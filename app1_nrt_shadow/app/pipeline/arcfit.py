"""
arcfit.py

Clean, from-scratch GNSS-IR arc-fit engine for APP1, replacing the V25-V29
notebook lineage per David's direction (2026-07-29): treat that lineage as
evidence only, not a base to extend.

Built following gnssrefl's documented method and parameters (Larson et al.
2013; Roesler & Larson 2018) as the specification -- gnssrefl's own code
could not be used directly in this environment (see CHANGELOG, Phase 0
finding: it shells out to POSIX tools that don't resolve on native Windows
Python). The orbit/geometry layer is carried over from v26.ipynb (proven
correct there); the parts that were NOT reliable in that lineage -- the
15-minute arc chopping, and an unresolved sign disagreement between V26 and
V27 -- are both fixed here, with the fix derived from first principles
below rather than copied from either version.

Surge sign, derived (not copied from V26 or V27):
    H          = totalanth - water_level        (reflector height above the
                                                   water surface)
    hpred      = totalanth - predlevel           (H if the tide were exactly
                                                   as predicted, no surge)
    Fitting searches over a candidate offset `dh` such that the hypothesis
    reflector height is H_hyp = hpred - dh. Solving for the implied water
    level: water_level_hyp = totalanth - H_hyp = predlevel + dh.
    So dh IS the surge (actual minus predicted) directly, with no sign flip:
        surge = +dh_sub
    (V27 derived this correctly in its comments; V26 used `-dh_sub`, which
    is backwards. V27 nonetheless scored worse against NOC_Obs historically
    -- but V27 also changed several consensus-config parameters at the same
    time, so that comparison doesn't isolate the sign. This derivation does.)

Change log
----------
v1  2026-07-29  Claude Code. Initial version (Phase 0/1 of plan
    `unified-growing-pony.md`): full-arc reconstruction (via
    concat_rinex_day.py), arc detection not tied to the 15-min RINEX file
    boundaries, corrected surge sign, config-driven azimuth/elevation mask.
    Tropospheric correction is still the simple Bennett (1982) mapping
    carried over from the local code -- NOT yet the more advanced MPF
    (Williams & Nievinski 2017) model gnssrefl documents; left as a
    follow-up, not silently skipped (see briefing SS9).
"""
import os
import warnings

import numpy as np
import pandas as pd
import georinex as gr
import pymap3d as pm

from pipeline.geometry import (
    SIG_PRIORITY, LAMBDA_MAP, get_sat_pos, get_tropo_delay, az_el_series,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")

RX_XYZ = np.array([3997184.446, -293144.113, 4939523.568], float)
TOTALANTH = 31.406  # m above Chart Datum, per CLAUDE.md / briefing SS2

# Baseline sectors -- the live, currently-deployed mask (briefing SS2).
# Candidate (b), the NW-expansion test, adds the North sector below (from
# V91/V101/V103's shared SECTORS dict) -- see pipeline/masks.py.
BASELINE_SECTORS = {
    "Chivenor": dict(minaz=5, maxaz=30, minel=0.5, maxel=6.5, weight=0.8),
    "Instow": dict(minaz=30, maxaz=105, minel=1.0, maxel=8.0, weight=0.8),
}

# QC defaults -- named after gnssrefl's documented parameters (make_gnssir_input),
# used here as the specification even though gnssrefl's own code isn't called.
DEFAULTS = dict(
    minelspread=0.5,     # deg; gnssrefl 'ediff'-style edge requirement, simplified
    delTmax_min=75.0,    # max arc duration, minutes
    gap_tolerance_s=300,  # 5 min, matching Purnell et al. (2024) arc-boundary rule
    peak2noise_min=2.8,  # gnssrefl default -- dimensionless ratio, comparable across implementations
    ampl_min=None,       # NOT gnssrefl's ampl=5.0 default: that's on gnssrefl's own LSP
                         # normalization, which isn't the same scale as `power[bestidx]` below
                         # (first real test: 10/10 good arcs, PNR 8-58, rejected here on amplitude
                         # 0.06-0.42 << 5.0). Disabled until recalibrated against this formula's
                         # own distribution; PNR is the working quality gate for now, matching
                         # what the local v26/v27 code relied on.
    lsp_dh_search=3.5,   # m, surge search half-window
    lsp_dh_step=0.01,    # m, grid resolution
    surge_plausible_max=1.5,  # m. Found 2026-07-29 (residual_diagnostics.py):
                         # 4 arcs at az~300-333deg (west edge of the candidate North
                         # sector) fit to implied water levels of -1.4 to -2.8m CD,
                         # against a known true tide of +0.3 to +0.75m CD that day --
                         # physically impossible (UK estuarine surge doesn't run
                         # metres beyond the true tide). Checked: at the *true* known
                         # height these azimuths land on real, already-classified wet
                         # ground only 200-260m from the antenna, well inside the
                         # +-3.5m search window -- so the window wasn't the problem;
                         # the periodogram fit locked onto the wrong peak (exactly the
                         # multi-candidate-peak failure mode Song et al. 2019 address,
                         # flagged in the Phase 2 plan but not yet implemented). This
                         # is a coarse stopgap, not that method: reject any fit whose
                         # |surge| exceeds this bound. 1.5m is generous (real storm
                         # surge here is well under 1m; CLAUDE.md's own operational
                         # trigger is 0.4m) -- meant to catch wrong-peak locks, not to
                         # trim genuine large surge.
    minpts=30,           # a POINT count, not a duration -- at the recommended 30s-decimated
                         # input (concat_rinex_day.py's decimate_s), this is a 15-minute minimum
                         # arc length, which is a sensible GNSS-IR floor (the one persistently
                         # unreliable arc found during debugging, "G03", was ~5 minutes). If you
                         # feed this an un-decimated 2-second-cadence file, the same minpts=30
                         # only requires 60 seconds -- too short to trust; decimate first.
)


def sector_for(az_deg, el_deg, sectors):
    az = float(az_deg) % 360.0
    for name, s in sectors.items():
        minaz = float(s["minaz"]) % 360.0
        maxaz = float(s["maxaz"]) % 360.0
        in_az = (minaz <= az <= maxaz) if minaz <= maxaz else (az >= minaz or az <= maxaz)
        if in_az and (s["minel"] <= el_deg <= s["maxel"]):
            return name, s
    return None, None


def get_or_download_navfile(basedir, targetdate, verbose=False):
    """Local BRDC search first, then download from IGS/BKG -- same pattern as v26.ipynb."""
    import requests

    dt = pd.to_datetime(targetdate)
    daystr = dt.strftime("%y%j")
    daydir = os.path.join(os.path.normpath(basedir), daystr)

    for pat in ("*MN.rnx*", "*.n*"):
        import glob
        matches = glob.glob(os.path.join(daydir, pat))
        if matches:
            navpath = max(matches, key=os.path.getsize)
            if verbose:
                print("LOCAL NAV", os.path.basename(navpath))
            return navpath

    doy = dt.timetuple().tm_yday
    yyyy = dt.year
    base = f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{yyyy}/{doy:03d}"
    for fname in (f"BRDC00IGSR_{yyyy}{doy:03d}0000_01D_MN.rnx.gz",
                  f"BRDC00IGS_R_{yyyy}{doy:03d}0000_01D_MN.rnx.gz"):
        savepath = os.path.join(daydir, fname)
        if os.path.exists(savepath) and os.path.getsize(savepath) > 10000:
            return savepath
        try:
            r = requests.get(f"{base}/{fname}", timeout=60)
            if r.status_code == 200 and len(r.content) > 10000:
                os.makedirs(daydir, exist_ok=True)
                with open(savepath, "wb") as fh:
                    fh.write(r.content)
                return savepath
        except Exception:
            pass
    return None


def detect_arcs(times, el_deg, gap_tolerance_s, minelspread, trend_noise_deg=0.02):
    """
    Split a satellite's already sector-filtered (in time order) samples into
    arcs: a new arc starts on a time gap > gap_tolerance_s, or when the
    elevation trend reverses direction mid-run (rise vs. set).

    Trend is tracked as a rolling smoothed direction (the sign of elevation
    change since the last point that moved by more than `trend_noise_deg`),
    not locked to the arc's first sample -- comparing every later step
    against a frozen first-step direction would flag the first bit of
    sample-to-sample noise as a reversal and fragment a genuine single-pass
    arc into unusable slivers.

    Returns a list of (start_idx, end_idx) index pairs into the input arrays,
    end_idx inclusive, for runs long enough to be worth fitting.
    """
    n = len(times)
    if n < 2:
        return []

    t_sec = np.array([(t - times[0]).total_seconds() for t in times])
    d_el = np.diff(el_deg)
    d_t = np.diff(t_sec)

    arcs = []
    start = 0
    trend = None  # +1 rising, -1 setting
    for i in range(len(d_t)):
        gap_break = d_t[i] > gap_tolerance_s
        trend_break = False
        if abs(d_el[i]) > trend_noise_deg:
            this_trend = 1 if d_el[i] > 0 else -1
            if trend is not None and this_trend != trend:
                trend_break = True
            trend = this_trend
        if gap_break or trend_break:
            if i - start >= 1:
                arcs.append((start, i))
            start = i + 1
            trend = None
        elif not gap_break:
            pass
    if n - 1 - start >= 1:
        arcs.append((start, n - 1))

    out = []
    for s, e in arcs:
        if (el_deg[s:e + 1].max() - el_deg[s:e + 1].min()) >= minelspread:
            out.append((s, e))
    return out


def fit_arc(el, az, snr_dbhz, hpred, lam, cfg):
    """
    Reflector-height / surge fit for one already-detected, already
    sector-filtered arc. Method: phase-referenced Lomb-Scargle with
    intra-arc tidal-phase correction (Williams & Nievinski 2017), voltage
    SNR + quadratic sin(el) detrend (Larson et al. 2013; Roesler & Larson
    2018), peak-to-noise QC (Roesler & Larson 2018).

    Returns (result_dict_or_None, diagnostics_dict).
    """
    el = np.asarray(el, float)
    snr_dbhz = np.asarray(snr_dbhz, float)
    hpred = np.asarray(hpred, float)

    diag = dict(npts=int(el.size))
    if el.size < cfg["minpts"]:
        diag["reject"] = "toofewpoints"
        return None, diag

    meanel = float(np.mean(el))
    hpred_mean = float(np.mean(hpred))

    snr_v = 10.0 ** (snr_dbhz / 20.0)
    sinel = np.sin(np.radians(el))
    try:
        p = np.polyfit(sinel, snr_v, 2)
        snrdet = snr_v - np.polyval(p, sinel)
    except Exception:
        snrdet = snr_v - np.mean(snr_v)

    var = float(np.var(snrdet))
    if var < 1e-30:
        diag["reject"] = "zerovar"
        return None, diag

    # Tropospheric correction disabled -- the Bennett-style get_tropo_delay() mapping
    # (carried over from the local V17-onward code, comment: "very simple mapping...
    # keep as-is") barely varies with elevation (its ad/bd/cd constants are ~1e-3,
    # so mdry/mwet stay ~1.001 at every elevation instead of growing sharply at low
    # elevation the way a real mapping function must). At el=4 deg that put a ~0.046
    # correction on top of sinel=0.070 -- a ~66% distortion for what should be a small
    # refraction refinement, and the actual source of the ~1-2m systematic surge bias
    # found in first testing (confirmed empirically: removing it fixes the bias; a
    # separate test showed the `tau` term removed above was NOT the cause). Proper
    # fix is a real mapping-function model (e.g. MPF / Williams & Nievinski 2017,
    # already flagged as a follow-up in briefing SS9) -- not implemented yet, so
    # disabled rather than left silently wrong.
    sinel_c = sinel

    K = 4.0 * np.pi / float(lam)
    N = len(snrdet)
    dh_grid = np.arange(-cfg["lsp_dh_search"], cfg["lsp_dh_search"] + cfg["lsp_dh_step"] * 0.5, cfg["lsp_dh_step"])
    power = np.empty(len(dh_grid))

    for i, dh in enumerate(dh_grid):
        # Hypothesis reflector height at this grid point: H_hyp = hpred - dh.
        # `hpred` is already per-point (the predicted level at each observation
        # instant, not the arc mean) -- that IS the Williams & Nievinski (2017)
        # intra-arc tidal correction. The old code (both V26 and V27) applied a
        # second, uncited "tau" adjustment here derived from
        # arctan2(sum sin(2*phi), sum cos(2*phi)); removed -- it isn't part of
        # the cited method, its derivation doesn't hold up under scrutiny (see
        # CHANGELOG), and it produced a ~1-2m systematic surge bias once arcs
        # were allowed to run their full length instead of being chopped to
        # 15 minutes, where the effect had been too small to notice.
        phi_t = K * (hpred - dh) * sinel_c
        ct, st = np.cos(phi_t), np.sin(phi_t)
        cc, ss = np.dot(ct, ct), np.dot(st, st)
        A = np.dot(snrdet, ct) ** 2 / (cc + 1e-30)
        B = np.dot(snrdet, st) ** 2 / (ss + 1e-30)
        power[i] = (A + B) / (2.0 * var * N)

    bestidx = int(np.argmax(power))
    if bestidx < 2 or bestidx > (len(dh_grid) - 3):
        diag["reject"] = "edgemin"
        return None, diag

    p0, p1, p2 = power[bestidx - 1], power[bestidx], power[bestidx + 1]
    denom = 2.0 * p1 - p0 - p2
    dh_sub = dh_grid[bestidx] + (0.5 * (p0 - p2) / denom * cfg["lsp_dh_step"] if abs(denom) > 1e-30 else 0.0)

    bestH = float(hpred_mean - dh_sub)
    surge = float(dh_sub)  # see module docstring for the sign derivation

    noise_mask = np.abs(dh_grid - dh_grid[bestidx]) > 2.0
    noise_floor = float(np.mean(power[noise_mask])) if noise_mask.sum() > 5 else 1e-6
    amplitude = float(power[bestidx])
    pnr = amplitude / (noise_floor + 1e-30)

    # Envelope decay: Simon Williams' suggestion (2026-07-28) that a reflection
    # off a small/disconnected water body might show "a different signature
    # (higher peak to noise or different decay parameter)" than one off the
    # open channel -- he flagged it as "likely too subtle" himself, and there
    # is no single citable "decay parameter" in the literature for this, so
    # this is our own simple operational definition, not a lifted formula:
    # the slope of a rolling-window RMS envelope of the detrended SNR signal
    # against elevation (SNR-volts per degree). Arcs are time-ordered with
    # monotonic elevation within an arc (detect_arcs splits on trend
    # reversal), so this is well-defined. Meant to be compared across arcs
    # classified wet-channel vs. wet-pool by pipeline/pnr_decay_check.py, not
    # over-interpreted on its own.
    win = max(5, N // 10)
    env = pd.Series(snrdet).rolling(win, center=True, min_periods=3).std().to_numpy()
    env_valid = ~np.isnan(env)
    if env_valid.sum() >= 5:
        decay_slope = float(np.polyfit(el[env_valid], env[env_valid], 1)[0])
    else:
        decay_slope = None

    meanaz = float(np.mean(az))
    diag.update(meanel=round(meanel, 3), meanaz=round(meanaz, 3), bestH=round(bestH, 4),
                surge=round(surge, 4), pnr=round(pnr, 3), amplitude=round(amplitude, 6),
                decay_slope=round(decay_slope, 6) if decay_slope is not None else None)

    if pnr < cfg["peak2noise_min"]:
        diag["reject"] = "lowpnr"
        return None, diag
    if abs(surge) > cfg["surge_plausible_max"]:
        diag["reject"] = "implausiblesurge"
        return None, diag
    if cfg["ampl_min"] is not None and amplitude < cfg["ampl_min"]:
        diag["reject"] = "lowamplitude"
        return None, diag

    return dict(bestH=bestH, surge=surge, pnr=pnr, amplitude=amplitude, meanel=meanel,
                meanaz=meanaz, decay_slope=decay_slope), diag


def build_predinterp(predfile):
    """Cubic interpolator over the harmonic tide prediction file (datetime,level, headerless CSV)."""
    from scipy.interpolate import interp1d

    pred_df = pd.read_csv(predfile, header=None, names=["datetime", "level"])
    pred_dt = pd.to_datetime(pred_df["datetime"], errors="coerce")
    if getattr(pred_dt.dt, "tz", None) is not None:
        pred_dt = pred_dt.dt.tz_convert(None)
    pred_level = pd.to_numeric(pred_df["level"], errors="coerce")
    pred_mask = pred_dt.notna() & pred_level.notna()
    pred_tsec = (pred_dt[pred_mask] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    return interp1d(pred_tsec.to_numpy(float), pred_level[pred_mask].to_numpy(float),
                     kind="cubic", fill_value="extrapolate")


def load_day_data(targetdate, basedir, rinex_day_file, verbose=False):
    """
    Load the (mask-independent) nav + obs data for one day, so multiple mask
    experiments on the same date don't each re-pay the expensive georinex
    parse (profiled in CHANGELOG entry 7 as ~94%+ of run_day's cost).
    """
    navfile = get_or_download_navfile(basedir, targetdate, verbose=verbose)
    if not navfile:
        raise RuntimeError(f"No nav file available for {targetdate}")
    nav = gr.load(navfile, use=["G", "E", "R", "C"])
    obs = gr.load(rinex_day_file)
    return nav, obs


def run_day(targetdate, basedir, rinex_day_file, predfile, sectors=None, cfg=None,
            verbose=False, nav=None, obs=None):
    """
    Full-day arc processing for one date: reconstructed daily RINEX file in,
    per-arc reflector-height/surge estimates out (one row per accepted arc).
    Does not chop by the original 15-minute file boundaries -- arcs run their
    natural length, detected by `detect_arcs`.

    Pass pre-loaded `nav`/`obs` (from `load_day_data`) to skip re-parsing when
    running multiple mask configs on the same date.
    """
    sectors = sectors or BASELINE_SECTORS
    cfg = {**DEFAULTS, **(cfg or {})}

    rx_lat, rx_lon, rx_alt = pm.ecef2geodetic(RX_XYZ[0], RX_XYZ[1], RX_XYZ[2])

    if nav is None or obs is None:
        nav, obs = load_day_data(targetdate, basedir, rinex_day_file, verbose=verbose)

    predinterp = build_predinterp(predfile)

    rows = []
    for sv in obs.sv.values:
        svstr = str(sv)
        const = svstr[0]
        if const not in SIG_PRIORITY:
            continue

        validsig = next((s for s in SIG_PRIORITY[const] if s in obs.data_vars), None)
        if validsig is None:
            continue
        lam = LAMBDA_MAP.get(validsig, 0.19029)

        raw = obs[validsig].sel(sv=sv).dropna(dim="time")
        if len(raw) < cfg["minpts"]:
            continue

        times = pd.to_datetime(raw.time.values)
        snr_all = raw.values.astype(float)

        az_all, el_all = az_el_series(nav, svstr, const, rx_lat, rx_lon, rx_alt, times)
        if az_all is None:
            continue

        sector_names = np.array([sector_for(a, e, sectors)[0] for a, e in zip(az_all, el_all)], dtype=object)
        in_mask = sector_names != None  # noqa: E711

        if not in_mask.any():
            continue

        idx_all = np.where(in_mask)[0]
        # process each maximal run of consecutive in-mask indices separately,
        # so detect_arcs only ever sees genuinely contiguous samples
        splits = np.where(np.diff(idx_all) > 1)[0]
        runs = np.split(idx_all, splits + 1)

        for run in runs:
            if len(run) < 2:
                continue
            t_run = times[run]
            el_run = el_all[run]
            az_run = az_all[run]
            snr_run = snr_all[run]
            sec_run = sector_names[run]

            for s, e in detect_arcs(t_run, el_run, cfg["gap_tolerance_s"], cfg["minelspread"]):
                dur_min = (t_run[e] - t_run[s]).total_seconds() / 60.0
                if dur_min > cfg["delTmax_min"]:
                    continue

                arc_t = t_run[s:e + 1]
                arc_el = el_run[s:e + 1]
                arc_az = az_run[s:e + 1]
                arc_snr = snr_run[s:e + 1]
                sector_name = sec_run[s]

                t_unix = (pd.DatetimeIndex(arc_t) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
                predlevel_pt = predinterp(t_unix.to_numpy(float))
                hpred_pt = TOTALANTH - predlevel_pt

                result, diag = fit_arc(arc_el, arc_az, arc_snr, hpred_pt, lam, cfg)
                row = dict(
                    sv=svstr, sig=validsig, sector=sector_name,
                    t_start=arc_t[0], t_end=arc_t[-1],
                    n=len(arc_t), status="ACCEPT" if result else "REJECT",
                )
                row.update(diag)
                if result:
                    row.update(result)
                    row["predlevel_mean"] = float(np.mean(predlevel_pt))
                rows.append(row)

    return pd.DataFrame(rows)
