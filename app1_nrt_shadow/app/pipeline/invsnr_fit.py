"""
invsnr_fit.py

Strandberg-style inverse modeling of GNSS-IR SNR data for APP1, following
Strandberg, Hobiger & Haas (2016, Radio Science 51:1286-1296) as
implemented in gnssrefl's spline_functions.py (residuals_cubspl_js /
snr2spline) and gnssir-rt's arcs2spline -- both read line-by-line for this
port (see CHANGELOG entry for the campaign).

Idea: instead of one reflector height per arc from a periodogram peak,
model the day's water level as a CONTINUOUS function h(t) (cubic spline
through knot values) and fit the detrended SNR of ALL arcs simultaneously:

    SNR_model(t) = [A_j sin(4 pi h(t) sin(e)/lambda)
                    + B_j cos(4 pi h(t) sin(e)/lambda)] * exp(-4 k^2 s sin^2 e)

with one (A_j, B_j) amplitude/phase pair per satellite-stream class j
(constellation x frequency, per Strandberg), an optional common roughness
parameter s (the exp damping; Strandberg eq. for a rough sea surface), and
h(t)'s knot values as the parameters of interest. Nonlinear least squares
(scipy least_squares, soft_l1 loss for outlier robustness), initialised
from the standard per-arc LSP results (spline through accepted arcs'
bestH, exactly gnssrefl invsnr's "spectral LS" initialisation).

Why this should help at APP1 specifically: the weighted-median consensus
treats each arc's peak as one vote and cannot exploit partial/weak arcs;
the inverse model uses every SNR sample, weights information by how well
it fits a *continuous* tide, and pools all constellations/frequencies into
one estimate -- published gains are 30-50% RMS vs the LSP method at both
calm (Onsala) and dynamic (Spring Bay) sites.

Change log
----------
v1  2026-08-03  Claude Code (Fable). Initial implementation.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import least_squares

from pipeline.arcfit import TOTALANTH, DEFAULTS, detect_arcs, sector_for, build_predinterp
from pipeline.geometry import LAMBDA_MAP
from pipeline.extraction_cache import load_cache, cached_run_day, default_refraction
from pipeline.validate import _predfile_for


def collect_arc_samples(targetdate, sectors, cfg=None, refraction_fn=None,
                         signal_mode="primary", cache_df=None, sector_streams=False,
                         predfile_override=None, end_time=None, start_time=None):
    """Run the standard cached pipeline for QC, then return (arc_df, samples):
    samples = one row per SNR observation of every ACCEPTED arc, with the
    detrended voltage SNR (same quadratic-in-sin(el) detrend as fit_arc),
    sin(apparent el), wavelength, stream class, and time.

    start_time/end_time bound the window -- together they turn this from
    a "whole day from midnight" fit into an arbitrary trailing window,
    which can span a day boundary (cache_df may already be a multi-day
    concatenation -- see pipeline/nrt_simulation.py)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    refraction_fn = refraction_fn or default_refraction
    df = cache_df if cache_df is not None else load_cache(targetdate)
    if start_time is not None:
        df = df[df["time"] >= pd.Timestamp(start_time)]
    if end_time is not None:
        df = df[df["time"] <= pd.Timestamp(end_time)]

    arc_df = cached_run_day(targetdate, sectors=sectors, cfg=cfg,
                             refraction_fn=refraction_fn, signal_mode=signal_mode,
                             cache_df=df, predfile_override=predfile_override)
    acc = arc_df[arc_df["status"] == "ACCEPT"]

    sample_frames = []
    for (svstr, sig), g in df.groupby(["sv", "sig"], sort=True):
        if signal_mode == "primary" and not g["primary"].iloc[0]:
            continue
        sub = acc[(acc["sv"] == svstr) & (acc["sig"] == sig)]
        if sub.empty:
            continue
        g = g.sort_values("time")
        times = pd.DatetimeIndex(g["time"])
        el_all = refraction_fn(g["el_raw"].to_numpy())
        snr_all = g["snr"].to_numpy()

        for _, arc in sub.iterrows():
            m = (times >= arc["t_start"]) & (times <= arc["t_end"])
            if m.sum() < 5:
                continue
            el = el_all[m]
            snr_v = 10.0 ** (snr_all[m] / 20.0)
            sinel = np.sin(np.radians(el))
            try:
                p = np.polyfit(sinel, snr_v, 2)
                snrdet = snr_v - np.polyval(p, sinel)
            except Exception:
                snrdet = snr_v - np.mean(snr_v)
            lam = LAMBDA_MAP.get(sig, 0.19029)
            # A/B pair per constellation x frequency (Strandberg); optionally
            # ALSO per sector -- physically motivated at APP1, where different
            # sectors reflect off different surfaces (always-wet channel vs.
            # partially-exposed sand at low water), so the interference phase
            # need not be shared across them.
            stream = f"{svstr[0]}_{sig}" + (f"_{arc['sector']}" if sector_streams else "")
            sample_frames.append(pd.DataFrame(dict(
                t=times[m], sinel=sinel, snrdet=snrdet, lam=lam, stream=stream,
                sv=svstr, sig=sig, sector=arc["sector"],
            )))

    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return arc_df, samples


def invsnr_day(targetdate, sectors, cfg=None, refraction_fn=None,
                signal_mode="primary", cache_df=None, knot_minutes=60,
                roughness=True, loss="soft_l1", f_scale=1.0, verbose=False,
                sector_streams=False, lowwater_wet_only_below=None,
                wet_sectors=("North",), reg_smooth=0.0,
                predfile_override=None, end_time=None, start_time=None,
                anchor_fn=None, anchor_weight=0.0,
                prior_info=None, state_anchor_weight=0.0):
    """Fit the inverse model for one day. Returns (series_df, info).

    predfile_override: alternative (datetime,level) headerless predfile --
    e.g. the June'26 draft harmonics vintage (see june26_compare.py) --
    used both for the surge-smoothness regularizer's target and for the
    upstream per-arc LSP QC pass (arcfit.fit_arc's hpred).
    end_time: truncate the day's cached samples to this timestamp before
    fitting -- the NRT-delay convergence test (june26_compare.py) re-fits
    the same day with progressively later cutoffs to see how much a
    near-the-edge estimate moves as more data arrives after it.

    anchor_fn/anchor_weight: HEIGHT-only persistence (entry 41's original
    fix) -- warm-start + soft penalty tying h(t) to a previous cycle's own
    converged spline. Validated at single target-time convergence tests,
    but a genuine multi-day SLIDING WINDOW simulation (nrt_simulation.py,
    2026-08-04) found this alone isn't enough: a bounded trailing window
    (e.g. 8h, as a real live system would use, not the ever-growing
    whole-day-from-midnight window the earlier tests happened to use) has
    far less data to constrain the per-stream amplitude/phase (A,B) pairs
    each cycle, and THOSE drifting between cycles degrades accuracy badly
    (bias +0.88m, rmse 1.04m in the first sliding-window test) even when
    the height trajectory itself is anchored.

    prior_info/state_anchor_weight: the fuller fix -- carries forward the
    WHOLE previous cycle's state (height spline AND matching streams'
    A/B AND roughness), not just height. Pass the previous invsnr_day()
    call's own `info` dict directly (built internally, no need for a
    separate anchor_fn). This is the direct, evidenced answer to "does
    the retrieval need a persistent multi-parameter state (i.e. something
    closer to a genuine Kalman filter over the full state vector), not
    just a height-only trick" -- see CHANGELOG for the before/after.

    series_df: report_time (15-min grid), finallevel (water level, m CD) --
    same shape score_vs_slipway expects."""
    arc_df, samples = collect_arc_samples(targetdate, sectors, cfg=cfg,
                                           refraction_fn=refraction_fn,
                                           signal_mode=signal_mode, cache_df=cache_df,
                                           sector_streams=sector_streams,
                                           predfile_override=predfile_override,
                                           end_time=end_time, start_time=start_time)
    if samples.empty:
        return pd.DataFrame(columns=["report_time", "finallevel"]), dict(fail="no samples")

    # Dynamic wet-surface gating: below `lowwater_wet_only_below` (m CD,
    # predicted tide), keep only samples from sectors whose reflecting
    # surface is known always-wet (David's own site observation + SWOT:
    # the North sector's channel). Instow/Chivenor reflect off banks that
    # progressively expose as the tide falls -- exactly the established
    # sand-contamination mechanism -- so their samples are excluded from
    # the fit in that regime rather than allowed to drag h(t) high.
    if lowwater_wet_only_below is not None:
        predinterp0 = build_predinterp(predfile_override or _predfile_for(targetdate))
        t_unix = (pd.DatetimeIndex(samples["t"]) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        predlev = predinterp0(t_unix.to_numpy(float))
        keep = (predlev >= lowwater_wet_only_below) | samples["sector"].isin(wet_sectors)
        samples = samples[keep].reset_index(drop=True)
        if samples.empty:
            return pd.DataFrame(columns=["report_time", "finallevel"]), dict(fail="no samples after gating")

    acc = arc_df[arc_df["status"] == "ACCEPT"].copy()
    t0 = samples["t"].min()
    tsec = (samples["t"] - t0) / pd.Timedelta(seconds=1)
    tsec = tsec.to_numpy(float)
    t_end = tsec.max()

    # knots spanning the data, spacing knot_minutes
    kdt = knot_minutes * 60.0
    n_knots = max(4, int(np.ceil(t_end / kdt)) + 1)
    knots = np.linspace(0.0, t_end, n_knots)

    # ---- initialisation: cubic spline through per-arc LSP heights ----
    acc["t_mid_sec"] = ((acc["t_start"] + (acc["t_end"] - acc["t_start"]) / 2) - t0) / pd.Timedelta(seconds=1)
    acc = acc.sort_values("t_mid_sec")
    # median-combine arcs in each knot interval for a robust initial curve
    arc_t = acc["t_mid_sec"].to_numpy(float)
    arc_h = acc["bestH"].to_numpy(float)
    kval0 = np.empty(n_knots)
    for i, k in enumerate(knots):
        m = np.abs(arc_t - k) <= kdt
        kval0[i] = np.median(arc_h[m]) if m.sum() else np.nan
    # fill gaps by interpolation from filled knots
    ok = ~np.isnan(kval0)
    if ok.sum() < 2:
        return pd.DataFrame(columns=["report_time", "finallevel"]), dict(fail="too few arcs")
    kval0[~ok] = np.interp(knots[~ok], knots[ok], kval0[ok])

    streams = sorted(samples["stream"].unique())
    stream_idx = {s: j for j, s in enumerate(streams)}
    js = samples["stream"].map(stream_idx).to_numpy()
    sinel = samples["sinel"].to_numpy(float)
    snrdet = samples["snrdet"].to_numpy(float)
    lam = samples["lam"].to_numpy(float)

    # normalise each stream's detrended SNR to unit RMS so the shared loss
    # weights streams evenly (different signals have very different raw
    # amplitudes; without this the largest-amplitude stream dominates)
    for s, j in stream_idx.items():
        m = js == j
        rms = np.sqrt(np.mean(snrdet[m] ** 2))
        if rms > 0:
            snrdet[m] = snrdet[m] / rms

    # Anchor: a callable(datetime_array) -> H (reflector height, same units as
    # kval) built from a PREVIOUS invsnr_day() call's own converged spline --
    # the cheap, testable version of "carry state forward between cycles"
    # (a poor-man's Kalman predict step, no covariance). Used for BOTH the
    # optimizer's initial guess (so it starts near the last known-good
    # trajectory instead of a fresh median-of-arcs guess) and, if
    # anchor_weight>0, as a soft residual penalty discouraging the new fit
    # from drifting away from that trajectory during the OVERLAP period
    # unless the new SNR data actively supports it. Only applied at knot
    # times within the anchor's own known-valid coverage (anchor_fn returns
    # NaN outside it) -- past that, the fit is free, unconstrained by a stale
    # prior. See invsnr_convergence_test in june26_compare.py for the
    # evidence this fixes (real, not hypothetical): a from-scratch batch fit
    # on 2026-06-17 found several near-equally-good (same per-sample cost)
    # h(t) solutions 0.4-0.6m apart -- exactly the situation a persistence
    # prior is designed to disambiguate.
    if anchor_fn is not None:
        anchor_h = anchor_fn(t0 + pd.to_timedelta(knots, unit="s"))
        anchor_h = np.asarray(anchor_h, float)
        anchor_valid = np.isfinite(anchor_h)
    else:
        anchor_h = None
        anchor_valid = None

    # Full-state persistence (prior_info/state_anchor_weight): height AND
    # per-stream A/B AND roughness, all carried forward from the previous
    # cycle's own converged solution -- see docstring for why height-only
    # anchoring wasn't enough in a genuinely bounded sliding window.
    ab_prior_target = None
    roughness_prior = None
    if prior_info is not None and state_anchor_weight > 0:
        p_t0, p_knots, p_kval = prior_info["t0"], prior_info["knots"], prior_info["kval"]
        p_times_abs = ((p_t0 + pd.to_timedelta(p_knots, unit="s")) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        h_interp = interpolate.interp1d(p_times_abs.to_numpy(float), p_kval, kind="cubic", fill_value="extrapolate")
        new_times_abs = ((t0 + pd.to_timedelta(knots, unit="s")) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        new_times_abs = new_times_abs.to_numpy(float)
        state_h = h_interp(new_times_abs)
        state_valid = (new_times_abs >= p_times_abs.min()) & (new_times_abs <= p_times_abs.max())
        if anchor_h is None:  # don't double up if a separate anchor_fn was also given
            anchor_h, anchor_valid = state_h, state_valid
            anchor_weight = anchor_weight or state_anchor_weight

        prior_streams = prior_info.get("streams")
        prior_ab = prior_info.get("ab")
        if prior_streams is not None and prior_ab is not None:
            prior_ab_of = {s: (prior_ab[2 * j], prior_ab[2 * j + 1]) for j, s in enumerate(prior_streams)}
            ab_prior_target = np.full(2 * len(streams), np.nan)
            for j, s in enumerate(streams):
                if s in prior_ab_of:
                    ab_prior_target[2 * j], ab_prior_target[2 * j + 1] = prior_ab_of[s]
        if prior_info.get("roughness") is not None:
            roughness_prior = prior_info["roughness"]

    nk, ns = n_knots, len(streams)
    kval_init = kval0.copy()
    if anchor_fn is not None and anchor_valid.any():
        kval_init[anchor_valid] = anchor_h[anchor_valid]

    ab_init = np.tile([0.5, 0.5], ns)
    if ab_prior_target is not None:
        ok_ab = np.isfinite(ab_prior_target)
        ab_init[ok_ab] = ab_prior_target[ok_ab]
    rough_init = [roughness_prior if roughness_prior is not None else 0.01] if roughness else []

    p0 = np.concatenate([kval_init, ab_init, rough_init])

    lo = np.concatenate([np.full(nk, 15.0), np.full(2 * ns, -10.0), [0.0] if roughness else []])
    hi = np.concatenate([np.full(nk, 35.0), np.full(2 * ns, 10.0), [0.5] if roughness else []])
    p0 = np.clip(p0, lo, hi)

    K = 4.0 * np.pi / lam
    k_wave = 2.0 * np.pi / lam

    # Surge-smoothness regularization (reg_smooth > 0): penalize the second
    # difference of (h(knot) - hpred(knot)). The TIDE varies fast, but the
    # SURGE (their difference) varies over hours -- so this stabilizes the
    # spline through arc-sparse stretches without pulling the level toward
    # the prediction itself (a constant or linearly-drifting surge incurs
    # zero penalty; only knot-to-knot wiggles do).
    predinterp_reg = build_predinterp(predfile_override or _predfile_for(targetdate))
    knot_unix = (t0 + pd.to_timedelta(knots, unit="s") - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    hpred_knots = TOTALANTH - predinterp_reg(knot_unix.to_numpy(float))

    def residuals(p):
        kval = p[:nk]
        ab = p[nk:nk + 2 * ns]
        s_rough = p[-1] if roughness else 0.0
        hf = interpolate.interp1d(knots, kval, kind="cubic", fill_value="extrapolate")
        h_t = hf(tsec)
        phase = K * h_t * sinel
        A = ab[2 * js]
        B = ab[2 * js + 1]
        model = A * np.sin(phase) + B * np.cos(phase)
        if roughness:
            model = model * np.exp(-4.0 * k_wave ** 2 * s_rough * sinel ** 2)
        res = model - snrdet
        if reg_smooth > 0:
            surge_k = kval - hpred_knots
            res = np.concatenate([res, reg_smooth * np.diff(surge_k, 2)])
        if anchor_fn is not None and anchor_weight > 0 and anchor_valid.any():
            res = np.concatenate([res, anchor_weight * (kval[anchor_valid] - anchor_h[anchor_valid])])
        if state_anchor_weight > 0:
            if ab_prior_target is not None:
                ok_ab = np.isfinite(ab_prior_target)
                if ok_ab.any():
                    res = np.concatenate([res, state_anchor_weight * (ab[ok_ab] - ab_prior_target[ok_ab])])
            if roughness and roughness_prior is not None:
                res = np.concatenate([res, [state_anchor_weight * (s_rough - roughness_prior)]])
        return res

    ls = least_squares(residuals, p0, bounds=(lo, hi), loss=loss, f_scale=f_scale,
                        max_nfev=200 * len(p0))
    kval = ls.x[:nk]

    # report on the 15-min grid, inside the fitted span only
    hf = interpolate.interp1d(knots, kval, kind="cubic")
    report = pd.date_range(t0.ceil("15min"), samples["t"].max().floor("15min"), freq="15min")
    rsec = (report - t0) / pd.Timedelta(seconds=1)
    rsec = np.clip(rsec.to_numpy(float), knots[0], knots[-1])
    h_rep = hf(rsec)
    series = pd.DataFrame(dict(report_time=report, finallevel=TOTALANTH - h_rep))

    info = dict(n_samples=len(samples), n_arcs=len(acc), n_streams=ns,
                n_knots=nk, cost=float(ls.cost),
                roughness=float(ls.x[-1]) if roughness else None,
                success=bool(ls.success),
                t0=t0, knots=knots, kval=kval,
                streams=streams, ab=ls.x[nk:nk + 2 * ns])
    if verbose:
        print(f"  invsnr {targetdate}: {info}")
    return series, info
