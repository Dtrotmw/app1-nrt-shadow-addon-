"""
extraction_cache.py

One-time-per-date extraction cache for GNSS-IR experiments, enabling the
2026-08-03 "have another go at closing the gap to Simon's accuracy"
campaign (David's brief).

Motivation (profiled, not assumed): RINEX/nav parsing via georinex costs
~260-300s per date and is identical for every mask/technique variant;
the arc-fit itself costs ~17-35s. Every experiment so far has re-paid the
parse cost per date per run. This module pays it ONCE per date, storing
the extracted per-satellite/per-signal SNR + azimuth + GEOMETRIC (raw,
refraction-free) elevation series to parquet; experiments then replay
from cache in seconds and can vary, without re-parsing:

  - azimuth/elevation masks (sector filtering happens at replay),
  - refraction model (cache stores raw elevation; the Bennett x1.15
    correction -- or any alternative -- is applied at replay),
  - signal selection (cache stores ALL SNR signals per satellite, not
    just the first-priority one run_day uses; `signal_mode="all"` turns
    each extra frequency into an independent arc stream, per Sepulveda
    et al. 2023 / gnssrefl practice),
  - every fit/QC parameter (fit happens at replay).

`cached_run_one_day(..., signal_mode="primary", refraction_fn=None)` is
validated to reproduce pipeline.validate.run_one_day exactly (same
accepted arcs, same surges) -- see validate_replay(). Anything else is
a deliberate experiment.

Cache files: refl_code/scratch/cache/extract_{YYYY-MM-DD}.parquet
Columns: sv, const, sig, primary(bool), time, az, el_raw, snr

Change log
----------
v1  2026-08-03  Claude Code (Fable). Initial version.
"""
import os
import time as _time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pymap3d as pm

from pipeline.arcfit import (
    BASELINE_SECTORS, DEFAULTS, RX_XYZ, TOTALANTH,
    detect_arcs, fit_arc, sector_for, build_predinterp,
)
from pipeline.geometry import (
    SIG_PRIORITY, LAMBDA_MAP, az_el_series, refraction_angle_correction,
)
from pipeline.consensus import build_series
from pipeline.validate import load_day, _predfile_for

# Backtesting-only, not used in live operation -- main.py always passes
# cache_df explicitly (its own rolling buffer), bypassing load_cache/
# build_cache entirely. See validate.py's equivalent comment.
CACHE_DIR = "/data/cache_unused"


def cache_path(targetdate):
    return os.path.join(CACHE_DIR, f"extract_{pd.Timestamp(targetdate).date()}.parquet")


def build_cache(targetdate, nav=None, obs=None, verbose=True):
    """Parse one date's RINEX (or reuse pre-loaded nav/obs) and store the full
    extraction to parquet. Returns the path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = cache_path(targetdate)
    if nav is None or obs is None:
        t0 = _time.time()
        nav, obs = load_day(targetdate)
        if verbose:
            print(f"  load_day: {_time.time()-t0:.0f}s")

    rx_lat, rx_lon, rx_alt = pm.ecef2geodetic(RX_XYZ[0], RX_XYZ[1], RX_XYZ[2])

    frames = []
    t0 = _time.time()
    for sv in obs.sv.values:
        svstr = str(sv)
        const = svstr[0]
        if const not in SIG_PRIORITY:
            continue
        # Dataset-level primary-signal rule, replicated from run_day exactly:
        # first priority signal present in obs.data_vars (NOT per-sv).
        primary_sig = next((s for s in SIG_PRIORITY[const] if s in obs.data_vars), None)

        # All signals with any data for this sv.
        sig_series = {}
        for sig in SIG_PRIORITY[const]:
            if sig not in obs.data_vars:
                continue
            raw = obs[sig].sel(sv=sv).dropna(dim="time")
            if len(raw) < 2:
                continue
            sig_series[sig] = (pd.to_datetime(raw.time.values), raw.values.astype(float))
        if not sig_series:
            continue

        # az/el once per satellite, at the union of all its signals' times,
        # GEOMETRIC (no refraction) so models can be swapped at replay.
        union_times = pd.DatetimeIndex(
            np.unique(np.concatenate([t.values for t, _ in sig_series.values()]))
        )
        az_u, el_u = az_el_series(nav, svstr, const, rx_lat, rx_lon, rx_alt,
                                   union_times, apply_refraction=False)
        if az_u is None:
            continue
        az_lookup = pd.Series(az_u, index=union_times)
        el_lookup = pd.Series(el_u, index=union_times)

        for sig, (times, snr) in sig_series.items():
            frames.append(pd.DataFrame(dict(
                sv=svstr, const=const, sig=sig, primary=(sig == primary_sig),
                time=times, az=az_lookup.reindex(times).to_numpy(),
                el_raw=el_lookup.reindex(times).to_numpy(), snr=snr,
            )))

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, index=False)
    if verbose:
        print(f"  extracted {len(df)} rows, {df['sv'].nunique()} sv, "
              f"{df.groupby(['sv','sig']).ngroups} sv-signal streams ({_time.time()-t0:.0f}s)")
    return out


def load_cache(targetdate):
    return pd.read_parquet(cache_path(targetdate))


def default_refraction(el_raw):
    """The production model: Bennett (1982) x RADIO_REFRACTIVITY_SCALE."""
    return el_raw + refraction_angle_correction(el_raw)


def cached_run_day(targetdate, sectors=None, cfg=None, refraction_fn=None,
                    signal_mode="primary", fit_fn=None, cache_df=None,
                    predfile_override=None, end_time=None, start_time=None):
    """run_day replayed from cache. With defaults (primary signals, Bennett
    refraction, arcfit.fit_arc) this reproduces arcfit.run_day exactly.

    signal_mode: "primary" (run_day-identical) or "all" (every cached
    sv/signal stream with enough points becomes an independent arc source).
    refraction_fn: maps raw elevation -> apparent elevation (default Bennett).
    fit_fn: alternative per-arc fitter with fit_arc's signature (experiments).
    predfile_override: path to an alternative (datetime,level) headerless
        predfile, e.g. an alternative harmonics vintage -- see
        pipeline/june26_compare.py. Bypasses the Oct'25-vintage default
        picked by validate._predfile_for.
    end_time: if given, drop all cached samples after this timestamp before
        fitting -- used by the NRT-delay convergence test in
        june26_compare.py to see how an estimate near the live edge would
        have looked with only data up to that moment.
    """
    sectors = sectors or BASELINE_SECTORS
    cfg = {**DEFAULTS, **(cfg or {})}
    refraction_fn = refraction_fn or default_refraction
    fit_fn = fit_fn or fit_arc

    df = cache_df if cache_df is not None else load_cache(targetdate)
    if start_time is not None:
        df = df[df["time"] >= pd.Timestamp(start_time)]
    if end_time is not None:
        df = df[df["time"] <= pd.Timestamp(end_time)]
    predinterp = build_predinterp(predfile_override or _predfile_for(targetdate))

    rows = []
    for (svstr, sig), g in df.groupby(["sv", "sig"], sort=True):
        if signal_mode == "primary" and not g["primary"].iloc[0]:
            continue
        g = g.sort_values("time")
        if len(g) < cfg["minpts"]:
            continue
        lam = LAMBDA_MAP.get(sig, 0.19029)

        times = pd.DatetimeIndex(g["time"])
        az_all = g["az"].to_numpy()
        el_all = refraction_fn(g["el_raw"].to_numpy())
        snr_all = g["snr"].to_numpy()

        sector_names = np.array([sector_for(a, e, sectors)[0] for a, e in zip(az_all, el_all)],
                                 dtype=object)
        in_mask = sector_names != None  # noqa: E711
        if not in_mask.any():
            continue

        idx_all = np.where(in_mask)[0]
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

                result, diag = fit_fn(arc_el, arc_az, arc_snr, hpred_pt, lam, cfg)
                row = dict(
                    sv=svstr, sig=sig, sector=sector_name,
                    t_start=arc_t[0], t_end=arc_t[-1],
                    n=len(arc_t), status="ACCEPT" if result else "REJECT",
                )
                row.update(diag)
                if result:
                    row.update(result)
                    row["predlevel_mean"] = float(np.mean(predlevel_pt))
                rows.append(row)

    return pd.DataFrame(rows)


def cached_run_one_day(targetdate, sectors=None, cfg=None, refraction_fn=None,
                        signal_mode="primary", fit_fn=None, cache_df=None,
                        series_fn=None, predfile_override=None):
    """Cache-replay equivalent of validate.run_one_day: (arc_df, series).
    series_fn: alternative consensus builder taking (arc_df, predinterp)."""
    arc_df = cached_run_day(targetdate, sectors=sectors, cfg=cfg,
                             refraction_fn=refraction_fn, signal_mode=signal_mode,
                             fit_fn=fit_fn, cache_df=cache_df,
                             predfile_override=predfile_override)
    predinterp = build_predinterp(predfile_override or _predfile_for(targetdate))
    series = (series_fn or build_series)(arc_df, predinterp)
    return arc_df, series


def validate_replay(targetdate, sectors=None):
    """Confirm cached replay == the original uncached path on one date.
    Compares accepted-arc keys and surge values. Returns (ok, detail)."""
    from pipeline.validate import run_one_day
    from pipeline.masks import MASK_NW_EXPANDED
    sectors = sectors or MASK_NW_EXPANDED

    nav, obs = load_day(targetdate)
    if not os.path.exists(cache_path(targetdate)):
        build_cache(targetdate, nav=nav, obs=obs)

    orig_df, _ = run_one_day(targetdate, sectors=sectors, nav=nav, obs=obs)
    rep_df = cached_run_day(targetdate, sectors=sectors, signal_mode="primary")

    def key(df):
        acc = df[df["status"] == "ACCEPT"].copy()
        acc["k"] = acc["sv"].astype(str) + "|" + acc["t_start"].astype(str)
        return acc.set_index("k")["surge"].sort_index()

    a, b = key(orig_df), key(rep_df)
    same_keys = list(a.index) == list(b.index)
    max_diff = float((a - b).abs().max()) if same_keys and len(a) else (np.nan if not same_keys else 0.0)
    ok = same_keys and (len(a) == 0 or max_diff < 1e-9)
    return ok, dict(n_orig=len(a), n_replay=len(b), same_keys=same_keys, max_surge_diff=max_diff)


ALL_SLIPWAY_DATES = [
    "2025-11-04", "2025-11-05", "2025-11-06", "2025-11-07", "2025-11-17",
    "2025-12-03", "2026-01-19", "2026-03-01", "2026-03-03", "2026-06-17",
    "2026-07-01", "2026-07-02", "2026-07-12", "2026-07-13", "2026-07-14",
    "2026-07-15", "2026-07-16", "2026-07-17", "2026-07-18", "2026-07-19",
]

if __name__ == "__main__":
    t0 = _time.time()
    for d in ALL_SLIPWAY_DATES:
        if os.path.exists(cache_path(d)):
            print(f"{d}: cached already")
            continue
        print(f"{d}: building...")
        try:
            build_cache(d)
        except Exception as e:
            print(f"{d}: FAILED -- {type(e).__name__}: {e}")
        print(f"  running total {_time.time()-t0:.0f}s")
    print(f"Done. Total {_time.time()-t0:.0f}s")
