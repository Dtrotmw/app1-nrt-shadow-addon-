"""
consensus.py

Combines per-arc reflector-height/surge estimates (from arcfit.run_day) into
a single water-level time series at a fixed report cadence.

Kept deliberately simple for Phase 1 (baseline-sector validation): weighted
median across whatever arcs are active in a trailing buffer window, weight
by PNR. This matches the general shape of the local V25-V29 code's
(previously unvalidated) consensus step, without inheriting its specific
bugs. Song et al. (2019)'s tidal-prediction-guided peak disambiguation is a
Phase 2 addition (needed once the mask experiment starts producing more
multi-peak-ambiguous arcs from the wider mask), not implemented here.

Change log
----------
v1  2026-07-29  Claude Code. Initial version (Phase 1 of plan
    `unified-growing-pony.md`).
"""
import numpy as np
import pandas as pd


def weighted_median(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    return float(values[np.searchsorted(cum, 0.5 * cum[-1])])


def build_series(arc_df, predinterp, freq="15min", buffer_minutes=60, outlier_gate_m=0.35):
    """
    Turn arc_df (arcfit.run_day's output, one row per accepted/rejected arc)
    into a report_time-indexed series of (predlevel, surgehat, finallevel).

    At each report time, uses all ACCEPTed arcs whose t_end falls within
    `buffer_minutes` before that report time (an arc "reports in" once it
    completes), weighted-median by PNR with a simple outlier gate around the
    median (same shape as the local code's consensus, not its specific bugs).

    `predlevel` at each report time is evaluated directly from `predinterp`
    (the same interpolator arcfit.run_day used, evaluated at the exact
    report_time) -- NOT averaged from each active arc's own arc-mean
    predlevel. That averaging was a real bug found during Phase 1
    validation: arcs in the buffer can be up to `buffer_minutes` old and each
    spans up to delTmax_min itself, so during rapid tide change (e.g. ~2m/hr
    near mid-tide) their individual arc-mean predlevels reflect meaningfully
    different times than the report instant -- averaging them smeared in
    stale tide level and produced errors up to ~1.8m timed almost exactly
    with the fastest parts of the tidal cycle, while washing out to a small
    net bias over a full day (see CHANGELOG). `surgehat` (the met/residual
    anomaly, not the tide's own fast motion) is legitimately fine to combine
    across a trailing buffer -- it's expected to vary slowly.
    """
    ok = arc_df[arc_df["status"] == "ACCEPT"].copy()
    if ok.empty:
        return pd.DataFrame(columns=["report_time", "predlevel", "surgehat", "finallevel", "n_arcs"])

    ok["t_end"] = pd.to_datetime(ok["t_end"])
    t0 = ok["t_end"].min().floor(freq)
    t1 = ok["t_end"].max().ceil(freq)
    report_times = pd.date_range(t0, t1, freq=freq)

    rows = []
    for rt in report_times:
        window_start = rt - pd.Timedelta(minutes=buffer_minutes)
        active = ok[(ok["t_end"] > window_start) & (ok["t_end"] <= rt)]
        rt_unix = (rt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        predlevel = float(predinterp(rt_unix))

        if active.empty:
            rows.append(dict(report_time=rt, predlevel=predlevel, surgehat=np.nan,
                              finallevel=np.nan, n_arcs=0))
            continue

        surges = active["surge"].to_numpy(float)
        weights = active["pnr"].to_numpy(float)
        anchor = weighted_median(surges, weights)
        keep = np.abs(surges - anchor) <= max(outlier_gate_m, 3 * np.median(np.abs(surges - anchor)))
        if keep.sum() == 0:
            keep = np.ones(len(surges), dtype=bool)

        surgehat = float(np.average(surges[keep], weights=np.clip(weights[keep], 1e-6, None)))
        rows.append(dict(report_time=rt, predlevel=predlevel, surgehat=surgehat,
                          finallevel=predlevel + surgehat, n_arcs=int(keep.sum())))

    return pd.DataFrame(rows)
