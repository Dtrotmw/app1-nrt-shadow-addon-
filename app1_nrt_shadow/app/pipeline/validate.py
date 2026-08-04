"""
validate.py

Phase 1 validation gate: run the arc-fit engine + consensus for one or more
full days against the current baseline sectors (unchanged from the live
config), score against ground truth, and compare to the published benchmark
figures cited in APP1_GNSS-IR_Briefing.md SS9 (Larson, Ray & Williams 2017;
Larson & Williams 2023) before any mask experiment (Phase 2) is trusted.

Change log
----------
v1  2026-07-29  Claude Code. Initial version (Phase 1 of plan
    `unified-growing-pony.md`).
"""
import os
import warnings

import numpy as np
import pandas as pd

from pipeline.concat_rinex_day import concat_rinex_day
from pipeline.arcfit import run_day, load_day_data, build_predinterp, BASELINE_SECTORS
from pipeline.consensus import build_series

warnings.filterwarnings("ignore")

# These backtesting-only paths (dev-machine snapshots, D:\GNSS\DATA archive)
# are NOT valid inside the add-on container and are not used in live
# operation -- main.py always calls the underlying functions with explicit
# cache_df/predfile_override/basedir arguments instead, bypassing every
# function that would read from these. Pointed at /data (the container's
# real, existing persistent volume) rather than left as bogus dev paths, so
# any accidental use fails clearly instead of silently resembling something
# plausible.
SCRATCH_DIR = "/data/scratch_unused"
BASEDIR = "/data/rinex_archive_unused"
PREDFILES = []
NOC_OBS_FILE = "/data/NOC_Obs_unused.csv"
SLIPWAY_FILE = "/data/SlipObs_unused.csv"


def _predfile_for(targetdate):
    """Pick whichever predfile snapshot actually covers this date."""
    t = pd.Timestamp(targetdate)
    for pf in PREDFILES:
        df = pd.read_csv(pf, header=None, names=["dt", "level"], nrows=1)
        first = pd.to_datetime(df["dt"].iloc[0], utc=True).tz_convert(None)
        df_last = pd.read_csv(pf, header=None, names=["dt", "level"]).tail(1)
        last = pd.to_datetime(df_last["dt"].iloc[0], utc=True).tz_convert(None)
        if first <= t <= last:
            return pf
    raise ValueError(f"No predfile snapshot covers {targetdate}")


def load_noc_obs():
    df = pd.read_csv(NOC_OBS_FILE)
    df["dt"] = pd.to_datetime(df["TimeStamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df["NOC_Obs"] = pd.to_numeric(df["NOC_Obs"], errors="coerce")
    return df.dropna(subset=["dt", "NOC_Obs"])[["dt", "NOC_Obs"]]


def load_slipway():
    df = pd.read_csv(SLIPWAY_FILE)
    df.columns = ["dt", "height"]
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def run_one_day(targetdate, sectors=None, decimate_s=30, cfg=None, verbose=False,
                 nav=None, obs=None):
    """Reconstruct, arc-fit, and build the consensus series for one full day.

    Pass pre-loaded `nav`/`obs` (from `pipeline.arcfit.load_day_data`) to avoid
    re-parsing the same day's RINEX when scoring multiple masks against it.
    """
    sectors = sectors or BASELINE_SECTORS
    t = pd.Timestamp(targetdate)
    doy = t.timetuple().tm_yday
    yy = t.year % 100

    rinex_path = os.path.join(SCRATCH_DIR, f"app1_{t.year}_{doy:03d}_dec{decimate_s}.26O")
    if not os.path.exists(rinex_path):
        daydir = os.path.join(BASEDIR, f"{yy:02d}{doy:03d}")
        concat_rinex_day(daydir, doy, yy, rinex_path, decimate_s=decimate_s)

    predfile = _predfile_for(targetdate)

    arc_df = run_day(
        targetdate=targetdate, basedir=BASEDIR, rinex_day_file=rinex_path,
        predfile=predfile, sectors=sectors, cfg=cfg, verbose=verbose,
        nav=nav, obs=obs,
    )
    predinterp = build_predinterp(predfile)
    series = build_series(arc_df, predinterp)
    return arc_df, series


def load_day(targetdate, decimate_s=30):
    """Load (nav, obs) once for a date, to reuse across multiple run_one_day calls (masks)."""
    t = pd.Timestamp(targetdate)
    doy = t.timetuple().tm_yday
    yy = t.year % 100
    rinex_path = os.path.join(SCRATCH_DIR, f"app1_{t.year}_{doy:03d}_dec{decimate_s}.26O")
    if not os.path.exists(rinex_path):
        daydir = os.path.join(BASEDIR, f"{yy:02d}{doy:03d}")
        concat_rinex_day(daydir, doy, yy, rinex_path, decimate_s=decimate_s)
    return load_day_data(targetdate, BASEDIR, rinex_path)


def score_against_ground_truth(series, targetdate):
    """Score one day's consensus series against NOC_Obs (continuous) and slipway marks (sparse)."""
    noc = load_noc_obs()
    slip = load_slipway()

    t = pd.Timestamp(targetdate)
    day_series = series.dropna(subset=["finallevel"]).copy()
    if day_series.empty:
        return dict(targetdate=str(t.date()), n=0, bias_noc=np.nan, rmse_noc=np.nan,
                    n_slip=0, slip_diffs=[])

    merged = pd.merge_asof(
        day_series.sort_values("report_time"), noc.sort_values("dt"),
        left_on="report_time", right_on="dt", tolerance=pd.Timedelta("10min"), direction="nearest",
    ).dropna(subset=["NOC_Obs"])

    err = merged["finallevel"] - merged["NOC_Obs"]
    bias_noc = float(err.mean()) if len(err) else np.nan
    rmse_noc = float(np.sqrt((err ** 2).mean())) if len(err) else np.nan

    day_slip = slip[(slip["dt"] >= t) & (slip["dt"] < t + pd.Timedelta(days=1))]
    slip_diffs = []
    for _, row in day_slip.iterrows():
        near = day_series.iloc[(day_series["report_time"] - row["dt"]).abs().argsort()[:1]]
        if len(near) and abs((near["report_time"].iloc[0] - row["dt"]).total_seconds()) <= 15 * 60:
            slip_diffs.append(float(near["finallevel"].iloc[0] - row["height"]))

    return dict(
        targetdate=str(t.date()), n=len(merged), bias_noc=bias_noc, rmse_noc=rmse_noc,
        n_slip=len(slip_diffs), slip_diffs=slip_diffs,
    )


if __name__ == "__main__":
    import sys
    dates = sys.argv[1:] or ["2026-01-27"]
    for d in dates:
        print(f"=== {d} ===")
        arc_df, series = run_one_day(d, verbose=True)
        print(f"  {len(arc_df)} arc rows, {(arc_df['status']=='ACCEPT').sum()} accepted")
        print(series.to_string())
        result = score_against_ground_truth(series, d)
        print(result)
