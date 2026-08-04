"""
main.py -- APP1 NRT Shadow Retrieval add-on

Reads new RINEX files from the Atom mini PC's Samba share as they appear,
runs the validated wide-mask + robust-anchor invsnr retrieval (2026-08
campaign: bias +0.246/rmse 0.288 raw at native 15-min cadence, n=15 vs
slipway truth, 5/184 implausible-jump rate), and LOGS the result -- does
not publish anything to Home Assistant, does not touch any existing
sensor, automation, or H:\\ file beyond a READ-ONLY mapping of /config
(for the live tide prediction, H:\\www\\gnss5mins.csv).

Purpose: a multi-week, fully reversible, zero-contact-with-the-live-system
shadow trial, to compare against Simon's Obs / the existing K2D output /
slipway ground truth before any decision about replacing anything.

State persisted to /data (survives add-on restarts/updates):
  - rolling_buffer.parquet: recent extracted SNR/geometry samples
  - anchor_history.pkl:     last N cycles' invsnr_day() info dicts, for
                             the robust median anchor (entry 43)
  - processed_files.json:   which RINEX files have already been ingested
  - results.csv:            the actual trial log -- append-only

Run as: python -u main.py  (the add-on's CMD)

Change log
----------
v1  2026-08-05  Claude Code. Initial version.
"""
import json
import logging
import os
import pickle
import sys
import time
import warnings
from collections import deque

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pymap3d as pm
import smbclient

from pipeline.arcfit import (
    RX_XYZ, get_or_download_navfile, build_predinterp,
)
from pipeline.geometry import SIG_PRIORITY, az_el_series
from pipeline.masks import MASK_NW_REFINED
from pipeline.invsnr_fit import invsnr_day
from pipeline.tropo_rh import site_met, bend_eqn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                     stream=sys.stdout)
log = logging.getLogger("nrt_shadow")

DATA_DIR = "/data"
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
BUFFER_PARQUET = os.path.join(DATA_DIR, "rolling_buffer.parquet")
HISTORY_PICKLE = os.path.join(DATA_DIR, "anchor_history.pkl")
PROCESSED_JSON = os.path.join(DATA_DIR, "processed_files.json")
NAV_DIR = os.path.join(DATA_DIR, "nav")

# Validated 2026-08 campaign config (CHANGELOG entries 39-46). Do not change
# without re-validating against the same evidence base -- see the briefing.
RETRIEVAL_KWARGS = dict(sectors=MASK_NW_REFINED, knot_minutes=45, reg_smooth=20.0)
WINDOW_HOURS = 8
STATE_ANCHOR_WEIGHT = 50.0
N_HISTORY = 12          # ~3h of anchor memory at 15-min report cadence (entry 45)
BUFFER_RETAIN_HOURS = 26  # window + anchor lookback + margin
RATE_FLAG_M_PER_HR = 2.5  # same threshold used throughout the validation


def load_options():
    with open("/data/options.json") as f:
        return json.load(f)


def ulich_rueger_refraction(when):
    """Best validated refraction model (entry 39): Ulich (1981) bending
    driven by Rueger (2002) radio refractivity, with GPT2w climatological
    met at APP1 -- ported here directly (not importing experiment_harness.py,
    which carries a lot of research-only scaffolding not needed live)."""
    p, t, e = site_met(pd.Timestamp(when))
    tk = t + 273.15
    K1r, K2r, K3r = 77.689, 71.2952, 375463.0
    n0 = K1r * (p - e) / tk + K2r * e / tk + K3r * e / tk ** 2  # ppm

    def fn(el):
        el_rad = np.deg2rad(el)
        f = np.cos(el_rad) / (np.sin(el_rad) + 0.00175 * np.tan(np.deg2rad(87.5) - el_rad))
        return el + np.rad2deg(n0 / 1e6 * f)
    return fn


def robust_prior_info(history):
    """Verbatim from pipeline/nrt_simulation.py (entry 43) -- copied rather
    than imported to avoid pulling in that module's backtesting-oriented
    dependencies (june26_compare, experiment_harness)."""
    from scipy import interpolate
    if len(history) == 1:
        return history[0]
    spans = []
    for info in history:
        t_abs = ((info["t0"] + pd.to_timedelta(info["knots"], unit="s"))
                  - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        spans.append((t_abs.to_numpy(float).min(), t_abs.to_numpy(float).max()))
    lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
    if hi <= lo:
        return history[-1]
    grid = np.linspace(lo, hi, max(4, int((hi - lo) / (15 * 60)) + 1))
    curves = []
    for info in history:
        t_abs = ((info["t0"] + pd.to_timedelta(info["knots"], unit="s"))
                  - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        f = interpolate.interp1d(t_abs.to_numpy(float), info["kval"], kind="cubic", fill_value="extrapolate")
        curves.append(f(grid))
    kval_med = np.median(np.vstack(curves), axis=0)
    t0 = pd.Timestamp("1970-01-01") + pd.to_timedelta(grid[0], unit="s")
    knots = grid - grid[0]
    from collections import defaultdict
    ab_vals = defaultdict(list)
    for info in history:
        for j, s in enumerate(info.get("streams", [])):
            ab_vals[s].append((info["ab"][2 * j], info["ab"][2 * j + 1]))
    streams = [s for s, v in ab_vals.items() if len(v) >= min(2, len(history))]
    ab = np.empty(2 * len(streams))
    for j, s in enumerate(streams):
        arr = np.array(ab_vals[s])
        ab[2 * j], ab[2 * j + 1] = np.median(arr[:, 0]), np.median(arr[:, 1])
    rough_vals = [info["roughness"] for info in history if info.get("roughness") is not None]
    roughness = float(np.median(rough_vals)) if rough_vals else None
    return dict(t0=t0, knots=knots, kval=kval_med, streams=streams, ab=ab, roughness=roughness)


# ---------------------------------------------------------------------------
# SMB
# ---------------------------------------------------------------------------

def smb_connect(opts):
    smbclient.ClientConfig(username=opts["smb_username"], password=opts["smb_password"])
    smbclient.register_session(opts["smb_host"], username=opts["smb_username"],
                                password=opts["smb_password"])
    log.info(f"SMB session established with {opts['smb_host']} as {opts['smb_username']}")


def smb_root(opts):
    return f"\\\\{opts['smb_host']}\\{opts['smb_share']}"


def smb_list_candidate_files(opts):
    """APP1's native 15-min files live in YYDOY-named subdirectories
    (D:\\GNSS\\DATA's own layout, mirrored on the Atom). Only look at
    today's and yesterday's directories -- more than enough margin for a
    15-min-cadence poll."""
    now = pd.Timestamp.utcnow()
    dirs = [(now - pd.Timedelta(days=d)) for d in (1, 0)]
    dirnames = [f"{d.strftime('%y')}{d.strftime('%j')}" for d in dirs]
    root = smb_root(opts)
    out = []
    for dn in dirnames:
        try:
            for fn in smbclient.listdir(f"{root}\\{dn}"):
                if fn.upper().endswith("O.GZ"):
                    out.append((dn, fn))
        except Exception as e:
            log.debug(f"listdir failed for {dn}: {e}")
    return out


def smb_fetch(opts, dn, fn, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, fn)
    remote_path = f"{smb_root(opts)}\\{dn}\\{fn}"
    with smbclient.open_file(remote_path, mode="rb") as rf, open(local_path, "wb") as lf:
        lf.write(rf.read())
    return local_path


# ---------------------------------------------------------------------------
# RINEX extraction (single file, decimated -- entry 47 benchmark: ~1s total)
# ---------------------------------------------------------------------------

def decimate_text(gz_path, decimate_s=30):
    """Text-level epoch filtering BEFORE georinex parses anything -- the
    already-established fix (concat_rinex_day.py) for georinex's expensive
    per-epoch xarray merge. Native cadence is 2s; second%30==0 is the
    coarsest cleanly-achievable decimation on that grid (CLAUDE.md notes
    15s silently collapses to 30s spacing for the same reason)."""
    import gzip
    from datetime import datetime
    out_path = gz_path[:-6] + "_dec.26O" if gz_path.lower().endswith(".26o.gz") else gz_path + "_dec.26O"
    with gzip.open(gz_path, "rt", errors="replace") as f:
        lines = f.readlines()
    end_idx = next(j for j, l in enumerate(lines) if "END OF HEADER" in l)
    header, body = lines[:end_idx + 1], lines[end_idx + 1:]
    with open(out_path, "w", encoding="ascii", errors="replace") as out:
        out.writelines(header)
        keep = False
        for l in body:
            if l.startswith(">"):
                parts = l.split()
                sec = float(parts[6])
                keep = int(sec) % decimate_s == 0
            if keep:
                out.write(l)
    return out_path


def extract_file(decimated_path, nav):
    import georinex as gr
    obs = gr.load(decimated_path)
    rx_lat, rx_lon, rx_alt = pm.ecef2geodetic(*RX_XYZ)
    frames = []
    for sv in obs.sv.values:
        svstr = str(sv)
        const = svstr[0]
        if const not in SIG_PRIORITY:
            continue
        primary_sig = next((s for s in SIG_PRIORITY[const] if s in obs.data_vars), None)
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
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state():
    buffer = pd.read_parquet(BUFFER_PARQUET) if os.path.exists(BUFFER_PARQUET) else pd.DataFrame()
    history = pickle.load(open(HISTORY_PICKLE, "rb")) if os.path.exists(HISTORY_PICKLE) else []
    processed = set(json.load(open(PROCESSED_JSON))) if os.path.exists(PROCESSED_JSON) else set()
    return buffer, history, processed


def save_buffer(buffer):
    buffer.to_parquet(BUFFER_PARQUET, index=False)


def save_history(history):
    with open(HISTORY_PICKLE, "wb") as f:
        pickle.dump(history, f)


def save_processed(processed):
    with open(PROCESSED_JSON, "w") as f:
        json.dump(sorted(processed), f)


def append_result(row):
    header = not os.path.exists(RESULTS_CSV)
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Nav (daily broadcast ephemeris)
# ---------------------------------------------------------------------------

_nav_cache = {}


def get_nav(targetdate):
    key = str(pd.Timestamp(targetdate).date())
    if key not in _nav_cache:
        import georinex as gr
        navfile = get_or_download_navfile(NAV_DIR, targetdate, verbose=True)
        if not navfile:
            raise RuntimeError(f"No nav file available for {targetdate}")
        _nav_cache.clear()  # only keep one day's nav resident at a time
        _nav_cache[key] = gr.load(navfile, use=["G", "E", "R", "C"])
        log.info(f"Loaded nav for {key}")
    return _nav_cache[key]


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_cycle(T, buffer, history, pred_path):
    window_start = T - pd.Timedelta(hours=WINDOW_HOURS)
    window_df = buffer[(buffer["time"] >= window_start) & (buffer["time"] <= T)]
    if window_df.empty:
        return None, None

    prior = robust_prior_info(history[-N_HISTORY:]) if history else None
    series, info = invsnr_day(
        str(T.date()), refraction_fn=ulich_rueger_refraction(T),
        predfile_override=pred_path, cache_df=window_df,
        start_time=window_start, end_time=T,
        prior_info=prior, state_anchor_weight=STATE_ANCHOR_WEIGHT,
        **RETRIEVAL_KWARGS,
    )
    if series.empty:
        return None, info
    value = float(series["finallevel"].iloc[-1])
    return value, info


def main():
    opts = load_options()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NAV_DIR, exist_ok=True)
    fetch_dir = os.path.join(DATA_DIR, "incoming")

    smb_connect(opts)
    buffer, history, processed = load_state()
    log.info(f"Startup: buffer={len(buffer)} rows, history={len(history)} cycles, "
             f"{len(processed)} files already processed")

    last_value = None
    last_report_time = None
    poll_s = opts["poll_interval_seconds"]
    report_min = opts["report_interval_minutes"]
    pred_path = opts["pred_file_path"]

    while True:
        try:
            candidates = smb_list_candidate_files(opts)
            new_files = [(dn, fn) for dn, fn in candidates if fn not in processed]
            for dn, fn in sorted(new_files):
                try:
                    local_gz = smb_fetch(opts, dn, fn, fetch_dir)
                    decimated = decimate_text(local_gz)
                    nav = get_nav(pd.Timestamp.utcnow())
                    new_rows = extract_file(decimated, nav)
                    if not new_rows.empty:
                        buffer = pd.concat([buffer, new_rows], ignore_index=True)
                        log.info(f"Ingested {fn}: {len(new_rows)} rows")
                    processed.add(fn)
                    os.remove(local_gz)
                    os.remove(decimated)
                except Exception as e:
                    log.warning(f"Failed to ingest {fn}: {type(e).__name__}: {e}")
                    processed.add(fn)  # don't retry a poison file forever

            if len(buffer):
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=BUFFER_RETAIN_HOURS)
                buffer = buffer[buffer["time"] >= cutoff].reset_index(drop=True)

            if len(buffer):
                latest_data_time = buffer["time"].max()
                candidate_report = latest_data_time.floor(f"{report_min}min")
                if last_report_time is None or candidate_report > last_report_time:
                    t0 = time.time()
                    value, info = run_cycle(candidate_report, buffer, history, pred_path)
                    elapsed = time.time() - t0
                    if value is not None:
                        rate = None
                        flagged = False
                        if last_value is not None and last_report_time is not None:
                            dt_hr = (candidate_report - last_report_time).total_seconds() / 3600.0
                            if dt_hr > 0:
                                rate = (value - last_value) / dt_hr
                                flagged = abs(rate) > RATE_FLAG_M_PER_HR
                        append_result(dict(
                            report_time=candidate_report, value=value,
                            n_arcs=info.get("n_arcs"), n_samples=info.get("n_samples"),
                            cost=info.get("cost"), roughness=info.get("roughness"),
                            rate_m_per_hr=rate, flagged=flagged, cycle_seconds=round(elapsed, 2),
                        ))
                        log.info(f"Cycle {candidate_report}: value={value:.3f}m "
                                 f"n_arcs={info.get('n_arcs')} rate={rate} flagged={flagged} "
                                 f"({elapsed:.1f}s)")
                        history.append(info)
                        history = history[-N_HISTORY:]
                        last_value, last_report_time = value, candidate_report
                        save_history(history)
                    else:
                        log.warning(f"Cycle {candidate_report}: no result "
                                    f"({info.get('fail') if info else 'empty window'})")
                    save_buffer(buffer)

            save_processed(processed)

        except Exception as e:
            log.error(f"Cycle loop error (continuing): {type(e).__name__}: {e}", exc_info=True)

        time.sleep(poll_s)


if __name__ == "__main__":
    main()
