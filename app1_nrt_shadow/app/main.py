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
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pymap3d as pm
import requests
import smbclient
from scipy import interpolate

from pipeline.arcfit import (
    RX_XYZ, TOTALANTH, get_or_download_navfile, build_predinterp,
)
from pipeline.geometry import SIG_PRIORITY, az_el_series
from pipeline.masks import MASK_NW_REFINED
from pipeline.invsnr_fit import invsnr_day
from pipeline.tropo_rh import site_met, bend_eqn
from pipeline.k2d_replica import k2d_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                     stream=sys.stdout)
log = logging.getLogger("nrt_shadow")
# smbprotocol logs every individual SMB2 protocol exchange (Negotiate,
# Session Setup, Create, Query Directory, Close, ...) at INFO level -- with
# our own root logger at INFO too, one directory listing produces ~15 lines
# of pure protocol noise, drowning out the handful of lines that actually
# matter (Ingested/Cycle/warnings). Found from a real deployment log
# (2026-08-06) where a single poll's listing was indistinguishable from the
# rest of the log at a glance. Raised to WARNING; our own "nrt_shadow"
# logger above is untouched and still logs at INFO.
logging.getLogger("smbprotocol").setLevel(logging.WARNING)
logging.getLogger("smbclient").setLevel(logging.WARNING)

DATA_DIR = "/data"
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
BUFFER_PARQUET = os.path.join(DATA_DIR, "rolling_buffer.parquet")
HISTORY_PICKLE = os.path.join(DATA_DIR, "anchor_history.pkl")
PROCESSED_JSON = os.path.join(DATA_DIR, "processed_files.json")
LAST_REPORT_JSON = os.path.join(DATA_DIR, "last_report.json")
K2D_STATE_PICKLE = os.path.join(DATA_DIR, "k2d_state.pkl")
NAV_DIR = os.path.join(DATA_DIR, "nav")

# K2D post-processing (Track A, CHANGELOG entry 53/55): forcing_surge (V13's
# separate 8-feature Ridge regression over met/river data) is deliberately
# NOT reimplemented here -- read live from the real deployed sensor instead,
# same design philosophy k2d_replica.py itself documents ("decouples testing
# K2D's own logic from re-deriving V13"). sensor.forcing_surge is V13's own
# direct output (David confirmed, 2026-08-06), not the gnss15k2d K2D sensor's
# own `forcing` attribute -- the latter only updates on K2D's own ~39min-
# lagged cadence (measured directly, entry 53); this dedicated sensor should
# be more current. Even so, the forcing value used each cycle can still be a
# little stale relative to our own report time -- accepted, since surge
# varies over hours, not minutes.
K2D_FORCING_ENTITY = "sensor.forcing_surge"
HA_API_BASE = "http://supervisor/core/api"
DJI_OBS_RAW_ENTITY = "sensor.dji_obs_raw"
DJI_OBS_K2D_ENTITY = "sensor.dji_obs_k2d"

# Validated 2026-08 campaign config (CHANGELOG entries 39-46). Do not change
# without re-validating against the same evidence base -- see the briefing.
RETRIEVAL_KWARGS = dict(sectors=MASK_NW_REFINED, knot_minutes=45, reg_smooth=20.0)
WINDOW_HOURS = 8
STATE_ANCHOR_WEIGHT = 50.0
N_HISTORY = 12          # ~3h of anchor memory at 15-min report cadence (entry 45)
BUFFER_RETAIN_HOURS = 26  # window + anchor lookback + margin
RATE_FLAG_M_PER_HR = 2.5  # same threshold used throughout the validation

# RINEX files only ever appear on a fixed 15-min cadence (never in between),
# and land at/very near their own nominal boundary (confirmed directly from
# real logs, entry 47/49: seconds, not minutes, of latency). Polling at a
# fixed short interval regardless of the clock (the original design) mostly
# just re-lists an unchanged directory and floods the log with smbprotocol's
# own per-call protocol noise for no benefit -- checking once shortly after
# each expected boundary is both far cheaper and, if anything, more
# responsive than polling blindly. Replaces the removed
# poll_interval_seconds option (same reasoning as report_interval_minutes's
# removal in v0.1.8).
POLL_MARGIN_SECONDS = 20


def utcnow_naive():
    """pd.Timestamp.utcnow() is tz-AWARE (unlike stdlib datetime.utcnow());
    buffer['time'] (built from georinex's RINEX timestamps) is naive, so
    comparing them directly raises TypeError. Found from a real deployment
    crash loop -- every cycle's buffer-trim comparison failed silently
    into the error log until this was fixed. Keep all live-clock reads
    naive for consistency with the buffer."""
    return pd.Timestamp.utcnow().tz_localize(None)


# ---------------------------------------------------------------------------
# Home Assistant Core API (via Supervisor's proxy -- SUPERVISOR_TOKEN is
# auto-injected into every add-on container when config.yaml declares
# homeassistant_api: true; no separate credential needed)
# ---------------------------------------------------------------------------

def ha_get_state(entity_id):
    """GET a live HA entity's {state, attributes, ...}. Returns None (logged,
    not raised) on any failure -- a permission/network hiccup here must not
    crash a cycle that would otherwise have a perfectly good raw retrieval."""
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        log.warning("SUPERVISOR_TOKEN not set -- is homeassistant_api: true in config.yaml?")
        return None
    try:
        r = requests.get(f"{HA_API_BASE}/states/{entity_id}",
                          headers={"Authorization": f"Bearer {token}"}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"ha_get_state({entity_id}) failed: {type(e).__name__}: {e}")
        return None


def ha_set_state(entity_id, state, attributes=None):
    """POST a new state for an entity this add-on owns (dji_obs_raw/k2d) --
    creates it on first use, same as any other HA REST-API-published sensor.
    Best-effort: logs and continues on failure, never raises."""
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return
    try:
        r = requests.post(f"{HA_API_BASE}/states/{entity_id}",
                           headers={"Authorization": f"Bearer {token}"},
                           json=dict(state=state, attributes=attributes or {}), timeout=10)
        r.raise_for_status()
    except Exception as e:
        log.warning(f"ha_set_state({entity_id}) failed: {type(e).__name__}: {e}")


def get_forcing_surge():
    """Current V13 surge forcing, read straight from the live deployed
    sensor (David confirmed sensor.forcing_surge is V13's own direct output,
    2026-08-06) -- deliberately not reimplemented here, see K2D_FORCING_ENTITY
    docstring above. Returns None (k2d_step tolerates this -- see its own
    docstring) if the sensor is unavailable or unparseable."""
    d = ha_get_state(K2D_FORCING_ENTITY)
    if d is None:
        return None
    try:
        return float(d["state"])
    except (KeyError, ValueError, TypeError):
        log.warning(f"{K2D_FORCING_ENTITY} state not a number: {d.get('state')!r}")
        return None


def load_k2d_state():
    if os.path.exists(K2D_STATE_PICKLE):
        with open(K2D_STATE_PICKLE, "rb") as f:
            return pickle.load(f)
    return dict(state_missing=True)  # proper cold-start seed, see k2d_step


def save_k2d_state(state):
    with open(K2D_STATE_PICKLE, "wb") as f:
        pickle.dump(state, f)


def seconds_until_next_check():
    """Next 15-min boundary + a small margin, not a fixed poll interval --
    see POLL_MARGIN_SECONDS. Floors a minimum of 5s in case a cycle's own
    processing time already ran past the intended check point."""
    now = utcnow_naive()
    next_boundary = now.ceil("15min")
    if next_boundary <= now:
        next_boundary += pd.Timedelta(minutes=15)
    target = next_boundary + pd.Timedelta(seconds=POLL_MARGIN_SECONDS)
    return max(5.0, (target - now).total_seconds())


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


HOUR_MASK = "abcdefghijklmnopqrstuvwx"  # matches pipeline/concat_rinex_day.py's convention


def rinex_file_end_time(fn):
    """The file's own nominal END boundary -- start-of-span + 15min -- parsed
    straight from its filename: APP1{doy}{hourletter}{minute:02d}.{yy}O[.gz],
    e.g. 'APP1216u30.26O.gz' covers [20:30,20:45) UTC on day 216 of 2026
    ('u' = 20th letter = hour 20). Confirmed against real log evidence
    before deploying: that exact file was ingested at 20:44:57 UTC, ~3s
    before this function's computed boundary of 20:45:00.

    Used instead of flooring the buffer's raw max sample timestamp (the
    original v1 approach) -- that requires the NEXT file to have arrived
    before a boundary is recognised at all (files only exist as complete
    15-min batches, per hardware.txt's 15-min upload cron), adding a full
    avoidable ~15min to every report's latency for no accuracy benefit.
    Keying off the arriving file's own known boundary directly removes
    that extra lag."""
    base = fn.split(".")[0]  # "APP1216u30"
    doy = base[4:7]
    hour = HOUR_MASK.index(base[7])
    minute = int(base[8:10])
    yy = fn.split(".")[1][:2]  # "26" from "26O" or "26O.gz"
    day = pd.Timestamp(datetime.strptime(f"{yy}{doy}", "%y%j"))
    return day + pd.Timedelta(hours=hour, minutes=minute + 15)


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
    now = utcnow_naive()
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
    last_value, last_report_time, latest_file_end = None, None, None
    if os.path.exists(LAST_REPORT_JSON):
        d = json.load(open(LAST_REPORT_JSON))
        last_value = d.get("last_value")
        if d.get("last_report_time"):
            last_report_time = pd.Timestamp(d["last_report_time"])
        if d.get("latest_file_end"):
            latest_file_end = pd.Timestamp(d["latest_file_end"])
    return buffer, history, processed, last_value, last_report_time, latest_file_end


def save_buffer(buffer):
    buffer.to_parquet(BUFFER_PARQUET, index=False)


def save_history(history):
    with open(HISTORY_PICKLE, "wb") as f:
        pickle.dump(history, f)


def save_processed(processed):
    with open(PROCESSED_JSON, "w") as f:
        json.dump(sorted(processed), f)


def save_last_report(last_value, last_report_time, latest_file_end):
    with open(LAST_REPORT_JSON, "w") as f:
        json.dump(dict(last_value=last_value,
                        last_report_time=str(last_report_time) if last_report_time is not None else None,
                        latest_file_end=str(latest_file_end) if latest_file_end is not None else None), f)


SHARE_RESULTS_CSV = "/share/app1_nrt_shadow_results.csv"


def append_result(row):
    # pandas' default CSV datetime formatting drops the time-of-day when a
    # column's only value happens to be exactly midnight (e.g. writes bare
    # "2026-08-06" instead of "2026-08-06 00:00:00") -- invisible normally
    # since almost every row appends a single value, but caught a real
    # report_time=00:00:00 row doing exactly this during a health-check
    # analysis, breaking a naive pd.read_csv() parse of the file. Format
    # explicitly so every row is unambiguous regardless of time-of-day.
    row = dict(row)
    row["report_time"] = pd.Timestamp(row["report_time"]).strftime("%Y-%m-%d %H:%M:%S")
    header = not os.path.exists(RESULTS_CSV)
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=header, index=False)
    # Also mirror to /share -- /data is a private per-add-on volume with no
    # network path; /share is visible to other add-ons (File editor, Samba
    # share, etc.) so David can actually read the trial results without
    # needing container/host shell access.
    try:
        import shutil
        shutil.copy(RESULTS_CSV, SHARE_RESULTS_CSV)
    except Exception as e:
        log.warning(f"Could not mirror results.csv to /share: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Nav (daily broadcast ephemeris)
# ---------------------------------------------------------------------------

class NavUnavailableError(RuntimeError):
    """Nav not (yet) published -- transient, worth retrying later, unlike a
    genuinely corrupt RINEX file. Kept distinct from other ingest failures
    so main()'s poison-file guard doesn't permanently skip files that only
    failed because nav wasn't out yet (a real bug: every file that failed
    with the old plain RuntimeError got marked processed forever, even
    after the underlying nav-lookback fix landed)."""


_nav_cache = {}


NAV_MAX_LOOKBACK_DAYS = 5  # see docstring below


def get_nav(targetdate):
    """This combined ('01D') BRDC nav product from IGS/BKG has a real
    publication lag of ~2 days, not "published later the same day" as a
    single-day fallback (v0.1.3) assumed -- confirmed directly from a real
    deployment stall (2026-08-05 08:45: both day 217 AND day 216 still
    404, day 215 the most recent actually available). Walks back day by
    day until it finds one that exists, rather than trying only one
    fallback. Satellite orbits don't change enough over a few days to
    meaningfully affect az/el pointing geometry for this application
    (unlike precise-orbit work, GNSS-IR reflectometry doesn't need cm-
    level orbit accuracy). Known simplification: once cached, does not
    later retry for a more-current file even after it becomes available
    -- acceptable given the minor effect, revisit if it matters."""
    key = str(pd.Timestamp(targetdate).date())
    if key not in _nav_cache:
        import georinex as gr
        navfile = None
        used_date = None
        for back in range(NAV_MAX_LOOKBACK_DAYS + 1):
            try_date = pd.Timestamp(targetdate) - pd.Timedelta(days=back)
            navfile = get_or_download_navfile(NAV_DIR, try_date, verbose=True)
            if navfile:
                used_date = try_date
                break
        if not navfile:
            raise NavUnavailableError(f"No nav file available for {targetdate} within "
                                       f"{NAV_MAX_LOOKBACK_DAYS} days back")
        if used_date.date() != pd.Timestamp(targetdate).date():
            log.warning(f"No nav for {key} yet (not published) -- "
                        f"using {used_date.date()}'s nav instead")
        _nav_cache.clear()  # only keep one day's nav resident at a time
        _nav_cache[key] = gr.load(navfile, use=["G", "E", "R", "C"])
        log.info(f"Loaded nav for {key} (from {used_date.date()})")
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
    if "kval" not in info:
        # invsnr_day's early-return "fail" dicts (no samples/too few arcs/
        # etc.) never populate kval/knots/t0 -- genuinely no result.
        return None, info

    # Evaluate the fitted spline exactly AT T, not at invsnr_day's own
    # internal report grid's last point (series["finallevel"].iloc[-1]),
    # which floors the actual last SAMPLE time to 15min and so lands one
    # bin (~15min) SHORT of T now that T is set from the arriving file's
    # own boundary (rinex_file_end_time) rather than a floor of the raw
    # buffer max -- using the old grid here would have silently thrown
    # away the entire latency benefit of that fix. Re-derives the same
    # cubic interpolation invsnr_day uses internally from info["t0"]/
    # ["knots"]/["kval"], entirely on the main.py side so the shared,
    # backtested invsnr_fit.py used throughout the whole accuracy
    # campaign is untouched.
    t_abs_sec = ((info["t0"] + pd.to_timedelta(info["knots"], unit="s"))
                 - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    t_abs_sec = t_abs_sec.to_numpy(float)
    hf = interpolate.interp1d(t_abs_sec, info["kval"], kind="cubic", fill_value="extrapolate")
    T_sec = (T - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    T_sec = float(np.clip(T_sec, t_abs_sec.min(), t_abs_sec.max()))
    value = float(TOTALANTH - hf(T_sec))
    return value, info


def run_k2d(value, T, pred_path, k2d_state):
    """One streaming K2D cycle on top of DJI Obs's own raw value -- Track A,
    CHANGELOG entries 53/55: confirmed on real live data to roughly halve
    bias/RMSE against Simon's obs and eliminate implausible jumps in the
    tested stretch. Pred comes from OUR OWN predinterp (always available at
    the exact report time T, unlike Simon's own sensor which only updates
    on his ~39min-lagged cadence -- entry 53) -- rebuilt fresh each cycle,
    consistent with how the rest of the pipeline already re-reads it (cheap:
    a CSV read + interp1d build), so a growing live feed file never goes
    stale here. forcing comes from the live sensor.forcing_surge (V13's own
    output, not reimplemented here -- see K2D_FORCING_ENTITY docstring).
    Streaming (one k2d_step() call per new cycle, persisted state) rather
    than replaying the whole history every time -- matches how the real
    deployed filter itself operates and stays cheap over weeks of runtime."""
    predinterp = build_predinterp(pred_path)
    T_unix = (T - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    pred_T = float(predinterp(np.array([T_unix]))[0])
    forcing = get_forcing_surge()
    kalman, new_state, diag = k2d_step(value, pred_T, forcing, k2d_state)
    return kalman, pred_T, forcing, new_state, diag


def main():
    opts = load_options()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NAV_DIR, exist_ok=True)
    fetch_dir = os.path.join(DATA_DIR, "incoming")

    # Retry startup SMB connect rather than crashing the container outright --
    # a wrong/not-yet-set password or a momentary network blip on the Atom
    # side must not end weeks of otherwise-unattended trial operation.
    backoff = 10
    while True:
        try:
            smb_connect(opts)
            break
        except Exception as e:
            log.error(f"SMB connect failed ({type(e).__name__}: {e}); "
                      f"check smb_password in the add-on Configuration tab. "
                      f"Retrying in {backoff}s.")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)

    buffer, history, processed, last_value, last_report_time, latest_file_end = load_state()
    k2d_state = load_k2d_state()
    log.info(f"Startup: buffer={len(buffer)} rows, history={len(history)} cycles, "
             f"{len(processed)} files already processed, "
             f"last_report_time={last_report_time}, latest_file_end={latest_file_end}, "
             f"k2d_state_missing={k2d_state.get('state_missing', False)}")

    pred_path = opts["pred_file_path"]

    while True:
        try:
            candidates = smb_list_candidate_files(opts)
            new_files = [(dn, fn) for dn, fn in candidates if fn not in processed]
            for dn, fn in sorted(new_files):
                local_gz, decimated = None, None
                try:
                    local_gz = smb_fetch(opts, dn, fn, fetch_dir)
                    decimated = decimate_text(local_gz)
                    # Nav date must match the FILE's own YYDOY directory (dn),
                    # not wall-clock "now" -- a backlog file from yesterday's
                    # directory needs yesterday's nav, and today's nav often
                    # isn't fully published yet this early in the day (bug
                    # found from a real deployment: every backlog file failed
                    # with "No nav file available" because it was requesting
                    # today's nav for yesterday's data).
                    from datetime import datetime as _dt
                    file_date = pd.Timestamp(_dt.strptime(dn, "%y%j"))
                    nav = get_nav(file_date)
                    new_rows = extract_file(decimated, nav)
                    if not new_rows.empty:
                        buffer = pd.concat([buffer, new_rows], ignore_index=True)
                        log.info(f"Ingested {fn}: {len(new_rows)} rows")
                    processed.add(fn)
                    # Report as soon as THIS file's own known boundary is
                    # reached, not once we've floored the buffer's raw max
                    # sample time (which needs the NEXT file to arrive first
                    # -- an avoidable extra ~15min of latency, since files
                    # only exist as complete 15-min batches to begin with).
                    file_end = rinex_file_end_time(fn)
                    if latest_file_end is None or file_end > latest_file_end:
                        latest_file_end = file_end
                except NavUnavailableError as e:
                    log.warning(f"Deferring {fn} (not marking processed): {e}")
                except Exception as e:
                    log.warning(f"Failed to ingest {fn}: {type(e).__name__}: {e}")
                    processed.add(fn)  # don't retry a genuinely poison file forever
                finally:
                    # Always clean up temp files, even on failure -- previously
                    # only happened on success, leaking files in /data/incoming
                    # on every ingest failure over weeks of unattended operation.
                    for p in (local_gz, decimated):
                        if p and os.path.exists(p):
                            os.remove(p)

            if len(buffer):
                cutoff = utcnow_naive() - pd.Timedelta(hours=BUFFER_RETAIN_HOURS)
                buffer = buffer[buffer["time"] >= cutoff].reset_index(drop=True)

            if len(buffer) and latest_file_end is not None:
                candidate_report = latest_file_end
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

                        try:
                            kalman, pred_T, forcing, k2d_state, k2d_diag = run_k2d(
                                value, candidate_report, pred_path, k2d_state)
                            save_k2d_state(k2d_state)
                        except Exception as e:
                            log.warning(f"K2D step failed (continuing with raw only): "
                                        f"{type(e).__name__}: {e}")
                            kalman, pred_T, forcing, k2d_diag = None, None, None, {}

                        append_result(dict(
                            report_time=candidate_report, value=value,
                            n_arcs=info.get("n_arcs"), n_samples=info.get("n_samples"),
                            cost=info.get("cost"), roughness=info.get("roughness"),
                            rate_m_per_hr=rate, flagged=flagged, cycle_seconds=round(elapsed, 2),
                            k2d_value=kalman, k2d_status=k2d_diag.get("status"),
                            pred=pred_T, forcing=forcing,
                        ))
                        log.info(f"Cycle {candidate_report}: value={value:.3f}m "
                                 f"k2d={kalman if kalman is None else round(kalman,3)}m "
                                 f"n_arcs={info.get('n_arcs')} rate={rate} flagged={flagged} "
                                 f"({elapsed:.1f}s)")

                        ha_set_state(DJI_OBS_RAW_ENTITY, round(value, 4), dict(
                            unit_of_measurement="m", friendly_name="DJI Obs (raw)",
                            report_time=str(candidate_report), n_arcs=info.get("n_arcs"),
                            rate_m_per_hr=rate, flagged=flagged))
                        if kalman is not None:
                            ha_set_state(DJI_OBS_K2D_ENTITY, round(kalman, 4), dict(
                                unit_of_measurement="m", friendly_name="DJI Obs (K2D filtered)",
                                report_time=str(candidate_report), status=k2d_diag.get("status"),
                                innovation=k2d_diag.get("innovation"),
                                adaptive_r=k2d_diag.get("adaptive_r")))

                        history.append(info)
                        history = history[-N_HISTORY:]
                        last_value, last_report_time = value, candidate_report
                        save_history(history)
                        save_last_report(last_value, last_report_time, latest_file_end)
                    else:
                        log.warning(f"Cycle {candidate_report}: no result "
                                    f"({info.get('fail') if info else 'empty window'})")
                    save_buffer(buffer)

            save_processed(processed)

        except Exception as e:
            log.error(f"Cycle loop error (continuing): {type(e).__name__}: {e}", exc_info=True)

        time.sleep(seconds_until_next_check())


if __name__ == "__main__":
    main()
