# APP1 NRT Shadow Retrieval

A Home Assistant add-on running the validated GNSS-IR retrieval pipeline
(wide mask + robust anchor, 2026-08 accuracy campaign) as a **read-only
shadow trial** alongside the existing, live Simon+K2D system. It does not
publish anything to Home Assistant, does not touch any existing sensor,
automation, or `H:\` file beyond a read-only mapping used to read the
live tide prediction. Purpose: log results for a multi-week comparison
before any decision about replacing anything.

## What it does, every ~15 minutes

1. Reads new 15-minute RINEX files from the Atom mini PC's Samba share
   as they appear (nothing changes on the Atom side).
2. Decimates (30s), parses (georinex), computes satellite geometry, and
   appends to a rolling buffer (~26h retained).
3. Runs the validated `invsnr` retrieval (Chivenor+Instow+North mask,
   robust median-of-history anchor, 8h trailing window) using the live
   tide prediction already served at `H:\www\gnss5mins.csv`.
4. Logs the result (timestamp, height, arc count, fit cost, implausible-
   jump flag) to `/data/results.csv` inside the add-on's persistent
   storage.

## Installing

1. In Home Assistant: **Settings → Add-ons → Add-on Store → ⋮ (top
   right) → Repositories**, add:
   `https://github.com/dtr0tmw/app1-nrt-shadow-addon`
2. Find "APP1 NRT Shadow Retrieval" in the store, install it.
3. Open its **Configuration** tab and set `smb_password` to the Atom's
   `dji` account password (this is the only secret the add-on needs;
   it's stored in Supervisor's own add-on config, never in this repo).
4. Start the add-on. Watch its **Log** tab for the first few cycles.

## Checking on it

- **Log tab** (Supervisor UI): live operational status, one line per
  ingested file and per retrieval cycle.
- **`results.csv`**: the actual trial data. Reachable via the add-on's
  `/data` volume (e.g. the Samba/SSH add-on, or `docker cp` if you have
  host access) -- columns: `report_time, value, n_arcs, n_samples, cost,
  roughness, rate_m_per_hr, flagged, cycle_seconds`.

## What it deliberately does NOT do (yet)

- Does not publish an MQTT sensor or anything else visible in HA's own
  dashboards -- logging-only, per your steer, to keep this fully
  additive and reversible during the trial.
- Does not implement the live plausibility gate (reject/hold-last-value
  on an implausible jump) as an actual *control* action -- it logs
  `flagged=True` for those moments so they're visible in the comparison,
  but doesn't act on them. That's a live-deployment concern, not a
  shadow-trial one.
- Does not touch K2D or `gnss_live.yaml` in any way.

## Repository layout

```
repository.yaml              -- HA add-on repository manifest
app1_nrt_shadow/
  config.yaml                -- add-on manifest (options, /config:ro map)
  Dockerfile
  DOCS.md                    -- shown in the Supervisor UI's add-on docs tab
  app/
    main.py                  -- the service loop
    pipeline/                -- ported, validated modules from the main
                                 TidalStudy research repo (geometry, arcfit,
                                 invsnr_fit, masks, tropo_rh, etc.) -- copied,
                                 not symlinked, so this repo is self-contained
```

## Validated configuration this add-on runs (do not change without
## re-testing against the same evidence base)

- Mask: Chivenor + Instow + North (`MASK_NW_REFINED`)
- Refraction: Ulich (1981) + Rueger (2002), GPT2w climatological met
- Knot spacing: 45 min, surge-smoothness regularizer weight 20
- 8h trailing window, robust median anchor over the last 12 cycles
  (~3h of memory at 15-min cadence)
- Full-state anchor weight 50 (height spline + per-stream A/B + roughness)

Result on the 2026-08 mid-July 8-day test: bias +0.246m, RMSE 0.288m
(n=15 vs slipway), 5/184 (2.7%) implausible-jump rate at hourly cadence;
17.9% raw / 4.6% post-K2D at native 15-min cadence with this wide mask.
See `APP1_GNSS-IR_Briefing.md` / `CHANGELOG.md` in the main TidalStudy
repo for the full validation history.
