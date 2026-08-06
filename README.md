# APP1 NRT Shadow Retrieval

A Home Assistant add-on running the validated GNSS-IR retrieval pipeline
(wide mask + robust anchor, 2026-08 accuracy campaign), post-processed
through a K2D replica (Track A), as a **shadow trial** alongside the
existing, live Simon/NOC+K2D system. It does not touch, modify, or depend on
any *existing* sensor, automation, or `H:\` file beyond a read-only
mapping used to read the live tide prediction -- it reads one existing
sensor (`sensor.forcing_surge`, V13's live surge output) and publishes
two new, additive sensors of its own. Purpose: a multi-week comparison
before any decision about replacing anything.

## What it does, every ~15 minutes

1. Reads new 15-minute RINEX files from the Atom mini PC's Samba share
   as they appear (nothing changes on the Atom side).
2. Decimates (30s), parses (georinex), computes satellite geometry, and
   appends to a rolling buffer (~26h retained).
3. Runs the validated `invsnr` retrieval (Chivenor+Instow+North mask,
   robust median-of-history anchor, 8h trailing window). The live tide
   prediction (`H:\www\gnss5mins.csv`) plays two specific roles here, and
   is **not** blended into the reported height directly: (a) upstream,
   per-arc quality control uses the predicted height to resolve the
   periodogram peak-selection ambiguity a single arc's raw SNR data is
   subject to; (b) inside the fit itself, a surge-smoothness regularizer
   penalizes rapid knot-to-knot changes in *(fitted height − predicted
   height)* rather than in the fitted height directly -- since the tide's
   own fast, large-amplitude curvature is known and free (tracking the
   prediction's shape costs nothing), only the slowly-varying leftover
   (surge) is discouraged from jittering. The reported value is always
   `TOTALANTH − h(T)` from the fitted spline alone; if the SNR
   observations disagree with the prediction, the fit follows the
   observations -- the regularizer penalizes the disagreement's rate of
   change, never its size.
4. Post-processes that raw value through a K2D replica (`pipeline/k2d_replica.py`,
   the same constants as the deployed filter), reading live surge forcing
   from `sensor.forcing_surge` via Home Assistant's own API -- confirmed
   on real live data (CHANGELOG entries 53/55) to roughly halve bias/RMSE
   against Simon's obs and eliminate implausible jumps in the tested stretch.
5. Logs both the raw and K2D-filtered result to `/data/results.csv`, and
   publishes them as `sensor.dji_obs_raw` / `sensor.dji_obs_k2d` for a
   Lovelace graph card.

## Installing

1. In Home Assistant: **Settings → Add-ons → Add-on Store → ⋮ (top
   right) → Repositories**, add:
   `https://github.com/Dtrotmw/app1-nrt-shadow-addon-`
2. Find "APP1 NRT Shadow Retrieval" in the store, install it.
3. Open its **Configuration** tab and set `smb_password` to the Atom's
   `dji` account password (this is the only secret the add-on needs;
   it's stored in Supervisor's own add-on config, never in this repo).
4. Start the add-on. Watch its **Log** tab for the first few cycles.

## Checking on it

- **Log tab** (Supervisor UI): live operational status, one line per
  ingested file and per retrieval cycle.
- **`results.csv`**: the actual trial data. Reachable at
  `\\<HA-server-IP>\share\app1_nrt_shadow_results.csv` (mirrored there
  automatically) -- columns: `report_time, value, n_arcs, n_samples, cost,
  roughness, rate_m_per_hr, flagged, cycle_seconds, k2d_value, k2d_status,
  pred, forcing`.
- **`sensor.dji_obs_raw` / `sensor.dji_obs_k2d`**: add either to a
  Lovelace history/graph card like any other sensor.

## What it deliberately does NOT do (yet)

- Does not implement the live plausibility gate (reject/hold-last-value
  on an implausible jump) as an actual *control* action -- it logs
  `flagged=True` for those moments so they're visible in the comparison,
  but doesn't act on them. That's a live-deployment concern, not a
  shadow-trial one.
- Does not touch `sensor.gnss15k2d`, `gnss_live.yaml`, or `forcing.yaml`
  in any way -- reads `sensor.forcing_surge` only, never writes to it or
  to any entity it doesn't itself own (`dji_obs_raw`/`dji_obs_k2d`).
- Does not reimplement V13's forcing/surge model -- reads its live output
  from the real sensor instead (same design philosophy `k2d_replica.py`
  documents for testing K2D's own logic in isolation).

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
On real live trial data (2026-08-06, clean 94-point stretch): K2D took
bias +0.249->+0.165, RMSE 0.343->0.195, implausible jumps 5/93->0/93.
See `APP1_GNSS-IR_Briefing.md` / `CHANGELOG.md` in the main TidalStudy
repo for the full validation history.

## Acknowledgments

**Simon Williams (National Oceanography Centre)** is the reason any of
this exists. His own GNSS-IR pipeline, run on the APP and APP1 antennas,
produced the harmonics UKHO currently publishes for Appledore, and his
ongoing ThingSpeak observation feed is the live reference this whole
comparison is checked against. K2D (the Kalman filter) and V13 (the
surge model behind `sensor.forcing_surge`), layered on top of Simon's
harmonics, are David's own work, not Simon's. This trial exists to test
whether specific, narrow refinements to the retrieval feeding into that
system can be evidenced against Simon's own product, using his own data
as the reference -- not as a critique of it. Any of this that turns out
useful is his to take or leave.

The retrieval method itself follows **Strandberg, Hobiger & Haas (2016)**,
*Radio Science* 51, 1286-1296 -- fitting a single continuous reflector-
height curve to all of a window's raw SNR data at once, rather than one
height per satellite arc. The refraction model is **Ulich (1981)** bending
driven by **Rueger (2002)** radio refractivity, with meteorology from the
**GPT2w** climatology grid (**Böhm et al., 2015**). Both the inverse-model
and refraction ports were built reading, line by line, the open-source
implementations in **Kristine Larson et al.'s `gnssrefl`**
(github.com/kristinemlarson/gnssrefl) and **David Purnell et al.'s
`gnssir-rt`** (github.com/purnelldj/gnssir-rt; see also Purnell et al.,
2024, *Geophysical Research Letters*, their real-time GNSS-IR system).
A tropospheric correction from **Santamaria-Gomez & Watson (2017)**,
*GPS Solutions* 21, 451-459, was also tested via `gnssir-rt`'s
implementation but not adopted -- its mean-elevation approximation
doesn't hold at APP1's unusually low reflection geometry. The original
(now superseded) refraction correction was **Bennett (1982)**, "The
Calculation of Astronomical Refraction in Marine Navigation". The
mask-boundary sweep method follows **Altuntas & Tunalioglu (2023)**.

This add-on, and the wider research repository it was extracted from,
were built with **Claude Code** (Anthropic), under David's direction --
he set every research question, evidence bar, and go/no-go call; Claude
wrote the code and ran the experiments.
