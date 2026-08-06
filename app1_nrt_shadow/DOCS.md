# APP1 NRT Shadow Retrieval

Shadow trial of a new GNSS-IR retrieval pipeline, post-processed through
a K2D replica. Logs a raw and K2D-filtered height estimate roughly every
15 minutes to `/data/results.csv`, and publishes them as
`sensor.dji_obs_raw` / `sensor.dji_obs_k2d`. Reads `sensor.forcing_surge`
(V13's live output) to drive the K2D step. Does not modify anything in
`/config`, and does not touch `sensor.gnss15k2d` or any entity it doesn't
itself own.

## Configuration

| Option | Meaning |
|---|---|
| `smb_host` | IP of the Atom mini PC (default `192.168.1.120`) |
| `smb_share` | Samba share name (default `AtomShare`) |
| `smb_username` | Samba account (default `dji`) |
| `smb_password` | **Set this** -- the only thing you need to configure |
| `pred_file_path` | Live tide prediction source, read-only from `/config` (default already correct: `/config/www/gnss5mins.csv`) |

Retrieval cycles run whenever a new RINEX file's own boundary is reached (native ~15min cadence, driven by file arrival, not a separate configurable interval -- see CHANGELOG 0.1.8). The Samba share is checked once shortly after each expected 15-min boundary rather than on a fixed short timer, since files never appear in between anyway -- see CHANGELOG 0.2.1.

## Log output

Watch the add-on's Log tab for lines like:

```
INFO Ingested APP1218a00.26O.gz: 2985 rows
INFO Cycle 2026-08-05 00:00:00: value=1.234m k2d=1.198m n_arcs=41 rate=0.32 flagged=False (0.9s)
```

`flagged=True` means the implied rate of change from the previous cycle
exceeded 2.5 m/hr -- logged for later analysis, not acted on. If
`sensor.forcing_surge` can't be reached, `k2d=None` and the cycle falls
back to logging the raw value only -- a permission/network hiccup on the
K2D side never blocks the raw retrieval or crashes the cycle.

## Data

`/data/results.csv` is the trial log (append-only, mirrored to
`/share/app1_nrt_shadow_results.csv`). `/data/rolling_buffer.parquet`,
`/data/anchor_history.pkl`, and `/data/k2d_state.pkl` are internal state,
persisted so a restart doesn't lose the last several hours of context.
