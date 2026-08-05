# APP1 NRT Shadow Retrieval

Read-only shadow trial of a new GNSS-IR retrieval pipeline. Logs a height
estimate roughly every 15 minutes to `/data/results.csv`. Publishes
nothing to Home Assistant and does not modify anything in `/config`.

## Configuration

| Option | Meaning |
|---|---|
| `smb_host` | IP of the Atom mini PC (default `192.168.1.120`) |
| `smb_share` | Samba share name (default `AtomShare`) |
| `smb_username` | Samba account (default `dji`) |
| `smb_password` | **Set this** -- the only thing you need to configure |
| `poll_interval_seconds` | How often to check for new RINEX files (default 60) |
| `pred_file_path` | Live tide prediction source, read-only from `/config` (default already correct: `/config/www/gnss5mins.csv`) |

Retrieval cycles run whenever a new RINEX file's own boundary is reached (native ~15min cadence, driven by file arrival, not a separate configurable interval -- see CHANGELOG 0.1.8).

## Log output

Watch the add-on's Log tab for lines like:

```
INFO Ingested APP1218a00.26O.gz: 2985 rows
INFO Cycle 2026-08-05 00:00:00: value=1.234m n_arcs=41 rate=0.32 flagged=False (0.9s)
```

`flagged=True` means the implied rate of change from the previous cycle
exceeded 2.5 m/hr -- logged for later analysis, not acted on.

## Data

`/data/results.csv` is the trial log (append-only). `/data/rolling_buffer.parquet`
and `/data/anchor_history.pkl` are internal state, persisted so a restart
doesn't lose the last several hours of context.
