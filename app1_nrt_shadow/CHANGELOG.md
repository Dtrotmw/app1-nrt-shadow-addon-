# Changelog

## 0.1.9
- Fixed a CSV formatting bug found during a health-check analysis: pandas
  drops the time-of-day when a single-row CSV append's report_time happens
  to be exactly midnight, writing bare "2026-08-06" instead of
  "2026-08-06 00:00:00" -- broke a naive pd.read_csv() parse of the file.
  report_time is now explicitly formatted on every row.

## 0.1.8
- Removed an avoidable ~15min of reporting latency. Previously, a report for
  boundary T only appeared once the buffer's raw latest-sample timestamp,
  floored to 15min, reached T -- which requires the *next* file to have
  already arrived (files only exist as complete 15-min batches), adding a
  full extra bin of delay for no accuracy benefit. Now keys the report
  boundary directly off each arriving file's own known nominal end time
  (parsed from its filename), so a report appears as soon as its own
  file lands.
- Since the report boundary can now be earlier than invsnr_day's own
  internal grid would produce (its last grid point is the actual last
  sample's time floored to 15min), the reported value is now evaluated
  directly at the report boundary via the fitted spline's own
  parameters, rather than read off that internal grid's last point.
- Removed the now-vestigial `report_interval_minutes` option -- reporting
  cadence is driven by file arrival, not a separate configurable interval.

## 0.1.7
- Fixed a crash when a retrieval window contains zero detected arcs at all
  (not just zero accepted) -- `arc_df` came back with no columns at all in
  that case, so filtering on `arc_df["status"]` raised `KeyError: 'status'`.
  Never hit in backtesting (full-day windows always found some arcs), but a
  short live window can legitimately come up empty. Now handled the same
  way as "zero usable samples" already was -- the cycle is skipped and
  logged, not crashed.

## 0.1.6
- Fixed nav publication-lag stall: the combined IGS/BKG BRDC nav product's
  real publication lag is ~2 days, not 1 -- `get_nav()` now walks back up
  to 5 days rather than trying only one fallback.
- Fixed the poison-file guard permanently skipping files that only failed
  because nav wasn't published yet (a transient condition) -- these now
  retry on the next poll instead of being written off forever.
- Fixed a temp-file leak: `local_gz`/`decimated` are now always cleaned up,
  even on ingest failure.
- Fixed a duplicate logged row after every restart -- `last_report_time`/
  `last_value` are now persisted across restarts.

## 0.1.5
- `results.csv` is now also mirrored to `/share/app1_nrt_shadow_results.csv`
  so it's actually reachable (e.g. via the Studio Code Server or File
  editor add-ons) without container/host shell access.

## 0.1.4
- Fixed a crash-loop: `pd.Timestamp.utcnow()` is timezone-aware, while the
  ingested RINEX timestamps are naive -- comparing them directly in the
  buffer-retention trim raised a `TypeError` every cycle.

## 0.1.3
- Added a same-day-nav-not-yet-published fallback (later found insufficient
  and extended in 0.1.6).

## 0.1.2
- Fixed nav requests using wall-clock "now" instead of the actual date of
  the RINEX file being processed -- backlog files from a previous day were
  requesting the wrong day's nav.

## 0.1.1
- Startup SMB connection now retries with backoff instead of crashing the
  container outright on a bad/unset password.

## 0.1.0
- Initial release: wide-mask + robust-anchor GNSS-IR retrieval, read-only
  shadow trial alongside the live Simon+K2D system.
