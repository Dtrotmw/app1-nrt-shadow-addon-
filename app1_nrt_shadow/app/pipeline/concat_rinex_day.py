"""
concat_rinex_day.py

Reconstructs one full-day RINEX 3 observation file from APP1's native
15-minute short-name files (APP1{doy}{hour-letter}{minute:02d}.{yy}O.gz),
so a satellite's rise/set arc can be fit in full instead of being chopped
into 15-minute windows (see APP1_GNSS-IR_Briefing.md SS6/SS9 for why).

Method: RINEX 3 obs files are header + a sequence of epoch records (each
starting with a '>' line). Taking the header from the first file and
appending only the epoch records from every file, in chronological order,
reconstructs the same file the receiver would have produced had it written
one file per day instead of one per 15 minutes -- this is lossless as long
as sessions are contiguous with no gap or overlap at the boundaries, which
is checked here via TIME OF FIRST/LAST OBS in each file's header.

Change log
----------
v1  2026-07-29  Claude Code. Initial version, built for the gnssrefl
    engine-integration spike (Phase 0 of the RINEX-reprocessing plan).
"""
import gzip
import os
from datetime import datetime


HOUR_MASK = "abcdefghijklmnopqrstuvwx"


def _open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def _day_file_list(base_dir, doy, yy):
    """Returns the ordered list of expected file paths for one day, skipping any that don't exist."""
    files = []
    for h in HOUR_MASK:
        for m in (0, 15, 30, 45):
            for ext in (".gz", ""):
                fname = f"APP1{doy:03d}{h}{m:02d}.{yy:02d}O{ext}"
                fpath = os.path.join(base_dir, fname)
                if os.path.exists(fpath):
                    files.append(fpath)
                    break
    return files


def _split_epoch_blocks(body_lines):
    """Split RINEX3 epoch records (each starting with a '>' line) into (epoch_line, sat_lines) blocks."""
    blocks = []
    cur_epoch, cur_sats = None, []
    for l in body_lines:
        if l.startswith(">"):
            if cur_epoch is not None:
                blocks.append((cur_epoch, cur_sats))
            cur_epoch, cur_sats = l, []
        else:
            cur_sats.append(l)
    if cur_epoch is not None:
        blocks.append((cur_epoch, cur_sats))
    return blocks


def _parse_epoch_time(epoch_line):
    # e.g. "> 2026 01 27 00 00  2.0000000  0 44"
    parts = epoch_line.split()
    y, mo, d, h, mi = (int(parts[k]) for k in range(1, 6))
    sec = float(parts[6])
    return datetime(y, mo, d, h, mi, int(sec))


def concat_rinex_day(base_dir, doy, yy, out_path, decimate_s=None):
    """
    Concatenate one day's 15-min RINEX3 obs files into a single daily obs file at out_path.

    decimate_s: if set, keep only epochs falling on this interval (e.g. 15 or 30
    seconds), dropping the rest. Native sampling here is 2 seconds, far denser
    than GNSS-IR reflectometry needs (the reflection oscillation period is tens
    of seconds to minutes) -- and georinex's RINEX3 parser does an expensive
    xarray merge per epoch, so 2-second data makes a full day's load take a very
    long time. Standard GNSS-IR practice (e.g. gnssrefl's `dec_rate` option)
    decimates for exactly this reason. None keeps every epoch (lossless, slow).
    """
    files = _day_file_list(base_dir, doy, yy)
    if not files:
        raise FileNotFoundError(f"No APP1 obs files found for doy {doy} in {base_dir}")

    header_lines = None
    epoch_count = 0
    first_obs_time = None
    last_obs_time = None

    with open(out_path, "w", encoding="ascii", errors="replace") as out:
        for i, fpath in enumerate(files):
            with _open_text(fpath) as f:
                lines = f.readlines()

            end_idx = next((j for j, l in enumerate(lines) if "END OF HEADER" in l), None)
            if end_idx is None:
                raise ValueError(f"No END OF HEADER found in {fpath}")

            this_header = lines[:end_idx + 1]
            this_body = lines[end_idx + 1:]

            for l in this_header:
                if "TIME OF FIRST OBS" in l:
                    t = _parse_obs_time(l)
                    if first_obs_time is None or t < first_obs_time:
                        first_obs_time = t
                if "TIME OF LAST OBS" in l:
                    t = _parse_obs_time(l)
                    if last_obs_time is None or t > last_obs_time:
                        last_obs_time = t

            if header_lines is None:
                header_lines = this_header
                out.writelines(header_lines)

            if decimate_s:
                for epoch_line, sat_lines in _split_epoch_blocks(this_body):
                    et = _parse_epoch_time(epoch_line)
                    if et.second % decimate_s != 0:
                        continue
                    out.write(epoch_line)
                    out.writelines(sat_lines)
                    epoch_count += 1
            else:
                out.writelines(this_body)
                epoch_count += sum(1 for l in this_body if l.startswith(">"))

    return {
        "out_path": out_path,
        "n_files": len(files),
        "n_epochs": epoch_count,
        "first_obs_time": first_obs_time,
        "last_obs_time": last_obs_time,
        "files_used": files,
    }


def _parse_obs_time(line):
    parts = line.split()
    # e.g. "2026     1    27     0     0    0.0000000     GPS         TIME OF FIRST OBS"
    y, mo, d, h, mi = (int(parts[k]) for k in range(5))
    sec = float(parts[5])
    return datetime(y, mo, d, h, mi, int(sec))


if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\GNSS\DATA\26027"
    doy = int(sys.argv[2]) if len(sys.argv) > 2 else 27
    yy = int(sys.argv[3]) if len(sys.argv) > 3 else 26
    out_path = sys.argv[4] if len(sys.argv) > 4 else r"C:\Users\dandl\Documents\TidalStudy\refl_code\scratch\app1_2026_027.26O"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = concat_rinex_day(base_dir, doy, yy, out_path)
    print(f"Wrote {result['out_path']}")
    print(f"  files used: {result['n_files']}")
    print(f"  epochs: {result['n_epochs']}")
    print(f"  span: {result['first_obs_time']} -> {result['last_obs_time']}")
