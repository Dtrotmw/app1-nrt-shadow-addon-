"""
Local, no-Docker, no-SMB test of the add-on's real main.py functions
(not a reimplementation) against real files from D:\\GNSS\\DATA, standing
in for the SMB feed. Exercises: decimate_text, extract_file, get_nav,
run_cycle -- simulating several consecutive "new file arrives" cycles.
"""
import sys, os, glob, importlib.util
sys.path.insert(0, r"C:\Users\dandl\Documents\APP1-NRT-ShadowAddon\app1_nrt_shadow\app")
import warnings
warnings.filterwarnings("ignore")

spec = importlib.util.spec_from_file_location("nrtmain", r"C:\Users\dandl\Documents\APP1-NRT-ShadowAddon\app1_nrt_shadow\app\main.py")
nrtmain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nrtmain)

import pandas as pd

# stand in for /data (real dirs, kept local to the test folder, not touching
# the actual add-on's /data path assumptions)
DATA_DIR = r"C:\Users\dandl\Documents\APP1-NRT-ShadowAddon\test_data"
nrtmain.NAV_DIR = os.path.join(DATA_DIR, "nav")
os.makedirs(nrtmain.NAV_DIR, exist_ok=True)

PRED_PATH = r"C:\Users\dandl\Documents\TidalStudy\refl_code\scratch\predNEW_headerless.csv"
# fall back to the Oct'25 vintage if predNEW isn't available for these dates
if not os.path.exists(PRED_PATH):
    PRED_PATH = None

# real, already-cached day: 2026-07-17 (doy 198), use its raw D:\GNSS\DATA files directly
day_dir = r"D:\GNSS\DATA\26198"
files = sorted(glob.glob(os.path.join(day_dir, "APP1198*.26O.gz")))
print(f"{len(files)} raw 15-min files available for the test day")

# use gnss5mins.csv (the live pred file, already confirmed to work with build_predinterp)
pred_path = r"H:\www\gnss5mins.csv"
if not os.path.exists(pred_path):
    pred_path = PRED_PATH
print("using pred file:", pred_path)

buffer = pd.DataFrame()
history = []
last_value = None
last_report_time = None

# process the first ~12 files (3 hours of data -- enough to exercise ingestion
# and at least attempt one retrieval cycle, without a full day's runtime)
test_files = files[:12]
target_date = "2026-07-17"

import time
t_start = time.time()
for fpath in test_files:
    fn = os.path.basename(fpath)
    t0 = time.time()
    decimated = nrtmain.decimate_text(fpath)
    nav = nrtmain.get_nav(target_date)
    new_rows = nrtmain.extract_file(decimated, nav)
    os.remove(decimated)
    if not new_rows.empty:
        buffer = pd.concat([buffer, new_rows], ignore_index=True)
    print(f"  ingested {fn}: {len(new_rows)} rows ({time.time()-t0:.1f}s), buffer now {len(buffer)} rows")

print(f"\nTotal ingestion time for {len(test_files)} files: {time.time()-t_start:.1f}s "
      f"({(time.time()-t_start)/len(test_files):.2f}s/file average)")

# now run one retrieval cycle at the latest available report time
latest_time = buffer["time"].max()
report_time = latest_time.floor("15min")
print(f"\nRunning retrieval cycle at {report_time} (buffer spans "
      f"{buffer['time'].min()} to {buffer['time'].max()})")

t0 = time.time()
value, info = nrtmain.run_cycle(report_time, buffer, history, pred_path)
print(f"Cycle took {time.time()-t0:.1f}s")
print(f"Result: value={value}, info={info if value is None else {k:v for k,v in info.items() if k not in ('kval','ab')}}")
