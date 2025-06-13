import os, glob
import pandas as pd
import numpy as np

# 1. Folders (relative to this script)
windowing_dir = "PatchTST_supervised/data/windowing"
dynamic_patch_dir = "PatchTST_supervised/data/dynamic_pathing"
os.makedirs(dynamic_patch_dir, exist_ok=True)

# 2. Loop over derived CSVs
for fp in glob.glob(os.path.join(windowing_dir, "*_windowing.csv")):
    df = pd.read_csv(fp, parse_dates=["Date"])
    
    # Dynamic Patch column added
    k = 3
    df["dynamic_patch_len"] = (df["AdaptivePatch"] // k).round().astype(int)
    df["dynamic_patch_len"] = df["dynamic_patch_len"].clip(lower=8)  # safety: min patch size = 8
   
    # save derived CSV
    base = os.path.basename(fp).replace("_windowing.csv", "")
    out = os.path.join(dynamic_patch_dir, f"{base}_dynamic_patch.csv")
    df.to_csv(out, index=False)
    print("Wrote →", out)
