import os, glob
import pandas as pd
import numpy as np

# 1. Folders (relative to this script)
windowing_dir = "PatchTST_supervised/data/windowing"
derived_dir = "PatchTST_supervised/data/derived"
os.makedirs(windowing_dir, exist_ok=True)

# 2. Loop over derived CSVs
for fp in glob.glob(os.path.join(derived_dir, "*_derived.csv")):
    df = pd.read_csv(fp, parse_dates=["Date"])

    # Step 2.1: Compute Rolling Volatility (Standard Deviation)
    rolling_window = 5  # 5-day rolling volatility
    df["Volatility"] = df["log_return"].rolling(window=rolling_window).std()
    
    # Step 2.2: Normalize Volatility to [0, 1]
    vol_min = df["Volatility"].min()
    vol_max = df["Volatility"].max()
    df["Volatility_Norm"] = (df["Volatility"] - vol_min) / (vol_max - vol_min)
    
    # Step 2.3: Adaptive Window Size (inverse of volatility)
    # Set min/max limits to keep it stable
    min_window = 10
    max_window = 50
    df["AdaptiveWindow"] = (
        max_window - df["Volatility_Norm"].fillna(0) * (max_window - min_window)
    ).astype(int)
   
    # save derived CSV
    base = os.path.basename(fp).replace("_derived.csv", "")
    out = os.path.join(windowing_dir, f"{base}_windowing.csv")
    df.to_csv(out, index=False)
    print("Wrote →", out)
