import os
import glob
import pandas as pd
import numpy as np

# 1. Define folders (relative to script’s location)
raw_dir    = "PatchTST_supervised/Raw data"
preprocessed_dir = "PatchTST_supervised/dataset"
os.makedirs(preprocessed_dir, exist_ok=True)

# 2. Loop over every CSV in raw
for raw_path in glob.glob(os.path.join(raw_dir, "*.csv")):
    # load
    df = pd.read_csv(raw_path, parse_dates=["Date"])

    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # sort & flag gaps
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["is_gap"] = df.groupby("Ticker")["Date"].diff().dt.days > 1

    # log return
    df["log_return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

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
   
   # 2.4: Adaptive Patch
   # Round to nearest multiple of 8
    df['AdaptivePatch'] = (df['AdaptiveWindow'] / 8).round().astype(int) * 8
    
    # Dynamic Patch column added
    k = 3
    df["dynamic_patch_len"] = (df["AdaptivePatch"] // k).round().astype(int)
    df["dynamic_patch_len"] = df["dynamic_patch_len"].clip(lower=8)  # safety: min patch size = 8

    df.rename(columns={"Date": "date"}, inplace=True)
   
    # prepare output path
    base = os.path.basename(raw_path).rsplit(".csv", 1)[0]
    out_path = os.path.join(preprocessed_dir, f"{base}_preprocessed.csv")

    # save
    df.to_csv(out_path, index=False)
    print(f"Wrote → {out_path}")
