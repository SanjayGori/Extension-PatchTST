import os
import pandas as pd
import numpy as np

# Input and output paths
raw_path = "PatchTST_supervised/electricity/electricity.csv"
preprocessed_dir = "PatchTST_supervised/dataset"
os.makedirs(preprocessed_dir, exist_ok=True)

# Load data
df = pd.read_csv(raw_path, parse_dates=["date"])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date").reset_index(drop=True)

# Compute log return
df["log_return"] = np.log(df["OT"] / df["OT"].shift(1))

# Rolling volatility
rolling_window = 5
df["Volatility"] = df["log_return"].rolling(window=rolling_window).std()

# Normalize volatility
vol_min, vol_max = df["Volatility"].min(), df["Volatility"].max()
df["Volatility_Norm"] = (df["Volatility"] - vol_min) / (vol_max - vol_min)

# Adaptive window and patch length
min_window, max_window = 10, 50
df["AdaptiveWindow"] = (
    max_window - df["Volatility_Norm"].fillna(0) * (max_window - min_window)
).astype(int)
df['AdaptivePatch'] = (df['AdaptiveWindow'] / 8).round().astype(int) * 8

# Final patch len
k = 3
df["dynamic_patch_len"] = (df["AdaptivePatch"] // k).round().astype(int)
df["dynamic_patch_len"] = df["dynamic_patch_len"].clip(lower=8)

# Save
base = os.path.basename(raw_path).rsplit(".csv", 1)[0]
out_path = os.path.join(preprocessed_dir, f"{base}_preprocessed.csv")
df.to_csv(out_path, index=False)
print(f"Wrote → {out_path}")