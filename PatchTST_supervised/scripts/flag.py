import os
import glob
import pandas as pd

# 1. Define folders (relative to script’s location)
raw_dir    = "PatchTST_supervised/data/raw"
flagged_dir = "PatchTST_supervised/data/flagged"
os.makedirs(flagged_dir, exist_ok=True)

# 2. Loop over every CSV in raw
for raw_path in glob.glob(os.path.join(raw_dir, "*.csv")):
    # load
    df = pd.read_csv(raw_path, parse_dates=["Date"])
    # sort & flag gaps
    df = df.sort_values(["Ticker", "Date"])
    df["is_gap"] = df.groupby("Ticker")["Date"].diff().dt.days > 1

    # prepare output path
    base = os.path.basename(raw_path).rsplit(".csv", 1)[0]
    out_path = os.path.join(flagged_dir, f"{base}_flagged.csv")

    # save
    df.to_csv(out_path, index=False)
    print(f"Wrote → {out_path}")
