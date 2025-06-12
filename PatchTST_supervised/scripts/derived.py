import os, glob
import numpy as np
import pandas as pd

# 1. Folders (relative to this script)
flagged_dir = "PatchTST_supervised/data/flagged"
derived_dir = "PatchTST_supervised/data/derived"
os.makedirs(derived_dir, exist_ok=True)

# 2. Loop over flagged CSVs
for fp in glob.glob(os.path.join(flagged_dir, "*_flagged.csv")):
    df = pd.read_csv(fp, parse_dates=["Date"])

    # log return
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # RSI (14)
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    # SMA & EMA (20)
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # MACD (12,26)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # ATR (14)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # save derived CSV
    base = os.path.basename(fp).replace("_flagged.csv", "")
    out = os.path.join(derived_dir, f"{base}_derived.csv")
    df.to_csv(out, index=False)
    print("Wrote →", out)
