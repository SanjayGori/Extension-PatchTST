import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# 1. Folder setup
# --------------------------
windowing_dir = "PatchTST_supervised/data/windowing"
derived_dir = "PatchTST_supervised/data/derived"
os.makedirs(windowing_dir, exist_ok=True)

# --------------------------
# 2. Loop over derived CSVs
# --------------------------
for i, fp in enumerate(glob.glob(os.path.join(derived_dir, "*_derived.csv"))):
    df = pd.read_csv(fp, parse_dates=["Date"])

    # --------------------------
    # Step 2.1 – Rolling Volatility
    # --------------------------
    rolling_window = 5
    df["Volatility"] = df["log_return"].rolling(window=rolling_window).std()

    # --------------------------
    # Step 2.2 – Normalize & Map to Window
    # --------------------------
    vol_min = df["Volatility"].min()
    vol_max = df["Volatility"].max()
    df["Volatility_Norm"] = (df["Volatility"] - vol_min) / (vol_max - vol_min)

    min_window = 10
    max_window = 50
    df["AdaptiveWindow"] = (
        max_window - df["Volatility_Norm"].fillna(0) * (max_window - min_window)
    ).astype(int)

    # --------------------------
    # Step 2.4 – Round to Patch
    # --------------------------
    df["AdaptivePatch"] = (df["AdaptiveWindow"] / 8).round().astype(int) * 8

    # --------------------------
    # Step 2.3 – Unit Test (only once)
    # --------------------------
    if i == 0:
        low_vol = df[df["Volatility_Norm"] < 0.3]["AdaptiveWindow"]
        high_vol = df[df["Volatility_Norm"] > 0.7]["AdaptiveWindow"]

        print("Low-vol window mean:", round(low_vol.mean(), 2))
        print("High-vol window mean:", round(high_vol.mean(), 2))

        assert high_vol.mean() < low_vol.mean(), "Unit Test Failed: High-vol window not smaller"
        print("Unit Test Passed: High volatility results in smaller window.")

    # --------------------------
    # Step 2.4 – Plot (only once)
    # --------------------------
    if i == 0:
        sns.set(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(df["Date"], df["Volatility_Norm"], label="Normalised 5-Day Volatility", color="blue", alpha=0.6)
        ax1.set_ylabel("Normalised 5-Day Volatility", color="blue")
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(df["Date"], df["AdaptivePatch"], label="Adaptive Patch round off to 8 Multiple", color="red", alpha=0.8)
        ax2.set_ylabel("Adaptive Patching (8 Multiple's)", color="red")
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title(f"Normalised Volatility vs Adaptive Patching for 'American Airlines Group Inc.'")
        fig.tight_layout()
        plt.show()
    
    break