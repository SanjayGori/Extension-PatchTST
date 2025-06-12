import pandas as pd

# 1. Load your raw CSV
df = pd.read_csv(
    "C:/Users/gorit/Downloads/archive/Set 1/ACN.csv",
    parse_dates=["Date"]
)

# 4. Flag gaps >1 day per ticker
df = df.sort_values(["Ticker", "Date"])
df["is_gap"] = df.groupby("Ticker")["Date"].diff().dt.days > 1

# 5. Save the flagged CSV
out_path = "C:/Users/gorit/Capstone project"
df.to_csv("ACN_flag",index=False)
#print("Wrote", out_path)
