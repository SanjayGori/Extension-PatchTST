import pandas as pd
import pandas_market_calendars as mcal

# 1. Read the raw CSV
df = pd.read_csv(
    "PatchTST_supervised/data/raw/ACN.csv",
    parse_dates=["Date"]
)

# 2. Get NYSE trading days between min and max date
nyse = mcal.get_calendar("NYSE")
sched = nyse.schedule(
    start_date=df["Date"].min(),
    end_date=df["Date"].max()
)
trading_days = sched.index

# 3. Flag holidays (non-trading days)
df["is_holiday"] = ~df["Date"].isin(trading_days)

# 4. Flag gaps (>1 calendar day) per ticker
df = df.sort_values(["Ticker", "Date"])
df["is_gap"] = df.groupby("Ticker")["Date"].diff().dt.days > 1

# 5. Save result
df.to_csv(
    "PatchTST_supervised/data/raw/ACN_flagged.csv",
    index=False
)
print("Saved ACN_flagged.csv")
