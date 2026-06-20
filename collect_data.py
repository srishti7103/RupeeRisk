import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # reads FRED_API_KEY from a local .env file if present (never committed to git)

# ── CONFIG ─────────────────────────────────────────────
START = "2015-01-01"
END   = datetime.today().strftime("%Y-%m-%d")
FRED_KEY = os.environ.get("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError(
        "FRED_API_KEY not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
        "then set it via: a) a .env file (FRED_API_KEY=your_key_here) loaded with python-dotenv, or "
        "b) Streamlit secrets (st.secrets['FRED_API_KEY']) when deployed, or "
        "c) export FRED_API_KEY=your_key_here in your shell before running this script."
    )
# ───────────────────────────────────────────────────────

print(f"Starting data collection and feature engineering pipeline up to {END}...")

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── 1. MARKET DATA VIA YFINANCE ────────────────────────
tickers = {
    "USDINR":    "USDINR=X",   # USD/INR exchange rate
    "NIFTY":     "^NSEI",      # Nifty 50
    "INDIAVIX":  "^INDIAVIX",  # India Volatility Index
    "GOLD":      "GC=F",       # Gold Futures (USD)
    "CRUDE":     "CL=F",       # Crude Oil WTI (USD)
    "DXY":       "DX-Y.NYB",   # US Dollar Index
}

market_data = {}
for name, ticker in tickers.items():
    print(f"Downloading {name} ({ticker}) from yfinance...")
    df_ticker = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    
    # Handle MultiIndex columns (yfinance >= 0.2.x)
    if isinstance(df_ticker.columns, pd.MultiIndex):
        close = df_ticker["Close"].iloc[:, 0]
    else:
        close = df_ticker["Close"]
        
    # Ensure it's a Series
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
        
    market_data[name] = close.rename(name)

# Merge all market data
market_df = pd.concat(market_data.values(), axis=1)
market_df.index = pd.to_datetime(market_df.index)
market_df.index = market_df.index.tz_localize(None)  # remove timezone
market_df.dropna(how='all', inplace=True)
market_df.to_csv("data/raw/market_data.csv")
print(f"Market data saved: {market_df.shape}")

# ── 2. MACRO DATA VIA FRED ─────────────────────────────
fred = Fred(api_key=FRED_KEY)
fred_series = {
    "US_CPI":       "CPIAUCSL",    # US Consumer Price Index
    "US_FEDFUNDS":  "FEDFUNDS",    # US Federal Funds Rate
    "US_10Y":       "GS10",        # US 10-Year Treasury Yield
}

macro_data = {}
for name, series_id in fred_series.items():
    print(f"Downloading {name} from FRED...")
    s = fred.get_series(series_id, observation_start=START, observation_end=END)
    macro_data[name] = s.rename(name)

macro_df = pd.DataFrame(macro_data)
macro_df.index = pd.to_datetime(macro_df.index)
macro_df.to_csv("data/raw/macro_fred.csv")
print(f"FRED macro data saved: {macro_df.shape}")

# ── 3. RBI REPO RATE ───────────────────────────────────
print("Setting up RBI Repo Rate database...")
rbi_repo = pd.DataFrame({
    "date": [
        "2015-01-01","2015-03-04","2015-06-02","2015-09-29",
        "2016-04-05","2016-10-04",
        "2018-06-06","2018-08-01",
        "2019-02-07","2019-04-04","2019-06-06","2019-08-07","2019-10-04",
        "2020-03-27","2020-05-22",
        "2022-05-04","2022-06-08","2022-08-05","2022-09-30","2022-12-07",
        "2023-02-08",
        "2024-10-09",
        "2025-02-07","2025-04-09","2025-06-06"
    ],
    "RBI_REPO": [
        7.75, 7.50, 7.25, 6.75,
        6.50, 6.25,
        6.25, 6.50,
        6.25, 6.00, 5.75, 5.40, 5.15,
        4.40, 4.00,
        4.40, 4.90, 5.40, 5.90, 6.25,
        6.50,
        6.25,
        6.25, 6.00, 5.75
    ]
}).set_index("date")
rbi_repo.index = pd.to_datetime(rbi_repo.index)
rbi_repo = rbi_repo.resample("D").ffill()
rbi_repo.to_csv("data/raw/rbi_repo.csv")

# ── 4. GEOPOLITICAL EVENTS ─────────────────────────────
print("Setting up Geopolitical Events Database...")
events = pd.DataFrame([
    {"date":"2016-09-29","event":"Uri Surgical Strikes","category":"India-Pak","channel":"Direct FX"},
    {"date":"2019-02-14","event":"Pulwama Attack","category":"India-Pak","channel":"Direct FX"},
    {"date":"2019-02-26","event":"Balakot Airstrike","category":"India-Pak","channel":"Direct FX"},
    {"date":"2020-06-15","event":"Galwan Valley Clash","category":"India-China","channel":"Direct FX + Trade"},
    {"date":"2025-05-07","event":"Operation Sindoor","category":"India-Pak","channel":"Direct FX"},
    {"date":"2019-05-12","event":"Gulf of Oman Tanker Attacks","category":"Strait of Hormuz","channel":"Oil Supply"},
    {"date":"2019-09-14","event":"Saudi Aramco Drone Attack","category":"Middle East","channel":"Oil Supply"},
    {"date":"2020-01-03","event":"Soleimani Killing","category":"US-Iran","channel":"Oil Supply + Risk-Off"},
    {"date":"2023-10-07","event":"Israel-Hamas War","category":"Middle East","channel":"Oil Supply + Risk-Off"},
    {"date":"2023-11-19","event":"Houthi Red Sea Attacks","category":"Strait of Hormuz","channel":"Trade Route + Oil"},
    {"date":"2022-02-24","event":"Russia-Ukraine War Begins","category":"Europe","channel":"Oil + Gas + Risk-Off + DXY"},
    {"date":"2022-10-05","event":"OPEC+ Production Cuts","category":"Global Energy","channel":"Oil Supply"},
    {"date":"2015-08-24","event":"China Stock Market Crash","category":"Global","channel":"Risk-Off + DXY"},
    {"date":"2020-03-23","event":"COVID Global Lockdown","category":"Global","channel":"Risk-Off + Oil Collapse"},
])
events["date"] = pd.to_datetime(events["date"])
events.to_csv("data/raw/geopolitical_events.csv", index=False)

# ── 5. MERGE INTO MASTER DATAFRAME & FEATURE ENGINEERING ─
print("Merging datasets and engineering features...")
master_df = market_df.copy()
master_df = master_df.join(macro_df.resample("D").ffill(), how="left")
master_df = master_df.join(rbi_repo, how="left")
master_df.ffill(inplace=True)
master_df.dropna(subset=["USDINR"], inplace=True)

# Engineer features (daily returns, rolling vol, US-India rate spread, geopolitical pulse)
for col in ["USDINR", "NIFTY", "GOLD", "CRUDE", "DXY"]:
    master_df[f"{col}_ret"] = master_df[col].pct_change() * 100

master_df["INR_vol_30d"] = master_df["USDINR_ret"].rolling(30).std()
master_df["Rate_Spread"] = master_df["US_FEDFUNDS"] - master_df["RBI_REPO"]

# Geopolitical tension pulse
master_df["Geo_Tension"] = 0
for _, row in events.iterrows():
    event_date = row["date"]
    mask = (master_df.index >= event_date) & (master_df.index <= event_date + pd.Timedelta(days=7))
    master_df.loc[mask, "Geo_Tension"] = 1

master_df["Month"] = master_df.index.month
master_df["Year"]  = master_df.index.year

# Save final processed dataset
master_df.to_csv("data/processed/master_df.csv")
print(f"Data collection pipeline completed successfully. Master dataframe saved: {master_df.shape}")
