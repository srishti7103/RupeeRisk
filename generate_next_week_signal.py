import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import json

# Load data
df = pd.read_csv("data/processed/master_df.csv", index_col=0, parse_dates=True)

# Weekly resampling
weekly = df[["USDINR", "CRUDE", "DXY", "Rate_Spread", "Geo_Tension"]].resample("W").mean()
weekly.dropna(inplace=True)

# Compute first differences
weekly["USDINR_diff"] = weekly["USDINR"].diff()
weekly["CRUDE_diff"] = weekly["CRUDE"].diff()
weekly["DXY_diff"] = weekly["DXY"].diff()
weekly["Rate_Spread_diff"] = weekly["Rate_Spread"].diff()

# Exogenous base variables to lag
exog_base = ["CRUDE_diff", "DXY_diff", "Rate_Spread_diff", "Geo_Tension"]
lagged_exog = weekly[exog_base].shift(1)
lagged_exog.columns = [c + "_lag1" for c in exog_base]

# Momentum & Calendar Dummies
weekly["inr_mom_4w"] = weekly["USDINR_diff"].rolling(4).mean().shift(1)
weekly["inr_mom_12w"] = weekly["USDINR_diff"].rolling(12).mean().shift(1)
weekly["is_fiscal_yr_end"] = (weekly.index.month == 3).astype(float)
weekly["is_qtr_end"] = weekly.index.month.isin([3, 6, 9, 12]).astype(float)

features_list = [
    "CRUDE_diff_lag1", "DXY_diff_lag1", "Rate_Spread_diff_lag1", "Geo_Tension_lag1",
    "inr_mom_4w", "inr_mom_12w", "is_fiscal_yr_end", "is_qtr_end"
]

model_df = pd.concat([
    weekly[["USDINR", "USDINR_diff"]],
    lagged_exog,
    weekly[["inr_mom_4w", "inr_mom_12w", "is_fiscal_yr_end", "is_qtr_end"]]
], axis=1).dropna()

# Fit Lasso on the entire dataset
X = model_df[features_list]
y = model_df["USDINR_diff"]
lasso = Lasso(alpha=0.001).fit(X, y)

# Construct features for the NEXT week (t_last + 1)
# The features for the next week are the base features at the last week
last_date = weekly.index[-1]
next_week_date = last_date + pd.Timedelta(weeks=1)

# Base features at t_last (to be lagged by 1 week relative to next week)
next_crude_diff_lag1 = weekly["CRUDE_diff"].iloc[-1]
next_dxy_diff_lag1 = weekly["DXY_diff"].iloc[-1]
next_rate_spread_diff_lag1 = weekly["Rate_Spread_diff"].iloc[-1]
next_geo_tension_lag1 = weekly["Geo_Tension"].iloc[-1]

# Momentum at t_last (known at last_date)
# 4w momentum ending at last_date: average of USDINR_diff over the last 4 weeks (including last_date)
next_inr_mom_4w = weekly["USDINR_diff"].iloc[-4:].mean()
next_inr_mom_12w = weekly["USDINR_diff"].iloc[-12:].mean()

# Calendar dummies for next week (t_last + 1)
next_is_fiscal_yr_end = float(next_week_date.month == 3)
next_is_qtr_end = float(next_week_date.month in [3, 6, 9, 12])

# Build next week's feature vector
X_next = pd.DataFrame([{
    "CRUDE_diff_lag1": next_crude_diff_lag1,
    "DXY_diff_lag1": next_dxy_diff_lag1,
    "Rate_Spread_diff_lag1": next_rate_spread_diff_lag1,
    "Geo_Tension_lag1": next_geo_tension_lag1,
    "inr_mom_4w": next_inr_mom_4w,
    "inr_mom_12w": next_inr_mom_12w,
    "is_fiscal_yr_end": next_is_fiscal_yr_end,
    "is_qtr_end": next_is_qtr_end
}])

# Predict next week's change
pred_diff = lasso.predict(X_next)[0]
current_rate = weekly["USDINR"].iloc[-1]
pred_rate = current_rate + pred_diff

signal = "STRENGTHEN" if pred_diff < 0 else "WEAKEN"
# In rupee terms, negative change means rate falls, so INR strengthens. Positive change means rate rises, so INR weakens.
# Let's verify: predicted change > 0 means rate rises (INR weakens), change < 0 means rate falls (INR strengthens).
# Change in paise is diff * 100
change_paise = abs(pred_diff) * 100

output = {
    "as_of_date": last_date.strftime("%Y-%m-%d"),
    "next_week_date": next_week_date.strftime("%Y-%m-%d"),
    "current_rate": float(current_rate),
    "predicted_change": float(pred_diff),
    "predicted_rate": float(pred_rate),
    "signal": signal,
    "change_paise": float(change_paise)
}

print(output)

# Save to file
with open("data/processed/next_week_signal.json", "w") as f:
    json.dump(output, f, indent=4)
print("Next week signal saved.")
