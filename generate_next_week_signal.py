import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import json
import os

# Load data
df = pd.read_csv("data/processed/master_df.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# Weekly resampling
tension_cols = [
    "Geo_Tension", 
    "Geo_Tension_DirectFX_IndiaPak", 
    "Geo_Tension_DirectFX_IndiaChina", 
    "Geo_Tension_OilSupply", 
    "Geo_Tension_RiskOff_RusUkr"
]
weekly = df[["USDINR", "CRUDE", "DXY", "Rate_Spread"] + tension_cols].resample("W").mean()
weekly.dropna(inplace=True)

# Compute first differences
weekly["USDINR_diff"] = weekly["USDINR"].diff()
weekly["CRUDE_diff"] = weekly["CRUDE"].diff()
weekly["DXY_diff"] = weekly["DXY"].diff()
weekly["Rate_Spread_diff"] = weekly["Rate_Spread"].diff()

# Exogenous base variables to lag
exog_base = ["CRUDE_diff", "DXY_diff", "Rate_Spread_diff"] + tension_cols
lagged_exog = weekly[exog_base].shift(1)
lagged_exog.columns = [c + "_lag1" for c in exog_base]

# Momentum & Calendar Dummies
weekly["inr_mom_4w"] = weekly["USDINR_diff"].rolling(4).mean().shift(1)
weekly["inr_mom_12w"] = weekly["USDINR_diff"].rolling(12).mean().shift(1)
weekly["is_fiscal_yr_end"] = (weekly.index.month == 3).astype(float)
weekly["is_qtr_end"] = weekly.index.month.isin([3, 6, 9, 12]).astype(float)

tension_lagged_cols = [c + "_lag1" for c in tension_cols]
features_list = [
    "CRUDE_diff_lag1", "DXY_diff_lag1", "Rate_Spread_diff_lag1"
] + tension_lagged_cols + [
    "inr_mom_4w", "inr_mom_12w", "is_fiscal_yr_end", "is_qtr_end"
]

model_df = pd.concat([
    weekly[["USDINR", "USDINR_diff"]],
    lagged_exog,
    weekly[["inr_mom_4w", "inr_mom_12w", "is_fiscal_yr_end", "is_qtr_end"]]
], axis=1).dropna()

X = model_df[features_list]
y = model_df["USDINR_diff"]

# ── DYNAMIC BEST-MODEL SELECTION ────────────────────────
candidate_models = ["arima", "arimax", "lasso", "rf", "gb", "arima_lasso", "arima_gb"]
metrics_path = "data/processed/model_metrics.csv"

# Default fallback model
best_model = "arima_lasso"

if os.path.exists(metrics_path):
    try:
        metrics_df = pd.read_csv(metrics_path, index_col=0)
        # Select best candidate model based on out-of-sample Sharpe Ratio (Rf=0)
        valid_candidates = [m for m in candidate_models if m in metrics_df.index]
        if valid_candidates:
            best_model = metrics_df.loc[valid_candidates, "Sharpe Ratio (Rf=0)"].idxmax()
            print(f"Dynamically selected top-performing model: {best_model.upper()}")
    except Exception as e:
        print(f"[WARNING] Error reading metrics, falling back to ARIMA+Lasso Hybrid. Error: {e}")
else:
    print("[WARNING] Metrics file not found, falling back to ARIMA+Lasso Hybrid.")

# ── MODEL TRAINING & PREDICTION ─────────────────────────
last_date = weekly.index[-1]
next_week_date = last_date + pd.Timedelta(weeks=1)

# Construct features for the NEXT week (t_last + 1)
next_crude_diff_lag1 = weekly["CRUDE_diff"].iloc[-1]
next_dxy_diff_lag1 = weekly["DXY_diff"].iloc[-1]
next_rate_spread_diff_lag1 = weekly["Rate_Spread_diff"].iloc[-1]
next_inr_mom_4w = weekly["USDINR_diff"].iloc[-4:].mean()
next_inr_mom_12w = weekly["USDINR_diff"].iloc[-12:].mean()
next_is_fiscal_yr_end = float(next_week_date.month == 3)
next_is_qtr_end = float(next_week_date.month in [3, 6, 9, 12])

X_next_dict = {
    "CRUDE_diff_lag1": next_crude_diff_lag1,
    "DXY_diff_lag1": next_dxy_diff_lag1,
    "Rate_Spread_diff_lag1": next_rate_spread_diff_lag1,
}
for col in tension_cols:
    X_next_dict[col + "_lag1"] = weekly[col].iloc[-1]

X_next_dict.update({
    "inr_mom_4w": next_inr_mom_4w,
    "inr_mom_12w": next_inr_mom_12w,
    "is_fiscal_yr_end": next_is_fiscal_yr_end,
    "is_qtr_end": next_is_qtr_end
})
X_next = pd.DataFrame([X_next_dict], columns=features_list)

# Find optimal ARIMA order for time-series components
print("Finding optimal ARIMA order on whole dataset...")
auto_model = auto_arima(
    y,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    max_p=5, max_q=5
)
order = auto_model.order
print(f"Optimal ARIMA order: {order}")

pred_diff = 0.0

if best_model == "arima":
    model = SARIMAX(y, order=order, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred_diff = model.forecast(steps=1).iloc[0]

elif best_model == "arimax":
    model = SARIMAX(y, exog=X, order=order, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred_diff = model.forecast(steps=1, exog=X_next).iloc[0]

elif best_model == "lasso":
    model = Lasso(alpha=0.001).fit(X, y)
    pred_diff = model.predict(X_next)[0]

elif best_model == "rf":
    model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
    pred_diff = model.predict(X_next)[0]

elif best_model == "gb":
    model = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X, y)
    pred_diff = model.predict(X_next)[0]

elif best_model == "arima_lasso":
    model_arima = SARIMAX(y, order=order, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred_arima_diff = model_arima.forecast(steps=1).iloc[0]
    arima_residuals = y - model_arima.fittedvalues
    model_lasso_resid = Lasso(alpha=0.001).fit(X, arima_residuals)
    pred_lasso_resid = model_lasso_resid.predict(X_next)[0]
    pred_diff = pred_arima_diff + pred_lasso_resid

elif best_model == "arima_gb":
    model_arima = SARIMAX(y, order=order, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    pred_arima_diff = model_arima.forecast(steps=1).iloc[0]
    arima_residuals = y - model_arima.fittedvalues
    model_gb_resid = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X, arima_residuals)
    pred_gb_resid = model_gb_resid.predict(X_next)[0]
    pred_diff = pred_arima_diff + pred_gb_resid

current_rate = weekly["USDINR"].iloc[-1]
pred_rate = current_rate + pred_diff

signal = "STRENGTHEN" if pred_diff < 0 else "WEAKEN"
change_paise = abs(pred_diff) * 100

# Map code name to UI-friendly model type string
model_type_mapping = {
    "arima": "ARIMA_Baseline",
    "arimax": "ARIMAX",
    "lasso": "Lasso",
    "rf": "Random_Forest",
    "gb": "Gradient_Boosting",
    "arima_lasso": "ARIMA+Lasso_Hybrid",
    "arima_gb": "ARIMA+Gradient_Boosting_Hybrid"
}
model_type_str = model_type_mapping.get(best_model, "ARIMA+Lasso_Hybrid")

output = {
    "as_of_date": last_date.strftime("%Y-%m-%d"),
    "next_week_date": next_week_date.strftime("%Y-%m-%d"),
    "current_rate": float(current_rate),
    "predicted_change": float(pred_diff),
    "predicted_rate": float(pred_rate),
    "signal": signal,
    "change_paise": float(change_paise),
    "model_type": model_type_str
}

print(output)

# Save to file
with open("data/processed/next_week_signal.json", "w") as f:
    json.dump(output, f, indent=4)
print("Next week signal saved.")
