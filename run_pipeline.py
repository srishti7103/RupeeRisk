import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pmdarima import auto_arima
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import json
import warnings
import os

warnings.filterwarnings("ignore")

# Load data
print("Loading data...")
df = pd.read_csv("data/processed/master_df.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# 1. Weekly resampling
tension_cols = [
    "Geo_Tension", 
    "Geo_Tension_DirectFX_IndiaPak", 
    "Geo_Tension_DirectFX_IndiaChina", 
    "Geo_Tension_OilSupply", 
    "Geo_Tension_RiskOff_RusUkr"
]
weekly = df[["USDINR", "CRUDE", "DXY", "Rate_Spread"] + tension_cols].resample("W").mean()
weekly.dropna(inplace=True)
print(f"Weekly data shape: {weekly.shape[0]} weeks")

# 2. Compute first differences
weekly["USDINR_diff"] = weekly["USDINR"].diff()
weekly["CRUDE_diff"] = weekly["CRUDE"].diff()
weekly["DXY_diff"] = weekly["DXY"].diff()
weekly["Rate_Spread_diff"] = weekly["Rate_Spread"].diff()

# Exogenous base variables to lag
exog_base = ["CRUDE_diff", "DXY_diff", "Rate_Spread_diff"] + tension_cols
lagged_exog = weekly[exog_base].shift(1)
lagged_exog.columns = [c + "_lag1" for c in exog_base]

# 3. Momentum & Calendar Dummies (Phase 2)
# 4-week and 12-week momentum (lagged by 1 week to avoid look-ahead bias)
weekly["inr_mom_4w"] = weekly["USDINR_diff"].rolling(4).mean().shift(1)
weekly["inr_mom_12w"] = weekly["USDINR_diff"].rolling(12).mean().shift(1)

# Calendar dummies (March fiscal year-end, and end of quarter)
weekly["is_fiscal_yr_end"] = (weekly.index.month == 3).astype(float)
weekly["is_qtr_end"] = weekly.index.month.isin([3, 6, 9, 12]).astype(float)

# Combine into model dataframe
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

print(f"Model dataset shape: {model_df.shape}")

# 4. Train/Test Split (200 Weeks Test Window)
TEST_WEEKS = 200
train = model_df.iloc[:-TEST_WEEKS]
test  = model_df.iloc[-TEST_WEEKS:]
print(f"Train size: {len(train)}, Test size: {len(test)}")
print(f"Test window: {test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')}")

# 5. Determine ARIMA order on initial train set (non-seasonal)
print("Finding optimal ARIMA order on training set...")
auto_model = auto_arima(
    train["USDINR_diff"],
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    information_criterion="aic",
    max_p=5, max_q=5
)
order = auto_model.order
print(f"Best ARIMA order: {order}")

# 6. Rolling 1-Step-Ahead Forecast Loop
preds = {
    "naive": [], "es": [], "arima": [], "arimax": [], "lasso": [], "rf": [], "gb": [],
    "arima_lasso": [], "arima_gb": []
}

for i in range(TEST_WEEKS):
    if i % 25 == 0:
        print(f"Rolling forecast step {i}/{TEST_WEEKS}...")

    curr_train = model_df.iloc[:-(TEST_WEEKS - i)]
    curr_test = model_df.iloc[-(TEST_WEEKS - i):].iloc[0]

    y_train_diff = curr_train["USDINR_diff"]
    y_train_level = curr_train["USDINR"]
    X_train_exog = curr_train[features_list]
    prev_level = y_train_level.iloc[-1]

    test_exog_val = pd.DataFrame([curr_test[features_list]], columns=features_list)

    # 1. Naïve Baseline (No change predicted)
    preds["naive"].append(prev_level)

    # 2. Exponential Smoothing (fit on levels)
    try:
        ses_model = SimpleExpSmoothing(y_train_level).fit(optimized=True, use_brute=True)
        pred_es_level = ses_model.forecast(steps=1).iloc[0]
        preds["es"].append(pred_es_level)
    except Exception as e:
        # Fallback to naive
        preds["es"].append(prev_level)

    # 3. ARIMA (non-seasonal, order from auto_model)
    arima_train_residuals = y_train_diff.copy()
    pred_arima_diff = 0.0
    try:
        model_arima = SARIMAX(
            y_train_diff,
            order=order,
            seasonal_order=(0,0,0,0),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        pred_arima_diff = model_arima.forecast(steps=1).iloc[0]
        preds["arima"].append(prev_level + pred_arima_diff)
        
        # In-sample residuals for hybrid models
        arima_fitted = model_arima.fittedvalues
        arima_train_residuals = y_train_diff - arima_fitted
    except Exception as e:
        preds["arima"].append(prev_level)

    # 4. ARIMAX (non-seasonal, order from auto_model)
    try:
        model_arimax = SARIMAX(
            y_train_diff,
            exog=X_train_exog,
            order=order,
            seasonal_order=(0,0,0,0),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        pred_arimax_diff = model_arimax.forecast(steps=1, exog=test_exog_val).iloc[0]
        preds["arimax"].append(prev_level + pred_arimax_diff)
    except Exception as e:
        preds["arimax"].append(prev_level)

    # 5. Lasso Regression (Standalone)
    model_lasso = Lasso(alpha=0.001).fit(X_train_exog, y_train_diff)
    pred_lasso_diff = model_lasso.predict(test_exog_val)[0]
    preds["lasso"].append(prev_level + pred_lasso_diff)

    # 6. Random Forest (Standalone)
    model_rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train_exog, y_train_diff)
    pred_rf_diff = model_rf.predict(test_exog_val)[0]
    preds["rf"].append(prev_level + pred_rf_diff)

    # 7. Gradient Boosting (Standalone)
    model_gb = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X_train_exog, y_train_diff)
    pred_gb_diff = model_gb.predict(test_exog_val)[0]
    preds["gb"].append(prev_level + pred_gb_diff)

    # 8. ARIMA + Lasso Hybrid
    try:
        # We train Lasso on X_train_exog to predict the ARIMA in-sample residuals
        model_lasso_resid = Lasso(alpha=0.001).fit(X_train_exog, arima_train_residuals)
        pred_lasso_resid = model_lasso_resid.predict(test_exog_val)[0]
        pred_arima_lasso_diff = pred_arima_diff + pred_lasso_resid
        preds["arima_lasso"].append(prev_level + pred_arima_lasso_diff)
    except Exception as e:
        preds["arima_lasso"].append(prev_level + pred_arima_diff)

    # 9. ARIMA + Gradient Boosting Hybrid (ARIMA + XGBoost equivalent)
    try:
        # We train GB on X_train_exog to predict the ARIMA in-sample residuals
        model_gb_resid = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X_train_exog, arima_train_residuals)
        pred_gb_resid = model_gb_resid.predict(test_exog_val)[0]
        pred_arima_gb_diff = pred_arima_diff + pred_gb_resid
        preds["arima_gb"].append(prev_level + pred_arima_gb_diff)
    except Exception as e:
        preds["arima_gb"].append(prev_level + pred_arima_diff)

# Save Lasso coefficients from final fit for interpretability
final_lasso = Lasso(alpha=0.001).fit(model_df[features_list], model_df["USDINR_diff"])
coef_df = pd.DataFrame({
    "Feature": features_list,
    "Coefficient": final_lasso.coef_
})
coef_df.to_csv("data/processed/lasso_coefficients.csv", index=False)
print("\nFinal Lasso coefficients saved.")

# 7. Compute Performance Metrics & Backtests
y_test_actual = test["USDINR"].values
prev_test_levels = model_df["USDINR"].iloc[-(TEST_WEEKS+1):-1].values
actual_returns = (y_test_actual - prev_test_levels) / prev_test_levels

results = {}
trading_cum_rets = {}

def directional_accuracy(y_true, y_pred, y_prev):
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    valid = true_dir != 0
    return np.mean(true_dir[valid] == pred_dir[valid]) * 100

# Calculate Naive RMSE first for Theil's U
naive_preds = np.array(preds["naive"])
naive_rmse = np.sqrt(mean_squared_error(y_test_actual, naive_preds))

for model_name, pred_levels in preds.items():
    pred_levels = np.array(pred_levels)

    # Core metrics
    mape = mean_absolute_percentage_error(y_test_actual, pred_levels) * 100
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_levels))
    theils_u = rmse / naive_rmse

    # Directional Accuracy
    mda = directional_accuracy(y_test_actual, pred_levels, prev_test_levels)

    # Backtest Simulation
    predicted_changes = pred_levels - prev_test_levels
    signals = np.sign(predicted_changes)
    
    # If signal is 0 (naive), return is 0
    strat_returns = signals * actual_returns
    
    # Cumulative returns
    cum_returns = np.cumprod(1 + strat_returns) - 1
    final_cum_ret = cum_returns[-1] * 100

    # Sharpe Ratio (Ann.) - Information Ratio (Rf = 0)
    # NOTE: use an epsilon guard, not `!= 0` -- a constant array's std() is never
    # exactly 0 in floating point (it lands around 1e-19), so `!= 0` lets a
    # near-zero denominator blow the ratio up to +/-1e16. EPS catches this.
    EPS = 1e-8
    std0 = np.std(strat_returns)
    if std0 > EPS:
        sharpe_rf0 = np.sqrt(52) * (np.mean(strat_returns) / std0)
    else:
        sharpe_rf0 = 0.0

    # Sharpe Ratio adjusted for India 91-day T-bill (~6.5% annual, which is ~0.125% per week)
    rf_weekly = 0.065 / 52
    strat_returns_excess = strat_returns - rf_weekly
    std65 = np.std(strat_returns_excess)
    if std65 > EPS:
        sharpe_rf6_5 = np.sqrt(52) * (np.mean(strat_returns_excess) / std65)
    else:
        sharpe_rf6_5 = 0.0

    results[model_name] = {
        "MAPE (%)": mape,
        "RMSE": rmse,
        "Theil's U": theils_u,
        "MDA (%)": mda,
        "Sharpe Ratio (Rf=0)": sharpe_rf0,
        "Sharpe Ratio (Rf=6.5%)": sharpe_rf6_5,
        "Cumulative Return (%)": final_cum_ret
    }
    trading_cum_rets[model_name] = list(cum_returns)

# Save metrics to CSV
results_df = pd.DataFrame(results).T
results_df["Sharpe Ratio"] = results_df["Sharpe Ratio (Rf=0)"]
results_df.to_csv("data/processed/model_metrics.csv")

# Save predictions JSON
predictions_json = {
    "dates": [d.strftime("%Y-%m-%d") for d in test.index],
    "actual": list(y_test_actual),
    "predictions": {m: list(p) for m, p in preds.items()},
    "cum_returns": trading_cum_rets
}
with open("data/processed/predictions.json", "w") as f:
    json.dump(predictions_json, f, indent=4)

print("\n=== LEAGUE TABLE (200 WEEKS WITH HYBRIDS) ===")
print(results_df.to_string())
print("\nPredictions and metrics saved successfully!")
