"""
Recomputes model_metrics.csv with the Sharpe-ratio division-by-near-zero bug fixed,
and produces a REAL, reproducible volatility regime analysis (replacing the
previously unbacked numbers in Report.md) -- all from data already saved in this
repo (master_df.csv + predictions.json). No external API calls needed.

Run this any time after run_pipeline.py to regenerate corrected outputs.
"""
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ── 1. Load existing artifacts ──────────────────────────
df = pd.read_csv("data/processed/master_df.csv", index_col=0)
df.index = pd.to_datetime(df.index)
weekly_levels = df["USDINR"].resample("W").mean().dropna()

with open("data/processed/predictions.json") as f:
    preds_data = json.load(f)

dates = pd.to_datetime(preds_data["dates"])
y_test_actual = np.array(preds_data["actual"])
preds = preds_data["predictions"]
TEST_WEEKS = len(dates)

# prev_test_levels: the actual level the week BEFORE each test point.
# For i>=1 this is just actual[i-1]. For i=0 we need the level immediately
# preceding the first test date, taken from the full weekly series.
first_test_date = dates[0]
pre_window = weekly_levels.loc[:first_test_date]
level_before_test = pre_window.iloc[-2]  # the week before the first test week
prev_test_levels = np.concatenate([[level_before_test], y_test_actual[:-1]])

actual_returns = (y_test_actual - prev_test_levels) / prev_test_levels

def directional_accuracy(y_true, y_pred, y_prev, mask=None):
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    valid = true_dir != 0
    if mask is not None:
        valid = valid & mask
    if valid.sum() == 0:
        return np.nan
    return np.mean(true_dir[valid] == pred_dir[valid]) * 100

naive_preds = np.array(preds["naive"])
naive_rmse = np.sqrt(mean_squared_error(y_test_actual, naive_preds))

EPS = 1e-8  # epsilon guard fixes the Sharpe(Rf=6.5%) blow-up on constant arrays

results = {}
for model_name, pred_levels in preds.items():
    pred_levels = np.array(pred_levels)

    mape = mean_absolute_percentage_error(y_test_actual, pred_levels) * 100
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_levels))
    theils_u = rmse / naive_rmse
    mda = directional_accuracy(y_test_actual, pred_levels, prev_test_levels)

    predicted_changes = pred_levels - prev_test_levels
    signals = np.sign(predicted_changes)
    strat_returns = signals * actual_returns
    cum_returns = np.cumprod(1 + strat_returns) - 1
    final_cum_ret = cum_returns[-1] * 100

    std0 = np.std(strat_returns)
    sharpe_rf0 = np.sqrt(52) * (np.mean(strat_returns) / std0) if std0 > EPS else 0.0

    rf_weekly = 0.065 / 52
    strat_returns_excess = strat_returns - rf_weekly
    std65 = np.std(strat_returns_excess)
    sharpe_rf65 = np.sqrt(52) * (np.mean(strat_returns_excess) / std65) if std65 > EPS else 0.0

    results[model_name] = {
        "MAPE (%)": mape,
        "RMSE": rmse,
        "Theil's U": theils_u,
        "MDA (%)": mda,
        "Sharpe Ratio (Rf=0)": sharpe_rf0,
        "Sharpe Ratio (Rf=6.5%)": sharpe_rf65,
        "Cumulative Return (%)": final_cum_ret,
    }

results_df = pd.DataFrame(results).T
results_df["Sharpe Ratio"] = results_df["Sharpe Ratio (Rf=0)"]
results_df.to_csv("data/processed/model_metrics.csv")
print("=== CORRECTED LEAGUE TABLE (Sharpe bug fixed) ===")
print(results_df.to_string())

# ── 2. REAL volatility regime analysis (replaces unbacked Report.md numbers) ─
abs_returns = np.abs(actual_returns)
threshold = 0.005  # 0.5% weekly move, same definition Report.md originally claimed
high_vol_mask = abs_returns > threshold
low_vol_mask = ~high_vol_mask

print(f"\nHigh-vol weeks (|return| > {threshold*100:.1f}%): {high_vol_mask.sum()}")
print(f"Low-vol weeks: {low_vol_mask.sum()}")

regime_rows = []
for model_name, pred_levels in preds.items():
    pred_levels = np.array(pred_levels)
    overall_mda = directional_accuracy(y_test_actual, pred_levels, prev_test_levels)
    hv_mda = directional_accuracy(y_test_actual, pred_levels, prev_test_levels, mask=high_vol_mask)
    lv_mda = directional_accuracy(y_test_actual, pred_levels, prev_test_levels, mask=low_vol_mask)
    regime_rows.append({
        "Model": model_name,
        "Overall MDA (%)": round(overall_mda, 2),
        "High-Vol MDA (%)": round(hv_mda, 2),
        "Low-Vol MDA (%)": round(lv_mda, 2),
        "N High-Vol": int(high_vol_mask.sum()),
        "N Low-Vol": int(low_vol_mask.sum()),
    })

regime_df = pd.DataFrame(regime_rows).set_index("Model")
regime_df.to_csv("data/processed/regime_metrics.csv")
print("\n=== REAL VOLATILITY REGIME ANALYSIS (reproducible, replaces old claim) ===")
print(regime_df.to_string())
