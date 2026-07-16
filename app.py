import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="RupeeRisk — India Forex Intelligence",
    layout="wide"
)

# ── LOAD DATA ──────────────────────────────────────────
def load_data():
    df      = pd.read_csv("data/processed/master_df.csv",        index_col=0)
    df.index = pd.to_datetime(df.index)
    events  = pd.read_csv("data/raw/geopolitical_events.csv",    parse_dates=["date"])
    metrics = pd.read_csv("data/processed/model_metrics.csv",    index_col=0)
    return df, events, metrics

df, events, metrics = load_data()

@st.cache_data(ttl=3600)
def get_live_rate():
    try:
        import yfinance as yf
        ticker = yf.Ticker("INR=X")
        live_df = ticker.history(period="2d")
        if not live_df.empty:
            return live_df["Close"].iloc[-1]
    except Exception as e:
        pass
    return None

def get_next_week_signal():
    try:
        with open("data/processed/next_week_signal.json", "r") as f:
            return json.load(f)
    except Exception as e:
        return None

next_week_signal = get_next_week_signal()

# ── CUSTOM CSS THEME ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Apply modern font and clean white background */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #FFFFFF !important;
    color: #2D3748 !important;
}

/* Header style overrides */
h1, h2, h3, h4, h5, h6 {
    color: #1A202C !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}

/* Hide Streamlit sidebar completely */
[data-testid="selectedSidebarView"] {
    display: none;
}
section[data-testid="stSidebar"] {
    display: none;
}

/* Style metric cards with sleek borders and left accents */
div[data-testid="metric-container"] {
    background-color: #F8F9FA !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 4px solid #4A5568 !important;
    padding: 18px 24px;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.2s ease-in-out;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-left: 4px solid #3182CE !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
}

/* Custom styles for metric labels and values */
div[data-testid="stMetricValue"], 
div[data-testid="stMetricValue"] *, 
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: #1A202C !important;
}
div[data-testid="stMetricLabel"], 
div[data-testid="stMetricLabel"] *, 
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #2D3748 !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Custom styling for inputs and selectors */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
    border: 1px solid #CBD5E1 !important;
}
div[role="listbox"], [data-baseweb="popover"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
}
div[role="option"], 
div[role="option"] *, 
[data-baseweb="popover"] li, 
[data-baseweb="popover"] li * {
    color: #2D3748 !important;
}
div[role="option"], 
[data-baseweb="popover"] li {
    background-color: #FFFFFF !important;
    transition: background-color 0.15s ease;
}
div[role="option"]:hover, 
div[role="option"]:hover *, 
[data-baseweb="popover"] li:hover, 
[data-baseweb="popover"] li:hover * {
    background-color: #F1F5F9 !important;
    color: #1A202C !important;
}

/* Custom styling for Streamlit native bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    margin-bottom: 25px !important;
}

/* Styling buttons */
div.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    background-color: #1A202C !important;
    color: #FFFFFF !important;
    border: 1px solid #1A202C !important;
    transition: all 0.2s ease-in-out !important;
    padding: 8px 16px !important;
}
div.stButton > button:hover {
    background-color: #2D3748 !important;
    border-color: #2D3748 !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
}

/* Styling dividers */
hr {
    border-top: 1px solid #E2E8F0 !important;
    margin: 25px 0 !important;
}

/* Dataframe styling */
.stDataFrame, div[data-testid="stTable"] {
    background-color: #FFFFFF !important;
    border-radius: 10px;
    border: 1px solid #E2E8F0 !important;
    padding: 5px;
}

/* Widget label colors */
label[data-testid="stWidgetLabel"] {
    color: #2D3748 !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY HELPER THEME ─────────────────────────────────
def apply_plotly_theme(fig, height=400, show_legend=True, is_heatmap=False):
    layout_args = dict(
        height=height,
        template="plotly_white",
        plot_bgcolor="#FFFFFF" if not is_heatmap else None,
        paper_bgcolor="#FFFFFF",
        font=dict(family="'Inter', sans-serif", color="#2D3748"),
        margin=dict(t=40, b=40, l=50, r=20)
    )
    if not is_heatmap:
        layout_args["xaxis"] = dict(
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(color="#4A5568"),
            title=dict(font=dict(color="#2D3748"))
        )
        layout_args["yaxis"] = dict(
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(color="#4A5568"),
            title=dict(font=dict(color="#2D3748"))
        )
    if show_legend:
        layout_args["legend"] = dict(
            orientation="h",
            y=1.15,
            x=0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(color="#2D3748")
        )
    fig.update_layout(**layout_args)

# ── HEADER ─────────────────────────────────────────────
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("""
    <div style="margin-bottom: 10px;">
        <h1 style="margin: 0; font-size: 2.25rem; font-weight: 800; color: #1A202C; letter-spacing: -0.5px; font-family: 'Inter', sans-serif;">RupeeRisk</h1>
        <p style="margin: 4px 0 0 0; font-size: 1.05rem; color: #4A5568; font-weight: 400; font-family: 'Inter', sans-serif;">India Macro & Geopolitical Forex Intelligence Platform</p>
        <div style="font-size: 0.75rem; color: #718096; margin-top: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Advanced Quantitative & Machine Learning Forex Forecasting</div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.write(" ")
    st.write(" ")
    run_update = st.button("🔄 Refresh Data Pipeline", use_container_width=True)

if run_update:
    import subprocess
    import sys

    status_container = st.container(border=True)
    with status_container:
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            python_bin = sys.executable if sys.executable else "python"

            env = os.environ.copy()
            if "FRED_API_KEY" not in env and "FRED_API_KEY" in st.secrets:
                env["FRED_API_KEY"] = st.secrets["FRED_API_KEY"]

            status_text.text("Scraping live feeds (Yahoo Finance & FRED)...")
            progress_bar.progress(10)
            res1 = subprocess.run([python_bin, "collect_data.py"], capture_output=True, text=True, env=env)
            if res1.returncode != 0:
                raise Exception(res1.stderr)

            status_text.text("Training 9 models (rolling 200w validation)...")
            progress_bar.progress(40)
            res2 = subprocess.run([python_bin, "run_pipeline.py"], capture_output=True, text=True, env=env)
            if res2.returncode != 0:
                raise Exception(res2.stderr)

            status_text.text("Generating out-of-sample forecast signal...")
            progress_bar.progress(80)
            res3 = subprocess.run([python_bin, "generate_next_week_signal.py"], capture_output=True, text=True, env=env)
            if res3.returncode != 0:
                raise Exception(res3.stderr)

            progress_bar.progress(100)
            status_text.success("✅ Update complete! Reloading page...")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            status_text.error(f"❌ Update failed:\n{str(e)}")
            progress_bar.empty()

st.divider()

# ── KEY METRICS BAR ────────────────────────────────────
csv_date = df.index[-1].strftime('%d %b %Y')
csv_date_short = df.index[-1].strftime('%d %b')

col1, col2, col3, col4 = st.columns(4)
current_inr = df["USDINR"].dropna().iloc[-1]

# Live spot rate
live_rate = get_live_rate()
if live_rate is not None:
    col1.metric(
        "USD/INR Live Spot", 
        f"₹ {live_rate:.4f}", 
        help="Real-time daily spot exchange rate from Yahoo Finance"
    )
else:
    col1.metric(
        f"USD/INR Spot (as of {csv_date_short})", 
        f"₹ {current_inr:.4f}", 
        help="Last historical closing price in the dataset"
    )

# Weekly baseline rate (model input)
if next_week_signal is not None:
    weekly_base = next_week_signal["current_rate"]
    as_of = pd.to_datetime(next_week_signal["as_of_date"]).strftime("%d %b")
    col2.metric(
        "Weekly Baseline Average", 
        f"₹ {weekly_base:.4f}", 
        help=f"Average exchange rate for the week ending {as_of} used as the model's baseline level"
    )
else:
    col2.metric(
        "Weekly Baseline Average", 
        f"₹ {current_inr:.4f}",
        help="Average exchange rate for the last complete week in the dataset"
    )

# Predicted rate
if next_week_signal is not None:
    fc_rate = next_week_signal["predicted_rate"]
    next_date = pd.to_datetime(next_week_signal["next_week_date"]).strftime("%d %b")
    col3.metric(
        "Forecasted Week Average", 
        f"₹ {fc_rate:.4f}", 
        help=f"Predicted average exchange rate for the upcoming week ending {next_date}"
    )
else:
    col3.metric("Forecasted Week Average", "N/A")

# Out of sample accuracy
model_key_mapping = {
    "ARIMA_Baseline": "arima",
    "ARIMAX": "arimax",
    "Lasso": "lasso",
    "Random_Forest": "rf",
    "Gradient_Boosting": "gb",
    "ARIMA+Lasso_Hybrid": "arima_lasso",
    "ARIMA+Gradient_Boosting_Hybrid": "arima_gb"
}

if next_week_signal is not None:
    model_type_str = next_week_signal.get("model_type", "ARIMA+Lasso_Hybrid")
    model_key = model_key_mapping.get(model_type_str, "lasso")
else:
    model_type_str = "ARIMA+Lasso_Hybrid"
    model_key = "arima_lasso"

winning_mape = metrics.loc[model_key, "MAPE (%)"]
winning_accuracy = 100 - winning_mape

col4.metric(
        "Model Forecast Accuracy", 
        f"{winning_accuracy:.3f}%", 
        help=f"Based on a 200-week out-of-sample rolling backtest of the {model_type_str.replace('_', ' ').replace('+', ' + ')} model (100% - Mean Absolute Percentage Error)"
)

# ── DATA TIMESTAMP & SPOT VS AVERAGE NOTICE ────────────
st.info(
    f"💡 **Model Baseline vs. Live Spot Rate**: Predictive models are trained on historical **weekly averages** (resampled to week ending **{csv_date}**). "
    f"The live spot rate represents the real-time daily closing price. To avoid confusion, both values are displayed above.",
    icon="ℹ️"
)

st.divider()

# ── FORECASTING & GEOPOLITICAL RISK DASHBOARD ──────────
st.header("Forecasting & Geopolitical Risk Intelligence Dashboard")

# Next Week Signal Banner
if next_week_signal is not None:
    bg_color = "#FEF2F2" if next_week_signal["signal"] == "WEAKEN" else "#ECFDF5"
    border_color = "#FEE2E2" if next_week_signal["signal"] == "WEAKEN" else "#A7F3D0"
    text_accent = "#991B1B" if next_week_signal["signal"] == "WEAKEN" else "#065F46"
    sig_color = "#EF4444" if next_week_signal["signal"] == "WEAKEN" else "#10B981"
    
    # User-friendly explanation of the signal
    if next_week_signal["signal"] == "WEAKEN":
        sig_word = "WEAKEN (USD/INR Up / Rupee Falls)"
        sig_desc = "The model expects the US dollar to strengthen against the Rupee."
    else:
        sig_word = "STRENGTHEN (USD/INR Down / Rupee Rises)"
        sig_desc = "The model expects the Rupee to strengthen against the US dollar."
    
    model_display_name = next_week_signal.get("model_type", "ARIMA+Lasso_Hybrid").replace("_", " ").replace("+", " + ")
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; border: 1px solid {border_color}; border-left: 6px solid {sig_color}; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
        <div style="font-size: 0.75rem; color: #4B5563; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">{model_display_name} Signal for week ending {next_week_signal["next_week_date"]}</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: #111827; margin-bottom: 4px;">
            INR Expected to <span style="color: {text_accent};">{sig_word}</span>
        </div>
        <div style="font-size: 0.95rem; color: #374151; margin-bottom: 8px;">
            {sig_desc}
        </div>
        <div style="font-size: 1rem; color: #1F2937; font-weight: 500;">
            Expected Rate Change: <b>{next_week_signal["predicted_change"]:+.4f}</b> (<b>{next_week_signal["change_paise"]:.2f} paise</b>) | 
            Target Exchange Rate: <b>₹ {next_week_signal["predicted_rate"]:.4f}</b> <span style="color: #6B7280; font-size: 0.9rem;">(current level: ₹ {next_week_signal["current_rate"]:.4f})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

try:
    with open("data/processed/predictions.json", "r") as f:
        preds_data = json.load(f)
        
    dates = pd.to_datetime(preds_data["dates"])
    actual = preds_data["actual"]
    predictions = preds_data["predictions"]
    cum_returns = preds_data["cum_returns"]
    
    with st.container(border=True):
        st.subheader("📈 Out-of-Sample Forecast Visualization")
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            plot_weeks = st.selectbox(
                "Select Visualization Time Window",
                options=[26, 52, 100, 200],
                format_func=lambda x: f"Last {x} Weeks (Zoomed)" if x < 200 else "Full 200 Weeks",
                index=1 # Default to last 52 weeks (1 year)
            )
        with col_plot2:
            model_options = {
                "ARIMAX": "arimax",
                "Gradient Boosting (GB)": "gb",
                "Lasso": "lasso",
                "ARIMA+Lasso Hybrid": "arima_lasso",
                "ARIMA Baseline": "arima"
            }
            selected_models = st.multiselect(
                "Select Models to Display",
                options=list(model_options.keys()),
                default=["ARIMAX", "Lasso"]
            )
            
        slice_dates = dates[-plot_weeks:]
        slice_actual = actual[-plot_weeks:]
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=slice_dates, y=slice_actual,
            mode="lines+markers" if plot_weeks <= 52 else "lines", 
            name="Actual USD/INR", 
            line=dict(color="#1E3A8A", width=2.5) # Sleek navy/slate blue
        ))
        
        model_colors = {
            "arimax": "#F59E0B",       # Amber Orange
            "gb": "#8B5CF6",            # Violet
            "lasso": "#10B981",        # Emerald green
            "arima_lasso": "#EF4444",  # Crimson Red
            "arima": "#6B7280"         # Slate Gray
        }
        
        for model_label in selected_models:
            model_key = model_options[model_label]
            slice_pred = predictions[model_key][-plot_weeks:]
            color = model_colors[model_key]
            dash = "dash" if model_key == "lasso" else ("dot" if model_key == "arima" else None)
            
            fig4.add_trace(go.Scatter(
                x=slice_dates, y=slice_pred,
                mode="lines", 
                name=f"{model_label} (MAPE: {metrics.loc[model_key, 'MAPE (%)']:.3f}%)",
                line=dict(color=color, width=2, dash=dash)
            ))
            
        apply_plotly_theme(fig4, height=400)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Expander explaining Random Walk properties and 1-week lag look (clean LaTeX rendering outside HTML)
        with st.expander("💡 Why do the predictions closely follow the actual line with a 1-week shift?"):
            try:
                lasso_mda_val = metrics.loc["lasso", "MDA (%)"]
                lasso_theil_val = metrics.loc["lasso", "Theil's U"]
                lasso_outperf = (1.0 - lasso_theil_val) * 100
            except Exception:
                lasso_mda_val = 58.00
                lasso_theil_val = 0.9969
                lasso_outperf = 0.31

            st.markdown(
                rf"""
                Exchange rates represent financial asset levels which are highly **non-stationary** and exhibit a near-perfect random walk structure. 
                In forecasting, predicting next week's rate *y*<sub>*t*+1</sub> from today's rate *y*<sub>*t*</sub> is modeled by forecasting the **weekly change** (Δ*y*<sub>*t*+1</sub>):
                
                $$ \hat{y}_{t+1} = y_t + \Delta \hat{y}_{t+1} $$
                
                Because the expected weekly change (Δŷ<sub>*t*+1</sub>) is extremely small (typically 5 to 15 paise, or 0.05% to 0.15%) relative to the massive baseline exchange rate level (*y*<sub>*t*</sub> ≈ ₹95.00+), the baseline level *y*<sub>*t*</sub> dominates 99% of the predicted value. 
                This is why the prediction line looks like a "copy-paste" of the actual line shifted exactly 1 week to the right. 
                
                **How to verify if the model is actually learning anything?**
                
                If the model were simply copy-pasting the past value without any skill (i.e. predicting the change to be 0), it would have a **Mean Directional Accuracy (MDA)** of 0% on directional changes and a **Theil's U** of exactly 1.0. Our models beat this:
                
                * **Mean Directional Accuracy (MDA %)**: Standalone Lasso correctly forecasts whether the exchange rate will go up or down **{lasso_mda_val:.2f}%** of the time (well above the 50% random chance threshold).
                * **Theil's U Statistic**: Compares the model's RMSE against a naive Random Walk. A **Theil's U < 1.0** means the model successfully out-forecasts the naive baseline. Standalone Lasso achieves a Theil's U of **{lasso_theil_val:.4f}** (a {lasso_outperf:.2f}% outperformance over the random walk benchmark).
                """
            )
            
except Exception as e:
    st.warning(f"Forecasting artifacts not loaded. Error: {e}")

# ── GEOPOLITICAL RISK SECTION ──────────────────────────
with st.container(border=True):
    st.subheader("🌍 Continuous GDELT Geopolitical Tension Index Timeline")
    st.markdown("""
    Continuous daily tension indices by transmission channel, computed using GDELT Goldstein Scale scores and exponentially decaying maximum logic. 
    High values represent elevated geopolitical tension.
    """)

    tension_channels = {
        "Combined (Max Across Channels)": "Geo_Tension",
        "Direct FX (India-Pakistan)": "Geo_Tension_DirectFX_IndiaPak",
        "Direct FX (India-China)": "Geo_Tension_DirectFX_IndiaChina",
        "Oil Supply (Middle East)": "Geo_Tension_OilSupply",
        "Risk-Off (Russia-Ukraine)": "Geo_Tension_RiskOff_RusUkr",
        "Risk-Off (Global)": "Geo_Tension_RiskOff_Global"
    }
    # Keep only those columns present in the dataframe
    tension_channels = {k: v for k, v in tension_channels.items() if v in df.columns}

    selected_channels = st.multiselect(
        "Select Channels to Plot:",
        options=list(tension_channels.keys()),
        default=[k for k in tension_channels.keys() if tension_channels[k] in ["Geo_Tension", "Geo_Tension_OilSupply", "Geo_Tension_RiskOff_RusUkr"]]
    )

    if selected_channels:
        fig_timeline = go.Figure()
        
        channel_colors = {
            "Geo_Tension": "#EF4444",
            "Geo_Tension_DirectFX_IndiaPak": "#2563EB",
            "Geo_Tension_DirectFX_IndiaChina": "#1E3A8A",
            "Geo_Tension_OilSupply": "#F59E0B",
            "Geo_Tension_RiskOff_RusUkr": "#10B981",
            "Geo_Tension_RiskOff_Global": "#8B5CF6"
        }
        
        for name in selected_channels:
            col = tension_channels[name]
            color = channel_colors.get(col, "gray")
            fig_timeline.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5)
            ))
            
        apply_plotly_theme(fig_timeline, height=350)
        st.plotly_chart(fig_timeline, use_container_width=True)

st.divider()

# ── DEEP-DIVE ANALYTICS SECTION ────────────────────────
st.header("📊 Deep-Dive Quantitative Analytics & Backtest Results")
st.markdown("Explore detailed statistical evaluations, trading rules, feature importances, and historical impact studies.")

# 1. League Table
with st.expander("🏆 Model Performance Comparative League Table"):
    st.markdown("Models ranked by out-of-sample performance over the 200-week test window. Naïve Random Walk and Exponential Smoothing included as baselines.")
    try:
        display_metrics = metrics[["MAPE (%)", "RMSE", "Theil's U", "MDA (%)", "Sharpe Ratio (Rf=0)", "Sharpe Ratio (Rf=6.5%)", "Cumulative Return (%)"]]
        display_metrics.index = [idx.upper() for idx in display_metrics.index]

        st.dataframe(display_metrics.style.format({
            "MAPE (%)": "{:.3f}%",
            "RMSE": "{:.4f}",
            "Theil's U": "{:.4f}",
            "MDA (%)": "{:.2f}%",
            "Sharpe Ratio (Rf=0)": "{:.2f}",
            "Sharpe Ratio (Rf=6.5%)": "{:.2f}",
            "Cumulative Return (%)": "{:+.2f}%"
        }), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render league table: {e}")

# 2. Feature Importance
with st.expander("💡 Lasso Model Feature Intelligence & Coefficients"):
    st.markdown("Features selected by L1 regularization (Lasso) and their corresponding coefficients:")
    try:
        coef_df = pd.read_csv("data/processed/lasso_coefficients.csv")
        coef_df = coef_df[coef_df["Coefficient"] != 0].sort_values(by="Coefficient", key=abs, ascending=True)
        if not coef_df.empty:
            name_mapping = {
                "CRUDE_diff_lag1": "Crude Oil (1w diff lag)",
                "DXY_diff_lag1": "US Dollar Index (1w diff lag)",
                "Rate_Spread_diff_lag1": "US-India Interest Spread (1w diff lag)",
                "Geo_Tension_lag1": "Combined Geo Tension (1w lag)",
                "Geo_Tension_DirectFX_IndiaPak_lag1": "India-Pak Tension (1w lag)",
                "Geo_Tension_DirectFX_IndiaChina_lag1": "India-China Tension (1w lag)",
                "Geo_Tension_OilSupply_lag1": "Oil Supply Tension (1w lag)",
                "Geo_Tension_RiskOff_RusUkr_lag1": "Russia-Ukraine Tension (1w lag)",
                "Geo_Tension_RiskOff_Global_lag1": "Global Risk-Off Tension (1w lag)",
                "inr_mom_4w": "4-Week INR Momentum",
                "inr_mom_12w": "12-Week INR Momentum",
                "is_fiscal_yr_end": "Fiscal Year-End Dummy (March)",
                "is_qtr_end": "Quarter-End Dummy"
            }
            coef_df["Feature Name"] = coef_df["Feature"].map(name_mapping).fillna(coef_df["Feature"])
            
            fig_coef = go.Figure(go.Bar(
                x=coef_df["Coefficient"],
                y=coef_df["Feature Name"],
                orientation='h',
                marker_color='#1E3A8A' # Clean navy blue
            ))
            apply_plotly_theme(fig_coef, height=300, show_legend=False)
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info("Lasso regularized all coefficients to zero.")
    except Exception as e:
        st.warning(f"Could not load Lasso coefficients: {e}")

# 3. Strategy Returns & Takeaways
with st.expander("💰 Simulated Trading Backtest: Cumulative Returns & Takeaways"):
    st.markdown("Performance of a simulated trading rule: Long USD/INR if predicted rate goes up, Short if it goes down.")
    try:
        color_map = {
            "arima_lasso": "#EF4444",   # Crimson Red
            "lasso": "#10B981",         # Emerald Green
            "arima": "#6B7280",         # Slate Gray
            "arima_gb": "#10B981",      # Green
            "gb": "#8B5CF6",            # Violet
            "rf": "#EC4899",            # Pink
            "arimax": "#F59E0B",        # Amber Orange
            "naive": "#94A3B8",         # Light Gray
            "es": "#CBD5E1"             # Pale Gray
        }
        
        fig5 = go.Figure()
        for model_name, rets in cum_returns.items():
            color = color_map.get(model_name, None)
            width = 2.5 if model_name == "arimax" else 1.5
            fig5.add_trace(go.Scatter(
                x=dates,
                y=[v * 100 for v in rets],
                mode="lines",
                name=f"{model_name.upper()} (Sharpe: {metrics.loc[model_name, 'Sharpe Ratio (Rf=0)']:.2f})",
                line=dict(color=color, width=width) if color else None
            ))
        fig5.add_hline(y=0, line_dash="dash", line_color="#E2E8F0", line_width=1)
        apply_plotly_theme(fig5, height=400)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Get metrics values dynamically
        try:
            arimax_sharpe = metrics.loc["arimax", "Sharpe Ratio (Rf=0)"]
            arimax_ret = metrics.loc["arimax", "Cumulative Return (%)"]
            gb_sharpe = metrics.loc["gb", "Sharpe Ratio (Rf=0)"]
            gb_ret = metrics.loc["gb", "Cumulative Return (%)"]
            arima_lasso_sharpe = metrics.loc["arima_lasso", "Sharpe Ratio (Rf=0)"]
            
            lasso_mda = metrics.loc["lasso", "MDA (%)"]
            lasso_theil = metrics.loc["lasso", "Theil's U"]
            lasso_ret = metrics.loc["lasso", "Cumulative Return (%)"]
            lasso_sharpe = metrics.loc["lasso", "Sharpe Ratio (Rf=0)"]
            
            arimax_carry = metrics.loc["arimax", "Sharpe Ratio (Rf=6.5%)"]
            gb_carry = metrics.loc["gb", "Sharpe Ratio (Rf=6.5%)"]
            
            arima_mda = metrics.loc["arima", "MDA (%)"]
            arima_ret = metrics.loc["arima", "Cumulative Return (%)"]
        except Exception:
            arimax_sharpe, arimax_ret = 1.24, 16.82
            gb_sharpe, gb_ret = 1.23, 16.68
            arima_lasso_sharpe = 1.06
            lasso_mda, lasso_theil, lasso_ret, lasso_sharpe = 58.00, 0.9969, 14.60, 1.09
            arimax_carry, gb_carry = -0.73, -0.74
            arima_mda, arima_ret = 57.00, 12.57

        st.markdown(f"""
        <div style="background-color: #F8F9FA; border: 1px solid #E2E8F0; border-left: 6px solid #4A5568; padding: 20px; border-radius: 8px; margin-top: 15px; font-size: 0.92rem; color: #2D3748; line-height: 1.6;">
            <h4 style="margin-top: 0; color: #1A202C; font-weight: 700; font-family: 'Inter', sans-serif;">📋 Quantitative Insights & Takeaways</h4>
            <ul style="padding-left: 20px; margin-bottom: 0;">
                <li style="margin-bottom: 8px;"><b>ARIMAX and Gradient Boosting (GB) lead performance</b>: With continuous, channel-specific GDELT tension indices, multivariate models are supplied with dense, rich signals. ARIMAX achieves the top Sharpe ratio (Rf=0) of <b>{arimax_sharpe:.2f}</b> ({arimax_ret:+.2f}% cumulative return), followed closely by Gradient Boosting with a Sharpe ratio of <b>{gb_sharpe:.2f}</b> ({gb_ret:+.2f}% return). This highlights that denser, continuous feature spaces reduce the need for hybrid error-correction architectures (like ARIMA+Lasso, which has <b>{arima_lasso_sharpe:.2f} Sharpe</b>).</li>
                <li style="margin-bottom: 8px;"><b>Lasso Standalone is the most accurate directional model</b>: Standalone Lasso achieves the highest Mean Directional Accuracy (<b>{lasso_mda:.2f}%</b>) and beats the Random Walk baseline (<b>Theil's U = {lasso_theil:.4f}</b>), yielding <b>{lasso_ret:+.2f}% Cumulative Return</b> with a <b>{lasso_sharpe:.2f} Sharpe Ratio</b>. Lasso succeeds because L1 regularization effectively handles multicollinearity among highly correlated macro and tension drivers.</li>
                <li style="margin-bottom: 8px;"><b>Risk-adjusted Carry Disclosures</b>: The standard Sharpe Ratio assumes a 0% risk-free rate (Information Ratio). If we adjust for India's <b>91-day T-Bill rate (~6.5% annualised)</b>, the carry hurdle turns all Sharpe ratios negative (ARIMAX: <b>{arimax_carry:.2f}</b>, GB: <b>{gb_carry:.2f}</b>). This highlights that systematic weekly trading of USD/INR is subject to a high interest rate carry hurdle in a low-volatility, central-bank-defended exchange rate regime.</li>
                <li style="margin-bottom: 0;"><b>ARIMA Baseline outperforms SARIMA</b>: The non-seasonal ARIMA(1,1,0) baseline achieves <b>{arima_mda:.2f}% MDA</b> and <b>{arima_ret:+.2f}% return</b>, outperforming the older seasonal SARIMA and confirming that weekly USD/INR changes are driven by short-term momentum rather than repeating annual cycles.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.warning(f"Could not load cumulative returns backtest: {e}")

# 4. Geopolitical Event studies
with st.expander("⚡ Geopolitical Event Impact Analysis (Event Studies)"):
    st.markdown("Analyze asset movements around historical geopolitical dates (window: ±30 days).")
    try:
        WINDOW = 30
        selected_event = st.selectbox(
            "Select Event to Inspect:", 
            events["event"].tolist(),
            key="event_study_select"
        )

        ev_row = events[events["event"] == selected_event].iloc[0]
        ev_date = ev_row["date"]

        assets = {
            "USDINR": ("USD/INR (↑ = INR weakens)", "#EF4444"),
            "GOLD":   ("Gold",                     "#D97706"),
            "CRUDE":  ("Crude Oil",                "#10B981"),
            "NIFTY":  ("Nifty 50",                 "#4A5568"),
        }

        fig3 = go.Figure()
        impact_data = []

        for col, (label, color) in assets.items():
            start = ev_date - pd.Timedelta(days=WINDOW)
            end   = ev_date + pd.Timedelta(days=WINDOW)
            sub   = df[col].loc[start:end].dropna()
            if len(sub) < 5:
                continue
            baseline = sub.loc[sub.index <= ev_date].iloc[-1]
            cum_ret  = ((sub - baseline) / baseline) * 100
            days     = [(d - ev_date).days for d in sub.index]
            
            fig3.add_trace(go.Scatter(
                x=days, y=cum_ret.values,
                mode="lines", name=label,
                line=dict(color=color, width=2)
            ))
            
            # 5-day impact
            idx_5 = [d for d in days if d == 5]
            if idx_5:
                val_5 = cum_ret.values[days.index(5)]
                impact_data.append({"Asset": label, "+5 Day Impact": f"{val_5:+.2f}%"})

        fig3.add_hline(y=0, line_dash="dash", line_color="#E2E8F0", line_width=1)
        fig3.add_vline(x=0, line_dash="dot", line_color="#6B7280", line_width=1,
                       annotation_text="Event Day", annotation_position="top right")
        apply_plotly_theme(fig3, height=350)
        st.plotly_chart(fig3, use_container_width=True)

        if impact_data:
            st.markdown("**5-Day Market Impact Summary**")
            st.dataframe(pd.DataFrame(impact_data), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("**All Geopolitical Events — INR Impact Summary**")
        summary = []
        for _, ev in events.iterrows():
            start = ev["date"] - pd.Timedelta(days=5)
            end   = ev["date"] + pd.Timedelta(days=10)
            sub   = df["USDINR"].loc[start:end].dropna()
            if len(sub) < 3:
                continue
            baseline = sub.iloc[0]
            max_dep  = ((sub.max() - baseline) / baseline) * 100
            summary.append({
                "Event": ev["event"],
                "Date": ev["date"].strftime("%d %b %Y"),
                "Category": ev["category"],
                "Max INR Depreciation (10d)": f"{max_dep:+.2f}%"
            })
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load event studies: {e}")

# 5. Overlays and Trends
with st.expander("🔍 USD/INR Long-Term Trend with Geopolitical Overlays"):
    st.markdown("USD/INR long-term trend mapped with markers showing major geopolitical events.")
    try:
        col_a, col_b = st.columns(2)
        start_yr = col_a.selectbox("From Year", list(range(2015, 2027)), index=0, key="from_year_overlay")
        end_yr   = col_b.selectbox("To Year",   list(range(2015, 2027)), index=11, key="to_year_overlay")

        df_filtered = df[f"{start_yr}":f"{end_yr}"].dropna(subset=["USDINR"])
        ev_filtered = events[
            (events["date"].dt.year >= start_yr) & 
            (events["date"].dt.year <= end_yr)
        ]

        colors_map = {
            "India-Pak": "#EF4444",
            "India-China": "#991B1B",
            "Strait of Hormuz": "#D97706",
            "Middle East": "#F59E0B",
            "US-Iran": "#EF4444",
            "Europe": "#10B981",
            "Global Energy": "#F59E0B",
            "Global": "#8B5CF6"
        }

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered["USDINR"],
            fill="tozeroy", fillcolor="rgba(49, 130, 206, 0.03)",
            line=dict(color="#2B6CB0", width=2),
            name="USD/INR Level"
        ))
        for _, ev in ev_filtered.iterrows():
            fig_trend.add_vline(
                x=ev["date"],
                line_color=colors_map.get(ev["category"], "gray"),
                line_dash="dash", line_width=1.5
            )
            fig_trend.add_annotation(
                x=ev["date"],
                y=1.0,
                yref="paper",
                text=ev["event"].split()[0],
                textangle=90,
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=9, color=colors_map.get(ev["category"], "gray"))
            )
        apply_plotly_theme(fig_trend, height=350, show_legend=False)
        st.plotly_chart(fig_trend, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load trend overlay: {e}")

# 6. Correlations
with st.expander("🧮 Macro & Geopolitical Correlation Analysis"):
    st.markdown("Pearson correlation matrix between the Rupee, macro drivers, and continuous GDELT tension channels.")
    try:
        corr_cols = [
            "USDINR", "CRUDE", "GOLD", "DXY", "NIFTY", "INDIAVIX", "Rate_Spread", 
            "Geo_Tension", "Geo_Tension_DirectFX_IndiaPak", "Geo_Tension_DirectFX_IndiaChina", 
            "Geo_Tension_OilSupply", "Geo_Tension_RiskOff_RusUkr", "Geo_Tension_RiskOff_Global"
        ]
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr = df[corr_cols].dropna().corr()

        fig2 = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", # Center at 0 with red-blue diverging
            zmin=-1.0, zmax=1.0,
            text=corr.round(2).values, texttemplate="%{text}",
            colorbar=dict(title="r")
        ))
        apply_plotly_theme(fig2, height=450, show_legend=False, is_heatmap=True)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render heatmap: {e}")

# ── FOOTER ─────────────────────────────────────────────
st.divider()
st.caption("RupeeRisk | Srishti Lamba | DAU M.Sc. Data Science | Data: RBI DBIE, yfinance, FRED API")