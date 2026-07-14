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
    df      = pd.read_csv("data/processed/master_df.csv",        index_col=0, parse_dates=True)
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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* Apply modern font and clean white background */
html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: #FFFFFF !important;
    color: #4A4238 !important;
}

/* Header style overrides */
h1, h2, h3, h4, h5, h6 {
    color: #2E2A25 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
}

/* Hide Streamlit sidebar since controls are now in header */
[data-testid="selectedSidebarView"] {
    display: none;
}
section[data-testid="stSidebar"] {
    display: none;
}

/* Style metric cards with sleek borders and left accents */
div[data-testid="metric-container"] {
    background-color: #FAF8F5 !important;
    border: 1px solid #E6DFD3 !important;
    border-left: 6px solid #C29F74 !important;
    padding: 18px 24px;
    border-radius: 12px !important;
    box-shadow: 0 4px 10px rgba(139, 115, 85, 0.03) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    border-left: 6px solid #B08E65 !important;
    box-shadow: 0 8px 18px rgba(139, 115, 85, 0.06) !important;
}

/* Custom styles for metric labels and values */
div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #2E2A25 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    color: #8C7E6E !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Custom styling for inputs and selectors */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid #E6DFD3 !important;
}
div[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E6DFD3 !important;
}

/* Custom styling for Streamlit native bordered containers (White/Beige Card layout) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FAF8F5 !important;
    border: 1px solid #EADFC9 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 4px 12px rgba(139, 115, 85, 0.02) !important;
    margin-bottom: 25px !important;
}

/* Styling buttons */
div.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    background-color: #C29F74 !important;
    color: #FFFFFF !important;
    border: 1px solid #B08E65 !important;
    transition: all 0.2s ease-in-out !important;
    padding: 10px 20px !important;
}
div.stButton > button:hover {
    background-color: #B08E65 !important;
    border-color: #8C6A3C !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
}

/* Styling dividers */
hr {
    border-top: 1px solid #EADFC9 !important;
    margin: 30px 0 !important;
}

/* Dataframe styling */
.stDataFrame, div[data-testid="stTable"] {
    background-color: #FFFFFF !important;
    border-radius: 12px;
    border: 1px solid #E6DFD3 !important;
    padding: 5px;
    box-shadow: 0 4px 10px rgba(139, 115, 85, 0.01) !important;
}

/* Widget label colors */
label[data-testid="stWidgetLabel"] {
    color: #4A4238 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY HELPER THEME ─────────────────────────────────
def apply_plotly_theme(fig, height=400, show_legend=True, is_heatmap=False):
    layout_args = dict(
        height=height,
        template="plotly_white",
        plot_bgcolor="#FAFAF9" if not is_heatmap else None,
        paper_bgcolor="#FAF8F5", # Blends seamlessly with our warm-beige cards
        font=dict(family="'Space Grotesk', sans-serif", color="#4A4238"),
        margin=dict(t=40, b=40, l=50, r=20)
    )
    if not is_heatmap:
        layout_args["xaxis"] = dict(
            gridcolor="#EADFC9",
            linecolor="#DCD5C8",
            tickfont=dict(color="#6E6254"),
            title=dict(font=dict(color="#4A4238"))
        )
        layout_args["yaxis"] = dict(
            gridcolor="#EADFC9",
            linecolor="#DCD5C8",
            tickfont=dict(color="#6E6254"),
            title=dict(font=dict(color="#4A4238"))
        )
    if show_legend:
        layout_args["legend"] = dict(
            orientation="h",
            y=1.18,
            x=0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(color="#4A4238")
        )
    fig.update_layout(**layout_args)

# ── HEADER ─────────────────────────────────────────────
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("""
    <div style="margin-bottom: 10px;">
        <h1 style="margin: 0; font-size: 3rem; font-weight: 700; background: linear-gradient(to right, #8C6A3C, #4A3E31); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Space Grotesk', sans-serif;">RupeeRisk</h1>
        <p style="margin: 6px 0 0 0; font-size: 1.2rem; color: #5C5245; font-weight: 400; letter-spacing: 0.3px;">India Macro & Geopolitical Forex Intelligence Platform</p>
        <div style="font-size: 0.85rem; color: #8C7E6E; margin-top: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px;">Advanced Analytics, Econometrics, and Machine Learning Backtesting</div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.write(" ")
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
col1, col2, col3, col4 = st.columns(4)
current_inr = df["USDINR"].dropna().iloc[-1]
inr_ytd_start = df["USDINR"].dropna()["2026-01-01":].iloc[0]
ytd_change = ((current_inr - inr_ytd_start) / inr_ytd_start) * 100

# Live rate fetch
live_rate = get_live_rate()
if live_rate is not None:
    col1.metric("Current USD/INR (Live)", f"₹ {live_rate:.4f}")
    live_ytd_change = ((live_rate - inr_ytd_start) / inr_ytd_start) * 100
    col2.metric("YTD Change", f"{live_ytd_change:+.2f}%", 
                delta=f"{live_ytd_change:+.2f}%", delta_color="inverse")
else:
    col1.metric("Current USD/INR (Stale)", f"₹ {current_inr:.4f}")
    col2.metric("YTD Change", f"{ytd_change:+.2f}%", 
                delta=f"{ytd_change:+.2f}%", delta_color="inverse")

col3.metric("Best Model (Lasso) MAPE", f"{metrics.loc['lasso','MAPE (%)']:.3f}%")
col4.metric("Geopolitical Events Analysed", f"{len(events)}")

# ── DATA TIMESTAMP & SPOT VS AVERAGE NOTICE ────────────
csv_date = df.index[-1].strftime('%d %b %Y')
st.markdown(f"""
<div style="background-color: #F0F5FA; border: 1px solid #D2DFEE; border-left: 6px solid #3182CE; padding: 15px 20px; border-radius: 10px; margin-bottom: 25px; font-size: 0.9rem; color: #2B6CB0; line-height: 1.5;">
    ℹ️ <b>System Notice</b>: Predictive models are trained and validated on historical <b>weekly averages</b> resampled up to <b>{csv_date}</b>. 
    The <b>Current USD/INR (Live)</b> metrics card above displays the <b>real-time daily spot exchange rate</b> (live from Yahoo Finance). 
    Because weekly averages smooth out daily microstructural noise, the baseline rate in the forecasting section below represents the last week's average (₹ {df['USDINR'].dropna().iloc[-1]:.4f}) rather than the instantaneous real-time spot rate.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── SECTION 1: PREDICTIVE FORECASTING & BACKTEST ANALYTICS ──
st.header("🎯 INR/USD Predictive Forecasting & Backtest Analytics")
st.markdown("""
This section presents the out-of-sample forecasting dashboard. Models are trained on historical weekly averages and evaluated over a rolling 200-week test window.
""")

# Next Week Signal Box
if next_week_signal is not None:
    st.markdown("### 🎯 Out-of-Sample Forex Signal (Upcoming Week)")
    bg_color = "#FFF5F5" if next_week_signal["signal"] == "WEAKEN" else "#F3FBF9"
    border_color = "#FEB2B2" if next_week_signal["signal"] == "WEAKEN" else "#C2E8DF"
    text_accent = "#9B2226" if next_week_signal["signal"] == "WEAKEN" else "#1A5F49"
    sig_color = "#D62828" if next_week_signal["signal"] == "WEAKEN" else "#2A9D8F"
    sig_word = "WEAKEN (USD/INR Up / Rupee Falls)" if next_week_signal["signal"] == "WEAKEN" else "STRENGTHEN (USD/INR Down / Rupee Rises)"
    
    model_display_name = next_week_signal.get("model_type", "ARIMA+Lasso_Hybrid").replace("_", " ").replace("+", " + ")
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; border: 1px solid {border_color}; border-left: 6px solid {sig_color}; padding: 22px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
        <div style="font-size: 0.85rem; color: #7A7062; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">{model_display_name} Signal for week ending {next_week_signal["next_week_date"]}</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #2E2A25; margin-bottom: 8px;">
            INR Expected to <span style="color: {text_accent};">{sig_word}</span>
        </div>
        <div style="font-size: 1.05rem; color: #4A4238; font-weight: 500;">
            Expected Rate Change: <b>{next_week_signal["predicted_change"]:+.4f}</b> (<b>{next_week_signal["change_paise"]:.2f} paise</b>)
        </div>
        <div style="font-size: 1.05rem; color: #4A4238; font-weight: 500; margin-top: 4px;">
            Target Exchange Rate: <b>₹ {next_week_signal["predicted_rate"]:.4f}</b> (current level: ₹ {next_week_signal["current_rate"]:.4f})
        </div>
        <div style="font-size: 0.78rem; color: #8C7E6E; font-style: italic; margin-top: 12px; border-top: 1px solid {border_color}; padding-top: 8px;">
            * This signal is generated out-of-sample by the {model_display_name} model trained on all historical data up to {next_week_signal["as_of_date"]}.
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
        
        # User controls for zoom window and model selection
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
            
        # Slice predictions and actuals based on the selected weeks
        slice_dates = dates[-plot_weeks:]
        slice_actual = actual[-plot_weeks:]
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=slice_dates, y=slice_actual,
            mode="lines+markers" if plot_weeks <= 52 else "lines", 
            name="Actual USD/INR", 
            line=dict(color="#3182CE", width=2.5) # Sleek slate blue
        ))
        
        # Consistent color mapping for the predictions chart
        model_colors = {
            "arimax": "#E28743",       # Warm Gold/Bronze
            "gb": "#76B5C5",            # Muted Teal Blue
            "lasso": "#2A9D8F",        # Emerald green
            "arima_lasso": "#D62828",  # Crimson Red
            "arima": "#8C7E6E"         # Slate Gray
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
        
        # Lagged Prediction Explanation Block
        st.markdown("""
        <div style="background-color: #FCF9F2; border: 1px solid #EADFCA; border-left: 6px solid #D4A373; padding: 20px; border-radius: 12px; margin-top: 15px; font-size: 0.9rem; color: #4A4238; line-height: 1.6; box-shadow: 0 4px 6px rgba(139, 115, 85, 0.02);">
            💡 <b>Why do the predictions closely follow the actual line with a 1-week shift?</b><br><br>
            Exchange rates represent financial asset levels which are highly <b>non-stationary</b> and exhibit a near-perfect random walk structure. 
            In forecasting, using today's exchange rate ($y_t$) to predict next week's rate ($y_{t+1}$) is modeled by predicting the <b>weekly change</b> ($\\Delta y_{t+1}$):
            <div style="text-align: center; margin: 12px 0; font-family: 'Courier New', monospace; font-weight: bold; font-size: 1.15rem; color: #8C6A3C;">
                ŷ<sub>t</sub> = y<sub>t-1</sub> + Δŷ<sub>t</sub>
            </div>
            Because weekly changes ($\\Delta \\hat{y}_t$) are extremely small (typically 5 to 15 paise, or 0.1% to 0.3%) relative to the massive baseline exchange rate level (₹95.00+), the $y_{t-1}$ level component dominates 99% of the prediction. 
            This is why the prediction line looks like a "copy-paste" of the actual line shifted exactly 1 week to the right. 
            <br><br>
            <b>How to verify if the model is actually learning anything?</b><br>
            If the model were simply copy-pasting the past value without any skill (i.e. predicting $\\Delta \\hat{y}_t = 0$), it would have a <b>Mean Directional Accuracy (MDA)</b> of 0% on directional changes and a <b>Theil's U</b> of exactly 1.0. Our models beat this:
            <ul>
                <li><b>Mean Directional Accuracy (MDA %)</b>: Standalone Lasso correctly forecasts whether the exchange rate will go up or down <b>58.00%</b> of the time (well above the 50% random chance threshold).</li>
                <li><b>Theil's U Statistic</b>: Compares the model's RMSE against a naive Random Walk. A <b>Theil's U &lt; 1.0</b> means the model successfully out-forecasts the naive baseline. Standalone Lasso achieves a Theil's U of <b>0.9969</b> (a 0.31% outperformance over the random walk benchmark).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("🏆 Model Performance League Table")
        st.caption("Models ranked by out-of-sample performance over the 200-week test window. Naïve Random Walk and Exponential Smoothing included as baselines.")
        
        # Select and format columns for the dashboard
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
    
    # Lasso Feature Importance
    with st.container(border=True):
        st.subheader("💡 Lasso Model Feature Intelligence")
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
                    marker_color='#C29F74' # Golden bronze color
                ))
                apply_plotly_theme(fig_coef, height=300, show_legend=False)
                st.plotly_chart(fig_coef, use_container_width=True)
            else:
                st.info("Lasso regularized all coefficients to zero.")
        except Exception as e:
            st.warning(f"Could not load Lasso coefficients: {e}")

    # Cumulative returns plot
    with st.container(border=True):
        st.subheader("💰 Trading Strategy Backtest: Cumulative Returns")
        st.caption("Performance of a simulated trading rule: Long USD/INR if predicted rate goes up, Short if it goes down.")
        
        color_map = {
            "arima_lasso": "#D62828",   # Crimson Red
            "lasso": "#2A9D8F",         # Teal Green
            "arima": "#8C7E6E",         # Slate Gray
            "arima_gb": "#4CAF50",      # Green
            "gb": "#76B5C5",            # Soft Blue
            "rf": "#9b59b6",            # Amethyst Purple
            "arimax": "#E28743",        # Warm Orange/Bronze
            "naive": "#B0A8A0",         # Light Gray
            "es": "#D2C9C0"             # Pale Gray
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
        fig5.add_hline(y=0, line_dash="dash", line_color="#DCD5C8", line_width=1)
        apply_plotly_theme(fig5, height=400)
        st.plotly_chart(fig5, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #FAF6F0; border: 1px solid #E6DFD3; border-left: 6px solid #8C6A3C; padding: 22px; border-radius: 12px; margin-top: 25px; font-size: 0.92rem; color: #4A4238; line-height: 1.6;">
            <h4 style="margin-top: 0; color: #2E2A25; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">📋 Quantitative Insights & Takeaways</h4>
            <ul style="padding-left: 20px; margin-bottom: 0;">
                <li style="margin-bottom: 10px;"><b>ARIMAX and Gradient Boosting (GB) lead performance</b>: With continuous, channel-specific GDELT tension indices, multivariate models are supplied with dense, rich signals. ARIMAX achieves the top Sharpe ratio (Rf=0) of <b>1.24</b> (+16.82% cumulative return), followed closely by Gradient Boosting with a Sharpe ratio of <b>1.23</b> (+16.68% return). This highlights that denser, continuous feature spaces reduce the need for hybrid error-correction architectures (like ARIMA+Lasso, which has <b>1.06 Sharpe</b>).</li>
                <li style="margin-bottom: 10px;"><b>Lasso Standalone is the most accurate directional model</b>: Standalone Lasso achieves the highest Mean Directional Accuracy (<b>58.00%</b>) and beats the Random Walk baseline (<b>Theil's U = 0.9969</b>), yielding <b>+14.60% Cumulative Return</b> with a <b>1.09 Sharpe Ratio</b>. Lasso succeeds because L1 regularization effectively handles multicollinearity among highly correlated macro and tension drivers.</li>
                <li style="margin-bottom: 10px;"><b>Risk-adjusted Carry Disclosures</b>: The standard Sharpe Ratio assumes a 0% risk-free rate (Information Ratio). If we adjust for India's <b>91-day T-Bill rate (~6.5% annualised)</b>, the carry hurdle turns all Sharpe ratios negative (ARIMAX: <b>-0.73</b>, GB: <b>-0.74</b>). This highlights that systematic weekly trading of USD/INR is subject to a high interest rate carry hurdle in a low-volatility, central-bank-defended exchange rate regime.</li>
                <li style="margin-bottom: 0;"><b>ARIMA Baseline outperforms SARIMA</b>: The non-seasonal ARIMA(1,1,0) baseline achieves <b>57.00% MDA</b> and <b>12.57% return</b>, outperforming the older seasonal SARIMA and confirming that weekly USD/INR changes are driven by short-term momentum rather than repeating annual cycles.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
except Exception as e:
    st.warning(f"Forecasting artifacts not loaded. Error: {e}")

st.divider()

# ── SECTION 2: GEOPOLITICAL RISK INDEX & EVENT IMPACT STUDIES ──
st.header("🌍 Geopolitical Risk Index & Event Impact Studies")
st.markdown("""
This section explores historical geopolitical events, details the continuous GDELT-sourced severity-weighted tension channels, and gauges their short-term impact on exchange rates and commodities.
""")

# Continuous GDELT Tension index timeline
with st.container(border=True):
    st.subheader("📊 GDELT-Sourced Severity-Weighted Geopolitical Tension Index Timeline")
    st.caption("Continuous daily tension indices by transmission channel, computed using GDELT Goldstein Scale scores and exponentially decaying maximum logic.")

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
            "Geo_Tension": "#D62828",
            "Geo_Tension_DirectFX_IndiaPak": "#457B9D",
            "Geo_Tension_DirectFX_IndiaChina": "#1D3557",
            "Geo_Tension_OilSupply": "#E69F00",
            "Geo_Tension_RiskOff_RusUkr": "#2A9D8F",
            "Geo_Tension_RiskOff_Global": "#9B59B6"
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
            
        apply_plotly_theme(fig_timeline, height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)

# Event study drop down selector
with st.container(border=True):
    st.subheader("⚡ Event Study — Market Response to Geopolitical Events")
    st.caption("Cumulative % change in each asset from the event date | Window: ±30 days")

    WINDOW = 30
    selected_event = st.selectbox(
        "Select Event to Inspect:", 
        events["event"].tolist()
    )

    ev_row = events[events["event"] == selected_event].iloc[0]
    ev_date = ev_row["date"]

    assets = {
        "USDINR": ("USD/INR (↑ = INR weakens)", "#D62828"),
        "GOLD":   ("Gold",                     "#B58900"),
        "CRUDE":  ("Crude Oil",                "#2A9D8F"),
        "NIFTY":  ("Nifty 50",                 "#4A3E31"),
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

    fig3.add_hline(y=0, line_dash="dash", line_color="#DCD5C8", line_width=1)
    fig3.add_vline(x=0, line_dash="dot", line_color="#8C7E6E", line_width=1,
                   annotation_text="Event Day", annotation_position="top right")
    apply_plotly_theme(fig3, height=420)
    st.plotly_chart(fig3, use_container_width=True)

    if impact_data:
        st.markdown("**5-Day Market Impact Summary**")
        st.dataframe(pd.DataFrame(impact_data), hide_index=True, use_container_width=True)

# All events summary table
with st.container(border=True):
    st.subheader("📋 All Geopolitical Events — INR Impact Summary")
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

st.divider()

# ── SECTION 3: MACRO TRENDS & CORRELATIONS ANALYSIS ──
st.header("📈 Macro Trends & Correlation Analysis")
st.markdown("""
This section analyzes the long-term trends of USD/INR alongside major geopolitical escalations, as well as the historical correlations between the rupee and macro factors.
""")

with st.container(border=True):
    st.subheader("🔍 INR/USD Long-Term Trend with Geopolitical Overlays")

    # Date filter
    col_a, col_b = st.columns(2)
    start_yr = col_a.selectbox("From Year", list(range(2015, 2027)), index=0)
    end_yr   = col_b.selectbox("To Year",   list(range(2015, 2027)), index=11)

    df_filtered = df[f"{start_yr}":f"{end_yr}"].dropna(subset=["USDINR"])
    ev_filtered = events[
        (events["date"].dt.year >= start_yr) & 
        (events["date"].dt.year <= end_yr)
    ]

    colors_map = {
        "India-Pak": "#D62828",
        "India-China": "#9B2226",
        "Strait of Hormuz": "#F77F00",
        "Middle East": "#F4A261",
        "US-Iran": "#E76F51",
        "Europe": "#2A9D8F",
        "Global Energy": "#E9C46A",
        "Global": "#6A0572"
    }

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered["USDINR"],
        fill="tozeroy", fillcolor="rgba(140, 106, 60, 0.04)",
        line=dict(color="#C29F74", width=2),
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
    apply_plotly_theme(fig_trend, height=400, show_legend=False)
    st.plotly_chart(fig_trend, use_container_width=True)

# Correlation heatmap
with st.container(border=True):
    st.subheader("🧮 Macro & Tension Correlations with INR")
    corr_cols = [
        "USDINR", "CRUDE", "GOLD", "DXY", "NIFTY", "INDIAVIX", "Rate_Spread", 
        "Geo_Tension", "Geo_Tension_DirectFX_IndiaPak", "Geo_Tension_DirectFX_IndiaChina", 
        "Geo_Tension_OilSupply", "Geo_Tension_RiskOff_RusUkr", "Geo_Tension_RiskOff_Global"
    ]
    # Filter out columns that don't exist yet to be safe
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].dropna().corr()

    fig2 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="YlOrBr", # Beautiful warm beige/yellow/orange heatmap matching light theme
        text=corr.round(2).values, texttemplate="%{text}",
        colorbar=dict(title="r")
    ))
    apply_plotly_theme(fig2, height=450, show_legend=False, is_heatmap=True)
    st.plotly_chart(fig2, use_container_width=True)

# ── FOOTER ─────────────────────────────────────────────
st.divider()
st.caption("RupeeRisk | Srishti Lamba | DAU M.Sc. Data Science | Data: RBI DBIE, yfinance, FRED API")