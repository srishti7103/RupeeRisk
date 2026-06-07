import streamlit as st
import pandas as pd
import numpy as np
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

@st.cache_data
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

/* Apply modern font */
html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Style metric cards with sleek borders and left accents */
div[data-testid="metric-container"] {
    background-color: #161B22;
    border: 1px solid #30363d;
    border-left: 6px solid #2A9D8F !important;
    padding: 18px 24px;
    border-radius: 12px;
    box-shadow: none;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    border-left: 6px solid #2A9D8F !important;
}

/* Custom styles for metric labels and values */
div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #F0F6FC !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    color: #8b949e !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Custom style for tab headers */
div[data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 2px solid #30363d;
    padding-bottom: 5px;
}
button[data-baseweb="tab"] {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #8b949e !important;
    border-bottom: 3px solid transparent !important;
    padding: 10px 16px !important;
    transition: all 0.25s ease;
}
button[data-baseweb="tab"]:hover {
    color: #F0F6FC !important;
}
button[aria-selected="true"] {
    color: #F0F6FC !important;
    border-bottom: 3px solid #2A9D8F !important;
}

/* Styling alert boxes */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid #30363d !important;
    background-color: #161B22 !important;
}

/* Rounded inputs and selectors */
div[data-baseweb="select"] {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #102a43, #0b1d33); padding: 35px; border-radius: 20px; border: 1px solid #102a43; color: white; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(16, 42, 67, 0.15);">
    <h1 style="margin: 0; font-size: 3rem; font-weight: 700; background: linear-gradient(to right, #00b4db, #2A9D8F); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">RupeeRisk</h1>
    <p style="margin: 6px 0 0 0; font-size: 1.15rem; color: #b4c6ef; font-weight: 300; letter-spacing: 0.3px;">India Macro & Geopolitical Forex Intelligence Platform</p>
    <div style="font-size: 0.85rem; color: #627d98; margin-top: 18px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">Advanced Analytics, Econometrics, and Machine Learning Backtesting</div>
</div>
""", unsafe_allow_html=True)

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

# Data timestamp notice
csv_date = df.index[-1].strftime('%d %b %Y')
st.caption(f"ℹ️ **System Notice**: Predictive models are trained on historical data up to **{csv_date}**. The live rate card fetches dynamically from Yahoo Finance and refreshes hourly.")

st.divider()

# ── TABS ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Macro Dashboard", "Geopolitical Risk", "Forecasting"])

# ─────────────────────────────────────────────────────────
# TAB 1: MACRO DASHBOARD
# ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("INR/USD Trend with Geopolitical Events")
    
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
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered["USDINR"],
        fill="tozeroy", fillcolor="rgba(26,58,92,0.08)",
        line=dict(color="#1A3A5C", width=1.5),
        name="USD/INR"
    ))
    for _, ev in ev_filtered.iterrows():
        # Add vertical line without built-in annotation to avoid Plotly datetime sum bug
        fig.add_vline(
            x=ev["date"],
            line_color=colors_map.get(ev["category"], "gray"),
            line_dash="dash", line_width=1.5
        )
        # Add annotation manually
        fig.add_annotation(
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
    fig.update_layout(height=400, template="plotly_white",
                      yaxis_title="INR per 1 USD", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Macro Factor Correlations with INR")
    corr_cols = ["USDINR","CRUDE","GOLD","DXY","NIFTY","INDIAVIX","Rate_Spread","Geo_Tension"]
    corr = df[corr_cols].dropna().corr()
    
    fig2 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmid=0,
        text=corr.round(2).values, texttemplate="%{text}",
        colorbar=dict(title="r")
    ))
    fig2.update_layout(height=400, template="plotly_white",
                       title="Correlation Matrix — INR vs Macro Drivers")
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────
# TAB 2: GEOPOLITICAL RISK
# ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Event Study — Market Response to India's Geopolitical Events")
    st.caption("Cumulative % change in each asset from the event date | Window: ±30 days")
    
    WINDOW = 30
    selected_event = st.selectbox(
        "Select Event:", 
        events["event"].tolist()
    )
    
    ev_row = events[events["event"] == selected_event].iloc[0]
    ev_date = ev_row["date"]
    
    assets = {
        "USDINR": ("USD/INR (↑=INR weakens)", "#D62828"),
        "GOLD":   ("Gold",                     "#F4A261"),
        "CRUDE":  ("Crude Oil",                "#2A9D8F"),
        "NIFTY":  ("Nifty 50",                 "#1A3A5C"),
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
    
    fig3.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.8)
    fig3.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1,
                   annotation_text="Event Day", annotation_position="top right")
    fig3.update_layout(
        title=f"Event Study: {selected_event}",
        xaxis_title="Days from Event", yaxis_title="Cumulative Return (%)",
        template="plotly_dark", height=420,
        plot_bgcolor="#0D1117",
        paper_bgcolor="#0D1117",
        font=dict(color="#F0F6FC"),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    if impact_data:
        st.markdown("**5-Day Market Impact Summary**")
        st.dataframe(pd.DataFrame(impact_data), hide_index=True, use_container_width=True)
    
    # All events summary table
    st.subheader("All Events — INR Impact Summary")
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

# ─────────────────────────────────────────────────────────
# TAB 3: FORECASTING
# ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("INR/USD Advanced Machine Learning & Quantitative Backtesting")
    st.markdown("""
    **Methodology Overview**:
    - **Data Frequency**: Shifted to **Weekly averages** to provide 590+ observations, enabling robust machine learning training.
    - **Stationarity & Spurious Regression**: Continuous variables are first-differenced (changes/returns) to eliminate spurious correlation risks.
    - **No Look-Ahead Bias**: Exogenous features (Crude, DXY, spreads) are **lagged by 1 week** ($X_{t-1}$).
    - **Validation**: Evaluated using a **Rolling 1-Step-Ahead Validation** loop (**200-week test set** covering 2022–2026), re-estimating models weekly and anchoring on the previous week's level ($y_{t-1}$).
    """)

    # Next Week Signal Box (Phase 3 addition)
    if next_week_signal is not None:
        st.markdown("### 🎯 Out-of-Sample Forex Signal (Upcoming Week)")
        sig_color = "#D62828" if next_week_signal["signal"] == "WEAKEN" else "#2A9D8F"
        sig_word = "WEAKEN (USD/INR Up / Rupee Falls)" if next_week_signal["signal"] == "WEAKEN" else "STRENGTHEN (USD/INR Down / Rupee Rises)"
        
        model_display_name = next_week_signal.get("model_type", "ARIMA+GradientBoosting_Hybrid").replace("_", " ").replace("+", " + ")
        
        st.markdown(f"""
        <div style="background-color: #161B22; border: 1px solid #30363d; border-left: 6px solid {sig_color}; padding: 20px; border-radius: 12px; margin-bottom: 25px;">
            <div style="font-size: 0.85rem; color: #8b949e; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px;">{model_display_name} Signal for week ending {next_week_signal["next_week_date"]}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #F0F6FC; margin-bottom: 8px;">
                INR Expected to <span style="color: {sig_color};">{sig_word}</span>
            </div>
            <div style="font-size: 1.05rem; color: #c9d1d9; font-weight: 400;">
                Expected Rate Change: <b>{next_week_signal["predicted_change"]:+.4f}</b> (<b>{next_week_signal["change_paise"]:.2f} paise</b>)
            </div>
            <div style="font-size: 1.05rem; color: #c9d1d9; font-weight: 400; margin-top: 4px;">
                Target Exchange Rate: <b>₹ {next_week_signal["predicted_rate"]:.4f}</b> (current level: ₹ {next_week_signal["current_rate"]:.4f})
            </div>
            <div style="font-size: 0.75rem; color: #8b949e; font-style: italic; margin-top: 12px; border-top: 1px solid #30363d; padding-top: 8px;">
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
        
        st.subheader("Model Performance League Table")
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
        
        st.subheader("Out-of-Sample Forecast Visualization")
        
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
                "ARIMA+Lasso Hybrid": "arima_lasso",
                "Lasso": "lasso",
                "ARIMA Baseline": "arima"
            }
            selected_models = st.multiselect(
                "Select Models to Display",
                options=list(model_options.keys()),
                default=["ARIMA+Lasso Hybrid"]
            )
            
        # Slice predictions and actuals based on the selected weeks
        slice_dates = dates[-plot_weeks:]
        slice_actual = actual[-plot_weeks:]
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=slice_dates, y=slice_actual,
            mode="lines+markers" if plot_weeks <= 52 else "lines", 
            name="Actual USD/INR", 
            line=dict(color="#3498db", width=2.5) # Sleek electric blue
        ))
        
        # Consistent color mapping for the predictions chart
        model_colors = {
            "arima_lasso": "#D62828",  # Crimson Red
            "lasso": "#2A9D8F",        # Teal Green
            "arima": "#F4A261"         # Warm Orange
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
            
        fig4.update_layout(height=400, template="plotly_dark",
                           yaxis_title="USD/INR Level", xaxis_title="Date",
                           plot_bgcolor="#0D1117",
                           paper_bgcolor="#0D1117",
                           font=dict(color="#F0F6FC"),
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig4, use_container_width=True)
        
        # Lasso Feature Importance (Lasso Coefficient Chart)
        st.subheader("Lasso Model Feature Intelligence")
        st.markdown("Features selected by L1 regularization (Lasso) and their corresponding coefficients:")
        try:
            coef_df = pd.read_csv("data/processed/lasso_coefficients.csv")
            coef_df = coef_df[coef_df["Coefficient"] != 0].sort_values(by="Coefficient", key=abs, ascending=True)
            if not coef_df.empty:
                # Nicer names for the feature importance chart
                name_mapping = {
                    "CRUDE_diff_lag1": "Crude Oil (1w diff lag)",
                    "DXY_diff_lag1": "US Dollar Index (1w diff lag)",
                    "Rate_Spread_diff_lag1": "US-India Interest Spread (1w diff lag)",
                    "Geo_Tension_lag1": "Geopolitical Tension (1w lag)",
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
                    marker_color='#2A9D8F'
                ))
                fig_coef.update_layout(
                    height=300,
                    template="plotly_dark",
                    xaxis_title="Coefficient Impact (paise)",
                    yaxis_title="",
                    plot_bgcolor="#0D1117",
                    paper_bgcolor="#0D1117",
                    font=dict(color="#F0F6FC"),
                    margin=dict(l=150, r=20, t=10, b=20)
                )
                st.plotly_chart(fig_coef, use_container_width=True)
            else:
                st.info("Lasso regularized all coefficients to zero.")
        except Exception as e:
            st.warning(f"Could not load Lasso coefficients: {e}")

        st.subheader("Trading Strategy Backtest: Cumulative Returns")
        st.caption("Performance of a simulated trading rule: Long USD/INR if predicted rate goes up, Short if it goes down.")
        
        # Consistent color mapping for backtest cumulative returns chart
        color_map = {
            "arima_lasso": "#D62828",   # Crimson Red (prominent)
            "lasso": "#2A9D8F",         # Teal Green
            "arima": "#F4A261",         # Warm Orange
            "arima_gb": "#2ecc71",      # Emerald Green
            "gb": "#e67e22",            # Orange
            "rf": "#9b59b6",            # Amethyst Purple
            "arimax": "#3498db",        # Sky Blue
            "naive": "#7f8c8d",         # Slate Gray (neutral baseline)
            "es": "#b2bec3"             # Light Gray (neutral baseline)
        }
        
        fig5 = go.Figure()
        # Render cumulative returns for all models
        for model_name, rets in cum_returns.items():
            color = color_map.get(model_name, None)
            width = 2.5 if model_name == "arima_lasso" else 1.5
            fig5.add_trace(go.Scatter(
                x=dates,
                y=[v * 100 for v in rets],
                mode="lines",
                name=f"{model_name.upper()} (Sharpe: {metrics.loc[model_name, 'Sharpe Ratio (Rf=0)']:.2f})",
                line=dict(color=color, width=width) if color else None
            ))
        fig5.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.8)
        fig5.update_layout(height=400, template="plotly_dark",
                           yaxis_title="Cumulative Return (%)", xaxis_title="Date",
                           plot_bgcolor="#0D1117",
                           paper_bgcolor="#0D1117",
                           font=dict(color="#F0F6FC"),
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig5, use_container_width=True)
        
        st.info("""
        **Quantitative Takeaways (200-Week Test Window with Hybrid Models)**:
        1. **ARIMA + Lasso Hybrid is the Top Performer**: The **ARIMA+Lasso Hybrid model dominates** the league table, achieving the highest cumulative return of **+18.07%** and a spectacular **Sharpe Ratio (Rf=0) of 1.33**. It achieves a **59.00% Mean Directional Accuracy (MDA)** and successfully beats the Naïve baseline (**Theil's U = 0.9966**). This hybrid model excels because it decomposes the prediction: ARIMA models the short-term linear time-series dynamics, while Lasso captures the macroeconomic exogenous relationship from the ARIMA residuals.
        2. **Lasso Standalone beats the Random Walk**: Standalone Lasso achieves the lowest overall RMSE (**0.3988**) and the lowest Theil's U (**0.9923**), with a **59.00% MDA** and a **1.02 Sharpe Ratio**, yielding **13.58% Cumulative Return**. Lasso succeeds because L1 regularization handles multicollinearity among macro drivers.
        3. **Risk-adjusted Carry Disclosures**: The standard Sharpe Ratio assumes a 0% risk-free rate (Information Ratio). If we adjust for India's **91-day T-Bill rate (~6.5% annualised)**, the carry hurdle turns the Sharpe ratios negative (ARIMA+Lasso Hybrid: **-0.65**). This highlights that systematic weekly trading of USD/INR is subject to a high interest rate carry hurdle in a central-bank-defended exchange rate regime.
        4. **ARIMA beats SARIMA**: The non-seasonal ARIMA(1,1,0) baseline outperforms the older seasonal SARIMA, achieving **57.00% MDA** and **12.85% return**. This confirms that weekly USD/INR changes are driven by short-term momentum rather than repeating annual weekly cycles.
        """)
        
    except Exception as e:
        st.warning(f"Run the forecasting notebook first. Error: {e}")
    
    st.divider()
    st.caption("Models: SARIMA, ARIMAX, Lasso, Random Forest, and Gradient Boosting Regressor trained on 1-week lagged exogenous features (Crude Diff, DXY Diff, Rate Spread Diff, Geo Tension Level)")

# ── FOOTER ─────────────────────────────────────────────
st.divider()
st.caption("RupeeRisk | Srishti Lamba | DAU M.Sc. Data Science | Data: RBI DBIE, yfinance, FRED API")