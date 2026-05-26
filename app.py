import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="RupeeRisk — India Forex Intelligence",
    page_icon="📉",
    layout="wide"
)

# ── LOAD DATA ──────────────────────────────────────────
@st.cache_data
def load_data():
    df      = pd.read_csv("data/processed/master_df.csv",        index_col=0, parse_dates=True)
    events  = pd.read_csv("data/raw/geopolitical_events.csv",    parse_dates=["date"])
    metrics = pd.read_csv("data/processed/model_metrics.csv",    index_col=0)
    return df, events, metrics

df, events, metrics = load_data()

# ── HEADER ─────────────────────────────────────────────
st.markdown("""
<h1 style='color:#1A3A5C;margin-bottom:0'>📉 RupeeRisk</h1>
<p style='color:#555;font-size:16px;margin-top:4px'>
India Macro & Geopolitical Forex Intelligence Platform — 
INR/USD analysis with war risk as an engineered feature
</p>
""", unsafe_allow_html=True)

st.divider()

# ── KEY METRICS BAR ────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
current_inr = df["USDINR"].dropna().iloc[-1]
inr_ytd_start = df["USDINR"].dropna()["2026-01-01":].iloc[0]
ytd_change = ((current_inr - inr_ytd_start) / inr_ytd_start) * 100

col1.metric("Current USD/INR", f"₹ {current_inr:.2f}")
col2.metric("YTD Change", f"{ytd_change:+.2f}%", 
            delta=f"{ytd_change:+.2f}%", delta_color="inverse")
col3.metric("Best Model (Lasso) MAPE", f"{metrics.loc['lasso','MAPE (%)']:.2f}%")
col4.metric("Geopolitical Events Analysed", f"{len(events)}")

st.divider()

# ── TABS ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Macro Dashboard", "⚔️ Geopolitical Risk", "🔮 Forecasting"])

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
    
    fig3.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig3.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1,
                   annotation_text="Event Day", annotation_position="top right")
    fig3.update_layout(
        title=f"Event Study: {selected_event}",
        xaxis_title="Days from Event", yaxis_title="Cumulative Return (%)",
        template="plotly_white", height=420,
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
    - **Validation**: Evaluated using a **Rolling 1-Step-Ahead Validation** loop (52-week test set), re-estimating models weekly and anchoring on the previous week's level ($y_{t-1}$).
    """)
    
    try:
        with open("data/processed/predictions.json", "r") as f:
            preds_data = json.load(f)
            
        dates = pd.to_datetime(preds_data["dates"])
        actual = preds_data["actual"]
        predictions = preds_data["predictions"]
        cum_returns = preds_data["cum_returns"]
        
        st.subheader("🏆 Model Performance League Table")
        st.caption("Models ranked by out-of-sample performance over the last 52 weeks.")
        # Renders the nice league table
        st.dataframe(metrics.style.format({
            "MAPE (%)": "{:.3f}%",
            "RMSE": "{:.4f}",
            "MDA (%)": "{:.2f}%",
            "Sharpe Ratio": "{:.2f}",
            "Cumulative Return (%)": "{:+.2f}%"
        }), use_container_width=True)
        
        st.subheader("📈 Out-of-Sample Predictions (Last 52 Weeks)")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=dates, y=actual,
            mode="lines+markers", name="Actual USD/INR", line=dict(color="#1A3A5C", width=2.5)))
        
        # Plot top 3 models: Lasso, Gradient Boosting (GB), and ARIMAX baseline
        fig4.add_trace(go.Scatter(x=dates, y=predictions["lasso"],
            mode="lines", name=f"Lasso (MAPE: {metrics.loc['lasso','MAPE (%)']:.2f}%)",
            line=dict(color="#2A9D8F", width=2)))
        fig4.add_trace(go.Scatter(x=dates, y=predictions["gb"],
            mode="lines", name=f"Gradient Boosting (MAPE: {metrics.loc['gb','MAPE (%)']:.2f}%)",
            line=dict(color="#D62828", width=2, dash="dash")))
        fig4.add_trace(go.Scatter(x=dates, y=predictions["sarima"],
            mode="lines", name=f"SARIMA Baseline (MAPE: {metrics.loc['sarima','MAPE (%)']:.2f}%)",
            line=dict(color="#F4A261", width=1.5, dash="dot")))
            
        fig4.update_layout(height=400, template="plotly_white",
                           yaxis_title="USD/INR Level", xaxis_title="Date",
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig4, use_container_width=True)
        
        st.subheader("💰 Trading Strategy Backtest: Cumulative Returns")
        st.caption("Performance of a simulated trading rule: Long USD/INR if predicted rate goes up, Short if it goes down.")
        
        fig5 = go.Figure()
        for model_name, rets in cum_returns.items():
            fig5.add_trace(go.Scatter(
                x=dates,
                y=[v * 100 for v in rets],
                mode="lines",
                name=f"{model_name.upper()} (Sharpe: {metrics.loc[model_name, 'Sharpe Ratio']:.2f})"
            ))
        fig5.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
        fig5.update_layout(height=400, template="plotly_white",
                           yaxis_title="Cumulative Return (%)", xaxis_title="Date",
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig5, use_container_width=True)
        
        st.info("""
        **Quantitative Takeaways**:
        1. **Lasso is the Top Model**: Lasso achieves the lowest **MAPE (0.46%)** and a spectacular **Mean Directional Accuracy (73.08%)**, yielding a **Sharpe Ratio of 2.68** and **11.70% Cumulative Return**. Lasso wins because L1 regularization prevents overfitting and resolves multicollinearity among correlated macro drivers.
        2. **Non-Linear Boosting**: **Gradient Boosting (Sharpe: 2.36, Return: 10.40%)** performs exceptionally well, capturing non-linear relationships during market shocks, while **Random Forest** overfits and struggles on test data.
        3. **ARIMAX beats SARIMA**: By differencing and lagging variables correctly, ARIMAX beats SARIMA (MDA **59.62%** vs **55.77%**), proving that macroeconomic indicators (oil, rates, DXY) do carry predictive signals when look-ahead bias is eliminated!
        """)
        
    except Exception as e:
        st.warning(f"Run the forecasting notebook first. Error: {e}")
    
    st.divider()
    st.caption("Models: SARIMA, ARIMAX, Lasso, Random Forest, and Gradient Boosting Regressor trained on 1-week lagged exogenous features (Crude Diff, DXY Diff, Rate Spread Diff, Geo Tension Level)")

# ── FOOTER ─────────────────────────────────────────────
st.divider()
st.caption("RupeeRisk | Srishti Lamba | DAU M.Sc. Data Science | Data: RBI DBIE, yfinance, FRED API")