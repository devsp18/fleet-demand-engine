"""
app.py  -  Fleet Demand Forecasting & Allocation Engine
--------------------------------------------------------
Streamlit front end tying together the forecast, seasonal signals,
allocation, and auto-briefing. Dark, restrained theme; the seasonal
signal strip is the signature element.

Run:  streamlit run app.py
"""

import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from forecasting import run_all
from allocation import seasonal_signals, proportional_allocation, what_if
from report import generate_briefing

# ----------------------------------------------------------------------
st.set_page_config(page_title="Fleet Demand Engine", layout="wide",
                   initial_sidebar_state="expanded")

# --- palette: deep slate base, one warm signal hue ---
INK = "#0E1117"; PANEL = "#161A23"; LINE = "#262C3A"
TEXT = "#E6E9EF"; MUTE = "#8A93A6"
WARM = "#E8853A"   # ramping / signal
COOL = "#4C7FB8"   # cooling
ACCENT = "#D8B26A" # gold hairline accent

st.markdown(f"""
<style>
.stApp {{ background:{INK}; color:{TEXT}; }}
section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {LINE}; }}
h1,h2,h3,h4 {{ color:{TEXT}; font-family:'Georgia',serif; letter-spacing:.2px; }}
.metric-card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:10px;
  padding:18px 20px; }}
.metric-card .label {{ color:{MUTE}; font-size:12px; text-transform:uppercase;
  letter-spacing:1.5px; }}
.metric-card .value {{ color:{TEXT}; font-size:30px; font-weight:600;
  font-family:'Georgia',serif; }}
.metric-card .sub {{ color:{ACCENT}; font-size:12px; }}
.sig {{ display:inline-block; padding:4px 10px; border-radius:999px;
  font-size:12px; font-weight:600; letter-spacing:.5px; }}
.eyebrow {{ color:{ACCENT}; font-size:12px; text-transform:uppercase;
  letter-spacing:3px; }}
.briefing {{ background:{PANEL}; border:1px solid {LINE}; border-left:3px solid {WARM};
  border-radius:8px; padding:18px 22px; font-family:'SF Mono','Menlo',monospace;
  font-size:13px; line-height:1.7; color:{TEXT}; white-space:pre-wrap; }}
hr {{ border-color:{LINE}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    out = run_all()
    return out

out = load()
panel, fc, metrics = out["panel"], out["forecast"], out["metrics"]
imp = out["importance"]; test_pred = out["test_pred"]

# ----------------------------------------------------------------------
# Sidebar controls
st.sidebar.markdown(f"<div class='eyebrow'>Controls</div>", unsafe_allow_html=True)
st.sidebar.markdown("### Fleet planning")
fleet_size = st.sidebar.slider("Demo fleet size", 60, 200, 120, step=5)
min_per = st.sidebar.slider("Minimum vehicles per market", 3, 15, 8, step=1)
st.sidebar.markdown("---")
st.sidebar.caption("Data: real dealership sales, 7 regions, Jan 2022 - Dec 2023 "
                   "(Kaggle). Sales aggregated to a weekly regional panel. "
                   "Demand = realized weekly units; no synthetic conversion values.")

sig = seasonal_signals(panel, fc)
alloc, _ = proportional_allocation(fc, fleet_size=fleet_size, min_per_region=min_per)

# ----------------------------------------------------------------------
# Header
st.markdown(f"<div class='eyebrow'>Marketing Fleet Operations</div>",
            unsafe_allow_html=True)
st.markdown("# Fleet Demand Forecasting & Allocation Engine")
st.markdown(f"<span style='color:{MUTE}'>Forecasts next-week demand per market, "
            f"flags seasonal momentum, and recommends how to distribute a fixed "
            f"demo fleet.</span>", unsafe_allow_html=True)
st.write("")

# KPI row
c1, c2, c3, c4 = st.columns(4)
total_fc = fc["forecast_units"].sum()
ramp_n = int((sig["signal"] == "RAMPING UP").sum())
with c1:
    st.markdown(f"<div class='metric-card'><div class='label'>Forecast week</div>"
                f"<div class='value'>{pd.Timestamp(fc['week'].iloc[0]).date()}</div>"
                f"<div class='sub'>7 markets</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><div class='label'>Projected demand</div>"
                f"<div class='value'>{total_fc:.0f}</div>"
                f"<div class='sub'>units next week</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><div class='label'>Model error (MAE)</div>"
                f"<div class='value'>{metrics['MAE']}</div>"
                f"<div class='sub'>{metrics['improvement_vs_naive_pct']}% better than naive</div></div>",
                unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='metric-card'><div class='label'>Markets ramping</div>"
                f"<div class='value'>{ramp_n}</div>"
                f"<div class='sub'>vs same week last year</div></div>",
                unsafe_allow_html=True)

st.write(""); st.markdown("---")

# ----------------------------------------------------------------------
# Signature: seasonal signal strip
st.markdown(f"<div class='eyebrow'>Seasonal momentum</div>", unsafe_allow_html=True)
st.markdown("### Which markets are entering strength?")
st.caption("Forecast vs the same week one year ago. Demand swings ~4.5x across the "
           "year, so year-over-year is the honest baseline for momentum.")

fig_sig = go.Figure()
s = sig.sort_values("yoy_pct")
colors = [WARM if v > 8 else COOL if v < -8 else MUTE for v in s["yoy_pct"]]
fig_sig.add_trace(go.Bar(
    x=s["yoy_pct"], y=s["Dealer_Region"], orientation="h",
    marker_color=colors,
    text=[f"{v:+.0f}%" for v in s["yoy_pct"]], textposition="outside",
    hovertemplate="%{y}: %{x:+.1f}% YoY<extra></extra>"))
fig_sig.update_layout(
    height=300, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font_color=TEXT, margin=dict(l=10, r=30, t=10, b=10),
    xaxis=dict(title="Forecast vs same week last year (%)", gridcolor=LINE, zerolinecolor=MUTE),
    yaxis=dict(gridcolor=LINE))
st.plotly_chart(fig_sig, use_container_width=True)

# ----------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.markdown(f"<div class='eyebrow'>Recommendation</div>", unsafe_allow_html=True)
    st.markdown("### Fleet allocation")
    fig_a = go.Figure()
    fig_a.add_trace(go.Bar(name="Recommended", x=alloc["Dealer_Region"],
                           y=alloc["recommended_vehicles"], marker_color=WARM))
    fig_a.add_trace(go.Bar(name="Even split", x=alloc["Dealer_Region"],
                           y=alloc["even_split"], marker_color=LINE))
    fig_a.update_layout(barmode="group", height=340, paper_bgcolor=PANEL,
                        plot_bgcolor=PANEL, font_color=TEXT,
                        legend=dict(orientation="h", y=1.1),
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(gridcolor=LINE), yaxis=dict(title="vehicles", gridcolor=LINE))
    st.plotly_chart(fig_a, use_container_width=True)
    st.dataframe(
        alloc[["Dealer_Region", "forecast_units", "forecast_share",
               "recommended_vehicles", "vs_even"]]
        .rename(columns={"Dealer_Region": "Market", "forecast_units": "Forecast",
                         "forecast_share": "Share %", "recommended_vehicles": "Vehicles",
                         "vs_even": "vs Even"}),
        use_container_width=True, hide_index=True)

with right:
    st.markdown(f"<div class='eyebrow'>Auto-generated</div>", unsafe_allow_html=True)
    st.markdown("### Weekly briefing")
    briefing = generate_briefing(metrics, sig, alloc, fleet_size)
    st.markdown(f"<div class='briefing'>{briefing}</div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------------------------
# Model diagnostics (the honesty section)
st.markdown(f"<div class='eyebrow'>Under the hood</div>", unsafe_allow_html=True)
st.markdown("### Model diagnostics")
d1, d2 = st.columns(2)

with d1:
    st.markdown("**Holdout validation** (time-based split, no leakage)")
    mdf = pd.DataFrame({
        "Metric": ["Test period start", "Train weeks", "Test weeks", "MAE (units)",
                   "RMSE", "MAPE %", "Naive MAE", "Improvement vs naive"],
        "Value": [metrics["test_start"], metrics["n_train"], metrics["n_test"],
                  metrics["MAE"], metrics["RMSE"], f"{metrics['MAPE_pct']}%",
                  metrics["naive_MAE"], f"{metrics['improvement_vs_naive_pct']}%"]})
    st.dataframe(mdf, use_container_width=True, hide_index=True)
    st.caption("Trained on the earliest ~80% of weeks, tested on the most recent "
               "~20%, the way a forecast is actually used. The model is held to a "
               "naive last-week baseline so it has to earn its accuracy.")

with d2:
    st.markdown("**What drives the forecast**")
    fi = imp.head(8).sort_values("importance")
    fig_i = go.Figure(go.Bar(x=fi["importance"], y=fi["feature"], orientation="h",
                             marker_color=ACCENT))
    fig_i.update_layout(height=300, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                        font_color=TEXT, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(gridcolor=LINE), yaxis=dict(gridcolor=LINE))
    st.plotly_chart(fig_i, use_container_width=True)
    st.caption("Seasonality (month, week-of-year) and recent demand (lags) lead, "
               "which matches the visible yearly cycle in the data.")

# Forecast vs actual on holdout
st.markdown("**Forecast vs actual** (held-out weeks, all markets)")
tp = test_pred.sort_values("week")
agg = tp.groupby("week").agg(actual=("units_sold", "sum"),
                             predicted=("pred", "sum")).reset_index()
fig_f = go.Figure()
fig_f.add_trace(go.Scatter(x=agg["week"], y=agg["actual"], name="Actual",
                           line=dict(color=TEXT, width=2)))
fig_f.add_trace(go.Scatter(x=agg["week"], y=agg["predicted"], name="Predicted",
                           line=dict(color=WARM, width=2, dash="dash")))
fig_f.update_layout(height=320, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                    font_color=TEXT, legend=dict(orientation="h", y=1.1),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(gridcolor=LINE), yaxis=dict(title="units", gridcolor=LINE))
st.plotly_chart(fig_f, use_container_width=True)

# What-if
st.markdown("---")
st.markdown(f"<div class='eyebrow'>Scenario</div>", unsafe_allow_html=True)
st.markdown("### What if the fleet were a different size?")
wi = what_if(fc, [80, 100, 120, 140, 160], min_per_region=min_per)
st.dataframe(wi.rename(columns={"fleet_size": "Fleet size",
                                "top_region": "Top market",
                                "top_region_vehicles": "Vehicles to top market"}),
             use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built by Satyam Patel. Data: Kaggle car sales (real, 2022-2023), "
           "reshaped into a weekly regional fleet panel. Forecast: XGBoost with "
           "time-based validation. Allocation: proportional-to-forecast with a "
           "per-market floor. Demand levels are similar across markets, so the "
           "engine's value is in seasonal timing, stated plainly rather than overclaimed.")
