# helix-analyser/pages/40_Hull_Monitoring.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date
import plotly.graph_objects as go
from plotly.colors import qualitative

from hull_dummy_data import get_hull_dummy_timeseries

st.set_page_config(page_title="Hull Monitoring", layout="wide")
st.title("Hull Monitoring")

# ---------- Load dummy data ----------
ts_df, ev_df = get_hull_dummy_timeseries()

# ---------- Controls ----------
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    vessel = st.selectbox("Vessel", sorted(ts_df["Vessel"].unique().tolist()))
with c2:
    d_from = st.date_input("Date from", value=date(2025, 1, 1))
with c3:
    d_to = st.date_input("Date to", value=date(2025, 6, 30))

# Filter
mask = (
    (ts_df["Vessel"] == vessel) &
    (ts_df["Date"] >= pd.to_datetime(d_from)) &
    (ts_df["Date"] <= pd.to_datetime(d_to))
)
ts_sel = ts_df.loc[mask].copy()
ev_sel = ev_df[
    (ev_df["Vessel"] == vessel) &
    (ev_df["DateTime"] >= pd.to_datetime(d_from)) &
    (ev_df["DateTime"] <= pd.to_datetime(d_to))
].copy()

# ---------- Convert to % fouling ----------
baseline = ts_sel["ResistanceIndex"].iloc[0]
ts_sel["Fouling_%"] = (ts_sel["ResistanceIndex"] / baseline - 1.0) * 100.0

# 7-day rolling average (centered on latest)
ts_sel["Fouling_%_roll7"] = ts_sel["Fouling_%"].rolling(7, min_periods=1).mean()

threshold = 15.0  # %
over_limit = ts_sel[ts_sel["Fouling_%_roll7"] > threshold]

if not over_limit.empty:
    st.warning(
        f"⚠️ Rolling fouling exceeds {threshold:.0f}% on {len(over_limit)} day(s). "
        "Inspection or hull cleaning recommended."
    )

# ---------- Chart ----------
st.subheader("Hull Resistance vs Time")

fig = go.Figure()

# Raw series (light grey)
fig.add_trace(go.Scatter(
    x=ts_sel["Date"], y=ts_sel["Fouling_%"],
    mode="lines",
    name="Fouling % (raw)",
    line=dict(color="rgba(120,120,120,0.35)", width=2),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Raw: %{y:.1f}%<extra></extra>",
))

# Rolling avg colored by threshold
roll = ts_sel["Fouling_%_roll7"]
below = roll.where(roll <= threshold)
above = roll.where(roll > threshold)

fig.add_trace(go.Scatter(
    x=ts_sel["Date"], y=below,
    mode="lines",
    name="Fouling % (7d avg)",
    line=dict(color="#2ecc71", width=3),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>7d avg: %{y:.1f}%<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=ts_sel["Date"], y=above,
    mode="lines",
    name="> 15% (7d avg)",
    line=dict(color="#e74c3c", width=4),
    hovertemplate="Date: %{x|%Y-%m-%d}<br>7d avg: %{y:.1f}%<extra></extra>",
))

# Threshold line
fig.add_hline(
    y=threshold,
    line_width=2, line_dash="dash", line_color="#e67e22",
    annotation_text="Inspection/Clean threshold (+15%)",
    annotation_position="top left",
    annotation_font_color="#e67e22",
)

# Events
event_colors = {
    "Dry dock": qualitative.Set1[0],
    "Hull clean": qualitative.Set1[2],
    "Propeller polish": qualitative.Set1[1],
}
for _, r in ev_sel.iterrows():
    x = r["DateTime"]
    etype = r["Event"]
    color = event_colors.get(etype, "#444")

    fig.add_vline(x=x, line_width=2, line_dash="dot", line_color=color, opacity=0.9)
    fig.add_annotation(
        x=x, y=max(ts_sel["Fouling_%_roll7"].max(), threshold + 5),
        text=f"{etype}",
        showarrow=False, yanchor="bottom", xanchor="left",
        font=dict(size=11, color=color),
        bgcolor="rgba(255,255,255,0.6)", bordercolor=color, borderwidth=1,
    )

# Layout
fig.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Hull resistance increase (%)")

st.plotly_chart(fig, use_container_width=True)

# ---------- Events table ----------
st.subheader("Events")
if ev_sel.empty:
    st.info("No events in the selected window.")
else:
    table = ev_sel.copy()
    table["Date"] = table["DateTime"].dt.date
    table["Time"] = table["DateTime"].dt.time
    table = table[["Date", "Time", "Location", "Event"]].reset_index(drop=True)
    st.dataframe(table, use_container_width=True)
