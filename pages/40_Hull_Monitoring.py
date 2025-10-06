# helix-analyser/pages/40_Hull_Monitoring.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date
import plotly.graph_objects as go
from plotly.colors import qualitative

# dummy data helper (sibling module)
from hull_dummy_data import get_hull_dummy_timeseries

st.set_page_config(page_title="Hull Monitoring", layout="wide")
st.title("Hull Monitoring")

# ---------- Load dummy data ----------
ts_df, ev_df = get_hull_dummy_timeseries()  # resistance time series + events

# ---------- Controls ----------
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    vessel = st.selectbox("Vessel", sorted(ts_df["Vessel"].unique().tolist()))
with c2:
    d_from = st.date_input("Date from", value=date(2025, 1, 1))
with c3:
    d_to = st.date_input("Date to", value=date(2025, 6, 30))

# filter
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

# ---------- Chart: Hull Resistance vs Time ----------
st.subheader("Hull Resistance vs Time")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ts_sel["Date"], y=ts_sel["ResistanceIndex"],
    mode="lines",
    name="Resistance index",
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Index: %{y:.3f}<extra></extra>",
))

# event vertical lines & labels
event_colors = {
    "Dry dock": qualitative.Set1[0],
    "Hull clean": qualitative.Set1[2],
    "Propeller polish": qualitative.Set1[1],
}

for _, r in ev_sel.iterrows():
    x = r["DateTime"]
    etype = r["Event"]
    color = event_colors.get(etype, "#444")

    # vertical line
    fig.add_vline(
        x=x,
        line_width=2,
        line_dash="dot",
        line_color=color,
        opacity=0.9,
    )
    # label near top
    fig.add_annotation(
        x=x, y=max(ts_sel["ResistanceIndex"].max(), 1.05),
        text=f"{etype}",
        showarrow=False,
        yanchor="bottom",
        xanchor="left",
        font=dict(size=11, color=color),
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor=color,
        borderwidth=1,
    )

# layout tweaks
fig.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Resistance Index (dimensionless)")

st.plotly_chart(fig, use_container_width=True)

# ---------- Events table ----------
st.subheader("Events")
if ev_sel.empty:
    st.info("No events in the selected window.")
else:
    table = ev_sel.copy()
    # split date/time columns for display
    table["Date"] = table["DateTime"].dt.date
    table["Time"] = table["DateTime"].dt.time
    table = table[["Date", "Time", "Location", "Event"]].reset_index(drop=True)
    st.dataframe(table, use_container_width=True)
