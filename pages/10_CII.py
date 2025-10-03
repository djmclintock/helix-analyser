# helix-analyser/pages/10_CII.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date
import plotly.express as px
import plotly.graph_objects as go

# Dummy data helpers (place this file next to your main app: helix-analyser/cii_dummy_data.py)
from cii_dummy_data import get_dummy_vessels, get_dummy_voyages


st.set_page_config(page_title="CII Planner & Actuals", layout="wide")
st.title("CII")

# ---------------------------
# Demo thresholds & colors (temporary until IMO refs wired)
# ---------------------------
def simple_rating(x: float) -> str:
    if x <= 13: return "A"
    if x <= 16: return "B"
    if x <= 19: return "C"
    if x <= 22: return "D"
    return "E"

RATING_COLORS = {
    "A": "#2ecc71",   # green
    "B": "#7fdb6a",   # light green
    "C": "#f1c40f",   # yellow
    "D": "#e67e22",   # orange
    "E": "#e74c3c",   # red
}

BANDS = [
    ("A", 0.0, 13.0,  "rgba(46, 204, 113, 0.10)"),
    ("B", 13.0, 16.0, "rgba(127, 219, 106, 0.10)"),
    ("C", 16.0, 19.0, "rgba(241, 196, 15,  0.10)"),
    ("D", 19.0, 22.0, "rgba(230, 126, 34,  0.10)"),
    ("E", 22.0, None, "rgba(231, 76,  60,  0.10)"),  # y1 filled later
]

# ---------------------------
# Load dummy data
# ---------------------------
vessels_df = get_dummy_vessels()   # Vessel, Type, DWT
voy_df     = get_dummy_voyages()   # Vessel, Voyage, DateFrom, DateTo, Distance_nm, HFO_t, ...

# Join DWT/type onto voyages so we can compute placeholder CII
df = voy_df.merge(vessels_df, on="Vessel", how="left")

# Compute placeholder Attained CII (gCO2 / DWT·nm)
df["CII_gCO2_DWTnm"] = (df["CO2_t"] * 1_000_000.0) / (df["Distance_nm"] * df["DWT"])
df["Rating"] = df["CII_gCO2_DWTnm"].apply(simple_rating)

# ---------------------------
# Fleet CII Bands view
# ---------------------------
st.subheader("CII Bands — Fleet view")
st.caption("Demo view with dummy data — real IMO bands will appear once reference tables are connected.")

fleet_point = df.groupby(["Vessel", "Type", "DWT"], as_index=False)["CII_gCO2_DWTnm"].mean()
fleet_point["Rating"] = fleet_point["CII_gCO2_DWTnm"].apply(simple_rating)

# Base scatter with labels
fig = px.scatter(
    fleet_point,
    x="DWT",
    y="CII_gCO2_DWTnm",
    color="Rating",
    color_discrete_map=RATING_COLORS,
    hover_data={"Vessel": True, "Type": True, "DWT": ":,", "CII_gCO2_DWTnm": ":.2f"},
    text="Vessel",
)
fig.update_traces(
    textposition="top center",
    marker=dict(size=10, line=dict(width=1, color="rgba(0,0,0,0.4)"))
)

# Background A–E band rectangles
x_min = max(0, float(fleet_point["DWT"].min() * 0.9)) if len(fleet_point) else 0
x_max = float(fleet_point["DWT"].max() * 1.05) if len(fleet_point) else 120_000
y_vals = fleet_point["CII_gCO2_DWTnm"] if len(fleet_point) else pd.Series([25.0])
y_max  = float(max(25.0, y_vals.max() * 1.2))
shapes, annotations = [], []
for name, y0, y1, rgba in BANDS:
    y1_eff = y_max if y1 is None else y1
    shapes.append(dict(
        type="rect", xref="x", yref="y",
        x0=x_min, x1=x_max, y0=y0, y1=y1_eff,
        fillcolor=rgba, layer="below", line=dict(width=0),
    ))
    annotations.append(dict(
        x=x_max, y=(y0 + y1_eff) / 2, xref="x", yref="y",
        text=name, showarrow=False, xanchor="right",
        font=dict(size=12, color="rgba(0,0,0,0.55)")
    ))

fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    legend_title_text="Rating",
    margin=dict(l=10, r=10, t=40, b=10),
)
fig.update_xaxes(title_text="DWT")
fig.update_yaxes(title_text="CII (gCO₂ / DWT·nm)")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Filters row: Vessel • Date Range • Voyage
# ---------------------------
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    vessel_opt = ["All"] + vessels_df["Vessel"].tolist()
    vessel_sel = st.selectbox("Vessel name", vessel_opt)
with c2:
    date_from = st.date_input("Date range — From", value=date(2025, 1, 1))
    date_to   = st.date_input("Date range — To",   value=date(2025,12,31))
with c3:
    voy_opt    = ["All"] + df["Voyage"].unique().tolist()
    voyage_sel = st.selectbox("Voyage number", voy_opt)

# Apply filters
filtered = df.copy()
if vessel_sel != "All":
    filtered = filtered[filtered["Vessel"] == vessel_sel]
if voyage_sel != "All":
    filtered = filtered[filtered["Voyage"] == voyage_sel]
if date_from and date_to:
    filtered = filtered[(filtered["DateFrom"] >= date_from) & (filtered["DateTo"] <= date_to)]

# ---------------------------
# CII table
# ---------------------------
st.subheader("CII table")
table_cols = [
    "Vessel","Type","Voyage","DateFrom","DateTo","Distance_nm",
    "HFO_t","MDO_t","LNG_t","CO2_t","DWT","CII_gCO2_DWTnm","Rating"
]
st.dataframe(filtered[table_cols], use_container_width=True)

# ---------------------------
# Planner (demo)
# ---------------------------
st.subheader("Planner")
st.caption("This mirrors the Excel planner at a high level using demo vessels or manual entry.")

mode = st.radio("Planner mode", ["Demo vessel", "Manual entry"], horizontal=True)

if mode == "Demo vessel":
    v = st.selectbox("Select demo vessel", vessels_df["Vessel"].tolist())
    row = vessels_df[vessels_df["Vessel"] == v].iloc[0]
    dwt = int(row["DWT"])
    st.write(f"**{v}** — {row['Type']}  •  DWT {dwt:,}")
else:
    colA, colB = st.columns(2)
    with colA:
        v = st.text_input("Vessel name", "Custom Vessel")
        t = st.selectbox("Type", ["LNG Carrier","Container","Gas Carrier","Bulk Carrier","Tanker"])
    with colB:
        dwt = st.number_input("Deadweight (DWT)", min_value=1, value=100_000, step=100)

# Planner inputs (distance + fuels)
col1, col2, col3, col4 = st.columns(4)
with col1:
    dist = st.number_input("Planned distance (nm)", min_value=1.0, value=12_000.0, step=100.0)
with col2:
    hfo = st.number_input("HFO (t)", min_value=0.0, value=500.0, step=10.0)
with col3:
    mdo = st.number_input("MDO (t)", min_value=0.0, value=50.0, step=5.0)
with col4:
    lng = st.number_input("LNG (t)", min_value=0.0, value=0.0, step=5.0)

# Simple CFs to compute demo CO2 + CII
CF = {"HFO": 3.114, "MDO": 3.206, "LNG": 2.750}
co2_t = hfo*CF["HFO"] + mdo*CF["MDO"] + lng*CF["LNG"]
attained = (co2_t * 1_000_000.0) / (dist * dwt)

# Rating + colored badge
rating = simple_rating(attained)
badge = f"""
<span style="
  display:inline-block;
  padding:4px 12px;
  border-radius:999px;
  background:{RATING_COLORS.get(rating,'#ccc')};
  color:white; font-weight:600;">
  {rating}
</span>
"""

colA, colB, colC, colD = st.columns(4)
colA.metric("Attained CII (g/DWT·nm)", f"{attained:.3f}")
colB.metric("Planned CO₂ (t)", f"{co2_t:,.1f}")
colC.metric("DWT", f"{dwt:,}")
with colD:
    st.markdown("**Rating**")
    st.markdown(badge, unsafe_allow_html=True)

st.caption("Rating uses temporary demo thresholds (A≤13, B≤16, C≤19, D≤22, else E). When ready, we’ll switch to IMO bands via `helix_cii`.")
