# helix-analyser/pages/10_CII.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datetime import date
from cii_dummy_data import get_dummy_vessels, get_dummy_voyages

st.set_page_config(page_title="CII Planner & Actuals", layout="wide")
st.title("CII")

# ---------------------------
# Load dummy data
# ---------------------------
vessels_df = get_dummy_vessels()   # Vessel, Type, DWT
voy_df     = get_dummy_voyages()   # Vessel, Voyage, DateFrom, DateTo, Distance_nm, HFO_t, ...

# Join DWT/type onto voyages so we can compute placeholder CII
df = voy_df.merge(vessels_df, on="Vessel", how="left")

# Compute placeholder Attained CII (gCO2 / DWT·nm)
# NOTE: purely illustrative; replace later with helix_cii.compute_cii(...) for accuracy.
df["CII_gCO2_DWTnm"] = (df["CO2_t"] * 1_000_000.0) / (df["Distance_nm"] * df["DWT"])

# Rough demo rating buckets (replace with real thresholds later)
def simple_rating(x: float) -> str:
    if x <= 13: return "A"
    if x <= 16: return "B"
    if x <= 19: return "C"
    if x <= 22: return "D"
    return "E"
df["Rating"] = df["CII_gCO2_DWTnm"].apply(simple_rating)

# ---------------------------
# Fleet CII Bands view (with colored A–E bands + labels)
# ---------------------------
st.subheader("CII Bands — Fleet view")
st.caption("Demo view with dummy data — bands use the page's temporary thresholds (A≤13, B≤16, C≤19, D≤22, else E).")

# Average per vessel point
fleet_point = df.groupby(["Vessel","Type","DWT"], as_index=False)["CII_gCO2_DWTnm"].mean()

# Ensure ratings exist for coloring
def simple_rating(x: float) -> str:
    if x <= 13: return "A"
    if x <= 16: return "B"
    if x <= 19: return "C"
    if x <= 22: return "D"
    return "E"
fleet_point["Rating"] = fleet_point["CII_gCO2_DWTnm"].apply(simple_rating)

# Colors per rating (you can tweak)
color_map = {
    "A": "#2ecc71",   # green
    "B": "#7fdb6a",   # light green
    "C": "#f1c40f",   # yellow
    "D": "#e67e22",   # orange
    "E": "#e74c3c",   # red
}

# Build base scatter with labels
fig = px.scatter(
    fleet_point,
    x="DWT",
    y="CII_gCO2_DWTnm",
    color="Rating",
    color_discrete_map=color_map,
    hover_data={"Vessel": True, "Type": True, "DWT": ":,", "CII_gCO2_DWTnm": ":.2f"},
    text="Vessel",  # name tags beside dots
)

# Nudge labels and style
fig.update_traces(textposition="top center", marker=dict(size=10, line=dict(width=1, color="rgba(0,0,0,0.4)")))

# Compute band rectangles (y0/y1) and x-span
x_min = max(0, float(fleet_point["DWT"].min()*0.9))
x_max = float(fleet_point["DWT"].max()*1.05)
y_vals = fleet_point["CII_gCO2_DWTnm"]
y_max = float(y_vals.max()*1.2 if len(y_vals) else 25.0)

bands = [
    ("A", 0.0, 13.0,  "rgba(46, 204, 113, 0.10)"),
    ("B", 13.0, 16.0, "rgba(127, 219, 106, 0.10)"),
    ("C", 16.0, 19.0, "rgba(241, 196, 15,  0.10)"),
    ("D", 19.0, 22.0, "rgba(230, 126, 34,  0.10)"),
    ("E", 22.0, y_max,"rgba(231, 76,  60,  0.10)"),
]

shapes = []
annotations = []
for band_name, y0, y1, rgba in bands:
    shapes.append(dict(
        type="rect", xref="x", yref="y",
        x0=x_min, x1=x_max, y0=y0, y1=y1,
        fillcolor=rgba, layer="below", line=dict(width=0),
    ))
    annotations.append(dict(
        x=x_max, y=(y0+y1)/2, xref="x", yref="y",
        text=band_name, showarrow=False, xanchor="right",
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
c1, c2, c3 = st.columns([1,1,1])
with c1:
    vessel_opt = ["All"] + vessels_df["Vessel"].tolist()
    vessel_sel = st.selectbox("Vessel name", vessel_opt)
with c2:
    # date filter is illustrative for now
    date_from = st.date_input("Date range — From", value=date(2025,1,1))
    date_to   = st.date_input("Date range — To",   value=date(2025,12,31))
with c3:
    voy_opt = ["All"] + df["Voyage"].unique().tolist()
    voyage_sel = st.selectbox("Voyage number", voy_opt)

# Apply filters (vessel + voyage; date range only filters if both set)
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

# Picker for a demo vessel OR manual entry
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

colA, colB, colC = st.columns(3)
colA.metric("Attained CII (g/DWT·nm)", f"{attained:.3f}")
colB.metric("Planned CO₂ (t)", f"{co2_t:,.1f}")
colC.metric("DWT", f"{dwt:,}")

st.success("Demo planner uses placeholder CFs and does not reference IMO thresholds yet. When ready, we can swap this to the `helix_cii` module and show A–E bands.")
