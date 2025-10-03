# helix-analyser/pages/10_CII.py
import streamlit as st
import pandas as pd
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
# Fleet CII "Bands" view (placeholder)
# ---------------------------
st.subheader("CII Bands — Fleet view")
# For now: scatter of CII by vessel (we'll overlay A–E bands when reference sheet is wired)
fleet_point = df.groupby(["Vessel","Type","DWT"], as_index=False)["CII_gCO2_DWTnm"].mean()
st.caption("Demo view with dummy data — real IMO bands will appear once reference tables are connected.")
st.scatter_chart(fleet_point, x="DWT", y="CII_gCO2_DWTnm")

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
