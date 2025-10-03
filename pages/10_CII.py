import streamlit as st
import pandas as pd
from helix_analyser.cii_dummy_data import get_dummy_cii_data

st.set_page_config(page_title="CII", layout="wide")
st.title("Carbon Intensity Indicator (CII)")

# --- Dummy fleet data
df = get_dummy_cii_data()

# --- Fleet CII Bands view (placeholder chart)
st.subheader("Fleet CII Bands")
st.line_chart(df.set_index("Vessel")["CII_gCO2_DWTnm"])

# --- Filters
col1, col2, col3 = st.columns(3)
with col1:
    vessel = st.selectbox("Vessel", ["All"] + df["Vessel"].unique().tolist())
with col2:
    start_date = st.date_input("From date")
with col3:
    voyage = st.selectbox("Voyage number", ["All"] + df["Voyage"].unique().tolist())

# Apply filters
filtered = df.copy()
if vessel != "All":
    filtered = filtered[filtered["Vessel"] == vessel]
if voyage != "All":
    filtered = filtered[filtered["Voyage"] == voyage]

# --- CII table
st.subheader("CII Table")
st.dataframe(filtered, use_container_width=True)

# --- Planner
st.subheader("CII Planner (Demo)")
demo_vessels = df["Vessel"].unique().tolist()
sel_vessel = st.selectbox("Select demo vessel", demo_vessels)

sel = df[df["Vessel"] == sel_vessel].iloc[0]
st.write(f"Planner for **{sel['Vessel']}** ({sel['Type']})")
st.metric("Distance (nm)", sel["Distance_nm"])
st.metric("Fuel (t)", sel["Fuel_t"])
st.metric("Attained CII", sel["CII_gCO2_DWTnm"])
st.metric("Rating", sel["Rating"])
