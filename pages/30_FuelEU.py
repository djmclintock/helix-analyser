# helix-analyser/pages/30_FuelEU.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date

from fueleu_dummy_data import load_dummy_fueleu_legs, compute_leg_energy_emissions

st.set_page_config(page_title="FuelEU Maritime — Summary", layout="wide")
st.title("FuelEU Maritime — Summary")

# -------------------------------------------------------
# Load dummy
# -------------------------------------------------------
legs = load_dummy_fueleu_legs()
legs = compute_leg_energy_emissions(legs)  # adds Energy_GJ, CO2e_t, Intensity_g_per_MJ

# -------------------------------------------------------
# Controls
# -------------------------------------------------------
c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.2])
with c1:
    vsel = st.selectbox("Vessel", ["All"] + sorted(legs["Vessel"].unique().tolist()))
with c2:
    d_from = st.date_input("From", value=date(2025,1,1))
with c3:
    d_to   = st.date_input("To",   value=date(2025,12,31))
with c4:
    pooling = st.toggle("Pool vessels", value=True, help="When ON, vessels are pooled so surpluses offset deficits.")

c5, c6 = st.columns([1.2, 1.2])
with c5:
    # Target GHG intensity (gCO2e/MJ). Leave editable.
    target = st.number_input("Target intensity (gCO₂e/MJ)", min_value=0.0, value=75.0, step=0.5)
with c6:
    # Price per tCO2e to estimate cost of covering deficits
    credit_price = st.number_input("Credit price (€/tCO₂e)", min_value=0.0, value=250.0, step=5.0,
                                   help="Edit to your assumed cost for purchasing credits/penalties.")

# -------------------------------------------------------
# Filter
# -------------------------------------------------------
mask = (legs["Start"].dt.date >= d_from) & (legs["End"].dt.date <= d_to)
if vsel != "All":
    mask &= legs["Vessel"].eq(vsel)
tbl = legs.loc[mask].copy()

# -------------------------------------------------------
# Vessel-level aggregation
# -------------------------------------------------------
group = ["Vessel"]
agg = tbl.groupby(group, as_index=False).agg(
    Energy_GJ=("Energy_GJ","sum"),
    CO2e_t=("CO2e_t","sum")
)
agg["Intensity_g_per_MJ"] = (agg["CO2e_t"] * 1000.0) / agg["Energy_GJ"]
agg["Delta_g_per_MJ"]     = agg["Intensity_g_per_MJ"] - target
# Convert intensity gap into tCO2e equivalent based on ship's energy: gap(g/MJ)*MJ/1e6
agg["Energy_MJ"]          = agg["Energy_GJ"] * 1000.0
agg["Deficit_t"]          = (agg["Delta_g_per_MJ"].clip(lower=0) * agg["Energy_MJ"]) / 1_000_000.0
agg["Surplus_t"]          = ((-agg["Delta_g_per_MJ"]).clip(lower=0) * agg["Energy_MJ"]) / 1_000_000.0

# -------------------------------------------------------
# Pooling logic (now computes a signed net)
# -------------------------------------------------------
if pooling:
    pool_E_GJ = float(agg["Energy_GJ"].sum())
    pool_CO2e = float(agg["CO2e_t"].sum())
    pooled_int = (pool_CO2e * 1000.0) / pool_E_GJ if pool_E_GJ > 0 else 0.0
    pool_delta = pooled_int - target
    pool_MJ = pool_E_GJ * 1000.0
    pool_deficit_t = max(0.0, pool_delta) * pool_MJ / 1_000_000.0
    pool_surplus_t = max(0.0, -pool_delta) * pool_MJ / 1_000_000.0
    net_signed_t = pool_deficit_t - pool_surplus_t  # + = deficit, - = surplus
else:
    total_deficit_t = float(agg["Deficit_t"].sum())
    total_surplus_t = float(agg["Surplus_t"].sum())
    net_signed_t = total_deficit_t - total_surplus_t  # + = deficit, - = surplus

total_energy = float(agg["Energy_GJ"].sum())
total_co2e   = float(agg["CO2e_t"].sum())
pooled_int   = (total_co2e * 1000.0) / total_energy if total_energy > 0 else 0.0

# Cost only if net is a deficit
net_cost_eur = max(0.0, net_signed_t) * credit_price

# -------------------------------------------------------
# KPIs (Net shows +/- with color; cost only if deficit)
# -------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total energy (GJ)", f"{total_energy:,.0f}")
m2.metric("Total CO₂e (t)",    f"{total_co2e:,.0f}")
m3.metric("Pooled intensity (g/MJ)", f"{pooled_int:,.1f}")

# Net (signed). Positive = deficit (red), Negative = surplus (green)
net_label = "Net obligation (t)" if pooling else "Net (t)"
net_value_display = f"{abs(net_signed_t):,.0f}"
net_delta_display = f"{net_signed_t:+,.0f} t"  # e.g., +1,234 t or -567 t
m4.metric(net_label, net_value_display, delta=net_delta_display, delta_color="inverse")

# Estimated cost: only on deficits
st.metric("Estimated cost (€)", f"{net_cost_eur:,.0f}")
st.caption(
    "Net shown as signed value: **positive (red) = deficit**, **negative (green) = surplus**. "
    "Estimated cost applies only to deficits: `max(0, net) × price`."
)


# -------------------------------------------------------
# Table
# -------------------------------------------------------
display_cols = ["Vessel","Energy_GJ","CO2e_t","Intensity_g_per_MJ","Delta_g_per_MJ","Deficit_t","Surplus_t"]
st.subheader("Vessel summary")
st.dataframe(agg[display_cols].round({
    "Energy_GJ":0,"CO2e_t":0,"Intensity_g_per_MJ":1,"Delta_g_per_MJ":1,"Deficit_t":0,"Surplus_t":0
}), use_container_width=True)

# -------------------------------------------------------
# Graphic — deficits vs. surpluses
# -------------------------------------------------------
st.subheader("Deficit / Surplus overview")
bars = go.Figure()
bars.add_bar(name="Deficit (t)",  x=agg["Vessel"], y=agg["Deficit_t"], marker_color="rgba(231, 76, 60, 0.85)")
bars.add_bar(name="Surplus (t)",  x=agg["Vessel"], y=agg["Surplus_t"], marker_color="rgba(46, 204, 113, 0.85)")
bars.update_layout(barmode="group", margin=dict(l=10,r=10,t=30,b=10), xaxis_title="", yaxis_title="tCO₂e")
st.plotly_chart(bars, use_container_width=True)

# -------------------------------------------------------
# Drilldown (optional): show legs for selected vessel
# -------------------------------------------------------
with st.expander("Voyage-leg details"):
    if vsel != "All":
        st.dataframe(
            tbl[tbl["Vessel"] == vsel][[
                "Year","Vessel","Voyage","Leg","From","To","Start","End","HFO_t","MGO_t","LNG_t",
                "Energy_GJ","CO2e_t","Intensity_g_per_MJ"
            ]].round({"Energy_GJ":0,"CO2e_t":0,"Intensity_g_per_MJ":1}),
            use_container_width=True
        )
    else:
        st.info("Select a single vessel above to see leg-level details.")
