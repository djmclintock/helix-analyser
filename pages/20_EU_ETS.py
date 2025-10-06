# helix-analyser/pages/20_EU_ETS.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import date
import plotly.graph_objects as go

st.set_page_config(page_title="EU ETS — Summary", layout="wide")
st.title("EU ETS — Summary")

# ------------------------------------------------------------------------------------
# Dummy dataset (replace with DB pull later)
# Columns: Vessel, Voyage, Leg, From, To, Scope, Start, End, CO2_t
# Scope: "EU-EU" -> 100%, "EU-In"/"EU-Out" -> 50%, "Non-EU" -> 0%
# ------------------------------------------------------------------------------------
def load_dummy_legs() -> pd.DataFrame:
    rows = [
        ("ShipNet LNG",       "LNG-001", 1, "Zeebrugge", "Ras Laffan", "EU-Out",  "2025-01-10 04:00", "2025-01-20 08:00",  9000.0),
        ("ShipNet LNG",       "LNG-001", 2, "Ras Laffan","Zeebrugge",  "EU-In",   "2025-02-01 06:00", "2025-02-10 21:00",  8700.0),

        ("ShipNet Container", "CON-101", 1, "Rotterdam", "Algeciras",  "EU-EU",   "2025-03-03 10:00", "2025-03-06 16:00",   950.0),
        ("ShipNet Container", "CON-101", 2, "Algeciras", "New York",   "EU-Out",  "2025-03-07 02:00", "2025-03-15 09:00",  2350.0),
        ("ShipNet Container", "CON-102", 1, "New York",  "Rotterdam",  "EU-In",   "2025-04-01 04:00", "2025-04-10 13:00",  2410.0),

        ("ShipNet Bulk",      "BLK-201", 1, "Piraeus",   "Constanta",  "EU-EU",   "2025-02-12 07:00", "2025-02-14 11:00",   420.0),
        ("ShipNet Bulk",      "BLK-202", 1, "Sohar",     "Koper",      "Non-EU",  "2025-03-20 05:00", "2025-03-27 20:00",  3100.0),
        ("ShipNet Bulk",      "BLK-202", 2, "Koper",     "Ravenna",    "EU-EU",   "2025-03-29 06:00", "2025-03-30 22:00",   300.0),

        ("ShipNet Gas",       "GAS-050", 1, "Gothenburg","Aarhus",     "EU-EU",   "2025-05-01 02:00", "2025-05-02 18:00",   210.0),
        ("ShipNet Tanker",    "TNK-777", 1, "Augusta",   "Fujairah",   "EU-Out",  "2025-06-01 09:00", "2025-06-12 03:00",  5200.0),
    ]
    df = pd.DataFrame(rows, columns=[
        "Vessel","Voyage","Leg","From","To","Scope","Start","End","CO2_t"
    ])
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"]   = pd.to_datetime(df["End"])
    df.insert(0, "Year", df["Start"].dt.year)
    return df

df = load_dummy_legs()

# ------------------------------------------------------------------------------------
# Controls (filters + assumptions)
# ------------------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1.2])

with c1:
    vessels = ["All"] + sorted(df["Vessel"].unique().tolist())
    vsel = st.selectbox("Vessel", vessels)
with c2:
    d_from = st.date_input("From", value=date(2025,1,1))
with c3:
    d_to   = st.date_input("To",   value=date(2025,12,31))
with c4:
    # Default EUA price (editable)
    eua_price = st.number_input("EUA price (€/tCO₂)", min_value=0.0, value=79.67, step=0.1)

c5, c6 = st.columns([1, 2])
with c5:
    apply_phase_in = st.checkbox("Apply maritime phase-in (40/70/100%)", value=True,
                                 help="40% (2024), 70% (2025), 100% (2026+).")
with c6:
    year_override = st.selectbox("Assumption year for phase-in", sorted(df["Year"].unique().tolist()), index=0)

# Filter table
mask = (df["Start"].dt.date >= d_from) & (df["End"].dt.date <= d_to)
if vsel != "All":
    mask &= df["Vessel"].eq(vsel)
tbl = df.loc[mask].copy()

# ------------------------------------------------------------------------------------
# ETS factors (internal calc — not shown as columns)
# ------------------------------------------------------------------------------------
AREA_FACTOR = {"EU-EU": 1.0, "EU-In": 0.5, "EU-Out": 0.5, "Non-EU": 0.0}
def phase_in_factor(y: int) -> float:
    if y <= 2023: return 0.0
    if y == 2024: return 0.40
    if y == 2025: return 0.70
    return 1.00

tbl["AreaFactor"] = tbl["Scope"].map(AREA_FACTOR).fillna(0.0)
if apply_phase_in:
    tbl["PhaseIn"] = tbl["Year"].apply(
        phase_in_factor if year_override is None else (lambda _: phase_in_factor(year_override))
    )
else:
    tbl["PhaseIn"] = 1.0

# ETS-accountable CO2 and EUAs per leg
tbl["CO2_ETS_t"]     = tbl["CO2_t"] * tbl["AreaFactor"] * tbl["PhaseIn"]
tbl["EUAs_required"] = tbl["CO2_ETS_t"]  # 1 EUA = 1 tCO2e

# ------------------------------------------------------------------------------------
# Top table (key data, no Area/Phase columns)
# ------------------------------------------------------------------------------------
show_cols = ["Year","Vessel","Voyage","Leg","From","To","Scope","Start","End",
             "CO2_t","CO2_ETS_t","EUAs_required"]
st.subheader("Voyage-leg details")
st.dataframe(
    tbl[show_cols].round({"CO2_t":0,"CO2_ETS_t":0,"EUAs_required":0}),
    use_container_width=True
)

# ------------------------------------------------------------------------------------
# Summary cards
# ------------------------------------------------------------------------------------
total_verified = float(tbl["CO2_t"].sum())
total_ets      = float(tbl["CO2_ETS_t"].sum())
total_euas     = float(tbl["EUAs_required"].sum())
est_cost_eur   = total_euas * eua_price

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total verified CO₂ (t)", f"{total_verified:,.0f}")
m2.metric("ETS-eligible CO₂ (t)",   f"{total_ets:,.0f}")
m3.metric("EUAs required",          f"{total_euas:,.0f}")
m4.metric("Est. cost (€)",          f"{est_cost_eur:,.0f}")

st.caption(
    "ETS-eligible CO₂ = Verified CO₂ × area factor (EU-EU 100%, EU in/out 50%, non-EU 0%) "
    "× phase-in (40% 2024, 70% 2025, 100% 2026+). 1 EUA = 1 tCO₂e; cost = EUAs × EUA price."
)

# ------------------------------------------------------------------------------------
# Simple summary graphic
# ------------------------------------------------------------------------------------
bar = go.Figure()
bar.add_trace(go.Bar(name="Verified CO₂ (t)",    x=["Total"], y=[total_verified],
                     marker_color="rgba(33,150,243,0.8)"))
bar.add_trace(go.Bar(name="ETS-eligible CO₂ (t)", x=["Total"], y=[total_ets],
                     marker_color="rgba(255,87,34,0.85)"))
bar.add_trace(go.Bar(name="EUAs (tCO₂e)",         x=["Total"], y=[total_euas],
                     marker_color="rgba(76,175,80,0.85)"))
bar.update_layout(barmode="group", margin=dict(l=10, r=10, t=30, b=10))
bar.update_yaxes(title_text="tCO₂ / EUAs")
st.plotly_chart(bar, use_container_width=True)
