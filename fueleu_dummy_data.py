# helix-analyser/fueleu_dummy_data.py
from __future__ import annotations
import pandas as pd
from datetime import datetime

# Very simple demo set — replace with DB pull later.
# Fuel -> (LHV_GJ_per_t, EF_tCO2e_per_t_fuel)  (tank-to-wake-style demo factors)
FUEL_FACTORS = {
    "HFO": (40.4, 3.114),  # 40.4 GJ/t, 3.114 tCO2/t
    "MGO": (42.7, 3.206),
    "LNG": (48.0, 2.750),
    # Add low/zero-carbon fuels as needed, e.g. e-methanol, HVO, etc.
}

def load_dummy_fueleu_legs() -> pd.DataFrame:
    """
    Returns rows with fuel tonnages per leg.
    Columns:
      Year, Vessel, Voyage, Leg, From, To, Start, End, HFO_t, MGO_t, LNG_t
    """
    rows = [
        ("ShipNet LNG",       "LNG-001", 1, "Zeebrugge", "Ras Laffan", "2025-01-10 04:00", "2025-01-20 08:00",  1200.0,   50.0,  300.0),
        ("ShipNet LNG",       "LNG-001", 2, "Ras Laffan","Zeebrugge",  "2025-02-01 06:00", "2025-02-10 21:00",   900.0,   40.0,  260.0),

        ("ShipNet Container", "CON-101", 1, "Rotterdam", "Algeciras",  "2025-03-03 10:00", "2025-03-06 16:00",    80.0,   20.0,    0.0),
        ("ShipNet Container", "CON-101", 2, "Algeciras", "New York",   "2025-03-07 02:00", "2025-03-15 09:00",   320.0,   60.0,   10.0),
        ("ShipNet Container", "CON-102", 1, "New York",  "Rotterdam",  "2025-04-01 04:00", "2025-04-10 13:00",   330.0,   55.0,   15.0),

        ("ShipNet Bulk",      "BLK-201", 1, "Piraeus",   "Constanta",  "2025-02-12 07:00", "2025-02-14 11:00",    35.0,   15.0,    0.0),
        ("ShipNet Bulk",      "BLK-202", 1, "Sohar",     "Koper",      "2025-03-20 05:00", "2025-03-27 20:00",   220.0,   35.0,    0.0),

        ("ShipNet Gas",       "GAS-050", 1, "Gothenburg","Aarhus",     "2025-05-01 02:00", "2025-05-02 18:00",    18.0,    6.0,    0.0),

        ("ShipNet Tanker",    "TNK-777", 1, "Augusta",   "Fujairah",   "2025-06-01 09:00", "2025-06-12 03:00",   420.0,   80.0,    0.0),
    ]
    df = pd.DataFrame(rows, columns=[
        "Vessel","Voyage","Leg","From","To","Start","End","HFO_t","MGO_t","LNG_t"
    ])
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"]   = pd.to_datetime(df["End"])
    df.insert(0, "Year", df["Start"].dt.year)
    return df

def fuel_energy_emissions(row) -> tuple[float, float]:
    """
    Convert fuel tonnages to total energy (GJ) and CO2e (t).
    """
    energy_gj = 0.0
    co2e_t    = 0.0
    for fuel, (lhv, ef) in FUEL_FACTORS.items():
        tonnes = float(row.get(f"{fuel}_t", 0.0) or 0.0)
        energy_gj += tonnes * lhv
        co2e_t    += tonnes * ef
    return energy_gj, co2e_t

def compute_leg_energy_emissions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    E, C = [], []
    for _, r in out.iterrows():
        e, c = fuel_energy_emissions(r)
        E.append(e); C.append(c)
    out["Energy_GJ"] = E
    out["CO2e_t"]    = C
    # Intensity gCO2e/MJ = (tCO2e * 1e6 g) / (Energy_GJ * 1000 MJ/GJ) = (tCO2e*1000) / Energy_GJ
    out["Intensity_g_per_MJ"] = (out["CO2e_t"] * 1000.0) / out["Energy_GJ"]
    return out
