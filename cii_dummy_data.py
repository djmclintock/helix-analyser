# helix-analyser/cii_dummy_data.py
from __future__ import annotations
import pandas as pd
from datetime import date

def get_dummy_vessels() -> pd.DataFrame:
    """
    Five standard vessels with typical particulars for demo calculations.
    Units:
      - DWT: tonnes
      - Type: high-level ship type
    """
    rows = [
        # name,             type,           DWT
        ("ShipNet LNG",      "LNG Carrier",  100_000),
        ("ShipNet Container","Container",     80_000),
        ("ShipNet Gas",      "Gas Carrier",   55_000),
        ("ShipNet Bulk",     "Bulk Carrier",  82_000),
        ("ShipNet Tanker",   "Tanker",       110_000),
    ]
    return pd.DataFrame(rows, columns=["Vessel", "Type", "DWT"])

def get_dummy_voyages() -> pd.DataFrame:
    """
    Minimal per-voyage demo data.
    Distance/Fuel are rough placeholders to drive the UI.
    CII is NOT precomputed here; your page can compute/override if needed.
    """
    rows = [
        # Vessel,            Voyage,   DateFrom,     DateTo,       Distance_nm, HFO_t, MDO_t, LNG_t
        ("ShipNet LNG",       "LNG001", date(2025,1,10), date(2025,1,20), 12000,  0.0,  0.0, 3000.0),
        ("ShipNet Bulk",      "BULK01", date(2025,2, 1), date(2025,2,15),  8000,1500.0,  50.0,   0.0),
        ("ShipNet Container", "CONT01", date(2025,3, 5), date(2025,3,12),  6000,1200.0,  80.0,   0.0),
        ("ShipNet Gas",       "GAS01",  date(2025,1,25), date(2025,2, 3),  5000, 900.0,  60.0, 150.0),
        ("ShipNet Tanker",    "TNK01",  date(2025,2,20), date(2025,2,28),  7000,1600.0,  70.0,   0.0),
    ]
    df = pd.DataFrame(rows, columns=[
        "Vessel","Voyage","DateFrom","DateTo","Distance_nm","HFO_t","MDO_t","LNG_t"
    ])
    # simple placeholder for CO2 and attained CII (gCO2/DWT·nm) using rough CFs
    CF = {"HFO_t": 3.114, "MDO_t": 3.206, "LNG_t": 2.750}  # tCO2 per t fuel
    df["CO2_t"] = df["HFO_t"]*CF["HFO_t"] + df["MDO_t"]*CF["MDO_t"] + df["LNG_t"]*CF["LNG_t"]
    return df
