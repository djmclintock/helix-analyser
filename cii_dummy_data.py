import pandas as pd

def get_dummy_cii_data():
    data = [
        # vessel, type, voyage, from, to, distance nm, fuel t, CO2 t, CII, rating
        ["ShipNet LNG",       "LNG Carrier",  "LNG001", "2025-01-10", "2025-01-20", 12000, 3000, 9342, 15.6, "C"],
        ["ShipNet Bulk",      "Bulk Carrier", "BULK01", "2025-02-01", "2025-02-15",  8000, 1500, 4671, 14.5, "B"],
        ["ShipNet Container", "Container",    "CONT01", "2025-03-05", "2025-03-12",  6000, 1200, 3798, 17.9, "D"],
        ["ShipNet Gas",       "Gas Carrier",  "GAS01",  "2025-01-25", "2025-02-03",  5000, 1100, 3480, 18.7, "D"],
        ["ShipNet Tanker",    "Tanker",       "TNK01",  "2025-02-20", "2025-02-28",  7000, 1600, 4982, 20.3, "E"],
    ]
    df = pd.DataFrame(data, columns=[
        "Vessel", "Type", "Voyage", "DateFrom", "DateTo",
        "Distance_nm", "Fuel_t", "CO2_t", "CII_gCO2_DWTnm", "Rating"
    ])
    return df
