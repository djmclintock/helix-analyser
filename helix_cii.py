
"""
helix_cii.py — CII calculations & plotting for Helix Analyser
Author: ChatGPT (for Helix)
Date: 2025-10-03

This module provides:
- Fuel -> CO2 conversion using IMO CF factors (with Excel override support)
- Attained CII calculation (gCO2 / DWT·nm)
- Required CII thresholds (A–E) loaded from an Excel "Reference Table" or "Reference Line"
- Interpolation by DWT within a Year
- Rating (A–E) evaluation
- Plotting on a Deadweight vs CII scale (matplotlib)

Designed to drop into the existing Helix Analyser codebase.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Defaults & Utilities
# -----------------------------

DEFAULT_CF_FACTORS = {
    # tCO2 per tonne of fuel (IMO typical values)
    # Source: IMO guidelines (commonly used values)
    "HFO": 3.114,
    "LFO": 3.151,
    "MDO": 3.206,
    "MGO": 3.206,
    "LNG": 2.750,
    "LPG_propane": 3.000,
    "LPG_butane": 3.030,
    "Methanol": 1.375,
    "Ethanol": 1.913,
    "Ammonia": 0.000,  # CO2 free at point of use (ignores upstream)
}

@dataclass
class CIIResult:
    attained_g_per_dwt_nm: float
    required_line: Optional[float]
    bands: Optional[Dict[str, float]]  # keys like "Upper A", "Lower B", etc.
    rating: Optional[str]  # 'A'..'E' or None when not computable


# -----------------------------
# CF Factor Handling
# -----------------------------

def load_cf_factors_from_excel(path: str, sheet_name: str = "Constants") -> Dict[str, float]:
    """
    Best-effort loader: if a fuel->CF table exists on the sheet, extract it.
    If not found, returns DEFAULT_CF_FACTORS.
    Expected format (flexible): columns containing 'Fuel' and 'CF' (case-insensitive).
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    except Exception:
        return DEFAULT_CF_FACTORS

    # Try to locate header row where two columns look like 'Fuel' and 'CF'
    header_row = None
    for i in range(min(40, len(df))):
        row = df.iloc[i].astype(str).str.strip().str.lower().tolist()
        if any("fuel" in c for c in row) and any("cf" in c or "co2" in c for c in row):
            header_row = i
            break

    if header_row is None:
        return DEFAULT_CF_FACTORS

    df2 = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    # Find columns
    cols = [c for c in df2.columns if isinstance(c, str)]
    fuel_col = next((c for c in cols if "fuel" in c.lower()), None)
    cf_col = next((c for c in cols if "cf" in c.lower() or "co2" in c.lower()), None)

    if fuel_col is None or cf_col is None:
        return DEFAULT_CF_FACTORS

    out: Dict[str, float] = {}
    for _, r in df2[[fuel_col, cf_col]].dropna().iterrows():
        key = str(r[fuel_col]).strip()
        try:
            val = float(r[cf_col])
        except Exception:
            continue
        if key:
            out[key] = val

    # Merge defaults for any missing common fuels
    merged = {**DEFAULT_CF_FACTORS, **out}
    return merged


# -----------------------------
# Reference Table Handling
# -----------------------------

REQUIRED_COLS = [
    "Year", "DWT",
    "Upper A", "Lower B",
    "Upper B", "Lower C",
    "Upper C", "Lower D",
    "Upper D", "Lower E",
    "Upper E"
]

def _normalize_reference_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a wide/messy 'Reference Table' or 'Reference Line' sheet into a tidy form
    with REQUIRED_COLS. Works with merged headers and extra unnamed columns.
    """
    # Drop entirely empty columns
    df = df.dropna(axis=1, how="all").copy()

    # Try to detect columns by fuzzy matching
    rename_map = {}
    for c in df.columns:
        cn = str(c).strip()
        l = cn.lower()
        if l in {"year"}:
            rename_map[c] = "Year"
        elif l in {"dwt", "deadweight", "deadweight tonnage"}:
            rename_map[c] = "DWT"
        elif "upper a" in l:
            rename_map[c] = "Upper A"
        elif "lower b" in l:
            rename_map[c] = "Lower B"
        elif "upper b" in l:
            rename_map[c] = "Upper B"
        elif "lower c" in l:
            rename_map[c] = "Lower C"
        elif "upper c" in l:
            rename_map[c] = "Upper C"
        elif "lower d" in l:
            rename_map[c] = "Lower D"
        elif "upper d" in l:
            rename_map[c] = "Upper D"
        elif "lower e" in l:
            rename_map[c] = "Lower E"
        elif "upper e" in l:
            rename_map[c] = "Upper E"
        elif "annual line" in l or "annaul line" in l or "required" in l:
            rename_map[c] = "Required"

    df = df.rename(columns=rename_map)

    # If no Year column found, try to infer from a column that looks like years in values
    if "Year" not in df.columns:
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
                # Heuristic: if values are between 2019 and 2035, assume year
                vals = df[c].dropna()
                if not vals.empty and (vals.between(2019, 2035).mean() > 0.6):
                    df = df.rename(columns={c: "Year"})
                    break

    # If no DWT column, try to infer from a column named like 'Capacity'
    if "DWT" not in df.columns:
        for c in df.columns:
            if "capacity" in str(c).lower():
                df = df.rename(columns={c: "DWT"})
                break

    # Keep only relevant cols if present
    keep = ["Year", "DWT", "Required",
            "Upper A", "Lower B", "Upper B", "Lower C", "Upper C",
            "Lower D", "Upper D", "Lower E", "Upper E"]
    present = [c for c in keep if c in df.columns]
    df = df[present].copy()

    # Drop rows with no Year or no DWT
    if "Year" in df.columns:
        df = df[~df["Year"].isna()]
    if "DWT" in df.columns:
        df = df[~df["DWT"].isna()]

    # Coerce numerics
    for c in df.columns:
        if c in {"Year", "DWT"} or c.startswith("Upper") or c.startswith("Lower") or c == "Required":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Deduplicate
    df = df.dropna(how="all")
    df = df.drop_duplicates(subset=[c for c in ["Year", "DWT"] if c in df.columns])

    return df.reset_index(drop=True)


class CIIReference:
    def __init__(self, table: pd.DataFrame):
        self.table = _normalize_reference_table(table)

    @classmethod
    def from_excel(cls, path: str, sheet_name: str = "Reference Table") -> "CIIReference":
        df = pd.read_excel(path, sheet_name=sheet_name)
        return cls(df)

    def thresholds_for(self, year: int, dwt: float) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """
        Returns (required_line, bands_dict) for a given year and DWT using linear interpolation on DWT.
        bands_dict keys: "Upper A","Lower B","Upper B","Lower C","Upper C","Lower D","Upper D","Lower E","Upper E"
        Any missing values are returned as None.
        """
        tbl = self.table.copy()
        if "Year" in tbl.columns:
            tbl = tbl[tbl["Year"] == year]
            if tbl.empty:
                # Fallback: nearest year available
                years = self.table["Year"].dropna().unique()
                if len(years):
                    nearest = int(min(years, key=lambda y: abs(y - year)))
                    tbl = self.table[self.table["Year"] == nearest]
                else:
                    return None, None

        # Require DWT column
        if "DWT" not in tbl.columns or tbl["DWT"].dropna().empty:
            return None, None

        tbl = tbl.sort_values("DWT")

        # If exact match
        if (tbl["DWT"] == dwt).any():
            row = tbl.loc[tbl["DWT"] == dwt].iloc[0]
            required = row.get("Required", np.nan)
            bands = {k: (row.get(k, np.nan) if k in tbl.columns else np.nan)
                     for k in ["Upper A","Lower B","Upper B","Lower C","Upper C","Lower D","Upper D","Lower E","Upper E"]}
            # normalize to None for NaNs
            bands = {k: (None if pd.isna(v) else float(v)) for k, v in bands.items()}
            required = None if pd.isna(required) else float(required)
            return required, bands

        # Interpolate between nearest DWTs
        below = tbl[tbl["DWT"] <= dwt].tail(1)
        above = tbl[tbl["DWT"] >= dwt].head(1)
        if below.empty or above.empty:
            # clamp to closest
            row = (below if not below.empty else above).iloc[0]
            required = row.get("Required", np.nan)
            required = None if pd.isna(required) else float(required)
            bands = {k: (row.get(k, np.nan) if k in tbl.columns else np.nan)
                     for k in ["Upper A","Lower B","Upper B","Lower C","Upper C","Lower D","Upper D","Lower E","Upper E"]}
            bands = {k: (None if pd.isna(v) else float(v)) for k, v in bands.items()}
            return required, bands

        row_lo = below.iloc[0]
        row_hi = above.iloc[0]
        x0, x1 = float(row_lo["DWT"]), float(row_hi["DWT"])
        t = 0.0 if x1 == x0 else (dwt - x0) / (x1 - x0)

        def lerp(a, b):
            if pd.isna(a) and pd.isna(b):
                return None
            if pd.isna(a): a = b
            if pd.isna(b): b = a
            return float(a + t * (b - a))

        required = None
        if "Required" in tbl.columns:
            required = lerp(row_lo.get("Required", np.nan), row_hi.get("Required", np.nan))

        bands = {}
        for k in ["Upper A","Lower B","Upper B","Lower C","Upper C","Lower D","Upper D","Lower E","Upper E"]:
            a = row_lo.get(k, np.nan)
            b = row_hi.get(k, np.nan)
            bands[k] = lerp(a, b)

        return required, bands


# -----------------------------
# Core Calculations
# -----------------------------

def total_co2_tonnes(fuel_tonnes: Dict[str, float], cf: Dict[str, float]) -> float:
    """
    Sum CO2 tonnes = sum_i (fuel_i_tonnes * CF_i).
    Unrecognised fuels fall back to 0 unless a CF exists in 'cf' dict.
    """
    total = 0.0
    for fuel, tonnes in (fuel_tonnes or {}).items():
        if tonnes is None:
            continue
        factor = cf.get(fuel, 0.0)
        total += float(tonnes) * float(factor)
    return total


def attained_cii_g_per_dwt_nm(co2_tonnes: float, distance_nm: float, dwt: float) -> float:
    """
    gCO2 / (DWT·nm) = (co2_tonnes * 1e6 g/tonne) / (distance_nm * dwt)
    """
    if distance_nm <= 0 or dwt <= 0:
        return float("nan")
    return (co2_tonnes * 1_000_000.0) / (float(distance_nm) * float(dwt))


def rating_from_thresholds(attained: float, bands: Optional[Dict[str, float]]) -> Optional[str]:
    """
    Given attained CII and band thresholds, return 'A'..'E'.
    Expected semantics (common convention):
      A: attained <= Upper A
      B: Upper A < attained <= Upper B
      C: Upper B < attained <= Upper C
      D: Upper C < attained <= Upper D
      E: attained > Upper D
    Falls back to thresholds provided (if some missing, uses those available).
    """
    if bands is None:
        return None
    ua = bands.get("Upper A")
    ub = bands.get("Upper B")
    uc = bands.get("Upper C")
    ud = bands.get("Upper D")

    if ua is not None and attained <= ua:
        return "A"
    if ub is not None and attained <= ub:
        return "B"
    if uc is not None and attained <= uc:
        return "C"
    if ud is not None and attained <= ud:
        return "D"
    return "E"


def compute_cii(
    dwt: float,
    distance_nm: float,
    fuel_tonnes_by_type: Dict[str, float],
    year: Optional[int] = None,
    cf_factors: Optional[Dict[str, float]] = None,
    reference: Optional[CIIReference] = None,
) -> CIIResult:
    """
    High-level API:
    - Computes attained CII
    - Looks up/interpolates required line + bands (if 'reference' and 'year' provided)
    - Returns rating
    """
    cf = cf_factors or DEFAULT_CF_FACTORS
    co2 = total_co2_tonnes(fuel_tonnes_by_type, cf)
    attained = attained_cii_g_per_dwt_nm(co2, distance_nm, dwt)

    required = None
    bands = None
    if reference is not None and year is not None:
        required, bands = reference.thresholds_for(int(year), float(dwt))

    rating = rating_from_thresholds(attained, bands) if bands else None
    return CIIResult(attained, required, bands, rating)


# -----------------------------
# Plotting
# -----------------------------

def plot_deadweight_scale(
    dwt: float,
    attained: float,
    year: int,
    reference: CIIReference,
    dwt_min: Optional[float] = None,
    dwt_max: Optional[float] = None,
    fig_size: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Returns a Matplotlib Figure showing A–E bands vs DWT for a given year and the vessel's point.
    No explicit colors are set (per Helix plotting guidance).
    """
    tbl = reference.table
    tbl = tbl[tbl["Year"] == year] if "Year" in tbl.columns else tbl.copy()

    # Guard: need DWT and at least one band column
    band_cols = [c for c in ["Upper A","Upper B","Upper C","Upper D"] if c in tbl.columns]
    if "DWT" not in tbl.columns or not band_cols:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.set_title(f"CII Bands (Year {year}) — data unavailable")
        ax.set_xlabel("Deadweight (DWT)")
        ax.set_ylabel("CII (gCO2 / DWT·nm)")
        ax.scatter([dwt], [attained], marker="x")
        return fig

    tbl = tbl.sort_values("DWT")
    if dwt_min is None:
        dwt_min = float(max(0, np.nanmin(tbl["DWT"])))
    if dwt_max is None:
        dwt_max = float(np.nanmax(tbl["DWT"]))

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    # Plot band boundaries as lines
    for col in band_cols:
        ax.plot(tbl["DWT"], tbl[col], label=col)

    # Vessel point
    ax.scatter([dwt], [attained], marker="x", s=60, label="Attained")

    ax.set_xlim(dwt_min, dwt_max)
    ax.set_title(f"CII Deadweight Scale — Year {year}")
    ax.set_xlabel("Deadweight (DWT)")
    ax.set_ylabel("CII (gCO2 / DWT·nm)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


# -----------------------------
# Convenience: Monthly / Voyage utilities
# -----------------------------

def group_and_compute(
    df: pd.DataFrame,
    dwt: float,
    group_col: str,
    distance_col: str,
    year_col: Optional[str],
    fuel_cols: Dict[str, str],
    cf_factors: Optional[Dict[str, float]] = None,
    reference: Optional[CIIReference] = None,
) -> pd.DataFrame:
    """
    Compute attained & rating by a grouping (e.g., 'Month', 'VoyageID').
    df: input DataFrame containing 'distance_col' and fuel columns (in tonnes).
    fuel_cols: mapping like {'HFO': 'hfo_t', 'LNG':'lng_t', ...}
    Returns a tidy DataFrame with attained, required, rating per group.
    """
    cf = cf_factors or DEFAULT_CF_FACTORS

    def _agg(group: pd.DataFrame) -> pd.Series:
        distance = float(group[distance_col].sum())
        fuel_map = {fuel: float(group[col].sum()) for fuel, col in fuel_cols.items() if col in group}
        co2 = total_co2_tonnes(fuel_map, cf)
        attained = attained_cii_g_per_dwt_nm(co2, distance, dwt)
        year = int(group[year_col].iloc[0]) if year_col else None
        required = None
        bands = None
        rating = None
        if reference is not None and year is not None:
            required, bands = reference.thresholds_for(year, dwt)
            rating = rating_from_thresholds(attained, bands)
        return pd.Series({
            "distance_nm": distance,
            "co2_tonnes": co2,
            "attained_g_per_dwt_nm": attained,
            "required_cii": required,
            "rating": rating
        })

    out = df.groupby(group_col, dropna=False).apply(_agg).reset_index()
    return out
