
# helix_commercial_analyzer.py
# Self-contained Streamlit app with CII module embedded.
# - Contains helix_cii module (as an internal section)
# - Contains the CII Streamlit panel
# - Provides a minimal app structure with tabs; drop-in friendly
#
# To run:
#   streamlit run helix_commercial_analyzer.py

import io
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Embedded helix_cii module
# -----------------------------

DEFAULT_CF_FACTORS = {
    "HFO": 3.114,
    "LFO": 3.151,
    "MDO": 3.206,
    "MGO": 3.206,
    "LNG": 2.750,
    "LPG_propane": 3.000,
    "LPG_butane": 3.030,
    "Methanol": 1.375,
    "Ethanol": 1.913,
    "Ammonia": 0.000,
}

@dataclass
class CIIResult:
    attained_g_per_dwt_nm: float
    required_line: Optional[float]
    bands: Optional[Dict[str, float]]
    rating: Optional[str]

def load_cf_factors_from_excel(path_or_buf, sheet_name: str = "Constants") -> Dict[str, float]:
    try:
        df = pd.read_excel(path_or_buf, sheet_name=sheet_name, header=None)
    except Exception:
        return DEFAULT_CF_FACTORS

    header_row = None
    for i in range(min(40, len(df))):
        row = df.iloc[i].astype(str).str.strip().str.lower().tolist()
        if any("fuel" in c for c in row) and any("cf" in c or "co2" in c for c in row):
            header_row = i
            break
    if header_row is None:
        return DEFAULT_CF_FACTORS

    df2 = pd.read_excel(path_or_buf, sheet_name=sheet_name, header=header_row)
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
    return {**DEFAULT_CF_FACTORS, **out}

REQUIRED_COLS = [
    "Year", "DWT", "Required",
    "Upper A", "Lower B", "Upper B", "Lower C",
    "Upper C", "Lower D", "Upper D", "Lower E", "Upper E"
]

def _normalize_reference_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all").copy()
    rename_map = {}
    for c in df.columns:
        cn = str(c).strip()
        l = cn.lower()
        if l == "year":
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

    if "Year" not in df.columns:
        for c in df.columns:
            vals = df[c].dropna()
            if not vals.empty and pd.api.types.is_numeric_dtype(vals):
                mask = (vals >= 2019) & (vals <= 2035)
                if mask.mean() > 0.6:
                    df = df.rename(columns={c: "Year"})
                    break
    if "DWT" not in df.columns:
        for c in df.columns:
            if "capacity" in str(c).lower():
                df = df.rename(columns={c: "DWT"})
                break

    keep = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[keep].copy()

    if "Year" in df.columns:
        df = df[~df["Year"].isna()]
    if "DWT" in df.columns:
        df = df[~df["DWT"].isna()]

    for c in df.columns:
        if c in {"Year", "DWT"} or c.startswith("Upper") or c.startswith("Lower") or c == "Required":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all")
    subset = [c for c in ["Year", "DWT"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset)
    return df.reset_index(drop=True)

class CIIReference:
    def __init__(self, table: pd.DataFrame):
        self.table = _normalize_reference_table(table)

    @classmethod
    def from_excel(cls, path_or_buf, sheet_name: str = "Reference Table"):
        df = pd.read_excel(path_or_buf, sheet_name=sheet_name)
        return cls(df)

    def thresholds_for(self, year: int, dwt: float) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        tbl = self.table.copy()
        if "Year" in tbl.columns:
            sub = tbl[tbl["Year"] == year]
            if sub.empty:
                years = self.table["Year"].dropna().unique()
                if len(years):
                    nearest = int(min(years, key=lambda y: abs(y - year)))
                    tbl = self.table[self.table["Year"] == nearest]
                else:
                    return None, None
            else:
                tbl = sub

        if "DWT" not in tbl.columns or tbl["DWT"].dropna().empty:
            return None, None

        tbl = tbl.sort_values("DWT")

        # exact
        if (tbl["DWT"] == dwt).any():
            row = tbl.loc[tbl["DWT"] == dwt].iloc[0]
            required = row.get("Required", np.nan)
            bands = {k: (row.get(k, np.nan) if k in tbl.columns else np.nan)
                     for k in ["Upper A","Lower B","Upper B","Lower C","Upper C","Lower D","Upper D","Lower E","Upper E"]}
            bands = {k: (None if pd.isna(v) else float(v)) for k, v in bands.items()}
            required = None if pd.isna(required) else float(required)
            return required, bands

        below = tbl[tbl["DWT"] <= dwt].tail(1)
        above = tbl[tbl["DWT"] >= dwt].head(1)
        if below.empty or above.empty:
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
            if pd.isna(a) and pd.isna(b): return None
            if pd.isna(a): a = b
            if pd.isna(b): b = a
            return float(a + t * (b - a))

        required = None
        if "Required" in tbl.columns:
            required = lerp(row_lo.get("Required", np.nan), row_hi.get("Required", np.nan))

        bands = {}
        for k in ["Upper A","Lower B","Upper B","Lower C","Upper C","Lower D","Upper D","Lower E","Upper E"]:
            bands[k] = lerp(row_lo.get(k, np.nan), row_hi.get(k, np.nan))

        return required, bands

def total_co2_tonnes(fuel_tonnes: Dict[str, float], cf: Dict[str, float]) -> float:
    total = 0.0
    for fuel, tonnes in (fuel_tonnes or {}).items():
        if tonnes is None: continue
        factor = cf.get(fuel, 0.0)
        total += float(tonnes) * float(factor)
    return total

def attained_cii_g_per_dwt_nm(co2_tonnes: float, distance_nm: float, dwt: float) -> float:
    if distance_nm <= 0 or dwt <= 0:
        return float("nan")
    return (co2_tonnes * 1_000_000.0) / (float(distance_nm) * float(dwt))

def rating_from_thresholds(attained: float, bands: Optional[Dict[str, float]]) -> Optional[str]:
    if bands is None: return None
    ua = bands.get("Upper A")
    ub = bands.get("Upper B")
    uc = bands.get("Upper C")
    ud = bands.get("Upper D")
    if ua is not None and attained <= ua: return "A"
    if ub is not None and attained <= ub: return "B"
    if uc is not None and attained <= uc: return "C"
    if ud is not None and attained <= ud: return "D"
    return "E"

def compute_cii(dwt: float, distance_nm: float, fuel_tonnes_by_type: Dict[str, float],
                year: Optional[int] = None, cf_factors: Optional[Dict[str, float]] = None,
                reference: Optional[CIIReference] = None) -> CIIResult:
    cf = cf_factors or DEFAULT_CF_FACTORS
    co2 = total_co2_tonnes(fuel_tonnes_by_type, cf)
    attained = attained_cii_g_per_dwt_nm(co2, distance_nm, dwt)
    required = None; bands = None
    if reference is not None and year is not None:
        required, bands = reference.thresholds_for(int(year), float(dwt))
    rating = rating_from_thresholds(attained, bands) if bands else None
    return CIIResult(attained, required, bands, rating)

def plot_deadweight_scale(dwt: float, attained: float, year: int, reference: CIIReference,
                          dwt_min: Optional[float] = None, dwt_max: Optional[float] = None,
                          fig_size: Tuple[int, int] = (8,5)) -> plt.Figure:
    tbl = reference.table
    tbl = tbl[tbl["Year"] == year] if "Year" in tbl.columns else tbl.copy()
    band_cols = [c for c in ["Upper A","Upper B","Upper C","Upper D"] if c in tbl.columns]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.set_xlabel("Deadweight (DWT)")
    ax.set_ylabel("CII (gCO2 / DWT·nm)")
    if "DWT" not in tbl.columns or not band_cols:
        ax.set_title(f"CII Bands (Year {year}) — data unavailable")
        ax.scatter([dwt], [attained], marker="x")
        return fig
    tbl = tbl.sort_values("DWT")
    if dwt_min is None: dwt_min = float(max(0, np.nanmin(tbl["DWT"])))
    if dwt_max is None: dwt_max = float(np.nanmax(tbl["DWT"]))
    for col in band_cols:
        ax.plot(tbl["DWT"], tbl[col], label=col)
    ax.scatter([dwt], [attained], marker="x", s=60, label="Attained")
    ax.set_xlim(dwt_min, dwt_max)
    ax.set_title(f"CII Deadweight Scale — Year {year}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig

def group_and_compute(df: pd.DataFrame, dwt: float, group_col: str, distance_col: str,
                      year_col: Optional[str], fuel_cols: Dict[str, str],
                      cf_factors: Optional[Dict[str, float]] = None,
                      reference: Optional[CIIReference] = None) -> pd.DataFrame:
    cf = cf_factors or DEFAULT_CF_FACTORS
    def _agg(group: pd.DataFrame) -> pd.Series:
        distance = float(group[distance_col].sum())
        fuel_map = {fuel: float(group[col].sum()) for fuel, col in fuel_cols.items() if col in group}
        co2 = total_co2_tonnes(fuel_map, cf)
        attained = attained_cii_g_per_dwt_nm(co2, distance, dwt)
        year = int(group[year_col].iloc[0]) if year_col else None
        required = bands = None; rating = None
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

# -----------------------------
# CII Streamlit panel
# -----------------------------

def _fuel_inputs(defaults: Dict[str, float]) -> Dict[str, float]:
    st.subheader("Fuel consumption (tonnes)")
    cols = st.columns(4)
    fuels = list(defaults.keys())
    fuel_map: Dict[str, float] = {}
    for i, fuel in enumerate(fuels):
        with cols[i % 4]:
            val = st.number_input(f"{fuel}", min_value=0.0, value=0.0, step=0.01, key=f"fuel_{fuel}")
            fuel_map[fuel] = float(val)
    return fuel_map

def _load_reference(xlsx_source, ref_sheet: str):
    try:
        ref = CIIReference.from_excel(xlsx_source, sheet_name=ref_sheet)
        return ref
    except Exception as e:
        st.error(f"Failed to load reference from sheet '{ref_sheet}': {e}")
        return None

def render_cii_panel(default_workbook_path: Optional[str] = None):
    with st.expander("📘 Data source", expanded=True):
        source_mode = st.radio("Reference source", ["Upload workbook", "Use server path"], horizontal=True)
        ref_sheet = st.text_input("Reference sheet name", value="Reference Table")
        const_sheet = st.text_input("Constants sheet name (for CF factors, optional)", value="Constants")

        xlsx_source = None
        if source_mode == "Upload workbook":
            xlsx = st.file_uploader("Upload the CII workbook (.xlsx)", type=["xlsx"], accept_multiple_files=False)
            if xlsx is not None:
                xlsx_source = xlsx
        else:
            default_path_val = default_workbook_path or ""
            xlsx_path = st.text_input("Server path to workbook (.xlsx)", value=default_path_val)
            if xlsx_path:
                xlsx_source = xlsx_path

        reference = _load_reference(xlsx_source, ref_sheet) if xlsx_source else None
        if xlsx_source:
            cf_factors = load_cf_factors_from_excel(xlsx_source, sheet_name=const_sheet)
        else:
            cf_factors = DEFAULT_CF_FACTORS

        with st.popover("View CF factors in use"):
            st.json(cf_factors)

    colL, colR = st.columns([1,1])
    with colL:
        st.subheader("Vessel & scenario")
        dwt = st.number_input("Deadweight (DWT)", min_value=1, value=100000, step=100)
        year_default = 2023
        if reference is not None and "Year" in reference.table.columns:
            years = sorted([int(y) for y in reference.table["Year"].dropna().unique()])
            idx = years.index(year_default) if year_default in years else 0
            year = st.selectbox("Year", years, index=idx)
        else:
            year = st.number_input("Year", min_value=2019, max_value=2035, value=year_default, step=1)
        distance_nm = st.number_input("Distance sailed (nm)", min_value=1.0, value=12000.0, step=10.0)

        fuel_map = _fuel_inputs(cf_factors)
        compute_btn = st.button("Compute CII", type="primary")

    with colR:
        st.subheader("Result")
        if compute_btn:
            res = compute_cii(
                dwt=dwt,
                distance_nm=distance_nm,
                fuel_tonnes_by_type=fuel_map,
                year=int(year),
                cf_factors=cf_factors,
                reference=reference,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Attained CII (g / DWT·nm)", f"{res.attained_g_per_dwt_nm:.3f}")
            m2.metric("Required line", f"{res.required_line:.3f}" if res.required_line is not None else "n/a")
            m3.metric("Rating", res.rating or "n/a")

            if reference is not None:
                fig = plot_deadweight_scale(
                    dwt=dwt,
                    attained=res.attained_g_per_dwt_nm,
                    year=int(year),
                    reference=reference,
                )
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Upload or reference a workbook to see band lines plotted.")

    st.markdown("---")
    st.subheader("Per-voyage / Monthly batch")
    st.caption("Upload a CSV with group-wise totals to compute CII per group. Template available below.")

    tmpl = pd.DataFrame({
        "Group": ["Voy1","Voy2"],
        "Year": [2023, 2023],
        "Distance_nm": [3200, 4100],
        **{k: [0.0, 0.0] for k in cf_factors.keys()},
    })
    buf = io.BytesIO()
    tmpl.to_csv(buf, index=False)
    st.download_button("Download CSV template", data=buf.getvalue(), file_name="cii_batch_template.csv", mime="text/csv")

    up = st.file_uploader("Upload batch CSV", type=["csv"], accept_multiple_files=False, key="batch_csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            fuel_cols = {fuel: fuel for fuel in cf_factors.keys() if fuel in df.columns}
            missing = [c for c in ["Group","Year","Distance_nm"] if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                out = group_and_compute(
                    df=df.rename(columns={"Group":"_Group"}),
                    dwt=dwt,
                    group_col="_Group",
                    distance_col="Distance_nm",
                    year_col="Year",
                    fuel_cols=fuel_cols,
                    cf_factors=cf_factors,
                    reference=reference,
                ).rename(columns={"_Group":"Group"})
                st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# -----------------------------
# App layout
# -----------------------------

st.set_page_config(page_title="Helix Analyser — Commercial", layout="wide")

st.title("Helix Commercial Analyzer")
tab_overview, tab_cii = st.tabs(["Overview", "CII"])

with tab_overview:
    st.write("Overview content goes here.")

with tab_cii:
    # If you have a server workbook, pass its path here:
    render_cii_panel(default_workbook_path="")
