
# cii_streamlit_panel.py
# Streamlit panel for CII planning & actuals using helix_cii.py
# Drop-in for your existing Helix Analyser app.
#
# Usage inside your app:
#   from cii_streamlit_panel import render_cii_panel
#   with tab_cii:
#       render_cii_panel()
#
# Or run standalone for testing:
#   streamlit run cii_streamlit_panel.py

import io
import json
import pandas as pd
import streamlit as st
from typing import Dict, Optional
from helix_cii import (
    CIIReference,
    compute_cii,
    load_cf_factors_from_excel,
    plot_deadweight_scale,
    group_and_compute,
    DEFAULT_CF_FACTORS,
)

st.set_page_config(page_title="CII Planner & Actuals", layout="wide")

def _pathlike_from_uploader(u):
    # Streamlit gives a BytesIO-like object; pass directly to pandas readers
    return u

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
    st.markdown("### CII Planning & Actuals")

    with st.expander("📘 Data source", expanded=True):
        source_mode = st.radio("Reference source", ["Upload workbook", "Use server path"], horizontal=True)
        ref_sheet = st.text_input("Reference sheet name", value="Reference Table")
        const_sheet = st.text_input("Constants sheet name (for CF factors, optional)", value="Constants")

        xlsx_source = None
        if source_mode == "Upload workbook":
            xlsx = st.file_uploader("Upload the CII workbook (.xlsx)", type=["xlsx"], accept_multiple_files=False)
            if xlsx is not None:
                xlsx_source = _pathlike_from_uploader(xlsx)
        else:
            default_path_val = default_workbook_path or ""
            xlsx_path = st.text_input("Server path to workbook (.xlsx)", value=default_path_val)
            if xlsx_path:
                xlsx_source = xlsx_path

        if not xlsx_source:
            st.info("Provide a workbook to enable thresholds & CF overrides (you can still use default CFs).")

        # Load reference & CF (best-effort)
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
        # year options from reference if available
        year_default = 2023
        if reference is not None and "Year" in reference.table.columns:
            years = sorted([int(y) for y in reference.table["Year"].dropna().unique()])
            year = st.selectbox("Year", years, index=(years.index(year_default) if year_default in years else 0))
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
            # Infer fuel columns present
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


# Standalone execution entrypoint
if __name__ == "__main__":
    render_cii_panel()
