# pages/10_CII.py
import streamlit as st
from cii_streamlit_panel import render_cii_panel

st.set_page_config(page_title="CII Planner & Actuals", layout="wide")
st.title("CII Planner & Actuals")

# Optionally set a default workbook path if you keep a shared reference on the server
render_cii_panel(default_workbook_path="")
