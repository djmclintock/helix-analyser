"""
Helix Commercial Analyzer — Streamlit MVP
(with Synapse profiles, Secrets support, and Cloud-friendly DBAPI)

This app lets you:
- Pick a server profile (Azure Synapse / SQL) from config or Streamlit Secrets
- Run SELECT queries (safe by default)
- Build charts and a multi-tile dashboard
- Save/load dashboard JSON

Cloud vs Local
- Local: default DBAPI = pyodbc (needs Microsoft ODBC Driver 18)
- Cloud (Streamlit Community Cloud): set HELIX_DBAPI=pytds and add creds via Secrets; requirements.txt should use python-tds
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pydantic import BaseModel, Field, validator
from urllib.parse import quote_plus

try:
    import yaml
except Exception:
    yaml = None

try:
    from sqlalchemy import create_engine, text
except Exception:
    create_engine = None
    text = None

# ---------------------------
# Config & Connection Profiles
# ---------------------------
# NOTE: For security, leave username/password blank here. Use Streamlit Secrets in Cloud.
DEFAULT_CONFIG_YAML = """
servers:
  - name: SNDEMO
    type: synapse
    host: snpoc-synapse-ondemand.sql.azuresynapse.net
    database: SNDEMO
    username: ""
    password: ""
    driver: ODBC Driver 18 for SQL Server
  - name: UGLAND
    type: synapse
    host: shipnetuglandanalytics-ondemand.sql.azuresynapse.net
    database: UglandProdDB
    username: ""
    password: ""
    driver: ODBC Driver 18 for SQL Server
  - name: LEMISSOLER
    type: synapse
    host: shipnetlemissoleranalytics-ondemand.sql.azuresynapse.net
    database: LEMISSOLER
    username: ""
    password: ""
    driver: ODBC Driver 18 for SQL Server
  - name: NavShipping
    type: synapse
    host: shipnetnavshipanalytics-ondemand.sql.azuresynapse.net
    database: NavShipping
    username: ""
    password: ""
    driver: ODBC Driver 18 for SQL Server
  - name: EPIC
    type: synapse
    host: shipnetepicanalytics-ondemand.sql.azuresynapse.net
    database: EPIC
    username: ""
    password: ""
    driver: ODBC Driver 18 for SQL Server
  - name: Emarat
    type: synapse
    host: shipnetemaratanalytics-ondemand.sql.azuresynapse.net
    database: Emarat
    username: ""
    password: ""
    driver: ODBC Driver 18 for SQL Server
  - name: Unigas
    type: synapse
    host: shipnetunigasanalytics-ondemand.sql.azuresynapse.net
    database: Unigas
    username: ""
    password: ""
    driver: ODBC Driver 18 for SQL Server

options:
  allow_unsafe_sql: false
"""

CONFIG_PATH = os.getenv("HELIX_CONFIG", "helix_config.yaml")
DBAPI = os.getenv("HELIX_DBAPI", "pyodbc").lower()  # set to "pytds" on Streamlit Cloud

@dataclass
class ServerProfile:
    name: str
    type: str
    host: str
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    driver: str = "ODBC Driver 18 for SQL Server"

@dataclass
class AppConfig:
    servers: List[ServerProfile]
    options: Dict[str, Any]


def load_servers_from_secrets() -> List[ServerProfile]:
    """Read server profiles from Streamlit Secrets if present.
    Expected shape:
    [servers.<NAME>]
    host = "..."
    database = "..."
    username = "..."
    password = "..."
    """
    try:
        sec = st.secrets.get("servers", {})
    except Exception:
        sec = {}
    servers: List[ServerProfile] = []
    if isinstance(sec, dict):
        for name, cfg in sec.items():
            servers.append(ServerProfile(
                name=name,
                type=cfg.get("type", "synapse"),
                host=cfg.get("host", ""),
                database=cfg.get("database", ""),
                username=cfg.get("username", ""),
                password=cfg.get("password", ""),
                driver=cfg.get("driver", "ODBC Driver 18 for SQL Server"),
            ))
    return servers


def load_config(path: str) -> AppConfig:
    cfg = None
    if yaml is None:
        # If PyYAML missing, just rely on Secrets or defaults
        cfg = json.loads(json.dumps({}))
    else:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CONFIG_YAML)
            st.info(f"No config found. Created example at {path}.")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    yaml_servers = [ServerProfile(**s) for s in (cfg or {}).get("servers", [])]
    options = (cfg or {}).get("options", {"allow_unsafe_sql": False})

    secret_servers = load_servers_from_secrets()
    servers = secret_servers if secret_servers else yaml_servers
    return AppConfig(servers=servers, options=options)


# ---------------------------
# SQL Connectivity Utilities
# ---------------------------

from sqlalchemy.engine import Engine

def get_sqlalchemy_url(profile: ServerProfile) -> str:
    user = (profile.username or "").strip()
    pwd  = (profile.password or "").strip()
    host = (profile.host or "").strip()
    db   = (profile.database or "").strip()
    if not (host and db):
        return ""

    if DBAPI == "pytds":
        # Pure-Python driver for Cloud (no ODBC dependency)
        # encrypt=yes is default for Synapse; specify port 1433
        if not (user and pwd):
            return ""
        return f"mssql+pytds://{quote_plus(user)}:{quote_plus(pwd)}@{host}:1433/{db}?encrypt=yes"
    else:
        # Local: pyodbc via ODBC Driver 18
        if not (user and pwd):
            # allow demo mode if no creds locally
            return ""
        driver = profile.driver or "ODBC Driver 18 for SQL Server"
        params = (
            f"DRIVER={{{driver}}};"
            f"SERVER={host};DATABASE={db};"
            f"UID={user};PWD={pwd};TrustServerCertificate=no;Encrypt=yes"
        )
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(params)}"


@st.cache_data(show_spinner=False)
def run_sql(profile: ServerProfile, sql: str, params: Optional[Dict[str, Any]] = None, limit: int = 100000) -> pd.DataFrame:
    if create_engine is None:
        st.error("SQLAlchemy not installed. `pip install sqlalchemy`.")
        return pd.DataFrame()

    if not sql.strip().lower().startswith("select") and not st.session_state.get("allow_unsafe_sql", False):
        st.error("Only SELECT statements are allowed. Enable unsafe SQL in sidebar if you know what you're doing.")
        return pd.DataFrame()

    url = get_sqlalchemy_url(profile)
    if not url:
        st.warning("No database credentials detected for selected profile. Falling back to demo data.")
        return demo_dataset()

    engine: Engine = create_engine(url)
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if len(df) > limit:
        df = df.head(limit)
    return df


# ---------------------------
# Demo Dataset (when no creds)
# ---------------------------

def demo_dataset() -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    vessels = ["LNG Aurora", "LNG Borealis", "LNG Caledonia", "LNG Dorado"]
    trades = ["Spot", "Time Charter", "COA"]
    ports = ["Ras Laffan", "Zeebrugge", "Gate", "South Hook", "Sines", "Montoir"]

    rows = []
    for d in dates:
        for v in vessels:
            rows.append({
                "date": d,
                "vessel": v,
                "trade_type": np.random.choice(trades, p=[0.5, 0.3, 0.2]),
                "load_port": np.random.choice(ports),
                "discharge_port": np.random.choice(ports),
                "tce_usd_day": np.random.normal(95000, 15000),
                "utilization": np.clip(np.random.normal(0.85, 0.08), 0, 1),
                "demurrage_usd": max(0, np.random.normal(20000, 40000)),
                "cargo_mmbtu": np.random.normal(3300, 300),
            })
    df = pd.DataFrame(rows)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


# ---------------------------
# Dashboard Models
# ---------------------------

class TileSpec(BaseModel):
    title: str = Field(...)
    chart_type: str = Field("line", description="line|bar|area|scatter|pie|table")
    dataset_sql: Optional[str] = Field(None, description="SQL used to build the tile's dataset")
    x: Optional[str] = None
    y: Optional[str] = None
    group: Optional[str] = None
    agg: Optional[str] = Field("sum", description="sum|mean|min|max|count")
    filters: Dict[str, Any] = Field(default_factory=dict)

    @validator("chart_type")
    def _check_chart(cls, v):
        allowed = {"line", "bar", "area", "scatter", "pie", "table"}
        if v not in allowed:
            raise ValueError(f"chart_type must be one of {allowed}")
        return v


class DashboardSpec(BaseModel):
    title: str
    tiles: List[TileSpec] = Field(default_factory=list)


# ---------------------------
# UI Helpers
# ---------------------------

def dataset_from_sql_or_demo(profile: ServerProfile, sql: Optional[str]) -> pd.DataFrame:
    if sql and sql.strip():
        return run_sql(profile, sql)
    return demo_dataset()


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, val in filters.items():
        if col not in out.columns:
            continue
        if isinstance(val, (list, tuple, set)):
            out = out[out[col].isin(list(val))]
        elif isinstance(val, dict) and {"start", "end"} <= set(val.keys()):
            start = pd.to_datetime(val.get("start")) if val.get("start") else None
            end = pd.to_datetime(val.get("end")) if val.get("end") else None
            if start is not None:
                out = out[out[col] >= start]
            if end is not None:
                out = out[out[col] <= end]
        else:
            out = out[out[col] == val]
    return out


def aggregate(df: pd.DataFrame, x: Optional[str], y: Optional[str], group: Optional[str], agg: str) -> pd.DataFrame:
    if y is None:
        return df
    aggfunc = {"sum": "sum", "mean": "mean", "min": "min", "max": "max", "count": "count"}.get(agg, "sum")
    keys = [c for c in [x, group] if c]
    if keys:
        grouped = df.groupby(keys, dropna=False)[y].agg(aggfunc).reset_index()
    else:
        grouped = df[[y]].agg(aggfunc).to_frame().T
    return grouped


def render_chart(df: pd.DataFrame, spec: TileSpec):
    if spec.chart_type == "table":
        st.dataframe(df, use_container_width=True)
        return

    if spec.chart_type == "pie":
        if spec.group and spec.y:
            aggdf = df.groupby(spec.group, dropna=False)[spec.y].sum().reset_index()
            fig = px.pie(aggdf, names=spec.group, values=spec.y, title=spec.title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Pie requires 'group' and 'y' fields.")
        return

    plotdf = aggregate(df, spec.x, spec.y, spec.group, spec.agg)
    if spec.chart_type == "line":
        fig = px.line(plotdf, x=spec.x, y=spec.y, color=spec.group, markers=True, title=spec.title)
    elif spec.chart_type == "bar":
        fig = px.bar(plotdf, x=spec.x or spec.group, y=spec.y, color=spec.group, barmode="group", title=spec.title)
    elif spec.chart_type == "area":
        fig = px.area(plotdf, x=spec.x, y=spec.y, color=spec.group, title=spec.title)
    else:  # scatter
        fig = px.scatter(plotdf, x=spec.x, y=spec.y, color=spec.group, title=spec.title)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Sidebar - Server & Options
# ---------------------------

def sidebar_controls(config: AppConfig) -> Tuple[ServerProfile, bool]:
    st.sidebar.title("Helix Commercial Analyzer")
    st.sidebar.caption("Select your server and options")

    if not config.servers:
        st.sidebar.warning("No servers configured. Add via helix_config.yaml or Streamlit Secrets.")
    profile_names = [s.name for s in config.servers] or ["Demo (no DB)"]
    choice = st.sidebar.selectbox("Server profile", profile_names, index=0)
    if config.servers:
        profile = next(s for s in config.servers if s.name == choice)
    else:
        profile = ServerProfile(name="demo", type="demo", host="", database="demo")

    allow_unsafe_default = bool(config.options.get("allow_unsafe_sql", False))
    st.session_state["allow_unsafe_sql"] = st.sidebar.toggle("Allow unsafe SQL (non-SELECT)", value=allow_unsafe_default)

    st.sidebar.markdown("---")
    st.sidebar.write("**Save / Load dashboard**")
    dl_spec = st.session_state.get("dashboard_spec")
    if dl_spec:
        st.sidebar.download_button("Download dashboard JSON", data=json.dumps(dl_spec, indent=2), file_name=f"{dl_spec['title']}.json", mime="application/json")
    uploaded = st.sidebar.file_uploader("Load dashboard JSON", type=["json"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            spec = json.load(uploaded)
            DashboardSpec(**spec)  # validate
            st.session_state["dashboard_spec"] = spec
            st.sidebar.success("Dashboard loaded.")
        except Exception as e:
            st.sidebar.error(f"Invalid dashboard JSON: {e}")

    return profile, st.session_state.get("allow_unsafe_sql", False)


# ---------------------------
# Page Sections
# ---------------------------

def section_query(profile: ServerProfile):
    st.header("Query")
    st.caption("Run a SELECT query or use the demo dataset if not connected.")
    default_sql = st.session_state.get("last_sql", "SELECT TOP 100 * FROM sys.tables")
    sql = st.text_area("SQL (SELECT-only by default)", value=default_sql, height=140)
    cols = st.columns([1,1,1,2])
    with cols[0]:
        limit = st.number_input("Row limit", 1000, 500000, 50000, step=1000)
    with cols[1]:
        run_btn = st.button("Run query", type="primary")
    with cols[2]:
        use_demo = st.toggle("Use demo data", value=False)
    with cols[3]:
        st.write("")

    if run_btn:
        st.session_state["last_sql"] = sql
        df = demo_dataset() if use_demo else run_sql(profile, sql, limit=limit)
        st.session_state["last_df"] = df

    df = st.session_state.get("last_df", demo_dataset())
    st.dataframe(df.head(1000), use_container_width=True)
    st.caption(f"Rows in memory: {len(df):,}")
    return df


def section_chart_builder(df: pd.DataFrame) -> TileSpec:
    st.header("Chart Builder")
    st.caption("Select fields and aggregations to build a chart tile.")

    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    title = st.text_input("Tile title", value="My Chart")
    chart_type = st.selectbox("Chart type", ["line", "bar", "area", "scatter", "pie", "table"], index=0)

    x = st.selectbox("X axis (or None)", [None] + cols, index=(cols.index("date")+1) if "date" in cols else 0)
    y = st.selectbox("Y (metric; for pie = values)", [None] + numeric_cols, index=(numeric_cols.index("tce_usd_day")+1) if "tce_usd_day" in numeric_cols else 0)
    group = st.selectbox("Group / series (optional; for pie = names)", [None] + cols, index=(cols.index("vessel")+1) if "vessel" in cols else 0)
    agg = st.selectbox("Aggregation", ["sum", "mean", "min", "max", "count"], index=0)

    st.subheader("Filters")
    st.caption("Apply quick filters (optional)")
    filters: Dict[str, Any] = {}

    with st.expander("Date filter", expanded=False):
        if datetime_cols:
            date_col = st.selectbox("Date column", datetime_cols, index=0)
            start = st.date_input("Start", value=None, format="YYYY-MM-DD")
            end = st.date_input("End", value=None, format="YYYY-MM-DD")
            if start:
                filters[date_col] = {"start": str(start)}
            if end:
                filters.setdefault(date_col, {}).update({"end": str(end)})
        else:
            st.info("No datetime columns detected.")

    with st.expander("Categorical filters", expanded=False):
        for c in [c for c in cols if c not in numeric_cols + datetime_cols]:
            vals = sorted([str(v) for v in df[c].dropna().unique().tolist()])
            if not vals:
                continue
            selected = st.multiselect(f"{c}", vals)
            if selected:
                filters[c] = selected

    build = st.button("Add tile to dashboard", type="primary")
    current_sql = st.session_state.get("last_sql")

    spec = TileSpec(title=title or "Untitled", chart_type=chart_type, dataset_sql=current_sql, x=x, y=y, group=group, agg=agg, filters=filters)

    if build:
        dash: DashboardSpec = DashboardSpec(**st.session_state.get("dashboard_spec", {"title": "Commercial Dashboard", "tiles": []}))
        dash.tiles.append(spec)
        st.session_state["dashboard_spec"] = json.loads(dash.json())
        st.success("Tile added.")

    st.subheader("Preview")
    df2 = apply_filters(df, filters)
    render_chart(df2, spec)

    return spec


def section_dashboard(profile: ServerProfile):
    st.header("Dashboard")
    spec = st.session_state.get("dashboard_spec", {"title": "Commercial Dashboard", "tiles": []})
    dash = DashboardSpec(**spec)
    st.text_input("Dashboard title", value=dash.title, key="dash_title")
    dash.title = st.session_state.get("dash_title", dash.title)

    tiles = dash.tiles
    if not tiles:
        st.info("No tiles yet. Build one above.")
    else:
        for i, t in enumerate(tiles):
            cols = st.columns(2)
            with cols[0]:
                st.subheader(t.title)
            with cols[1]:
                if st.button(f"Delete", key=f"del_{i}"):
                    tiles.pop(i)
                    st.session_state["dashboard_spec"] = json.loads(DashboardSpec(title=dash.title, tiles=tiles).json())
                    st.rerun()
            df = dataset_from_sql_or_demo(profile, t.dataset_sql)
            df = apply_filters(df, t.filters)
            render_chart(df, t)


# ---------------------------
# KPI Rail (example LNG/commercial KPIs)
# ---------------------------

def kpi_rail(df: pd.DataFrame):
    st.header("KPIs")
    avg_tce = df["tce_usd_day"].mean() if "tce_usd_day" in df else np.nan
    util = df["utilization"].mean() if "utilization" in df else np.nan
    dem = df["demurrage_usd"].sum() if "demurrage_usd" in df else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg TCE (USD/day)", f"{avg_tce:,.0f}" if pd.notna(avg_tce) else "—")
    c2.metric("Fleet Utilization", f"{util*100:,.1f}%" if pd.notna(util) else "—")
    c3.metric("Total Demurrage (USD)", f"{dem:,.0f}" if pd.notna(dem) else "—")


# ---------------------------
# Main App
# ---------------------------

def main():
    st.set_page_config(page_title="Helix Commercial Analyzer", layout="wide")

    config = load_config(CONFIG_PATH)
    profile, _ = sidebar_controls(config)

    df = section_query(profile)
    kpi_rail(df)
    section_chart_builder(df)
    section_dashboard(profile)


if __name__ == "__main__":
    main()
