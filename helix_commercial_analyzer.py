"""
Helix Commercial Analyzer — Streamlit MVP
- COMM/VW picker with preview + “Use in Query”
- Per-server caching & server-change reset
- Plotly: config= usage (no deprecated kwargs)
- Pydantic v2 (@field_validator)
- Connection diagnostics
- Reads server profiles from:
    1) Secrets UI (preferred), or
    2) .streamlit/secrets.toml in the repo (fallback), else
    3) helix_config.yaml
- TLS for pytds: pass cafile/validate_host via create_engine(connect_args=...)
"""

from __future__ import annotations

import json, os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pydantic import BaseModel, Field, field_validator
from urllib.parse import quote_plus

import sqlalchemy as sa
from sqlalchemy.engine import Engine

# NEW: certifi gives us a portable CA bundle path for TLS
import certifi

# Optional YAML for local dev config
try:
    import yaml
except Exception:
    yaml = None

# TOML parser for fallback .streamlit/secrets.toml
try:
    import tomllib  # Py 3.11+
except Exception:
    tomllib = None

try:
    from sqlalchemy import create_engine, text
except Exception:
    create_engine = None
    text = None

# -------------------------- Config --------------------------

CONFIG_PATH = os.getenv("HELIX_CONFIG", "helix_config.yaml")

def _get_dbapi() -> str:
    # Prefer Secrets UI; then env var; default pyodbc
    try:
        sec_val = st.secrets.get("HELIX_DBAPI", None)
        if isinstance(sec_val, str) and sec_val.strip():
            return sec_val.strip().lower()
    except Exception:
        pass
    return os.getenv("HELIX_DBAPI", "pyodbc").lower()

DBAPI = _get_dbapi()

DEFAULT_CONFIG_YAML = """
servers:
  - name: SNDEMO
    type: synapse
    host: snpoc-synapse-ondemand.sql.azuresynapse.net
    database: SNDEMO
    username: ""
    password: ""
    driver: ODBC Driver 18 for SQL Server
options:
  allow_unsafe_sql: false
"""

# -------------------------- Models --------------------------

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

# -------------------------- Load profiles --------------------------

def _profiles_from_secrets_ui() -> List[ServerProfile]:
    try:
        sec = st.secrets.get("servers", {})
    except Exception:
        sec = {}
    out: List[ServerProfile] = []
    if isinstance(sec, dict):
        for name, cfg in sec.items():
            out.append(ServerProfile(
                name=name,
                type=cfg.get("type", "synapse"),
                host=cfg.get("host", ""),
                database=cfg.get("database", ""),
                username=cfg.get("username", ""),
                password=cfg.get("password", ""),
                driver=cfg.get("driver", "ODBC Driver 18 for SQL Server"),
            ))
    return out

def _profiles_from_secrets_file() -> List[ServerProfile]:
    path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(path) or tomllib is None:
        return []
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        servers = data.get("servers", {})
        out: List[ServerProfile] = []
        if isinstance(servers, dict):
            for name, cfg in servers.items():
                out.append(ServerProfile(
                    name=name,
                    type=cfg.get("type", "synapse"),
                    host=cfg.get("host", ""),
                    database=cfg.get("database", ""),
                    username=cfg.get("username", ""),
                    password=cfg.get("password", ""),
                    driver=cfg.get("driver", "ODBC Driver 18 for SQL Server"),
                ))
        return out
    except Exception:
        return []

def _profiles_from_yaml(path: str) -> Tuple[List[ServerProfile], Dict[str, Any]]:
    cfg = None
    if yaml is not None:
        if not os.path.exists(path):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(DEFAULT_CONFIG_YAML)
            except Exception:
                pass
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = None
    servers = [ServerProfile(**s) for s in (cfg or {}).get("servers", [])]
    options = (cfg or {}).get("options", {"allow_unsafe_sql": False})
    return servers, options

def load_config(path: str) -> AppConfig:
    s1 = _profiles_from_secrets_ui()
    s2 = _profiles_from_secrets_file() if not s1 else []
    y_servers, y_opts = _profiles_from_yaml(path)

    if s1:
        source = "secrets-ui"; servers = s1; options = y_opts
    elif s2:
        source = "secrets-file"; servers = s2; options = y_opts
    else:
        source = "yaml"; servers = y_servers; options = y_opts

    # Diagnostics (safe: names only)
    st.session_state["profiles_source"] = source
    st.session_state["profiles_from_secrets_ui"] = [s.name for s in s1]
    st.session_state["profiles_from_secrets_file"] = [s.name for s in s2]
    st.session_state["profiles_from_yaml"] = [s.name for s in y_servers]

    return AppConfig(servers=servers, options=options)

# -------------------------- Connectivity --------------------------

def get_sqlalchemy_url(p: ServerProfile) -> str:
    user = (p.username or "").strip()
    pwd  = (p.password or "").strip()
    host = (p.host or "").strip()
    db   = (p.database or "").strip()
    if not (host and db):
        return ""
    if DBAPI == "pytds":
        # Base URL without TLS args; we'll pass TLS via connect_args in create_engine.
        if not (user and pwd):
            return ""
        return f"mssql+pytds://{quote_plus(user)}:{quote_plus(pwd)}@{host}:1433/{db}"
    else:
        if not (user and pwd):
            return ""
        driver = p.driver or "ODBC Driver 18 for SQL Server"
        params = (
            f"DRIVER={{{driver}}};SERVER={host};DATABASE={db};"
            f"UID={user};PWD={pwd};TrustServerCertificate=no;Encrypt=yes"
        )
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(params)}"

def connection_key(p: ServerProfile) -> str:
    return f"{p.name}|{p.host}|{p.database}|{DBAPI}"

@st.cache_resource(show_spinner=False)
def get_engine_for_url(url: str) -> Optional[Engine]:
    if create_engine is None or not url:
        return None
    # **TLS for pytds** — pass DBAPI args explicitly (recommended by SQLAlchemy)
    # so we don't rely on the dialect parsing URL query params. :contentReference[oaicite:1]{index=1}
    if DBAPI == "pytds":
        return create_engine(
            url,
            connect_args={
                "cafile": certifi.where(),   # enables TLS in python-tds :contentReference[oaicite:2]{index=2}
                "validate_host": True,       # hostname verification (default True)
            },
            pool_pre_ping=True,
        )
    return create_engine(url, pool_pre_ping=True)

def try_connect(engine: Optional[Engine]) -> Tuple[bool, str]:
    if engine is None:
        return False, "No engine (missing URL/credentials)."
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True, ""
    except Exception as e:
        msg = str(e)
        for k in ["UID", "PWD", "username", "password"]:
            msg = msg.replace(k, "***")
        return False, msg[:500]

@st.cache_data(show_spinner=False)
def run_sql_cached(url: str, sql: str, params: Optional[Dict[str, Any]], limit: int) -> pd.DataFrame:
    engine: Engine = get_engine_for_url(url)  # reuse same connect_args
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if len(df) > limit:
        df = df.head(limit)
    return df

def run_sql(profile: ServerProfile, sql: str, params: Optional[Dict[str, Any]] = None, limit: int = 100000) -> pd.DataFrame:
    if create_engine is None:
        st.error("SQLAlchemy missing. Install dependencies.")
        return pd.DataFrame()
    if not sql.strip().lower().startswith("select") and not st.session_state.get("allow_unsafe_sql", False):
        st.error("Only SELECT statements are allowed.")
        return pd.DataFrame()
    url = get_sqlalchemy_url(profile)
    if not url:
        st.warning("No database credentials detected for selected profile. Using demo data.")
        return demo_dataset()
    return run_sql_cached(url, sql, params, limit)

# -------------------------- COMM/VW listing (URL-based cache) --------------------------

@st.cache_data(ttl=300, show_spinner=False)
def list_comm_vw_objects(url: str) -> pd.DataFrame:
    if not url:
        return pd.DataFrame(columns=["schema_name", "object_name", "object_type"])
    engine = get_engine_for_url(url)
    if engine is None:
        return pd.DataFrame(columns=["schema_name", "object_name", "object_type"])
    try:
        dialect = engine.dialect.name.lower()
        with engine.connect() as conn:
            if dialect == "mssql":
                sql = sa.text("""
                    SELECT s.name AS schema_name, t.name AS object_name, 'BASE TABLE' AS object_type
                    FROM sys.tables t JOIN sYS.schemas s ON s.schema_id = t.schema_id
                    WHERE (LOWER(t.name) LIKE 'comm%' AND LOWER(t.name) NOT LIKE 'commercial%')
                       OR LOWER(t.name) LIKE 'vw%'
                    UNION ALL
                    SELECT s.name, et.name, 'EXTERNAL TABLE'
                    FROM sys.external_tables et JOIN sys.schemas s ON s.schema_id = et.schema_id
                    WHERE (LOWER(et.name) LIKE 'comm%' AND LOWER(et.name) NOT LIKE 'commercial%')
                       OR LOWER(et.name) LIKE 'vw%'
                    UNION ALL
                    SELECT s.name, v.name, 'VIEW'
                    FROM sys.views v JOIN sys.schemas s ON s.schema_id = v.schema_id
                    WHERE (LOWER(v.name) LIKE 'comm%' AND LOWER(v.name) NOT LIKE 'commercial%')
                       OR LOWER(v.name) LIKE 'vw%'
                    ORDER BY schema_name, object_name;
                """)
            else:
                sql = sa.text("""
                    SELECT table_schema AS schema_name, table_name AS object_name, table_type AS object_type
                    FROM information_schema.tables
                    WHERE table_type IN ('BASE TABLE','VIEW','EXTERNAL TABLE')
                      AND ((LOWER(table_name) LIKE 'comm%' AND LOWER(table_name) NOT LIKE 'commercial%')
                           OR LOWER(table_name) LIKE 'vw%')
                    ORDER BY table_schema, table_name;
                """)
            return pd.read_sql(sql, conn)
    except Exception:
        return pd.DataFrame(columns=["schema_name", "object_name", "object_type"])

@st.cache_data(ttl=120, show_spinner=False)
def preview_object(url: str, schema: str, name: str, limit: int) -> pd.DataFrame:
    if not url:
        return pd.DataFrame()
    engine = get_engine_for_url(url)
    if engine is None:
        return pd.DataFrame()
    dialect = engine.dialect.name.lower()
    with engine.connect() as conn:
        if dialect == "mssql":
            sql = sa.text(f"SELECT TOP {int(limit)} * FROM [{schema}].[{name}]")
        else:
            prep = sa.sql.compiler.IdentifierPreparer(engine.dialect)
            fq = f"{prep.quote_identifier(schema)}.{prep.quote_identifier(name)}"
            sql = sa.text(f"SELECT * FROM {fq} LIMIT {int(limit)}")
        return pd.read_sql(sql, conn)

# -------------------------- Demo data --------------------------

def demo_dataset() -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    vessels = ["LNG Aurora", "LNG Borealis", "LNG Caledonia", "LNG Dorado"]
    trades = ["Spot", "Time Charter", "COA"]
    ports = ["Ras Laffan", "Zeebrugge", "Gate", "South Hook", "Sines", "Montoir"]
    rows = []
    for d in dates:
        for v in vessels:
            rows.append({
                "date": d, "vessel": v,
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

# -------------------------- Dashboard models --------------------------

class TileSpec(BaseModel):
    title: str = Field(...)
    chart_type: str = Field("line")
    dataset_sql: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    group: Optional[str] = None
    agg: Optional[str] = Field("sum")
    filters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("chart_type")
    @classmethod
    def _check_chart(cls, v: str) -> str:
        allowed = {"line","bar","area","scatter","pie","table"}
        if v not in allowed:
            raise ValueError(f"chart_type must be one of {allowed}")
        return v

class DashboardSpec(BaseModel):
    title: str
    tiles: List[TileSpec] = Field(default_factory=list)

# -------------------------- UI helpers --------------------------

def dataset_from_sql_or_demo(profile: ServerProfile, sql: Optional[str]) -> pd.DataFrame:
    return run_sql(profile, sql) if (sql and sql.strip()) else demo_dataset()

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, val in filters.items():
        if col not in out.columns:
            continue
        if isinstance(val, (list, tuple, set)):
            out = out[out[col].isin(list(val))]
        elif isinstance(val, dict) and {"start","end"} <= set(val.keys()):
            start = pd.to_datetime(val.get("start")) if val.get("start") else None
            end = pd.to_datetime(val.get("end")) if val.get("end") else None
            if start is not None: out = out[out[col] >= start]
            if end   is not None: out = out[out[col] <= end]
        else:
            out = out[out[col] == val]
    return out

def aggregate(df: pd.DataFrame, x: Optional[str], y: Optional[str], group: Optional[str], agg: str) -> pd.DataFrame:
    if y is None: return df
    aggfunc = {"sum":"sum","mean":"mean","min":"min","max":"max","count":"count"}.get(agg,"sum")
    keys = [c for c in [x, group] if c]
    return df.groupby(keys, dropna=False)[y].agg(aggfunc).reset_index() if keys else df[[y]].agg(aggfunc).to_frame().T

def render_chart(df: pd.DataFrame, spec: TileSpec):
    cfg = {"responsive": True, "displaylogo": False}
    if spec.chart_type == "table":
        st.dataframe(df)
        return
    if spec.chart_type == "pie":
        if spec.group and spec.y:
            aggdf = df.groupby(spec.group, dropna=False)[spec.y].sum().reset_index()
            fig = px.pie(aggdf, names=spec.group, values=spec.y, title=spec.title)
            st.plotly_chart(fig, config=cfg)
        else:
            st.warning("Pie requires 'group' and 'y'.")
        return
    plotdf = aggregate(df, spec.x, spec.y, spec.group, spec.agg)
    if spec.chart_type == "line":
        fig = px.line(plotdf, x=spec.x, y=spec.y, color=spec.group, markers=True, title=spec.title)
    elif spec.chart_type == "bar":
        fig = px.bar(plotdf, x=spec.x or spec.group, y=spec.y, color=spec.group, barmode="group", title=spec.title)
    elif spec.chart_type == "area":
        fig = px.area(plotdf, x=spec.x, y=spec.y, color=spec.group, title=spec.title)
    else:
        fig = px.scatter(plotdf, x=spec.x, y=spec.y, color=spec.group, title=spec.title)
    fig.update_layout(margin=dict(l=8,r=8,t=48,b=8))
    st.plotly_chart(fig, config=cfg)

# -------------------------- Sidebar --------------------------

def sidebar_controls(config: AppConfig) -> Tuple[ServerProfile, bool, Optional[Engine], str]:
    st.sidebar.title("Helix Commercial Analyzer")
    st.sidebar.caption("Select your server and options")
    if not config.servers:
        st.sidebar.warning("No servers configured (add Secrets or .streamlit/secrets.toml).")
    names = [s.name for s in config.servers] or ["Demo (no DB)"]
    choice = st.sidebar.selectbox("Server profile", names, index=0)
    profile = next((s for s in config.servers if s.name == choice), ServerProfile("demo","demo","",""))
    allow_unsafe_default = bool(config.options.get("allow_unsafe_sql", False))
    st.session_state["allow_unsafe_sql"] = st.sidebar.toggle("Allow unsafe SQL (non-SELECT)", value=allow_unsafe_default)

    # ---- Connection Status + Diagnostics ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("Connection status")
    st.sidebar.caption(f"DBAPI: **{DBAPI}**")

    src = st.session_state.get("profiles_source")
    st.sidebar.caption(f"Profiles source: **{src}**")
    with st.sidebar.expander("Profiles from Secrets UI"):
        st.write(", ".join(st.session_state.get("profiles_from_secrets_ui", [])) or "—")
    with st.sidebar.expander("Profiles from Secrets FILE"):
        st.write(", ".join(st.session_state.get("profiles_from_secrets_file", [])) or "—")
    with st.sidebar.expander("Profiles from YAML"):
        st.write(", ".join(st.session_state.get("profiles_from_yaml", [])) or "—")

    st.sidebar.write(
        "Fields present — "
        f"Host: {'✅' if profile.host else '—'} · "
        f"DB: {'✅' if profile.database else '—'} · "
        f"User: {'✅' if profile.username else '—'} · "
        f"Pwd: {'✅' if profile.password else '—'}"
    )

    url = get_sqlalchemy_url(profile)
    st.sidebar.write(f"Credentials present: **{'Yes' if bool(url) else 'No'}**")
    if DBAPI == "pytds" and url:
        st.sidebar.caption(f"TLS (pytds) CA: `{certifi.where()}`")

    engine = get_engine_for_url(url) if url else None
    ok, err = try_connect(engine)
    if ok:
        st.sidebar.success("Connected")
    else:
        st.sidebar.info("Not connected")
        if err:
            st.sidebar.code(err)

    # Save/Load dashboard
    st.sidebar.markdown("---")
    st.sidebar.write("**Save / Load dashboard**")
    dl = st.session_state.get("dashboard_spec")
    if dl:
        st.sidebar.download_button("Download dashboard JSON", data=json.dumps(dl, indent=2),
                                   file_name=f"{dl['title']}.json", mime="application/json")
    up = st.sidebar.file_uploader("Load dashboard JSON", type=["json"])
    if up is not None:
        try:
            spec = json.load(up)
            DashboardSpec(**spec)
            st.session_state["dashboard_spec"] = spec
            st.sidebar.success("Dashboard loaded.")
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")

    return profile, st.session_state.get("allow_unsafe_sql", False), engine, url

# -------------------------- Page sections --------------------------

def section_query(profile: ServerProfile):
    st.header("Query")
    st.caption("Run a SELECT query or use the demo dataset if not connected.")
    default_sql = st.session_state.get("last_sql", "SELECT TOP 100 * FROM sys.tables")
    sql = st.text_area("SQL (SELECT-only by default)", value=default_sql, height=140)
    c1, c2, c3, _ = st.columns([1,1,1,2])
    with c1:
        limit = st.number_input("Row limit", 1000, 500000, 50000, step=1000)
    with c2:
        run_btn = st.button("Run query")
    with c3:
        use_demo = st.toggle("Use demo data", value=False)

    if run_btn:
        st.session_state["last_sql"] = sql
        df = demo_dataset() if use_demo else run_sql(profile, sql, limit=limit)
        st.session_state["last_df"] = df

    df = st.session_state.get("last_df", demo_dataset())
    st.dataframe(df)
    st.caption(f"Rows in memory: {len(df):,}")
    return df

def section_chart_builder(df: pd.DataFrame) -> TileSpec:
    st.header("Chart Builder")
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    title = st.text_input("Tile title", value="My Chart")
    chart_type = st.selectbox("Chart type", ["line","bar","area","scatter","pie","table"], index=0)
    x = st.selectbox("X axis (or None)", [None]+cols, index=(cols.index("date")+1) if "date" in cols else 0)
    y = st.selectbox("Y (metric; for pie = values)", [None]+numeric_cols, index=(numeric_cols.index("tce_usd_day")+1) if "tce_usd_day" in numeric_cols else 0)
    group = st.selectbox("Group / series (optional; for pie = names)", [None]+cols, index=(cols.index("vessel")+1) if "vessel" in cols else 0)
    agg = st.selectbox("Aggregation", ["sum","mean","min","max","count"], index=0)

    st.subheader("Filters")
    filters: Dict[str, Any] = {}
    with st.expander("Date filter", expanded=False):
        if dt_cols:
            dc = st.selectbox("Date column", dt_cols, index=0)
            start = st.date_input("Start", value=None, format="YYYY-MM-DD")
            end = st.date_input("End", value=None, format="YYYY-MM-DD")
            if start: filters[dc] = {"start": str(start)}
            if end:   filters.setdefault(dc, {}).update({"end": str(end)})
        else:
            st.info("No datetime columns detected.")
    with st.expander("Categorical filters", expanded=False):
        for c in [c for c in cols if c not in numeric_cols + dt_cols]:
            vals = sorted([str(v) for v in df[c].dropna().unique().tolist()])
            if not vals: continue
            selected = st.multiselect(c, vals)
            if selected: filters[c] = selected

    add = st.button("Add tile to dashboard")
    current_sql = st.session_state.get("last_sql")
    spec = TileSpec(title=title or "Untitled", chart_type=chart_type, dataset_sql=current_sql, x=x, y=y, group=group, agg=agg, filters=filters)

    if add:
        dash: DashboardSpec = DashboardSpec(**st.session_state.get("dashboard_spec", {"title":"Commercial Dashboard","tiles":[]}))
        dash.tiles.append(spec)
        st.session_state["dashboard_spec"] = json.loads(dash.model_dump_json())
        st.success("Tile added.")

    st.subheader("Preview")
    render_chart(apply_filters(df, filters), spec)
    return spec

def section_dashboard(profile: ServerProfile):
    st.header("Dashboard")
    spec = st.session_state.get("dashboard_spec", {"title":"Commercial Dashboard","tiles":[]})
    dash = DashboardSpec(**spec)
    st.text_input("Dashboard title", value=dash.title, key="dash_title")
    dash.title = st.session_state.get("dash_title", dash.title)

    if not dash.tiles:
        st.info("No tiles yet. Build one above.")
    else:
        for i, t in enumerate(dash.tiles):
            c1, c2 = st.columns(2)
            with c1: st.subheader(t.title)
            with c2:
                if st.button("Delete", key=f"del_{i}"):
                    dash.tiles.pop(i)
                    st.session_state["dashboard_spec"] = json.loads(DashboardSpec(title=dash.title, tiles=dash.tiles).model_dump_json())
                    st.rerun()
            df = dataset_from_sql_or_demo(profile, t.dataset_sql)
            render_chart(apply_filters(df, t.filters), t)

# -------------------------- KPIs --------------------------

def kpi_rail(df: pd.DataFrame):
    st.header("KPIs")
    avg_tce = df["tce_usd_day"].mean() if "tce_usd_day" in df else np.nan
    util = df["utilization"].mean() if "utilization" in df else np.nan
    dem = df["demurrage_usd"].sum() if "demurrage_usd" in df else np.nan
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg TCE (USD/day)", f"{avg_tce:,.0f}" if pd.notna(avg_tce) else "—")
    c2.metric("Fleet Utilization", f"{util*100:,.1f}%" if pd.notna(util) else "—")
    c3.metric("Total Demurrage (USD)", f"{dem:,.0f}" if pd.notna(dem) else "—")

# -------------------------- Main --------------------------

def main():
    st.set_page_config(page_title="Helix Commercial Analyzer", layout="wide")

    config = load_config(CONFIG_PATH)
    profile, _, engine, url = sidebar_controls(config)

    # Reset state when server changes
    cur_key = connection_key(profile)
    prev_key = st.session_state.get("active_profile_key")
    if prev_key != cur_key:
        st.session_state["active_profile_key"] = cur_key
        st.session_state.pop("last_df", None)
        st.session_state.pop("last_sql", None)
        st.cache_data.clear()

    # COMM/VW picker (URL-based)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Choose COMM/VW table/view")
    objs = list_comm_vw_objects(url)
    if objs.empty:
        st.sidebar.info("No objects matching comm* (excluding commercial*) or vw*.")
    else:
        q = st.sidebar.text_input("Filter", placeholder="e.g. comm_register, vw_", value="").strip().lower()
        filt = objs[
            objs["schema_name"].str.lower().str.contains(q) | objs["object_name"].str.lower().str.contains(q)
        ] if q else objs
        if filt.empty:
            st.sidebar.info("No matches for current filter.")
        else:
            labels = [f"{r.schema_name}.{r.object_name}  •  {r.object_type}" for r in filt.itertuples(index=False)]
            chosen = st.sidebar.selectbox("Table/View", labels, index=0)
            idx = labels.index(chosen)
            sel = filt.iloc[idx]
            st.sidebar.caption(f"Selected: `{sel['schema_name']}.{sel['object_name']}`")
            n = st.sidebar.number_input("Preview rows", 10, 2000, 200, step=10)
            cA, cB = st.sidebar.columns(2)
            with cA:
                if st.button("Preview"):
                    st.dataframe(preview_object(url, sel["schema_name"], sel["object_name"], int(n)))
            with cB:
                if st.button("Use in Query"):
                    st.session_state["last_sql"] = f"SELECT TOP 1000 * FROM [{sel['schema_name']}].[{sel['object_name']}]"
                    try:
                        st.session_state["last_df"] = preview_object(url, sel["schema_name"], sel["object_name"], 1000)
                        st.success("Query editor updated from selection.")
                    except Exception as e:
                        st.warning(str(e)[:300])

    df = section_query(profile)
    kpi_rail(df)
    section_chart_builder(df)
    section_dashboard(profile)

if __name__ == "__main__":
    main()
