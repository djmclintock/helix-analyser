"""
Helix Commercial Analyzer — Lean Viewer
- LEFT: server + DB + COMM/VW picker (with preview)
- RIGHT: data viewer with global search + filters (no SQL editor, no chart builder)
- Dashboard renderer (read-only) — load JSON from sidebar
- Synapse-friendly: auto-fallback to master, DB dropdown, TLS for pytds via connect_args
"""

from __future__ import annotations

import json, os
from dataclasses import dataclass, replace
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pydantic import BaseModel, Field, field_validator
from urllib.parse import quote_plus

import sqlalchemy as sa
from sqlalchemy.engine import Engine
import certifi  # TLS CA bundle (pytds)

# Optional YAML for local fallback config
try:
    import yaml
except Exception:
    yaml = None

# TOML parser to read repo-side secrets if present
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
        val = st.secrets.get("HELIX_DBAPI", None)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
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

# -------------------------- Profile loading --------------------------

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

    st.session_state["profiles_source"] = source
    st.session_state["profiles_from_secrets_ui"] = [s.name for s in s1]
    st.session_state["profiles_from_secrets_file"] = [s.name for s in s2]
    st.session_state["profiles_from_yaml"] = [s.name for s in y_servers]
    return AppConfig(servers=servers, options=options)

# -------------------------- Connectivity --------------------------

def build_url(p: ServerProfile, db_override: Optional[str] = None) -> str:
    user = (p.username or "").strip()
    pwd  = (p.password or "").strip()
    host = (p.host or "").strip()
    db   = (db_override if db_override is not None else p.database or "").strip()
    if not (host and db):
        return ""
    if DBAPI == "pytds":
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

def connection_key(name: str, host: str, db: str) -> str:
    return f"{name}|{host}|{db}|{DBAPI}"

@st.cache_resource(show_spinner=False)
def get_engine_for_url(url: str) -> Optional[Engine]:
    if create_engine is None or not url:
        return None
    if DBAPI == "pytds":
        return create_engine(
            url,
            connect_args={"cafile": certifi.where(), "validate_host": True},
            pool_pre_ping=True,
        )
    return create_engine(url, pool_pre_ping=True)

def try_connect(url: str) -> Tuple[bool, str]:
    engine = get_engine_for_url(url) if url else None
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
    engine: Engine = get_engine_for_url(url)
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df.head(limit) if len(df) > limit else df

# -------------------------- Discovery + object APIs (URL-keyed caches) --------------------------

@st.cache_data(ttl=300, show_spinner=False)
def list_databases(url: str) -> List[str]:
    if not url:
        return []
    engine = get_engine_for_url(url)
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(sa.text("SELECT name FROM sys.databases WHERE database_id > 4 ORDER BY name;")).fetchall()
            return [r[0] for r in rows]
    except Exception:
        return []

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
                    FROM sys.tables t JOIN sys.schemas s ON s.schema_id = t.schema_id
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
def fetch_table_sample(url: str, schema: str, name: str, limit: int) -> pd.DataFrame:
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
        df = pd.read_sql(sql, conn)
    # Best-effort: parse datetimes
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_datetime(df[c], errors="ignore", utc=False)
            except Exception:
                pass
    return df

# -------------------------- Demo dataset --------------------------

def demo_dataset() -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = [{"date": d, "category": np.random.choice(["A","B","C"]), "value": np.random.randint(1, 100)} for d in dates]
    return pd.DataFrame(rows)

# -------------------------- UI helpers (filters + search) --------------------------

def apply_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    q = str(query).strip().lower()
    if not q:
        return df
    # string-match across all columns (safe stringify)
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        try:
            col_as_str = df[c].astype(str).str.lower()
            mask |= col_as_str.str.contains(q, na=False)
        except Exception:
            continue
    return df[mask]

def apply_filters(df: pd.DataFrame, dt_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    with st.expander("Filters", expanded=False):
        # Date range filters (per datetime column)
        if dt_cols:
            for dc in dt_cols:
                st.caption(f"Date filter — {dc}")
                col1, col2 = st.columns(2)
                with col1:
                    start = st.date_input(f"Start ({dc})", value=None, format="YYYY-MM-DD", key=f"start_{dc}")
                with col2:
                    end = st.date_input(f"End ({dc})", value=None, format="YYYY-MM-DD", key=f"end_{dc}")
                if start is not None:
                    out = out[out[dc] >= pd.to_datetime(start)]
                if end is not None:
                    out = out[out[dc] <= pd.to_datetime(end)]

        # Categorical quick filters (top 8 distinct values per column)
        cat_cols = [c for c in out.columns if out[c].dtype == "object" or pd.api.types.is_categorical_dtype(out[c])]
        if cat_cols:
            st.caption("Categorical filters")
            for c in cat_cols[:12]:
                vals = out[c].dropna().astype(str).unique().tolist()
                vals = sorted(vals)[:200]
                choice = st.multiselect(c, vals, key=f"cat_{c}")
                if choice:
                    out = out[out[c].astype(str).isin(choice)]
    return out

# -------------------------- Sidebar (server/DB/table) --------------------------

def sidebar_controls(config: AppConfig) -> Tuple[ServerProfile, str, Optional[Dict[str, str]]]:
    st.sidebar.title("Helix Commercial Analyzer")
    st.sidebar.caption("Select your server and source table")

    if not config.servers:
        st.sidebar.warning("No servers configured (add Secrets or .streamlit/secrets.toml).")

    names = [s.name for s in config.servers] or ["Demo (no DB)"]
    choice = st.sidebar.selectbox("Server profile", names, index=0)
    base_profile = next((s for s in config.servers if s.name == choice), ServerProfile("demo","demo","",""))

    # Diagnostics
    st.sidebar.markdown("---")
    st.sidebar.subheader("Connection status")
    st.sidebar.caption(f"DBAPI: **{DBAPI}**")
    st.sidebar.caption(f"Profiles source: **{st.session_state.get('profiles_source')}**")
    with st.sidebar.expander("Profiles from Secrets UI"):
        st.write(", ".join(st.session_state.get("profiles_from_secrets_ui", [])) or "—")
    with st.sidebar.expander("Profiles from Secrets FILE"):
        st.write(", ".join(st.session_state.get("profiles_from_secrets_file", [])) or "—")
    with st.sidebar.expander("Profiles from YAML"):
        st.write(", ".join(st.session_state.get("profiles_from_yaml", [])) or "—")

    st.sidebar.write(
        "Fields — "
        f"Host: {'✅' if base_profile.host else '—'} · "
        f"DB: {'✅' if base_profile.database else '—'} · "
        f"User: {'✅' if base_profile.username else '—'} · "
        f"Pwd: {'✅' if base_profile.password else '—'}"
    )

    # Try configured DB; if it fails ("Cannot open database"), fall back to master to enumerate DBs
    configured_url = build_url(base_profile, None)
    ok, err = try_connect(configured_url)
    effective_db = base_profile.database
    effective_url = configured_url
    if not ok and ("Cannot open database" in err or "Login failed" in err):
        master_url = build_url(base_profile, "master")
        ok2, err2 = try_connect(master_url)
        if ok2:
            effective_db = "master"
            effective_url = master_url
            st.sidebar.info("Connected to **master** to discover databases (configured DB not accessible).")
        else:
            st.sidebar.info("Not connected")
            st.sidebar.code(err or err2)

    st.sidebar.write(f"Credentials present: **{'Yes' if bool(configured_url) else 'No'}**")
    if DBAPI == "pytds" and (configured_url or effective_url):
        st.sidebar.caption(f"TLS (pytds) CA: `{certifi.where()}`")

    if effective_url:
        st.sidebar.success("Connected")

    # Database dropdown
    dbs = list_databases(effective_url) if effective_url else []
    initial = effective_db if effective_db in dbs else (dbs[0] if dbs else effective_db or "master")
    selected_db = st.sidebar.selectbox("Database", dbs or [initial], index=(dbs.index(initial) if dbs and initial in dbs else 0))
    effective_url = build_url(base_profile, selected_db)
    effective_profile = replace(base_profile, database=selected_db)

    # COMM/VW picker
    st.sidebar.markdown("---")
    st.sidebar.subheader("Choose COMM/VW table/view")
    objs = list_comm_vw_objects(effective_url)
    chosen = None
    if objs.empty:
        st.sidebar.info("No objects matching comm* (excluding commercial*) or vw* in this DB.")
    else:
        q = st.sidebar.text_input("Filter", placeholder="e.g. comm_register, vw_", value="").strip().lower()
        filt = objs[
            objs["schema_name"].str.lower().str.contains(q) | objs["object_name"].str.lower().str.contains(q)
        ] if q else objs
        if filt.empty:
            st.sidebar.info("No matches for current filter.")
        else:
            labels = [f"{r.schema_name}.{r.object_name}  •  {r.object_type}" for r in filt.itertuples(index=False)]
            chosen_label = st.sidebar.selectbox("Table/View", labels, index=0, key="object_select")
            idx = labels.index(chosen_label)
            chosen = filt.iloc[idx]
            st.sidebar.caption(f"Selected: `{chosen['schema_name']}.{chosen['object_name']}`")
            n = st.sidebar.number_input("Preview rows", 10, 50000, 1000, step=50, key="preview_rows")
            if st.sidebar.button("Refresh sample"):
                # Force refresh by clearing this key in cache for current URL/object
                st.cache_data.clear()
            if chosen is not None:
                # Auto-load sample to main view
                st.session_state["last_df"] = fetch_table_sample(effective_url, chosen["schema_name"], chosen["object_name"], int(n))
                st.session_state["data_source"] = f"{selected_db}.{chosen['schema_name']}.{chosen['object_name']}"

    # Save / Load dashboard JSON (read-only rendering)
    st.sidebar.markdown("---")
    st.sidebar.write("**Load dashboard JSON**")
    up = st.sidebar.file_uploader("Load dashboard JSON", type=["json"])
    if up is not None:
        try:
            spec = json.load(up)
            DashboardSpec(**spec)  # validate
            st.session_state["dashboard_spec"] = spec
            st.sidebar.success("Dashboard loaded.")
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")

    state = {"profile": effective_profile, "url": effective_url, "chosen": chosen}
    return effective_profile, effective_url, state

# -------------------------- Dashboard rendering (read-only) --------------------------

def aggregate(df: pd.DataFrame, x: Optional[str], y: Optional[str], group: Optional[str], agg: str) -> pd.DataFrame:
    if y is None: return df
    aggfunc = {"sum":"sum","mean":"mean","min":"min","max":"max","count":"count"}.get(agg,"sum")
    keys = [c for c in [x, group] if c]
    return df.groupby(keys, dropna=False)[y].agg(aggfunc).reset_index() if keys else df[[y]].agg(aggfunc).to_frame().T

def dataset_for_tile(url: str, t: TileSpec, fallback_df: pd.DataFrame) -> pd.DataFrame:
    if t.dataset_sql and url:
        try:
            return run_sql_cached(url, t.dataset_sql, None, 200000)
        except Exception as e:
            st.warning(f"Tile '{t.title}' failed to load SQL dataset; using main table. ({str(e)[:140]})")
    return fallback_df

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

def section_dashboard(url: str, main_df: pd.DataFrame):
    st.header("Dashboard")
    spec = st.session_state.get("dashboard_spec")
    if not spec:
        st.info("Load a dashboard JSON from the sidebar to display charts.")
        return
    dash = DashboardSpec(**spec)
    st.subheader(dash.title)
    for i, t in enumerate(dash.tiles):
        st.markdown(f"**{t.title}**")
        df = dataset_for_tile(url, t, main_df)
        # Apply per-tile filters (if provided in JSON)
        # Only a basic equality/membership filter here; extend as needed.
        if t.filters:
            for col, val in t.filters.items():
                if col in df.columns:
                    if isinstance(val, (list, tuple, set)):
                        df = df[df[col].isin(list(val))]
                    else:
                        df = df[df[col] == val]
        render_chart(df, t)

# -------------------------- Main (lean viewer) --------------------------

def main():
    st.set_page_config(page_title="Helix Commercial Analyzer", layout="wide")

    config = load_config(CONFIG_PATH)
    profile, url, state = sidebar_controls(config)

    # Reset when server/DB changes
    cur_key = connection_key(profile.name, profile.host, profile.database)
    prev_key = st.session_state.get("active_profile_key")
    if prev_key != cur_key:
        st.session_state["active_profile_key"] = cur_key
        st.session_state.pop("last_df", None)
        st.session_state["data_source"] = "Demo dataset"
        st.cache_data.clear()

    # ---------- DATA VIEW ----------
    st.header("Data")
    source_label = st.session_state.get("data_source", "Demo dataset")
    df = st.session_state.get("last_df", demo_dataset())
    st.caption(f"Source: **{source_label}** • Rows loaded: **{len(df):,}**")

    # Global search (string contains across all columns)
    search_q = st.text_input("Search", placeholder="Type to find any text across all columns…", key="global_search")
    df_filtered = apply_search(df, search_q)

    # Filters (date + categorical)
    dt_cols = [c for c in df_filtered.columns if pd.api.types.is_datetime64_any_dtype(df_filtered[c])]
    df_filtered = apply_filters(df_filtered, dt_cols)

    st.dataframe(df_filtered)

    # ---------- DASHBOARD (read-only) ----------
    section_dashboard(url, df_filtered)

if __name__ == "__main__":
    main()
