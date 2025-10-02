"""
Helix Commercial Analyzer — Environmental Tracker

LEFT (unchanged):
  - Server + DB + COMM/VW picker with preview
RIGHT:
  - Data grid with global search + filters
  - Environmental tabs:
      • CII (fleet & vessel views, bands overlay via CSV/DB)
      • EU MRV (voyage/month/year KPIs + charts)
      • EEXI (attained vs required, compliance margin)
      • Hull Fouling indicator (rolling normalized fuel-per-nm)
      • EU ETS estimator (phase-in, voyage share controls)
      • FuelEU (GHG intensity vs target)
  - Dashboard viewer (read-only JSON)

Important:
  - This code provides scaffolding + indicative calcs.
  - Plug in official factors/thresholds from CSV/DB views for production compliance.
"""

from __future__ import annotations
import os, json
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import sqlalchemy as sa
import streamlit as st
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.engine import Engine
from urllib.parse import quote_plus

import certifi  # TLS bundle for pytds

# Optional YAML/TOML (local fallback config and repo-side secrets)
try:
    import yaml
except Exception:
    yaml = None
try:
    import tomllib  # Py 3.11+
except Exception:
    tomllib = None
try:
    from sqlalchemy import create_engine, text
except Exception:
    create_engine = None
    text = None

# -------------------------- App config --------------------------

CONFIG_PATH = os.getenv("HELIX_CONFIG", "helix_config.yaml")

def _get_dbapi() -> str:
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

# -------------------------- Data classes --------------------------

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
        if v not in allowed: raise ValueError(f"chart_type must be one of {allowed}")
        return v

class DashboardSpec(BaseModel):
    title: str
    tiles: List[TileSpec] = Field(default_factory=list)

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
                name=name, type=cfg.get("type","synapse"),
                host=cfg.get("host",""), database=cfg.get("database",""),
                username=cfg.get("username",""), password=cfg.get("password",""),
                driver=cfg.get("driver","ODBC Driver 18 for SQL Server")))
    return out

def _profiles_from_secrets_file() -> List[ServerProfile]:
    path = os.path.join(".streamlit","secrets.toml")
    if not os.path.exists(path) or tomllib is None: return []
    try:
        with open(path,"rb") as f: data = tomllib.load(f)
        servers = data.get("servers", {})
        out: List[ServerProfile] = []
        if isinstance(servers, dict):
            for name, cfg in servers.items():
                out.append(ServerProfile(
                    name=name, type=cfg.get("type","synapse"),
                    host=cfg.get("host",""), database=cfg.get("database",""),
                    username=cfg.get("username",""), password=cfg.get("password",""),
                    driver=cfg.get("driver","ODBC Driver 18 for SQL Server")))
        return out
    except Exception:
        return []

def _profiles_from_yaml(path: str) -> Tuple[List[ServerProfile], Dict[str, Any]]:
    cfg = None
    if yaml is not None:
        if not os.path.exists(path):
            try:
                with open(path,"w",encoding="utf-8") as f: f.write(DEFAULT_CONFIG_YAML)
            except Exception:
                pass
        try:
            with open(path,"r",encoding="utf-8") as f:
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
    if s1: source, servers, options = "secrets-ui", s1, y_opts
    elif s2: source, servers, options = "secrets-file", s2, y_opts
    else: source, servers, options = "yaml", y_servers, y_opts
    st.session_state["profiles_source"] = source
    st.session_state["profiles_from_secrets_ui"] = [s.name for s in s1]
    st.session_state["profiles_from_secrets_file"] = [s.name for s in s2]
    st.session_state["profiles_from_yaml"] = [s.name for s in y_servers]
    return AppConfig(servers=servers, options=options)

# -------------------------- Connectivity --------------------------

def build_url(p: ServerProfile, db_override: Optional[str] = None) -> str:
    user, pwd = (p.username or "").strip(), (p.password or "").strip()
    host, db = (p.host or "").strip(), (db_override if db_override is not None else p.database or "").strip()
    if not (host and db): return ""
    if DBAPI == "pytds":
        if not (user and pwd): return ""
        return f"mssql+pytds://{quote_plus(user)}:{quote_plus(pwd)}@{host}:1433/{db}"
    else:
        if not (user and pwd): return ""
        driver = p.driver or "ODBC Driver 18 for SQL Server"
        params = f"DRIVER={{{driver}}};SERVER={host};DATABASE={db};UID={user};PWD={pwd};TrustServerCertificate=no;Encrypt=yes"
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(params)}"

def connection_key(name: str, host: str, db: str) -> str:
    return f"{name}|{host}|{db}|{DBAPI}"

@st.cache_resource(show_spinner=False)
def get_engine_for_url(url: str) -> Optional[Engine]:
    if create_engine is None or not url: return None
    if DBAPI == "pytds":
        return create_engine(url, connect_args={"cafile": certifi.where(), "validate_host": True}, pool_pre_ping=True)
    return create_engine(url, pool_pre_ping=True)

def try_connect(url: str) -> Tuple[bool, str]:
    engine = get_engine_for_url(url) if url else None
    if engine is None: return False, "No engine (missing URL/credentials)."
    try:
        with engine.connect() as conn: conn.exec_driver_sql("SELECT 1")
        return True, ""
    except Exception as e:
        msg = str(e)
        for k in ["UID","PWD","username","password"]: msg = msg.replace(k, "***")
        return False, msg[:500]

@st.cache_data(show_spinner=False)
def run_sql_cached(url: str, sql: str, params: Optional[Dict[str, Any]], limit: int) -> pd.DataFrame:
    engine: Engine = get_engine_for_url(url)
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df.head(limit) if len(df) > limit else df

# -------------------------- Discovery + object APIs --------------------------

@st.cache_data(ttl=300, show_spinner=False)
def list_databases(url: str) -> List[str]:
    if not url: return []
    engine = get_engine_for_url(url)
    if engine is None: return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(sa.text("SELECT name FROM sys.databases WHERE database_id > 4 ORDER BY name;")).fetchall()
            return [r[0] for r in rows]
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def list_comm_vw_objects(url: str) -> pd.DataFrame:
    if not url: return pd.DataFrame(columns=["schema_name","object_name","object_type"])
    engine = get_engine_for_url(url)
    if engine is None: return pd.DataFrame(columns=["schema_name","object_name","object_type"])
    try:
        with engine.connect() as conn:
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
            return pd.read_sql(sql, conn)
    except Exception:
        return pd.DataFrame(columns=["schema_name","object_name","object_type"])

@st.cache_data(ttl=120, show_spinner=False)
def fetch_table_sample(url: str, schema: str, name: str, limit: int) -> pd.DataFrame:
    if not url: return pd.DataFrame()
    engine = get_engine_for_url(url)
    if engine is None: return pd.DataFrame()
    dialect = engine.dialect.name.lower()
    with engine.connect() as conn:
        if dialect == "mssql":
            sql = sa.text(f"SELECT TOP {int(limit)} * FROM [{schema}].[{name}]")
        else:
            prep = sa.sql.compiler.IdentifierPreparer(engine.dialect)
            fq = f"{prep.quote_identifier(schema)}.{prep.quote_identifier(name)}"
            sql = sa.text(f"SELECT * FROM {fq} LIMIT {int(limit)}")
        df = pd.read_sql(sql, conn)
    # best-effort datetime parsing
    for c in df.columns:
        if df[c].dtype == object:
            try: df[c] = pd.to_datetime(df[c], errors="ignore")
            except Exception: pass
    return df

# -------------------------- Demo data --------------------------

def demo_dataset() -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rows = []
    vessels = ["LNG Aurora","LNG Borealis","LNG Caledonia"]
    for v in vessels:
        dwt = np.random.randint(140000, 180000)
        for d in dates:
            dist = np.random.uniform(0, 500)
            fuel = dist * np.random.uniform(0.015, 0.030)  # t/day proxy
            rows.append({
                "vessel": v,
                "date": d,
                "distance_nm": dist,
                "fuel_t": fuel,
                "co2_t": fuel*3.114,  # HFO-ish
                "dwt": dwt,
                "voyage_id": f"{v.split()[1]}-{d.month}",
                "ship_type": "LNG",
                "speed_kt": np.random.uniform(9, 19),
                "power_mw": np.random.uniform(10, 25),
            })
    return pd.DataFrame(rows)

# -------------------------- UI helpers (search + filters) --------------------------

def apply_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query: return df
    q = str(query).strip().lower()
    if not q: return df
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        try:
            mask |= df[c].astype(str).str.lower().str.contains(q, na=False)
        except Exception:
            pass
    return df[mask]

def apply_filters(df: pd.DataFrame, dt_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    with st.expander("Filters", expanded=False):
        if dt_cols:
            for dc in dt_cols:
                st.caption(f"Date filter — {dc}")
                c1, c2 = st.columns(2)
                with c1: start = st.date_input(f"Start ({dc})", value=None, format="YYYY-MM-DD", key=f"start_{dc}")
                with c2: end   = st.date_input(f"End ({dc})", value=None, format="YYYY-MM-DD", key=f"end_{dc}")
                if start is not None: out = out[out[dc] >= pd.to_datetime(start)]
                if end   is not None: out = out[out[dc] <= pd.to_datetime(end)]
        cat_cols = [c for c in out.columns if out[c].dtype == "object" or pd.api.types.is_categorical_dtype(out[c])]
        if cat_cols:
            st.caption("Categorical filters")
            for c in cat_cols[:12]:
                vals = sorted(out[c].dropna().astype(str).unique().tolist())[:200]
                sel = st.multiselect(c, vals, key=f"cat_{c}")
                if sel: out = out[out[c].astype(str).isin(sel)]
    return out

# -------------------------- Sidebar (server/DB/table) --------------------------

def sidebar_controls(config: AppConfig) -> Tuple[ServerProfile, str, Dict[str, Any]]:
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
    with st.sidebar.expander("Profiles from Secrets UI"): st.write(", ".join(st.session_state.get("profiles_from_secrets_ui", [])) or "—")
    with st.sidebar.expander("Profiles from Secrets FILE"): st.write(", ".join(st.session_state.get("profiles_from_secrets_file", [])) or "—")
    with st.sidebar.expander("Profiles from YAML"): st.write(", ".join(st.session_state.get("profiles_from_yaml", [])) or "—")

    st.sidebar.write(
        "Fields — " +
        f"Host: {'✅' if base_profile.host else '—'} · " +
        f"DB: {'✅' if base_profile.database else '—'} · " +
        f"User: {'✅' if base_profile.username else '—'} · " +
        f"Pwd: {'✅' if base_profile.password else '—'}"
    )

    configured_url = build_url(base_profile, None)
    ok, err = try_connect(configured_url)
    effective_db, effective_url = base_profile.database, configured_url
    if not ok and ("Cannot open database" in err or "Login failed" in err):
        master_url = build_url(base_profile, "master")
        ok2, err2 = try_connect(master_url)
        if ok2:
            effective_db, effective_url = "master", master_url
            st.sidebar.info("Connected to **master** to discover databases (configured DB not accessible).")
        else:
            st.sidebar.info("Not connected")
            st.sidebar.code(err or err2)

    st.sidebar.write(f"Credentials present: **{'Yes' if bool(configured_url) else 'No'}**")
    if DBAPI == "pytds" and (configured_url or effective_url):
        st.sidebar.caption(f"TLS (pytds) CA: `{certifi.where()}`")
    if effective_url: st.sidebar.success("Connected")

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
            n = st.sidebar.number_input("Preview rows", 10, 100000, 2000, step=100, key="preview_rows")
            if st.sidebar.button("Refresh sample"): st.cache_data.clear()
            if chosen is not None:
                st.session_state["last_df"] = fetch_table_sample(effective_url, chosen["schema_name"], chosen["object_name"], int(n))
                st.session_state["data_source"] = f"{selected_db}.{chosen['schema_name']}.{chosen['object_name']}"

    # Dashboard JSON loader (viewer)
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

    return effective_profile, effective_url, {"chosen": chosen}

# -------------------------- Dashboard viewer --------------------------

def aggregate(df: pd.DataFrame, x: Optional[str], y: Optional[str], group: Optional[str], agg: str) -> pd.DataFrame:
    if y is None: return df
    aggfunc = {"sum":"sum","mean":"mean","min":"min","max":"max","count":"count"}.get(agg,"sum")
    keys = [c for c in [x, group] if c]
    return df.groupby(keys, dropna=False)[y].agg(aggfunc).reset_index() if keys else df[[y]].agg(aggfunc).to_frame().T

def dataset_for_tile(url: str, t: TileSpec, fallback_df: pd.DataFrame) -> pd.DataFrame:
    if t.dataset_sql and url:
        try: return run_sql_cached(url, t.dataset_sql, None, 200000)
        except Exception as e:
            st.warning(f"Tile '{t.title}' failed to load SQL dataset; using main table. ({str(e)[:140]})")
    return fallback_df

def render_chart(df: pd.DataFrame, spec: TileSpec):
    cfg = {"responsive": True, "displaylogo": False}
    if spec.chart_type == "table":
        st.dataframe(df); return
    if spec.chart_type == "pie":
        if spec.group and spec.y:
            aggdf = df.groupby(spec.group, dropna=False)[spec.y].sum().reset_index()
            fig = px.pie(aggdf, names=spec.group, values=spec.y, title=spec.title)
            st.plotly_chart(fig, config=cfg)
        else:
            st.warning("Pie requires 'group' and 'y'."); return
    else:
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
        st.info("Load a dashboard JSON from the sidebar to display charts."); return
    dash = DashboardSpec(**spec)
    st.subheader(dash.title)
    for t in dash.tiles:
        st.markdown(f"**{t.title}**")
        df = dataset_for_tile(url, t, main_df)
        if t.filters:
            for col, val in t.filters.items():
                if col in df.columns:
                    if isinstance(val, (list,tuple,set)): df = df[df[col].isin(list(val))]
                    else: df = df[df[col] == val]
        render_chart(df, t)

# -------------------------- Environmental: helpers --------------------------

# Basic TTW CO2 emission factors (tCO2 per tonne of fuel) — placeholder values.
EF_T_CO2_PER_T_FUEL = {
    "HFO": 3.114, "LFO": 3.151, "MGO": 3.206, "MDO": 3.206, "LNG": 2.750, "LPG": 3.000,
}

def co2_from_fuel(df: pd.DataFrame, fuel_t_col: str, fuel_type_col: Optional[str]) -> pd.Series:
    if fuel_type_col and fuel_type_col in df.columns:
        ef = df[fuel_type_col].map(EF_T_CO2_PER_T_FUEL).fillna(3.114)  # default HFO
        return df[fuel_t_col].fillna(0) * ef
    return df[fuel_t_col].fillna(0) * 3.114  # default

def cii_mapper_ui(df: pd.DataFrame) -> Dict[str, str]:
    cols = df.columns.tolist()
    st.caption("Map your CII columns")
    c1, c2, c3 = st.columns(3)
    with c1:
        vessel = st.selectbox("Vessel", cols, index=(cols.index("vessel") if "vessel" in cols else 0), key="cii_vessel")
        date = st.selectbox("Date", cols, index=(cols.index("date") if "date" in cols else 0), key="cii_date")
    with c2:
        dist = st.selectbox("Distance (nm)", cols, index=(cols.index("distance_nm") if "distance_nm" in cols else 0), key="cii_dist")
        dwt = st.selectbox("DWT / Capacity", cols, index=(cols.index("dwt") if "dwt" in cols else 0), key="cii_dwt")
    with c3:
        co2 = st.selectbox("CO2 (t) [optional]", [None]+cols, index=(cols.index("co2_t")+1 if "co2_t" in cols else 0), key="cii_co2")
        fuel_t = st.selectbox("Fuel (t) [optional]", [None]+cols, index=(cols.index("fuel_t")+1 if "fuel_t" in cols else 0), key="cii_fuel_t")
    ship_type = st.selectbox("Ship type", [c for c in ["LNG","Tanker","Bulkcarrier","Container","Other"]], index=0, key="cii_shiptype")
    fuel_type = st.selectbox("Fuel type col [optional]", [None]+cols, index=(cols.index("fuel_type")+1 if "fuel_type" in cols else 0), key="cii_fueltype")
    return {
        "vessel": vessel, "date": date, "distance_nm": dist, "dwt": dwt,
        "co2_t": co2 or "", "fuel_t": fuel_t or "", "fuel_type": fuel_type or "", "ship_type": ship_type
    }

def compute_aer(df: pd.DataFrame, m: Dict[str,str]) -> pd.DataFrame:
    x = df.copy()
    # CO2 selection
    if m["co2_t"]:
        x["co2_t_calc"] = pd.to_numeric(x[m["co2_t"]], errors="coerce").fillna(0.0)
    elif m["fuel_t"]:
        x["co2_t_calc"] = co2_from_fuel(x, m["fuel_t"], m["fuel_type"] or None)
    else:
        x["co2_t_calc"] = 0.0
    # base fields
    x["distance_nm_calc"] = pd.to_numeric(x[m["distance_nm"]], errors="coerce").fillna(0.0)
    x["dwt_calc"] = pd.to_numeric(x[m["dwt"]], errors="coerce").fillna(np.nan)
    x["vessel_calc"] = x[m["vessel"]].astype(str)
    x["date_calc"] = pd.to_datetime(x[m["date"]], errors="coerce")
    # AER gCO2 per dwt-nm
    denom = (x["dwt_calc"] * x["distance_nm_calc"]).replace(0, np.nan)
    x["aer_g_per_dwt_nm"] = (x["co2_t_calc"] * 1_000_000) / (denom * 1000)  # 1 t = 1e6 g ; 1 dwt ~ 1 t
    # tidy
    return x[["vessel_calc","date_calc","dwt_calc","distance_nm_calc","co2_t_calc","aer_g_per_dwt_nm"]].rename(
        columns={"vessel_calc":"vessel","date_calc":"date","dwt_calc":"dwt","distance_nm_calc":"distance_nm","co2_t_calc":"co2_t"}
    )

def cii_band_data_ui(url: str) -> Optional[pd.DataFrame]:
    st.caption("CII bands source (optional)")
    ups = st.file_uploader("Upload CSV with columns: ship_type,year,dwt_min,dwt_max,A,B,C,D,E (AER thresholds gCO2/dwt-nm)", type=["csv"], key="cii_bands_up")
    if ups is not None:
        try:
            df = pd.read_csv(ups)
            st.success("Loaded bands CSV.")
            return df
        except Exception as e:
            st.error(f"Bad CSV: {e}")
    # Future: read from DB view here if you maintain one (e.g., vw_cii_bands)
    return None

def fleet_cii_view(aer_df: pd.DataFrame, ship_type: str, bands: Optional[pd.DataFrame]):
    st.subheader("Fleet view")
    min_date, max_date = st.date_input("Date range", value=(aer_df["date"].min(), aer_df["date"].max()), format="YYYY-MM-DD")
    mask = (aer_df["date"] >= pd.to_datetime(min_date)) & (aer_df["date"] <= pd.to_datetime(max_date))
    g = aer_df.loc[mask].groupby("vessel", as_index=False).agg(
        aer=("aer_g_per_dwt_nm","mean"),
        dwt=("dwt","median")
    )
    fig = px.scatter(g, x="dwt", y="aer", hover_name="vessel", title="AER vs DWT (mean over selected range)")
    if bands is not None and not bands.empty:
        yr = st.number_input("Band year", value=int(pd.Timestamp.today().year), step=1)
        b = bands[(bands["ship_type"].str.lower()==ship_type.lower()) & (bands["year"]==int(yr))]
        # draw shaded bands
        for _, row in b.iterrows():
            # Expect one row per dwt range with A..E thresholds; draw horizontal-ish steps with rectangles
            # For simplicity, show thresholds as lines at mid DWT band
            pass
        st.caption("Bands CSV loaded — overlay not drawn to avoid wrong assumptions; plug your visuals as desired.")
    st.plotly_chart(fig, config={"responsive": True, "displaylogo": False})

def vessel_cii_view(aer_df: pd.DataFrame):
    st.subheader("Vessel detail")
    vessels = sorted(aer_df["vessel"].dropna().unique().tolist())
    v = st.selectbox("Vessel", vessels, index=0)
    mode = st.radio("View by", ["Voyage (requires voyage_id)", "Monthly", "Yearly"], horizontal=True)
    dfv = aer_df[aer_df["vessel"]==v].copy()
    if mode.startswith("Voyage") and "voyage_id" in dfv.columns:
        grp = dfv.groupby("voyage_id", as_index=False).agg(aer=("aer_g_per_dwt_nm","mean"))
        fig = px.bar(grp, x="voyage_id", y="aer", title=f"{v} — AER by voyage")
    elif mode == "Monthly":
        dfv["month"] = dfv["date"].dt.to_period("M").astype(str)
        grp = dfv.groupby("month", as_index=False).agg(aer=("aer_g_per_dwt_nm","mean"))
        fig = px.line(grp, x="month", y="aer", markers=True, title=f"{v} — AER monthly")
    else:
        dfv["year"] = dfv["date"].dt.year
        grp = dfv.groupby("year", as_index=False).agg(aer=("aer_g_per_dwt_nm","mean"))
        fig = px.line(grp, x="year", y="aer", markers=True, title=f"{v} — AER yearly")
    st.plotly_chart(fig, config={"responsive": True, "displaylogo": False})

def mrv_mapper_ui(df: pd.DataFrame) -> Dict[str, str]:
    cols = df.columns.tolist()
    st.caption("Map your MRV columns")
    c1, c2, c3 = st.columns(3)
    with c1:
        vessel = st.selectbox("Vessel", cols, index=(cols.index("vessel") if "vessel" in cols else 0), key="mrv_vessel")
        date = st.selectbox("Date", cols, index=(cols.index("date") if "date" in cols else 0), key="mrv_date")
    with c2:
        dist = st.selectbox("Distance (nm)", cols, index=(cols.index("distance_nm") if "distance_nm" in cols else 0), key="mrv_dist")
        cargo = st.selectbox("Cargo carried (t) [optional]", [None]+cols, index=0, key="mrv_cargo")
    with c3:
        co2 = st.selectbox("CO2 (t)", cols, index=(cols.index("co2_t") if "co2_t" in cols else 0), key="mrv_co2")
        port = st.selectbox("Port / Area [optional]", [None]+cols, index=0, key="mrv_port")
    return {"vessel":vessel,"date":date,"distance_nm":dist,"co2_t":co2,"cargo_t":cargo or "","port":port or ""}

def mrv_kpis_and_plots(df: pd.DataFrame, m: Dict[str,str]):
    x = df.copy()
    x["date"] = pd.to_datetime(x[m["date"]], errors="coerce")
    x["co2_t"] = pd.to_numeric(x[m["co2_t"]], errors="coerce").fillna(0.0)
    x["distance_nm"] = pd.to_numeric(x[m["distance_nm"]], errors="coerce").fillna(0.0)
    if m["cargo_t"]:
        x["cargo_t"] = pd.to_numeric(x[m["cargo_t"]], errors="coerce").fillna(np.nan)
    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Total CO₂ (t)", f"{x['co2_t'].sum():,.0f}")
    k2.metric("Total distance (nm)", f"{x['distance_nm'].sum():,.0f}")
    if m["cargo_t"]:
        denom = (x["cargo_t"]*x["distance_nm"]).replace(0,np.nan)
        ef = (x["co2_t"]*1_000_000)/(denom*1000)  # g CO2 / t-nm
        k3.metric("Avg gCO₂/t-nm", f"{ef.mean():,.1f}")
    else:
        k3.metric("Avg gCO₂/t-nm", "—")
    # Charts
    x["month"] = x["date"].dt.to_period("M").astype(str)
    st.plotly_chart(px.bar(x.groupby("month", as_index=False)["co2_t"].sum(), x="month", y="co2_t",
                           title="CO₂ by month"), config={"responsive": True, "displaylogo": False})

def eexi_mapper_ui(df: pd.DataFrame) -> Dict[str,str]:
    cols = df.columns.tolist()
    st.caption("Map your EEXI columns (static per vessel)")
    c1, c2, c3 = st.columns(3)
    with c1:
        vessel = st.selectbox("Vessel", cols, index=(cols.index("vessel") if "vessel" in cols else 0), key="eexi_vessel")
    with c2:
        attained = st.selectbox("Attained EEXI", cols, index=(cols.index("eexi_attained") if "eexi_attained" in cols else 0), key="eexi_attained")
    with c3:
        required = st.selectbox("Required EEXI", cols, index=(cols.index("eexi_required") if "eexi_required" in cols else 0), key="eexi_required")
    return {"vessel":vessel,"attained":attained,"required":required}

def eexi_view(df: pd.DataFrame, m: Dict[str,str]):
    x = df.copy()
    x["attained"] = pd.to_numeric(x[m["attained"]], errors="coerce")
    x["required"] = pd.to_numeric(x[m["required"]], errors="coerce")
    x["vessel"] = x[m["vessel"]].astype(str)
    x["margin_pct"] = (x["required"] - x["attained"]) / x["required"] * 100.0
    st.dataframe(x[["vessel","attained","required","margin_pct"]].sort_values("margin_pct", ascending=False))
    st.plotly_chart(px.bar(x, x="vessel", y="margin_pct", title="EEXI margin (%) — positive is compliant"), config={"responsive": True, "displaylogo": False})

def hull_mapper_ui(df: pd.DataFrame) -> Dict[str,str]:
    cols = df.columns.tolist()
    st.caption("Map your performance columns for hull fouling indicator")
    c1, c2, c3 = st.columns(3)
    with c1:
        date = st.selectbox("Date", cols, index=(cols.index("date") if "date" in cols else 0), key="hull_date")
        vessel = st.selectbox("Vessel", cols, index=(cols.index("vessel") if "vessel" in cols else 0), key="hull_vessel")
    with c2:
        speed = st.selectbox("Speed (kt)", cols, index=(cols.index("speed_kt") if "speed_kt" in cols else 0), key="hull_speed")
        fuel_t = st.selectbox("Fuel per day (t) / Consumption", cols, index=(cols.index("fuel_t") if "fuel_t" in cols else 0), key="hull_fuel")
    with c3:
        dist = st.selectbox("Distance (nm)", cols, index=(cols.index("distance_nm") if "distance_nm" in cols else 0), key="hull_dist")
        group = st.selectbox("Group baseline by (e.g., draft/route) [optional]", [None]+cols, index=0, key="hull_group")
    return {"date":date,"vessel":vessel,"speed":speed,"fuel_t":fuel_t,"distance_nm":dist,"group":group or ""}

def hull_indicator(df: pd.DataFrame, m: Dict[str,str]):
    x = df.copy()
    x["date"] = pd.to_datetime(x[m["date"]], errors="coerce")
    x["vessel"] = x[m["vessel"]].astype(str)
    x["speed"] = pd.to_numeric(x[m["speed"]], errors="coerce")
    x["fuel_t"] = pd.to_numeric(x[m["fuel_t"]], errors="coerce")
    x["distance_nm"] = pd.to_numeric(x[m["distance_nm"]], errors="coerce").replace(0,np.nan)
    x["fuel_per_nm"] = x["fuel_t"] / x["distance_nm"]
    key_cols = ["vessel"] + ([m["group"]] if m["group"] else [])
    # rolling 30-day moving average per vessel/group
    x = x.sort_values("date")
    x["rolling_fpnm"] = x.groupby(key_cols)["fuel_per_nm"].transform(lambda s: s.rolling(30, min_periods=5).mean())
    base = x.groupby(key_cols, as_index=False)["fuel_per_nm"].transform("median")
    x["degradation_pct"] = (x["rolling_fpnm"] - base) / base * 100.0
    st.plotly_chart(px.line(x, x="date", y="degradation_pct", color="vessel", title="Hull performance degradation vs baseline (%)"),
                    config={"responsive": True, "displaylogo": False})
    thr = st.slider("Alert threshold (%)", 2.0, 20.0, 8.0, step=0.5)
    alerts = x.groupby("vessel", as_index=False)["degradation_pct"].last()
    alerts["recommend_cleaning"] = alerts["degradation_pct"] >= thr
    st.dataframe(alerts.sort_values("degradation_pct", ascending=False))

def ets_mapper_ui(df: pd.DataFrame) -> Dict[str,str]:
    cols = df.columns.tolist()
    st.caption("Map your ETS columns")
    c1, c2, c3 = st.columns(3)
    with c1:
        vessel = st.selectbox("Vessel", cols, index=(cols.index("vessel") if "vessel" in cols else 0), key="ets_vessel")
        date = st.selectbox("Date", cols, index=(cols.index("date") if "date" in cols else 0), key="ets_date")
    with c2:
        co2 = st.selectbox("CO2 (t)", cols, index=(cols.index("co2_t") if "co2_t" in cols else 0), key="ets_co2")
        voyage_type = st.selectbox("Voyage type (intraEU/extraEU/inbound/outbound)", [None]+cols, index=0, key="ets_type")
    with c3:
        price = st.number_input("ETS price (€/tCO2)", min_value=0.0, value=65.0, step=1.0)
    return {"vessel":vessel,"date":date,"co2_t":co2,"type":voyage_type or "", "price": price}

def ets_estimator(df: pd.DataFrame, m: Dict[str,str]):
    x = df.copy()
    x["date"] = pd.to_datetime(x[m["date"]], errors="coerce")
    x["co2_t"] = pd.to_numeric(x[m["co2_t"]], errors="coerce").fillna(0.0)
    if m["type"]:
        x["type"] = x[m["type"]].str.lower().fillna("unknown")
    else:
        x["type"] = "unknown"
    year = st.number_input("Compliance year", min_value=2024, max_value=2035, value=pd.Timestamp.today().year, step=1)
    # Phase-in factor — user control (official values may differ over time)
    phase_in = st.slider("Phase-in coverage (%)", 0, 100, 70 if year==2025 else (40 if year==2024 else 100), step=5)
    # Voyage share weights
    intra = st.slider("Intra-EU coverage (%)", 0, 100, 100, step=10)
    extra = st.slider("Extra-EU (inbound/outbound) coverage (%)", 0, 100, 50, step=10)
    def coverage(t):
        if t.startswith("intra"): return intra/100.0
        if t in ("extraeu","inbound","outbound"): return extra/100.0
        return 0.0
    x["covered_t"] = x["co2_t"] * x["type"].map(coverage)
    x["allowances_t"] = x["covered_t"] * (phase_in/100.0)
    k1,k2,k3 = st.columns(3)
    k1.metric("Total CO₂ (t)", f"{x['co2_t'].sum():,.0f}")
    k2.metric("Covered CO₂ (t)", f"{x['covered_t'].sum():,.0f}")
    k3.metric("ETS allowances (t)", f"{x['allowances_t'].sum():,.0f}")
    st.metric("Cost estimate (€)", f"{x['allowances_t'].sum()*m['price']:,.0f}")
    x["month"] = x["date"].dt.to_period("M").astype(str)
    st.plotly_chart(px.bar(x.groupby("month", as_index=False)["allowances_t"].sum(), x="month", y="allowances_t",
                           title="ETS allowances by month (t)"), config={"responsive": True, "displaylogo": False})

def fueleu_mapper_ui(df: pd.DataFrame) -> Dict[str,str]:
    cols = df.columns.tolist()
    st.caption("Map your FuelEU columns")
    c1, c2, c3 = st.columns(3)
    with c1:
        vessel = st.selectbox("Vessel", cols, index=(cols.index("vessel") if "vessel" in cols else 0), key="fe_vessel")
        date = st.selectbox("Date", cols, index=(cols.index("date") if "date" in cols else 0), key="fe_date")
    with c2:
        energy_mj = st.selectbox("Energy used (MJ) [optional]", [None]+cols, index=0, key="fe_mj")
        co2eq = st.selectbox("CO₂e (kg) [optional]", [None]+cols, index=0, key="fe_co2eq")
    with c3:
        fuel_t = st.selectbox("Fuel (t) [optional]", [None]+cols, index=(cols.index("fuel_t")+1 if "fuel_t" in cols else 0), key="fe_fuel_t")
        fuel_type = st.selectbox("Fuel type col [optional]", [None]+cols, index=(cols.index("fuel_type")+1 if "fuel_type" in cols else 0), key="fe_fueltype")
    target = st.number_input("Target GHG intensity (gCO₂e/MJ)", min_value=0.0, value=91.0, step=0.1)
    return {"vessel":vessel,"date":date,"energy_mj":energy_mj or "","co2eq":co2eq or "","fuel_t":fuel_t or "","fuel_type":fuel_type or "","target":target}

def fueleu_view(df: pd.DataFrame, m: Dict[str,str]):
    x = df.copy()
    x["date"] = pd.to_datetime(x[m["date"]], errors="coerce")
    # compute intensity
    if m["energy_mj"] and m["co2eq"]:
        e = pd.to_numeric(x[m["energy_mj"]], errors="coerce").fillna(np.nan)
        co2e = pd.to_numeric(x[m["co2eq"]], errors="coerce").fillna(np.nan)
        x["intensity"] = (co2e*1000)/e  # g/MJ if co2e is kg
    elif m["fuel_t"]:
        # rough proxy: convert fuel_t -> energy via LHV, fuel_t -> CO2eq via EF (assumes TTW)
        LHV_MJ_PER_T = {"HFO": 40_400, "MGO": 42_700, "LNG": 49_500}
        ft = x[m["fuel_type"]].map(LHV_MJ_PER_T).fillna(40_000) if m["fuel_type"] else 40_000
        energy = pd.to_numeric(x[m["fuel_t"]], errors="coerce").fillna(0.0) * ft
        co2e = co2_from_fuel(x, m["fuel_t"], m["fuel_type"] or None) * 1000  # t -> kg
        x["intensity"] = co2e / energy.replace(0,np.nan)
    else:
        x["intensity"] = np.nan
    x["month"] = x["date"].dt.to_period("M").astype(str)
    grp = x.groupby("month", as_index=False)["intensity"].mean()
    line = px.line(grp, x="month", y="intensity", markers=True, title="GHG intensity (gCO₂e/MJ)")
    line.add_hline(y=m["target"], line_dash="dot", annotation_text=f"Target {m['target']} g/MJ")
    st.plotly_chart(line, config={"responsive": True, "displaylogo": False})
    st.dataframe(x[["vessel","date","intensity"]].sort_values("date"))

# -------------------------- Main --------------------------

def main():
    st.set_page_config(page_title="Helix Commercial Analyzer — Environmental", layout="wide")

    config = load_config(CONFIG_PATH)
    profile, url, _ = sidebar_controls(config)

    # Reset when server/DB changes
    cur_key = connection_key(profile.name, profile.host, profile.database)
    if st.session_state.get("active_profile_key") != cur_key:
        st.session_state["active_profile_key"] = cur_key
        st.session_state.pop("last_df", None)
        st.session_state["data_source"] = "Demo dataset"
        st.cache_data.clear()

    # DATA VIEW
    st.header("Data")
    df = st.session_state.get("last_df", demo_dataset())
    source_label = st.session_state.get("data_source","Demo dataset")
    st.caption(f"Source: **{source_label}** • Rows: **{len(df):,}**")
    search_q = st.text_input("Search", placeholder="Type to find text across all columns…")
    df_filtered = apply_search(df, search_q)
    dt_cols = [c for c in df_filtered.columns if pd.api.types.is_datetime64_any_dtype(df_filtered[c])]
    df_filtered = apply_filters(df_filtered, dt_cols)
    st.dataframe(df_filtered)

    # ENVIRONMENTAL TABS
    st.header("Environmental")
    tab_cii, tab_mrv, tab_eexi, tab_hull, tab_ets, tab_fueleu = st.tabs(["CII","EU MRV","EEXI","Hull fouling","EU ETS","FuelEU"])

    with tab_cii:
        if df_filtered.empty:
            st.info("No data loaded.")
        else:
            m = cii_mapper_ui(df_filtered)
            aer_df = compute_aer(df_filtered, m)
            bands = cii_band_data_ui(url)  # optional CSV
            st.markdown("---")
            colA, colB = st.columns(2)
            with colA: fleet_cii_view(aer_df, m["ship_type"], bands)
            with colB: vessel_cii_view(aer_df)

    with tab_mrv:
        if df_filtered.empty:
            st.info("No data loaded.")
        else:
            m = mrv_mapper_ui(df_filtered); st.markdown("---"); mrv_kpis_and_plots(df_filtered, m)

    with tab_eexi:
        if df_filtered.empty:
            st.info("No data loaded.")
        else:
            m = eexi_mapper_ui(df_filtered); st.markdown("---"); eexi_view(df_filtered, m)

    with tab_hull:
        if df_filtered.empty:
            st.info("No data loaded.")
        else:
            m = hull_mapper_ui(df_filtered); st.markdown("---"); hull_indicator(df_filtered, m)

    with tab_ets:
        if df_filtered.empty:
            st.info("No data loaded.")
        else:
            m = ets_mapper_ui(df_filtered); st.markdown("---"); ets_estimator(df_filtered, m)

    with tab_fueleu:
        if df_filtered.empty:
            st.info("No data loaded.")
        else:
            m = fueleu_mapper_ui(df_filtered); st.markdown("---"); fueleu_view(df_filtered, m)

    # DASHBOARD VIEWER (optional)
    section_dashboard(url, df_filtered)

if __name__ == "__main__":
    main()
