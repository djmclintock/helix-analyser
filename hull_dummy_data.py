# helix-analyser/hull_dummy_data.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

VESSELS = [
    ("ShipNet LNG",       "LNG Carrier"),
    ("ShipNet Bulk",      "Bulk Carrier"),
    ("ShipNet Container", "Container"),
    ("ShipNet Gas",       "Gas Carrier"),
    ("ShipNet Tanker",    "Tanker"),
]

PORTS = ["Ras Laffan", "Zeebrugge", "Gate", "South Hook", "Sines", "Singapore", "Fujairah", "Rotterdam"]

EVENT_TYPES = {
    "Dry dock":          {"effect": -0.15},  # big improvement
    "Hull clean":        {"effect": -0.08},  # medium improvement
    "Propeller polish":  {"effect": -0.04},  # small improvement
}

def _make_events(start: pd.Timestamp, end: pd.Timestamp, rng: np.random.Generator) -> List[Dict]:
    """Create 1–3 events in the window."""
    n = rng.integers(1, 4)
    days = (end - start).days
    events = []
    for _ in range(n):
        dt = start + pd.Timedelta(days=int(rng.integers(10, max(11, days-10))))
        etype = rng.choice(list(EVENT_TYPES.keys()), p=[0.2, 0.5, 0.3])
        loc = rng.choice(PORTS)
        events.append({
            "DateTime": pd.Timestamp(dt.replace(hour=int(rng.integers(0,24)), minute=int(rng.integers(0,60)))),
            "Location": loc,
            "Type": etype,
        })
    # sort chronological
    events.sort(key=lambda e: e["DateTime"])
    return events

def get_hull_dummy_timeseries(start="2025-01-01", end="2025-06-30", seed=7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      ts_df: columns [Vessel, Type, Date, ResistanceIndex]
      ev_df: columns [Vessel, Type, DateTime, Location, Type]
    ResistanceIndex is dimensionless, baseline ~1.00; fouling drift pushes it up;
    maintenance events push it down.
    """
    rng = np.random.default_rng(seed)
    start = pd.to_datetime(start)
    end   = pd.to_datetime(end)
    dates = pd.date_range(start, end, freq="D")

    ts_rows = []
    ev_rows = []

    for vessel, vtype in VESSELS:
        # events
        events = _make_events(start, end, rng)
        for e in events:
            ev_rows.append({
                "Vessel": vessel,
                "Type": vtype,
                "DateTime": e["DateTime"],
                "Location": e["Location"],
                "Event": e["Type"],
            })

        # time series: gradual fouling + noise; apply event effects
        idx = 1.0 + rng.normal(0, 0.01)  # starting resistance index
        drift_per_day = rng.uniform(0.0003, 0.0007)  # slow upward drift
        event_ptr = 0

        for d in dates:
            # apply event if today passed an event timestamp
            while event_ptr < len(events) and d >= events[event_ptr]["DateTime"].normalize():
                eff = EVENT_TYPES[events[event_ptr]["Type"]]["effect"]
                idx = max(0.7, idx + eff)  # drop index (improvement), clamp
                event_ptr += 1

            # add drift + noise
            idx += drift_per_day + rng.normal(0, 0.0025)
            idx = float(max(0.7, idx))  # clamp lower bound

            ts_rows.append({
                "Vessel": vessel,
                "Type": vtype,
                "Date": d,
                "ResistanceIndex": idx,
            })

    ts_df = pd.DataFrame(ts_rows)
    ev_df = pd.DataFrame(ev_rows).sort_values(["Vessel", "DateTime"]).reset_index(drop=True)
    return ts_df, ev_df
