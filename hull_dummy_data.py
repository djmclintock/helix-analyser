# helix-analyser/hull_dummy_data.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict

VESSELS = [
    ("ShipNet LNG",       "LNG Carrier"),
    ("ShipNet Bulk",      "Bulk Carrier"),
    ("ShipNet Container", "Container"),
    ("ShipNet Gas",       "Gas Carrier"),
    ("ShipNet Tanker",    "Tanker"),
]

PORTS = ["Ras Laffan", "Zeebrugge", "Gate", "South Hook", "Sines", "Singapore", "Fujairah", "Rotterdam"]

# Event effects are multiplicative on the resistance index (e.g., -0.15 => -15%)
EVENT_TYPES = {
    "Dry dock":          {"effect": -0.18},  # big improvement
    "Hull clean":        {"effect": -0.12},  # medium improvement
    "Propeller polish":  {"effect": -0.06},  # small improvement
}

def _make_events(start: pd.Timestamp, end: pd.Timestamp, rng: np.random.Generator) -> List[Dict]:
    """Create 1–3 events within the window."""
    days = (end - start).days
    n = int(rng.integers(1, 4)) if days >= 30 else 1
    picks = sorted(rng.integers(low=10, high=max(11, days - 5), size=n))
    events: List[Dict] = []
    for d in picks:
        dt = start + pd.Timedelta(days=int(d))
        etype = rng.choice(list(EVENT_TYPES.keys()), p=[0.25, 0.5, 0.25])
        hr = int(rng.integers(0, 24))
        mn = int(rng.integers(0, 60))
        events.append({
            "DateTime": pd.Timestamp(dt.replace(hour=hr, minute=mn)),
            "Location": rng.choice(PORTS),
            "Type": etype,
        })
    return events

def get_hull_dummy_timeseries(start="2025-01-01", end="2025-06-30", seed=7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      ts_df: [Vessel, Type, Date, ResistanceIndex] (dimensionless, >= 1.0)
      ev_df: [Vessel, Type, DateTime, Location, Event]
    Behavior:
      - Starts ~1.00 (0% fouling) and drifts upward daily (0.10–0.30% / day).
      - Events reduce resistance index by 6–18%.
    """
    rng = np.random.default_rng(seed)
    start = pd.to_datetime(start)
    end   = pd.to_datetime(end)
    dates = pd.date_range(start, end, freq="D")

    ts_rows: List[Dict] = []
    ev_rows: List[Dict] = []

    for vessel, vtype in VESSELS:
        events = _make_events(start, end, rng)
        for e in events:
            ev_rows.append({
                "Vessel": vessel, "Type": vtype,
                "DateTime": e["DateTime"], "Location": e["Location"], "Event": e["Type"],
            })

        # Start near baseline with tiny positive noise
        idx = max(1.0, 1.0 + float(rng.normal(0.003, 0.002)))   # ~ +0.3% start
        # Daily upward drift (0.10–0.30% / day)
        drift = float(rng.uniform(0.0010, 0.0030))

        ev_ptr = 0
        events_sorted = sorted(events, key=lambda x: x["DateTime"])

        for d in dates:
            # Apply events that occurred at/just before this day
            while ev_ptr < len(events_sorted) and d >= events_sorted[ev_ptr]["DateTime"].normalize():
                eff = EVENT_TYPES[events_sorted[ev_ptr]["Type"]]["effect"]  # negative
                idx = idx * (1.0 + eff)                      # multiplicative drop
                idx = max(1.0, idx)                          # never below baseline
                ev_ptr += 1

            # Add drift + small noise (noise centered ~0)
            noise = float(rng.normal(0.0, 0.0008))
            idx = idx * (1.0 + drift + noise)
            idx = max(1.0, idx)

            ts_rows.append({
                "Vessel": vessel, "Type": vtype,
                "Date": d, "ResistanceIndex": float(idx),
            })

    ts_df = pd.DataFrame(ts_rows)
    ev_df = pd.DataFrame(ev_rows).sort_values(["Vessel", "DateTime"]).reset_index(drop=True)
    return ts_df, ev_df
