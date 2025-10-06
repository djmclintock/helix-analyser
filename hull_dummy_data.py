# helix-analyser/hull_dummy_data.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

VESSELS = [
    ("ShipNet LNG",       "LNG Carrier"),
    ("ShipNet Bulk",      "Bulk Carrier"),
    ("ShipNet Container", "Container"),
    ("ShipNet Gas",       "Gas Carrier"),
    ("ShipNet Tanker",    "Tanker"),
]

PORTS = ["Ras Laffan", "Zeebrugge", "Gate", "South Hook", "Sines", "Singapore", "Fujairah", "Rotterdam"]

# Event effects are multiplicative on the resistance index (e.g., -0.15 => -15% drop)
EVENT_TYPES = {
    "Dry dock":          {"effect": -0.18},  # big improvement
    "Hull clean":        {"effect": -0.12},  # medium improvement
    "Propeller polish":  {"effect": -0.06},  # small improvement
}

def _evenly_spaced_events(start: pd.Timestamp, end: pd.Timestamp, rng: np.random.Generator) -> List[Dict]:
    """
    Create 1–3 maintenance events spaced across the window (≥ ~30 days apart when possible).
    """
    days = (end - start).days
    if days < 45:
        n = 1
    elif days < 120:
        n = 2
    else:
        n = 3

    # Even fractions of the range with small jitter
    fracs = np.linspace(0.35, 0.85, n)
    picks = []
    for f in fracs:
        base_day = int(f * days)
        jitter = int(rng.integers(-5, 6))  # ±5 days
        d = max(5, min(days - 5, base_day + jitter))
        picks.append(d)

    events: List[Dict] = []
    for d in sorted(set(picks)):
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
      ts_df: [Vessel, Type, Date, ResistanceIndex]  (dimensionless, >= 1.0)
      ev_df: [Vessel, Type, DateTime, Location, Event]

    Behaviour:
      - ResistanceIndex starts near 1.0 (clean), never goes below 1.0.
      - Between events it rises smoothly toward a target ~+10–15%.
      - At each event it drops by an event-specific fraction, clamped to >= 1.0,
        then begins rising again.
    """
    rng = np.random.default_rng(seed)
    start = pd.to_datetime(start)
    end   = pd.to_datetime(end)
    all_dates = pd.date_range(start, end, freq="D")

    ts_rows: List[Dict] = []
    ev_rows: List[Dict] = []

    for vessel, vtype in VESSELS:
        events = _evenly_spaced_events(start, end, rng)
        for e in events:
            ev_rows.append({
                "Vessel": vessel, "Type": vtype,
                "DateTime": e["DateTime"], "Location": e["Location"], "Event": e["Type"],
            })

        # Work in segments between events
        seg_boundaries = [start] + [e["DateTime"].normalize() for e in sorted(events, key=lambda x: x["DateTime"])] + [end]
        idx = 1.0 + float(rng.uniform(0.000, 0.005))  # start at ~0–0.5% fouling

        for si in range(len(seg_boundaries) - 1):
            seg_start = seg_boundaries[si]
            # segment end is the day *before* the event (or the final end)
            seg_end = (seg_boundaries[si + 1] - pd.Timedelta(days=1)) if si < len(seg_boundaries) - 2 else seg_boundaries[si + 1]
            seg_dates = pd.date_range(seg_start, seg_end, freq="D")
            if not len(seg_dates):
                # If event days collide, still record the index at seg_start
                ts_rows.append({"Vessel": vessel, "Type": vtype, "Date": seg_start, "ResistanceIndex": float(idx)})
            else:
                # Choose a target between +10% and +15% (relative to absolute clean baseline = 1.0)
                target_pct = float(rng.uniform(0.10, 0.15))
                target_idx = 1.0 + target_pct

                # Compute daily growth factor to move from current idx to target_idx over the segment
                n_days = len(seg_dates)
                # ensure positive growth per day (minimum floor so it keeps rising)
                desired = (target_idx / max(idx, 1.0)) ** (1.0 / n_days) - 1.0
                daily_growth = max(desired, 0.0005)  # ≥ 0.05% per day minimal rise

                for d in seg_dates:
                    # small noise but never allow negative net growth on the day
                    noise = float(rng.normal(0.0000, 0.0004))
                    net = max(0.0, daily_growth + noise)
                    idx = idx * (1.0 + net)
                    idx = max(1.0, idx)  # never below baseline
                    ts_rows.append({
                        "Vessel": vessel, "Type": vtype,
                        "Date": d, "ResistanceIndex": float(idx),
                    })

            # Apply event at boundary (except after the final segment)
            if si < len(seg_boundaries) - 2:
                e = sorted(events, key=lambda x: x["DateTime"])[si]
                eff = EVENT_TYPES[e["Type"]]["effect"]  # negative
                idx = idx * (1.0 + eff)
                idx = max(1.0, idx)  # clamp to baseline

        # End for vessel

    ts_df = pd.DataFrame(ts_rows)
    ev_df = pd.DataFrame(ev_rows).sort_values(["Vessel", "DateTime"]).reset_index(drop=True)
    return ts_df, ev_df
