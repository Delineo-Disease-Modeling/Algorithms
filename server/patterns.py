
import ast
import csv
import json
import math
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _parse_hour_list(val) -> List[int]:
    """
    popularity_by_hour is either already a list or a string like "[0,1,...,24 items]".
    Return a 24-length list of ints (fallback to zeros).
    """
    if isinstance(val, list):
        arr = val
    else:
        try:
            arr = ast.literal_eval(val)
            # Handle double-encoded JSON from new CSV files
            if isinstance(arr, str):
                arr = ast.literal_eval(arr)
        except Exception:
            arr = []
    if not isinstance(arr, list) or len(arr) != 24:
        return [0] * 24
    # ensure ints
    return [int(x) if x is not None else 0 for x in arr]


def _parse_day_map(val) -> Dict[str, int]:
    """
    popularity_by_day is a JSON/dict mapping weekday->int. It may be a JSON string.
    """
    if isinstance(val, dict):
        d = val
    else:
        try:
            d = json.loads(val)
            # Handle double-encoded JSON from new CSV files
            if isinstance(d, str):
                d = json.loads(d)
        except Exception:
            try:
                d = ast.literal_eval(val)
            except Exception:
                d = {}
    out = {k: int(v) for k, v in d.items() if k in WEEKDAYS}
    # ensure all keys exist
    for k in WEEKDAYS:
        out.setdefault(k, 0)
    return out


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize(v: List[float]) -> List[float]:
    s = float(sum(v))
    if s <= 0:
        return [0.0 for _ in v]
    return [x / s for x in v]


def _ceil_hours_from_minutes(m: Optional[float]) -> int:
    if m is None:
        return 1
    try:
        m = float(m)
    except Exception:
        return 1
    return max(1, int(math.ceil(m / 60.0)))


def _build_stats_from_df(df: pd.DataFrame,
                         placekey_to_place_id: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Build per-place stats from a DataFrame that already has the needed columns
    (placekey, median_dwell, popularity_by_hour, popularity_by_day).
    Column names must already be lowercase.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    needed = set(placekey_to_place_id.keys())

    subset = df[df["placekey"].isin(needed)]
    for _, row in subset.iterrows():
        placekey = row["placekey"]
        place_id = placekey_to_place_id.get(placekey)
        if place_id is None:
            continue

        median_dwell_minutes = row.get("median_dwell", None)
        if pd.isna(median_dwell_minutes):
            median_dwell_minutes = 60

        hour_list = _parse_hour_list(row.get("popularity_by_hour", "[]"))
        day_map = _parse_day_map(row.get("popularity_by_day", "{}"))

        hour_weights = _normalize([float(x) for x in hour_list])
        day_counts = [float(day_map.get(k, 0)) for k in WEEKDAYS]
        day_weights_list = _normalize(day_counts)
        day_weights = {k: day_weights_list[i] for i, k in enumerate(WEEKDAYS)}

        stats[place_id] = {
            "median_dwell_hours": _ceil_hours_from_minutes(median_dwell_minutes),
            "hour_weights": hour_weights,
            "day_weights": day_weights,
            "raw_hour_counts": hour_list,
            "raw_day_counts": day_map,
        }
    return stats


def _overall_busy_factor(stats: Dict[str, Dict[str, Any]], weekday: str, hour: int) -> Tuple[List[str], List[float]]:
    """
    Build a global "intensity" distribution across places for a given weekday+hour.
    Returns (place_ids, weights) where weights are unnormalized intensities.
    """
    ids: List[str] = []
    wts: List[float] = []
    for pid, s in stats.items():
        hw = s["hour_weights"]
        dw = s["day_weights"].get(weekday, 0.0)
        hour_w = hw[hour] if 0 <= hour < 24 else 0.0
        weight = hour_w * dw
        if weight > 0:
            ids.append(pid)
            wts.append(weight)
    return ids, wts


def _compute_peak_busyness(stats: Dict[str, Dict[str, Any]]) -> float:
    """
    Pre-compute the maximum aggregate raw activity across all 7×24 timeslots.
    For each (weekday, hour) pair, sum raw_hour_counts[hour] * raw_day_counts[weekday]
    across all facilities. Return the maximum such sum.

    This gives us a baseline to normalize against: the busiest hour of the busiest day.
    """
    peak = 0.0
    for weekday in WEEKDAYS:
        for hour in range(24):
            total = 0.0
            for s in stats.values():
                raw_h = s["raw_hour_counts"][hour]
                raw_d = s["raw_day_counts"].get(weekday, 0)
                total += raw_h * raw_d
            if total > peak:
                peak = total
    return peak


def _aggregate_busyness(stats: Dict[str, Dict[str, Any]], weekday: str, hour: int) -> float:
    """
    Compute aggregate raw activity for a specific (weekday, hour) timeslot.
    Sum raw_hour_counts[hour] * raw_day_counts[weekday] across all facilities.
    """
    total = 0.0
    for s in stats.values():
        raw_h = s["raw_hour_counts"][hour]
        raw_d = s["raw_day_counts"].get(weekday, 0)
        total += raw_h * raw_d
    return total


def gen_patterns(papdata: Dict[str, Any], start_time: datetime, duration: int = 168,
                 shared_data=None) -> Dict[str, Any]:
    """
    Simulate, hour-by-hour, moving people from homes to places using SafeGraph-like stats:
      - popularity_by_hour (time-of-day)
      - popularity_by_day (day-of-week)
      - median_dwell (stay length in hours = ceil(median_dwell/60))
    Inputs:
      papdata: dict with keys 'people', 'homes', 'places' (already loaded)
      start_time: simulation start timestamp (datetime)
      duration: hours to simulate
      shared_data: Pre-loaded PatternsData. If None/empty, no place stats are
          available and movement falls back to home-only.
    Output format matches the original: a dict keyed by cumulative minutes,
    each mapping to {"homes": {home_id: [person_ids]}, "places": {place_id: [person_ids]}}.
    """
    # Map placekey -> pap place_id
    placekey_to_place_id: Dict[str, str] = {}
    for pid, desc in papdata.get("places", {}).items():
        pk = desc.get("placekey")
        if pk:
            placekey_to_place_id[pk] = pid

    # Derive stats from the shared PatternsData. If none/empty, stats is empty.
    if shared_data is not None and not shared_data.is_empty():
        placekey_set = set(placekey_to_place_id.keys())
        stats_df = shared_data.for_patterns_stats(placekey_set)
        stats = _build_stats_from_df(stats_df, placekey_to_place_id)
    else:
        stats = {}

    # Pre-compute peak busyness across the entire week so we can derive a
    # meaningful 0-1 movement probability from raw visitor counts.
    peak_busyness = _compute_peak_busyness(stats)

    # People state
    # At any time, a person is either at home or at a place with a 'leave_time' scheduled.
    # (We don't model inter-POI hopping here; that could be added by sampling again after leaving.)
    people_state: Dict[int, Dict[str, Any]] = {}
    for sid, info in papdata.get("people", {}).items():
        pid = int(sid)
        people_state[pid] = {
            "home": str(info.get("home")),
            "at_place": False,
            "place_id": None,
            "leave_time_idx": None,  # hour index when they will leave the place
        }

    output: Dict[str, Any] = {}

    rng = np.random.default_rng(seed=42)  # deterministic but easy to change/remove

    for hour_idx in range(duration):
        current_time = start_time + timedelta(hours=hour_idx)
        weekday = WEEKDAYS[current_time.weekday()]
        hour_of_day = current_time.hour

        # People who need to leave now (normal dwell time expiry)
        for pid, st in people_state.items():
            if st["at_place"] and st["leave_time_idx"] == hour_idx:
                # send home
                st["at_place"] = False
                st["place_id"] = None
                st["leave_time_idx"] = None

        # Build intensity distribution for this hour (normalized — for destination selection)
        place_ids, raw_weights = _overall_busy_factor(stats, weekday, hour_of_day)
        if raw_weights:
            place_probs = np.array(raw_weights, dtype=float)
            place_probs = place_probs / place_probs.sum()
        else:
            place_probs = np.array([])

        # Compute movement probability from raw visitor counts.
        # overall_busy is a 0-1 ratio: current aggregate activity / peak activity
        # across the whole week. This properly captures "how busy is right now"
        # using absolute magnitudes rather than the normalized distribution.
        if peak_busyness > 0:
            overall_busy = _aggregate_busyness(stats, weekday, hour_of_day) / peak_busyness
        else:
            overall_busy = 0.0

        base_move_prob = min(0.35, 0.85 * overall_busy)

        # Decide moves for each person at home
        for pid, st in people_state.items():
            if st["at_place"]:
                continue  # already out

            if len(place_ids) == 0:
                continue  # nothing open/busy for this hour

            if rng.random() < base_move_prob:
                # pick a destination
                choice_idx = int(rng.choice(len(place_ids), p=place_probs))
                dest_place_id = place_ids[choice_idx]

                # dwell time from median_dwell
                median_hours = stats[dest_place_id]["median_dwell_hours"]
                # add a little noise around the median (triangular around [median-1, median+1])
                lo = max(1, median_hours - 1)
                hi = median_hours + 1
                dwell_hours = int(rng.integers(lo, hi + 1))

                st["at_place"] = True
                st["place_id"] = dest_place_id
                st["leave_time_idx"] = min(duration, hour_idx + dwell_hours)

        # Snapshot at the end of this hour
        homes_map: Dict[str, List[str]] = {}
        places_map: Dict[str, List[str]] = {}
        for pid, st in people_state.items():
            if st["at_place"] and st["place_id"] is not None:
                places_map.setdefault(str(st["place_id"]), []).append(str(pid))
            else:
                homes_map.setdefault(str(st["home"]), []).append(str(pid))

        # Key is minutes from start_time. hour_idx=0 processes the first hour (0:00-1:00),
        # so we store at minute 60 to represent the state AT hour 1.
        # Frontend interprets key "60" as "1 hour from start" which is correct.
        current_minutes = (hour_idx + 1) * 60
        output[str(current_minutes)] = {"homes": homes_map, "places": places_map}

    return output


if __name__ == "__main__":
    # Example manual run when used as a script
    with open("./output/papdata.json", "r") as f:
        pap = json.load(f)
        
    patterns = gen_patterns(pap, datetime.now(), 168)
    
    with open("patterns_out.json", "w") as f:
        json.dump(patterns, f, indent=2)
    
    print("Exported patterns_out.json")
