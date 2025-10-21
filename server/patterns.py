
import ast
import csv
import json
import math
import numpy as np
import pandas as pd
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


def load_patterns_csv(patterns_csv_path: str,
                      placekey_to_place_id: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Load only rows whose placekey exists in papdata['places'], and build:
      stats[place_id] = {
        'median_dwell_hours': int,
        'hour_weights': [24 floats sum=1.0],
        'day_weights':  {weekday: float, ...} normalized to sum=1.0 across 7 days
      }
    Uses chunked pandas to handle full files.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    if not placekey_to_place_id:
        return stats

    needed = set(placekey_to_place_id.keys())

    usecols = [
        "placekey",
        "median_dwell",
        "popularity_by_hour",
        "popularity_by_day",
    ]

    for chunk in pd.read_csv(patterns_csv_path, usecols=usecols, chunksize=20000):
        # Filter to places we actually have in papdata
        subset = chunk[chunk["placekey"].isin(needed)]
        for _, row in subset.iterrows():
            placekey = row["placekey"]
            place_id = placekey_to_place_id.get(placekey)
            if place_id is None:
                continue

            median_dwell_minutes = row.get("median_dwell", None)
            # sometimes is NaN; replace with 60 minutes
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


def gen_patterns(papdata: Dict[str, Any], start_time: datetime, duration: int = 168) -> Dict[str, Any]:
    """
    Simulate, hour-by-hour, moving people from homes to places using SafeGraph-like stats:
      - popularity_by_hour (time-of-day)
      - popularity_by_day (day-of-week)
      - median_dwell (stay length in hours = ceil(median_dwell/60))
    Inputs:
      papdata: dict with keys 'people', 'homes', 'places' (already loaded)
      start_time: simulation start timestamp (datetime)
      duration: hours to simulate
    Output format matches the original: a dict keyed by cumulative minutes,
    each mapping to {"homes": {home_id: [person_ids]}, "places": {place_id: [person_ids]}}.
    """
    # Map placekey -> pap place_id
    placekey_to_place_id: Dict[str, str] = {}
    for pid, desc in papdata.get("places", {}).items():
        pk = desc.get("placekey")
        if pk:
            placekey_to_place_id[pk] = pid

    # Load CSV stats only for the known places
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "./data/patterns.csv")
    stats = load_patterns_csv(csv_path, placekey_to_place_id)

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

        # People who need to leave now
        for pid, st in people_state.items():
            if st["at_place"] and st["leave_time_idx"] == hour_idx:
                # send home
                st["at_place"] = False
                st["place_id"] = None
                st["leave_time_idx"] = None

        # Build intensity distribution for this hour
        place_ids, raw_weights = _overall_busy_factor(stats, weekday, hour_of_day)
        if raw_weights:
            place_probs = np.array(raw_weights, dtype=float)
            place_probs = place_probs / place_probs.sum()
            overall_busy = float(place_probs.max())  # quick proxy for busyness
        else:
            place_probs = np.array([])
            overall_busy = 0.0

        # Move-from-home probability scales with overall busyness (capped)
        # You can tune the multiplier for more/less mobility.
        base_move_prob = min(0.35, 0.15 + 0.85 * overall_busy)

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
