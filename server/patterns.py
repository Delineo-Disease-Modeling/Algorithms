
import ast
import contextlib
import csv
import json
import logging
import math
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_PATTERNS_LOGGER = logging.getLogger(__name__)


def _perf_timings_enabled() -> bool:
    return os.getenv("DELINEO_PERF_TIMINGS", "").lower() in {"1", "true", "yes", "on"}


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
    perf_accum: Dict[str, float] = {}
    perf_on = _perf_timings_enabled()

    @contextlib.contextmanager
    def _timed(label: str):
        if not perf_on:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            perf_accum[label] = perf_accum.get(label, 0.0) + (time.perf_counter() - started)

    with _timed("gen_patterns/setup_stats"):
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

        # People state held in parallel arrays so the mover-decision hot loop
        # can run as a batched numpy op instead of one rng.random + rng.choice
        # + rng.integers call per person. is_home_arr is the source of truth
        # for whether each person is at home or out; leave_time_arr holds the
        # hour at which they'll return; dest_place_id_arr holds the destination
        # place_id (only meaningful when at place).
        people = list(papdata.get("people", {}).items())
        n_people = len(people)
        pid_str_list: List[str] = [None] * n_people  # str(int(sid))
        home_str_list: List[str] = [None] * n_people  # str(info["home"])
        for i, (sid, info) in enumerate(people):
            pid_str_list[i] = str(int(sid))
            home_str_list[i] = str(info.get("home"))
        is_home_arr = np.ones(n_people, dtype=bool)
        leave_time_arr = np.full(n_people, -1, dtype=np.int64)
        dest_place_id_arr: List[Any] = [None] * n_people

    output: Dict[str, Any] = {}

    rng = np.random.default_rng(seed=42)  # deterministic but easy to change/remove

    for hour_idx in range(duration):
        current_time = start_time + timedelta(hours=hour_idx)
        weekday = WEEKDAYS[current_time.weekday()]
        hour_of_day = current_time.hour

        # People whose dwell time expires this hour return home.
        with _timed("gen_patterns/leave_time_expiry"):
            expired_mask = (leave_time_arr == hour_idx)
            if expired_mask.any():
                is_home_arr[expired_mask] = True
                leave_time_arr[expired_mask] = -1
                # dest_place_id_arr entries are not cleared on return; they
                # are only read when is_home_arr[i] is False, so stale values
                # are unreachable.

        # Build intensity distribution for this hour (normalized — for destination selection)
        with _timed("gen_patterns/intensity_distribution"):
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
        with _timed("gen_patterns/aggregate_busyness"):
            if peak_busyness > 0:
                overall_busy = _aggregate_busyness(stats, weekday, hour_of_day) / peak_busyness
            else:
                overall_busy = 0.0

            base_move_prob = min(0.35, 0.85 * overall_busy)

        # Vectorized mover decision: batch the rng draws across all people at
        # home this hour, then apply assignments only to those who actually
        # move. RNG draw ORDER differs from the original per-person loop, so
        # outputs are NOT byte-identical to the pre-vectorization code; the
        # statistical distribution is preserved (each person has the same
        # base_move_prob; destinations are still drawn with place_probs;
        # dwell still triangular around median-1..median+1).
        with _timed("gen_patterns/mover_decision"):
            home_indices = np.where(is_home_arr)[0]
            n_home = int(home_indices.size)
            n_places = len(place_ids)
            if n_home > 0 and n_places > 0:
                move_rolls = rng.random(n_home)
                movers_mask = move_rolls < base_move_prob
                n_movers = int(movers_mask.sum())
                if n_movers > 0:
                    mover_indices = home_indices[movers_mask]
                    dest_choice_idx = rng.choice(n_places, size=n_movers, p=place_probs)
                    dest_choice_list = dest_choice_idx.tolist()
                    dest_place_ids_for_movers = [place_ids[c] for c in dest_choice_list]
                    median_per_mover = np.fromiter(
                        (stats[pid]["median_dwell_hours"] for pid in dest_place_ids_for_movers),
                        dtype=np.int64,
                        count=n_movers,
                    )
                    lo = np.maximum(1, median_per_mover - 1)
                    hi_inclusive = median_per_mover + 1
                    dwell_hours_arr = rng.integers(lo, hi_inclusive + 1)
                    new_leave_time = np.minimum(duration, hour_idx + dwell_hours_arr)

                    is_home_arr[mover_indices] = False
                    leave_time_arr[mover_indices] = new_leave_time
                    for arr_idx, place_id in zip(mover_indices.tolist(), dest_place_ids_for_movers):
                        dest_place_id_arr[arr_idx] = place_id

        # Snapshot at the end of this hour
        with _timed("gen_patterns/snapshot_assembly"):
            homes_map: Dict[str, List[str]] = {}
            places_map: Dict[str, List[str]] = {}
            # tolist() up front turns 50k numpy bool indexing into 50k local
            # truthiness checks, which is measurably faster in CPython.
            is_home_list = is_home_arr.tolist()
            for i in range(n_people):
                pid_str = pid_str_list[i]
                if is_home_list[i]:
                    home_id = home_str_list[i]
                    bucket = homes_map.get(home_id)
                    if bucket is None:
                        homes_map[home_id] = [pid_str]
                    else:
                        bucket.append(pid_str)
                else:
                    place_id = str(dest_place_id_arr[i])
                    bucket = places_map.get(place_id)
                    if bucket is None:
                        places_map[place_id] = [pid_str]
                    else:
                        bucket.append(pid_str)

            # Key is minutes from start_time. hour_idx=0 processes the first hour (0:00-1:00),
            # so we store at minute 60 to represent the state AT hour 1.
            # Frontend interprets key "60" as "1 hour from start" which is correct.
            current_minutes = (hour_idx + 1) * 60
            output[str(current_minutes)] = {"homes": homes_map, "places": places_map}

    if perf_on:
        # Emit via print so the line is visible even when the algorithms logger
        # configuration is changed by callers.
        for label in sorted(perf_accum):
            print(f"[perf] {label}: {perf_accum[label]:.3f}s", flush=True)

    return output


if __name__ == "__main__":
    # Example manual run when used as a script
    with open("./output/papdata.json", "r") as f:
        pap = json.load(f)
        
    patterns = gen_patterns(pap, datetime.now(), 168)
    
    with open("patterns_out.json", "w") as f:
        json.dump(patterns, f, indent=2)
    
    print("Exported patterns_out.json")
