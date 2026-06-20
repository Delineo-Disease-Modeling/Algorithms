
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


_PANEL_DEFAULT = 19.7   # SafeGraph panel->population factor (OK ~19.7); overridden from data
_MONTH_DAYS = 30.0      # popularity_by_hour is a monthly device-hours total


def _movement_scale(stats_df) -> float:
    """Absolute occupancy scale: popularity_by_hour (monthly sample device-hours)
    -> typical-day population occupancy = popularity * panel / days. The panel
    factor is read from the data (normalized_visits_by_state_scaling / raw_visit
    _counts, ~constant per state); DELINEO_MOVEMENT_SCALE overrides the whole
    factor for calibration."""
    override = os.getenv("DELINEO_MOVEMENT_SCALE", "")
    if override.strip():
        try:
            return float(override)
        except ValueError:
            pass
    panel = _PANEL_DEFAULT
    try:
        nsc = pd.to_numeric(stats_df.get("normalized_visits_by_state_scaling"), errors="coerce")
        rvc = pd.to_numeric(stats_df.get("raw_visit_counts"), errors="coerce")
        ratio = (nsc / rvc).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            panel = float(ratio.median())
    except Exception:
        pass
    return panel / _MONTH_DAYS


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


_DAY_ABBR = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
             "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}

# Fallback open windows by 2-digit NAICS sector (open_hour inclusive,
# close_hour exclusive, local time), used only when a POI has no open_hours of
# its own. Deliberately broad; these are guesses, so they gate SOFTLY (_OPEN_EPS)
# rather than hard-zeroing.
_NAICS_DEFAULT_HOURS = {
    "44": (8, 21), "45": (8, 21),                  # retail
    "72": (6, 23),                                 # accommodation & food service
    "71": (9, 22),                                 # arts, entertainment, recreation
    "61": (7, 18),                                 # educational services
    "62": (7, 18),                                 # health care
    "81": (8, 18),                                 # other services
    "92": (8, 17),                                 # public administration
    "51": (8, 18), "52": (8, 18), "53": (8, 18),
    "54": (8, 18), "55": (8, 18), "56": (8, 18),   # info/finance/realestate/prof/admin
    "23": (7, 17),                                 # construction
    "31": (6, 18), "32": (6, 18), "33": (6, 18),   # manufacturing
}
_OPEN_EPS = 1e-3   # weight multiplier for hours gated CLOSED by a NAICS default


def _parse_clock(s) -> Optional[float]:
    """'8:00' -> 8.0, '17:30' -> 17.5, '24:00' -> 24.0. None if unparseable."""
    try:
        s = str(s).strip()
        if not s:
            return None
        hh, _, mm = s.partition(":")
        return int(hh) + (int(mm) / 60.0 if mm else 0.0)
    except Exception:
        return None


def _open_hours_to_sets(val) -> Optional[Dict[str, set]]:
    """Parse SafeGraph open_hours into {weekday_full: set(open hour-of-day ints)}.
    Returns None when the field is missing or unparseable (caller then falls back
    to a NAICS default or to no gating)."""
    if isinstance(val, dict):
        d = val
    elif val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    else:
        try:
            d = json.loads(val)
            if isinstance(d, str):
                d = json.loads(d)
        except Exception:
            return None
    if not isinstance(d, dict) or not d:
        return None
    out: Dict[str, set] = {}
    for abbr, intervals in d.items():
        wd = _DAY_ABBR.get(str(abbr)[:3].title())
        if wd is None or not isinstance(intervals, list):
            continue
        hours: set = set()
        for iv in intervals:
            if not (isinstance(iv, (list, tuple)) and len(iv) == 2):
                continue
            o = _parse_clock(iv[0])
            c = _parse_clock(iv[1])
            if o is None or c is None:
                continue
            if c == o:                       # e.g. 0:00-0:00 == open all day
                hours.update(range(24))
                continue
            spans = [(o, c)] if c > o else [(o, 24.0), (0.0, c)]   # wrap past midnight
            for a, b in spans:
                for h in range(24):
                    if a < h + 1 and b > h:  # interval overlaps the clock-hour [h, h+1)
                        hours.add(h)
        if hours:
            out[wd] = hours
    return out or None


def _build_open_gate(open_hours_val, naics_code) -> Dict[str, List[float]]:
    """Per-(weekday, hour) weight multiplier for the destination kernel.

    - POI has real open_hours  -> HARD gate (closed hours = 0.0); the data is
      authoritative, so a 3am dentist "visit" becomes impossible.
    - else, NAICS-default window -> SOFT gate (closed hours = _OPEN_EPS); the
      window is a guess, so we damp rather than forbid.
    - else (no hours, unknown sector) -> no gating (all 1.0).
    """
    parsed = _open_hours_to_sets(open_hours_val)
    if parsed is not None:
        return {wd: [1.0 if h in parsed.get(wd, set()) else 0.0 for h in range(24)]
                for wd in WEEKDAYS}
    sector = None
    if naics_code is not None:
        ns = str(naics_code).strip()
        if ns and ns.lower() != "nan":
            sector = ns[:2]
    window = _NAICS_DEFAULT_HOURS.get(sector) if sector else None
    if window is None:
        return {wd: [1.0] * 24 for wd in WEEKDAYS}
    lo, hi = window
    row = [1.0 if lo <= h < hi else _OPEN_EPS for h in range(24)]
    return {wd: list(row) for wd in WEEKDAYS}


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


def _parse_cbg_counts(val) -> Dict[str, float]:
    """Parse a SafeGraph {cbg: visitor_count} field (visitor_home_cbgs) into a
    dict keyed by zero-padded 12-digit CBG. Empty dict if missing/unparseable."""
    if isinstance(val, dict):
        d = val
    elif val is None or (isinstance(val, float) and pd.isna(val)):
        return {}
    else:
        try:
            d = json.loads(val)
            if isinstance(d, str):
                d = json.loads(d)
        except Exception:
            return {}
    if not isinstance(d, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k).strip().zfill(12)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _catchment_fraction(home_cbgs_val, cluster_cbgs: set) -> Optional[float]:
    """f_j: the fraction of a POI's observed visitors whose home CBG is inside the
    simulated population. Returns None when visitor_home_cbgs is missing/empty so
    the caller can substitute the per-run fallback."""
    counts = _parse_cbg_counts(home_cbgs_val)
    total = sum(counts.values())
    if total <= 0:
        return None
    inside = sum(v for cbg, v in counts.items() if cbg in cluster_cbgs)
    return inside / total


def _build_stats_from_df(df: pd.DataFrame,
                         placekey_to_place_id: Dict[str, str],
                         cluster_cbgs: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """
    Build per-place stats from a DataFrame that already has the needed columns
    (placekey, median_dwell, popularity_by_hour, popularity_by_day).
    Column names must already be lowercase.

    When ``cluster_cbgs`` is given, also compute each POI's catchment fraction
    ``catchment_fj`` (Stage 2) — observed from visitor_home_cbgs where present,
    else the median observed fraction across the zone (a per-run flat fallback;
    the no-data POIs are low-traffic and carry ~0 weight regardless).
    """
    cluster_cbgs = cluster_cbgs or set()
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

        day_counts = [float(day_map.get(k, 0)) for k in WEEKDAYS]
        # Mean-normalized weekday shape (~1.0 on an average day) for the absolute
        # occupancy target; keeps magnitude while reflecting weekday variation.
        mean_day = (sum(day_counts) / 7.0) if day_counts else 0.0
        if mean_day > 0:
            day_factor = {k: day_counts[i] / mean_day for i, k in enumerate(WEEKDAYS)}
        else:
            day_factor = {k: 1.0 for k in WEEKDAYS}

        stats[place_id] = {
            "median_dwell_hours": _ceil_hours_from_minutes(median_dwell_minutes),
            "day_factor": day_factor,
            "raw_hour_counts": hour_list,
            # open-hours gate (hard for real hours, soft NAICS-default otherwise).
            "open_gate": _build_open_gate(
                row.get("open_hours", None), row.get("naics_code", None)),
            # catchment fraction f_j (None -> filled with the fallback below).
            "catchment_fj": _catchment_fraction(
                row.get("visitor_home_cbgs", None), cluster_cbgs),
        }

    # Fill missing f_j with the median observed fraction (flat per-run fallback).
    observed = [s["catchment_fj"] for s in stats.values() if s["catchment_fj"] is not None]
    fallback = float(np.median(observed)) if observed else 1.0
    for s in stats.values():
        if s["catchment_fj"] is None:
            s["catchment_fj"] = fallback
    return stats


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

        # Home CBGs of the simulated population, for Stage 2 catchment scaling.
        cluster_cbgs = set()
        for home in papdata.get("homes", {}).values():
            cbg = (home.get("cbg") if isinstance(home, dict)
                   else (home[0] if isinstance(home, (list, tuple)) and home else home))
            if cbg is not None:
                cluster_cbgs.add(str(cbg).strip().zfill(12))

        # Movement REQUIRES SafeGraph patterns data. Fail loudly rather than
        # silently produce a degenerate everyone-stays-home run.
        if shared_data is None or shared_data.is_empty():
            raise ValueError(
                "gen_patterns requires non-empty patterns data (shared_data); "
                "refusing to generate a movement-free run.")
        placekey_set = set(placekey_to_place_id.keys())
        stats_df = shared_data.for_patterns_stats(placekey_set)
        stats = _build_stats_from_df(stats_df, placekey_to_place_id, cluster_cbgs)

        # Demand-pull precompute: a stable global place ordering plus the per-POI
        # inputs to the occupancy target, as numpy matrices so each hour's fill is
        # a vectorized op.
        all_place_ids: List[str] = list(stats.keys())
        n_places_total = len(all_place_ids)
        place_pos = {pid: i for i, pid in enumerate(all_place_ids)}
        if n_places_total == 0:
            raise ValueError(
                "gen_patterns: no POIs with usable stats for this zone's placekeys; "
                "cannot generate movement.")
        movement_scale = _movement_scale(stats_df)
        if not os.getenv("DELINEO_MOVEMENT_SCALE", "").strip():
            _PATTERNS_LOGGER.warning(
                "movement_scale=%.4f is the data-derived default (panel/days) and is "
                "UNCALIBRATED; set DELINEO_MOVEMENT_SCALE from validation.", movement_scale)
        pop_hour_mat = np.array([stats[p]["raw_hour_counts"] for p in all_place_ids], dtype=float)        # (P,24)
        day_factor_mat = np.array([[stats[p]["day_factor"][wd] for wd in WEEKDAYS] for p in all_place_ids], dtype=float)  # (P,7)
        fj_arr = np.array([stats[p].get("catchment_fj", 1.0) for p in all_place_ids], dtype=float)        # (P,)
        gate_mat = np.array([[[stats[p]["open_gate"][wd][h] for h in range(24)] for wd in WEEKDAYS]
                             for p in all_place_ids], dtype=float)                                        # (P,7,24)
        dwell_arr = np.array([stats[p]["median_dwell_hours"] for p in all_place_ids], dtype=np.int64)     # (P,)

        # People state held in parallel arrays so the mover-decision hot loop
        # can run as a batched numpy op instead of one rng.random + rng.choice
        # + rng.integers call per person. is_home_arr is the source of truth
        # for whether each person is at home or out; leave_time_arr holds the
        # hour at which they'll return; dest_idx_arr holds the destination's
        # global place index (only meaningful when out).
        people = list(papdata.get("people", {}).items())
        n_people = len(people)
        pid_str_list: List[str] = [None] * n_people  # str(int(sid))
        home_str_list: List[str] = [None] * n_people  # str(info["home"])
        for i, (sid, info) in enumerate(people):
            pid_str_list[i] = str(int(sid))
            home_str_list[i] = str(info.get("home"))
        is_home_arr = np.ones(n_people, dtype=bool)
        leave_time_arr = np.full(n_people, -1, dtype=np.int64)
        # Destination as a global place index (-1 = home), so current occupancy
        # is a vectorized bincount and the snapshot maps index -> place_id.
        dest_idx_arr = np.full(n_people, -1, dtype=np.int64)

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
                # dest_idx_arr entries are not cleared on return; they are only
                # read when is_home_arr[i] is False, so stale values are unreachable.

        # Demand-pull: fill each POI to its realistic occupancy target
        # O_j = popularity_by_hour * day_factor * f_j * open_gate * scale by topping
        # up from the home pool. Occupancy IS the target (not a move-rate flood
        # re-normalized), so no POI over-fills and everyone not pulled stays home.
        with _timed("gen_patterns/demand_pull"):
            wd_idx = WEEKDAYS.index(weekday)
            targets = (pop_hour_mat[:, hour_of_day] * day_factor_mat[:, wd_idx]
                       * gate_mat[:, wd_idx, hour_of_day] * fj_arr * movement_scale)
            target_int = np.floor(targets + 0.5).astype(np.int64)

            out_mask = ~is_home_arr
            current_occ = (np.bincount(dest_idx_arr[out_mask], minlength=n_places_total)
                           if out_mask.any() else np.zeros(n_places_total, dtype=np.int64))
            needed = np.maximum(0, target_int - current_occ)
            total_needed = int(needed.sum())

            home_indices = np.where(is_home_arr)[0]
            n_home = int(home_indices.size)
            if total_needed > 0 and n_home > 0:
                if total_needed > n_home:
                    # More demand than residents available: fill proportionally.
                    needed = np.floor(needed * (n_home / total_needed)).astype(np.int64)
                    total_needed = int(needed.sum())
                if total_needed > 0:
                    dest_assign = np.repeat(np.arange(n_places_total), needed)
                    movers = rng.choice(home_indices, size=total_needed, replace=False)
                    med = dwell_arr[dest_assign]
                    lo = np.maximum(1, med - 1)
                    dwell_hours_arr = rng.integers(lo, med + 2)
                    is_home_arr[movers] = False
                    dest_idx_arr[movers] = dest_assign
                    leave_time_arr[movers] = np.minimum(duration, hour_idx + dwell_hours_arr)

        # Snapshot at the end of this hour
        with _timed("gen_patterns/snapshot_assembly"):
            homes_map: Dict[str, List[str]] = {}
            places_map: Dict[str, List[str]] = {}
            # tolist() up front turns 50k numpy bool indexing into 50k local
            # truthiness checks, which is measurably faster in CPython.
            is_home_list = is_home_arr.tolist()
            dest_idx_list = dest_idx_arr.tolist()
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
                    place_id = all_place_ids[dest_idx_list[i]]
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
