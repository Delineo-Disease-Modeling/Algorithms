import ast
import csv
import json
import math
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Cache for CBG centroid lookup to avoid reloading the YAML repeatedly
_CBG_CENTROIDS: Optional[Dict[str, Tuple[float, float]]] = None

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


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Compute great-circle distance in kilometers between two (lat, lon) pairs.
    """
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _sample_truncated_normal_float(mean: float,
                                   std: float,
                                   lower: float,
                                   upper: float,
                                   rng: np.random.Generator) -> float:
    """
    Sample a float from a normal distribution and clamp to [lower, upper].
    """
    raw = rng.normal(loc=mean, scale=max(0.001, std))
    return float(max(lower, min(upper, raw)))


def _sample_truncated_normal(mean: float,
                             std: float,
                             lower: int,
                             upper: int,
                             rng: np.random.Generator) -> int:
    """
    Draw a single integer sample from a normal distribution but clamp to [lower, upper].
    This keeps times realistic (e.g., within a 0-23 hour window).
    """
    raw = rng.normal(loc=mean, scale=max(0.1, std))
    return int(max(lower, min(upper, round(raw))))


def _resolve_cbg_info_path() -> Optional[str]:
    """
    Locate the cbg_info.yaml file. We check a few likely locations so running
    from different working directories still succeeds.
    """
    import os

    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "../data/cbg_info.yaml"),
        os.path.join(here, "data/cbg_info.yaml"),
        os.path.join(here, "../output/cbg_info.yaml"),
        os.path.join(here, "output/cbg_info.yaml"),
        os.path.join(os.getcwd(), "data/cbg_info.yaml"),
        os.path.join(os.getcwd(), "output/cbg_info.yaml"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def _load_cbg_centroids() -> Dict[str, Tuple[float, float]]:
    """
    Load CBG centroid locations from YAML once and cache them.
    Expected formats (any of these will be handled):
      - dict: {cbg: {"lat": .., "lon": ..}}
      - list of dicts: each with GEOID10/cbg and lat/lon style fields
    """
    global _CBG_CENTROIDS
    if _CBG_CENTROIDS is not None:
        return _CBG_CENTROIDS

    path = _resolve_cbg_info_path()
    centroids: Dict[str, Tuple[float, float]] = {}
    if path is None:
        _CBG_CENTROIDS = centroids
        return centroids

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except Exception:
        _CBG_CENTROIDS = centroids
        return centroids

    def _extract_entry(entry: Any):
        cbg = entry.get("GEOID10") or entry.get("cbg") or entry.get("cbg_id")
        lat = entry.get("lat") or entry.get("latitude") or entry.get("centroid_lat")
        lon = entry.get("lon") or entry.get("lng") or entry.get("longitude") or entry.get("centroid_lon")
        if cbg is None or lat is None or lon is None:
            return
        try:
            centroids[str(cbg)] = (float(lat), float(lon))
        except Exception:
            return

    if isinstance(raw, dict):
        for cbg, entry in raw.items():
            if isinstance(entry, dict):
                lat = entry.get("lat") or entry.get("latitude") or entry.get("centroid_lat")
                lon = entry.get("lon") or entry.get("lng") or entry.get("longitude") or entry.get("centroid_lon")
                if lat is None or lon is None:
                    continue
                try:
                    centroids[str(cbg)] = (float(lat), float(lon))
                except Exception:
                    continue
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                _extract_entry(entry)

    _CBG_CENTROIDS = centroids
    return centroids


def _bucket_places_by_category(places: Dict[str, Any],
                               stats: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Group places (only ones that have patterns stats) into rough activity buckets.
    We use lightweight string checks on top_category to keep the logic data-driven.
    """
    buckets = {"school": [], "college": [], "work": []}
    for pid, meta in places.items():
        # Only keep places that actually have usage stats loaded
        if pid not in stats:
            continue
        cat = (meta.get("top_category") or "").lower()
        if "college" in cat or "university" in cat:
            buckets["college"].append(pid)
            continue
        if "school" in cat:
            buckets["school"].append(pid)
            continue
        if "day care" in cat or "child care" in cat:
            buckets["school"].append(pid)
            continue
        # Everything else is a potential workplace
        buckets["work"].append(pid)

    # Provide broad fallbacks so we always have something to choose from
    if not buckets["work"]:
        buckets["work"] = list(stats.keys())
    if not buckets["school"]:
        buckets["school"] = buckets["work"][:]
    if not buckets["college"]:
        buckets["college"] = buckets["school"][:]
    return buckets


def _choose_anchor_place_random(preferred: str,
                                buckets: Dict[str, List[str]],
                                stats: Dict[str, Dict[str, Any]],
                                rng: np.random.Generator) -> Optional[str]:
    """
    Legacy random selection within preferred bucket; falls back to other buckets or all stats.
    """
    options = buckets.get(preferred, [])
    if not options:
        if preferred == "college":
            options = buckets.get("school", []) or buckets.get("work", [])
        elif preferred == "school":
            options = buckets.get("work", [])
    if not options:
        options = list(stats.keys())
    if not options:
        return None
    return str(rng.choice(options))


def _choose_anchor_place(age: int,
                         home_cbg: Optional[str],
                         buckets: Dict[str, List[str]],
                         stats: Dict[str, Dict[str, Any]],
                         places: Dict[str, Any],
                         cbg_centroids: Dict[str, Tuple[float, float]],
                         rng: np.random.Generator) -> Optional[str]:
    """
    Pick a "primary" place for the person using age (which bucket) and home location:
      - 5-17  -> school
      - 18-22 -> college (falls back to school/work)
      - 23-65 -> work
      - others stay home (no anchor)
    If we know the home CBG centroid, we sample a target commute distance (normal, truncated)
    and choose the place whose distance to home is closest to that target. If we cannot compute
    distance, we fall back to the legacy random bucket choice.
    """
    if age is None:
        return None
    if age < 5:
        return None  # too young; keep them home
    if age < 18:
        preferred = "school"
        dist_params = (2.0, 1.0, 0.2, 10.0)
    elif age < 23:
        preferred = "college"
        dist_params = (5.0, 3.0, 0.5, 30.0)
    elif age <= 65:
        preferred = "work"
        dist_params = (15.0, 8.0, 0.5, 60.0)
    else:
        return None  # retirement age; leave flexible

    # Fallback if we cannot locate the home centroid
    if not cbg_centroids or not home_cbg or home_cbg not in cbg_centroids:
        return _choose_anchor_place_random(preferred, buckets, stats, rng)

    home_coord = cbg_centroids.get(home_cbg)
    if home_coord is None:
        return _choose_anchor_place_random(preferred, buckets, stats, rng)

    # Choose the applicable bucket (with the same fallbacks as the legacy logic)
    candidate_ids = buckets.get(preferred, [])
    if not candidate_ids:
        if preferred == "college":
            candidate_ids = buckets.get("school", []) or buckets.get("work", [])
        elif preferred == "school":
            candidate_ids = buckets.get("work", [])
    if not candidate_ids:
        return _choose_anchor_place_random(preferred, buckets, stats, rng)

    candidates: List[Tuple[str, float]] = []
    for pid in candidate_ids:
        meta = places.get(pid, {})
        lat = meta.get("latitude")
        lon = meta.get("longitude")
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            continue
        dist_km = _haversine_km(home_coord, (lat_f, lon_f))
        candidates.append((pid, dist_km))

    if not candidates:
        return _choose_anchor_place_random(preferred, buckets, stats, rng)

    # Sample a target commute distance and pick the closest candidate
    mean, std, dmin, dmax = dist_params
    target = _sample_truncated_normal_float(mean, std, dmin, dmax, rng)
    # Find candidates closest to target; break ties randomly
    min_diff = None
    closest: List[str] = []
    for pid, dist in candidates:
        diff = abs(dist - target)
        if min_diff is None or diff < min_diff - 1e-6:
            min_diff = diff
            closest = [pid]
        elif min_diff is not None and abs(diff - min_diff) <= 1e-6:
            closest.append(pid)

    if closest:
        return str(rng.choice(closest))
    return _choose_anchor_place_random(preferred, buckets, stats, rng)


def _build_anchor_schedule(people: Dict[str, Any],
                           homes: Dict[str, Any],
                           places: Dict[str, Any],
                           stats: Dict[str, Dict[str, Any]],
                           cbg_centroids: Dict[str, Tuple[float, float]],
                           rng: np.random.Generator) -> Dict[int, Dict[str, Any]]:
    """
    Build a weekly "anchor" schedule for people:
      - assigns each person a primary place (school/college/work) based on age
        and home CBG location (commute distance model)
      - generates weekday blocks with normal-distributed start/duration
      - weekends are intentionally skipped to avoid school/work on Sat/Sun
    Returned structure:
      anchor[person_id] = {
        "place_id": str,
        "by_weekday": {weekday_index: {"start_hour": int, "duration": int}}
      }
    """
    buckets = _bucket_places_by_category(places, stats)
    anchor: Dict[int, Dict[str, Any]] = {}
    weekday_indices = list(range(5))  # Monday-Friday only

    for sid, info in people.items():
        pid = int(sid)
        age = _safe_int(info.get("age"), default=0)
        home_id = info.get("home")
        home_desc = homes.get(str(home_id)) if home_id is not None else None
        home_cbg = None
        if home_desc:
            cbg_val = home_desc.get("cbg")
            if cbg_val is not None:
                home_cbg = str(cbg_val)

        place_id = _choose_anchor_place(
            age,
            home_cbg,
            buckets,
            stats,
            places,
            cbg_centroids,
            rng,
        )
        if place_id is None:
            continue

        # Age-tailored timing parameters
        if age < 18:
            start_mean, start_std = 8.0, 0.8     # kids start ~8am
            dur_mean, dur_std = 6.5, 1.0         # school day ~6-7h
        elif age < 23:
            start_mean, start_std = 9.5, 1.2     # college mornings start later
            dur_mean, dur_std = 5.0, 1.5         # shorter, more varied days
        else:
            start_mean, start_std = 8.5, 1.0     # workday starts around 8:30
            dur_mean, dur_std = 8.5, 1.5         # 8-9h workdays

        by_weekday: Dict[int, Dict[str, int]] = {}
        # Optionally make college schedules a little sparser (e.g., 4 days / week)
        active_days = weekday_indices[:]
        if 18 <= age < 23:
            # randomly drop one weekday to mimic no-class day
            drop_one = rng.choice(active_days)
            active_days = [d for d in active_days if d != drop_one]

        for wd in active_days:
            start_hour = _sample_truncated_normal(start_mean, start_std, 6, 20, rng)
            duration = _sample_truncated_normal(dur_mean, dur_std, 3, 12, rng)
            by_weekday[wd] = {"start_hour": start_hour, "duration": duration}

        anchor[pid] = {"place_id": place_id, "by_weekday": by_weekday}
    return anchor


def _sample_dwell_hours(median_hours: int, rng: np.random.Generator) -> int:
    """
    Sample a dwell time using a normal distribution centered on the median.
    The std dev scales with the median to add realistic spread but is clamped
    so we stay in [1, 12] hours.
    """
    std = max(0.5, median_hours * 0.35)
    sampled = _sample_truncated_normal(median_hours, std, 1, 12, rng)
    return int(max(1, sampled))


def _resolve_patterns_csv_path() -> str:
    """
    Locate the patterns.csv file. We try a few common project-relative paths so
    running this module directly (python server/patterns.py) does not fail.
    """
    import os

    env_path = os.environ.get("PATTERNS_CSV")
    if env_path:
        if os.path.exists(env_path):
            return os.path.abspath(env_path)
        raise FileNotFoundError(f"PATTERNS_CSV was set but not found: {env_path}")

    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "../data/patterns.csv"),  # project-root data folder
        os.path.join(here, "data/patterns.csv"),     # sibling data folder
        os.path.join(os.getcwd(), "data/patterns.csv"),  # cwd-based
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError("Could not find data/patterns.csv; checked: {}".format(candidates))


def _resolve_csv_usecols(csv_path: str,
                         desired: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Resolve desired column names against the CSV header, case-insensitively.
    Returns (usecols, rename_map) for pandas read_csv.
    """
    header = pd.read_csv(csv_path, nrows=0)
    lower_map = {c.lower(): c for c in header.columns}
    missing = [name for name in desired if name.lower() not in lower_map]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    usecols = [lower_map[name.lower()] for name in desired]
    rename_map = {lower_map[name.lower()]: name for name in desired}
    return usecols, rename_map


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

    desired_cols = [
        "placekey",
        "median_dwell",
        "popularity_by_hour",
        "popularity_by_day",
    ]
    usecols, rename_map = _resolve_csv_usecols(patterns_csv_path, desired_cols)

    for chunk in pd.read_csv(patterns_csv_path, usecols=usecols, chunksize=20000):
        chunk = chunk.rename(columns=rename_map)
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
    Added behavior:
      - age-based "anchor" schedules that send kids to school, young adults to college,
        and adults to work on weekdays with normal-distributed start/duration windows.
      - weekends skip these anchors to keep people home or on leisure trips instead.
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
    csv_path = _resolve_patterns_csv_path()
    stats = load_patterns_csv(csv_path, placekey_to_place_id)

    rng = np.random.default_rng(seed=42)  # deterministic but easy to change/remove

    # Build age-aware anchor schedules (school/college/work) so people have
    # predictable daytime locations on weekdays. This is computed once so the
    # simulation uses the same "primary" place throughout a run.
    anchor_schedule = _build_anchor_schedule(
        papdata.get("people", {}),
        papdata.get("homes", {}),
        papdata.get("places", {}),
        stats,
        _load_cbg_centroids(),
        rng,
    )

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

    for hour_idx in range(duration):
        current_time = start_time + timedelta(hours=hour_idx)
        weekday = WEEKDAYS[current_time.weekday()]
        hour_of_day = current_time.hour
        day_offset = hour_idx - hour_of_day  # start index of the current calendar day

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

            # Check if this hour sits inside the person's weekday anchor window
            in_anchor_window = False
            anchor_info = anchor_schedule.get(pid)
            if anchor_info:
                plan = anchor_info["by_weekday"].get(current_time.weekday())
                if plan:
                    start_hour = plan["start_hour"]
                    end_hour = min(24, start_hour + plan["duration"])
                    planned_start_idx = day_offset + start_hour
                    planned_end_idx = min(duration, day_offset + end_hour)
                    if planned_start_idx <= hour_idx < planned_end_idx:
                        in_anchor_window = True
                        # Force them to their primary place and set the leave time
                        st["at_place"] = True
                        st["place_id"] = anchor_info["place_id"]
                        st["leave_time_idx"] = planned_end_idx
                    elif st["place_id"] == anchor_info["place_id"] and hour_idx >= planned_end_idx:
                        # If somehow lingering past schedule, mark for immediate return
                        st["leave_time_idx"] = hour_idx

            if st["at_place"]:
                continue  # anchored for this block

            if in_anchor_window:
                continue  # should already be moved above

            if len(place_ids) == 0:
                continue  # nothing open/busy for this hour

            if rng.random() < base_move_prob:
                # pick a destination from the busy distribution
                choice_idx = int(rng.choice(len(place_ids), p=place_probs))
                dest_place_id = place_ids[choice_idx]

                # dwell time from median_dwell using a normal distribution
                median_hours = stats[dest_place_id]["median_dwell_hours"]
                dwell_hours = _sample_dwell_hours(median_hours, rng)

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
