"""Persistent v1 worker assignment for synthetic residents.

This deliberately does not generate a full activity schedule. It only annotates
papdata people with stable employment/workplace metadata so later movement work
can use persistent anchors without re-solving employment.
"""
import json
import os
import random
from statistics import median
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd


EMPLOYMENT_PROB_18_64_DEFAULT = 0.65
EMPLOYMENT_PROB_65_PLUS_DEFAULT = 0.10
MIN_WORKPLACE_AREA_M2 = 50.0
MAX_WORKPLACE_AREA_M2 = 2000.0


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _clip_probability(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return min(1.0, max(0.0, parsed))


def _coerce_age(value: Any) -> Optional[int]:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def employment_probability(age: Any) -> float:
    age_int = _coerce_age(age)
    if age_int is None or age_int < 18:
        return 0.0
    if age_int <= 64:
        return _clip_probability(
            _env_float("DELINEO_WORKER_PROB_18_64", EMPLOYMENT_PROB_18_64_DEFAULT),
            EMPLOYMENT_PROB_18_64_DEFAULT,
        )
    return _clip_probability(
        _env_float("DELINEO_WORKER_PROB_65_PLUS", EMPLOYMENT_PROB_65_PLUS_DEFAULT),
        EMPLOYMENT_PROB_65_PLUS_DEFAULT,
    )


def effective_workplace_area_weight(area: Any) -> float:
    try:
        parsed = float(area)
    except (TypeError, ValueError):
        parsed = MIN_WORKPLACE_AREA_M2
    if parsed != parsed or parsed <= 0:
        parsed = MIN_WORKPLACE_AREA_M2
    return min(MAX_WORKPLACE_AREA_M2, max(MIN_WORKPLACE_AREA_M2, parsed))


def _capture_values(home_origin_capture: Optional[Dict[str, Any]]) -> list[float]:
    values = []
    for raw in (home_origin_capture or {}).values():
        if isinstance(raw, dict):
            raw = raw.get("p_inside")
        clipped = _clip_probability(raw)
        if clipped is not None:
            values.append(float(clipped))
    return values


def _capture_for_cbg(
    home_origin_capture: Optional[Dict[str, Any]],
    cbg: Any,
    fallback: float,
) -> float:
    key = str(cbg).strip().zfill(12)
    raw = (home_origin_capture or {}).get(key)
    if isinstance(raw, dict):
        raw = raw.get("p_inside")
    clipped = _clip_probability(raw)
    return float(fallback if clipped is None else clipped)


def _place_area_weights(places: Dict[str, Dict[str, Any]]) -> tuple[list[str], list[float]]:
    place_ids: list[str] = []
    weights: list[float] = []
    for place_id, place in places.items():
        if not isinstance(place, dict):
            continue
        weight = effective_workplace_area_weight(place.get("area"))
        if weight <= 0:
            continue
        place_ids.append(str(place_id))
        weights.append(float(weight))
    return place_ids, weights


def assign_workers(
    papdata: Dict[str, Any],
    home_origin_capture: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Annotate papdata people with persistent v1 work assignments.

    Employment is age-probability based. In-zone workers are assigned to exactly
    one persistent POI, sampled by capped POI area. Out-of-zone workers are only
    annotated; movement scheduling deliberately does not send them to a synthetic
    facility yet because Simulation currently treats every facility as a normal
    transmission location.
    """
    people = papdata.get("people", {})
    places = papdata.get("places", {})
    place_ids, place_weights = _place_area_weights(places)

    capture_values = _capture_values(home_origin_capture)
    if capture_values:
        p_inside_fallback = float(median(capture_values))
    else:
        p_inside_fallback = 1.0 if place_ids else 0.0

    summary = {
        "version": "v1_area_weighted",
        "employment_probability_18_64": employment_probability(18),
        "employment_probability_65_plus": employment_probability(65),
        "area_weight_min_m2": MIN_WORKPLACE_AREA_M2,
        "area_weight_max_m2": MAX_WORKPLACE_AREA_M2,
        "home_origin_capture_cbg_count": len(capture_values),
        "p_inside_fallback": p_inside_fallback,
        "eligible_place_count": len(place_ids),
        "worker_count": 0,
        "in_zone_worker_count": 0,
        "out_of_zone_worker_count": 0,
        "non_worker_count": 0,
        "workplace_poi_count": 0,
    }
    workers_by_poi: Dict[str, int] = {}

    for person in people.values():
        age = person.get("age") if isinstance(person, dict) else None
        prob = employment_probability(age)
        if prob <= 0.0 or random.random() >= prob:
            if isinstance(person, dict):
                person["is_worker"] = False
                person["work_location_type"] = "none"
                person["work_poi"] = None
                person["work_p_inside"] = None
            summary["non_worker_count"] += 1
            continue

        summary["worker_count"] += 1
        home_cbg = person.get("home_cbg") if isinstance(person, dict) else None
        p_inside = _capture_for_cbg(home_origin_capture, home_cbg, p_inside_fallback)
        person["is_worker"] = True
        person["work_p_inside"] = p_inside

        if place_ids and random.random() < p_inside:
            work_poi = random.choices(place_ids, weights=place_weights, k=1)[0]
            person["work_location_type"] = "poi"
            person["work_poi"] = work_poi
            summary["in_zone_worker_count"] += 1
            workers_by_poi[work_poi] = workers_by_poi.get(work_poi, 0) + 1
        else:
            person["work_location_type"] = "out_of_zone"
            person["work_poi"] = None
            summary["out_of_zone_worker_count"] += 1

    summary["workplace_poi_count"] = len(workers_by_poi)
    if workers_by_poi:
        counts = sorted(workers_by_poi.values())
        summary["workers_per_workplace_poi"] = {
            "min": counts[0],
            "median": counts[len(counts) // 2],
            "max": counts[-1],
        }
    else:
        summary["workers_per_workplace_poi"] = {
            "min": 0,
            "median": 0,
            "max": 0,
        }

    papdata["worker_assignment"] = summary
    return summary


def _parse_cbg_counts(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        parsed = value
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    else:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}

    counts: Dict[str, float] = {}
    for raw_cbg, raw_count in parsed.items():
        try:
            cbg = str(raw_cbg).strip().zfill(12)
            count = float(raw_count)
        except (TypeError, ValueError):
            continue
        if cbg and count > 0:
            counts[cbg] = counts.get(cbg, 0.0) + count
    return counts


def _normalize_cbg_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(12)
    )


def _accumulate_home_origin_capture(
    df: pd.DataFrame,
    cluster_cbgs: set[str],
    source_cbgs: Optional[set[str]],
    inside_by_cbg: Dict[str, float],
    total_by_cbg: Dict[str, float],
) -> None:
    if df.empty:
        return
    df = df.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]
    if "poi_cbg" not in df.columns or "visitor_home_cbgs" not in df.columns:
        return

    df["poi_cbg"] = _normalize_cbg_series(df["poi_cbg"])
    for row in df[["poi_cbg", "visitor_home_cbgs"]].itertuples(index=False):
        destination_inside = row.poi_cbg in cluster_cbgs
        for home_cbg, count in _parse_cbg_counts(row.visitor_home_cbgs).items():
            if source_cbgs is not None and home_cbg not in source_cbgs:
                continue
            total_by_cbg[home_cbg] = total_by_cbg.get(home_cbg, 0.0) + count
            if destination_inside:
                inside_by_cbg[home_cbg] = inside_by_cbg.get(home_cbg, 0.0) + count


def compute_home_origin_capture(
    patterns_df: pd.DataFrame,
    cluster_cbgs: Iterable[str],
    source_cbgs: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute p_inside by home CBG from SafeGraph visitor_home_cbgs rows."""
    cluster_set = {str(cbg).strip().zfill(12) for cbg in cluster_cbgs}
    source_set = (
        {str(cbg).strip().zfill(12) for cbg in source_cbgs}
        if source_cbgs is not None
        else None
    )
    inside_by_cbg: Dict[str, float] = {}
    total_by_cbg: Dict[str, float] = {}
    _accumulate_home_origin_capture(
        patterns_df,
        cluster_set,
        source_set,
        inside_by_cbg,
        total_by_cbg,
    )
    return {
        cbg: (inside_by_cbg.get(cbg, 0.0) / total)
        for cbg, total in total_by_cbg.items()
        if total > 0
    }


def _read_pattern_flow_columns(path: str) -> pd.DataFrame:
    lower_cols = ["poi_cbg", "visitor_home_cbgs"]
    upper_cols = ["POI_CBG", "VISITOR_HOME_CBGS"]
    if str(path).endswith(".parquet"):
        try:
            df = pd.read_parquet(path, columns=lower_cols)
        except Exception:
            df = pd.read_parquet(path, columns=upper_cols)
        df.columns = [str(column).strip().lower() for column in df.columns]
        return df

    return pd.read_csv(
        path,
        usecols=lambda column: str(column).strip().lower() in set(lower_cols),
    )


def load_home_origin_capture(
    file_paths: Sequence[str],
    cluster_cbgs: Iterable[str],
    source_cbgs: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Load narrow SafeGraph columns and compute p_inside by home CBG."""
    cluster_set = {str(cbg).strip().zfill(12) for cbg in cluster_cbgs}
    source_set = (
        {str(cbg).strip().zfill(12) for cbg in source_cbgs}
        if source_cbgs is not None
        else None
    )
    inside_by_cbg: Dict[str, float] = {}
    total_by_cbg: Dict[str, float] = {}
    for path in file_paths:
        df = _read_pattern_flow_columns(str(path))
        _accumulate_home_origin_capture(
            df,
            cluster_set,
            source_set,
            inside_by_cbg,
            total_by_cbg,
        )
    return {
        cbg: (inside_by_cbg.get(cbg, 0.0) / total)
        for cbg, total in total_by_cbg.items()
        if total > 0
    }
