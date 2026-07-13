"""Persistent v1 activity anchors for synthetic residents.

This deliberately does not generate a complete activity schedule. It annotates
papdata people with stable work/school metadata so movement generation can use
persistent anchors without re-solving those assignments each run.
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
EXTERNAL_WORKPLACE_LABEL = "Out of Zone Work"
EXTERNAL_WORKPLACE_TYPE = "out_of_zone_work"
EXTERNAL_SCHOOL_LABEL = "Out of Zone School"
EXTERNAL_SCHOOL_TYPE = "out_of_zone_school"
STUDENT_MIN_AGE = 5
STUDENT_MAX_AGE = 17
SCHOOL_NAICS_CODES = {"611110"}
SCHOOL_TOP_CATEGORIES = {"elementary and secondary schools"}


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
        if place.get("external_location_type"):
            continue
        weight = effective_workplace_area_weight(place.get("area"))
        if weight <= 0:
            continue
        place_ids.append(str(place_id))
        weights.append(float(weight))
    return place_ids, weights


def _next_place_id(places: Dict[str, Dict[str, Any]]) -> str:
    numeric_ids = []
    for place_id in places:
        try:
            numeric_ids.append(int(place_id))
        except (TypeError, ValueError):
            continue
    external_id = str((max(numeric_ids) + 1) if numeric_ids else len(places))
    while external_id in places:
        external_id = str(int(external_id) + 1)
    return external_id


def _ensure_external_location(
    papdata: Dict[str, Any],
    external_location_type: str,
    label: str,
    id_key: str,
) -> str:
    places = papdata.setdefault("places", {})
    for place_id, place in places.items():
        if isinstance(place, dict) and place.get("external_location_type") == external_location_type:
            papdata[id_key] = str(place_id)
            return str(place_id)

    external_id = _next_place_id(places)
    places[external_id] = {
        "placekey": None,
        "label": label,
        "cbg": None,
        "latitude": None,
        "longitude": None,
        "top_category": "External",
        "street_address": None,
        "postal_code": None,
        "naics_code": None,
        "footprint": None,
        "area": None,
        "catchment_fj": None,
        "external_location_type": external_location_type,
    }
    papdata[id_key] = external_id
    return external_id


def ensure_external_workplace(papdata: Dict[str, Any]) -> Optional[str]:
    """Ensure the synthetic off-map workplace exists when needed."""
    people = papdata.get("people", {})
    has_out_of_zone_worker = any(
        isinstance(person, dict) and person.get("work_location_type") == "out_of_zone"
        for person in people.values()
    )
    if not has_out_of_zone_worker:
        return None
    return _ensure_external_location(
        papdata,
        EXTERNAL_WORKPLACE_TYPE,
        EXTERNAL_WORKPLACE_LABEL,
        "external_workplace_id",
    )


def ensure_external_school(papdata: Dict[str, Any]) -> Optional[str]:
    """Ensure the synthetic off-map school exists when needed."""
    people = papdata.get("people", {})
    has_out_of_zone_student = any(
        isinstance(person, dict) and person.get("school_location_type") == "out_of_zone"
        for person in people.values()
    )
    if not has_out_of_zone_student:
        return None
    return _ensure_external_location(
        papdata,
        EXTERNAL_SCHOOL_TYPE,
        EXTERNAL_SCHOOL_LABEL,
        "external_school_id",
    )


def ensure_external_locations(papdata: Dict[str, Any]) -> Dict[str, Optional[str]]:
    return {
        "workplace": ensure_external_workplace(papdata),
        "school": ensure_external_school(papdata),
    }


def _normalize_naics(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return "".join(ch for ch in text if ch.isdigit())


def _is_school_place(place: Dict[str, Any]) -> bool:
    if place.get("external_location_type"):
        return False
    naics = _normalize_naics(place.get("naics_code"))
    if naics in SCHOOL_NAICS_CODES:
        return True
    category = str(place.get("top_category") or "").strip().lower()
    return category in SCHOOL_TOP_CATEGORIES


def _school_area_weights(places: Dict[str, Dict[str, Any]]) -> tuple[list[str], list[float]]:
    school_ids: list[str] = []
    weights: list[float] = []
    for place_id, place in places.items():
        if not isinstance(place, dict) or not _is_school_place(place):
            continue
        weight = effective_workplace_area_weight(place.get("area"))
        if weight <= 0:
            continue
        school_ids.append(str(place_id))
        weights.append(float(weight))
    return school_ids, weights


def assign_students(papdata: Dict[str, Any]) -> Dict[str, Any]:
    """Annotate school-age residents with persistent v1 school assignments."""
    people = papdata.get("people", {})
    places = papdata.get("places", {})
    school_ids, school_weights = _school_area_weights(places)
    summary = {
        "version": "v1_school_area_weighted",
        "student_min_age": STUDENT_MIN_AGE,
        "student_max_age": STUDENT_MAX_AGE,
        "eligible_school_poi_count": len(school_ids),
        "student_count": 0,
        "in_zone_student_count": 0,
        "out_of_zone_student_count": 0,
        "non_student_count": 0,
        "school_poi_count": 0,
    }
    students_by_school: Dict[str, int] = {}

    for person in people.values():
        age = person.get("age") if isinstance(person, dict) else None
        age_int = _coerce_age(age)
        if age_int is None or age_int < STUDENT_MIN_AGE or age_int > STUDENT_MAX_AGE:
            if isinstance(person, dict):
                person["is_student"] = False
                person["school_location_type"] = "none"
                person["school_poi"] = None
            summary["non_student_count"] += 1
            continue

        person["is_student"] = True
        summary["student_count"] += 1
        if school_ids:
            school_poi = random.choices(school_ids, weights=school_weights, k=1)[0]
            person["school_location_type"] = "poi"
            person["school_poi"] = school_poi
            summary["in_zone_student_count"] += 1
            students_by_school[school_poi] = students_by_school.get(school_poi, 0) + 1
        else:
            person["school_location_type"] = "out_of_zone"
            person["school_poi"] = None
            summary["out_of_zone_student_count"] += 1

    summary["school_poi_count"] = len(students_by_school)
    if students_by_school:
        counts = sorted(students_by_school.values())
        summary["students_per_school_poi"] = {
            "min": counts[0],
            "median": counts[len(counts) // 2],
            "max": counts[-1],
        }
    else:
        summary["students_per_school_poi"] = {
            "min": 0,
            "median": 0,
            "max": 0,
        }
    papdata["school_assignment"] = summary
    return summary


def assign_workers(
    papdata: Dict[str, Any],
    home_origin_capture: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Annotate papdata people with persistent v1 work assignments.

    Employment is age-probability based. Workers are assigned to exactly one
    in-zone persistent POI, sampled by capped POI area. ``work_p_inside`` is
    retained as diagnostic metadata from observed home-origin capture, but it
    no longer routes workers out of the zone.
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
        "version": "v1_area_weighted_in_zone_only",
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

        if not place_ids:
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

        work_poi = random.choices(place_ids, weights=place_weights, k=1)[0]
        person["work_location_type"] = "poi"
        person["work_poi"] = work_poi
        summary["in_zone_worker_count"] += 1
        workers_by_poi[work_poi] = workers_by_poi.get(work_poi, 0) + 1

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
