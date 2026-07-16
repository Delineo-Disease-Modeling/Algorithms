"""Robust dwell models for generic POI visits.

SafeGraph's dwell buckets mix customers, employees, residents, and host-facility
traffic.  Generic movement therefore uses an empirical-Bayes model: a POI's
bucket distribution is shrunk toward comparable NAICS peers according to its
number of unique visitors, while strongly corroborated outliers use the peer
distribution completely.  Scheduled workers and students are handled outside
this module; the open-ended bucket is represented by a decreasing category tail
rather than a blanket maximum for otherwise trustworthy POIs.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DWELL_BUCKET_LABELS: Tuple[str, ...] = (
    "<5", "5-20", "21-60", "61-240", ">240",
)
DWELL_PRIOR_VISITORS = 20.0
DWELL_MIN_CATEGORY_PEERS = 30

_OUTLIER_QUANTILE = 0.99
_LEGACY_BUCKET_MAP = {
    "<5": 0,
    "5-10": 1,
    "11-20": 1,
    "5-20": 1,
    "21-60": 2,
    "61-120": 3,
    "121-240": 3,
    "61-240": 3,
    ">240": 4,
}

_OBSERVATION_COLUMNS = (
    "placekey", "region", "naics_code", "median_dwell",
    "bucketed_dwell_times", "raw_visitor_counts", "raw_visit_counts",
    "popularity_by_hour", "popularity_by_day", "distance_from_home",
    "polygon_class", "parent_placekey", "enclosed", "closed_on",
    "date_range_start",
)


@dataclass(frozen=True)
class DwellModel:
    """Resolved dwell distribution and audit metadata for one POI."""

    probabilities: np.ndarray
    tail_geometric_p: float
    category_level: Optional[str]
    peer_count: int
    poi_weight: float
    hard_fallback: bool
    fallback_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class DwellReference:
    """Compact state/category profiles built before zone rows are filtered."""

    profiles: Dict[Tuple[str, str, str], "_CategoryProfile"]


@dataclass(frozen=True)
class _EmpiricalReference:
    lower_bound: float
    upper_bound: float

    def is_outlier(self, value: Optional[float]) -> bool:
        if value is None or not math.isfinite(value):
            return False
        return (
            value < self.lower_bound - 1e-12
            or value > self.upper_bound + 1e-12
        )

    def is_upper_outlier(self, value: Optional[float]) -> bool:
        return bool(
            value is not None
            and math.isfinite(value)
            and value > self.upper_bound + 1e-12
        )


@dataclass(frozen=True)
class _Observation:
    placekey: str
    region: str
    naics: str
    median_minutes: Optional[float]
    probabilities: Optional[np.ndarray]
    unique_visitors: Optional[float]
    repeat_log: Optional[float]
    night_log_ratio: Optional[float]
    weekend_logit: Optional[float]
    distance_from_home: Optional[float]
    shared_child: bool
    stale_closed: bool


@dataclass
class _CategoryProfile:
    level: str
    prior_probabilities: np.ndarray
    signal_probabilities: np.ndarray
    trusted_count: int
    typical_median_minutes: float
    median_reference: _EmpiricalReference
    long_reference: _EmpiricalReference
    repeat_reference: Optional[_EmpiricalReference]
    night_reference: Optional[_EmpiricalReference]
    weekend_reference: Optional[_EmpiricalReference]
    js_threshold: float

    def signals(
        self,
        observation: _Observation,
    ) -> Tuple[bool, bool, bool, bool, bool, bool]:
        probabilities = observation.probabilities
        if probabilities is None:
            return False, False, False, False, False, False
        median_log = (_safe_log1p(observation.median_minutes)
                      if observation.median_minutes is not None else None)
        long_logit = _logit(float(probabilities[4]))
        median_outlier = self.median_reference.is_outlier(median_log)
        long_outlier = self.long_reference.is_outlier(long_logit)
        shape_outlier = (
            _jensen_shannon_distance(probabilities, self.signal_probabilities)
            > self.js_threshold + 1e-12
        )
        repeat_outlier = bool(
            self.repeat_reference is not None
            and self.repeat_reference.is_upper_outlier(observation.repeat_log)
        )
        night_outlier = bool(
            self.night_reference is not None
            and self.night_reference.is_upper_outlier(observation.night_log_ratio)
        )
        weekend_outlier = bool(
            self.weekend_reference is not None
            and self.weekend_reference.is_upper_outlier(observation.weekend_logit)
        )
        return (
            median_outlier, long_outlier, shape_outlier, repeat_outlier,
            night_outlier, weekend_outlier,
        )


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_nonnegative_float(value) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _decode_mapping(value) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    if _is_missing(value):
        return None
    decoded = value
    for _ in range(2):
        if isinstance(decoded, dict):
            return decoded
        if not isinstance(decoded, str):
            return None
        try:
            decoded = json.loads(decoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            try:
                decoded = ast.literal_eval(decoded)
            except (SyntaxError, ValueError, TypeError):
                return None
    return decoded if isinstance(decoded, dict) else None


def _decode_sequence(value) -> Optional[list]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if _is_missing(value):
        return None
    decoded = value
    for _ in range(2):
        if isinstance(decoded, (list, tuple, np.ndarray)):
            return list(decoded)
        if not isinstance(decoded, str):
            return None
        try:
            decoded = json.loads(decoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            try:
                decoded = ast.literal_eval(decoded)
            except (SyntaxError, ValueError, TypeError):
                return None
    return list(decoded) if isinstance(decoded, (list, tuple, np.ndarray)) else None


def _night_log_ratio(value) -> Optional[float]:
    hours = _decode_sequence(value)
    if hours is None or len(hours) != 24:
        return None
    numeric = [_as_nonnegative_float(item) for item in hours]
    if any(item is None for item in numeric):
        return None
    night_mean = float(np.mean(numeric[0:6]))
    business_mean = float(np.mean(numeric[9:17]))
    if night_mean <= 0 and business_mean <= 0:
        return None
    return math.log((night_mean + 1.0) / (business_mean + 1.0))


def _weekend_logit(value) -> Optional[float]:
    days = _decode_mapping(value)
    if not days:
        return None
    numeric = {
        str(key).strip().title(): _as_nonnegative_float(count)
        for key, count in days.items()
    }
    if any(count is None for count in numeric.values()):
        return None
    total = float(sum(numeric.values()))
    if total <= 0:
        return None
    weekend = float(numeric.get("Saturday", 0.0) + numeric.get("Sunday", 0.0))
    return _logit(weekend / total)


def _truthy(value) -> bool:
    return bool(
        not _is_missing(value)
        and str(value).strip().lower() in {"1", "true", "yes", "y"}
    )


def _iso_date(value) -> Optional[str]:
    if _is_missing(value):
        return None
    text = str(value).strip()[:10]
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text
    return None


def parse_dwell_bucket_counts(value) -> Optional[np.ndarray]:
    """Return modern five-bin counts from modern or legacy SafeGraph JSON.

    Production parquet values may be dictionaries, JSON strings, or
    double-encoded JSON strings.  Older exports split two of the modern bins;
    those counts are combined here rather than discarded.
    """
    mapping = _decode_mapping(value)
    if not mapping:
        return None
    counts = np.zeros(len(DWELL_BUCKET_LABELS), dtype=float)
    recognized = False
    for raw_key, raw_value in mapping.items():
        index = _LEGACY_BUCKET_MAP.get(str(raw_key).strip())
        if index is None:
            continue
        count = _as_nonnegative_float(raw_value)
        if count is None:
            return None
        counts[index] += count
        recognized = True
    total = float(counts.sum())
    if not recognized or total <= 0:
        return None
    return counts


def _normalize_counts(counts: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if counts is None:
        return None
    total = float(np.sum(counts))
    if not math.isfinite(total) or total <= 0:
        return None
    return np.asarray(counts, dtype=float) / total


def _clean_naics(value) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return "".join(ch for ch in text if ch.isdigit())


def _safe_log1p(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value) or value < 0:
        return None
    return math.log1p(value)


def _logit(probability: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(p / (1.0 - p))


def _empirical_reference(values: Iterable[Optional[float]]) -> _EmpiricalReference:
    arr = np.asarray([
        value for value in values
        if value is not None and math.isfinite(value)
    ], dtype=float)
    if arr.size == 0:
        return _EmpiricalReference(0.0, 0.0)
    lower_bound = float(np.quantile(arr, 1.0 - _OUTLIER_QUANTILE))
    upper_bound = float(np.quantile(arr, _OUTLIER_QUANTILE))
    return _EmpiricalReference(lower_bound, upper_bound)


def _robust_probability_center(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    center = np.median(matrix, axis=0)
    total = float(center.sum())
    if total <= 0:
        center = np.mean(matrix, axis=0)
        total = float(center.sum())
    if total <= 0:
        return np.full(len(DWELL_BUCKET_LABELS), 1.0 / len(DWELL_BUCKET_LABELS))
    return center / total


def _jensen_shannon_distance(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    midpoint = 0.5 * (p + q)

    def _kl(a, b) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return math.sqrt(max(0.0, 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)))


def _jensen_shannon_distances(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(matrix, dtype=float)
    reference = np.broadcast_to(np.asarray(center, dtype=float), probabilities.shape)
    midpoint = 0.5 * (probabilities + reference)

    left_terms = np.zeros_like(probabilities)
    left_mask = probabilities > 0
    left_terms[left_mask] = probabilities[left_mask] * np.log(
        probabilities[left_mask] / midpoint[left_mask]
    )
    right_terms = np.zeros_like(reference)
    right_mask = reference > 0
    right_terms[right_mask] = reference[right_mask] * np.log(
        reference[right_mask] / midpoint[right_mask]
    )
    divergence = 0.5 * (
        np.sum(left_terms, axis=1) + np.sum(right_terms, axis=1)
    )
    return np.sqrt(np.maximum(0.0, divergence))


def _is_statistical_outlier(
    median_outlier: bool,
    long_outlier: bool,
    shape_outlier: bool,
) -> bool:
    return sum((median_outlier, long_outlier, shape_outlier)) >= 2


def _observation_from_values(values) -> Optional[_Observation]:
    (
        placekey_value, region_value, naics_value, median_value, bucket_value,
        visitor_value, visit_value, popularity_hour_value, popularity_day_value,
        distance_value, polygon_value, parent_value, enclosed_value,
        closed_value, date_range_start_value,
    ) = values
    if _is_missing(placekey_value):
        return None
    placekey = str(placekey_value).strip()
    counts = parse_dwell_bucket_counts(bucket_value)
    probabilities = _normalize_counts(counts)
    median_minutes = _as_nonnegative_float(median_value)
    unique_visitors = _as_nonnegative_float(visitor_value)
    raw_visits = _as_nonnegative_float(visit_value)
    repeat_log = None
    if raw_visits is not None and unique_visitors is not None and unique_visitors > 0:
        repeat_log = math.log1p(raw_visits / unique_visitors)
    polygon_class = str(polygon_value).strip().upper()
    parent_present = not _is_missing(parent_value)
    closed_on = _iso_date(closed_value)
    date_range_start = _iso_date(date_range_start_value)
    return _Observation(
        placekey=placekey,
        region=("" if _is_missing(region_value)
                else str(region_value).strip().upper()),
        naics=_clean_naics(naics_value),
        median_minutes=median_minutes,
        probabilities=probabilities,
        unique_visitors=unique_visitors,
        repeat_log=repeat_log,
        night_log_ratio=_night_log_ratio(popularity_hour_value),
        weekend_logit=_weekend_logit(popularity_day_value),
        distance_from_home=_as_nonnegative_float(distance_value),
        shared_child=(
            parent_present
            and (polygon_class == "SHARED_POLYGON" or _truthy(enclosed_value))
        ),
        stale_closed=bool(
            closed_on is not None
            and date_range_start is not None
            and closed_on < date_range_start
        ),
    )


def _profile_for(
    level: str,
    observations: List[_Observation],
) -> Optional[_CategoryProfile]:
    usable = [
        observation for observation in observations
        if observation.probabilities is not None and observation.median_minutes is not None
    ]
    if not usable:
        return None

    probability_matrix = np.asarray([
        observation.probabilities for observation in usable
    ], dtype=float)
    preliminary_prior = _robust_probability_center(probability_matrix)
    median_reference = _empirical_reference(
        _safe_log1p(observation.median_minutes) for observation in usable
    )
    long_reference = _empirical_reference(
        _logit(float(observation.probabilities[4])) for observation in usable
    )
    repeat_values = [
        observation.repeat_log for observation in usable
        if observation.repeat_log is not None
    ]
    repeat_reference = (
        _empirical_reference(repeat_values) if repeat_values else None
    )
    night_values = [
        observation.night_log_ratio for observation in usable
        if observation.night_log_ratio is not None
    ]
    night_reference = (
        _empirical_reference(night_values) if night_values else None
    )
    weekend_values = [
        observation.weekend_logit for observation in usable
        if observation.weekend_logit is not None
    ]
    weekend_reference = (
        _empirical_reference(weekend_values) if weekend_values else None
    )
    js_distances = _jensen_shannon_distances(probability_matrix, preliminary_prior)
    js_threshold = float(np.quantile(js_distances, _OUTLIER_QUANTILE))

    trusted: List[_Observation] = []
    for observation, js_distance in zip(usable, js_distances):
        median_outlier = median_reference.is_outlier(
            _safe_log1p(observation.median_minutes)
        )
        long_outlier = long_reference.is_outlier(
            _logit(float(observation.probabilities[4]))
        )
        shape_outlier = js_distance > js_threshold + 1e-12
        repeat_outlier = bool(
            repeat_reference is not None
            and repeat_reference.is_upper_outlier(observation.repeat_log)
        )
        night_outlier = bool(
            night_reference is not None
            and night_reference.is_upper_outlier(observation.night_log_ratio)
        )
        weekend_outlier = bool(
            weekend_reference is not None
            and weekend_reference.is_upper_outlier(observation.weekend_logit)
        )
        statistical_outlier = _is_statistical_outlier(
            median_outlier, long_outlier, shape_outlier
        )
        home_like = (
            observation.distance_from_home is not None
            and observation.distance_from_home <= 100.0
            and repeat_outlier
        )
        temporal_mismatch = night_outlier and weekend_outlier
        if (
            not statistical_outlier
            and not observation.shared_child
            and not home_like
            and not temporal_mismatch
            and not observation.stale_closed
        ):
            trusted.append(observation)

    if trusted:
        prior = _robust_probability_center([
            observation.probabilities for observation in trusted
        ])
        typical_median = float(np.median([
            observation.median_minutes for observation in trusted
        ]))
    else:
        # Keep a profile object for diagnostics, but trusted_count=0 prevents
        # selection and makes the caller back off to a broader category.
        prior = preliminary_prior
        typical_median = float(np.median([
            observation.median_minutes for observation in usable
        ]))
    return _CategoryProfile(
        level=level,
        prior_probabilities=prior,
        signal_probabilities=preliminary_prior,
        trusted_count=len(trusted),
        typical_median_minutes=typical_median,
        median_reference=median_reference,
        long_reference=long_reference,
        repeat_reference=repeat_reference,
        night_reference=night_reference,
        weekend_reference=weekend_reference,
        js_threshold=js_threshold,
    )


def _category_keys(observation: _Observation) -> List[Tuple[str, str, str]]:
    keys: List[Tuple[str, str, str]] = []
    region = observation.region
    if len(observation.naics) >= 6:
        keys.append((region, "naics6", observation.naics[:6]))
    if len(observation.naics) >= 4:
        keys.append((region, "naics4", observation.naics[:4]))
    if len(observation.naics) >= 2:
        keys.append((region, "naics2", observation.naics[:2]))
    keys.append((region, "global", "*"))
    return keys


def _tail_geometric_probability(typical_median_minutes: float) -> float:
    """Calibrate a decreasing >240-minute tail from the trusted category median."""
    target_hour = int(math.ceil(max(0.0, typical_median_minutes) / 60.0))
    target_hour = max(5, target_hour)
    increments_to_median = max(1, target_hour - 4)
    probability = 1.0 - (0.5 ** (1.0 / increments_to_median))
    return min(1.0, max(0.01, probability))


def _observations_from_df(df: pd.DataFrame) -> List[_Observation]:
    if df is None or df.empty or "placekey" not in df.columns:
        return []
    observations: List[_Observation] = []
    seen = set()
    projected = df.reindex(columns=_OBSERVATION_COLUMNS)
    for values in projected.itertuples(index=False, name=None):
        observation = _observation_from_values(values)
        if observation is None or observation.placekey in seen:
            continue
        observations.append(observation)
        seen.add(observation.placekey)
    return observations


def _observation_is_trusted(
    observation: _Observation,
    profile: _CategoryProfile,
) -> bool:
    if observation.probabilities is None or observation.median_minutes is None:
        return False
    (
        median_outlier, long_outlier, shape_outlier, repeat_outlier,
        night_outlier, weekend_outlier,
    ) = profile.signals(observation)
    statistical_outlier = _is_statistical_outlier(
        median_outlier, long_outlier, shape_outlier
    )
    home_like = (
        observation.distance_from_home is not None
        and observation.distance_from_home <= 100.0
        and repeat_outlier
    )
    temporal_mismatch = night_outlier and weekend_outlier
    return (
        not statistical_outlier
        and not observation.shared_child
        and not home_like
        and not temporal_mismatch
        and not observation.stale_closed
    )


def build_dwell_reference(df: pd.DataFrame) -> DwellReference:
    """Reduce a full monthly/state frame to compact hierarchical profiles."""
    observations = _observations_from_df(df)
    groups: Dict[Tuple[str, str, str], List[_Observation]] = {}
    for observation in observations:
        if observation.probabilities is None or observation.median_minutes is None:
            continue
        for key in _category_keys(observation):
            groups.setdefault(key, []).append(observation)
    profiles = {
        key: profile
        for key, group in groups.items()
        if (profile := _profile_for(key[1], group)) is not None
    }
    return DwellReference(profiles=profiles)


def merge_dwell_references(references: Iterable[DwellReference]) -> DwellReference:
    """Merge independently built state profiles.

    Region is part of every key, so normal multi-state loads are disjoint. If
    malformed inputs omit region in more than one file, retain the profile with
    more trusted POIs rather than silently combining incompatible quantiles.
    """
    profiles: Dict[Tuple[str, str, str], _CategoryProfile] = {}
    for reference in references:
        for key, profile in reference.profiles.items():
            current = profiles.get(key)
            if current is None or profile.trusted_count > current.trusted_count:
                profiles[key] = profile
    return DwellReference(profiles=profiles)


def build_dwell_models(
    df: pd.DataFrame,
    needed_placekeys: Optional[Iterable[str]] = None,
    reference: Optional[DwellReference] = None,
) -> Dict[str, DwellModel]:
    """Build category-shrunk dwell models for requested POIs.

    Peer selection requires 30 trusted *other* POIs and backs off from exact
    six-digit NAICS to four-digit, sector, then the loaded global population.
    A hard statistical fallback requires at least two category-tail signals
    among median dwell, >240-minute share, and full bucket-distribution shape.
    """
    observations = _observations_from_df(df)
    if not observations:
        return {}
    profiles = (reference or build_dwell_reference(df)).profiles

    needed = (
        {str(placekey) for placekey in needed_placekeys}
        if needed_placekeys is not None else None
    )
    models: Dict[str, DwellModel] = {}
    for observation in observations:
        if needed is not None and observation.placekey not in needed:
            continue

        selected: Optional[_CategoryProfile] = None
        peer_count = 0
        for key in _category_keys(observation):
            profile = profiles.get(key)
            if profile is None:
                continue
            trusted_count = profile.trusted_count
            if _observation_is_trusted(observation, profile):
                trusted_count -= 1
            if trusted_count >= DWELL_MIN_CATEGORY_PEERS:
                selected = profile
                peer_count = trusted_count
                break

        if selected is None:
            if observation.probabilities is None:
                continue
            median_for_tail = observation.median_minutes or 300.0
            models[observation.placekey] = DwellModel(
                probabilities=observation.probabilities.copy(),
                tail_geometric_p=_tail_geometric_probability(median_for_tail),
                category_level=None,
                peer_count=0,
                poi_weight=1.0,
                hard_fallback=False,
                fallback_reasons=(),
            )
            continue

        (
            median_outlier, long_outlier, shape_outlier, repeat_outlier,
            night_outlier, weekend_outlier,
        ) = selected.signals(observation)
        reasons: List[str] = []
        if _is_statistical_outlier(median_outlier, long_outlier, shape_outlier):
            reasons.append("statistical_outlier")
        if observation.shared_child:
            reasons.append("shared_child")
        if (
            observation.distance_from_home is not None
            and observation.distance_from_home <= 100.0
            and repeat_outlier
        ):
            reasons.append("home_like_repeat_visitors")
        if night_outlier and weekend_outlier:
            reasons.append("temporal_mismatch")
        if observation.stale_closed:
            reasons.append("stale_closed")
        if observation.probabilities is None:
            reasons.append("missing_bucket_distribution")

        hard_fallback = bool(reasons)
        if hard_fallback:
            poi_weight = 0.0
            resolved = selected.prior_probabilities.copy()
        else:
            unique_visitors = observation.unique_visitors
            poi_weight = (
                unique_visitors / (unique_visitors + DWELL_PRIOR_VISITORS)
                if unique_visitors is not None and unique_visitors > 0 else 0.0
            )
            resolved = (
                poi_weight * observation.probabilities
                + (1.0 - poi_weight) * selected.prior_probabilities
            )

        models[observation.placekey] = DwellModel(
            probabilities=resolved,
            tail_geometric_p=_tail_geometric_probability(
                selected.typical_median_minutes
            ),
            category_level=selected.level,
            peer_count=peer_count,
            poi_weight=float(poi_weight),
            hard_fallback=hard_fallback,
            fallback_reasons=tuple(reasons),
        )
    return models


def sample_dwell_hours(
    rng: np.random.Generator,
    destination_indices: np.ndarray,
    probability_matrix: np.ndarray,
    modeled_places: np.ndarray,
    tail_geometric_p: np.ndarray,
    legacy_median_hours: np.ndarray,
) -> np.ndarray:
    """Vectorized sampling for a batch of generic POI arrivals."""
    destinations = np.asarray(destination_indices, dtype=np.int64)
    result = np.ones(destinations.size, dtype=np.int64)
    if destinations.size == 0:
        return result

    has_model = modeled_places[destinations]
    modeled_positions = np.flatnonzero(has_model)
    if modeled_positions.size:
        modeled_destinations = destinations[modeled_positions]
        probabilities = probability_matrix[modeled_destinations]
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0
        bucket_draw = rng.random(modeled_positions.size)
        buckets = np.sum(bucket_draw[:, None] > cumulative, axis=1)
        finite_mask = buckets < 4
        if finite_mask.any():
            finite_buckets = buckets[finite_mask]
            lows = np.choose(finite_buckets, [1, 5, 21, 61]).astype(np.int64)
            highs = np.choose(finite_buckets, [4, 20, 60, 240]).astype(np.int64)
            spans = np.maximum(1, highs - lows + 1)
            minute_draws = lows + np.floor(
                rng.random(finite_buckets.size) * spans
            ).astype(np.int64)
            sampled_hours = (minute_draws + 59) // 60
            result[modeled_positions[finite_mask]] = sampled_hours

        tail_mask = buckets == 4
        if tail_mask.any():
            tail_destinations = modeled_destinations[tail_mask]
            probabilities_p = tail_geometric_p[tail_destinations]
            uniforms = rng.random(tail_destinations.size)
            increments = np.ones(tail_destinations.size, dtype=np.int64)
            non_unit = probabilities_p < 1.0
            if non_unit.any():
                increments[non_unit] = (
                    np.floor(
                        np.log1p(-uniforms[non_unit])
                        / np.log1p(-probabilities_p[non_unit])
                    ).astype(np.int64)
                    + 1
                )
            sampled_hours = 4 + increments
            result[modeled_positions[tail_mask]] = sampled_hours

    legacy_positions = np.flatnonzero(~has_model)
    if legacy_positions.size:
        legacy_destinations = destinations[legacy_positions]
        medians = legacy_median_hours[legacy_destinations]
        lows = np.maximum(1, medians - 1)
        highs = medians + 1
        spans = np.maximum(1, highs - lows + 1)
        result[legacy_positions] = lows + np.floor(
            rng.random(legacy_positions.size) * spans
        ).astype(np.int64)

    return np.maximum(1, result)
