from dataclasses import dataclass
from typing import Optional

from common_geo import normalize_cbg

from .algorithm_params import (
    normalize_cluster_algorithm,
    parse_czi_balanced_params,
    parse_czi_optimal_params,
    parse_seed_guard_params,
    parse_ttwa_params,
)
from .constants import (
    DEFAULT_CONTAINMENT_THRESHOLD,
    DEFAULT_DISTANCE_PENALTY_WEIGHT,
    DEFAULT_DISTANCE_SCALE_KM,
    DEFAULT_OPTIMAL_CANDIDATE_LIMIT,
    DEFAULT_OPTIMAL_MAX_ITERS,
    DEFAULT_OPTIMAL_MIP_REL_GAP,
    DEFAULT_OPTIMAL_POP_FLOOR_RATIO,
    DEFAULT_OPTIMAL_TIME_LIMIT_SEC,
    DEFAULT_SEED_GUARD_DISTANCE_KM,
    TEST_PATTERNS_FILE,
)
from .errors import ApiError
from .pattern_resolution import (
    get_test_seed_cbg,
    resolve_patterns_file_for_request,
    validate_csv_columns,
)


@dataclass(frozen=True)
class PatternSelection:
    file_path: str
    source: str
    month: Optional[str]
    use_test_data: bool


def parse_json_payload(flask_request):
    try:
        payload = flask_request.get_json(force=True)
    except Exception:
        raise ApiError('Bad Request', status_code=400)

    if not payload:
        raise ApiError('Please supply adequate JSON data', status_code=400)
    return payload


def require_cbg(payload, field_name):
    value = payload.get(field_name)
    if value is None:
        raise ApiError(f"Missing required field: '{field_name}'", status_code=400)
    normalized = normalize_cbg(value)
    if not normalized:
        raise ApiError(f"Invalid '{field_name}': expected exactly 12 digits", status_code=400)
    return normalized


def optional_cbg(payload, field_name, required_message=None):
    normalized = normalize_cbg(payload.get(field_name))
    if not normalized:
        raise ApiError(required_message or f"Invalid or missing '{field_name}'", status_code=400)
    return normalized


def normalize_cbg_list(values, field_name):
    if not isinstance(values, list) or len(values) == 0:
        raise ApiError(f"Missing or invalid '{field_name}'", status_code=400)

    normalized = []
    seen = set()
    for cbg in values:
        cbg_norm = normalize_cbg(cbg)
        if not cbg_norm or cbg_norm in seen:
            continue
        seen.add(cbg_norm)
        normalized.append(cbg_norm)

    if not normalized:
        raise ApiError(f"No valid CBGs in '{field_name}'", status_code=400)

    return normalized


def resolve_pattern_selection(seed_cbg, payload):
    use_test_data = bool(payload.get('use_test_data'))
    try:
        file_path, source, month = resolve_patterns_file_for_request(
            seed_cbg,
            start_date_raw=payload.get('start_date'),
            use_test_data=use_test_data,
        )
    except ValueError as exc:
        raise ApiError(str(exc), status_code=400) from exc
    return PatternSelection(file_path=file_path, source=source, month=month, use_test_data=use_test_data)


def parse_cluster_algorithm_config(payload):
    algorithm = normalize_cluster_algorithm(payload.get('algorithm'))
    if not algorithm:
        raise ApiError(
            "Invalid 'algorithm'. Valid options: czi_balanced, czi_optimal_cap, greedy_fast, greedy_ratio, greedy_ttwa, greedy_weight, greedy_weight_seed_guard",
            status_code=400,
        )

    czi_params = {}
    optimal_params = {}
    seed_guard_params = {}
    ttwa_params = {}

    if algorithm == 'czi_balanced':
        czi_params, err = parse_czi_balanced_params(payload)
    elif algorithm == 'czi_optimal_cap':
        optimal_params, err = parse_czi_optimal_params(payload)
    elif algorithm == 'greedy_weight_seed_guard':
        seed_guard_params, err = parse_seed_guard_params(payload)
    elif algorithm == 'greedy_ttwa':
        ttwa_params, err = parse_ttwa_params(payload)
    else:
        err = None

    if err:
        raise ApiError(err, status_code=400)

    effective_czi_params = {}
    effective_optimal_params = {}
    effective_seed_guard_params = {}
    effective_ttwa_params = {}

    if algorithm == 'czi_balanced':
        effective_czi_params = {
            'distance_penalty_weight': (
                czi_params.get('distance_penalty_weight')
                if czi_params.get('distance_penalty_weight') is not None
                else DEFAULT_DISTANCE_PENALTY_WEIGHT
            ),
            'distance_scale_km': (
                czi_params.get('distance_scale_km')
                if czi_params.get('distance_scale_km') is not None
                else DEFAULT_DISTANCE_SCALE_KM
            ),
        }
    elif algorithm == 'czi_optimal_cap':
        effective_optimal_params = {
            'optimal_candidate_limit': (
                optimal_params.get('optimal_candidate_limit')
                if optimal_params.get('optimal_candidate_limit') is not None
                else DEFAULT_OPTIMAL_CANDIDATE_LIMIT
            ),
            'optimal_population_floor_ratio': (
                optimal_params.get('optimal_population_floor_ratio')
                if optimal_params.get('optimal_population_floor_ratio') is not None
                else DEFAULT_OPTIMAL_POP_FLOOR_RATIO
            ),
            'optimal_mip_rel_gap': (
                optimal_params.get('optimal_mip_rel_gap')
                if optimal_params.get('optimal_mip_rel_gap') is not None
                else DEFAULT_OPTIMAL_MIP_REL_GAP
            ),
            'optimal_time_limit_sec': (
                optimal_params.get('optimal_time_limit_sec')
                if optimal_params.get('optimal_time_limit_sec') is not None
                else DEFAULT_OPTIMAL_TIME_LIMIT_SEC
            ),
            'optimal_max_iters': (
                optimal_params.get('optimal_max_iters')
                if optimal_params.get('optimal_max_iters') is not None
                else DEFAULT_OPTIMAL_MAX_ITERS
            ),
        }
    elif algorithm == 'greedy_weight_seed_guard':
        effective_seed_guard_params = {
            'seed_guard_distance_km': (
                seed_guard_params.get('seed_guard_distance_km')
                if seed_guard_params.get('seed_guard_distance_km') is not None
                else DEFAULT_SEED_GUARD_DISTANCE_KM
            ),
        }
    elif algorithm == 'greedy_ttwa':
        effective_ttwa_params = {
            'containment_threshold': (
                ttwa_params.get('containment_threshold')
                if ttwa_params.get('containment_threshold') is not None
                else DEFAULT_CONTAINMENT_THRESHOLD
            ),
        }

    return {
        'algorithm': algorithm,
        'czi_params': czi_params,
        'optimal_params': optimal_params,
        'seed_guard_params': seed_guard_params,
        'ttwa_params': ttwa_params,
        'hierarchical_params': {},
        'effective_czi_params': effective_czi_params,
        'effective_optimal_params': effective_optimal_params,
        'effective_seed_guard_params': effective_seed_guard_params,
        'effective_ttwa_params': effective_ttwa_params,
        'effective_hierarchical_params': {},
    }


def parse_test_seed_if_needed(payload, cbg):
    if not bool(payload.get('use_test_data')) or cbg is not None:
        return cbg

    ok, missing, _headers = validate_csv_columns(TEST_PATTERNS_FILE, ['poi_cbg', 'visitor_daytime_cbgs'])
    if not ok:
        raise ApiError(
            f"TEST data is missing required columns for clustering: {', '.join(missing)}",
            status_code=400,
            extra={'required_columns': ['poi_cbg', 'visitor_daytime_cbgs'], 'test_file': TEST_PATTERNS_FILE},
        )
    inferred = get_test_seed_cbg(TEST_PATTERNS_FILE)
    if inferred is None:
        raise ApiError(
            "TEST data must include at least one valid 12-digit 'poi_cbg' value to infer the seed CBG",
            status_code=400,
            extra={'test_file': TEST_PATTERNS_FILE},
        )
    return inferred
