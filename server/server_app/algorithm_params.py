from .constants import VALID_CLUSTER_ALGORITHMS


def normalize_cluster_algorithm(algorithm):
    if algorithm is None:
        return 'czi_balanced'

    alg = str(algorithm).strip().lower()
    aliases = {
        'balanced': 'czi_balanced',
        'czi': 'czi_balanced',
        'optimal': 'czi_optimal_cap',
        'rigorous': 'czi_optimal_cap',
        'milp': 'czi_optimal_cap',
        'fast': 'greedy_fast',
        'weight': 'greedy_weight',
        'seed_guard': 'greedy_weight_seed_guard',
        'weight_guard': 'greedy_weight_seed_guard',
        'ratio': 'greedy_ratio',
        'ttwa': 'greedy_ttwa',
        'hierarchical': 'hierarchical_core_satellites',
        'hierarchy': 'hierarchical_core_satellites',
        'core_satellites': 'hierarchical_core_satellites',
    }
    alg = aliases.get(alg, alg)
    return alg if alg in VALID_CLUSTER_ALGORITHMS else None


def _parse_float(payload, name, min_value=None, max_value=None, strictly_positive=False):
    raw = payload.get(name)
    if raw is None or raw == '':
        return None, None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"Invalid '{name}': expected a number"
    if strictly_positive and value <= 0:
        return None, f"Invalid '{name}': must be > 0"
    if min_value is not None and value < min_value:
        return None, f"Invalid '{name}': must be >= {min_value}"
    if max_value is not None and value > max_value:
        return None, f"Invalid '{name}': must be <= {max_value}"
    return value, None


def _parse_int(payload, name, min_value=None, max_value=None, strictly_positive=False):
    raw = payload.get(name)
    if raw is None or raw == '':
        return None, None
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return None, f"Invalid '{name}': expected an integer"
    if strictly_positive and value <= 0:
        return None, f"Invalid '{name}': must be > 0"
    if min_value is not None and value < min_value:
        return None, f"Invalid '{name}': must be >= {min_value}"
    if max_value is not None and value > max_value:
        return None, f"Invalid '{name}': must be <= {max_value}"
    return value, None


def parse_czi_balanced_params(payload):
    payload = payload or {}

    distance_penalty_weight, err = _parse_float(
        payload, 'distance_penalty_weight', min_value=0.0, max_value=1.0
    )
    if err:
        return None, err

    distance_scale_km, err = _parse_float(
        payload, 'distance_scale_km', min_value=0.1, max_value=500.0, strictly_positive=True
    )
    if err:
        return None, err

    return {
        'distance_penalty_weight': distance_penalty_weight,
        'distance_scale_km': distance_scale_km,
    }, None


def parse_seed_guard_params(payload):
    payload = payload or {}
    raw = payload.get('seed_guard_distance_km')
    if raw is None or raw == '':
        return {'seed_guard_distance_km': None}, None

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, "Invalid 'seed_guard_distance_km': expected a number"

    if value < 0:
        return None, "Invalid 'seed_guard_distance_km': must be >= 0"
    if value > 500:
        return None, "Invalid 'seed_guard_distance_km': must be <= 500"

    return {'seed_guard_distance_km': value}, None


def parse_ttwa_params(payload):
    payload = payload or {}
    raw = payload.get('containment_threshold')
    if raw is None or raw == '':
        return {'containment_threshold': None}, None

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, "Invalid 'containment_threshold': expected a number"

    if value < 0 or value > 1:
        return None, "Invalid 'containment_threshold': must be between 0 and 1"

    return {'containment_threshold': value}, None


def parse_hierarchical_params(payload):
    payload = payload or {}

    local_radius_km, err = _parse_float(
        payload, 'local_radius_km', min_value=1.0, max_value=250.0, strictly_positive=True
    )
    if err:
        return None, err

    core_containment_threshold, err = _parse_float(
        payload, 'core_containment_threshold', min_value=0.0, max_value=1.0
    )
    if err:
        return None, err

    core_improvement_epsilon, err = _parse_float(
        payload, 'core_improvement_epsilon', min_value=0.0, max_value=0.25
    )
    if err:
        return None, err

    satellite_flow_threshold, err = _parse_float(
        payload, 'satellite_flow_threshold', min_value=0.0, max_value=1.0
    )
    if err:
        return None, err

    max_satellites, err = _parse_int(
        payload, 'max_satellites', min_value=0, max_value=25
    )
    if err:
        return None, err

    return {
        'local_radius_km': local_radius_km,
        'core_containment_threshold': core_containment_threshold,
        'core_improvement_epsilon': core_improvement_epsilon,
        'satellite_flow_threshold': satellite_flow_threshold,
        'max_satellites': max_satellites,
    }, None


def parse_czi_optimal_params(payload):
    payload = payload or {}

    candidate_limit, err = _parse_int(
        payload, 'optimal_candidate_limit', min_value=20, max_value=400, strictly_positive=True
    )
    if err:
        return None, err

    population_floor_ratio, err = _parse_float(
        payload, 'optimal_population_floor_ratio', min_value=0.0, max_value=1.0
    )
    if err:
        return None, err

    mip_rel_gap, err = _parse_float(
        payload, 'optimal_mip_rel_gap', min_value=0.0, max_value=0.2
    )
    if err:
        return None, err

    time_limit_sec, err = _parse_float(
        payload, 'optimal_time_limit_sec', min_value=1.0, max_value=300.0, strictly_positive=True
    )
    if err:
        return None, err

    max_iters, err = _parse_int(
        payload, 'optimal_max_iters', min_value=1, max_value=30, strictly_positive=True
    )
    if err:
        return None, err

    return {
        'optimal_candidate_limit': candidate_limit,
        'optimal_population_floor_ratio': population_floor_ratio,
        'optimal_mip_rel_gap': mip_rel_gap,
        'optimal_time_limit_sec': time_limit_sec,
        'optimal_max_iters': max_iters,
    }, None
