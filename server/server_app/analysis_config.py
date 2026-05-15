from common_geo import normalize_cbg
from czcode import Config


class LightweightAnalysisConfig:
    """Minimal config for graph analyses that already have their graph inputs."""

    def __init__(self, seed_cbg, min_pop=0, patterns_file=None, month=None):
        self.core_cbg = str(seed_cbg)
        self.min_cluster_pop = min_pop
        self.month = month
        self.states = []
        self.paths = {
            'patterns_csv': patterns_file,
            'population_csv': './data/cbg_b01.csv',
        }


def build_analysis_config(seed_cbg, min_pop=0, patterns_file=None, month=None):
    if normalize_cbg(seed_cbg):
        return Config(seed_cbg, min_pop, patterns_file=patterns_file, month=month)
    return LightweightAnalysisConfig(seed_cbg, min_pop, patterns_file=patterns_file, month=month)


EFFECTIVE_PARAM_KEYS_BY_ALGORITHM = {
    'czi_balanced': 'effective_czi_params',
    'czi_optimal_cap': 'effective_optimal_params',
    'greedy_weight_seed_guard': 'effective_seed_guard_params',
    'greedy_ttwa': 'effective_ttwa_params',
    'hierarchical_core_satellites': 'effective_hierarchical_params',
    'mobility_prune': 'effective_mobility_prune_params',
}


def effective_params_for_algorithm(algorithm_config):
    key = EFFECTIVE_PARAM_KEYS_BY_ALGORITHM.get(algorithm_config.get('algorithm'))
    if not key:
        return {}
    return algorithm_config.get(key, {}) or {}
