from dataclasses import dataclass

from .graph import GraphBuilder


VALID_ALGORITHMS = (
    'czi_balanced',
    'czi_optimal_cap',
    'greedy_fast',
    'greedy_weight',
    'greedy_weight_seed_guard',
    'greedy_ratio',
    'greedy_ttwa',
    'mobility_prune',
)

SEED_REGION_ALGORITHMS = {'mobility_prune'}


TRACE_NOTES = {
    'czi_optimal_cap': (
        "czi_optimal_cap is solved as a global optimization and does not have a "
        "single greedy add-one expansion sequence."
    ),
    'mobility_prune': (
        "Trace steps show bounded mobility-envelope growth followed by reverse "
        "pruning. CBGs are removed by lowest movement loss per resident removed."
    ),
}

TRACE_METADATA_ALGORITHMS = {'mobility_prune'}


@dataclass
class AlgorithmRun:
    result: tuple
    metadata: dict


def normalize_algorithm_key(algorithm):
    return str(algorithm or 'czi_balanced').strip().lower()


def normalize_seed_cbgs(seed_cbgs, fallback_seed):
    normalized = []
    seen = set()
    for seed_cbg in seed_cbgs or [fallback_seed]:
        if not seed_cbg or seed_cbg in seen:
            continue
        seen.add(seed_cbg)
        normalized.append(seed_cbg)
    return normalized


def valid_algorithm_message(algorithm):
    return (
        f"Invalid clustering algorithm '{algorithm}'. "
        f"Valid options: {', '.join(VALID_ALGORITHMS)}"
    )


def extract_algorithm_metadata(algorithm_result):
    if len(algorithm_result) > 2 and isinstance(algorithm_result[2], dict):
        return algorithm_result[2]
    return {}


def build_trace_payload(algorithm_key, seed_cbg, trace_steps, algorithm_metadata=None):
    payload = {
        'algorithm': algorithm_key,
        'seed_cbg': seed_cbg,
        'supports_stepwise': algorithm_key != 'czi_optimal_cap',
        'steps': trace_steps or [],
    }
    note = TRACE_NOTES.get(algorithm_key)
    if note:
        payload['note'] = note
    if algorithm_key in TRACE_METADATA_ALGORITHMS:
        payload['algorithm_metadata'] = algorithm_metadata or {}
    return payload


class AlgorithmRunner:
    def __init__(self, clustering_algo, config, logger, graph, patterns_df, cbg_centers, cache_service):
        self.clustering_algo = clustering_algo
        self.config = config
        self.logger = logger
        self.graph = graph
        self.patterns_df = patterns_df
        self.cbg_centers = cbg_centers
        self.cache_service = cache_service

    def run(self, algorithm_key, normalized_seed_cbgs, trace_steps=None, **params):
        handlers = self._handlers()
        handler = handlers.get(algorithm_key)
        if handler is None:
            raise ValueError(valid_algorithm_message(params.get('algorithm', algorithm_key)))

        self._validate_seed_presence(algorithm_key, normalized_seed_cbgs)
        algorithm_result = handler(normalized_seed_cbgs, trace_steps, params)
        return AlgorithmRun(
            result=algorithm_result,
            metadata=extract_algorithm_metadata(algorithm_result),
        )

    def _handlers(self):
        return {
            'czi_balanced': self._run_czi_balanced,
            'czi_optimal_cap': self._run_czi_optimal_cap,
            'greedy_fast': self._run_greedy_fast,
            'greedy_weight': self._run_greedy_weight,
            'greedy_weight_seed_guard': self._run_greedy_weight_seed_guard,
            'greedy_ratio': self._run_greedy_ratio,
            'greedy_ttwa': self._run_greedy_ttwa,
            'mobility_prune': self._run_mobility_prune,
        }

    def _validate_seed_presence(self, algorithm_key, normalized_seed_cbgs):
        if algorithm_key in SEED_REGION_ALGORITHMS:
            if not any(seed_cbg in self.graph for seed_cbg in normalized_seed_cbgs):
                raise ValueError(
                    "None of the resolved seed-region CBGs are present in the mobility graph. "
                    "Try a different location or date."
                )
            return

        if self.config.core_cbg not in self.graph:
            raise ValueError(
                f"Seed CBG {self.config.core_cbg} is not present in the mobility graph for "
                f"{self.config.paths['patterns_csv']}. Try a different start date or location."
            )

    def _directed_graph(self):
        digraph_cache_key = (self.config.paths['patterns_csv'], frozenset(self.config.states), 'directed')
        digraph = self.cache_service.get_or_build_graph(
            digraph_cache_key,
            lambda: GraphBuilder(self.logger).gen_digraph(self.patterns_df),
        )
        if digraph.number_of_nodes() == 0:
            raise ValueError("No directed mobility graph could be built from the selected patterns data.")
        return digraph

    def _add_trace_collector(self, kwargs, trace_steps):
        if trace_steps is not None:
            kwargs['trace_collector'] = trace_steps
        return kwargs

    def _run_czi_balanced(self, _normalized_seed_cbgs, trace_steps, params):
        kwargs = {'cbg_centers': self.cbg_centers}
        if params.get('distance_penalty_weight') is not None:
            kwargs['distance_penalty_weight'] = float(params['distance_penalty_weight'])
        if params.get('distance_scale_km') is not None:
            kwargs['distance_scale_km'] = float(params['distance_scale_km'])
        self._add_trace_collector(kwargs, trace_steps)
        return self.clustering_algo.greedy_czi_balanced(
            self.graph,
            self.config.core_cbg,
            self.config.min_cluster_pop,
            **kwargs
        )

    def _run_czi_optimal_cap(self, _normalized_seed_cbgs, _trace_steps, params):
        kwargs = {}
        if params.get('optimal_candidate_limit') is not None:
            kwargs['candidate_limit'] = int(params['optimal_candidate_limit'])
        if params.get('optimal_population_floor_ratio') is not None:
            kwargs['population_floor_ratio'] = float(params['optimal_population_floor_ratio'])
        if params.get('optimal_mip_rel_gap') is not None:
            kwargs['mip_rel_gap'] = float(params['optimal_mip_rel_gap'])
        if params.get('optimal_time_limit_sec') is not None:
            kwargs['time_limit_sec'] = float(params['optimal_time_limit_sec'])
        if params.get('optimal_max_iters') is not None:
            kwargs['max_dinkelbach_iters'] = int(params['optimal_max_iters'])
        return self.clustering_algo.czi_optimal_cap(
            self.graph,
            self.config.core_cbg,
            self.config.min_cluster_pop,
            **kwargs
        )

    def _run_greedy_fast(self, _normalized_seed_cbgs, trace_steps, _params):
        return self.clustering_algo.greedy_fast(
            self.graph,
            self.config.core_cbg,
            self.config.min_cluster_pop,
            trace_collector=trace_steps,
        )

    def _run_greedy_weight(self, _normalized_seed_cbgs, trace_steps, _params):
        return self.clustering_algo.greedy_weight(
            self.graph,
            self.config.core_cbg,
            self.config.min_cluster_pop,
            trace_collector=trace_steps,
        )

    def _run_greedy_weight_seed_guard(self, _normalized_seed_cbgs, trace_steps, params):
        kwargs = {'cbg_centers': self.cbg_centers}
        if params.get('seed_guard_distance_km') is not None:
            kwargs['seed_guard_distance_km'] = float(params['seed_guard_distance_km'])
        self._add_trace_collector(kwargs, trace_steps)
        return self.clustering_algo.greedy_weight_seed_guard(
            self.graph,
            self.config.core_cbg,
            self.config.min_cluster_pop,
            **kwargs
        )

    def _run_greedy_ratio(self, _normalized_seed_cbgs, trace_steps, _params):
        return self.clustering_algo.greedy_ratio(
            self.graph,
            self.config.core_cbg,
            self.config.min_cluster_pop,
            trace_collector=trace_steps,
        )

    def _run_greedy_ttwa(self, _normalized_seed_cbgs, trace_steps, params):
        digraph = self._directed_graph()
        if self.config.core_cbg not in digraph:
            raise ValueError(
                f"Seed CBG {self.config.core_cbg} is not present in the directed mobility graph for "
                f"{self.config.paths['patterns_csv']}. Try a different start date or location."
            )
        kwargs = {}
        if params.get('containment_threshold') is not None:
            kwargs['containment_threshold'] = float(params['containment_threshold'])
        self._add_trace_collector(kwargs, trace_steps)
        return self.clustering_algo.greedy_ttwa(
            digraph,
            self.config.core_cbg,
            self.config.min_cluster_pop,
            **kwargs
        )

    def _run_mobility_prune(self, normalized_seed_cbgs, trace_steps, params):
        kwargs = {}
        if params.get('mobility_prune_min_seed_capture') is not None:
            kwargs['min_seed_capture'] = float(params['mobility_prune_min_seed_capture'])
        self._add_trace_collector(kwargs, trace_steps)
        return self.clustering_algo.mobility_prune(
            self.graph,
            normalized_seed_cbgs,
            self.config.min_cluster_pop,
            **kwargs
        )
