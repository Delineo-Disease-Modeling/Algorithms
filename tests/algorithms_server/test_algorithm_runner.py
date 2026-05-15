import logging

import networkx as nx

from czcode_modules.algorithm_runner import (
    AlgorithmRunner,
    build_trace_payload,
    normalize_seed_cbgs,
)


class DummyConfig:
    core_cbg = 'seed'
    min_cluster_pop = 100
    states = ['MD']
    paths = {'patterns_csv': '/tmp/patterns.parquet'}


class DummyCache:
    def get_or_build_graph(self, _cache_key, build_fn):
        return build_fn()


class FakeClustering:
    def __init__(self):
        self.calls = []

    def greedy_czi_balanced(self, graph, seed_cbg, min_pop, **kwargs):
        self.calls.append(('czi_balanced', graph, seed_cbg, min_pop, kwargs))
        return ['seed'], 100

    def mobility_prune(self, graph, seed_cbgs, min_pop, **kwargs):
        self.calls.append(('mobility_prune', graph, seed_cbgs, min_pop, kwargs))
        return ['seed'], 100, {'bounded_envelope': True}


def build_runner(fake_clustering):
    graph = nx.Graph()
    graph.add_node('seed')
    return AlgorithmRunner(
        clustering_algo=fake_clustering,
        config=DummyConfig(),
        logger=logging.getLogger('test-algorithm-runner'),
        graph=graph,
        patterns_df=None,
        cbg_centers={'seed': (0.0, 0.0)},
        cache_service=DummyCache(),
    )


def test_algorithm_runner_dispatches_czi_balanced_params():
    fake_clustering = FakeClustering()
    runner = build_runner(fake_clustering)
    trace_steps = []

    result = runner.run(
        'czi_balanced',
        ['seed'],
        trace_steps=trace_steps,
        distance_penalty_weight='0.25',
        distance_scale_km='15',
    )

    assert result.result == (['seed'], 100)
    algorithm, _graph, seed_cbg, min_pop, kwargs = fake_clustering.calls[0]
    assert algorithm == 'czi_balanced'
    assert seed_cbg == 'seed'
    assert min_pop == 100
    assert kwargs['distance_penalty_weight'] == 0.25
    assert kwargs['distance_scale_km'] == 15.0
    assert kwargs['cbg_centers'] == {'seed': (0.0, 0.0)}
    assert kwargs['trace_collector'] is trace_steps


def test_algorithm_runner_extracts_metadata_from_region_algorithm():
    fake_clustering = FakeClustering()
    runner = build_runner(fake_clustering)

    result = runner.run(
        'mobility_prune',
        ['seed', 'seed'],
        mobility_prune_min_seed_capture='0.6',
    )

    assert result.metadata == {'bounded_envelope': True}
    algorithm, _graph, seed_cbgs, min_pop, kwargs = fake_clustering.calls[0]
    assert algorithm == 'mobility_prune'
    assert seed_cbgs == ['seed', 'seed']
    assert min_pop == 100
    assert kwargs['min_seed_capture'] == 0.6


def test_trace_payload_carries_notes_and_metadata():
    payload = build_trace_payload(
        'mobility_prune',
        'seed',
        [{'iteration': 0}],
        {'bounded_envelope': True},
    )

    assert payload['supports_stepwise'] is True
    assert payload['steps'] == [{'iteration': 0}]
    assert payload['algorithm_metadata'] == {'bounded_envelope': True}
    assert 'bounded mobility-envelope growth' in payload['note']


def test_normalize_seed_cbgs_deduplicates_preserving_order():
    assert normalize_seed_cbgs(['a', 'b', 'a', None, 'c'], 'seed') == ['a', 'b', 'c']
    assert normalize_seed_cbgs([], 'seed') == ['seed']
