import logging
import random

import networkx as nx
import pytest

from czcode_modules.algorithm_runner import AlgorithmRunner, build_trace_payload
from czcode_modules.clustering import Clustering
from server_app.request_parsing import parse_cluster_algorithm_config


class DummyConfig:
    def __init__(self, *args, **kwargs):
        pass


def _patch_population(monkeypatch, populations):
    def fake_population(cbg, _config, _logger):
        return populations.get(cbg, 0)

    monkeypatch.setattr('czcode_modules.seed_prune.cbg_population', fake_population)


def _clustering():
    return Clustering(DummyConfig(), logging.getLogger('test-seed-prune'))


def _graph(populations, self_weights, edges):
    graph = nx.Graph()
    for cbg in populations:
        graph.add_node(cbg, self_weight=float(self_weights.get(cbg, 0.0)))
    for a, b, weight in edges:
        graph.add_edge(a, b, weight=float(weight))
    return graph


def test_seed_prune_takes_seed_neighbors_then_removes_lowest_seed_movement_per_resident(
    monkeypatch,
):
    populations = {
        'seed_a': 100,
        'seed_b': 100,
        'core_1': 100,
        'low_value_large': 300,
        'low_value_small': 200,
        'second_order': 500,
        'zero_pop': 0,
    }
    _patch_population(monkeypatch, populations)
    graph = _graph(
        populations,
        {'seed_a': 10.0, 'seed_b': 10.0, 'core_1': 4.0},
        [
            ('seed_a', 'seed_b', 50.0),
            ('seed_a', 'core_1', 30.0),
            ('seed_b', 'core_1', 20.0),
            ('seed_a', 'low_value_large', 1.0),
            ('seed_b', 'low_value_small', 5.0),
            ('core_1', 'second_order', 100.0),
            ('seed_a', 'zero_pop', 3.0),
        ],
    )

    trace = []
    cluster, population, metadata = _clustering().seed_prune(
        graph,
        ['seed_a', 'seed_b'],
        min_pop=500,
        min_seed_capture=0.95,
        trace_collector=trace,
    )

    # Total seed movement 129: self 20 + seed-seed 50 + neighbours 50 + 1 + 5 + 3.
    assert cluster == ['core_1', 'low_value_small', 'seed_a', 'seed_b']
    assert population == 500
    assert metadata['bounded_envelope'] is False
    assert metadata['universe_rule'] == 'seed_neighbors'
    assert metadata['initial_cbg_count'] == 5
    assert metadata['initial_population'] == 800
    assert metadata['excluded_zero_population_cbg_count'] == 1
    assert metadata['excluded_zero_population_seed_movement'] == pytest.approx(3.0)
    assert metadata['seed_movement_total'] == pytest.approx(129.0)
    assert metadata['initial_seed_capture_share'] == pytest.approx(126 / 129)
    assert metadata['final_seed_capture_share'] == pytest.approx(125 / 129)
    assert metadata['seed_capture_target_met'] is True
    assert metadata['stopped_by_seed_capture_floor'] is True
    assert metadata['removed_cbg_count'] == 1
    assert metadata['blocked_cbg_count'] == 2
    assert metadata['population_reduced'] == 300
    assert metadata['minimum_population_used'] is False
    assert metadata['legacy_min_population'] == 500

    assert len(trace) == 1
    step = trace[0]
    assert step['metrics_after']['stage'] == 'reverse_prune'
    assert step['selected_cbg'] == 'low_value_large'
    assert step['candidates'][0]['cbg'] == 'low_value_large'
    assert step['candidates'][0]['selected'] is True
    assert 'czi_after' in step['candidates'][0]
    assert 'second_order' not in step['cluster_before']
    assert 'zero_pop' not in step['cluster_before']
    assert step['metrics_after']['seed_capture'] == pytest.approx(125 / 129)


def test_seed_prune_skips_blocked_candidates_and_keeps_pruning(monkeypatch):
    populations = {'seed': 100, 'big_cheap': 1000, 'tiny_dear': 10, 'keep': 100}
    _patch_population(monkeypatch, populations)
    graph = _graph(
        populations,
        {},
        [
            ('seed', 'big_cheap', 40.0),
            ('seed', 'tiny_dear', 5.0),
            ('seed', 'keep', 55.0),
        ],
    )

    cluster, population, metadata = _clustering().seed_prune(
        graph,
        ['seed'],
        min_pop=0,
        min_seed_capture=0.90,
    )

    # big_cheap ranks first but would drop capture to 0.60, so it is skipped;
    # tiny_dear ranks after it and is still removed.
    assert cluster == ['big_cheap', 'keep', 'seed']
    assert population == 1200
    assert metadata['final_seed_capture_share'] == pytest.approx(0.95)
    assert metadata['blocked_cbg_count'] == 2
    assert metadata['removed_cbg_count'] == 1


def test_seed_prune_never_removes_seeds_and_reports_missing_seeds(monkeypatch):
    populations = {'seed_a': 50, 'seed_b': 0, 'n1': 100}
    _patch_population(monkeypatch, populations)
    graph = _graph(
        populations,
        {'seed_a': 1.0},
        [('seed_a', 'seed_b', 1.0), ('seed_b', 'n1', 1.0)],
    )

    cluster, population, metadata = _clustering().seed_prune(
        graph,
        ['seed_a', 'seed_b', 'not_in_graph'],
        min_pop=0,
        min_seed_capture=0.0,
    )

    assert cluster == ['seed_a', 'seed_b']
    assert population == 50
    assert metadata['seed_cbgs'] == ['seed_a', 'seed_b']
    assert metadata['missing_seed_cbgs'] == ['not_in_graph']
    assert metadata['stopped_by_seed_capture_floor'] is False


def test_seed_prune_raises_when_no_seed_is_in_graph(monkeypatch):
    _patch_population(monkeypatch, {})
    with pytest.raises(ValueError):
        _clustering().seed_prune(nx.Graph(), ['missing'], min_pop=0)


def _reference_iterative_prune(graph, seeds, populations, threshold):
    """Literal statement of the rule: repeatedly remove the lowest-scoring
    removable candidate until none can be removed."""
    seed_set = set(seeds)
    total = sum(graph.nodes[s].get('self_weight', 0.0) for s in seed_set)
    always = total
    by_cbg = {}
    for a, b, data in graph.edges(data=True):
        weight = data['weight']
        if a in seed_set or b in seed_set:
            total += weight
            if a in seed_set and b in seed_set:
                always += weight
            else:
                other = b if a in seed_set else a
                by_cbg[other] = by_cbg.get(other, 0.0) + weight
    zone = set(seed_set) | {cbg for cbg in by_cbg if populations[cbg] > 0}
    captured = always + sum(by_cbg[cbg] for cbg in zone - seed_set)
    while True:
        best = None
        for cbg in zone - seed_set:
            loss = by_cbg[cbg]
            if (captured - loss) / total < threshold and loss > 1e-12:
                continue
            key = (loss / populations[cbg], loss, -populations[cbg], cbg)
            if best is None or key < best[0]:
                best = (key, cbg)
        if best is None:
            return sorted(zone)
        zone.remove(best[1])
        captured -= by_cbg[best[1]]


@pytest.mark.parametrize('trial', range(40))
def test_seed_prune_single_pass_matches_iterative_rule(monkeypatch, trial):
    rng = random.Random(trial)
    graph = nx.gnm_random_graph(40, 140, seed=trial)
    graph = nx.relabel_nodes(graph, {n: f'cbg_{n:02d}' for n in graph.nodes})
    for a, b in graph.edges:
        graph.edges[a, b]['weight'] = float(rng.choice([0, 1, 2, 4, 8, 20, 60]))
    populations = {}
    for node in graph.nodes:
        graph.nodes[node]['self_weight'] = float(rng.randint(0, 30))
        populations[node] = rng.choice([0, 5, 50, 300, 1200, 2500])
    _patch_population(monkeypatch, populations)
    seeds = rng.sample(sorted(graph.nodes), rng.randint(1, 4))
    threshold = rng.choice([0.5, 0.7, 0.8, 0.9])

    cluster, population, metadata = _clustering().seed_prune(
        graph, seeds, min_pop=0, min_seed_capture=threshold
    )

    expected = _reference_iterative_prune(graph, seeds, populations, threshold)
    assert cluster == expected
    assert population == sum(populations[cbg] for cbg in expected)
    if metadata['initial_seed_capture_share'] >= threshold:
        assert metadata['final_seed_capture_share'] + 1e-9 >= threshold


class _FakeRunnerClustering:
    def __init__(self):
        self.calls = []

    def seed_prune(self, graph, seed_cbgs, min_pop, **kwargs):
        self.calls.append((seed_cbgs, min_pop, kwargs))
        return ['seed'], 100, {'bounded_envelope': False}


class _RunnerConfig:
    core_cbg = 'seed'
    min_cluster_pop = 100
    states = ['OK']
    paths = {'patterns_csv': '/tmp/patterns.parquet'}


def test_algorithm_runner_dispatches_seed_prune_with_seed_capture_floor():
    fake = _FakeRunnerClustering()
    graph = nx.Graph()
    graph.add_node('seed')
    runner = AlgorithmRunner(
        clustering_algo=fake,
        config=_RunnerConfig(),
        logger=logging.getLogger('test-seed-prune-runner'),
        graph=graph,
        patterns_df=None,
        cbg_centers={},
        cache_service=None,
    )
    trace_steps = []

    result = runner.run(
        'seed_prune',
        ['seed', 'other'],
        trace_steps=trace_steps,
        mobility_prune_min_seed_capture='0.7',
    )

    assert result.metadata == {'bounded_envelope': False}
    seed_cbgs, min_pop, kwargs = fake.calls[0]
    assert seed_cbgs == ['seed', 'other']
    assert min_pop == 100
    assert kwargs['min_seed_capture'] == 0.7
    assert kwargs['trace_collector'] is trace_steps


def test_seed_prune_trace_payload_carries_note_and_metadata():
    payload = build_trace_payload('seed_prune', 'seed', [{'iteration': 0}], {'universe_rule': 'seed_neighbors'})

    assert payload['supports_stepwise'] is True
    assert payload['algorithm_metadata'] == {'universe_rule': 'seed_neighbors'}
    assert 'direct movement link to the seed' in payload['note']


@pytest.mark.parametrize('alias', ['seed_prune', 'take_all_prune', 'seed_neighbor_prune'])
def test_request_parsing_accepts_seed_prune_with_default_floor(alias):
    config = parse_cluster_algorithm_config({'algorithm': alias})

    assert config['algorithm'] == 'seed_prune'
    assert config['effective_mobility_prune_params'] == {'min_seed_capture': 0.80}


def test_request_parsing_passes_seed_capture_floor_to_seed_prune():
    config = parse_cluster_algorithm_config({
        'algorithm': 'seed_prune',
        'mobility_prune_min_seed_capture': 0.65,
    })

    assert config['effective_mobility_prune_params'] == {'min_seed_capture': 0.65}


def test_cluster_cbgs_route_accepts_seed_prune(client, app, monkeypatch):
    monkeypatch.setattr(
        'server_app.request_parsing.resolve_patterns_file_for_request',
        lambda seed_cbg, start_date_raw=None, use_test_data=False: ('/tmp/patterns.parquet', 'monthly', '2021-04'),
    )
    captured = {}

    def fake_start_cluster_job(cbg_str, min_pop, pattern_selection, algorithm_config, include_trace, seed_cbgs=None):
        captured['algorithm_config'] = algorithm_config
        captured['seed_cbgs'] = seed_cbgs
        return 7

    monkeypatch.setattr(app.config['analysis_service'], 'start_cluster_job', fake_start_cluster_job)

    response = client.post('/cluster-cbgs', json={
        'cbg': '401139400081',
        'seed_cbgs': ['401139400081', '401139400082'],
        'min_pop': 5000,
        'algorithm': 'seed_prune',
        'start_date': '2021-04-01',
    })

    assert response.status_code == 200
    assert captured['algorithm_config']['algorithm'] == 'seed_prune'
    assert captured['algorithm_config']['effective_mobility_prune_params'] == {
        'min_seed_capture': 0.80,
    }
    assert captured['seed_cbgs'] == ['401139400081', '401139400082']
