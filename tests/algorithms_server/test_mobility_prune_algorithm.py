import logging
import random

import networkx as nx
import pytest

from czcode_modules.clustering import Clustering


class DummyConfig:
    def __init__(self, *args, **kwargs):
        pass


def _patch_population(monkeypatch, populations):
    def fake_population(cbg, _config, _logger):
        return populations.get(cbg, 0)

    monkeypatch.setattr('czcode_modules.mobility_prune.cbg_population', fake_population)


def _clustering():
    return Clustering(DummyConfig(), logging.getLogger('test-mobility-prune'))


def _graph(populations, self_weights, edges):
    graph = nx.Graph()
    for cbg in populations:
        graph.add_node(cbg, self_weight=float(self_weights.get(cbg, 0.0)))
    for a, b, weight in edges:
        graph.add_edge(a, b, weight=float(weight))
    return graph


def test_mobility_prune_takes_seed_neighbors_then_removes_lowest_seed_movement_per_resident(
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
    cluster, population, metadata = _clustering().mobility_prune(
        graph,
        ['seed_a', 'seed_b'],
        min_pop=500,
        min_seed_capture=0.95,
        trace_collector=trace,
    )

    # Total seed movement 129: self 20 + seed-seed 50 + neighbours 50 + 1 + 5 + 3.
    assert cluster == ['core_1', 'low_value_small', 'seed_a', 'seed_b']
    assert population == 500
    assert metadata['bounded_envelope'] is True
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


def test_mobility_prune_skips_blocked_candidates_and_keeps_pruning(monkeypatch):
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

    cluster, population, metadata = _clustering().mobility_prune(
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


def test_mobility_prune_never_removes_seeds_and_reports_missing_seeds(monkeypatch):
    populations = {'seed_a': 50, 'seed_b': 0, 'n1': 100}
    _patch_population(monkeypatch, populations)
    graph = _graph(
        populations,
        {'seed_a': 1.0},
        [('seed_a', 'seed_b', 1.0), ('seed_b', 'n1', 1.0)],
    )

    cluster, population, metadata = _clustering().mobility_prune(
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


def test_mobility_prune_raises_when_no_seed_is_in_graph(monkeypatch):
    _patch_population(monkeypatch, {})
    with pytest.raises(ValueError):
        _clustering().mobility_prune(nx.Graph(), ['missing'], min_pop=0)


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
def test_mobility_prune_single_pass_matches_iterative_rule(monkeypatch, trial):
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

    cluster, population, metadata = _clustering().mobility_prune(
        graph, seeds, min_pop=0, min_seed_capture=threshold
    )

    expected = _reference_iterative_prune(graph, seeds, populations, threshold)
    assert cluster == expected
    assert population == sum(populations[cbg] for cbg in expected)
    if metadata['initial_seed_capture_share'] >= threshold:
        assert metadata['final_seed_capture_share'] + 1e-9 >= threshold
    # Every retained non-seed CBG is a direct seed neighbour, so the zone can
    # never be disconnected from the seed region.
    seed_set = set(seeds)
    for cbg in set(cluster) - seed_set:
        assert any(neighbor in seed_set for neighbor in graph.adj[cbg])
