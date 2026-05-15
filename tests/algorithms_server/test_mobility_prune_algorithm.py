import logging

import networkx as nx

from czcode_modules.clustering import Clustering


class DummyConfig:
    def __init__(self, *args, **kwargs):
        pass


def test_mobility_prune_removes_lowest_movement_loss_per_person(monkeypatch):
    populations = {
        'seed_a': 100,
        'seed_b': 100,
        'core_1': 100,
        'low_value_large': 300,
        'low_value_small': 200,
    }

    def fake_population(cbg, _config, _logger):
        return populations.get(cbg, 0)

    monkeypatch.setattr('czcode_modules.mobility_prune.cbg_population', fake_population)

    graph = nx.Graph()
    for cbg in populations:
        graph.add_node(cbg, self_weight=0.0)

    graph.nodes['seed_a']['self_weight'] = 10.0
    graph.nodes['seed_b']['self_weight'] = 10.0
    graph.nodes['core_1']['self_weight'] = 4.0

    graph.add_edge('seed_a', 'seed_b', weight=50.0)
    graph.add_edge('seed_a', 'core_1', weight=30.0)
    graph.add_edge('seed_b', 'core_1', weight=20.0)
    graph.add_edge('seed_a', 'low_value_large', weight=1.0)
    graph.add_edge('seed_b', 'low_value_small', weight=5.0)

    trace = []
    clustering = Clustering(DummyConfig(), logging.getLogger('test-mobility-prune'))
    cluster, population, metadata = clustering.mobility_prune(
        graph,
        ['seed_a', 'seed_b'],
        min_pop=500,
        min_seed_capture=0.97,
        trace_collector=trace,
    )

    assert cluster == ['core_1', 'low_value_small', 'seed_a', 'seed_b']
    assert population == 500
    assert 'low_value_large' not in cluster
    assert {'seed_a', 'seed_b'}.issubset(set(cluster))
    assert metadata['initial_population'] == 800
    assert metadata['population_reduced'] == 300
    assert metadata['removed_cbg_count'] == 1
    assert metadata['minimum_population_used'] is False
    assert metadata['legacy_min_population'] == 500
    assert metadata['final_seed_capture_share'] >= 0.97
    assert metadata['stopped_by_seed_capture_floor'] is True
    assert trace[0]['metrics_after']['stage'] == 'bounded_envelope_growth'
    prune_steps = [
        step for step in trace
        if step.get('metrics_after', {}).get('stage') == 'reverse_prune'
    ]
    assert prune_steps[0]['selected_cbg'] == 'low_value_large'
    assert prune_steps[0]['candidates'][0]['cbg'] == 'low_value_large'


def test_mobility_prune_does_not_disconnect_retained_non_seed_cbgs(monkeypatch):
    populations = {
        'seed': 100,
        'bridge': 100,
        'leaf': 300,
    }

    def fake_population(cbg, _config, _logger):
        return populations.get(cbg, 0)

    monkeypatch.setattr('czcode_modules.mobility_prune.cbg_population', fake_population)

    graph = nx.Graph()
    for cbg in populations:
        graph.add_node(cbg, self_weight=0.0)
    graph.add_edge('seed', 'bridge', weight=1.0)
    graph.add_edge('bridge', 'leaf', weight=100.0)

    clustering = Clustering(DummyConfig(), logging.getLogger('test-mobility-prune'))
    cluster, population, _metadata = clustering.mobility_prune(
        graph,
        ['seed'],
        min_pop=400,
    )

    assert cluster == ['bridge', 'seed']
    assert population == 200


def test_mobility_prune_prunes_remote_component_after_large_envelope(monkeypatch):
    populations = {
        'seed': 100,
        'strong_a': 300,
        'strong_b': 300,
        'weak_bridge': 300,
        'remote_huge': 1_000_000,
    }

    def fake_population(cbg, _config, _logger):
        return populations.get(cbg, 0)

    monkeypatch.setattr('czcode_modules.mobility_prune.cbg_population', fake_population)

    graph = nx.Graph()
    for cbg in populations:
        graph.add_node(cbg, self_weight=0.0)

    graph.add_edge('seed', 'strong_a', weight=50.0)
    graph.add_edge('seed', 'strong_b', weight=40.0)
    graph.add_edge('seed', 'weak_bridge', weight=1.0)
    graph.add_edge('weak_bridge', 'remote_huge', weight=1.0)

    clustering = Clustering(DummyConfig(), logging.getLogger('test-mobility-prune'))
    cluster, population, metadata = clustering.mobility_prune(
        graph,
        ['seed'],
        min_pop=500,
        min_seed_capture=0.95,
    )

    assert cluster == ['seed', 'strong_a', 'strong_b']
    assert population == 700
    assert 'remote_huge' not in cluster
    assert metadata['bounded_envelope'] is True
    assert metadata['envelope_population_target'] == 100000
    assert metadata['initial_cbg_count'] == 5
    assert metadata['initial_population'] == 1001000
    assert metadata['removed_cbg_count'] == 2
