import logging

import networkx as nx

from czcode_modules.clustering import Clustering


class DummyConfig:
    def __init__(self, *args, **kwargs):
        pass


def test_hierarchical_core_satellites_grows_local_core_then_adds_zip_satellite(monkeypatch):
    populations = {
        'seed_a': 100,
        'seed_b': 100,
        'local_1': 100,
        'zip2_a': 100,
        'zip2_b': 100,
        'far_1': 100,
    }

    def fake_population(cbg, _config, _logger):
        return populations.get(cbg, 0)

    monkeypatch.setattr('czcode_modules.clustering.cbg_population', fake_population)

    dg = nx.DiGraph()
    dg.add_node('seed_a', self_weight=30.0)
    dg.add_node('seed_b', self_weight=30.0)
    dg.add_node('local_1', self_weight=20.0)
    dg.add_node('zip2_a', self_weight=15.0)
    dg.add_node('zip2_b', self_weight=15.0)
    dg.add_node('far_1', self_weight=10.0)

    dg.add_edge('seed_a', 'local_1', weight=10.0)
    dg.add_edge('local_1', 'seed_a', weight=12.0)
    dg.add_edge('seed_b', 'local_1', weight=8.0)
    dg.add_edge('local_1', 'seed_b', weight=8.0)

    dg.add_edge('seed_a', 'zip2_a', weight=3.0)
    dg.add_edge('zip2_a', 'seed_a', weight=4.0)
    dg.add_edge('seed_b', 'zip2_b', weight=3.0)
    dg.add_edge('zip2_b', 'seed_b', weight=4.0)
    dg.add_edge('zip2_a', 'zip2_b', weight=10.0)
    dg.add_edge('zip2_b', 'zip2_a', weight=10.0)

    dg.add_edge('seed_a', 'far_1', weight=0.5)
    dg.add_edge('far_1', 'seed_a', weight=0.5)

    cbg_to_zip = {
        'seed_a': '11111',
        'seed_b': '11111',
        'local_1': '11111',
        'zip2_a': '22222',
        'zip2_b': '22222',
        'far_1': '33333',
    }
    zip_to_cbgs = {
        '11111': ['seed_a', 'seed_b', 'local_1'],
        '22222': ['zip2_a', 'zip2_b'],
        '33333': ['far_1'],
    }
    cbg_centers = {
        'seed_a': (0.0, 0.0),
        'seed_b': (0.0, 0.01),
        'local_1': (0.0, 0.03),
        'zip2_a': (0.0, 0.40),
        'zip2_b': (0.0, 0.41),
        'far_1': (0.0, 0.60),
    }

    clustering = Clustering(DummyConfig(), logging.getLogger('test-hierarchical'))
    cluster, population, metadata = clustering.hierarchical_core_satellites(
        dg,
        ['seed_a', 'seed_b'],
        min_pop=500,
        cbg_to_zip=cbg_to_zip,
        zip_to_cbgs=zip_to_cbgs,
        cbg_centers=cbg_centers,
        local_radius_km=10.0,
        core_containment_threshold=0.60,
        satellite_flow_threshold=0.05,
        max_satellites=2,
    )

    assert cluster == ['seed_a', 'seed_b', 'local_1', 'zip2_a', 'zip2_b']
    assert population == 500
    assert metadata['core_cluster'] == ['seed_a', 'seed_b', 'local_1']
    assert metadata['core_population'] == 300
    assert metadata['selected_satellites'][0]['unit_id'] == '22222'
    assert metadata['population_target_met'] is True
