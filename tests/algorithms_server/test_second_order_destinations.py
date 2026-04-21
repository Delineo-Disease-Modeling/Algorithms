import networkx as nx

from server_app.analysis_service import PreviewClusteringService
from server_app import seed_regions


class DummyPatternSelection:
    def __init__(self):
        self.file_path = '/tmp/patterns.parquet'
        self.source = 'monthly'
        self.month = '2021-07'
        self.use_test_data = False


class StubResources:
    def __init__(self, digraph):
        self._digraph = digraph

    def get_directed_mobility_graph(self, seed_cbg, patterns_file=None, patterns_folder=None, month=None, cache_tag='v3'):
        _ = (seed_cbg, patterns_file, patterns_folder, month, cache_tag)
        return self._digraph


def test_compute_second_order_destinations_ranks_zip_regions_by_seed_outflow(monkeypatch):
    populations = {
        'seed_a': 100,
        'seed_b': 120,
        'zip2_a': 80,
        'zip2_b': 90,
        'zip3_a': 70,
    }

    def fake_population(cbg, _config, _logger):
        return populations.get(cbg, 0)

    monkeypatch.setattr('server_app.analysis_service.cbg_population', fake_population)
    monkeypatch.setattr('server_app.analysis_service.Config', lambda *args, **kwargs: DummyPatternSelection())
    monkeypatch.setattr('server_app.analysis_service.setup_logging', lambda _config: None)
    monkeypatch.setattr(
        'server_app.analysis_service.get_cbg_to_zip_map',
        lambda: {
            'seed_a': '11111',
            'seed_b': '11111',
            'zip2_a': '22222',
            'zip2_b': '22222',
            'zip3_a': '33333',
        },
    )
    monkeypatch.setattr(
        'server_app.analysis_service.get_zip_to_cbgs_map',
        lambda: {
            '11111': ['seed_a', 'seed_b'],
            '22222': ['zip2_a', 'zip2_b'],
            '33333': ['zip3_a'],
        },
    )

    digraph = nx.DiGraph()
    digraph.add_node('seed_a', self_weight=50.0)
    digraph.add_node('seed_b', self_weight=40.0)
    digraph.add_node('zip2_a', self_weight=5.0)
    digraph.add_node('zip2_b', self_weight=4.0)
    digraph.add_node('zip3_a', self_weight=3.0)

    digraph.add_edge('seed_a', 'seed_b', weight=15.0)
    digraph.add_edge('seed_a', 'zip2_a', weight=20.0)
    digraph.add_edge('seed_b', 'zip2_b', weight=10.0)
    digraph.add_edge('seed_b', 'zip3_a', weight=5.0)
    digraph.add_edge('zip2_a', 'seed_a', weight=7.0)
    digraph.add_edge('zip3_a', 'seed_b', weight=2.0)

    service = PreviewClusteringService(clustering_store=None, resources=StubResources(digraph))
    result = service.compute_second_order_destinations(
        seed_cbg='seed_a',
        seed_cbgs=['seed_a', 'seed_b'],
        limit=10,
        pattern_selection=DummyPatternSelection(),
    )

    assert result['seed_cbgs'] == ['seed_a', 'seed_b']
    assert result['seed_population'] == 220
    assert result['destination_count'] == 2
    assert result['destinations'][0]['unit_id'] == 'zip:22222'
    assert result['destinations'][0]['outbound_flow'] == 30.0
    assert result['destinations'][0]['inbound_flow'] == 7.0
    assert result['destinations'][0]['population'] == 170
    assert result['destinations'][0]['recommended'] is True
    assert result['destinations'][1]['unit_id'] == 'zip:33333'
    assert round(result['total_seed_external_outbound_flow'], 4) == 35.0
    assert round(result['recommended_captured_external_outbound_share'], 4) == 1.0


def test_get_zip_to_cbgs_map_uses_bundled_fallback_when_primary_missing(tmp_path, monkeypatch):
    fallback_path = tmp_path / 'bundled' / 'zip_to_cbg.json'
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text('{"74002": ["401139400082"]}', encoding='utf-8')

    monkeypatch.setattr(
        seed_regions,
        'ZIP_TO_CBG_PATHS',
        (
            str(tmp_path / 'missing' / 'zip_to_cbg.json'),
            str(fallback_path),
        ),
    )
    seed_regions.get_zip_to_cbgs_map.cache_clear()
    seed_regions.get_cbg_to_zip_map.cache_clear()

    try:
        assert seed_regions.get_zip_to_cbgs_map() == {
            '74002': ['401139400082']
        }
    finally:
        seed_regions.get_zip_to_cbgs_map.cache_clear()
        seed_regions.get_cbg_to_zip_map.cache_clear()
