import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server_app import seed_regions
from server_app.analysis_service import PreviewClusteringService


class FakeDigraph:
    def __init__(self):
        self.nodes = {}
        self._out_edges = {}
        self._in_edges = {}

    def add_node(self, node, self_weight=0.0):
        self.nodes.setdefault(node, {'self_weight': float(self_weight)})

    def add_edge(self, src, dst, weight):
        self.add_node(src)
        self.add_node(dst)
        payload = (src, dst, {'weight': float(weight)})
        self._out_edges.setdefault(src, []).append(payload)
        self._in_edges.setdefault(dst, []).append(payload)

    def out_edges(self, node, data=True):
        return list(self._out_edges.get(node, ()))

    def in_edges(self, node, data=True):
        return list(self._in_edges.get(node, ()))

    def __contains__(self, node):
        return node in self.nodes


class FakeSearchEngine:
    def __init__(self, records):
        self.records = records

    def by_zipcode(self, zip_code):
        return self.records.get(str(zip_code))


class SecondOrderDestinationTests(unittest.TestCase):
    def test_compute_second_order_destinations_groups_zips_into_city_candidates(self):
        graph = FakeDigraph()
        graph.add_node('100000000001', self_weight=10)
        graph.add_node('100000000002', self_weight=5)
        graph.add_node('200000000001')
        graph.add_node('200000000002')
        graph.add_node('200000000003')
        graph.add_node('200000000004')
        graph.add_node('300000000001')

        graph.add_edge('100000000001', '100000000002', 7)
        graph.add_edge('100000000001', '200000000001', 30)
        graph.add_edge('100000000001', '200000000002', 20)
        graph.add_edge('100000000002', '200000000003', 15)
        graph.add_edge('100000000002', '300000000001', 12)
        graph.add_edge('200000000001', '100000000001', 8)
        graph.add_edge('200000000003', '100000000002', 4)
        graph.add_edge('300000000001', '100000000001', 1)

        resources = SimpleNamespace(
            get_directed_mobility_graph=lambda *args, **kwargs: graph
        )
        service = PreviewClusteringService(clustering_store=SimpleNamespace(), resources=resources)

        population_by_cbg = {
            '100000000001': 40,
            '100000000002': 35,
            '200000000001': 100,
            '200000000002': 50,
            '200000000003': 60,
            '200000000004': 25,
            '300000000001': 80,
        }
        cbg_to_zip = {
            '100000000001': '74002',
            '100000000002': '74002',
            '200000000001': '74103',
            '200000000002': '74104',
            '200000000003': '74105',
            '200000000004': '74103',
            '300000000001': '74006',
        }
        zip_to_cbgs = {
            '74002': ['100000000001', '100000000002'],
            '74103': ['200000000001', '200000000004'],
            '74104': ['200000000002'],
            '74105': ['200000000003'],
            '74006': ['300000000001'],
        }

        def describe_zip(zip_code):
            if zip_code == '74002':
                return {
                    'unit_id': 'city:OK:barnsdall',
                    'label': 'Barnsdall, OK',
                    'unit_type': 'city_approximation',
                }
            if zip_code in {'74103', '74104', '74105'}:
                return {
                    'unit_id': 'city:OK:tulsa',
                    'label': 'Tulsa, OK',
                    'unit_type': 'city_approximation',
                }
            if zip_code == '74006':
                return {
                    'unit_id': 'city:OK:bartlesville',
                    'label': 'Bartlesville, OK',
                    'unit_type': 'city_approximation',
                }
            return None

        pattern_selection = SimpleNamespace(
            file_path='patterns.csv',
            month='2019-01',
            source='unit-test',
            use_test_data=False,
        )

        with patch('server_app.analysis_service.Config', return_value=SimpleNamespace()), \
             patch('server_app.analysis_service.setup_logging', return_value=SimpleNamespace()), \
             patch('server_app.analysis_service.cbg_population', side_effect=lambda cbg, *_args, **_kwargs: population_by_cbg.get(cbg, 0)), \
             patch('server_app.analysis_service.get_cbg_to_zip_map', return_value=cbg_to_zip), \
             patch('server_app.analysis_service.get_zip_to_cbgs_map', return_value=zip_to_cbgs), \
             patch('server_app.analysis_service.describe_city_approximation_for_zip', side_effect=describe_zip):
            result = service.compute_second_order_destinations(
                '100000000001',
                ['100000000001', '100000000002'],
                limit=10,
                pattern_selection=pattern_selection,
            )

        self.assertEqual(result['seed_city_labels'], ['Barnsdall, OK'])
        self.assertEqual(result['seed_zip_codes'], ['74002'])
        self.assertEqual(result['destination_count'], 2)
        self.assertEqual(result['recommended_unit_ids'], ['city:OK:tulsa', 'city:OK:bartlesville'])

        tulsa = result['destinations'][0]
        self.assertEqual(tulsa['unit_id'], 'city:OK:tulsa')
        self.assertEqual(tulsa['label'], 'Tulsa, OK')
        self.assertEqual(tulsa['zip_codes'], ['74103', '74104', '74105'])
        self.assertEqual(tulsa['cbgs'], ['200000000001', '200000000002', '200000000003'])
        self.assertEqual(tulsa['cbg_count'], 3)
        self.assertEqual(tulsa['city_cbg_count'], 4)
        self.assertEqual(tulsa['population'], 210)
        self.assertEqual(tulsa['city_population'], 235)
        self.assertAlmostEqual(tulsa['outbound_flow'], 65.0)
        self.assertAlmostEqual(tulsa['inbound_flow'], 12.0)
        self.assertAlmostEqual(tulsa['bidirectional_flow'], 77.0)
        self.assertAlmostEqual(tulsa['share_of_seed_external_bidirectional'], 77.0 / 90.0)
        self.assertAlmostEqual(tulsa['share_of_seed_external_outbound'], 65.0 / 77.0)
        self.assertAlmostEqual(tulsa['captured_bidirectional_flow_share'], 1.0)

        bartlesville = result['destinations'][1]
        self.assertEqual(bartlesville['unit_id'], 'city:OK:bartlesville')
        self.assertEqual(bartlesville['zip_codes'], ['74006'])
        self.assertEqual(bartlesville['population'], 80)
        self.assertEqual(bartlesville['city_population'], 80)
        self.assertAlmostEqual(bartlesville['outbound_flow'], 12.0)
        self.assertAlmostEqual(bartlesville['inbound_flow'], 1.0)
        self.assertAlmostEqual(bartlesville['share_of_seed_external_bidirectional'], 13.0 / 90.0)

        self.assertEqual(result['recommended_explicit_population'], 365)
        self.assertEqual(result['recommended_explicit_population_cap'], 25000)

    def test_describe_city_approximation_for_zip_falls_back_to_zip_when_city_missing(self):
        class ZipRecord:
            state = 'OK'
            major_city = None
            post_office_city = None
            city = None
            common_city_list = []

        seed_regions.describe_city_approximation_for_zip.cache_clear()
        with patch.object(
            seed_regions.DEFAULT_ALGORITHM_CACHE,
            'get_search_engine',
            return_value=FakeSearchEngine({'74006': ZipRecord()}),
        ):
            result = seed_regions.describe_city_approximation_for_zip('74006')
        seed_regions.describe_city_approximation_for_zip.cache_clear()

        self.assertEqual(result['unit_id'], 'zip:74006')
        self.assertEqual(result['label'], 'ZIP 74006')
        self.assertEqual(result['unit_type'], 'zip_fallback')
        self.assertEqual(result['state'], 'OK')


if __name__ == '__main__':
    unittest.main()
