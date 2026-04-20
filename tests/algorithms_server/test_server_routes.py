def test_cluster_cbgs_route_normalizes_seed_and_returns_job_id(client, app, monkeypatch):
    monkeypatch.setattr(
        'server_app.request_parsing.resolve_patterns_file_for_request',
        lambda seed_cbg, start_date_raw=None, use_test_data=False: ('/tmp/patterns.parquet', 'monthly', '2021-01'),
    )

    captured = {}

    def fake_start_cluster_job(cbg_str, min_pop, pattern_selection, algorithm_config, include_trace, seed_cbgs=None):
        captured['cbg_str'] = cbg_str
        captured['min_pop'] = min_pop
        captured['pattern_selection'] = pattern_selection
        captured['algorithm_config'] = algorithm_config
        captured['include_trace'] = include_trace
        captured['seed_cbgs'] = seed_cbgs
        return 17

    monkeypatch.setattr(app.config['analysis_service'], 'start_cluster_job', fake_start_cluster_job)

    response = client.post('/cluster-cbgs', json={
        'cbg': '12345678901',
        'min_pop': 9000,
        'algorithm': 'balanced',
        'start_date': '2021-01-15',
        'include_trace': True,
    })

    assert response.status_code == 200
    assert response.get_json() == {'clustering_id': 17}
    assert captured['cbg_str'] == '012345678901'
    assert captured['min_pop'] == 9000
    assert captured['pattern_selection'].file_path == '/tmp/patterns.parquet'
    assert captured['pattern_selection'].month == '2021-01'
    assert captured['algorithm_config']['algorithm'] == 'czi_balanced'
    assert captured['include_trace'] is True
    assert captured['seed_cbgs'] == ['012345678901']


def test_cluster_cbgs_route_passes_seed_region_for_hierarchical_algorithm(client, app, monkeypatch):
    monkeypatch.setattr(
        'server_app.request_parsing.resolve_patterns_file_for_request',
        lambda seed_cbg, start_date_raw=None, use_test_data=False: ('/tmp/patterns.parquet', 'monthly', '2021-05'),
    )

    captured = {}

    def fake_start_cluster_job(cbg_str, min_pop, pattern_selection, algorithm_config, include_trace, seed_cbgs=None):
        captured['cbg_str'] = cbg_str
        captured['min_pop'] = min_pop
        captured['algorithm_config'] = algorithm_config
        captured['seed_cbgs'] = seed_cbgs
        return 29

    monkeypatch.setattr(app.config['analysis_service'], 'start_cluster_job', fake_start_cluster_job)

    response = client.post('/cluster-cbgs', json={
        'cbg': '240010001001',
        'seed_cbgs': ['240010001001', '240010002002', '240010002002'],
        'algorithm': 'hierarchical',
        'min_pop': 12000,
        'start_date': '2021-05-15',
    })

    assert response.status_code == 200
    assert response.get_json() == {'clustering_id': 29}
    assert captured['cbg_str'] == '240010001001'
    assert captured['seed_cbgs'] == ['240010001001', '240010002002']
    assert captured['algorithm_config']['algorithm'] == 'hierarchical_core_satellites'
    assert captured['algorithm_config']['effective_hierarchical_params']['local_radius_km'] == 20.0


def test_frontier_candidates_route_deduplicates_cbgs_and_parses_limit(client, app, monkeypatch):
    monkeypatch.setattr(
        'server_app.request_parsing.resolve_patterns_file_for_request',
        lambda seed_cbg, start_date_raw=None, use_test_data=False: ('/tmp/patterns.parquet', 'monthly', '2021-02'),
    )

    captured = {}

    def fake_compute_frontier_candidates(seed_cbg, normalized_cluster, min_pop, limit, pattern_selection, algorithm_config):
        captured['seed_cbg'] = seed_cbg
        captured['normalized_cluster'] = normalized_cluster
        captured['min_pop'] = min_pop
        captured['limit'] = limit
        captured['pattern_selection'] = pattern_selection
        captured['algorithm_config'] = algorithm_config
        return {
            'seed_cbg': seed_cbg,
            'cluster_size': len(normalized_cluster),
            'candidate_count': 1,
            'candidates': [{'cbg': '240010001001', 'rank': 1, 'selected': False}],
            'algorithm': algorithm_config['algorithm'],
            'patterns_file_used': pattern_selection.file_path,
            'patterns_source': pattern_selection.source,
            'patterns_month': pattern_selection.month,
            'missing_cluster_cbgs': [],
            'use_test_data': pattern_selection.use_test_data,
        }

    monkeypatch.setattr(app.config['analysis_service'], 'compute_frontier_candidates', fake_compute_frontier_candidates)

    response = client.post('/frontier-candidates', json={
        'seed_cbg': '240010001001',
        'cbg_list': ['240010001001', '240010001001', '240010002002'],
        'algorithm': 'weight',
        'min_pop': 5000,
        'limit': '3',
        'start_date': '2021-02-01T00:00:00Z',
    })

    body = response.get_json()
    assert response.status_code == 200
    assert body['candidate_count'] == 1
    assert captured['seed_cbg'] == '240010001001'
    assert captured['normalized_cluster'] == ['240010001001', '240010002002']
    assert captured['limit'] == 3
    assert captured['algorithm_config']['algorithm'] == 'greedy_weight'


def test_candidate_pois_route_normalizes_inputs_and_returns_service_payload(client, app, monkeypatch):
    monkeypatch.setattr(
        'server_app.request_parsing.resolve_patterns_file_for_request',
        lambda seed_cbg, start_date_raw=None, use_test_data=False: ('/tmp/patterns.parquet', 'monthly', '2021-03'),
    )

    captured = {}

    def fake_compute_candidate_pois(seed_cbg, candidate_cbg, normalized_cluster, limit, pattern_selection):
        captured['seed_cbg'] = seed_cbg
        captured['candidate_cbg'] = candidate_cbg
        captured['normalized_cluster'] = normalized_cluster
        captured['limit'] = limit
        return {
            'candidate_cbg': candidate_cbg,
            'cluster_size': len(normalized_cluster),
            'pois': [{'placekey': 'abc', 'rank': 1, 'cluster_flow': 10.0}],
            'patterns_file_used': pattern_selection.file_path,
            'patterns_source': pattern_selection.source,
            'patterns_analysis_mode': 'raw',
            'patterns_month': pattern_selection.month,
            'use_test_data': pattern_selection.use_test_data,
        }

    monkeypatch.setattr(app.config['analysis_service'], 'compute_candidate_pois', fake_compute_candidate_pois)

    response = client.post('/candidate-pois', json={
        'seed_cbg': '240010001001',
        'candidate_cbg': '240010002002',
        'cluster_cbgs': ['240010001001', '240010001001', '240010003003'],
        'limit': 5,
        'start_date': '2021-03-10',
    })

    body = response.get_json()
    assert response.status_code == 200
    assert body['candidate_cbg'] == '240010002002'
    assert body['pois'][0]['placekey'] == 'abc'
    assert captured['normalized_cluster'] == ['240010001001', '240010003003']
    assert captured['limit'] == 5


def test_second_order_destinations_route_deduplicates_seed_region_and_parses_limit(client, app, monkeypatch):
    monkeypatch.setattr(
        'server_app.request_parsing.resolve_patterns_file_for_request',
        lambda seed_cbg, start_date_raw=None, use_test_data=False: ('/tmp/patterns.parquet', 'monthly', '2021-06'),
    )

    captured = {}

    def fake_compute_second_order_destinations(seed_cbg, seed_cbgs, limit, pattern_selection):
        captured['seed_cbg'] = seed_cbg
        captured['seed_cbgs'] = seed_cbgs
        captured['limit'] = limit
        captured['pattern_selection'] = pattern_selection
        return {
            'seed_cbg': seed_cbg,
            'seed_cbgs': seed_cbgs,
            'destination_count': 1,
            'destinations': [{'unit_id': '74056', 'recommended': True}],
            'recommended_unit_ids': ['74056'],
            'patterns_file_used': pattern_selection.file_path,
            'patterns_source': pattern_selection.source,
            'patterns_month': pattern_selection.month,
            'use_test_data': pattern_selection.use_test_data,
        }

    monkeypatch.setattr(app.config['analysis_service'], 'compute_second_order_destinations', fake_compute_second_order_destinations)

    response = client.post('/second-order-destinations', json={
        'cbg': '240010001001',
        'seed_cbgs': ['240010001001', '240010001001', '240010002002'],
        'limit': '6',
        'start_date': '2021-06-01T00:00:00Z',
    })

    body = response.get_json()
    assert response.status_code == 200
    assert body['destination_count'] == 1
    assert body['recommended_unit_ids'] == ['74056']
    assert captured['seed_cbg'] == '240010001001'
    assert captured['seed_cbgs'] == ['240010001001', '240010002002']
    assert captured['limit'] == 6


def test_cz_metrics_route_returns_service_shape(client, app, monkeypatch):
    monkeypatch.setattr(
        'server_app.request_parsing.resolve_patterns_file_for_request',
        lambda seed_cbg, start_date_raw=None, use_test_data=False: ('/tmp/patterns.parquet', 'monthly', '2021-04'),
    )

    captured = {}

    def fake_compute_cz_metrics(seed_cbg, normalized_cbgs, pattern_selection):
        captured['seed_cbg'] = seed_cbg
        captured['normalized_cbgs'] = normalized_cbgs
        return {
            'movement_inside': 10.0,
            'movement_boundary': 2.0,
            'czi': 0.8333,
            'containment_ratio': 0.8333,
            'cbg_count': len(normalized_cbgs),
            'patterns_file_used': pattern_selection.file_path,
            'patterns_source': pattern_selection.source,
            'patterns_month': pattern_selection.month,
            'use_test_data': pattern_selection.use_test_data,
        }

    monkeypatch.setattr(app.config['analysis_service'], 'compute_cz_metrics', fake_compute_cz_metrics)

    response = client.post('/cz-metrics', json={
        'seed_cbg': '240010001001',
        'cbg_list': ['240010001001', '240010002002', '240010002002'],
        'start_date': '2021-04-20T12:00:00Z',
    })

    body = response.get_json()
    assert response.status_code == 200
    assert body['movement_inside'] == 10.0
    assert body['movement_boundary'] == 2.0
    assert body['czi'] == 0.8333
    assert captured['normalized_cbgs'] == ['240010001001', '240010002002']
