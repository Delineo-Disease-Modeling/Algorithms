import json
import re
from datetime import datetime
from io import BytesIO

import folium
import geopandas as gpd
from flask import Flask, Response as FlaskResponse, jsonify, make_response, request, send_file
from flask_cors import cross_origin
from jsonschema import validate

from geojsongen import get_cbg_at_point, get_cbg_geojson
from run_report import RunReport
from schema import gen_cz_schema

from .algorithm_params import parse_czi_balanced_params, parse_czi_optimal_params, parse_seed_guard_params
from .analysis_helpers import compute_geojson_bounds
from .constants import (
    DATA_DIR,
    TEST_SIM_COLUMNS,
    VALID_CLUSTER_ALGORITHMS,
)
from .errors import ApiError
from .jobs import stream_events
from .logging_utils import log_candidate_pois
from .pattern_resolution import (
    extract_month_key,
    list_available_months_for_state,
    months_in_range,
    validate_csv_columns,
)
from .request_parsing import (
    normalize_cbg_list,
    optional_cbg,
    parse_cluster_algorithm_config,
    parse_json_payload,
    parse_test_seed_if_needed,
    require_cbg,
    resolve_pattern_selection,
)


def _api_error_response(error):
    return make_response(jsonify(error.to_payload()), error.status_code)


def register_routes(
    app: Flask,
    generation_service,
    analysis_service,
    generation_store,
    clustering_store,
):
    @app.route('/generate-cz', methods=['POST'])
    @cross_origin()
    def route_generate_cz():
        try:
            payload = parse_json_payload(request)

            cbg_value = payload.get('cbg')
            if cbg_value is None:
                raise ApiError("Missing required field: 'cbg'", status_code=400)
            cbg_str = str(cbg_value).strip()
            if not cbg_str.isdigit():
                raise ApiError(
                    "Invalid 'cbg': must be digits only (12-digit Census Block Group GEOID)",
                    status_code=400,
                )
            if len(cbg_str) == 11:
                cbg_str = cbg_str.zfill(12)
                payload['cbg'] = cbg_str
            if len(cbg_str) != 12 or not re.fullmatch(r"\d{12}", cbg_str):
                raise ApiError(
                    "Invalid 'cbg': expected exactly 12 digits (e.g., '060590117212')",
                    status_code=400,
                    extra={'cbg_received': str(cbg_value), 'cbg_normalized': cbg_str},
                )

            algorithm_config = parse_cluster_algorithm_config(payload)
            payload['algorithm'] = algorithm_config['algorithm']
            if algorithm_config['algorithm'] == 'czi_balanced':
                payload.update({k: v for k, v in algorithm_config['czi_params'].items() if v is not None})
            elif algorithm_config['algorithm'] == 'czi_optimal_cap':
                payload.update({k: v for k, v in algorithm_config['optimal_params'].items() if v is not None})
            elif algorithm_config['algorithm'] == 'greedy_weight_seed_guard':
                payload.update({k: v for k, v in algorithm_config['seed_guard_params'].items() if v is not None})

            try:
                validate(instance=payload, schema=gen_cz_schema)
            except Exception as exc:
                raise ApiError(
                    f'JSON data not valid: {str(exc)}',
                    status_code=400,
                    extra={'hint': "Check 'cbg' is a 12-digit Census Block Group GEOID (leading zeros matter)."},
                ) from exc

            payload = payload.copy()
            report = RunReport(
                run_type="cz_generation",
                name=f"CZ Generation: {payload.get('name', payload['cbg'])}",
                parameters={
                    'cbg': payload['cbg'],
                    'name': payload.get('name'),
                    'description': payload.get('description'),
                    'min_pop': payload.get('min_pop'),
                    'algorithm': payload.get('algorithm'),
                    'distance_penalty_weight': payload.get('distance_penalty_weight'),
                    'distance_scale_km': payload.get('distance_scale_km'),
                    'seed_guard_distance_km': payload.get('seed_guard_distance_km'),
                    'optimal_candidate_limit': payload.get('optimal_candidate_limit'),
                    'optimal_population_floor_ratio': payload.get('optimal_population_floor_ratio'),
                    'optimal_mip_rel_gap': payload.get('optimal_mip_rel_gap'),
                    'optimal_time_limit_sec': payload.get('optimal_time_limit_sec'),
                    'optimal_max_iters': payload.get('optimal_max_iters'),
                    'length': payload.get('length'),
                },
                user_id=payload.get('user_id'),
            )

            pattern_selection = resolve_pattern_selection(payload['cbg'], payload)
            if pattern_selection.use_test_data:
                ok, missing, _headers = validate_csv_columns(pattern_selection.file_path, TEST_SIM_COLUMNS)
                if not ok:
                    raise ApiError(
                        f"TEST data is missing required columns for simulation patterns: {', '.join(missing)}",
                        status_code=400,
                    )

            result = generation_service.create_cz(payload, report, pattern_selection, algorithm_config)
            if result['status_code'] != 200:
                return make_response(jsonify(result['payload']), result['status_code'])
            return json.dumps(result['payload'])
        except ApiError as error:
            return _api_error_response(error)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return make_response(jsonify({'message': str(exc)}), 500)

    @app.route('/cluster-cbgs', methods=['POST'])
    @cross_origin()
    def route_cluster_cbgs():
        try:
            payload = parse_json_payload(request)
            cbg_value = parse_test_seed_if_needed(payload, payload.get('cbg'))
            if cbg_value is None:
                raise ApiError("Missing required field: 'cbg'", status_code=400)
            cbg_str = require_cbg({'cbg': cbg_value}, 'cbg')
            algorithm_config = parse_cluster_algorithm_config(payload)
            pattern_selection = resolve_pattern_selection(cbg_str, payload)
            include_trace = bool(payload.get('include_trace', False))
            cid = analysis_service.start_cluster_job(
                cbg_str,
                payload.get('min_pop', 5000),
                pattern_selection,
                algorithm_config,
                include_trace,
            )
            return jsonify({'clustering_id': cid})
        except ApiError as error:
            return _api_error_response(error)

    @app.route('/finalize-cz', methods=['POST'])
    @cross_origin()
    def route_finalize_cz():
        try:
            payload = parse_json_payload(request)
            cbg_list = normalize_cbg_list(payload.get('cbg_list', []), 'cbg_list')
            seed_cbg = cbg_list[0]
            pattern_selection = resolve_pattern_selection(seed_cbg, payload)
            if pattern_selection.use_test_data:
                ok, missing, _headers = validate_csv_columns(pattern_selection.file_path, TEST_SIM_COLUMNS)
                if not ok:
                    raise ApiError(
                        f"TEST data is missing required columns for simulation patterns: {', '.join(missing)}",
                        status_code=400,
                        extra={'required_columns': TEST_SIM_COLUMNS, 'test_file': pattern_selection.file_path},
                    )

            payload = payload.copy()
            payload['cbg_list'] = cbg_list
            report = RunReport(
                run_type="cz_generation",
                name=f"CZ Generation: {payload.get('name', 'unnamed')}",
                parameters={
                    'cbg_list': cbg_list,
                    'name': payload.get('name'),
                    'description': payload.get('description'),
                    'length': payload.get('length'),
                },
                user_id=payload.get('user_id'),
            )
            result = generation_service.finalize_cz(payload, report, pattern_selection)
            return make_response(jsonify(result['payload']), result['status_code'])
        except ApiError as error:
            return _api_error_response(error)
        except Exception as exc:
            return make_response(jsonify({'message': str(exc)}), 500)

    @app.route('/generation-progress/<int:czone_id>', methods=['GET'])
    @cross_origin()
    def route_generation_progress(czone_id):
        return FlaskResponse(
            stream_events(generation_store, czone_id, interval_seconds=1.0),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            }
        )

    @app.route('/clustering-progress/<int:cid>', methods=['GET'])
    @cross_origin()
    def route_clustering_progress(cid):
        return FlaskResponse(
            stream_events(clustering_store, cid, interval_seconds=0.5, include_result_on_done=True),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            }
        )

    @app.route('/cz-metrics', methods=['POST'])
    @cross_origin()
    def route_cz_metrics():
        try:
            payload = parse_json_payload(request)
            seed_cbg = require_cbg(payload, 'seed_cbg')
            normalized_cbgs = normalize_cbg_list(payload.get('cbg_list', []), 'cbg_list')
            pattern_selection = resolve_pattern_selection(seed_cbg, payload)
            return jsonify(analysis_service.compute_cz_metrics(seed_cbg, normalized_cbgs, pattern_selection))
        except ApiError as error:
            return _api_error_response(error)
        except ValueError as exc:
            return make_response(jsonify({'message': str(exc)}), 400)
        except Exception as exc:
            return make_response(jsonify({'message': f'Error computing CZ metrics: {str(exc)}'}), 500)

    @app.route('/frontier-candidates', methods=['POST'])
    @cross_origin()
    def route_frontier_candidates():
        try:
            payload = parse_json_payload(request)
            seed_cbg = optional_cbg(payload, 'seed_cbg')
            normalized_cluster = normalize_cbg_list(payload.get('cbg_list', []), 'cbg_list')
            pattern_selection = resolve_pattern_selection(seed_cbg, payload)
            algorithm_config = parse_cluster_algorithm_config(payload)

            raw_limit = payload.get('limit')
            parsed_limit = None
            if raw_limit is not None:
                try:
                    parsed_limit = int(raw_limit)
                except (TypeError, ValueError) as exc:
                    raise ApiError("Invalid 'limit': expected integer", status_code=400) from exc
                if parsed_limit <= 0:
                    parsed_limit = None

            result = analysis_service.compute_frontier_candidates(
                seed_cbg,
                normalized_cluster,
                payload.get('min_pop', 0),
                parsed_limit,
                pattern_selection,
                algorithm_config,
            )
            return jsonify(result)
        except ApiError as error:
            return _api_error_response(error)
        except ValueError as exc:
            return make_response(jsonify({'message': str(exc)}), 400)
        except Exception as exc:
            return make_response(jsonify({'message': f'Error computing frontier candidates: {str(exc)}'}), 500)

    @app.route('/candidate-pois', methods=['POST'])
    @cross_origin()
    def route_candidate_pois():
        try:
            payload = parse_json_payload(request)
            seed_cbg = optional_cbg(payload, 'seed_cbg')
            candidate_cbg = optional_cbg(payload, 'candidate_cbg', required_message="Invalid or missing 'candidate_cbg'")
            normalized_cluster = normalize_cbg_list(payload.get('cluster_cbgs', []), 'cluster_cbgs')
            pattern_selection = resolve_pattern_selection(seed_cbg, payload)
            limit = payload.get('limit', 8)
            return jsonify(analysis_service.compute_candidate_pois(
                seed_cbg,
                candidate_cbg,
                normalized_cluster,
                limit,
                pattern_selection,
            ))
        except ApiError as error:
            return _api_error_response(error)
        except Exception as exc:
            log_candidate_pois('candidate-pois error=%s', exc, level=40, exc_info=True)
            return make_response(jsonify({'message': f'Error computing candidate POIs: {str(exc)}'}), 500)

    @app.route('/export-cz-map-html', methods=['POST'])
    @cross_origin()
    def route_export_cz_map_html():
        try:
            payload = parse_json_payload(request)
            cbg_list = normalize_cbg_list(payload.get('cbg_list', []), 'cbg_list')
            zone_name = str(payload.get('name') or 'cz-map').strip()

            geojson = get_cbg_geojson(cbg_list, include_neighbors=False)
            if not isinstance(geojson, dict) or not geojson.get('features'):
                raise ApiError('No map geometry found for the selected CBGs', status_code=400)

            bounds = compute_geojson_bounds(geojson)
            center = [bounds['center_lat'], bounds['center_lng']] if bounds else [39.3290708, -76.6219753]

            map_obj = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap')
            folium.GeoJson(
                geojson,
                style_function=lambda _: {
                    'fillColor': '#2563eb',
                    'color': '#1d4ed8',
                    'weight': 1.5,
                    'fillOpacity': 0.45,
                },
                highlight_function=lambda _: {
                    'fillColor': '#1d4ed8',
                    'color': '#1e3a8a',
                    'weight': 2,
                    'fillOpacity': 0.55,
                }
            ).add_to(map_obj)

            if bounds and bounds['min_lat'] != bounds['max_lat'] and bounds['min_lng'] != bounds['max_lng']:
                map_obj.fit_bounds([
                    [bounds['min_lat'], bounds['min_lng']],
                    [bounds['max_lat'], bounds['max_lng']]
                ])

            safe_name = re.sub(r'[^A-Za-z0-9._-]+', '-', zone_name).strip('-') or 'cz-map'
            timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            download_name = f'{safe_name}-{timestamp}.html'
            html_content = map_obj.get_root().render().encode('utf-8')
            html_buffer = BytesIO(html_content)
            html_buffer.seek(0)

            return send_file(
                html_buffer,
                mimetype='text/html',
                as_attachment=True,
                download_name=download_name
            )
        except ApiError as error:
            return _api_error_response(error)
        except Exception as exc:
            return make_response(jsonify({'message': f'Error generating CBG geometry: {str(exc)}'}), 500)

    @app.route('/cbg-geojson', methods=['GET'])
    @cross_origin()
    def route_cbg_geojson():
        cbgs_param = request.args.get('cbgs', '')
        include_neighbors = request.args.get('include_neighbors', 'false').lower() == 'true'

        if not cbgs_param:
            return make_response(jsonify({'message': 'Missing cbgs parameter'}), 400)

        cbg_list = [cbg.strip() for cbg in cbgs_param.split(',') if cbg.strip()]
        if not cbg_list:
            return make_response(jsonify({'message': 'No valid CBGs provided'}), 400)

        try:
            geojson = get_cbg_geojson(cbg_list, include_neighbors=include_neighbors)
            return jsonify(geojson)
        except Exception as exc:
            return make_response(jsonify({'message': f'Error generating GeoJSON: {str(exc)}'}), 500)

    @app.route('/cbg-at-point', methods=['GET'])
    @cross_origin()
    def route_cbg_at_point():
        latitude = request.args.get('latitude')
        longitude = request.args.get('longitude')
        state_fips = request.args.get('state_fips')

        if latitude is None or longitude is None:
            return make_response(jsonify({'message': 'Missing latitude/longitude parameters'}), 400)

        try:
            lat = float(latitude)
            lng = float(longitude)
        except (TypeError, ValueError):
            return make_response(jsonify({'message': 'Invalid latitude/longitude'}), 400)

        if lat < -90 or lat > 90 or lng < -180 or lng > 180:
            return make_response(jsonify({'message': 'Latitude/longitude out of range'}), 400)

        if state_fips:
            state_fips = str(state_fips).strip().zfill(2)

        try:
            result = get_cbg_at_point(lat, lng, state_fips=state_fips)
            if not result:
                return make_response(jsonify({'message': 'No CBG found at that location'}), 404)

            return jsonify({
                'cbg': result['GEOID'],
                'population': result['population'],
            })
        except Exception as exc:
            return make_response(jsonify({'message': f'Error resolving CBG at point: {str(exc)}'}), 500)

    @app.route('/pattern-availability', methods=['GET'])
    @cross_origin()
    def route_pattern_availability():
        state_abbr = (request.args.get('state') or '').strip().upper()
        start_date_raw = request.args.get('start_date')
        end_date_raw = request.args.get('end_date')

        if not state_abbr:
            return make_response(jsonify({'message': 'Missing state parameter'}), 400)
        if not start_date_raw or not end_date_raw:
            return make_response(jsonify({'message': 'Missing start_date or end_date parameter'}), 400)

        start_month = extract_month_key(start_date_raw)
        end_month = extract_month_key(end_date_raw)
        if not start_month or not end_month:
            return make_response(jsonify({'message': 'Invalid start_date or end_date format'}), 400)

        required_months = months_in_range(start_month, end_month)

        patterns_dir = f'{DATA_DIR}/patterns/{state_abbr}'
        available_months = []
        if re and patterns_dir:
            import os
            if os.path.isdir(patterns_dir):
                pat = re.compile(r'^(\d{4}-\d{2})-[A-Z]{2}\.parquet$', re.IGNORECASE)
                found = set()
                for filename in os.listdir(patterns_dir):
                    match = pat.match(filename)
                    if match:
                        found.add(match.group(1))
                available_months = sorted(found)

        has_any_data = len(available_months) > 0
        has_coverage = has_any_data
        missing_months = [] if has_coverage else required_months

        return jsonify({
            'data': {
                'state': state_abbr,
                'available_months': available_months,
                'required_months': required_months,
                'missing_months': missing_months,
                'has_any_data': has_any_data,
                'has_coverage': has_coverage,
            }
        })
