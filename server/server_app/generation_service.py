import json
import threading
from datetime import datetime

import geopandas as gpd

from czcode import Config, cbg_population, generate_cz, setup_logging
from geojsongen import get_cbg_geojson
from patterns import gen_patterns
from patterns_loader import PatternsData, resolve_patterns_files, states_from_cbgs
from popgen import gen_pop


class ConvenienceZoneGenerationService:
    def __init__(self, fullstack_client, generation_store):
        self.fullstack_client = fullstack_client
        self.generation_store = generation_store

    def gen_and_upload_data(self, geoids, czone_id, start_date, length, report, gdf=None,
                            patterns_file=None):
        try:
            self.generation_store.update(czone_id, 'Loading patterns data...', 5)
            shared_data = None
            cbg_set = set(geoids.keys())

            if patterns_file:
                shared_data = PatternsData.load([patterns_file], cbg_set=cbg_set)
            else:
                states = states_from_cbgs(list(cbg_set))
                if states:
                    resolved_files = resolve_patterns_files(states, start_date)
                    if resolved_files:
                        report.info(f'Auto-resolved patterns files: {resolved_files}')
                        shared_data = PatternsData.load(resolved_files, cbg_set=cbg_set)

            report.info(f'Patterns data loaded: {len(shared_data.df) if shared_data else 0} rows')
            self.generation_store.update(czone_id, 'Patterns data loaded', 20)

            report.info('Generating synthetic population (papdata)...')
            self.generation_store.update(czone_id, 'Generating synthetic population...', 25)
            papdata = gen_pop(geoids, gdf=gdf, shared_data=shared_data)
            people_count = len(papdata.get('people', {}))
            homes_count = len(papdata.get('homes', {}))
            places_count = len(papdata.get('places', {}))

            homes_with_coords = sum(1 for h in papdata.get('homes', {}).values()
                                    if h.get('latitude') is not None)
            if homes_with_coords > 0:
                report.info(f'Generated papdata: {people_count} people, {homes_count} homes ({homes_with_coords} with coordinates), {places_count} places')
            else:
                report.info(f'Generated papdata: {people_count} people, {homes_count} homes, {places_count} places')
            self.generation_store.update(czone_id, f'Population generated: {people_count} people, {places_count} places', 45)

            report.info('Generating movement patterns...')
            self.generation_store.update(czone_id, 'Generating movement patterns...', 50)
            patterns = gen_patterns(
                papdata,
                start_date,
                length,
                shared_data=shared_data
            )
            patterns_count = len(patterns)
            report.info(f'Generated {patterns_count} timestep patterns')
            self.generation_store.update(czone_id, f'Generated {patterns_count} movement patterns', 75)

            del shared_data

            report.info('Uploading data to DB API...')
            self.generation_store.update(czone_id, 'Uploading data...', 80)
            resp = self.fullstack_client.upload_patterns(czone_id, papdata, patterns)

            if resp.ok:
                report.info('Data uploaded successfully!')
                report.complete(summary={
                    'people_count': people_count,
                    'homes_count': homes_count,
                    'places_count': places_count,
                    'patterns_count': patterns_count,
                    'cbg_count': len(geoids),
                })
                self.generation_store.update(czone_id, 'Done', 100, done=True)
            else:
                report.fail(f'Error uploading data: HTTP {resp.status_code}')
                self.generation_store.update(czone_id, f'Upload failed: HTTP {resp.status_code}', 80, error=True)
        except Exception as exc:
            report.capture_exception()
            self.generation_store.update(czone_id, f'Error: {str(exc)}', 0, error=True)

    def create_cz(self, data, report, pattern_selection, algorithm_config):
        report.info(f"Generating CZ from CBG: {data['cbg']}")
        report.info(f"Target minimum population: {data['min_pop']}")

        seed_cbg = data['cbg']

        cz_start_date = None
        if data.get('start_date'):
            try:
                sd = data['start_date'].replace("Z", "+00:00")
                cz_start_date = datetime.fromisoformat(sd)
            except Exception:
                pass

        report.info(
            f"Using patterns source: {pattern_selection.source}"
            + (f" ({pattern_selection.month})" if pattern_selection.month else "")
        )

        geoids, map_obj, gdf = generate_cz(
            seed_cbg,
            data['min_pop'],
            patterns_file=pattern_selection.file_path,
            start_date=cz_start_date,
            algorithm=algorithm_config['algorithm'],
            distance_penalty_weight=algorithm_config['czi_params'].get('distance_penalty_weight'),
            distance_scale_km=algorithm_config['czi_params'].get('distance_scale_km'),
            optimal_candidate_limit=algorithm_config['optimal_params'].get('optimal_candidate_limit'),
            optimal_population_floor_ratio=algorithm_config['optimal_params'].get('optimal_population_floor_ratio'),
            optimal_mip_rel_gap=algorithm_config['optimal_params'].get('optimal_mip_rel_gap'),
            optimal_time_limit_sec=algorithm_config['optimal_params'].get('optimal_time_limit_sec'),
            optimal_max_iters=algorithm_config['optimal_params'].get('optimal_max_iters'),
            seed_guard_distance_km=algorithm_config['seed_guard_params'].get('seed_guard_distance_km')
        )

        cluster = list(geoids.keys())
        size = sum(list(geoids.values()))

        report.info(f'Clustered {len(cluster)} CBGs with total population {size}')
        report.debug(f'Cluster CBGs: {cluster}')

        resp = self.fullstack_client.create_convenience_zone({
            'name': data['name'],
            'description': data['description'],
            'latitude': map_obj.location[0],
            'longitude': map_obj.location[1],
            'cbg_list': cluster,
            'start_date': data['start_date'],
            'length': data['length'],
            'size': size,
            'user_id': data['user_id']
        })

        if not resp.ok:
            detail = self.fullstack_client.response_detail(resp)
            report.fail(f'Error creating CZ record: HTTP {resp.status_code}')
            return {
                'status_code': 500,
                'payload': {
                    'message': f'Error creating CZ record (HTTP {resp.status_code})',
                    'detail': detail
                }
            }

        czone_id = resp.json()['data']['id']
        report.info(f'Created CZ record with ID: {czone_id}')

        def background_generate():
            start_date = datetime.fromisoformat(data['start_date'].replace("Z", "+00:00"))
            self.gen_and_upload_data(
                geoids,
                czone_id,
                start_date,
                data.get('length', 168),
                report,
                gdf=gdf,
                patterns_file=pattern_selection.file_path
            )

        threading.Thread(target=background_generate, daemon=True).start()

        return {
            'status_code': 200,
            'payload': {
                'id': czone_id,
                'cluster': cluster,
                'size': size,
                'map': map_obj._repr_html_()
            }
        }

    def finalize_cz(self, data, report, pattern_selection):
        cbg_list = data['cbg_list']

        config = Config(cbg_list[0], 0, patterns_file=pattern_selection.file_path, month=pattern_selection.month)
        logger = setup_logging(config)
        geoids = {cbg: cbg_population(cbg, config, logger) for cbg in cbg_list}

        cluster = list(geoids.keys())
        size = sum(list(geoids.values()))

        report.info(f'Finalizing CZ with {len(cluster)} CBGs, total population {size}')
        report.info(
            f"Using patterns source: {pattern_selection.source}"
            + (f" ({pattern_selection.month})" if pattern_selection.month else "")
        )

        gdf = None
        try:
            geojson = get_cbg_geojson(cluster, include_neighbors=False)
            if isinstance(geojson, dict) and geojson.get('features'):
                gdf = gpd.GeoDataFrame.from_features(geojson['features'], crs='EPSG:4326')
                report.info(f'Prepared residential sampling geometry for {len(gdf)} CBGs')
            else:
                report.warning('No CBG geometry available for residential sampling during finalize-cz')
        except Exception as exc:
            report.warning(f'Could not prepare CBG geometry for residential sampling: {exc}')

        resp = self.fullstack_client.create_convenience_zone({
            'name': data.get('name', ''),
            'description': data.get('description', ''),
            'latitude': data.get('latitude', 0),
            'longitude': data.get('longitude', 0),
            'cbg_list': cluster,
            'start_date': data.get('start_date'),
            'length': data.get('length', 168),
            'size': size,
            'user_id': data.get('user_id')
        })

        if not resp.ok:
            report.fail(f'Error creating CZ record: HTTP {resp.status_code}')
            detail = self.fullstack_client.response_detail(resp)
            return {
                'status_code': 500,
                'payload': {
                    'message': f'Error creating CZ record (HTTP {resp.status_code})',
                    'detail': detail
                }
            }

        czone_id = resp.json()['data']['id']
        report.info(f'Created CZ record with ID: {czone_id}')

        try:
            start_date_raw = data.get('start_date')
            if not start_date_raw:
                raise ValueError("Missing required field: start_date")
            start_date = datetime.fromisoformat(start_date_raw.replace("Z", "+00:00"))
        except Exception as exc:
            report.fail(f"Invalid start_date for CZ finalize: {exc}")
            return {
                'status_code': 400,
                'payload': {'message': f'Invalid start_date: {exc}'}
            }

        def background_generate():
            self.gen_and_upload_data(
                geoids,
                czone_id,
                start_date,
                data.get('length', 168),
                report,
                gdf=gdf,
                patterns_file=pattern_selection.file_path
            )

        self.generation_store.update(czone_id, 'Starting generation...', 0)
        threading.Thread(target=background_generate, daemon=True).start()

        return {
            'status_code': 200,
            'payload': {
                'id': czone_id,
                'cluster': cluster,
                'size': size
            }
        }
