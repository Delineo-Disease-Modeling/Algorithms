"""Coordinate region previews and Census input preparation without legacy finalization."""
import logging
import threading

from . import GEOGRAPHY_YEAR, LODES_JOB_TYPE, LODES_RELEASE, LODES_YEAR
from .geography import (
    TIGER_RELEASE, build_preview_geojson, load_selected_cbgs,
    neighboring_workplace_states, resolve_core_cbgs, state_fips_for_states,
)
from .lodes import (
    COMMUTING_COVERAGE_TARGET, STATE_ABBRS, expand_cbgs, load_commute_flows, load_manifest,
)
from .prepare_lodes import prepare
from .census import CensusInputsClient, complete_sum
from .agents import build_town
from .papdata import to_papdata
from .places import load_home_points, load_places

LOGGER = logging.getLogger(__name__)
_PREPARATION_LOCK = threading.Lock()


def ensure_lodes_ready(required_states, progress_callback):
    """Extend the shared cache as needed, then return this preview's state scope."""
    required_states = set(required_states)
    while not _PREPARATION_LOCK.acquire(timeout=10):
        progress_callback('Waiting for commuting data preparation...', 10)
    try:
        try:
            manifest = load_manifest()
        except (ValueError, FileNotFoundError):
            manifest = None
        attempted = set(manifest.get('requested_states', manifest['states'])) if manifest else set()
        if manifest is None or not required_states <= attempted:
            progress_callback(
                'Preparing regional commuting data. This may take a while...', 10
            )
            prepare(
                attempted | required_states,
                progress_callback=lambda message: progress_callback(message, 10),
            )
            manifest = load_manifest()
        loaded = required_states & set(manifest['states'])
        if not loaded:
            raise ValueError('Commuting data is unavailable for this region.')
        return {
            **manifest,
            'states': sorted(loaded),
            'requested_states': sorted(required_states),
            'unavailable_states': sorted(required_states - loaded),
            'coverage_complete': loaded == set(STATE_ABBRS),
            'coverage_scope': 'Homes and workplaces within core states and directly adjoining states',
        }
    finally:
        _PREPARATION_LOCK.release()


class V2RegionService:
    def __init__(self, clustering_store, population_client, fullstack_client=None):
        self.clustering_store = clustering_store
        self.population_client = population_client
        self.fullstack_client = fullstack_client
        # Large per-job data (places, town, papdata, patterns). Kept out of the
        # progress store because its result is streamed to the browser. In memory
        # only, like the rest of the v2 jobs: cleared on server restart.
        self.artifacts = {}

    def _require(self, job_id, stage):
        result = self.clustering_store.get_result(job_id)
        if not result or result.get('stage') != stage:
            raise ValueError(f'Completed {stage} job {job_id} is unavailable. Run that step again.')
        return result

    def _artifact(self, job_id):
        artifact = self.artifacts.get(job_id)
        if artifact is None:
            raise ValueError(f'Data for job {job_id} is unavailable (server restarted?). Run that step again.')
        return artifact

    def _start_job(self, label, work):
        cid = self.clustering_store.next_id()

        def progress(message, pct):
            self.clustering_store.update(cid, message, pct)

        def run():
            try:
                work(cid, progress)
            except Exception as exc:
                LOGGER.exception('V2 %s job %s failed', label, cid)
                self.clustering_store.update(cid, str(exc), 0, error=True)

        threading.Thread(target=run, daemon=True).start()
        return cid

    def start_places_job(self, census_id):
        """OSM places and residential buildings for a confirmed census job's CBGs."""
        census = self._require(census_id, 'census_inputs')

        def work(cid, progress):
            progress('Loading region boundaries...', 2)
            geometries = load_selected_cbgs(census['cluster'])
            places, missing = load_places(geometries, lambda m, f: progress(m, 5 + int(45 * f)))
            home_points = load_home_points(geometries, lambda m, f: progress(m, 50 + int(45 * f)))
            self.artifacts[cid] = {'places': places}
            by_function = {}
            for place in places:
                by_function[place['function']] = by_function.get(place['function'], 0) + 1
            result = {
                'stage': 'places', 'census_id': census_id, 'preview_id': census['preview_id'],
                'seed': census['seed'], 'cluster': census['cluster'], 'missing_fields': missing,
                'summary': {
                    'place_count': len(places), 'places_by_function': by_function,
                    'cbgs_without_places': len(set(census['cluster']) - {p['cbg'] for p in places}),
                    'osm_home_points': sum(len(v) for v in home_points.values()),
                    'cbgs_without_osm_homes': len(set(census['cluster']) - home_points.keys()),
                },
            }
            self.clustering_store.update(cid, 'Places ready', 100, done=True, result=result)

        return self._start_job('places', work)

    def start_town_job(self, places_id, seed=None):
        """Build people, homes and assignments from census, OSM and LODES data."""
        places_result = self._require(places_id, 'places')
        census_result = self._require(places_result['census_id'], 'census_inputs')
        places = self._artifact(places_id)['places']
        seed = census_result['seed'] if seed is None else seed

        def work(cid, progress):
            cluster = census_result['cluster']
            progress('Loading region boundaries...', 2)
            geometries = load_selected_cbgs(cluster)
            home_points = load_home_points(geometries, lambda m, f: progress(m, 3 + int(12 * f)))
            progress('Loading commuting flows...', 15)
            flows = load_commute_flows(cluster, load_manifest(), lambda m, f: progress(m, 15 + int(15 * f)))
            town = build_town(geometries, census_result['census'], places, home_points, flows, seed,
                              lambda m, f: progress(m, 30 + int(65 * f)))
            expected = sum(b['population'] for b in census_result['census']['block_groups'].values())
            preview = self.clustering_store.get_result(census_result['preview_id']) or {}
            self.artifacts[cid] = {'town': town, 'center': preview.get('center'), 'cluster': cluster,
                                   'size': expected}
            result = {
                'stage': 'town', 'places_id': places_id, 'census_id': places_result['census_id'],
                'preview_id': census_result['preview_id'], 'seed': seed,
                'summary': {**town.summary, 'census_population': expected,
                            'population_matches_census': len(town.people) == expected,
                            'place_count': len(town.places)},
            }
            self.clustering_store.update(cid, 'Town ready', 100, done=True, result=result)

        return self._start_job('town', work)

    def start_patterns_job(self, town_id, start_date, length, seed=None, publish=None):
        """Run the Mesa model; optionally create the zone and upload to Fullstack."""
        town_result = self._require(town_id, 'town')
        artifact = self._artifact(town_id)
        if publish is not None and self.fullstack_client is None:
            raise ValueError('Fullstack client is not configured; cannot publish.')
        seed = town_result['seed'] if seed is None else seed

        def work(cid, progress):
            from .mesa_model import run_patterns  # Mesa is only needed for this step
            papdata = to_papdata(artifact['town'])
            patterns = run_patterns(
                papdata, start_date, length, seed,
                lambda done, total: progress(f'Simulating movement: hour {done}/{total}', 5 + int(75 * done / total)))
            self.artifacts[cid] = {'papdata': papdata, 'patterns': patterns}
            czone_id = None
            if publish is not None:
                progress('Creating convenience zone...', 85)
                center = artifact['center'] or [0, 0]
                response = self.fullstack_client.create_convenience_zone({
                    'name': publish['name'], 'description': publish.get('description', ''),
                    'latitude': center[0], 'longitude': center[1], 'cbg_list': artifact['cluster'],
                    'start_date': start_date.isoformat(), 'length': length, 'size': artifact['size'],
                    'user_id': publish.get('user_id'), 'guest_claim_token': publish.get('guest_claim_token'),
                })
                if not response.ok:
                    raise ValueError(f'Error creating zone: HTTP {response.status_code} '
                                     f'{self.fullstack_client.response_detail(response)}')
                czone_id = response.json()['data']['id']
                progress('Uploading papdata and patterns...', 92)
                response = self.fullstack_client.upload_patterns(czone_id, papdata, patterns)
                if not response.ok:
                    raise ValueError(f'Error uploading patterns: HTTP {response.status_code} '
                                     f'{self.fullstack_client.response_detail(response)}')
            result = {
                'stage': 'patterns', 'town_id': town_id, 'seed': seed, 'czone_id': czone_id,
                'summary': {'people': len(papdata['people']), 'homes': len(papdata['homes']),
                            'places': len(papdata['places']), 'timesteps': len(patterns),
                            'start_date': start_date.isoformat(), 'length': length},
            }
            self.clustering_store.update(cid, 'Patterns ready', 100, done=True, result=result)

        return self._start_job('patterns', work)

    def get_artifact(self, job_id, key):
        artifact = self.artifacts.get(job_id)
        return None if artifact is None else artifact.get(key)

    def start_population_job(self, preview_id, seed):
        preview = self.clustering_store.get_result(preview_id)
        if (not preview or preview.get('region_metadata', {}).get('method') != 'v2'
                or not preview.get('population_by_cbg')):
            raise ValueError('Completed region preview is unavailable. Generate the preview again before confirming.')
        cid = self.clustering_store.next_id()

        def run():
            try:
                client = CensusInputsClient(api_key=self.population_client.api_key)
                census = client.prepare(
                    preview['cluster'], preview['population_by_cbg'],
                    lambda message, pct: self.clustering_store.update(cid, message, pct),
                )
                result = {
                    'stage': 'census_inputs', 'preview_id': preview_id, 'seed': seed,
                    'cluster': preview['cluster'], 'census': census,
                    'summary': {
                        'cbg_count': len(census['block_groups']),
                        'county_count': len(census['distributions_by_county']),
                        'population': sum(block['population'] for block in census['block_groups'].values()),
                        'household_count': complete_sum(block['households']['total_households'] for block in census['block_groups'].values()),
                        'household_population': complete_sum(block['residence']['household_population'] for block in census['block_groups'].values()),
                        'group_quarters_population': complete_sum(block['residence']['group_quarters_population'] for block in census['block_groups'].values()),
                    },
                }
                message = 'Census inputs ready—with missing data' if census['missing_fields'] else 'Census inputs ready'
                self.clustering_store.update(cid, message, 100, done=True, result=result)
            except Exception as exc:
                LOGGER.exception('Census preparation failed for preview %s', preview_id)
                self.clustering_store.update(cid, str(exc), 0, error=True)

        threading.Thread(target=run, daemon=True).start()
        return cid

    def start_region_job(self, zip_code, coverage_target=COMMUTING_COVERAGE_TARGET):
        cid = self.clustering_store.next_id()
        def run():
            try:
                result = self.build_region(zip_code, lambda message, pct: self.clustering_store.update(cid, message, pct), coverage_target)
                self.clustering_store.update(cid, 'Done', 100, done=True, result=result)
            except Exception as exc:
                LOGGER.exception('V2 region preview failed for ZIP %s', zip_code)
                self.clustering_store.update(cid, str(exc), 0, error=True)
        threading.Thread(target=run, daemon=True).start()
        return cid

    def build_region(self, zip_code, progress_callback, coverage_target=COMMUTING_COVERAGE_TARGET):
        progress_callback('Resolving ZIP and Census block groups...', 5)
        core, center = resolve_core_cbgs(zip_code)
        required_states = neighboring_workplace_states(core)
        progress_callback(f'Regional workplace states: {", ".join(required_states).upper()}', 10)
        manifest = ensure_lodes_ready(required_states, progress_callback)
        cluster, coverage = expand_cbgs(
            core, manifest, progress_callback,
            allowed_state_fips=state_fips_for_states(required_states),
            coverage_target=coverage_target,
        )
        progress_callback('Loading expanded region boundaries...', 65)
        geometries = load_selected_cbgs(cluster)
        progress_callback('Loading Census population totals...', 80)
        populations = self.population_client.pull_cbg_population_totals(cluster)
        progress_callback('Building region map...', 95)
        return {
            'cluster': cluster, 'size': sum(populations.values()), 'center': center,
            'population_by_cbg': populations,
            'geojson': build_preview_geojson(geometries, populations, core),
            'algorithm': 'lodes_commuting',
            'region_metadata': {
                'method': 'v2', 'zip_code': zip_code, 'core_cbgs': core,
                'added_cbgs': sorted(set(cluster) - set(core)), 'coverage': coverage,
                'sources': {
                    'geography_vintage': GEOGRAPHY_YEAR, 'tiger_release': TIGER_RELEASE,
                    'lodes_release': LODES_RELEASE,
                    'lodes_year': LODES_YEAR, 'job_type': LODES_JOB_TYPE,
                    'population_year': self.population_client.ACS_YEAR,
                    'coverage_complete': manifest['coverage_complete'],
                    'workplace_states': manifest['states'], 'coverage_scope': manifest.get('coverage_scope'),
                    'requested_workplace_states': manifest['requested_states'],
                    'unavailable_workplace_states': manifest.get('unavailable_states', []),
                },
            },
        }
