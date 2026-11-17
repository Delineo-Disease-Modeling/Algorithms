"""Coordinate region previews and Census input preparation without legacy finalization."""
import logging
import threading

from . import GEOGRAPHY_YEAR, LODES_JOB_TYPE, LODES_RELEASE, LODES_YEAR
from .geography import (
    TIGER_RELEASE, build_preview_geojson, load_selected_cbgs,
    neighboring_workplace_states, resolve_core_cbgs, state_fips_for_states,
)
from .lodes import COMMUTING_COVERAGE_TARGET, STATE_ABBRS, expand_cbgs, load_manifest
from .prepare_lodes import prepare
from .census import CensusInputsClient, complete_sum

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
    def __init__(self, clustering_store, population_client):
        self.clustering_store = clustering_store
        self.population_client = population_client

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
