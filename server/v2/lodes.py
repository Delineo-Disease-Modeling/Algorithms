"""One-hop region expansion from cached LODES primary-job OD flows."""
import json
from collections import Counter

import pandas as pd

from . import CACHE_DIR, GEOGRAPHY_YEAR, LODES_JOB_TYPE, LODES_RELEASE, LODES_YEAR

PREPARATION_VERSION = 1
CACHE_SIGNATURE = {
    'release': LODES_RELEASE, 'year': LODES_YEAR, 'job_type': LODES_JOB_TYPE,
    'geography_vintage': GEOGRAPHY_YEAR, 'preparation_version': PREPARATION_VERSION,
}
# Development target: restore 0.95 for normal regional coverage.
COMMUTING_COVERAGE_TARGET = 0.05
# Coverage claim is for the 50 states and DC, not territories.
STATE_ABBRS = tuple('al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy'.split())
LODES_DIR = CACHE_DIR / 'lodes' / f'{LODES_RELEASE}_{LODES_YEAR}_{LODES_JOB_TYPE}'


def load_manifest():
    path = LODES_DIR / 'manifest.json'
    if not path.exists():
        raise ValueError('LODES cache is not prepared. From Algorithms/server run python -m v2.prepare_lodes.')
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in CACHE_SIGNATURE.items()):
        raise ValueError('Incompatible v2 LODES cache; rerun python -m v2.prepare_lodes.')
    states = manifest.get('states', [])
    if not states or len(states) != len(set(states)) or not set(states) <= set(STATE_ABBRS):
        raise ValueError('Invalid states in v2 LODES manifest.')
    for state in states:
        if not (LODES_DIR / f'{state}.csv.gz').is_file():
            raise ValueError(f'Missing prepared LODES file for {state}.')
    manifest['coverage_complete'] = set(states) == set(STATE_ABBRS)
    return manifest


def calculate_core_connections(core_cbgs, manifest, progress_callback=None, *, allowed_state_fips=None):
    core = set(core_cbgs)
    inbound, outbound = Counter(), Counter()
    for index, state in enumerate(manifest['states']):
        if progress_callback:
            progress_callback(f'Reading commuting connections ({state.upper()})...', 20 + int(40 * index / len(manifest['states'])))
        for frame in pd.read_csv(LODES_DIR / f'{state}.csv.gz',
                                 usecols=['home_cbg', 'work_cbg', 'jobs'],
                                 dtype={'home_cbg': str, 'work_cbg': str, 'jobs': 'int64'},
                                 chunksize=250_000):
            if allowed_state_fips is not None:
                # Scope both endpoints before totals and 95% selections are computed.
                frame = frame.loc[
                    frame.home_cbg.str[:2].isin(allowed_state_fips)
                    & frame.work_cbg.str[:2].isin(allowed_state_fips)
                ]
            home_inside = frame.home_cbg.isin(core)
            work_inside = frame.work_cbg.isin(core)
            outbound.update(frame.loc[home_inside & ~work_inside].groupby('work_cbg').jobs.sum().to_dict())
            inbound.update(frame.loc[~home_inside & work_inside].groupby('home_cbg').jobs.sum().to_dict())
    return inbound, outbound


def load_commute_flows(cluster, manifest, progress_callback=None):
    """Primary-job flows for residents of the region: home_cbg -> {work_cbg: jobs}.

    Workplaces may be inside or outside the region; the caller decides which
    destinations are off-map. Every prepared workplace state is read because the
    prepared files are keyed by workplace state.
    """
    homes = set(cluster)
    flows = {}
    for index, state in enumerate(manifest['states']):
        if progress_callback:
            progress_callback(f'Reading commuting flows ({state.upper()})...', index / len(manifest['states']))
        for frame in pd.read_csv(LODES_DIR / f'{state}.csv.gz',
                                 usecols=['home_cbg', 'work_cbg', 'jobs'],
                                 dtype={'home_cbg': str, 'work_cbg': str, 'jobs': 'int64'},
                                 chunksize=250_000):
            frame = frame.loc[frame.home_cbg.isin(homes)]
            for (home, work), jobs in frame.groupby(['home_cbg', 'work_cbg']).jobs.sum().items():
                destinations = flows.setdefault(home, {})
                destinations[work] = destinations.get(work, 0) + int(jobs)
    return flows


def select_connected_cbgs(counts_by_cbg, coverage_target=COMMUTING_COVERAGE_TARGET):
    total = sum(counts_by_cbg.values())
    selected, cumulative = [], 0
    if total == 0 or coverage_target == 0:
        return selected
    for cbg, count in sorted(counts_by_cbg.items(), key=lambda item: (-item[1], item[0])):
        selected.append(cbg)
        cumulative += count
        if cumulative >= coverage_target * total:
            break
    return selected


def expand_cbgs(core_cbgs, manifest, progress_callback=None, *, allowed_state_fips=None,
                coverage_target=COMMUTING_COVERAGE_TARGET):
    inbound, outbound = calculate_core_connections(
        core_cbgs, manifest, progress_callback, allowed_state_fips=allowed_state_fips
    )
    added = set(select_connected_cbgs(inbound, coverage_target)) | set(select_connected_cbgs(outbound, coverage_target))
    cluster = sorted(set(core_cbgs) | added)
    def coverage(counts):
        total = sum(counts.values())
        return sum(counts.get(cbg, 0) for cbg in cluster) / total if total else None
    return cluster, {
        'target': coverage_target,
        'inbound_achieved': coverage(inbound), 'outbound_achieved': coverage(outbound),
        'inbound_external_jobs': int(sum(inbound.values())),
        'outbound_external_jobs': int(sum(outbound.values())),
    }
