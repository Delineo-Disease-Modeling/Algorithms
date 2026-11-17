"""Pinned Census geometry, independent of the legacy 2016 shapefile loader."""
from functools import lru_cache

import geopandas as gpd

from . import CACHE_DIR, download_cached
from .lodes import STATE_ABBRS

# TIGER 2021 publishes the 2020 Census geography (TIGER 2020 ZCTAs are 2010).
TIGER_RELEASE = 2021
TIGER_ROOT = f'https://www2.census.gov/geo/tiger/TIGER{TIGER_RELEASE}'


@lru_cache(maxsize=1)
def _zctas():
    path = download_cached(
        f'{TIGER_ROOT}/ZCTA520/tl_2021_us_zcta520.zip',
        CACHE_DIR / 'geography' / 'tl_2021_us_zcta520.zip',
    )
    return gpd.read_file(path).to_crs('EPSG:4326')


def load_zcta(zip_code):
    frame = _zctas()
    selected = frame.loc[frame['ZCTA5CE20'] == zip_code].copy()
    if selected.empty:
        raise ValueError(f'ZIP {zip_code} has no matching 2020 Census ZCTA.')
    return selected


@lru_cache(maxsize=1)
def _states():
    path = download_cached(
        f'{TIGER_ROOT}/STATE/tl_2021_us_state.zip',
        CACHE_DIR / 'geography' / 'tl_2021_us_state.zip',
    )
    return gpd.read_file(path).to_crs('EPSG:4326')


def find_intersecting_states(zcta):
    polygon = zcta.geometry.union_all()
    states = _states()
    candidates = states.loc[states.geometry.intersects(polygon)]
    return sorted(candidates.loc[candidates.geometry.intersection(polygon).map(lambda geom: geom.area > 0), 'STATEFP'])


def neighboring_workplace_states(core_cbgs):
    """Select core states and directly adjoining states, including point touches."""
    state_codes = {cbg[:2] for cbg in core_cbgs}
    states = _states()
    states = states.loc[states['STUSPS'].str.lower().isin(STATE_ABBRS)]
    core_states = states.loc[states['STATEFP'].isin(state_codes)]
    if not state_codes or set(core_states['STATEFP']) != state_codes:
        raise ValueError('Commuting previews support the 50 US states and DC only.')
    boundary = core_states.geometry.union_all()
    neighbors = states.loc[states.geometry.intersects(boundary)]
    return sorted(neighbors['STUSPS'].str.lower())


def state_fips_for_states(abbreviations):
    states = _states()
    return set(states.loc[states['STUSPS'].str.lower().isin(abbreviations), 'STATEFP'])


@lru_cache(maxsize=8)
def load_state_cbgs(state_fips):
    name = f'tl_2021_{state_fips}_bg.zip'
    path = download_cached(f'{TIGER_ROOT}/BG/{name}', CACHE_DIR / 'geography' / name)
    frame = gpd.read_file(path).to_crs('EPSG:4326')
    frame['GEOID'] = frame['GEOID'].astype(str).str.zfill(12)
    return frame


def resolve_core_cbgs(zip_code):
    zcta = load_zcta(zip_code)
    polygon = zcta.geometry.union_all()
    core = []
    for state in find_intersecting_states(zcta):
        frame = load_state_cbgs(state)
        candidates = frame.loc[frame.geometry.intersects(polygon)]
        # Area is used only to distinguish polygons from boundary touches, not as a weight.
        core.extend(candidates.loc[candidates.geometry.intersection(polygon).map(lambda geom: geom.area > 0), 'GEOID'])
    if not core:
        raise ValueError(f'No CBGs overlap ZCTA {zip_code}.')
    point = polygon.representative_point()
    return sorted(set(core)), [point.y, point.x]


def load_selected_cbgs(cbg_ids):
    selected = {}
    for state in sorted({cbg[:2] for cbg in cbg_ids}):
        frame = load_state_cbgs(state)
        for row in frame.loc[frame['GEOID'].isin(cbg_ids)].itertuples():
            selected[row.GEOID] = row.geometry
    missing = set(cbg_ids) - selected.keys()
    if missing:
        raise ValueError(f'2020 geometry is missing for CBGs: {sorted(missing)[:10]}')
    return selected


def build_preview_geojson(geometries, populations, core_cbgs):
    core = set(core_cbgs)
    return {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': geometries[cbg].__geo_interface__, 'properties': {
            'GEOID': cbg, 'CensusBlockGroup': cbg, 'population': populations[cbg],
            'in_cluster': True, 'region_role': 'core' if cbg in core else 'expanded',
        }} for cbg in sorted(geometries)
    ]}
