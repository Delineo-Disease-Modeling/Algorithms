from datetime import datetime

import pytest
from shapely.geometry import box

from v2 import osm
from v2.agents import build_town
from v2.mesa_model import run_patterns
from v2.papdata import to_papdata
from v2.places import classify_tags, load_home_points, load_places

CBG = '240230001001'
OTHER = '240230001002'
POLYGONS = {CBG: box(-79.42, 39.40, -79.41, 39.42), OTHER: box(-79.41, 39.40, -79.40, 39.42)}


def elements():
    return [
        {'type': 'node', 'id': 1, 'lat': 39.41, 'lon': -79.415, 'tags': {'amenity': 'school', 'name': 'Oak'}},
        {'type': 'node', 'id': 2, 'lat': 39.411, 'lon': -79.416, 'tags': {'shop': 'supermarket', 'name': 'Mart'}},
        {'type': 'way', 'id': 3, 'center': {'lat': 39.412, 'lon': -79.405}, 'tags': {'office': 'yes', 'name': 'Firm'}},
        {'type': 'node', 'id': 4, 'lat': 39.413, 'lon': -79.417, 'tags': {'amenity': 'pharmacy'}},
        {'type': 'node', 'id': 5, 'lat': 10.0, 'lon': 10.0, 'tags': {'shop': 'bakery', 'name': 'Far'}},
        {'type': 'node', 'id': 6, 'lat': 39.41, 'lon': -79.41, 'tags': {'amenity': 'bench'}},
    ]


@pytest.fixture
def fake_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(osm, 'OSM_DIR', tmp_path)
    calls = []

    def fetch(query, min_elements=0):
        calls.append(query)
        if 'building' in query:
            return {'elements': [{'type': 'way', 'id': 9, 'center': {'lat': 39.405, 'lon': -79.415}},
                                 {'type': 'way', 'id': 10, 'center': {'lat': 39.406, 'lon': -79.405}}]}
        return {'elements': elements()}

    fetch.calls = calls
    return fetch


def test_classify_tags():
    assert classify_tags({'amenity': 'kindergarten'}) == ('school', 'preschool')
    assert classify_tags({'amenity': 'university'}) == ('school', 'college')
    assert classify_tags({'shop': 'convenience'}) == ('grocery', None)
    assert classify_tags({'shop': 'clothes'}) == ('workplace', None)
    assert classify_tags({'amenity': 'bench'}) is None


def test_load_places_assigns_cbg_and_caches(fake_fetch):
    places, missing = load_places(POLYGONS, fetch=fake_fetch)
    assert {p['name']: p['cbg'] for p in places} == {'Oak': CBG, 'Mart': CBG, 'Firm': OTHER, 'pharmacy': CBG}
    assert [m['field'] for m in missing] == []
    calls = len(fake_fetch.calls)
    assert calls >= 1
    load_places(POLYGONS, fetch=fake_fetch)
    assert len(fake_fetch.calls) == calls  # second run is served from the tile cache


def test_load_home_points(fake_fetch):
    points = load_home_points(POLYGONS, fetch=fake_fetch)
    assert points == {CBG: [(-79.415, 39.405)], OTHER: [(-79.405, 39.406)]}


def census():
    def bucket(sex, low, high, count):
        return {'sex': sex, 'min_age': low, 'max_age': high, 'count': count}
    buckets = [bucket('M', 0, 17, 10), bucket('M', 18, 64, 10), bucket('F', 0, 17, 10), bucket('F', 18, 64, 10)]
    households = {'total_households': 14, 'family_households': 10, 'total_family_households': 10,
                  'families_with_own_children_under_18': 5, 'size_2': 4, 'size_3': 4, 'size_4': 2,
                  'nonfamily_size_1': 4}
    block = {'population': 40, 'county_geoid': '24023', 'sex_age': buckets, 'households': households,
             'residence': {'household_population': 36, 'group_quarters_population': 4},
             'employment': {'civilian_employed': 10}, 'enrollment': {'enrolled': 8}}
    return {'block_groups': {CBG: block}, 'distributions_by_county': {}}


def town(seed=42, flows=None):
    places, _ = load_places_stub()
    return build_town({CBG: POLYGONS[CBG]}, census(), places, {CBG: [(-79.415, 39.405)]},
                      flows if flows is not None else {CBG: {OTHER: 3, CBG: 1}}, seed)


def load_places_stub():
    def place(osm_id, name, function, cbg, level=None):
        return {'osm_id': osm_id, 'name': name, 'function': function, 'school_level': level,
                'latitude': 39.41, 'longitude': -79.415, 'cbg': cbg, 'street_address': None, 'postal_code': None}
    return [
        place('node/1', 'Oak', 'school', CBG, 'k12'), place('node/2', 'College', 'school', CBG, 'college'),
        place('node/3', 'Mart', 'grocery', CBG), place('node/4', 'Rx', 'pharmacy', CBG),
        place('node/5', 'Shop', 'workplace', CBG),
    ], []


def test_town_matches_census_totals():
    result = town()
    assert len(result.people) == 40
    assert sum(h.members for h in result.homes) == 40
    assert sum(1 for h in result.homes if h.household_type == 'group_quarters') == 4
    assert sum(p.employment_status == 'employed' for p in result.people) == 10
    assert result.summary['homes_from_osm'] == 1
    for person in result.people:
        if person.role == 'student':
            assert person.school_location_type in ('poi', 'out_of_zone')
        if person.work_location_type == 'poi':
            assert person.work_id is not None


def test_town_is_deterministic_and_seeded():
    assert town(1).people == town(1).people
    assert town(1).people != town(2).people


def test_out_of_region_commuters_become_out_of_zone():
    result = town(flows={CBG: {'999999999999': 1}})
    workers = [p for p in result.people if p.role == 'worker']
    assert workers and all(p.work_location_type == 'out_of_zone' for p in workers)
    papdata = to_papdata(result)
    assert papdata['places'][papdata['external_workplace_id']]['external_location_type'] == 'out_of_zone_work'


def test_mesa_patterns_are_full_snapshots_the_codec_accepts():
    pytest.importorskip('mesa')
    from patterns_codec import build_arrays_from_legacy
    papdata = to_papdata(town())
    patterns = run_patterns(papdata, datetime(2026, 10, 13), 48, seed=7)
    assert list(patterns)[0] == '60' and len(patterns) == 48
    matrix, minutes, pids, _, _ = build_arrays_from_legacy(patterns, papdata)  # raises if any cell is unset
    assert matrix.shape == (48, 40)
    worker = next(pid for pid, p in papdata['people'].items() if p['role'] == 'worker')
    placed = patterns['600']['places']  # Tuesday 10:00
    assert any(worker in members for members in placed.values())
    assert patterns == run_patterns(papdata, datetime(2026, 10, 13), 48, seed=7)
