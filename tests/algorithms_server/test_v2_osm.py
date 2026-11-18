import pytest
from shapely.geometry import box

from v2 import osm
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
    assert missing == []
    calls = len(fake_fetch.calls)
    assert calls >= 1
    load_places(POLYGONS, fetch=fake_fetch)
    assert len(fake_fetch.calls) == calls  # second run is served from the tile cache


def test_load_home_points(fake_fetch):
    points = load_home_points(POLYGONS, fetch=fake_fetch)
    assert points == {CBG: [(-79.415, 39.405)], OTHER: [(-79.405, 39.406)]}
