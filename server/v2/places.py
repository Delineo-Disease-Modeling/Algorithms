"""OSM places (schools, groceries, health care, workplaces) and homes for a CBG region."""
import geopandas as gpd

from .osm import element_point, fetch_tiles, homes_query, places_query
from .overpass import fetch_overpass

_AMENITY_FUNCTIONS = {
    'school': ('school', 'k12'),
    'kindergarten': ('school', 'preschool'),
    'college': ('school', 'college'),
    'university': ('school', 'college'),
    'pharmacy': ('pharmacy', None),
    'clinic': ('clinic', None),
    'doctors': ('clinic', None),
    'dentist': ('clinic', None),
    'hospital': ('clinic', None),
}
_GROCERY_SHOPS = {'supermarket', 'convenience', 'greengrocer', 'grocery'}
REQUIRED_FUNCTIONS = ('school', 'grocery', 'workplace')


def _tag(tags, key):
    value = tags.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    text = None if value is None else str(value).strip()
    return text or None


def classify_tags(tags):
    """(function, school_level) for one OSM tag set, or None when it is not a place."""
    amenity, shop, office = _tag(tags, 'amenity'), _tag(tags, 'shop'), _tag(tags, 'office')
    if amenity in _AMENITY_FUNCTIONS:
        return _AMENITY_FUNCTIONS[amenity]
    if shop in _GROCERY_SHOPS:
        return 'grocery', None
    if shop or office:
        return 'workplace', None
    return None


def _cbg_frame(geometries):
    return gpd.GeoDataFrame({'cbg': list(geometries)}, geometry=list(geometries.values()), crs='EPSG:4326')


def _assign_cbg(rows, geometries):
    """Attach the containing selected CBG to each row; drop rows outside the region."""
    if not rows:
        return []
    frame = gpd.GeoDataFrame(
        rows, geometry=gpd.points_from_xy([r['longitude'] for r in rows], [r['latitude'] for r in rows]),
        crs='EPSG:4326')
    joined = gpd.sjoin(frame, _cbg_frame(geometries), how='inner', predicate='intersects')
    # A point exactly on a shared border matches two CBGs; keep the lowest GEOID.
    joined = joined.sort_values('cbg')
    joined = joined.loc[~joined.index.duplicated(keep='first')].sort_index()
    return [{**rows[index], 'cbg': cbg} for index, cbg in zip(joined.index, joined['cbg'])]


def load_places(geometries, progress_callback=None, fetch=fetch_overpass):
    """Classified OSM places inside the selected CBGs, sorted for stable ids.

    Returns (places, missing_fields). Each place dict has osm_id, name, function,
    school_level, latitude, longitude, cbg, street_address and postal_code.
    """
    rows, seen = [], set()
    for element in fetch_tiles('places', places_query, geometries, progress_callback, fetch):
        tags = element.get('tags') or {}
        classified = classify_tags(tags)
        point = element_point(element)
        if classified is None or point is None:
            continue
        function, level = classified
        name = _tag(tags, 'name') or function
        longitude, latitude = point
        key = (function, name.lower(), round(latitude, 4), round(longitude, 4))
        if name != function and key in seen:
            continue  # a node and a building outline for the same shop
        seen.add(key)
        street = ' '.join(part for part in (_tag(tags, 'addr:housenumber'), _tag(tags, 'addr:street')) if part)
        rows.append({
            'osm_id': f"{element['type']}/{element['id']}", 'name': name, 'function': function,
            'school_level': level or ('k12' if function == 'school' else None),
            'latitude': round(latitude, 6), 'longitude': round(longitude, 6),
            'street_address': street or None, 'postal_code': _tag(tags, 'addr:postcode'),
        })
    places = _assign_cbg(rows, geometries)
    places.sort(key=lambda p: (p['function'], p['name'], p['latitude'], p['longitude'], p['osm_id']))
    present = {place['function'] for place in places}
    missing = [{'field': f'places.{function}', 'reason': 'none_in_region'}
               for function in REQUIRED_FUNCTIONS if function not in present]
    return places, missing


def load_home_points(geometries, progress_callback=None, fetch=fetch_overpass):
    """cbg -> sorted [(longitude, latitude)] of OSM residential building centers."""
    rows, seen = [], set()
    for element in fetch_tiles('homes', homes_query, geometries, progress_callback, fetch):
        point = element_point(element)
        if point is None or (element['type'], element['id']) in seen:
            continue
        seen.add((element['type'], element['id']))
        rows.append({'longitude': round(point[0], 6), 'latitude': round(point[1], 6)})
    points = {}
    for row in _assign_cbg(rows, geometries):
        points.setdefault(row['cbg'], []).append((row['longitude'], row['latitude']))
    return {cbg: sorted(values) for cbg, values in points.items()}
