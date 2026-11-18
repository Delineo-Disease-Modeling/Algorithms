"""Tiled, cached Overpass queries for a selected CBG region.

The region is covered by fixed 0.1-degree grid tiles. Each tile is fetched once
and cached under server/data/v2/osm/, so overlapping regions share downloads and
an interrupted run resumes where it stopped.
"""
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from shapely.geometry import box

from . import CACHE_DIR
from .overpass import OverpassError, fetch_overpass, validate_payload

TILE_DEGREES = 0.1
CACHE_VERSION = 1
OSM_DIR = CACHE_DIR / 'osm'

PLACE_AMENITIES = 'school|kindergarten|college|university|pharmacy|clinic|doctors|dentist|hospital'
HOME_BUILDINGS = ('house|residential|apartments|detached|semidetached_house|terrace|bungalow|'
                  'duplex|dormitory|farm|cabin|static_caravan')


def tile_bounds(i, j):
    """(south, west, north, east) of grid tile (i, j)."""
    return (round(j * TILE_DEGREES, 6), round(i * TILE_DEGREES, 6),
            round((j + 1) * TILE_DEGREES, 6), round((i + 1) * TILE_DEGREES, 6))


def tiles_for(geometries):
    """Sorted grid tiles that intersect at least one CBG polygon."""
    found = set()
    for geometry in geometries.values():
        minx, miny, maxx, maxy = geometry.bounds
        for i in range(math.floor(minx / TILE_DEGREES), math.floor(maxx / TILE_DEGREES) + 1):
            for j in range(math.floor(miny / TILE_DEGREES), math.floor(maxy / TILE_DEGREES) + 1):
                if (i, j) in found:
                    continue
                south, west, north, east = tile_bounds(i, j)
                if geometry.intersects(box(west, south, east, north)):
                    found.add((i, j))
    return sorted(found)


def places_query(bbox):
    south, west, north, east = bbox
    box_text = f'{south},{west},{north},{east}'
    return (
        '[out:json][timeout:180];('
        f'nwr["amenity"~"^({PLACE_AMENITIES})$"]({box_text});'
        f'nwr["shop"]({box_text});'
        f'nwr["office"]({box_text});'
        ');out center tags qt;'
    )


def homes_query(bbox):
    south, west, north, east = bbox
    return (
        '[out:json][timeout:180];'
        f'nwr["building"~"^({HOME_BUILDINGS})$"]({south},{west},{north},{east});'
        'out center ids qt;'
    )


def element_point(element):
    """(longitude, latitude) of a node or a way/relation center, else None."""
    lat, lon = element.get('lat'), element.get('lon')
    if lat is None or lon is None:
        center = element.get('center') or {}
        lat, lon = center.get('lat'), center.get('lon')
    if lat is None or lon is None:
        return None
    return float(lon), float(lat)


def _read_tile(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return validate_payload(json.loads(path.read_text(encoding='utf-8')))
    except (ValueError, OverpassError):
        return None


def _write_tile(path, payload):
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(payload, output)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_tiles(kind, query_for, geometries, progress_callback=None, fetch=fetch_overpass):
    """Deduplicated OSM elements for every tile touching the region.

    ``progress_callback(message, fraction)`` is optional; fraction runs 0..1.
    A tile that cannot be fetched raises OverpassError. It is never skipped,
    because a silent gap would look like a town without places.
    """
    tiles = tiles_for(geometries)
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    elements = {}
    for index, (i, j) in enumerate(tiles):
        if progress_callback:
            progress_callback(f'Loading OSM {kind} tile {index + 1}/{len(tiles)}...', index / max(len(tiles), 1))
        path = OSM_DIR / f'{kind}_v{CACHE_VERSION}_{TILE_DEGREES}_{i}_{j}.json'
        payload = _read_tile(path)
        if payload is None:
            payload = fetch(query_for(tile_bounds(i, j)), min_elements=0)
            _write_tile(path, {'elements': payload['elements']})
        for element in payload['elements']:
            elements[(element.get('type'), element.get('id'))] = element
    return list(elements.values())
