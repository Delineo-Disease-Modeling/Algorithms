import json
import os
import re
from functools import lru_cache

from common_geo import normalize_cbg
from czcode_modules.cache_service import DEFAULT_ALGORITHM_CACHE

from .constants import DATA_DIR


ZIP_TO_CBG_PATHS = (
    f"{DATA_DIR}/zip_to_cbg.json",
    os.path.join(os.path.dirname(__file__), 'bundled_seed_data', 'zip_to_cbg.json'),
)


def normalize_zip(value):
    text = str(value or '').strip()
    if not text:
        return None
    digits = ''.join(ch for ch in text if ch.isdigit())
    if len(digits) == 5:
        return digits
    return None


def normalize_seed_cbgs(values):
    normalized = []
    seen = set()
    for cbg in values or []:
        cbg_norm = normalize_cbg(cbg)
        if not cbg_norm or cbg_norm in seen:
            continue
        seen.add(cbg_norm)
        normalized.append(cbg_norm)
    return normalized


@lru_cache(maxsize=1)
def get_zip_to_cbgs_map():
    raw = None
    checked_paths = []
    for zip_to_cbg_path in ZIP_TO_CBG_PATHS:
        checked_paths.append(zip_to_cbg_path)
        try:
            with open(zip_to_cbg_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            break
        except FileNotFoundError:
            continue

    if raw is None:
        raise FileNotFoundError(
            "Required seed-region data file is missing. "
            f"Checked: {', '.join(checked_paths)}. "
            "Guided connected cities requires zip_to_cbg.json either in the mounted algorithms data directory "
            "or in the bundled fallback path."
        )

    normalized = {}
    for zip_code, cbgs in raw.items():
        zip_norm = normalize_zip(zip_code)
        if not zip_norm:
            continue
        cbg_list = normalize_seed_cbgs(cbgs)
        if cbg_list:
            normalized[zip_norm] = cbg_list
    return normalized


@lru_cache(maxsize=1)
def get_cbg_to_zip_map():
    inverse = {}
    for zip_code, cbgs in get_zip_to_cbgs_map().items():
        for cbg in cbgs:
            inverse.setdefault(cbg, zip_code)
    return inverse


def seed_cbgs_for_zip(zip_code):
    zip_norm = normalize_zip(zip_code)
    if not zip_norm:
        return []
    return list(get_zip_to_cbgs_map().get(zip_norm, ()))


def resolve_seed_region_for_zip(zip_code):
    zip_norm = normalize_zip(zip_code)
    if not zip_norm:
        return None

    seed_cbgs = seed_cbgs_for_zip(zip_norm)
    if not seed_cbgs:
        return None

    city_info = describe_city_approximation_for_zip(zip_norm) or {}
    label = _clean_location_text(city_info.get('label')) or f'ZIP {zip_norm}'
    city = _clean_location_text(city_info.get('city'))
    state = _clean_location_text(city_info.get('state'))

    return {
        'zip': zip_norm,
        'seed_cbgs': seed_cbgs,
        'seed_name': label,
        'city': city,
        'state': state.upper() if state else None,
        'unit_id': city_info.get('unit_id'),
        'unit_type': city_info.get('unit_type') or 'zip_fallback',
        'label': label,
    }


def _clean_location_text(value):
    text = str(value or '').strip()
    if not text:
        return None
    return re.sub(r'\s+', ' ', text)


def _slugify_location_text(value):
    slug = re.sub(r'[^a-z0-9]+', '-', str(value or '').strip().lower()).strip('-')
    return slug or 'unknown'


@lru_cache(maxsize=8192)
def describe_city_approximation_for_zip(zip_code):
    zip_norm = normalize_zip(zip_code)
    if not zip_norm:
        return None

    city = None
    state = None

    try:
        search = DEFAULT_ALGORITHM_CACHE.get_search_engine()
        zipcode = search.by_zipcode(zip_norm)
    except Exception:
        zipcode = None

    if zipcode is not None:
        city = (
            _clean_location_text(getattr(zipcode, 'major_city', None))
            or _clean_location_text(getattr(zipcode, 'post_office_city', None))
            or _clean_location_text(getattr(zipcode, 'city', None))
        )
        state = (
            _clean_location_text(getattr(zipcode, 'state', None))
            or _clean_location_text(getattr(zipcode, 'state_abbr', None))
        )

        if not city:
            common_cities = getattr(zipcode, 'common_city_list', None) or []
            for candidate in common_cities:
                city = _clean_location_text(candidate)
                if city:
                    break

    if city and state:
        unit_id = f"city:{state.upper()}:{_slugify_location_text(city)}"
        return {
            'unit_id': unit_id,
            'label': f'{city}, {state.upper()}',
            'city': city,
            'state': state.upper(),
            'zip_code': zip_norm,
            'unit_type': 'city_approximation',
        }

    return {
        'unit_id': f'zip:{zip_norm}',
        'label': f'ZIP {zip_norm}',
        'city': city,
        'state': state.upper() if state else None,
        'zip_code': zip_norm,
        'unit_type': 'zip_fallback',
    }
