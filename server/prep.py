import os
from typing import Any, Optional
from datetime import datetime

import pandas as pd

from monthly_patterns import (
  MonthlyPatternsManager,
  date_to_month_key,
  get_month_boundaries,
  get_months_in_range,
  get_simulation_minutes_for_month,
  parse_month_key,
)
from patterns import gen_patterns
from storage_api import load_papdata


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def resolve_patterns_folder(state_filter=None, explicit_patterns_folder=None):
  if explicit_patterns_folder:
    return explicit_patterns_folder

  if state_filter:
    state_folder = os.path.join(DATA_DIR, state_filter)
    if os.path.isdir(state_folder):
      return state_folder

  return DATA_DIR


def get_pattern_availability(state_filter, start_date=None, end_date=None, patterns_folder=None):
  folder_path = resolve_patterns_folder(state_filter, patterns_folder)
  manager = MonthlyPatternsManager(folder_path, state_filter)
  available_months = manager.available_months
  required_months = []

  if start_date and end_date:
    required_months = get_months_in_range(start_date, end_date)

  missing_months = [month for month in required_months if month not in manager.files]

  return {
    'state': state_filter,
    'patterns_folder': folder_path,
    'available_months': available_months,
    'required_months': required_months,
    'missing_months': missing_months,
    'has_any_data': len(available_months) > 0,
    'has_coverage': len(available_months) > 0 and len(missing_months) == 0,
  }


def load_papdata_for_preparation(czone_id=None, papdata=None):
  if papdata is not None:
    if not isinstance(papdata, dict) or 'people' not in papdata:
      raise ValueError("Provided papdata must be an object with at least a 'people' key")
    return papdata

  if czone_id is None:
    raise ValueError("Missing required preparation input: provide czone_id or papdata")

  return load_papdata(int(czone_id))


def _normalize_placekey(value: Any) -> Optional[str]:
  placekey = str(value or '').strip()
  return placekey or None


def _normalize_location_name(value: Any) -> Optional[str]:
  name = str(value or '').strip().lower()
  return name or None


def _normalize_coord(value: Any) -> Optional[float]:
  try:
    return round(float(value), 6)
  except (TypeError, ValueError):
    return None


def _location_signature(label: Any, latitude: Any, longitude: Any):
  name = _normalize_location_name(label)
  lat = _normalize_coord(latitude)
  lon = _normalize_coord(longitude)
  if not name or lat is None or lon is None:
    return None
  return (name, lat, lon)


def _has_street_address(place: dict[str, Any]) -> bool:
  return bool(str(place.get('street_address') or '').strip())


def _has_placekey(place: dict[str, Any]) -> bool:
  return bool(_normalize_placekey(place.get('placekey')))


def _needs_pattern_metadata(place: dict[str, Any]) -> bool:
  return not _has_street_address(place) or not _has_placekey(place)


def enrich_papdata_pattern_metadata(
  papdata: dict[str, Any],
  patterns_files: list[str],
) -> dict[str, int]:
  places = papdata.get('places')
  if not isinstance(places, dict) or not patterns_files:
    return {
      'street_addresses_backfilled': 0,
      'placekeys_backfilled': 0,
    }

  indexed_places: dict[str, dict[str, Any]] = {}
  placekey_index: dict[str, set[str]] = {}
  signature_index = {}

  for place_id_raw, place in places.items():
    if not isinstance(place, dict) or not _needs_pattern_metadata(place):
      continue

    place_id = str(place_id_raw)
    indexed_places[place_id] = place

    placekey = _normalize_placekey(place.get('placekey'))
    if placekey:
      placekey_index.setdefault(placekey, set()).add(place_id)

    signature = _location_signature(
      place.get('label'),
      place.get('latitude'),
      place.get('longitude'),
    )
    if signature:
      signature_index.setdefault(signature, set()).add(place_id)

  if not indexed_places:
    return {
      'street_addresses_backfilled': 0,
      'placekeys_backfilled': 0,
    }

  counts = {
    'street_addresses_backfilled': 0,
    'placekeys_backfilled': 0,
  }

  def _drop_from_index(index: dict, key, place_id: str) -> None:
    if key not in index:
      return
    index[key].discard(place_id)
    if not index[key]:
      del index[key]

  def assign_metadata(place_ids: set[str], placekey: Optional[str], street_address: str) -> None:
    for place_id in list(place_ids):
      place = indexed_places.get(place_id)
      if place is None:
        continue

      old_placekey = _normalize_placekey(place.get('placekey'))
      old_signature = _location_signature(
        place.get('label'),
        place.get('latitude'),
        place.get('longitude'),
      )

      if placekey and not _has_placekey(place):
        place['placekey'] = placekey
        counts['placekeys_backfilled'] += 1

      if street_address and not _has_street_address(place):
        place['street_address'] = street_address
        counts['street_addresses_backfilled'] += 1

      _drop_from_index(placekey_index, old_placekey, place_id)
      _drop_from_index(signature_index, old_signature, place_id)

      if not _needs_pattern_metadata(place):
        indexed_places.pop(place_id, None)
        continue

      new_placekey = _normalize_placekey(place.get('placekey'))
      if new_placekey:
        placekey_index.setdefault(new_placekey, set()).add(place_id)

      new_signature = _location_signature(
        place.get('label'),
        place.get('latitude'),
        place.get('longitude'),
      )
      if new_signature:
        signature_index.setdefault(new_signature, set()).add(place_id)

  for patterns_file in patterns_files:
    if not indexed_places:
      break

    for chunk in pd.read_csv(
      patterns_file,
      chunksize=10000,
      usecols=lambda c: c.lower() in ('placekey', 'location_name', 'latitude', 'longitude', 'street_address'),
    ):
      chunk = chunk.copy()
      chunk.columns = [str(col).strip().lower() for col in chunk.columns]
      if 'street_address' not in chunk.columns:
        continue

      for _, row in chunk.iterrows():
        street_address = str(row.get('street_address') or '').strip()
        row_placekey = _normalize_placekey(row.get('placekey'))

        if row_placekey and row_placekey in placekey_index:
          assign_metadata(set(placekey_index[row_placekey]), row_placekey, street_address)
          if not indexed_places:
            break

        signature = _location_signature(
          row.get('location_name'),
          row.get('latitude'),
          row.get('longitude'),
        )
        if signature and signature in signature_index:
          assign_metadata(set(signature_index[signature]), row_placekey, street_address)
          if not indexed_places:
            break

      if not indexed_places:
        break

  return counts


def prepare_multi_month_inputs(
  state_filter,
  start_date,
  end_date,
  *,
  czone_id=None,
  papdata=None,
  patterns_folder=None,
):
  availability = get_pattern_availability(
    state_filter,
    start_date=start_date,
    end_date=end_date,
    patterns_folder=patterns_folder,
  )

  if not availability['has_coverage']:
    missing = availability.get('missing_months', [])
    raise ValueError(f"Missing pattern files for months: {missing}")

  resolved_papdata = load_papdata_for_preparation(czone_id=czone_id, papdata=papdata)
  resolved_patterns_folder = availability['patterns_folder']
  manager = MonthlyPatternsManager(resolved_patterns_folder, state_filter)
  start_month = date_to_month_key(start_date)
  end_month = date_to_month_key(end_date)
  month_entries = list(manager.iter_months(start_month, end_month))
  backfill_counts = enrich_papdata_pattern_metadata(
    resolved_papdata,
    [patterns_csv_file for _, patterns_csv_file in month_entries],
  )
  prepared_months = []

  for month, patterns_csv_file in month_entries:
    length_minutes = get_simulation_minutes_for_month(month, start_date, end_date)
    duration_hours = length_minutes // 60
    year, month_num = parse_month_key(month)
    month_start_dt, month_end_dt = get_month_boundaries(year, month_num)
    segment_start_dt = max(
      month_start_dt,
      datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0),
    )
    segment_end_dt = min(
      month_end_dt,
      datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59),
    )

    prepared_months.append({
      'month': month,
      'patterns_csv_file': patterns_csv_file,
      'length_minutes': length_minutes,
      'duration_hours': duration_hours,
      'segment_start': segment_start_dt.isoformat(),
      'segment_end': segment_end_dt.isoformat(),
      'patterns_data': gen_patterns(
        resolved_papdata,
        start_time=segment_start_dt,
        duration=duration_hours,
        patterns_file=patterns_csv_file,
      ),
    })

  return {
    'state': state_filter,
    'patterns_folder': resolved_patterns_folder,
    'papdata': resolved_papdata,
    'months': prepared_months,
    'summary': {
      'start_month': start_month,
      'end_month': end_month,
      'total_months': len(prepared_months),
      'street_addresses_backfilled': backfill_counts['street_addresses_backfilled'],
      'placekeys_backfilled': backfill_counts['placekeys_backfilled'],
    },
  }
