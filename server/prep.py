import os
from datetime import datetime

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
  prepared_months = []

  for month, patterns_csv_file in manager.iter_months(start_month, end_month):
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
    },
  }
