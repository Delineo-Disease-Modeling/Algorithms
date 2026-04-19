import csv
import os
import re
from datetime import datetime

from common_geo import STATE_FIPS_TO_ABBR, normalize_cbg

from .constants import DATA_DIR, TEST_CLUSTER_COLUMNS, TEST_PATTERNS_FILE


def read_csv_headers(csv_path):
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader, [])
    return [h.strip() for h in headers if h is not None]


def validate_csv_columns(csv_path, required_columns):
    if not os.path.exists(csv_path):
        return False, list(required_columns), []
    headers = read_csv_headers(csv_path)
    missing = [c for c in required_columns if c not in headers]
    return len(missing) == 0, missing, headers


def get_test_seed_cbg(csv_path):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cbg = normalize_cbg(row.get('poi_cbg'))
            if cbg:
                return cbg
    return None


def extract_month_key(start_date_raw):
    if start_date_raw is None:
        return None
    raw = str(start_date_raw).strip()
    if not raw:
        return None

    match = re.match(r'^(\d{4})-(\d{2})', raw)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    try:
        parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        return parsed.strftime('%Y-%m')
    except ValueError:
        return None


def resolve_monthly_patterns_file(cbg_str, month_key):
    if not cbg_str or not month_key:
        return None

    state_fips = str(cbg_str)[:2]
    state_abbr = STATE_FIPS_TO_ABBR.get(state_fips)
    if not state_abbr:
        return None

    stem = f'{month_key}-{state_abbr}'
    state_dir = os.path.join(DATA_DIR, 'patterns', state_abbr)
    for ext in ('.parquet', '.csv.gz', '.converted.csv', '.csv'):
        path = os.path.join(state_dir, f'{stem}{ext}')
        if os.path.exists(path):
            return path

    return None


def list_available_months_for_state(cbg_str):
    state_fips = str(cbg_str)[:2] if cbg_str else ''
    state_abbr = STATE_FIPS_TO_ABBR.get(state_fips)
    if not state_abbr:
        return []

    pat = re.compile(r'^(\d{4}-\d{2})-[A-Z]{2}\.(?:parquet|csv(?:\.gz)?|converted\.csv)$', re.IGNORECASE)
    months = set()
    state_dir = os.path.join(DATA_DIR, 'patterns', state_abbr)
    if os.path.isdir(state_dir):
        for filename in os.listdir(state_dir):
            match = pat.match(filename)
            if match:
                months.add(match.group(1))
    return sorted(months)


def resolve_patterns_file_for_request(seed_cbg, start_date_raw=None, use_test_data=False):
    patterns_file = None
    patterns_source = 'monthly'
    patterns_month = None

    if use_test_data:
        ok, missing, _headers = validate_csv_columns(TEST_PATTERNS_FILE, TEST_CLUSTER_COLUMNS)
        if not ok:
            raise ValueError(
                f"TEST data is missing required columns for clustering: {', '.join(missing)}"
            )
        return TEST_PATTERNS_FILE, 'test', None

    requested_month = extract_month_key(start_date_raw)
    patterns_month = requested_month
    if requested_month:
        monthly_file = resolve_monthly_patterns_file(seed_cbg, requested_month)
        if monthly_file:
            patterns_file = monthly_file
            resolved_month = extract_month_key(os.path.basename(monthly_file))
            if resolved_month:
                patterns_month = resolved_month
        else:
            state_fips = str(seed_cbg)[:2]
            state_abbr = STATE_FIPS_TO_ABBR.get(state_fips)
            available = list_available_months_for_state(seed_cbg)
            if available:
                from patterns_loader import closest_month
                nearest = closest_month(requested_month, available)
                if nearest:
                    monthly_file = resolve_monthly_patterns_file(seed_cbg, nearest)
                    if monthly_file:
                        patterns_file = monthly_file
                        patterns_month = nearest
            if not patterns_file:
                if state_abbr:
                    raise ValueError(
                        f"No monthly patterns files found for state {state_abbr} (requested month '{requested_month}')."
                    )
                raise ValueError(
                    f"No monthly patterns file found for requested month '{requested_month}'."
                )

    if not patterns_file and not requested_month:
        state_fips = str(seed_cbg)[:2]
        state_abbr = STATE_FIPS_TO_ABBR.get(state_fips)
        available = list_available_months_for_state(seed_cbg)
        if available:
            latest_month = available[-1]
            monthly_file = resolve_monthly_patterns_file(seed_cbg, latest_month)
            if monthly_file:
                patterns_file = monthly_file
                patterns_month = latest_month
        if not patterns_file and state_abbr:
            raise ValueError(
                f"Missing start_date for patterns resolution, and no monthly patterns files were found for state {state_abbr}. "
                f"Expected files like '{DATA_DIR}/patterns/{state_abbr}/YYYY-MM-{state_abbr}.parquet'."
            )
        if not patterns_file:
            raise ValueError(
                "Missing start_date for patterns resolution, and no state-specific monthly patterns files are available."
            )

    if not patterns_file:
        raise ValueError(
            "No patterns file available for this request (monthly and default lookups failed)."
        )

    return patterns_file, patterns_source, patterns_month


def resolve_localized_patterns_extract(seed_cbg, patterns_file, month=None, cache_tag='v3'):
    if not seed_cbg or not patterns_file:
        return patterns_file, 'raw'

    source_csv = str(patterns_file)
    source_stem = os.path.splitext(os.path.basename(source_csv))[0]
    source_mtime = int(os.path.getmtime(source_csv)) if os.path.exists(source_csv) else 0
    month_part = str(month).strip() if month else 'nomonth'
    output_dir = os.path.join(DATA_DIR, '..', 'output')
    filename = f"{seed_cbg}_{month_part}_{source_stem}_{source_mtime}_{cache_tag}.csv"
    localized_path = os.path.join(output_dir, filename)
    if os.path.exists(localized_path):
        return localized_path, 'localized_cache'

    return patterns_file, 'raw'


def months_in_range(start_month, end_month):
    result = []
    year, month = int(start_month[:4]), int(start_month[5:7])
    end_year, end_month_value = int(end_month[:4]), int(end_month[5:7])
    while (year, month) <= (end_year, end_month_value):
        result.append(f'{year:04d}-{month:02d}')
        month += 1
        if month > 12:
            month = 1
            year += 1
    return result
