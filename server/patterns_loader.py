"""
Shared Patterns Data Loader

Resolves state-specific monthly pattern files and loads them once for
shared use across all algorithm steps (CZ clustering, popgen, patterns gen).

File layout (parquet preferred, gzipped CSV and plain CSV also supported):
    data/patterns/{STATE}/{YYYY-MM}-{STATE}.{parquet,csv.gz,csv}

Column names in source files are UPPERCASE; this module normalizes them
to lowercase so downstream code works unchanged.
"""

import os
import logging
import pandas as pd
from typing import List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)

PATTERNS_BASE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'patterns')
_PATTERN_EXTS = ('.parquet', '.csv.gz', '.converted.csv', '.csv')

# All columns any algorithm step might need (lowercase canonical names).
# CZ clustering:  poi_cbg, visitor_daytime_cbgs, postal_code
# Popgen:         poi_cbg, placekey, location_name, top_category, latitude, longitude, postal_code, polygon_wkt
# Patterns gen:   placekey, median_dwell, popularity_by_hour, popularity_by_day
ALL_NEEDED_COLUMNS = [
    'poi_cbg', 'visitor_daytime_cbgs', 'postal_code',
    'placekey', 'location_name', 'top_category', 'latitude', 'longitude',
    'polygon_wkt',
    'median_dwell', 'popularity_by_hour', 'popularity_by_day',
]

# FIPS code -> state abbreviation (50 states + DC + territories)
FIPS_TO_STATE = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
    '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
    '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
    '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
    '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
    '56': 'WY', '72': 'PR', '78': 'VI',
}


def states_from_cbgs(cbgs: List[str]) -> List[str]:
    """
    Derive unique state abbreviations from a list of Census Block Group IDs.
    CBG format: SSCCCTTTTTTB  (first 2 digits = state FIPS).
    """
    fips_codes = set()
    for cbg in cbgs:
        cbg_str = str(cbg).strip()
        if len(cbg_str) >= 2:
            fips_codes.add(cbg_str[:2])
    states = []
    for fips in sorted(fips_codes):
        abbr = FIPS_TO_STATE.get(fips)
        if abbr:
            states.append(abbr)
        else:
            logger.warning(f"Unknown state FIPS code: {fips}")
    return states


def _available_months_for_state(state: str, base_dir: str) -> List[str]:
    """List available month keys (YYYY-MM) for a state directory, sorted."""
    import re
    state_dir = os.path.join(base_dir, state.upper())
    if not os.path.isdir(state_dir):
        return []
    pat = re.compile(r'^(\d{4}-\d{2})-[A-Z]{2}\.(?:parquet|csv(?:\.gz)?|converted\.csv)$', re.IGNORECASE)
    months = set()
    for f in os.listdir(state_dir):
        m = pat.match(f)
        if m:
            months.add(m.group(1))
    return sorted(months)


def closest_month(requested: str, available: List[str]) -> Optional[str]:
    """
    Return the available month closest to the requested month.
    If equidistant, prefers the earlier month.
    """
    if not available:
        return None
    if requested in available:
        return requested
    # Parse to comparable (year, month) tuples
    def _ym(mk):
        parts = mk.split('-')
        return int(parts[0]) * 12 + int(parts[1])
    req = _ym(requested)
    best = min(available, key=lambda m: (abs(_ym(m) - req), _ym(m)))
    return best


def resolve_patterns_files(states: List[str], start_date: datetime,
                           base_dir: str = None) -> List[str]:
    """
    Resolve patterns file path(s) for given states and date.

    Looks in: {base_dir}/{STATE}/{YYYY-MM}-{STATE}.parquet

    Args:
        states: State abbreviations (e.g. ['OK', 'TX'])
        start_date: Determines which monthly file to use
        base_dir: Override for data/patterns/ directory

    Returns:
        List of existing file paths (may be empty if none found)
    """
    if base_dir is None:
        base_dir = PATTERNS_BASE_DIR

    month_key = start_date.strftime('%Y-%m')
    files = []

    for state in states:
        state_upper = state.upper()
        found = False
        # Try exact month first across all supported extensions
        stem = f'{month_key}-{state_upper}'
        for ext in _PATTERN_EXTS:
            path = os.path.join(base_dir, state_upper, f'{stem}{ext}')
            if os.path.exists(path):
                files.append(path)
                found = True
                break
        if found:
            continue

        # Fall back to closest available month within the same state
        available = _available_months_for_state(state_upper, base_dir)
        nearest = closest_month(month_key, available)
        if nearest and nearest != month_key:
            nstem = f'{nearest}-{state_upper}'
            for ext in _PATTERN_EXTS:
                path = os.path.join(base_dir, state_upper, f'{nstem}{ext}')
                if os.path.exists(path):
                    logger.info(f"No exact match for {state_upper}/{month_key}, "
                                f"using closest available month: {nearest}")
                    files.append(path)
                    found = True
                    break

        if not found:
            logger.warning(f"No patterns file found for state={state_upper} month={month_key}")

    return files



class PatternsData:
    """
    Pre-loaded patterns data shared across algorithm steps.

    Loads the CSV once with only the columns needed, normalizes column names
    to lowercase, and provides filtered views for each consumer.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @classmethod
    def load(cls, file_paths: List[str],
             zip_codes: Optional[List[int]] = None,
             cbg_set: Optional[Set[str]] = None,
             chunksize: int = 20_000) -> 'PatternsData':
        """
        Load and merge patterns from one or more files.

        Supports parquet (preferred) and legacy CSV fallback.
        Filters rows early (by zip_codes or cbg_set) to keep memory low.

        Args:
            file_paths: Parquet or CSV files to read (may span multiple states)
            zip_codes: Filter by postal_code (used during CZ clustering)
            cbg_set: Filter by poi_cbg (used for popgen + patterns)
            chunksize: Rows per chunk for CSV streaming read
        """
        all_chunks: List[pd.DataFrame] = []

        for path in file_paths:
            if path.endswith('.parquet'):
                logger.info(f"Loading parquet patterns from {path}")
                df = pd.read_parquet(path)
                df.columns = df.columns.str.lower()
                if 'poi_cbg' in df.columns:
                    df['poi_cbg'] = df['poi_cbg'].astype(str).str.strip().str.zfill(12)
                if cbg_set and 'poi_cbg' in df.columns:
                    df = df[df['poi_cbg'].isin(cbg_set)]
                elif zip_codes and 'postal_code' in df.columns:
                    df = df[df['postal_code'].isin(zip_codes)]
                if not df.empty:
                    all_chunks.append(df)
            else:
                # Legacy CSV fallback
                logger.info(f"Loading CSV patterns from {path}")
                df = pd.read_csv(path)
                df.columns = df.columns.str.lower()
                if 'poi_cbg' in df.columns:
                    df['poi_cbg'] = pd.to_numeric(df['poi_cbg'], errors='coerce')
                    df.dropna(subset=['poi_cbg'], inplace=True)
                    df['poi_cbg'] = df['poi_cbg'].astype('int64').astype(str).str.zfill(12)
                if cbg_set and 'poi_cbg' in df.columns:
                    df = df[df['poi_cbg'].isin(cbg_set)]
                elif zip_codes and 'postal_code' in df.columns:
                    df = df[df['postal_code'].isin(zip_codes)]
                if not df.empty:
                    all_chunks.append(df)

        if all_chunks:
            df = pd.concat(all_chunks, ignore_index=True)
        else:
            df = pd.DataFrame(columns=[c.lower() for c in ALL_NEEDED_COLUMNS])

        logger.info(f"Loaded {len(df)} pattern rows total")
        return cls(df)

    @property
    def df(self) -> pd.DataFrame:
        """The full pre-filtered DataFrame."""
        return self._df

    def is_empty(self) -> bool:
        return len(self._df) == 0

    # -- Views for each algorithm step --

    def for_clustering(self) -> pd.DataFrame:
        """DataFrame for CZ clustering (poi_cbg, visitor_daytime_cbgs)."""
        cols = [c for c in ['poi_cbg', 'visitor_daytime_cbgs', 'postal_code']
                if c in self._df.columns]
        return self._df[cols]

    def get_placekeys_for_cbgs(self, cbg_set: Set[str]) -> List[str]:
        """Return unique placekeys whose poi_cbg is in cbg_set."""
        if 'poi_cbg' not in self._df.columns or 'placekey' not in self._df.columns:
            return []
        mask = self._df['poi_cbg'].isin(cbg_set)
        return self._df.loc[mask, 'placekey'].dropna().unique().tolist()

    def for_popgen_places(self, placekeys: List[str]) -> pd.DataFrame:
        """Place info for popgen: placekey, location_name, top_category, lat, lon, postal_code."""
        cols = [c for c in ['placekey', 'location_name', 'top_category',
                            'latitude', 'longitude', 'postal_code', 'polygon_wkt']
                if c in self._df.columns]
        if 'placekey' not in self._df.columns:
            return pd.DataFrame()
        mask = self._df['placekey'].isin(set(placekeys))
        return (self._df.loc[mask, cols]
                .drop_duplicates(subset=['placekey'])
                .reset_index(drop=True))

    def for_patterns_stats(self, placekeys: Set[str]) -> pd.DataFrame:
        """SafeGraph stats for patterns generation."""
        cols = [c for c in ['placekey', 'median_dwell',
                            'popularity_by_hour', 'popularity_by_day']
                if c in self._df.columns]
        if 'placekey' not in self._df.columns:
            return pd.DataFrame()
        mask = self._df['placekey'].isin(placekeys)
        return self._df.loc[mask, cols]
