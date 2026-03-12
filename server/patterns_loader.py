"""
Shared Patterns Data Loader

Resolves state-specific monthly pattern files and loads them once for
shared use across all algorithm steps (CZ clustering, popgen, patterns gen).

New file layout (compressed, preferred):
    data/patterns/{STATE}/{YYYY-MM}-{STATE}.csv.gz
Legacy/uncompressed fallback:
    data/patterns/{STATE}/{YYYY-MM}-{STATE}.csv

Column names in new files are UPPERCASE; this module normalizes them to
lowercase so downstream code works unchanged.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)

PATTERNS_BASE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'patterns')

# All columns any algorithm step might need (lowercase canonical names).
# CZ clustering:  poi_cbg, visitor_daytime_cbgs, postal_code
# Popgen:         poi_cbg, placekey, location_name, top_category, latitude, longitude, postal_code
# Patterns gen:   placekey, median_dwell, popularity_by_hour, popularity_by_day
ALL_NEEDED_COLUMNS = [
    'poi_cbg', 'visitor_daytime_cbgs', 'postal_code',
    'placekey', 'location_name', 'top_category', 'latitude', 'longitude',
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
    pat = re.compile(r'^(\d{4}-\d{2})-[A-Z]{2}\.csv(?:\.gz)?$', re.IGNORECASE)
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

    Prefers .csv.gz, falls back to .csv.
    Looks in: {base_dir}/{STATE}/{YYYY-MM}-{STATE}.csv.gz  (then .csv)

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
        # Try exact month first, then fall back to closest available
        months_to_try = [month_key]
        available = _available_months_for_state(state_upper, base_dir)
        nearest = closest_month(month_key, available)
        if nearest and nearest != month_key:
            months_to_try.append(nearest)
            logger.info(f"No exact match for {state_upper}/{month_key}, "
                        f"using closest available month: {nearest}")

        found = False
        for mk in months_to_try:
            stem = f'{mk}-{state_upper}'
            candidates = [
                os.path.join(base_dir, state_upper, f'{stem}.csv.gz'),
                os.path.join(base_dir, state_upper, f'{stem}.csv'),
                os.path.join(base_dir, f'{stem}.csv.gz'),
                os.path.join(base_dir, f'{stem}.csv'),
            ]
            for path in candidates:
                if os.path.exists(path):
                    files.append(path)
                    found = True
                    break
            if found:
                break
        if not found:
            logger.warning(f"No patterns file found for state={state_upper} month={month_key}")

    return files


def _detect_usecols(csv_path: str) -> List[str]:
    """
    Read the header of a CSV and return the actual column names that match
    our needed columns (case-insensitive).
    """
    sample = pd.read_csv(csv_path, nrows=0)
    header_map = {c.lower(): c for c in sample.columns}
    return [header_map[c] for c in ALL_NEEDED_COLUMNS if c in header_map]


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
        Load and merge patterns from one or more state CSV files.

        Filters rows early (by zip_codes or cbg_set) to keep memory low.
        Only reads the columns the algorithms actually need.

        Args:
            file_paths: CSV files to read (may span multiple states)
            zip_codes: Filter by postal_code (used during CZ clustering)
            cbg_set: Filter by poi_cbg (used for popgen + patterns)
            chunksize: Rows per chunk for streaming read
        """
        all_chunks: List[pd.DataFrame] = []

        for path in file_paths:
            usecols = _detect_usecols(path)
            if not usecols:
                logger.warning(f"No recognized columns in {path}, skipping")
                continue

            logger.info(f"Loading patterns from {path} ({len(usecols)} columns)")

            for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
                # Normalize column names to lowercase
                chunk.columns = chunk.columns.str.lower()

                # Normalize poi_cbg to string
                if 'poi_cbg' in chunk.columns:
                    chunk['poi_cbg'] = pd.to_numeric(chunk['poi_cbg'], errors='coerce')
                    chunk.dropna(subset=['poi_cbg'], inplace=True)
                    chunk['poi_cbg'] = chunk['poi_cbg'].astype('int64').astype(str)

                # Apply geographic filter
                if cbg_set and 'poi_cbg' in chunk.columns:
                    chunk = chunk[chunk['poi_cbg'].isin(cbg_set)]
                elif zip_codes and 'postal_code' in chunk.columns:
                    chunk = chunk[chunk['postal_code'].isin(zip_codes)]

                if len(chunk) > 0:
                    all_chunks.append(chunk)

        if all_chunks:
            df = pd.concat(all_chunks, ignore_index=True)
        else:
            df = pd.DataFrame(columns=[c.lower() for c in ALL_NEEDED_COLUMNS])

        logger.info(f"Loaded {len(df)} pattern rows total")
        return cls(df)

    @classmethod
    def from_legacy_csv(cls, csv_path: str,
                        zip_codes: Optional[List[int]] = None,
                        cbg_set: Optional[Set[str]] = None,
                        chunksize: int = 20_000) -> 'PatternsData':
        """
        Load from a single legacy patterns.csv (lowercase columns).
        Convenience wrapper for backwards compatibility.
        """
        return cls.load([csv_path], zip_codes=zip_codes, cbg_set=cbg_set,
                        chunksize=chunksize)

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
                            'latitude', 'longitude', 'postal_code']
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
