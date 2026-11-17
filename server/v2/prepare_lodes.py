"""Run from Algorithms/server: python -m v2.prepare_lodes [--states md pa wv].

Default: request all 50 states and DC. Missing sources produce an explicitly incomplete cache.
Large raw downloads are retained so an interrupted preparation can be resumed.
"""
import argparse
import gzip
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from tempfile import NamedTemporaryFile
from pathlib import Path

import pandas as pd
import requests

from . import LODES_JOB_TYPE, LODES_RELEASE, LODES_YEAR, download_cached
from .lodes import CACHE_SIGNATURE, LODES_DIR, STATE_ABBRS


def report(message, progress_callback):
    print(message, flush=True)
    if progress_callback is not None:
        progress_callback(message)


def prepare_state(state, progress_callback=None):
    flows = Counter()
    sources = []
    total_jobs = 0
    for part in ('main', 'aux'):
        name = f'{state}_od_{part}_{LODES_JOB_TYPE}_{LODES_YEAR}.csv.gz'
        url = f'https://lehd.ces.census.gov/data/lodes/{LODES_RELEASE}/{state}/od/{name}'
        report(f'Loading {name}', progress_callback)
        path = download_cached(
            url, LODES_DIR / 'raw' / name,
            progress_callback=lambda message: report(message, progress_callback),
        )
        createdates = set()
        report(f'Processing {name}...', progress_callback)
        processed_rows = 0
        last_report = time.monotonic()
        for frame in pd.read_csv(path, dtype={'h_geocode': str, 'w_geocode': str, 'createdate': str},
                                 usecols=['h_geocode', 'w_geocode', 'S000', 'createdate'], chunksize=250_000):
            for column in ('h_geocode', 'w_geocode'):
                if not frame[column].str.fullmatch(r'\d{15}', na=False).all():
                    raise ValueError(f'Invalid block GEOID in {name}')
            jobs = pd.to_numeric(frame.S000, errors='raise')
            if jobs.isna().any() or (jobs < 0).any() or (jobs >= 2**63).any() or (jobs % 1 != 0).any():
                raise ValueError(f'Invalid job count in {name}')
            # Python integers prevent overflow before pandas groups int64 counts.
            total_jobs += sum(int(value) for value in jobs)
            if total_jobs >= 2**63:
                raise ValueError(f'Job totals exceed int64 capacity for {state}')
            frame['home_cbg'] = frame.h_geocode.str[:12]
            frame['work_cbg'] = frame.w_geocode.str[:12]
            frame['S000'] = jobs.astype('int64')
            flows.update(frame.groupby(['home_cbg', 'work_cbg']).S000.sum().to_dict())
            createdates.update(frame.createdate.dropna().unique())
            processed_rows += len(frame)
            if time.monotonic() - last_report >= 10:
                report(f'Processing {name}: {processed_rows:,} rows', progress_callback)
                last_report = time.monotonic()
        report(f'Recording source checksum for {name}...', progress_callback)
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        sources.append({'url': url, 'sha256': digest, 'createdates': sorted(createdates)})
    temporary = None
    report(f'Saving {state.upper()} commuting data...', progress_callback)
    last_report = time.monotonic()
    try:
        with NamedTemporaryFile(dir=LODES_DIR, suffix='.csv.gz', delete=False) as output:
            temporary = Path(output.name)
        with gzip.open(temporary, 'wt') as output:
            output.write('home_cbg,work_cbg,jobs\n')
            for (home, work), jobs in sorted(flows.items()):
                if jobs:
                    output.write(f'{home},{work},{int(jobs)}\n')
                if time.monotonic() - last_report >= 10:
                    report(f'Saving {state.upper()} commuting data...', progress_callback)
                    last_report = time.monotonic()
        temporary.replace(LODES_DIR / f'{state}.csv.gz')
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    report(f'Prepared {state.upper()}: {len(flows):,} CBG pairs', progress_callback)
    return sources


def load_completed_state(state):
    """Reuse application-managed aggregates; size is not a corruption checksum."""
    try:
        record = json.loads((LODES_DIR / f'{state}.prepared.json').read_text())
        aggregate_size = (LODES_DIR / f'{state}.csv.gz').stat().st_size
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if any(record.get(key) != value for key, value in CACHE_SIGNATURE.items()):
        return None
    if (record.get('state') != state or record.get('aggregate_bytes') != aggregate_size
            or not isinstance(record.get('sources'), list) or len(record['sources']) != 2):
        return None
    return record


def write_state_record(state, record):
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', dir=LODES_DIR, delete=False) as output:
            temporary = Path(output.name)
            json.dump(record, output, indent=2)
        temporary.replace(LODES_DIR / f'{state}.prepared.json')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(states, force=False, progress_callback=None):
    states = sorted(set(states))
    if not states or not set(states) <= set(STATE_ABBRS):
        raise ValueError('Provide at least one valid workplace state.')
    LODES_DIR.mkdir(parents=True, exist_ok=True)
    # The service serializes preparation; standalone CLI use requires a stopped server.
    (LODES_DIR / 'manifest.json').unlink(missing_ok=True)
    sources = []
    loaded_states = []
    unavailable_states = []
    for index, state in enumerate(states, start=1):
        report(f'Preparing commuting data: {state.upper()} ({index}/{len(states)})', progress_callback)
        record = None if force else load_completed_state(state)
        if record is None:
            (LODES_DIR / f'{state}.prepared.json').unlink(missing_ok=True)
            try:
                if progress_callback is None:
                    state_sources = prepare_state(state)
                else:
                    state_sources = prepare_state(state, progress_callback=progress_callback)
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                unavailable_states.append(state)
                report(
                    f'{state.upper()}: required {LODES_YEAR} commuting data is unavailable '
                    '(HTTP 404); continuing with incomplete source coverage.',
                    progress_callback,
                )
                continue
            record = {
                **CACHE_SIGNATURE, 'state': state,
                'aggregate_bytes': (LODES_DIR / f'{state}.csv.gz').stat().st_size,
                'sources': state_sources,
            }
            write_state_record(state, record)
        else:
            report(f'Reusing prepared {state.upper()}', progress_callback)
        sources.extend(record['sources'])
        loaded_states.append(state)
    if not loaded_states:
        raise ValueError(f'No complete {LODES_YEAR} LODES sources were available for the requested states.')
    manifest = {
        **CACHE_SIGNATURE, 'states': loaded_states,
        'requested_states': states, 'unavailable_states': unavailable_states,
        'coverage_complete': set(loaded_states) == set(STATE_ABBRS),
        'coverage_scope': '50 US states and District of Columbia workplaces; excludes territories',
        'prepared_at': datetime.now(timezone.utc).isoformat(), 'sources': sources,
    }
    (LODES_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    report(
        'Commuting data is ready.' if not unavailable_states else
        f'Commuting data is ready with missing workplace states: {", ".join(unavailable_states).upper()}.',
        progress_callback,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', nargs='+', choices=STATE_ABBRS, default=list(STATE_ABBRS),
                        help='Workplace states to load. Defaults to all 50 states and DC; unavailable sources are reported.')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild aggregates, reusing cached raw downloads.')
    args = parser.parse_args()
    prepare(set(args.states), force=args.force)


if __name__ == '__main__':
    main()
