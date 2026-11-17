import gzip
import json
from unittest.mock import Mock

import pandas as pd
import pytest

from v2 import lodes, prepare_lodes, service

CORE = '240010001001'
A = '420010001001'
B = '540010001001'
C = '510010001001'


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(lodes, 'LODES_DIR', tmp_path)
    monkeypatch.setattr(prepare_lodes, 'LODES_DIR', tmp_path)
    return tmp_path


def write_raw(cache, state, rows, part='main'):
    path = cache / 'raw' / f'{state}_od_{part}_JT01_2023.csv.gz'
    path.parent.mkdir(exist_ok=True)
    with gzip.open(path, 'wt') as output:
        output.write('h_geocode,w_geocode,S000,createdate\n')
        for home, work, jobs in rows:
            output.write(f'{home},{work},{jobs},20260101\n')


def seed(cache, state='md'):
    write_raw(cache, state, [(CORE + '001', A + '001', 60),
                             (CORE + '002', A + '002', 35),
                             (CORE + '001', B + '001', 5),
                             (CORE + '001', CORE + '002', 1000)])
    write_raw(cache, state, [(B + '001', CORE + '001', 95),
                             (C + '001', CORE + '001', 5)], part='aux')


def test_preparation_and_coverage(cache, monkeypatch):
    seed(cache)
    prepare_lodes.prepare({'md'})
    manifest = lodes.load_manifest()
    assert manifest['coverage_complete'] is False
    monkeypatch.setattr(lodes, 'load_manifest', Mock(side_effect=AssertionError('loaded twice')))
    cluster, coverage = lodes.expand_cbgs([CORE], manifest)
    assert cluster == sorted([CORE, A, B])
    assert coverage == {'target': .95, 'inbound_achieved': .95,
                        'outbound_achieved': 1.0, 'inbound_external_jobs': 100,
                        'outbound_external_jobs': 100}
    frame = pd.read_csv(cache / 'md.csv.gz')
    assert sorted(frame.jobs) == [5, 5, 95, 95, 1000]
    assert lodes.select_connected_cbgs({B: 10, A: 10}) == [A, B]
    _, empty = lodes.expand_cbgs(['010010001001'], manifest)
    assert empty['inbound_achieved'] is None
    assert empty['outbound_achieved'] is None


def test_resume_after_interruption_and_force(cache, monkeypatch):
    seed(cache, 'md')
    seed(cache, 'pa')
    original = prepare_lodes.prepare_state
    calls = []

    def interrupted(state):
        calls.append(state)
        if state == 'pa':
            raise RuntimeError('interrupted')
        return original(state)

    monkeypatch.setattr(prepare_lodes, 'prepare_state', interrupted)
    with pytest.raises(RuntimeError):
        prepare_lodes.prepare({'md', 'pa'})
    assert not (cache / 'manifest.json').exists()
    assert (cache / 'md.prepared.json').exists()

    def tracked(state):
        calls.append(state)
        return original(state)

    monkeypatch.setattr(prepare_lodes, 'prepare_state', tracked)
    calls.clear()
    prepare_lodes.prepare({'md', 'pa'})
    assert calls == ['pa']
    assert len(lodes.load_manifest()['sources']) == 4
    calls.clear()
    prepare_lodes.prepare({'md', 'pa'}, force=True)
    assert calls == ['md', 'pa']
    prepare_lodes.prepare({'md'})
    assert lodes.load_manifest()['states'] == ['md']


@pytest.mark.parametrize('damage', ['version', 'missing', 'size', 'json'])
def test_invalid_state_record_is_rebuilt(cache, monkeypatch, damage):
    seed(cache)
    prepare_lodes.prepare({'md'})
    marker = cache / 'md.prepared.json'
    if damage == 'version':
        record = json.loads(marker.read_text())
        record['preparation_version'] = 0
        marker.write_text(json.dumps(record))
    elif damage == 'missing':
        (cache / 'md.csv.gz').unlink()
    elif damage == 'size':
        (cache / 'md.csv.gz').write_bytes(b'bad')
    else:
        marker.write_text('{')
    wrapped = Mock(wraps=prepare_lodes.prepare_state)
    monkeypatch.setattr(prepare_lodes, 'prepare_state', wrapped)
    prepare_lodes.prepare({'md'})
    wrapped.assert_called_once_with('md')
    assert prepare_lodes.load_completed_state('md') is not None


def test_legacy_manifest_requires_rebuild(cache):
    (cache / 'md.csv.gz').touch()
    manifest = {**lodes.CACHE_SIGNATURE, 'states': ['md']}
    manifest.pop('preparation_version')
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Incompatible'):
        lodes.load_manifest()


@pytest.mark.parametrize('jobs', [-1, 1.5, 2**63, ''])
def test_invalid_source_counts_never_complete(cache, jobs):
    write_raw(cache, 'md', [(CORE + '001', A + '001', jobs)])
    with pytest.raises(ValueError):
        prepare_lodes.prepare({'md'})
    assert not (cache / 'md.prepared.json').exists()
    assert not (cache / 'manifest.json').exists()


def test_aggregate_overflow_never_completes(cache):
    write_raw(cache, 'md', [(CORE + '001', A + '001', 2**62),
                           (CORE + '002', A + '002', 2**62)])
    with pytest.raises(ValueError, match='capacity'):
        prepare_lodes.prepare({'md'})
    assert not (cache / 'md.prepared.json').exists()


def test_service_loads_manifest_once(cache, monkeypatch):
    seed(cache)
    prepare_lodes.prepare({'md'})
    load = Mock(wraps=lodes.load_manifest)
    monkeypatch.setattr(service, 'load_manifest', load)
    monkeypatch.setattr(service, 'resolve_core_cbgs', lambda _: ([CORE], [39, -77]))
    monkeypatch.setattr(service, 'load_selected_cbgs', lambda _: {})
    population = Mock(ACS_YEAR=2023)
    population.pull_cbg_population_totals.return_value = {CORE: 100, A: 200, B: 300}
    result = service.V2RegionService(None, population).build_region('21550', Mock())
    load.assert_called_once_with()
    assert result['size'] == 600
    assert result['cluster'] == sorted([CORE, A, B])
