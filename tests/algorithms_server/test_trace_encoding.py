import json
import random
from types import SimpleNamespace

import pytest

from czcode_modules.trace_encoding import (
    apply_cluster_delta,
    encode_trace_payload,
    encode_trace_steps,
)
from server_app import analysis_service as analysis_service_module
from server_app.analysis_service import PreviewClusteringService
from server_app.jobs import ProgressStore
from server_app.request_parsing import parse_cluster_algorithm_config


def decode_trace_steps(steps):
    """Reference decoder; Fullstack's expandTraceSteps implements the same rule."""
    decoded = []
    previous_after = []
    for step in steps:
        out = {
            key: value
            for key, value in step.items()
            if key not in ('cluster_before_delta', 'cluster_after_delta')
        }
        if 'cluster_before_delta' in step:
            out['cluster_before'] = apply_cluster_delta(previous_after, step['cluster_before_delta'])
        if 'cluster_after_delta' in step:
            out['cluster_after'] = apply_cluster_delta(out['cluster_before'], step['cluster_after_delta'])
        decoded.append(out)
        previous_after = out.get('cluster_after') or []
    return decoded


def _step(iteration, before, after, selected):
    return {
        'iteration': iteration,
        'cluster_before': list(before),
        'population_before': 10 * len(before),
        'candidates': [{'cbg': selected, 'score': 0.5, 'rank': 1, 'selected': True}],
        'selected_cbg': selected,
        'selected_population': 10,
        'cluster_after': list(after),
        'population_after': 10 * len(after),
        'metrics_after': {'stage': 'reverse_prune'},
    }


def _prune_trace(size):
    cluster = [f'cbg_{i:04d}' for i in range(size)]
    steps = []
    for iteration, cbg in enumerate(list(cluster[1:])):
        after = [c for c in cluster if c != cbg]
        steps.append(_step(iteration, cluster, after, cbg))
        cluster = after
    return steps


def test_prune_trace_round_trips_and_carries_only_the_removed_cbg():
    steps = _prune_trace(6)

    encoded = encode_trace_steps(steps)

    assert decode_trace_steps(encoded) == steps
    assert encoded[0]['cluster_before_delta'] == {
        'added': steps[0]['cluster_before'],
        'removed': [],
    }
    for original, step in zip(steps, encoded):
        assert 'cluster_before' not in step
        assert 'cluster_after' not in step
        assert step['cluster_after_delta'] == {'added': [], 'removed': [original['selected_cbg']]}
    for step in encoded[1:]:
        assert step['cluster_before_delta'] == {'added': [], 'removed': []}


def test_growth_trace_round_trips():
    cluster = ['seed']
    steps = []
    for iteration, cbg in enumerate(['b', 'a', 'c']):
        after = cluster + [cbg]
        steps.append(_step(iteration, cluster, after, cbg))
        cluster = after

    encoded = encode_trace_steps(steps)

    assert decode_trace_steps(encoded) == steps
    assert [step['cluster_after_delta']['added'] for step in encoded] == [['b'], ['a'], ['c']]


def test_list_a_delta_cannot_reproduce_is_sent_in_full():
    steps = [
        _step(0, ['a', 'b', 'c'], ['c', 'b', 'a'], 'x'),  # reordered
        _step(1, ['c', 'b', 'a'], ['c', 'a', 'a'], 'b'),  # duplicate entry
        _step(2, ['z'], ['z', 'y'], 'y'),  # not chained to the previous step
    ]

    encoded = encode_trace_steps(steps)

    assert decode_trace_steps(encoded) == steps
    assert encoded[0]['cluster_after'] == ['c', 'b', 'a']
    assert 'cluster_before_delta' in encoded[1]
    assert encoded[1]['cluster_after'] == ['c', 'a', 'a']
    assert encoded[2]['cluster_before_delta'] == {'added': ['z'], 'removed': ['c', 'a', 'a']}


def test_steps_without_cluster_lists_pass_through():
    steps = [{'iteration': 0, 'candidates': []}, _step(1, ['a', 'b'], ['a'], 'b')]

    encoded = encode_trace_steps(steps)

    assert encoded[0] is steps[0]
    assert decode_trace_steps(encoded) == steps


@pytest.mark.parametrize('trial', range(25))
def test_random_traces_round_trip(trial):
    rng = random.Random(trial)
    pool = [f'cbg_{i}' for i in range(30)]
    current = rng.sample(pool, rng.randint(0, 10))
    steps = []
    for iteration in range(rng.randint(1, 15)):
        if rng.random() < 0.15:
            before = rng.sample(pool, rng.randint(0, 10))  # break the chain
        else:
            before = list(current)
        after = list(before)
        for _ in range(rng.randint(0, 3)):
            if after and rng.random() < 0.5:
                after.pop(rng.randrange(len(after)))
            else:
                after.insert(rng.randint(0, len(after)), rng.choice(pool))
        steps.append(_step(iteration, before, after, rng.choice(pool)))
        current = after

    assert decode_trace_steps(encode_trace_steps(steps)) == steps


def test_payload_is_only_encoded_on_request():
    payload = {'algorithm': 'mobility_prune', 'steps': _prune_trace(4), 'note': 'n'}

    assert encode_trace_payload(payload, None) is payload
    assert encode_trace_payload(payload, 'something-else') is payload

    encoded = encode_trace_payload(payload, 'delta')
    assert encoded['step_encoding'] == 'delta'
    assert encoded['note'] == 'n'
    assert decode_trace_steps(encoded['steps']) == payload['steps']
    assert 'step_encoding' not in payload


def test_delta_encoding_shrinks_a_long_prune_trace():
    payload = {'steps': _prune_trace(400)}

    full = len(json.dumps(payload))
    compact = len(json.dumps(encode_trace_payload(payload, 'delta')))

    assert compact < full / 10


def _run_cluster_job(monkeypatch, trace_payload, trace_encoding, defer_trace=False, service=None):
    geojson_requests = []
    monkeypatch.setattr(
        'server_app.analysis_service.get_cbg_geojson',
        lambda cbgs, include_neighbors=False: geojson_requests.append(sorted(cbgs)) or {'features': []},
    )

    class _SyncThread:
        def __init__(self, target, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr('server_app.analysis_service.threading.Thread', _SyncThread)

    if service is None:
        service = PreviewClusteringService(ProgressStore(with_results=True, with_counter=True), resources=object())
    store = service.clustering_store
    service.cluster_cbgs = lambda *args, **kwargs: ({'a': 10, 'c': 10}, [36.5, -96.1], trace_payload)
    pattern_selection = SimpleNamespace(
        file_path='/tmp/patterns.parquet', month='2021-04', source='monthly', use_test_data=False,
    )

    cid = service.start_cluster_job(
        'a',
        5000,
        pattern_selection,
        parse_cluster_algorithm_config({'algorithm': 'mobility_prune'}),
        True,
        seed_cbgs=['a'],
        trace_encoding=trace_encoding,
        defer_trace=defer_trace,
    )
    result = store.get_result(cid)
    if defer_trace:
        return result, geojson_requests, service, cid
    return result, geojson_requests


def test_cluster_job_compacts_trace_after_building_trace_geojson(monkeypatch):
    steps = [
        _step(0, ['a', 'b', 'c', 'far'], ['a', 'c', 'far'], 'b'),
        _step(1, ['a', 'c', 'far'], ['a', 'c'], 'far'),
    ]
    trace_payload = {'algorithm': 'mobility_prune', 'steps': steps, 'algorithm_metadata': {'x': 1}}

    result, geojson_requests = _run_cluster_job(monkeypatch, trace_payload, 'delta')

    # Trace GeoJSON still covers every CBG named in the full per-step lists.
    assert geojson_requests[-1] == ['a', 'b', 'c', 'far']
    assert result['trace']['step_encoding'] == 'delta'
    assert decode_trace_steps(result['trace']['steps']) == steps
    assert result['algorithm_metadata'] == {'x': 1}


def test_cluster_job_keeps_full_trace_without_encoding(monkeypatch):
    steps = [_step(0, ['a', 'b', 'c'], ['a', 'c'], 'b')]

    result, _ = _run_cluster_job(monkeypatch, {'algorithm': 'mobility_prune', 'steps': steps}, None)

    assert 'step_encoding' not in result['trace']
    assert result['trace']['steps'] == steps


def test_deferred_trace_is_summarized_and_fetched_later(monkeypatch):
    steps = [
        _step(0, ['a', 'b', 'c', 'far'], ['a', 'c', 'far'], 'b'),
        _step(1, ['a', 'c', 'far'], ['a', 'c'], 'far'),
    ]
    trace_payload = {
        'algorithm': 'mobility_prune',
        'supports_stepwise': True,
        'note': 'n',
        'steps': steps,
        'algorithm_metadata': {'x': 1},
    }

    result, geojson_requests, service, cid = _run_cluster_job(
        monkeypatch, trace_payload, 'delta', defer_trace=True,
    )

    assert result['trace'] == {
        'algorithm': 'mobility_prune',
        'supports_stepwise': True,
        'note': 'n',
        'algorithm_metadata': {'x': 1},
        'step_count': 2,
        'deferred': True,
        'clustering_id': cid,
    }
    assert 'trace_geojson' not in result
    assert result['algorithm_metadata'] == {'x': 1}
    assert geojson_requests == [['a', 'c']]  # zone only; trace GeoJSON waits

    fetched = service.get_deferred_trace(cid, 'delta')
    assert fetched['trace']['step_encoding'] == 'delta'
    assert decode_trace_steps(fetched['trace']['steps']) == steps
    assert geojson_requests[-1] == ['a', 'b', 'c', 'far']
    assert service.get_deferred_trace(cid)['trace']['steps'] == steps
    assert len(geojson_requests) == 2  # trace GeoJSON built once, then reused


def test_deferred_traces_are_bounded(monkeypatch):
    monkeypatch.setattr(analysis_service_module, 'MAX_DEFERRED_TRACES', 2)
    service = PreviewClusteringService(ProgressStore(with_results=True, with_counter=True), resources=object())
    trace_payload = {'algorithm': 'mobility_prune', 'steps': [_step(0, ['a', 'b'], ['a'], 'b')]}

    cids = [
        _run_cluster_job(monkeypatch, trace_payload, None, defer_trace=True, service=service)[3]
        for _ in range(3)
    ]

    assert service.get_deferred_trace(cids[0]) is None
    assert service.get_deferred_trace(cids[1]) is not None
    assert service.get_deferred_trace(cids[2]) is not None
    assert service.get_deferred_trace(999) is None

