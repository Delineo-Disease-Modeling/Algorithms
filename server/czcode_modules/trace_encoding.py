"""Compact encoding of clustering trace steps for the /cluster-cbgs response.

Every trace step records the zone before and after one addition or removal,
so sending both lists in full makes the payload grow with steps x zone size
(tens of MB for a city-scale mobility_prune trace). A client that sends
``trace_encoding: 'delta'`` gets each list as a change against the list
before it instead:

    cluster_before_delta: {'added': [...], 'removed': [...]}
        against the previous step's cluster_after ([] for the first step)
    cluster_after_delta: {'added': [...], 'removed': [...]}
        against this step's cluster_before

Decoding drops the ``removed`` CBGs from the reference list, keeping its
order, then appends ``added``. A list that decoding would not reproduce
exactly, order included, is sent in full under its usual key, so the round
trip is lossless. The payload carries ``step_encoding: 'delta'`` when this
encoding was applied; without it every step has full lists as before.
"""

TRACE_ENCODING_DELTA = 'delta'

_LIST_KEYS = ('cluster_before', 'cluster_after')


def apply_cluster_delta(reference, delta):
    removed = set(delta['removed'])
    return [cbg for cbg in reference if cbg not in removed] + list(delta['added'])


def _cluster_delta(reference, target):
    reference_set = set(reference)
    target_set = set(target)
    delta = {
        'added': [cbg for cbg in target if cbg not in reference_set],
        'removed': [cbg for cbg in reference if cbg not in target_set],
    }
    if apply_cluster_delta(reference, delta) != list(target):
        return None
    return delta


def _encode_list(step_out, key, reference, target):
    delta = _cluster_delta(reference, target)
    if delta is None:
        step_out[key] = list(target)
    else:
        step_out[f'{key}_delta'] = delta


def encode_trace_steps(steps):
    encoded = []
    previous_after = []
    for step in steps:
        before = step.get('cluster_before')
        after = step.get('cluster_after')
        if before is None or after is None:
            encoded.append(step)
            previous_after = after if after is not None else []
            continue

        step_out = {key: value for key, value in step.items() if key not in _LIST_KEYS}
        _encode_list(step_out, 'cluster_before', previous_after, before)
        _encode_list(step_out, 'cluster_after', before, after)
        encoded.append(step_out)
        previous_after = after
    return encoded


def encode_trace_payload(trace_payload, trace_encoding):
    """Return the payload in the requested encoding.

    Unknown or missing encodings leave the payload untouched, which is also
    what a server without this module returns, so clients must handle both.
    """
    if trace_encoding != TRACE_ENCODING_DELTA or not isinstance(trace_payload, dict):
        return trace_payload
    encoded = dict(trace_payload)
    encoded['steps'] = encode_trace_steps(trace_payload.get('steps') or [])
    encoded['step_encoding'] = TRACE_ENCODING_DELTA
    return encoded
