"""Producer-side binary patterns: round-trip + the upload-part flag switch."""
import numpy as np

from patterns_codec import (
    build_arrays_from_legacy,
    decode_patterns_binary,
    encode_patterns_binary,
)
from server_app.fullstack_client import FullstackClient

# Home id "1" and place id "1" collide -> locations keyed by (id, is_home).
PAPDATA = {
    "people": {"1": {}, "2": {}, "3": {}},
    "homes": {"1": {}, "2": {}},
    "places": {"1": {}},
}
PATTERNS = {
    "60": {"homes": {"1": ["1"], "2": ["2"]}, "places": {"1": ["3"]}},
    "120": {"homes": {"1": ["1", "2"]}, "places": {"1": ["3"]}},
}


def test_producer_binary_round_trips():
    M, ts, pids, loc_ids, n_homes = build_arrays_from_legacy(PATTERNS, PAPDATA)
    bp = decode_patterns_binary(encode_patterns_binary(M, ts, pids, loc_ids, n_homes))
    assert np.array_equal(bp.loc_matrix, M)
    # Producer enumerates persons sorted-by-int, so the reconstruction is
    # byte-identical to the source patterns (order included).
    assert dict(bp.items()) == PATTERNS


def test_patterns_part_defaults_to_binary(monkeypatch):
    monkeypatch.delenv("DELINEO_PATTERNS_BINARY", raising=False)
    name, body, ctype = FullstackClient._patterns_part(PAPDATA, PATTERNS)
    assert name == "patterns.bin"
    assert ctype == "application/octet-stream"
    # The emitted bytes must decode back to the same patterns.
    bp = decode_patterns_binary(body.getvalue())
    assert dict(bp.items()) == PATTERNS


def test_empty_env_value_still_defaults_to_binary(monkeypatch):
    # docker-compose passes ${DELINEO_PATTERNS_BINARY:-} (empty) by default.
    monkeypatch.setenv("DELINEO_PATTERNS_BINARY", "")
    name, _body, ctype = FullstackClient._patterns_part(PAPDATA, PATTERNS)
    assert name == "patterns.bin"


def test_patterns_part_falls_back_to_json_when_disabled(monkeypatch):
    monkeypatch.setenv("DELINEO_PATTERNS_BINARY", "0")
    name, _body, ctype = FullstackClient._patterns_part(PAPDATA, PATTERNS)
    assert name == "patterns.json"
    assert ctype == "text/plain"
