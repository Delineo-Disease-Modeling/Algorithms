import json
import os
from io import BytesIO

import requests


def _binary_patterns_enabled():
    """Binary (DLNOPAT) patterns are the DEFAULT. Set DELINEO_PATTERNS_BINARY=0
    (or false/no/off) as a kill-switch to fall back to legacy JSON.

    Safe to leave on once the Simulation consumer (DLNOPAT decoder + zstandard) is
    deployed — it auto-detects either format. See COLUMNAR_PATTERNS_DESIGN.md."""
    val = os.getenv('DELINEO_PATTERNS_BINARY')
    if val is None or val == '':
        return True
    return val.lower() in {'1', 'true', 'yes', 'on'}


class FullstackClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')

    def create_convenience_zone(self, payload):
        return requests.post(f'{self.base_url}/api/convenience-zones', json=payload)

    def upload_patterns(self, czone_id, papdata, patterns):
        return requests.post(
            f'{self.base_url}/api/patterns',
            data={'czone_id': int(czone_id)},
            files={
                'papdata': ('papdata.json', BytesIO(json.dumps(papdata).encode()), 'text/plain'),
                'patterns': self._patterns_part(papdata, patterns),
            },
        )

    @staticmethod
    def _patterns_part(papdata, patterns):
        """The multipart 'patterns' part: the compact binary (DLNOPAT) format by
        default, or legacy JSON when DELINEO_PATTERNS_BINARY is disabled. The
        Fullstack store sniffs the magic bytes, so the filename/content-type here
        are only advisory."""
        if _binary_patterns_enabled():
            # Lazy import so the JSON path needs neither numpy nor zstandard.
            from patterns_codec import build_arrays_from_legacy, encode_patterns_binary

            arrays = build_arrays_from_legacy(patterns, papdata)
            blob = encode_patterns_binary(*arrays)
            return ('patterns.bin', BytesIO(blob), 'application/octet-stream')
        return ('patterns.json', BytesIO(json.dumps(patterns).encode()), 'text/plain')

    @staticmethod
    def response_detail(response, limit=500):
        try:
            return response.text[:limit]
        except Exception:
            return ''
