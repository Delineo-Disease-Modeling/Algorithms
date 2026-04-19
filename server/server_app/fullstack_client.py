import json
from io import BytesIO

import requests


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
                'patterns': ('patterns.json', BytesIO(json.dumps(patterns).encode()), 'text/plain'),
            },
        )

    @staticmethod
    def response_detail(response, limit=500):
        try:
            return response.text[:limit]
        except Exception:
            return ''
