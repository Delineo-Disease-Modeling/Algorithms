import json
import os
from io import BytesIO

import requests


def _normalize_base_url(url):
  return str(url).rstrip('/')


def get_storage_api_base_urls():
  explicit = os.getenv('DELINEO_STORAGE_API_BASE_URL')
  if explicit:
    urls = [_normalize_base_url(item) for item in explicit.split(',') if item.strip()]
    if urls:
      return urls

  # Prefer the current Ryads-branch Fullstack API during the transition.
  candidates = [
    'http://localhost:1890',
    'http://localhost:3000/api',
  ]

  seen = set()
  ordered = []
  for candidate in candidates:
    normalized = _normalize_base_url(candidate)
    if normalized in seen:
      continue
    seen.add(normalized)
    ordered.append(normalized)
  return ordered


def _request_with_fallback(method, path, timeout=30, **kwargs):
  errors = []
  for base_url in get_storage_api_base_urls():
    url = f'{base_url}{path}'
    try:
      response = requests.request(method, url, timeout=timeout, **kwargs)
      if response.ok:
        return response
      errors.append(f'{url}: HTTP {response.status_code}')
      response.close()
    except requests.exceptions.RequestException as exc:
      errors.append(f'{url}: {exc}')

  raise RuntimeError(
    'Storage API unavailable. Tried: ' + ' | '.join(errors)
  )


def _post_json_with_fallback(path, payload, timeout=30):
  return _request_with_fallback('POST', path, json=payload, timeout=timeout)


def create_convenience_zone(payload, timeout=30):
  return _post_json_with_fallback('/convenience-zones', payload, timeout=timeout)


def upload_patterns(czone_id, papdata, patterns, timeout=120):
  errors = []
  papdata_bytes = json.dumps(papdata).encode()
  patterns_bytes = json.dumps(patterns).encode()

  for base_url in get_storage_api_base_urls():
    url = f'{base_url}/patterns'
    try:
      return requests.post(
        url,
        data={
          'czone_id': int(czone_id),
        },
        files={
          'papdata': ('papdata.json', BytesIO(papdata_bytes), 'text/plain'),
          'patterns': ('patterns.json', BytesIO(patterns_bytes), 'text/plain'),
        },
        timeout=timeout,
      )
    except requests.exceptions.RequestException as exc:
      errors.append(f'{url}: {exc}')

  raise RuntimeError(
    'Storage API unavailable. Tried: ' + ' | '.join(errors)
  )


def load_papdata(czone_id, timeout=60):
  response = _request_with_fallback(
    'GET',
    f'/patterns/{int(czone_id)}',
    timeout=timeout,
    stream=True,
  )

  try:
    for line in response.iter_lines(decode_unicode=True):
      if not line:
        continue

      payload = json.loads(line)
      papdata = payload.get('papdata', payload)
      if not isinstance(papdata, dict) or 'people' not in papdata:
        raise RuntimeError(f'Invalid papdata returned for czone {czone_id}')
      return papdata
  finally:
    response.close()

  raise RuntimeError(f'No papdata returned for czone {czone_id}')
