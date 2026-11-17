"""ACS population totals for v2 previews, independent of legacy household code."""
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import requests

from . import CACHE_DIR


class PopulationTotalsClient:
    ACS_YEAR = 2023

    def __init__(self, api_key=None):
        self.api_key = (api_key or os.environ.get('CENSUS_API_KEY') or '').strip()

    def pull_cbg_population_totals(self, cbg_ids) -> dict[str, int]:
        """Fetch ACS five-year population totals for the v2 preview only.

        Query each county once, merge by full GEOID, and fail on missing/suppressed
        values instead of rendering an unknown population as zero.
        """
        wanted = set(cbg_ids)
        if any(not isinstance(cbg, str) or not re.fullmatch(r'[0-9]{12}', cbg) for cbg in wanted):
            raise ValueError('Expected twelve-digit CBG identifiers.')
        cache = CACHE_DIR / 'census' / f'acs5_{self.ACS_YEAR}'
        cache.mkdir(parents=True, exist_ok=True)
        result = {}
        for county_geoid in sorted({cbg[:5] for cbg in wanted}):
            path = cache / f'{county_geoid}_population.json'
            fetched = not path.exists()
            if fetched:
                rows = self.fetch_table({
                    'get': 'NAME,B01003_001E', 'for': 'block group:*',
                    'in': f'state:{county_geoid[:2]} county:{county_geoid[2:]} tract:*',
                }, county_geoid)
            else:
                rows = json.loads(path.read_text())
            required = {'state', 'county', 'tract', 'block group', 'B01003_001E'}
            if not isinstance(rows, list) or not rows or not isinstance(rows[0], list) or not required <= set(rows[0]):
                raise ValueError(f'Invalid Census response for county {county_geoid}.')
            county_values = {}
            for values in rows[1:]:
                if not isinstance(values, list) or len(values) != len(rows[0]):
                    raise ValueError('Malformed Census population row.')
                row = dict(zip(rows[0], values))
                geoid = ''.join(row[key] for key in ('state', 'county', 'tract', 'block group'))
                if not re.fullmatch(r'[0-9]{12}', geoid) or not geoid.startswith(county_geoid):
                    raise ValueError('Census response geography does not match the request.')
                if geoid in county_values:
                    raise ValueError(f'Duplicate Census population row for {geoid}.')
                try:
                    county_values[geoid] = int(row['B01003_001E'])
                except (ValueError, TypeError):
                    county_values[geoid] = None
            for cbg in sorted(cbg for cbg in wanted if cbg.startswith(county_geoid)):
                population = county_values.get(cbg)
                if population is None or population < 0:
                    raise ValueError(f'Population unavailable for CBG {cbg}; check ACS/geography compatibility.')
                result[cbg] = population
            if fetched:
                self.save_table(path, rows)
        return result

    def fetch_table(self, params, county_geoid):
        """Fetch one ACS five-year table without exposing the API key in errors."""
        if not self.api_key:
            raise ValueError(
                'Census population data requires an API key. Set CENSUS_API_KEY '
                'in the Algorithms server environment (Deploy/.env.local for local '
                'development), restart Algorithms, and retry. '
                'Request a key at https://api.census.gov/data/key_signup.html.'
            )
        url = f'https://api.census.gov/data/{self.ACS_YEAR}/acs/acs5'
        params = {**params, 'key': self.api_key}
        try:
            response = requests.get(url, params=params, timeout=(15, 90))
        except requests.RequestException:
            raise ValueError('Census population request failed; check network access.') from None
        if response.status_code == 204:
            raise ValueError(f'ACS {self.ACS_YEAR} has no data for county {county_geoid}; check geography compatibility.')
        if response.status_code != 200:
            raise ValueError(f'Census population request for county {county_geoid} returned HTTP {response.status_code}.')
        # Census can redirect authentication failures to HTML with HTTP 200.
        if urlsplit(response.url).path.endswith(('missing_key.html', 'invalid_key.html')):
            raise ValueError(
                'Census did not accept the API key. Check that CENSUS_API_KEY '
                'is valid and activated, then restart Algorithms and retry.'
            )
        try:
            rows = response.json()
        except ValueError:
            raise ValueError(
                f'Census returned a non-JSON response for county {county_geoid} '
                'despite HTTP 200. The API may be unavailable or rejecting the request; '
                'verify CENSUS_API_KEY and retry.'
            ) from None
        return rows

    @staticmethod
    def save_table(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                json.dump(rows, output)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
