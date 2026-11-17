"""Prepare ACS demographic inputs for an already-confirmed CBG selection."""
import json
import re

from cbg_demographics import B01001_SEX_AGE_BUCKETS
from . import CACHE_DIR
from .population import PopulationTotalsClient

SEX_AGE_BUCKETS = [
    (f"B01001_{int(bucket.field.split('e')[1]):03d}E", bucket)
    for bucket in B01001_SEX_AGE_BUCKETS
]
# Household table mapping adapted into Delineo; no prototype imports.
HOUSEHOLD_CODES = {
    "total_households": "B11001_001E",
    "family_households": "B11001_002E",
    "nonfamily_households": "B11001_007E",
    "total_family_households": "B11003_001E",
    "married_with_children": "B11003_003E",
    "single_father_with_children": "B11003_010E",
    "single_mother_with_children": "B11003_016E",
    "size_2": "B11016_003E",
    "size_3": "B11016_004E",
    "size_4": "B11016_005E",
    "size_5": "B11016_006E",
    "size_6": "B11016_007E",
    "size_7_plus": "B11016_008E",
    "nonfamily_size_1": "B11016_010E",
    "nonfamily_size_2": "B11016_011E",
    "nonfamily_size_3": "B11016_012E",
    "nonfamily_size_4": "B11016_013E",
    "nonfamily_size_5": "B11016_014E",
    "nonfamily_size_6": "B11016_015E",
    "nonfamily_size_7_plus": "B11016_016E",
}


# Local targets: only confirmed block groups contribute to the region totals.
RESIDENCE_CODES = {
    "household_population": "B25008_001E",
}
EMPLOYMENT_CODES = {
    "population_16_plus": "B23025_001E",
    "civilian_employed": "B23025_004E",
    "unemployed": "B23025_005E",
    "armed_forces": "B23025_006E",
    "not_in_labor_force": "B23025_007E",
}
ENROLLMENT_CODES = {
    "population_3_plus": "B14001_001E",
    "enrolled": "B14001_002E",
    "preschool": "B14001_003E",
    "kindergarten": "B14001_004E",
    "grades_1_to_4": "B14001_005E",
    "grades_5_to_8": "B14001_006E",
    "grades_9_to_12": "B14001_007E",
    "undergraduate": "B14001_008E",
    "graduate_or_professional": "B14001_009E",
    "not_enrolled": "B14001_010E",
}

COMMUTING_CODES = {
    "workers": "B08301_001E",
    "worked_from_home": "B08301_021E",
}
# County distributions guide age-specific assignment, never region totals.
# Keep ACS field IDs so denominators/subtotals remain distinct from leaf counts.
COUNTY_DISTRIBUTION_CODES = {
    "employment_by_sex_age": [f"B23001_{i:03d}E" for i in range(1, 174)],
    "enrollment_by_sex_age": [f"B14003_{i:03d}E" for i in range(1, 58)],
}


def table_rows(table, required, county):
    if (not isinstance(table, list) or not table or not isinstance(table[0], list)
            or not set(required) <= set(table[0])):
        raise ValueError(f'Invalid Census demographics response for county {county}.')
    rows = []
    for values in table[1:]:
        if not isinstance(values, list) or len(values) != len(table[0]):
            raise ValueError(f'Malformed Census demographics row for county {county}.')
        row = dict(zip(table[0], values))
        if row['state'] + row['county'] != county:
            raise ValueError(f'Census demographics geography mismatch for county {county}.')
        rows.append(row)
    return rows


def count(value, label):
    try:
        number = int(str(value))
    except (ValueError, TypeError):
        raise ValueError(f'Census count unavailable for {label}.') from None
    if number < 0:
        raise ValueError(f'Census count suppressed or unavailable for {label}.')
    return number


def complete_sum(values):
    """A complete total is unknown if any component is unknown."""
    values = list(values)
    return None if any(value is None for value in values) else sum(values)


def optional_count(value, location, field, code, missing_fields):
    reason = None
    if value is None:
        reason = 'not_available'
    else:
        try:
            number = int(str(value))
        except (ValueError, TypeError):
            raise ValueError(f'Malformed Census count for {location} {code}.') from None
        if number < 0:
            reason = 'suppressed'
        else:
            return number
    missing_fields.append({**location, 'field': field, 'census_variable': code, 'reason': reason})
    return None


class CensusInputsClient(PopulationTotalsClient):
    def load_counts(self, county, name, codes, selected=None, version=1):
        """Cache county batches; return only selected CBGs or one county row."""
        cache = CACHE_DIR / 'census' / f'acs5_{self.ACS_YEAR}'
        result = {}
        codes = list(dict.fromkeys(codes))
        geography = ({'for': 'block group:*',
                      'in': f'state:{county[:2]} county:{county[2:]} tract:*'}
                     if selected is not None else
                     {'for': f'county:{county[2:]}', 'in': f'state:{county[:2]}'})
        expected = selected if selected is not None else {county}
        for offset in range(0, len(codes), 50):
            batch = codes[offset:offset + 50]
            path = cache / f'{county}_{name}_v{version}_{offset // 50}.json'
            table = json.loads(path.read_text()) if path.exists() else self.fetch_table(
                {'get': ','.join(batch), **geography}, county)
            geo_fields = ['state', 'county'] + (['tract', 'block group'] if selected is not None else [])
            rows = table_rows(table, [*geo_fields, *batch], county)
            seen = set()
            for row in rows:
                geoid = ''.join(row[field] for field in geo_fields)
                if geoid not in expected:
                    continue
                if geoid in seen:
                    raise ValueError(f'Duplicate Census {name} row for {geoid}.')
                seen.add(geoid)
                result.setdefault(geoid, {}).update({
                    code: row[code] for code in batch
                })
            if seen != expected:
                raise ValueError(f'ACS {self.ACS_YEAR} {name} missing for: {sorted(expected - seen)[:10]}. Check geography compatibility.')
            if not path.exists():
                self.save_table(path, table)
        return result

    def prepare(self, cbg_ids, populations, progress_callback):
        wanted = set(cbg_ids)
        if not wanted or any(not re.fullmatch(r'[0-9]{12}', cbg) for cbg in wanted):
            raise ValueError('Expected twelve-digit CBG identifiers.')
        counties = sorted({cbg[:5] for cbg in wanted})
        cache = CACHE_DIR / 'census' / f'acs5_{self.ACS_YEAR}'
        blocks, county_distributions = {}, {}
        missing_fields = []
        for index, county in enumerate(counties):
            progress_callback(f'Loading Census demographics for county {county} ({index + 1}/{len(counties)})...',
                              5 + int(85 * index / len(counties)))
            selected = {cbg for cbg in wanted if cbg.startswith(county)}
            age_codes = [code for code, _ in SEX_AGE_BUCKETS]
            path = cache / f'{county}_sex_age_v1.json'
            table = json.loads(path.read_text()) if path.exists() else self.fetch_table({
                'get': ','.join(age_codes), 'for': 'block group:*',
                'in': f'state:{county[:2]} county:{county[2:]} tract:*',
            }, county)
            rows = table_rows(table, ['state', 'county', 'tract', 'block group', *age_codes], county)
            for row in rows:
                cbg = county + row['tract'] + row['block group']
                if cbg not in selected:
                    continue
                if cbg in blocks:
                    raise ValueError(f'Duplicate Census demographics row for {cbg}.')
                buckets = [
                    {'sex': bucket.sex, 'min_age': bucket.min_age, 'max_age': bucket.max_age,
                     'count': count(row[code], f'{cbg} {code}')}
                    for code, bucket in SEX_AGE_BUCKETS
                ]
                population = populations[cbg]
                if sum(bucket['count'] for bucket in buckets) != population:
                    raise ValueError(f'Age/sex counts disagree with preview population for {cbg}. Regenerate the preview.')
                blocks[cbg] = {'population': population, 'county_geoid': county, 'sex_age': buckets}
            missing = selected - blocks.keys()
            if missing:
                raise ValueError(f'ACS {self.ACS_YEAR} demographics missing for CBGs: {sorted(missing)[:10]}. Check geography compatibility.')
            if not path.exists():
                self.save_table(path, table)

            progress_callback(f'Loading household, employment and enrollment counts for county {county}...',
                              10 + int(85 * index / len(counties)))
            mappings = {'households': HOUSEHOLD_CODES, 'residence': RESIDENCE_CODES,
                        'employment': EMPLOYMENT_CODES, 'enrollment': ENROLLMENT_CODES,
                        'commuting': COMMUTING_CODES}
            codes = [code for mapping in mappings.values() for code in mapping.values()]
            local = self.load_counts(county, 'cbg_inputs', codes, selected, version=3)
            for cbg, row in local.items():
                for name, mapping in mappings.items():
                    blocks[cbg][name] = {
                        field: optional_count(row[code], {'cbg': cbg}, f'{name}.{field}', code, missing_fields)
                        for field, code in mapping.items()
                    }
                # B26001 can be unavailable at CBG level. Use the same-year
                # total minus residents in occupied housing units instead.
                residence = blocks[cbg]['residence']
                group_quarters = (None if residence['household_population'] is None else
                                  blocks[cbg]['population'] - residence['household_population'])
                if group_quarters is not None and group_quarters < 0:
                    raise ValueError(f'Household population exceeds total population for CBG {cbg}; check Census inputs.')
                residence['group_quarters_population'] = group_quarters
                # These are families with OWN children under 18, not all households with children.
                households = blocks[cbg]['households']
                households['families_with_own_children_under_18'] = complete_sum(households[name] for name in (
                    'married_with_children', 'single_father_with_children', 'single_mother_with_children'))

                for field, value in (
                    ('residence.group_quarters_population', group_quarters),
                    ('households.families_with_own_children_under_18', households['families_with_own_children_under_18']),
                ):
                    if value is None:
                        missing_fields.append({'cbg': cbg, 'field': field,
                                               'census_variable': None, 'reason': 'not_available'})

            county_distributions[county] = {}
            for name, codes in COUNTY_DISTRIBUTION_CODES.items():
                row = self.load_counts(county, name, codes)[county]
                county_distributions[county][name] = {
                    code: optional_count(row[code], {'county': county}, f'{name}.{code}', code, missing_fields)
                    for code in codes
                }
        return {'block_groups': blocks, 'distributions_by_county': county_distributions,
                'missing_fields': missing_fields,
                'sources': {'dataset': 'acs/acs5', 'year': self.ACS_YEAR,
                            'tables': ['B01003', 'B01001', 'B11001', 'B11003', 'B11016',
                                       'B25008', 'B23025', 'B14001', 'B08301',
                                       'B23001', 'B14003'],
                            'derived_fields': {'group_quarters_population': 'B01003_001E - B25008_001E'},
                            'county_distribution_tables': ['B23001', 'B14003']}}
