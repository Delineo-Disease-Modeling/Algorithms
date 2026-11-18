"""Agent schema and the synthetic town builder for the v2 Mesa model.

Inputs are the confirmed Census block-group records, OSM places and home points,
and LODES commuting flows. Everything is seeded; the same inputs and seed give
the same town.

Census totals are honoured per block group: household members fill the ACS
household population, group-quarters residents fill the remainder, and exactly
the ACS number of employed people and enrolled students are chosen, weighted by
county sex/age rates (B23001, B14003).
"""
import random
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point

SCHEMA_VERSION = 1

# Rough relative staff size per kind of place. LODES gives where people work by
# CBG; this only splits a CBG's workers between its places. Tunable.
EMPLOYER_WEIGHTS = {'workplace': 1.0, 'grocery': 3.0, 'pharmacy': 2.0, 'clinic': 8.0, 'school': 10.0}

FAMILY_SIZE_FIELDS = {2: 'size_2', 3: 'size_3', 4: 'size_4', 5: 'size_5', 6: 'size_6', 7: 'size_7_plus'}
NONFAMILY_SIZE_FIELDS = {1: 'nonfamily_size_1', 2: 'nonfamily_size_2', 3: 'nonfamily_size_3',
                         4: 'nonfamily_size_4', 5: 'nonfamily_size_5', 6: 'nonfamily_size_6',
                         7: 'nonfamily_size_7_plus'}

# B23001: (min_age, max_age, total_row, employed_row_offset), male rows; female = +86.
EMPLOYMENT_GROUPS = [
    (16, 19, 3, 4), (20, 21, 10, 4), (22, 24, 17, 4), (25, 29, 24, 4), (30, 34, 31, 4),
    (35, 44, 38, 4), (45, 54, 45, 4), (55, 59, 52, 4), (60, 61, 59, 4), (62, 64, 66, 4),
    (65, 69, 73, 2), (70, 74, 78, 2), (75, 120, 83, 2),
]
EMPLOYMENT_FEMALE_OFFSET = 86
# B14003: age groups; male public/private/not-enrolled group rows start at 4/13/22; female = +28.
ENROLLMENT_AGES = [(3, 4), (5, 9), (10, 14), (15, 17), (18, 19), (20, 24), (25, 34), (35, 120)]
ENROLLMENT_FEMALE_OFFSET = 28


@dataclass(frozen=True)
class Home:
    home_id: int
    cbg: str
    county: str
    latitude: float
    longitude: float
    members: int
    household_type: str          # family | nonfamily | group_quarters
    has_children: bool
    location_source: str         # osm_building | cbg_sample


@dataclass(frozen=True)
class Place:
    place_id: int
    osm_id: str
    name: str
    function: str                # school | grocery | pharmacy | clinic | workplace
    school_level: str | None     # preschool | k12 | college (schools only)
    latitude: float
    longitude: float
    cbg: str
    street_address: str | None
    postal_code: str | None


@dataclass(frozen=True)
class Person:
    person_id: int
    home_id: int
    home_cbg: str
    county: str
    sex: int                     # 0 male, 1 female
    age: int
    household_size: int
    household_type: str
    employment_status: str       # employed | not_employed | under_16
    role: str                    # student | worker | home
    school_level: str | None
    school_id: int | None
    school_location_type: str    # poi | out_of_zone | none
    work_id: int | None
    work_cbg: str | None
    work_location_type: str      # poi | out_of_zone | none
    work_assignment: str         # lodes | fallback | none
    grocery_id: int | None
    healthcare_id: int | None


@dataclass(frozen=True)
class Town:
    people: tuple
    homes: tuple
    places: tuple
    summary: dict

    def to_dict(self):
        return {'schema_version': SCHEMA_VERSION, 'people': [asdict(p) for p in self.people],
                'homes': [asdict(h) for h in self.homes], 'places': [asdict(p) for p in self.places]}


def _code(table, row):
    return f'{table}_{row:03d}E'


def employment_rates(distribution):
    """sex -> [(min_age, max_age, P(employed) or None)] from county B23001 counts."""
    rates = {}
    for sex, offset in (('M', 0), ('F', EMPLOYMENT_FEMALE_OFFSET)):
        rows = []
        for low, high, total_row, employed_offset in EMPLOYMENT_GROUPS:
            total = distribution.get(_code('B23001', total_row + offset))
            employed = distribution.get(_code('B23001', total_row + offset + employed_offset))
            rows.append((low, high, employed / total if total and employed is not None else None))
        rates[sex] = rows
    return rates


def enrollment_rates(distribution):
    """sex -> [(min_age, max_age, P(enrolled) or None)] from county B14003 counts."""
    rates = {}
    for sex, offset in (('M', 0), ('F', ENROLLMENT_FEMALE_OFFSET)):
        rows = []
        for index, (low, high) in enumerate(ENROLLMENT_AGES):
            values = [distribution.get(_code('B14003', start + index + offset)) for start in (4, 13, 22)]
            total = None if any(v is None for v in values) else sum(values)
            rows.append((low, high, (values[0] + values[1]) / total if total else None))
        rates[sex] = rows
    return rates


def _lookup(rows, age, default):
    for low, high, rate in rows:
        if low <= age <= high:
            return default if rate is None else rate
    return default


def _default_employment(age):
    return 0.25 if age < 18 else 0.65 if age < 65 else 0.10


def _default_enrollment(age):
    return 0.5 if age < 5 else 0.97 if age < 18 else 0.35 if age < 25 else 0.02


def _select(weights, target, rng):
    """Indices of exactly ``target`` items, weighted, without replacement (A-Res)."""
    keyed = sorted(((rng.random() ** (1.0 / w), index) for index, w in enumerate(weights) if w > 0), reverse=True)
    return {index for _, index in keyed[:max(0, target)]}


def _xyz(latitude, longitude):
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


class NearestIndex:
    """Nearest-place lookup on the sphere (no projection, works across UTM zones)."""

    def __init__(self, places):
        self.places = list(places)
        self.tree = cKDTree(_xyz([p.latitude for p in self.places], [p.longitude for p in self.places])) \
            if self.places else None

    def pick(self, latitude, longitude, rng, k=1):
        if self.tree is None:
            return None
        k = min(k, len(self.places))
        _, indexes = self.tree.query(_xyz([latitude], [longitude])[0], k=k)
        indexes = np.atleast_1d(indexes)
        return self.places[int(indexes[0] if k == 1 else rng.choice(list(indexes)))]


def _sample_person(buckets, low, high, rng):
    eligible = [b for b in buckets if b['count'] > 0 and b['max_age'] >= low and b['min_age'] <= high]
    if not eligible:
        return rng.choice(('M', 'F')), rng.randint(low, high)
    bucket = rng.choices(eligible, weights=[b['count'] for b in eligible], k=1)[0]
    return bucket['sex'], rng.randint(max(low, bucket['min_age']), min(high, bucket['max_age']))


def _weighted_size(households, fields, rng, default):
    sizes = [(size, households.get(field)) for size, field in fields.items()]
    sizes = [(size, count) for size, count in sizes if count]
    if not sizes:
        return default
    return rng.choices([s for s, _ in sizes], weights=[c for _, c in sizes], k=1)[0]


def _draft_households(block, rng):
    """[{type, children, members:[(sex, age)]}] filling the block group population."""
    households = block['households']
    population = block['population']
    household_population = block['residence'].get('household_population')
    household_population = population if household_population is None else min(household_population, population)
    total, family = households.get('total_households'), households.get('family_households')
    family_share = family / total if total and family is not None else 0.65
    family_total, with_children = households.get('total_family_households'), \
        households.get('families_with_own_children_under_18')
    child_share = with_children / family_total if family_total and with_children is not None else 0.4
    buckets = block['sex_age']

    drafts, remaining = [], household_population
    while remaining > 0:
        is_family = rng.random() < family_share
        fields, default = (FAMILY_SIZE_FIELDS, 2) if is_family else (NONFAMILY_SIZE_FIELDS, 1)
        size = max(1, min(remaining, _weighted_size(households, fields, rng, default)))
        children = rng.randint(1, size - 1) if is_family and size > 1 and rng.random() < child_share else 0
        members = [_sample_person(buckets, 18, 95, rng) for _ in range(size - children)]
        members += [_sample_person(buckets, 0, 17, rng) for _ in range(children)]
        drafts.append({'type': 'family' if is_family else 'nonfamily', 'children': children > 0, 'members': members})
        remaining -= size
    for _ in range(population - household_population):
        drafts.append({'type': 'group_quarters', 'children': False,
                       'members': [_sample_person(buckets, 18, 95, rng)]})
    return drafts


def _sample_point(polygon, rng):
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(1000):
        point = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if polygon.covers(point):
            return point.x, point.y
    point = polygon.representative_point()
    return point.x, point.y


def build_town(geometries, census, places, home_points, flows, seed=42, progress_callback=None):
    """Build the synthetic town.

    geometries   cbg -> shapely polygon (WGS84) for every selected CBG
    census       the census_inputs result's ``census`` dict
    places       dicts from places.load_places
    home_points  cbg -> [(lon, lat)] OSM residential centers
    flows        lodes.load_commute_flows output (home_cbg -> {work_cbg: jobs})
    """
    rng = random.Random(seed)
    blocks = census['block_groups']
    region = set(blocks)
    place_objects = tuple(Place(place_id=index, **{k: p[k] for k in (
        'osm_id', 'name', 'function', 'school_level', 'latitude', 'longitude', 'cbg',
        'street_address', 'postal_code')}) for index, p in enumerate(places, start=1))

    schools = [p for p in place_objects if p.function == 'school']
    school_index = {level: NearestIndex([p for p in schools if p.school_level == level])
                    for level in ('preschool', 'k12', 'college')}
    any_school = NearestIndex(schools)
    grocery_index = NearestIndex([p for p in place_objects if p.function == 'grocery'])
    health_index = NearestIndex([p for p in place_objects if p.function in ('pharmacy', 'clinic')])
    employers_by_cbg = {}
    for place in place_objects:
        employers_by_cbg.setdefault(place.cbg, []).append(place)
    all_employers = list(place_objects)

    def choose_employer(candidates):
        return rng.choices(candidates, weights=[EMPLOYER_WEIGHTS[p.function] for p in candidates], k=1)[0]

    rate_cache = {}

    def county_rates(county):
        if county not in rate_cache:
            dist = census.get('distributions_by_county', {}).get(county, {})
            rate_cache[county] = (employment_rates(dist.get('employment_by_sex_age', {})),
                                  enrollment_rates(dist.get('enrollment_by_sex_age', {})))
        return rate_cache[county]

    people, homes = [], []
    summary = {'schema_version': SCHEMA_VERSION, 'seed': seed, 'people': 0, 'homes': 0, 'workers': 0,
               'students': 0, 'out_of_zone_workers': 0, 'out_of_zone_students': 0,
               'workers_fallback_assignment': 0, 'homes_from_osm': 0, 'homes_sampled': 0}

    for block_index, cbg in enumerate(sorted(blocks)):
        if progress_callback:
            progress_callback(f'Building residents ({block_index + 1}/{len(blocks)} block groups)...',
                              block_index / len(blocks))
        block = blocks[cbg]
        county = block['county_geoid']
        employment_table, enrollment_table = county_rates(county)
        drafts = _draft_households(block, rng)

        # Exact ACS totals for employed and enrolled, picked by county age/sex rates.
        flat = [(sex, age) for d in drafts for sex, age in d['members']]
        employ_weights = [_lookup(employment_table[sex], age, _default_employment(age)) if age >= 16 else 0.0
                          for sex, age in flat]
        enroll_weights = [_lookup(enrollment_table[sex], age, _default_enrollment(age)) if age >= 3 else 0.0
                          for sex, age in flat]
        employed_target = block['employment'].get('civilian_employed')
        enrolled_target = block['enrollment'].get('enrolled')
        employed = _select(employ_weights, round(sum(employ_weights)) if employed_target is None
                           else employed_target, rng)
        enrolled = _select(enroll_weights, round(sum(enroll_weights)) if enrolled_target is None
                           else enrolled_target, rng)

        polygon = geometries[cbg]
        candidates = list(home_points.get(cbg, ()))
        rng.shuffle(candidates)
        cursor = 0
        for draft in drafts:
            if candidates:
                longitude, latitude = candidates.pop()
                source = 'osm_building'
                summary['homes_from_osm'] += 1
            else:
                longitude, latitude = _sample_point(polygon, rng)
                source = 'cbg_sample'
                summary['homes_sampled'] += 1
            home_id = len(homes) + 1
            size = len(draft['members'])
            homes.append(Home(home_id, cbg, county, round(latitude, 6), round(longitude, 6), size,
                              draft['type'], draft['children'], source))
            for sex, age in draft['members']:
                is_employed, is_enrolled = cursor in employed, cursor in enrolled
                cursor += 1
                role = ('student' if is_enrolled and age < 18 else 'worker' if is_employed
                        else 'student' if is_enrolled else 'home')
                level = (None if not is_enrolled else 'preschool' if age < 5 else 'k12' if age < 18 else 'college')
                school = None
                school_type = 'none'
                if role == 'student':
                    school = (school_index[level].pick(latitude, longitude, rng)
                              or any_school.pick(latitude, longitude, rng))
                    school_type = 'poi' if school else 'out_of_zone'
                    summary['students'] += 1
                    summary['out_of_zone_students'] += school is None
                work, work_cbg, work_type, assignment = None, None, 'none', 'none'
                if role == 'worker':
                    summary['workers'] += 1
                    destinations = flows.get(cbg)
                    assignment = 'lodes'
                    if destinations:
                        work_cbg = rng.choices(list(destinations), weights=list(destinations.values()), k=1)[0]
                    if work_cbg is not None and work_cbg not in region:
                        work_type = 'out_of_zone'
                        summary['out_of_zone_workers'] += 1
                    else:
                        local = employers_by_cbg.get(work_cbg) if work_cbg else None
                        if local is None:
                            assignment = 'fallback'
                            summary['workers_fallback_assignment'] += 1
                            local = all_employers
                        if local:
                            work = choose_employer(local)
                            work_cbg, work_type = work.cbg, 'poi'
                        else:
                            work_type = 'out_of_zone'
                            summary['out_of_zone_workers'] += 1
                grocery = grocery_index.pick(latitude, longitude, rng, k=3) if age >= 16 else None
                health = health_index.pick(latitude, longitude, rng, k=3)
                people.append(Person(
                    person_id=len(people) + 1, home_id=home_id, home_cbg=cbg, county=county,
                    sex=0 if sex == 'M' else 1, age=age, household_size=size,
                    household_type=draft['type'],
                    employment_status='under_16' if age < 16 else 'employed' if is_employed else 'not_employed',
                    role=role, school_level=level, school_id=school.place_id if school else None,
                    school_location_type=school_type, work_id=work.place_id if work else None,
                    work_cbg=work_cbg, work_location_type=work_type, work_assignment=assignment,
                    grocery_id=grocery.place_id if grocery else None,
                    healthcare_id=health.place_id if health else None,
                ))
    summary['people'], summary['homes'] = len(people), len(homes)
    return Town(tuple(people), tuple(homes), place_objects, summary)
