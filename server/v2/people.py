"""Households and people for a v2 region, built from the confirmed Census inputs.

Upstream:  census.py output (per-CBG sex/age, household tables, employment,
           enrollment; per-county employment/enrollment rates), CBG polygons from
           geography.py, and optionally OSM home points from places.py.
Downstream: commute.py gives the employed people workplaces; the Mesa model reads
           the resulting Person/Home records (schema.py).

"""
import random
from dataclasses import replace

from shapely.geometry import Point

from .schema import Home, Person

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


# DECISION: fallbacks when a county's rate table is missing or suppressed.
def _default_employment(age):
    return 0.25 if age < 18 else 0.65 if age < 65 else 0.10


def _default_enrollment(age):
    return 0.5 if age < 5 else 0.97 if age < 18 else 0.35 if age < 25 else 0.02


def _select(weights, target, rng):
    """Indices of exactly ``target`` items, weighted, without replacement (A-Res)."""
    keyed = sorted(((rng.random() ** (1.0 / w), index) for index, w in enumerate(weights) if w > 0), reverse=True)
    return {index for _, index in keyed[:max(0, target)]}


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


def _fit_sizes(counts, target, default, rng):
    """Exactly ``target`` household sizes, drawn from the census size counts.

    Uses the table's own multiset of sizes. If the table has more households
    than ``target`` a random few are dropped; if fewer, extra sizes are drawn
    in proportion to the counts (``default`` when the table is empty).
    """
    sizes = [size for size, n in sorted(counts.items()) for _ in range(n)]
    rng.shuffle(sizes)
    sizes = sizes[:target]
    positive = [(size, n) for size, n in counts.items() if n > 0]
    while len(sizes) < target:
        sizes.append(rng.choices([s for s, _ in positive], weights=[n for _, n in positive], k=1)[0]
                     if positive else default)
    return sizes


def _draft_constrained(block, rng, notes):
    """Households that match the census counts, or None when the tables can't support it.

    Fixed by the census: the number of family and nonfamily households (B11001),
    their size distributions (B11016), how many families have own children under
    18 (B11003), and the household population (B25008). The member total is made
    to equal the household population exactly by moving a few households up or
    down one size at a time, since ACS tables are separate estimates and rarely
    add up perfectly.
    """
    households = block['households']
    family, nonfamily = households.get('family_households'), households.get('nonfamily_households')
    population = block['population']
    household_population = block['residence'].get('household_population')
    household_population = population if household_population is None else min(household_population, population)
    if family is None or nonfamily is None or family + nonfamily == 0:
        return None
    if household_population < 2 * family + nonfamily:     # not enough people for the minimum sizes
        return None

    homes = [['family', size] for size in _fit_sizes(
        {size: households.get(field) or 0 for size, field in FAMILY_SIZE_FIELDS.items()}, family, 2, rng)]
    homes += [['nonfamily', size] for size in _fit_sizes(
        {size: households.get(field) or 0 for size, field in NONFAMILY_SIZE_FIELDS.items()}, nonfamily, 1, rng)]

    # Reconcile sizes with the household population, one person at a time.
    # Households of size 7 may really be "7 or more", so they grow first.
    gap = household_population - sum(size for _, size in homes)
    notes['size_adjustment_people'] = notes.get('size_adjustment_people', 0) + abs(gap)
    while gap > 0:
        open_ended = [h for h in homes if h[1] >= 7]
        rng.choice(open_ended or homes)[1] += 1
        gap -= 1
    while gap < 0:
        rng.choice([h for h in homes if h[1] > (2 if h[0] == 'family' else 1)])[1] -= 1
        gap += 1
    rng.shuffle(homes)

    # Exactly this many families have own children under 18 (fallback: a share).
    family_ids = [i for i, (kind, _) in enumerate(homes) if kind == 'family']
    with_children_count = households.get('families_with_own_children_under_18')
    if with_children_count is None:
        with_children = {i for i in family_ids if rng.random() < 0.4}   # DECISION: fallback share
    else:
        with_children = set(rng.sample(family_ids, min(with_children_count, len(family_ids))))

    buckets = block['sex_age']
    drafts = []
    for index, (kind, size) in enumerate(homes):
        children = rng.randint(1, size - 1) if index in with_children and size > 1 else 0
        members = [_sample_person(buckets, 18, 95, rng) for _ in range(size - children)]
        members += [_sample_person(buckets, 0, 17, rng) for _ in range(children)]
        drafts.append({'type': kind, 'members': members})
    for _ in range(population - household_population):
        drafts.append({'type': 'group_quarters', 'members': [_sample_person(buckets, 18, 95, rng)]})
    return drafts


def draft_households(block, rng, notes=None):
    """[{type, members:[(sex, age)]}] for one block group, matching its census counts.

    Uses the census household counts when the block group's tables allow it
    (``_draft_constrained``) and otherwise falls back to sampling households until
    the population is used up (``_draft_unconstrained``). ``notes`` (a dict) collects
    how often each path ran.
    """
    notes = {} if notes is None else notes
    drafts = _draft_constrained(block, rng, notes)
    if drafts is None:
        notes['unconstrained_block_groups'] = notes.get('unconstrained_block_groups', 0) + 1
        return _draft_unconstrained(block, rng)
    notes['constrained_block_groups'] = notes.get('constrained_block_groups', 0) + 1
    return drafts


def _draft_unconstrained(block, rng):
    """[{type, members:[(sex, age)]}] filling the block group's exact ACS population.

    Fallback used when household counts are missing. The number of households
    emerges from random sampling. Household members fill ``household_population``;
    group-quarters residents (population minus household population) become
    one-person homes.
    """
    households = block['households']
    population = block['population']
    household_population = block['residence'].get('household_population')
    household_population = population if household_population is None else min(household_population, population)
    total, family = households.get('total_households'), households.get('family_households')
    family_share = family / total if total and family is not None else 0.65      # DECISION: fallback
    family_total = households.get('total_family_households')
    with_children = households.get('families_with_own_children_under_18')
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
        drafts.append({'type': 'family' if is_family else 'nonfamily', 'members': members})
        remaining -= size
    for _ in range(population - household_population):
        drafts.append({'type': 'group_quarters', 'members': [_sample_person(buckets, 18, 95, rng)]})
    return drafts


def _sample_point(polygon, rng):
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(1000):
        point = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if polygon.covers(point):
            return point.x, point.y
    point = polygon.representative_point()
    return point.x, point.y


def build_population(census, geometries, home_points=None, seed=42, progress_callback=None):
    """Households and people for every confirmed block group.

    census        the census_inputs result's ``census`` dict
    geometries    cbg -> shapely polygon (WGS84), used when no OSM home point is left
    home_points   optional cbg -> [(lon, lat)] from places.load_home_points
    Returns (people, homes, summary). Workplaces are not assigned here; call
    commute.assign_workplaces on worker_counts(people), then attach_workplaces.
    """
    rng = random.Random(seed)
    home_points = home_points or {}
    blocks = census['block_groups']
    rate_cache = {}

    def county_rates(county):
        if county not in rate_cache:
            dist = census.get('distributions_by_county', {}).get(county, {})
            rate_cache[county] = (employment_rates(dist.get('employment_by_sex_age', {})),
                                  enrollment_rates(dist.get('enrollment_by_sex_age', {})))
        return rate_cache[county]

    people, homes = [], []
    notes = {}
    summary = {'seed': seed, 'homes_from_osm': 0, 'homes_sampled': 0, 'workers': 0, 'students': 0}
    for index, cbg in enumerate(sorted(blocks)):
        if progress_callback:
            progress_callback(f'Generating residents ({index + 1}/{len(blocks)} block groups)...',
                              index / len(blocks))
        block = blocks[cbg]
        employment_table, enrollment_table = county_rates(block['county_geoid'])
        drafts = draft_households(block, rng, notes)

        # Exact ACS totals for employed and enrolled, picked by county age/sex rates.
        flat = [(sex, age) for draft in drafts for sex, age in draft['members']]
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

        candidates = list(home_points.get(cbg, ()))
        rng.shuffle(candidates)
        cursor = 0
        for draft in drafts:
            if candidates:
                longitude, latitude = candidates.pop()
                source = 'osm_building'
                summary['homes_from_osm'] += 1
            else:
                longitude, latitude = _sample_point(geometries[cbg], rng)
                source = 'cbg_sample'
                summary['homes_sampled'] += 1
            home_id = str(len(homes) + 1)
            size = len(draft['members'])
            homes.append(Home(home_id, cbg, round(latitude, 6), round(longitude, 6), size,
                              draft['type'], source))
            for sex, age in draft['members']:
                is_employed, is_enrolled = cursor in employed, cursor in enrolled
                cursor += 1
                # DECISION: a 16-17 year old who is both employed and enrolled counts as a
                # student; an adult who is both counts as a worker.
                role = ('student' if is_enrolled and age < 18 else 'worker' if is_employed
                        else 'student' if is_enrolled else 'home')
                level = (None if role != 'student' else 'preschool' if age < 5
                         else 'k12' if age < 18 else 'college')
                summary['workers'] += role == 'worker'
                summary['students'] += role == 'student'
                people.append(Person(
                    person_id=str(len(people) + 1), home_id=home_id, home_cbg=cbg,
                    sex=0 if sex == 'M' else 1, age=age, role=role, household_size=size,
                    household_type=draft['type'], school_level=level))
    summary['people'], summary['homes'] = len(people), len(homes)
    summary['census_population'] = sum(b['population'] for b in blocks.values())
    census_households = [b['households'].get('total_households') for b in blocks.values()]
    summary['census_households'] = None if None in census_households else sum(census_households)
    summary['generated_households'] = sum(h.household_type != 'group_quarters' for h in homes)
    summary.update({'constrained_block_groups': 0, 'unconstrained_block_groups': 0,
                    'size_adjustment_people': 0, **notes})
    return people, homes, summary


def worker_counts(people):
    """home CBG -> number of people with role 'worker' (input to commute.assign_workplaces)."""
    counts = {}
    for person in people:
        if person.role == 'worker':
            counts[person.home_cbg] = counts.get(person.home_cbg, 0) + 1
    return counts


def attach_workplaces(people, assignments):
    """Copy commute assignments onto the matching workers, in order, and return new people."""
    queues = {cbg: iter(entries) for cbg, entries in assignments.items()}
    result = []
    for person in people:
        if person.role != 'worker':
            result.append(person)
            continue
        try:
            entry = next(queues[person.home_cbg])
        except (KeyError, StopIteration):
            raise ValueError(f'No workplace assignment left for a worker in CBG {person.home_cbg}.') from None
        result.append(replace(person, work_osm_id=entry['osm_id'], work_type=entry['type']))
    return result
