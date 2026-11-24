from datetime import datetime

import pytest
from shapely.geometry import box

from v2.commute import assign_workplaces
from v2.people import attach_workplaces, build_population, worker_counts

CBG = '240230001001'
WORK_CBG = '240230001002'
POLYGONS = {CBG: box(-79.42, 39.40, -79.41, 39.42), WORK_CBG: box(-79.41, 39.40, -79.40, 39.42)}


def census():
    def bucket(sex, low, high, count):
        return {'sex': sex, 'min_age': low, 'max_age': high, 'count': count}
    buckets = [bucket('M', 0, 17, 10), bucket('M', 18, 64, 10), bucket('F', 0, 17, 10), bucket('F', 18, 64, 10)]
    households = {'total_households': 14, 'family_households': 10, 'nonfamily_households': 4,
                  'total_family_households': 10, 'families_with_own_children_under_18': 5,
                  'size_2': 4, 'size_3': 4, 'size_4': 2, 'nonfamily_size_1': 4}
    block = {'population': 40, 'county_geoid': '24023', 'sex_age': buckets, 'households': households,
             'residence': {'household_population': 36, 'group_quarters_population': 4},
             'employment': {'civilian_employed': 10}, 'enrollment': {'enrolled': 8}}
    return {'block_groups': {CBG: block}, 'distributions_by_county': {}}


def build(seed=42, home_points=None):
    return build_population(census(), {CBG: POLYGONS[CBG]}, home_points, seed)


def test_population_matches_census_totals():
    people, homes, summary = build()
    assert len(people) == 40 == summary['census_population']
    assert sum(h.members for h in homes) == 40
    assert sum(h.household_type == 'group_quarters' for h in homes) == 4
    # exactly 10 employed and 8 enrolled; a few may be counted as students instead of workers
    assert summary['workers'] + sum(p.role == 'student' and p.age < 18 for p in people) >= 10 - 2
    assert summary['students'] >= 8 - 2 and summary['workers'] <= 10
    assert {p.person_id for p in people} == {str(i) for i in range(1, 41)}   # integer-like ids
    assert all(p.school_level is None for p in people if p.role != 'student')


def test_seeded_and_osm_homes_used_first():
    assert build(1)[0] == build(1)[0] and build(1)[0] != build(2)[0]
    _, homes, summary = build(home_points={CBG: [(-79.415, 39.405)]})
    assert summary['homes_from_osm'] == 1 and homes[0].location_source in ('osm_building', 'cbg_sample')
    assert sum(h.location_source == 'osm_building' for h in homes) == 1


def test_flow_into_commute_and_mesa_skeleton():
    people, _, _ = build()
    places = [{'osm_id': 'node/1', 'function': 'workplace', 'cbg': WORK_CBG}]
    assignments, summary = assign_workplaces(worker_counts(people), {CBG: {WORK_CBG: 1}},
                                             places, {CBG, WORK_CBG}, seed=3)
    assert summary['workers'] == sum(p.role == 'worker' for p in people)
    placed = attach_workplaces(people, assignments)
    assert all(p.work_type == 'poi' and p.work_osm_id == 'node/1' for p in placed if p.role == 'worker')
    assert all(p.work_type == 'none' for p in placed if p.role != 'worker')

    pytest.importorskip('mesa')
    from v2.mesa_model import run_patterns
    patterns = run_patterns(placed, datetime(2026, 10, 13), 12)   # Tuesday
    at_work = patterns['600']['places'].get('node/1', [])          # 10:00
    assert sorted(at_work) == sorted(p.person_id for p in placed if p.role == 'worker')


def test_attach_requires_matching_assignments():
    people, _, _ = build()
    with pytest.raises(ValueError):
        attach_workplaces(people, {})


def households_of(people):
    grouped = {}
    for person in people:
        grouped.setdefault(person.home_id, []).append(person)
    return grouped


def test_household_counts_match_the_census():
    people, homes, summary = build()
    assert summary['constrained_block_groups'] == 1 and summary['unconstrained_block_groups'] == 0
    family = [h for h in homes if h.household_type == 'family']
    nonfamily = [h for h in homes if h.household_type == 'nonfamily']
    quarters = [h for h in homes if h.household_type == 'group_quarters']
    assert (len(family), len(nonfamily)) == (10, 4)
    assert summary['generated_households'] == summary['census_households'] == 14
    assert sum(h.members for h in family + nonfamily) == 36            # household population, exactly
    assert all(h.members >= 2 for h in family) and all(h.members >= 1 for h in nonfamily)
    members = households_of(people)
    assert sum(any(p.age < 18 for p in members[h.home_id]) for h in family) == 5   # families with own children
    assert not any(p.age < 18 for h in nonfamily + quarters for p in members[h.home_id])


def test_sizes_are_reconciled_with_the_household_population():
    for household_population in (30, 36, 60):       # the table's sizes add up to 32 people
        data = census()
        block = data['block_groups'][CBG]
        block['residence'] = {'household_population': household_population}
        block['population'] = household_population
        people, homes, _ = build_population(data, {CBG: POLYGONS[CBG]}, None, seed=3)
        assert sum(h.members for h in homes) == len(people) == household_population


def test_generate_town_and_sample_trace():
    pytest.importorskip('mesa')
    from v2.town import generate_town, run_movement, sample_trace
    places = [{'osm_id': 'node/1', 'function': 'workplace', 'cbg': WORK_CBG}]
    geometries = {CBG: POLYGONS[CBG], WORK_CBG: POLYGONS[WORK_CBG]}
    data = census()
    data['block_groups'][WORK_CBG] = data['block_groups'][CBG]       # both CBGs are in the region
    flows = {CBG: {WORK_CBG: 1}, WORK_CBG: {WORK_CBG: 1}}
    people, homes, summary = generate_town(data, geometries, places, {}, flows, seed=4)
    assert summary['commute']['poi'] == summary['population']['workers'] > 0
    patterns = run_movement(people, hours=24, seed=4)   # default start is a Monday
    trace = sample_trace(people, patterns)
    worker = next(v for v in trace.values() if v['work_type'] == 'poi')
    assert [m[1:] for m in worker['moves'][:3]] == [['home', worker['moves'][0][2]], ['place', 'node/1'], ['home', worker['moves'][0][2]]]


def test_falls_back_when_household_counts_are_missing():
    data = census()
    del data['block_groups'][CBG]['households']['nonfamily_households']
    people, homes, summary = build_population(data, {CBG: POLYGONS[CBG]}, None, seed=5)
    assert summary['unconstrained_block_groups'] == 1 and len(people) == 40
