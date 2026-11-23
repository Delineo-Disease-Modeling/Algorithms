from datetime import datetime

import pytest

from v2.commute import assign_workplaces, missing_worker_counts, workers_by_home_cbg
from v2.schema import Person

A, B, C, OUTSIDE = '240230001001', '240230001002', '240230001003', '420010001001'
REGION = {A, B, C}
PLACES = [
    {'osm_id': 'node/1', 'function': 'workplace', 'cbg': B},
    {'osm_id': 'node/2', 'function': 'grocery', 'cbg': B},
    {'osm_id': 'node/3', 'function': 'bench', 'cbg': A},   # not an employer function
]


def test_workers_come_from_census_employment():
    census = {'block_groups': {A: {'employment': {'civilian_employed': 5}},
                               B: {'employment': {'civilian_employed': None}}}}
    assert workers_by_home_cbg(census) == {A: 5}
    assert missing_worker_counts(census) == [B]


def test_lodes_picks_the_cbg_and_osm_picks_the_place():
    flows = {A: {B: 6, C: 2, OUTSIDE: 2}}
    assignments, summary = assign_workplaces({A: 1000}, flows, PLACES, REGION, seed=1)
    entries = assignments[A]
    assert summary['workers'] == 1000 and sum(summary[k] for k in
        ('poi', 'out_of_zone', 'unmatched_no_osm', 'no_flows')) == 1000
    assert all(e['osm_id'] in ('node/1', 'node/2') and e['work_cbg'] == B for e in entries if e['type'] == 'poi')
    assert all(e['work_cbg'] == C for e in entries if e['type'] == 'unmatched_no_osm')   # C has no OSM place
    assert all(e['work_cbg'] == OUTSIDE for e in entries if e['type'] == 'out_of_zone')
    assert 500 < summary['poi'] < 700 and summary['out_of_zone'] > 100   # roughly 60% / 20%


def test_missing_flows_and_determinism():
    _, summary = assign_workplaces({A: 3}, {}, PLACES, REGION)
    assert summary['no_flows'] == 3
    flows = {A: {B: 1, C: 1}}
    assert assign_workplaces({A: 50}, flows, PLACES, REGION, seed=7) == \
        assign_workplaces({A: 50}, flows, PLACES, REGION, seed=7)


def test_mesa_skeleton_puts_workers_at_work_in_work_hours():
    pytest.importorskip('mesa')
    from v2.mesa_model import run_patterns
    people = [Person('1', 'H1', A, 0, 40, 'worker', 'node/1', 'poi'),
              Person('2', 'H2', A, 1, 40, 'home')]
    patterns = run_patterns(people, datetime(2026, 10, 13), 24)   # Tuesday
    assert '1' in patterns['600']['places']['node/1']    # 10:00
    assert '1' in patterns['1200']['homes']['H1']        # 20:00
    assert all('2' in patterns[m]['homes']['H2'] for m in patterns)
