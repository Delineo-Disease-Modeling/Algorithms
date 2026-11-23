"""Join Census employment, LODES commuting flows and OSM places into workplace assignments.

Three inputs, one question: for each employed resident, which workplace?
  1. Census: how many employed people live in each home CBG (civilian_employed).
  2. LODES:  for a worker living in home CBG A, how likely each work CBG B is.
  3. OSM:    which actual places exist inside B to work at.
A work CBG outside the region becomes "out_of_zone". A work CBG inside the region
with no OSM place is reported as "unmatched_no_osm" instead of being hidden.
"""
import random

POI = 'poi'
OUT_OF_ZONE = 'out_of_zone'
UNMATCHED_NO_OSM = 'unmatched_no_osm'   # LODES sends workers here, OSM has no place there
NO_FLOWS = 'no_flows'                   # LODES has no flows from this home CBG

# DECISION: which OSM places count as employers, and how a CBG's workers split
# between them. Equal weight is a placeholder; staff-size weights or NAICS-based
# shares are the natural next step.
EMPLOYER_FUNCTIONS = ('workplace', 'grocery', 'pharmacy', 'clinic', 'school')


def workers_by_home_cbg(census):
    """home CBG -> number of employed residents, from the census_inputs result.

    CBGs whose employed count is missing (suppressed or unavailable) are left
    out and listed by the caller via ``missing_worker_counts``.
    """
    blocks = census['block_groups']
    counts = {cbg: block['employment'].get('civilian_employed') for cbg, block in blocks.items()}
    return {cbg: n for cbg, n in counts.items() if n is not None}


def missing_worker_counts(census):
    return sorted(cbg for cbg, block in census['block_groups'].items()
                  if block['employment'].get('civilian_employed') is None)


def employers_by_cbg(places):
    grouped = {}
    for place in places:
        if place['function'] in EMPLOYER_FUNCTIONS:
            grouped.setdefault(place['cbg'], []).append(place)
    return grouped


def assign_workplaces(workers, flows, places, region, seed=42):
    """Assign every employed resident a work CBG (LODES) and a place in it (OSM).

    workers  home CBG -> employed resident count
    flows    home CBG -> {work CBG: jobs}   (lodes.load_commute_flows)
    places   dicts from places.load_places  (need 'cbg', 'function', 'osm_id')
    region   set of CBGs inside the selected region

    Returns (assignments, summary). assignments maps each home CBG to a list with
    one dict per worker: {'work_cbg', 'osm_id', 'type'}. Deterministic for a seed.
    """
    rng = random.Random(seed)
    employers = employers_by_cbg(places)
    assignments = {}
    summary = {POI: 0, OUT_OF_ZONE: 0, UNMATCHED_NO_OSM: 0, NO_FLOWS: 0}
    for home in sorted(workers):
        destinations = flows.get(home)
        entries = []
        for _ in range(workers[home]):
            if not destinations:
                entry = {'work_cbg': None, 'osm_id': None, 'type': NO_FLOWS}
            else:
                work_cbg = rng.choices(list(destinations), weights=list(destinations.values()), k=1)[0]
                if work_cbg not in region:
                    entry = {'work_cbg': work_cbg, 'osm_id': None, 'type': OUT_OF_ZONE}
                elif work_cbg not in employers:
                    entry = {'work_cbg': work_cbg, 'osm_id': None, 'type': UNMATCHED_NO_OSM}
                else:
                    place = rng.choice(employers[work_cbg])
                    entry = {'work_cbg': work_cbg, 'osm_id': place['osm_id'], 'type': POI}
            summary[entry['type']] += 1
            entries.append(entry)
        assignments[home] = entries
    return assignments, {**summary, 'workers': sum(summary.values())}
