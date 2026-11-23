"""Run the whole v2 chain for one ZIP code and print a movement trace.

From Algorithms/server:
    python -m v2.run_trace --zip 21550 [--coverage 5] [--hours 48] [--seed 42]

Needs CENSUS_API_KEY and internet access. The first run downloads Census
geography, LODES commuting files and OSM tiles into server/data/v2/ (git-ignored);
later runs reuse that cache. Writes people, homes, places, patterns and the trace
to server/data/v2/runs/<zip>_<seed>.json.
"""
import argparse
import json
import os
from datetime import datetime, timedelta

from . import CACHE_DIR
from .census import CensusInputsClient
from .geography import load_selected_cbgs
from .lodes import load_commute_flows, load_manifest
from .places import load_home_points, load_places
from .population import PopulationTotalsClient
from .service import V2RegionService
from .town import generate_town, run_movement, sample_trace


def say(message, *_):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zip', required=True, help='five-digit ZIP code')
    parser.add_argument('--coverage', type=float, default=5.0,
                        help='percent of commuting jobs to cover when expanding the region (default 5)')
    parser.add_argument('--hours', type=int, default=48, help='hours to simulate (default 48)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start', default='2026-10-12', help='midnight start date, default a Monday')
    args = parser.parse_args()
    if not os.environ.get('CENSUS_API_KEY'):
        raise SystemExit('Set CENSUS_API_KEY first (https://api.census.gov/data/key_signup.html).')

    say('1/6 Region: ZIP -> core CBGs -> commuting expansion -> population totals')
    preview = V2RegionService(None, PopulationTotalsClient()).build_region(
        args.zip, say, args.coverage / 100)
    cluster = preview['cluster']
    say(f'    {len(cluster)} CBGs, population {preview["size"]}')

    say('2/6 Census inputs (households, age/sex, employment, enrollment)')
    census = CensusInputsClient().prepare(cluster, preview['population_by_cbg'], say)

    say('3/6 OSM places and residential buildings')
    geometries = load_selected_cbgs(cluster)
    places, missing = load_places(geometries, say)
    home_points = load_home_points(geometries, say)
    say(f'    {len(places)} places, {sum(len(v) for v in home_points.values())} home points; '
        f'missing: {[m["field"] for m in missing] or "none"}')

    say('4/6 LODES commuting flows')
    flows = load_commute_flows(cluster, load_manifest(), say)

    say('5/6 People, households and workplaces')
    people, homes, summary = generate_town(census, geometries, places, home_points, flows, args.seed, say)
    say(json.dumps(summary, indent=2))

    say(f'6/6 Mesa movement for {args.hours} hours')
    start = datetime.fromisoformat(args.start)
    patterns = run_movement(people, args.hours, start, args.seed)
    trace = sample_trace(people, patterns)
    for pid, entry in trace.items():
        print(f'person {pid} ({entry["role"]}, work: {entry["work_type"]})')
        for minute, kind, location in entry['moves'][:8]:
            print(f'    {start + timedelta(minutes=minute):%a %H:%M}  {kind} {location}')

    out = CACHE_DIR / 'runs'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{args.zip}_{args.seed}.json'
    path.write_text(json.dumps({
        'summary': summary, 'trace': trace, 'places': places,
        'people': [p.__dict__ for p in people], 'homes': [h.__dict__ for h in homes],
        'patterns': patterns}))
    say(f'Wrote {path}')


if __name__ == '__main__':
    main()
