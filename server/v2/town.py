"""Glue for a first end-to-end run: people -> workplaces -> Mesa movement -> a readable trace.

Used by both the HTTP job (service.start_town_job, POST /v2/town) and the command
line runner (python -m v2.run_trace). Inputs are the outputs of earlier v2 steps:
the census_inputs ``census`` dict, CBG polygons, OSM places and home points, and
LODES flows.
"""
from datetime import datetime

from .commute import assign_workplaces
from .people import attach_workplaces, build_population, worker_counts


def generate_town(census, geometries, places, home_points, flows, seed=42, progress_callback=None):
    """People and homes with workplaces attached. Returns (people, homes, summary)."""
    people, homes, population_summary = build_population(
        census, geometries, home_points, seed, progress_callback)
    assignments, commute_summary = assign_workplaces(
        worker_counts(people), flows, places, set(census['block_groups']), seed)
    summary = {'population': population_summary, 'commute': commute_summary,
               'places': len(places)}
    return attach_workplaces(people, assignments), homes, summary


def run_movement(people, hours=48, start=datetime(2026, 10, 12), seed=42):
    """Run the Mesa model. ``start`` should be midnight; the default is a Monday."""
    from .mesa_model import run_patterns   # Mesa is only needed for this step
    return run_patterns(people, start, hours, seed)


def sample_trace(people, patterns, workers=5, others=2):
    """Where a few people are over time, as location changes only.

    Returns {person_id: {'role', 'work_type', 'moves': [[minute, 'home'|'place', location_id]]}}.
    Picks workers with a matched OSM workplace first, so the trace shows movement.
    """
    chosen = [p for p in people if p.work_type == 'poi'][:workers]
    chosen += [p for p in people if p.role != 'worker'][:others]
    wanted = {p.person_id for p in chosen}
    moves = {pid: [] for pid in wanted}
    for minute in sorted(patterns, key=int):
        step = patterns[minute]
        for kind, bucket in (('home', step['homes']), ('place', step['places'])):
            for location, members in bucket.items():
                for pid in wanted.intersection(members):
                    if not moves[pid] or moves[pid][-1][1:] != [kind, location]:
                        moves[pid].append([int(minute), kind, location])
    return {p.person_id: {'role': p.role, 'work_type': p.work_type, 'moves': moves[p.person_id]}
            for p in chosen}
