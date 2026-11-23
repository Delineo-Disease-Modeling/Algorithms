"""STARTING LAYOUT for the v2 Mesa model. One rule only: workers are at work in work hours.

One Mesa step = one hour. Every person is placed somewhere every step, because
the existing patterns format needs a full snapshot. Snapshots are keyed by
cumulative minutes ("60", "120", ...) like gen_patterns. Output shape:
    {"60": {"homes": {home_id: [person_id]}, "places": {osm_id: [person_id]}}, ...}
Place keys are OSM IDs here; converting them to the numeric place IDs the
existing papdata/codec require is a later step.

DECISIONS still open: schedules for students and non-workers, errand trips
(grocery, health care), weekend behaviour, calibration against gen_patterns.
"""
import random
from datetime import timedelta

from mesa import Agent, Model

WORK_HOURS = (9, 17)   # DECISION: placeholder, replace with a cited schedule
WORK_WEEKDAYS = range(5)


def choose_location(person, when):
    """('home', home_id) or ('place', osm_id) for one person at this clock time."""
    at_work_time = when.weekday() in WORK_WEEKDAYS and WORK_HOURS[0] <= when.hour < WORK_HOURS[1]
    if person.role == 'worker' and person.work_type == 'poi' and at_work_time:
        return 'place', person.work_osm_id
    # out_of_zone / unmatched workers stay 'home' for now. DECISION: send out_of_zone
    # workers to an external place so contact with the region is removed during work.
    return 'home', person.home_id


class PersonAgent(Agent):
    def __init__(self, model, person):
        super().__init__(model)
        self.person = person
        self.kind, self.location = 'home', person.home_id

    def step(self):
        self.kind, self.location = choose_location(self.person, self.model.when)


class TownModel(Model):
    def __init__(self, people, start, seed=42):
        super().__init__()
        self.start, self.when = start, start
        self.rng = random.Random(seed)   # unused by this rule; later rules draw from it
        self.patterns = {}
        for person in people:
            PersonAgent(self, person)

    def step(self):
        self.when += timedelta(hours=1)
        self.agents.do('step')
        homes, places = {}, {}
        for agent in self.agents:
            bucket = places if agent.kind == 'place' else homes
            bucket.setdefault(agent.location, []).append(agent.person.person_id)
        self.patterns[str(int((self.when - self.start).total_seconds() // 60))] = {
            'homes': homes, 'places': places}


def run_patterns(people, start, hours, seed=42):
    model = TownModel(people, start, seed)
    for _ in range(hours):
        model.step()
    return model.patterns
