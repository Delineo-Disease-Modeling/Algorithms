"""Hourly Mesa model that turns papdata into the minute-keyed patterns the simulator replays.

One Mesa step is one hour. Each agent is placed at home or at exactly one place
every hour, so every snapshot is complete (the binary patterns codec requires
that). Snapshots are keyed by cumulative minutes from ``start``; the first is
"60", matching gen_patterns. Output shape:
    {"60": {"homes": {home_id: [person_id]}, "places": {place_id: [person_id]}}, ...}
"""
import random
from datetime import datetime, timedelta

from mesa import Agent, Model

GROCERY_PROBABILITY = 0.3          # Saturday 10:00-13:00, per hour
HEALTHCARE_PROBABILITY = 0.004     # weekdays 10:00-16:00, per hour
SCHOOL_HOURS = {'preschool': (8, 12), 'k12': (8, 15), 'college': (9, 15)}
WORK_HOURS = (9, 17)


def choose_location(plan, when, rng):
    """('home' | 'place', id) for one agent at this clock time."""
    weekday, hour = when.weekday(), when.hour
    if weekday < 5:
        if plan['role'] == 'student':
            start, end = SCHOOL_HOURS.get(plan['school_level'], SCHOOL_HOURS['k12'])
            if start <= hour < end and plan['school']:
                return 'place', plan['school']
        elif plan['role'] == 'worker' and WORK_HOURS[0] <= hour < WORK_HOURS[1] and plan['work']:
            return 'place', plan['work']
        if plan['healthcare'] and 10 <= hour < 16 and rng.random() < HEALTHCARE_PROBABILITY:
            return 'place', plan['healthcare']
    elif weekday == 5 and plan['grocery'] and 10 <= hour < 13 and rng.random() < GROCERY_PROBABILITY:
        return 'place', plan['grocery']
    return 'home', plan['home']


def agent_plan(person_id, person, papdata):
    """Pin each destination once; out-of-zone work/school map to the external place."""
    work = person.get('work_poi')
    if person.get('work_location_type') == 'out_of_zone':
        work = papdata.get('external_workplace_id')
    school = person.get('school_poi')
    if person.get('school_location_type') == 'out_of_zone':
        school = papdata.get('external_school_id')
    return {
        'person_id': person_id, 'home': person['home'], 'role': person.get('role', 'home'),
        'school_level': person.get('school_level'), 'school': school, 'work': work,
        'grocery': person.get('grocery_poi'), 'healthcare': person.get('healthcare_poi'),
    }


class PersonAgent(Agent):
    def __init__(self, model, plan):
        super().__init__(model)
        self.plan = plan
        self.location_kind = 'home'
        self.location_id = plan['home']

    def step(self):
        self.location_kind, self.location_id = choose_location(self.plan, self.model.when, self.model.clock)


class TownModel(Model):
    def __init__(self, papdata, start, seed=42):
        super().__init__()
        self.start = start
        self.when = start
        self.clock = random.Random(seed)
        self.patterns = {}
        for person_id in sorted(papdata['people'], key=int):
            PersonAgent(self, agent_plan(person_id, papdata['people'][person_id], papdata))

    def step(self):
        self.when += timedelta(hours=1)
        self.agents.do('step')
        homes, places = {}, {}
        for agent in self.agents:
            bucket = places if agent.location_kind == 'place' else homes
            bucket.setdefault(agent.location_id, []).append(agent.plan['person_id'])
        minute = int((self.when - self.start).total_seconds() // 60)
        self.patterns[str(minute)] = {'homes': homes, 'places': places}


def run_patterns(papdata, start, hours, seed=42, progress_callback=None):
    """Step the Mesa model ``hours`` times; return the legacy patterns dict."""
    if not isinstance(start, datetime):
        raise ValueError('start must be a datetime.')
    if type(hours) is not int or hours < 1:
        raise ValueError('hours must be a positive integer.')
    model = TownModel(papdata, start.replace(tzinfo=None), seed)
    for hour in range(hours):
        model.step()
        if progress_callback and hour % 12 == 0:
            progress_callback(hour + 1, hours)
    return model.patterns
