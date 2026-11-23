"""STARTING LAYOUT for the v2 agent schema. Deliberately small and incomplete.

Only fields the current generation and rules read. Add a field when a rule or the
disease model reads it, not before. Open decisions are marked DECISION.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Person:
    person_id: str                 # integer-like string, as papdata and the patterns codec require
    home_id: str
    home_cbg: str
    sex: int                       # 0 male, 1 female (matches papdata)
    age: int
    role: str                      # DECISION: student | worker | home. Add more roles if rules need them
    work_osm_id: str | None = None  # key into the places list (set by commute assignment)
    work_type: str = 'none'        # poi | out_of_zone | unmatched_no_osm | no_flows | none
    household_size: int = 1
    household_type: str = 'nonfamily'   # family | nonfamily | group_quarters
    school_level: str | None = None     # preschool | k12 | college, only for enrolled people
    # TODO: school/grocery/health-care assignment and any disease-model attributes
    #       (vaccination, comorbidity, ...).


@dataclass(frozen=True)
class Home:
    home_id: str
    cbg: str
    latitude: float
    longitude: float
    members: int = 1
    household_type: str = 'nonfamily'
    location_source: str = 'cbg_sample'   # osm_building | cbg_sample
