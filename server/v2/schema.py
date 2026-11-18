"""STARTING LAYOUT for the v2 agent schema. Deliberately small and incomplete.

Only fields the first Mesa rule needs. Add a field when a rule or the disease
model reads it, not before. Open decisions are marked DECISION.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Person:
    person_id: str
    home_id: str
    home_cbg: str
    sex: int                       # 0 male, 1 female (matches papdata)
    age: int
    role: str                      # DECISION: student | worker | home. Add more roles if rules need them
    work_osm_id: str | None = None  # key into the places list
    work_type: str = 'none'        # poi | out_of_zone | unmatched_no_osm | no_flows | none
    # TODO: school assignment, household composition, grocery/health-care IDs,
    #       and any disease-model attributes (vaccination, comorbidity, ...).


@dataclass(frozen=True)
class Home:
    home_id: str
    cbg: str
    latitude: float
    longitude: float
    # TODO: members, household type (family / nonfamily / group quarters).
