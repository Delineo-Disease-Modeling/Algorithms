"""Demand-pull movement model (docs/MOVEMENT_MODEL_REDESIGN.md §9).

Destinations are filled to a realistic per-POI occupancy target
(popularity_by_hour x day_factor x f_j x open_gate x movement_scale) by topping
up from the home pool; everyone not pulled stays home. There is no legacy /
supply-push fallback — missing patterns data fails loudly.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
# The shared conftest may have cached patterns/patterns_loader from a different
# checkout (it hardcodes the main Algorithms/server path); evict so the imports
# below resolve to THIS worktree's server dir.
for _m in ("patterns", "patterns_loader"):
    sys.modules.pop(_m, None)

from patterns import (  # noqa: E402
    _build_open_gate,
    _catchment_fraction,
    gen_patterns,
)

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CBG_IN, CBG_OUT = "400010001001", "999990000001"


def _shared(pois):
    df = pd.DataFrame(pois)

    class _Stub:
        def is_empty(self):
            return False

        def for_patterns_stats(self, placekeys):
            return df[df["placekey"].isin(placekeys)]

    return _Stub()


def _poi(pk, hour_counts, *, dwell=30, open_hours=None, naics="445110",
         home_cbgs=None, day=10):
    return {
        "placekey": pk, "median_dwell": dwell,
        "popularity_by_hour": hour_counts,
        "popularity_by_day": {d: day for d in WEEK},
        "open_hours": json.dumps(open_hours) if open_hours else None,
        "naics_code": naics,
        "visitor_home_cbgs": json.dumps(home_cbgs or {CBG_IN: 10}),
    }


def _papdata(placekeys, n_people=400, n_homes=20):
    return {
        "people": {str(p): {"home": f"home-{p % n_homes}"} for p in range(n_people)},
        "homes": {f"home-{h}": {"cbg": CBG_IN} for h in range(n_homes)},
        "places": {str(i): {"placekey": pk} for i, pk in enumerate(placekeys)},
    }


def _peak_occ(out):
    peak = {}
    for snap in out.values():
        for pid, occ in snap.get("places", {}).items():
            peak[pid] = max(peak.get(pid, 0), len(occ))
    return peak


def _location_of(snapshot, person_id):
    person_id = str(person_id)
    for home_id, people in snapshot.get("homes", {}).items():
        if person_id in [str(pid) for pid in people]:
            return "homes", home_id
    for place_id, people in snapshot.get("places", {}).items():
        if person_id in [str(pid) for pid in people]:
            return "places", place_id
    return "unknown", "unknown"


# --- fail loudly, no silent fallback ----------------------------------------

def test_missing_patterns_data_fails_loudly():
    pap = _papdata(["pk-0"])
    with pytest.raises(ValueError):
        gen_patterns(pap, datetime(2021, 1, 4, 0), duration=1, shared_data=None)

    class _Empty:
        def is_empty(self):
            return True

        def for_patterns_stats(self, pks):
            return pd.DataFrame()

    with pytest.raises(ValueError):
        gen_patterns(pap, datetime(2021, 1, 4, 0), duration=1, shared_data=_Empty())


def test_zone_with_no_matching_pois_fails_loudly():
    # non-empty patterns data, but no POI whose placekey matches the zone's places
    shared = _shared([_poi("other", [10] * 24)])
    pap = _papdata(["unmatched-pk"])
    with pytest.raises(ValueError):
        gen_patterns(pap, datetime(2021, 1, 4, 0), duration=1, shared_data=shared)


# --- a night-active, open-at-night POI is filled at night ---------------------

def test_night_active_poi_fills_at_night(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    h = [0] * 24
    h[22] = h[23] = 120                          # busy late at night
    shared = _shared([_poi("late", h, open_hours={a: [["0:00", "24:00"]] for a in ABBR})])
    out = gen_patterns(_papdata(["late"]), datetime(2021, 1, 4, 22),  # start 10pm
                       duration=2, shared_data=shared)
    assert _peak_occ(out).get("0", 0) > 50


# --- absolute volume: a 2-visit noise POI stays ~empty, a busy one fills ------

def test_noise_poi_stays_empty_busy_poi_fills(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    noise_h = [0] * 24
    noise_h[3] = 1                               # one 3am blip all month
    busy_h = [0] * 24
    for h in range(8, 18):
        busy_h[h] = 120
    shared = _shared([
        _poi("noise", noise_h, naics="621210"),
        _poi("busy", busy_h, open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
    ])
    out = gen_patterns(_papdata(["noise", "busy"]), datetime(2021, 1, 4, 0),
                       duration=24, shared_data=shared)
    peak = _peak_occ(out)
    assert peak.get("1", 0) > 50          # busy POI filled toward its target
    assert peak.get("0", 0) <= 2          # noise POI essentially empty


# --- open-hours hard gate: no one is placed during known-closed hours ---------

def test_open_hours_gate_blocks_closed_hours(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    h = [50] * 24                               # data claims activity every hour
    shared = _shared([_poi("shop", h,
                           open_hours={a: [["8:00", "17:00"]] for a in ["Mon", "Tue", "Wed", "Thu", "Fri"]})])
    out = gen_patterns(_papdata(["shop"]), datetime(2021, 1, 4, 3),  # start at 3am Monday
                       duration=1, shared_data=shared)
    # the only snapshot is the 3am hour -> shop is closed -> nobody there
    assert _peak_occ(out).get("0", 0) == 0


# --- catchment: a mostly-external POI draws fewer of our residents ------------

def test_catchment_downscales_external_poi(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    h = [0] * 24
    for k in range(8, 18):
        h[k] = 100
    shared = _shared([
        _poi("local", list(h), home_cbgs={CBG_IN: 100}),               # f_j = 1.0
        _poi("external", list(h), home_cbgs={CBG_IN: 10, CBG_OUT: 90}),  # f_j = 0.1
    ])
    out = gen_patterns(_papdata(["local", "external"]), datetime(2021, 1, 4, 0),
                       duration=24, shared_data=shared)
    peak = _peak_occ(out)
    assert peak.get("0", 0) > 5 * peak.get("1", 1)   # local >> external


# --- released visitors spend a full snapshot at home -------------------------

def test_expired_generic_visitor_is_not_reselected_in_same_timestep(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    h = [0] * 24
    h[0] = h[1] = h[2] = h[3] = 100
    shared = _shared([
        _poi("shop", h, dwell=1,
             open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
    ])

    out = gen_patterns(_papdata(["shop"], n_people=1, n_homes=1),
                       datetime(2021, 1, 4, 0), 4, shared_data=shared)

    locations = [_location_of(out[str(minute)], "0")
                 for minute in (60, 120, 180, 240)]
    assert locations[0] == ("places", "0")
    first_home = locations.index(("homes", "home-0"))
    assert locations[first_home + 1] == ("places", "0")


# --- workers use their persistent assigned workplace -------------------------

def test_assigned_worker_goes_to_work_poi_on_weekday(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    h = [0] * 24
    pap = _papdata(["work", "other"], n_people=3, n_homes=1)
    pap["people"]["0"].update({
        "is_worker": True,
        "work_location_type": "poi",
        "work_poi": "0",
        "work_p_inside": 1.0,
    })
    pap["people"]["1"].update({
        "is_worker": True,
        "work_location_type": "out_of_zone",
        "work_poi": None,
        "work_p_inside": 0.0,
    })
    shared = _shared([
        _poi("work", h, open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
        _poi("other", h, open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
    ])

    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 24, shared_data=shared)

    assert _location_of(out["480"], "0") == ("homes", "home-0")
    assert _location_of(out["540"], "0") == ("places", "0")
    assert _location_of(out["960"], "0") == ("places", "0")
    assert _location_of(out["1020"], "0") == ("homes", "home-0")
    assert _location_of(out["540"], "1") == ("places", "2")
    assert pap["places"]["2"]["label"] == "Out of Zone Work"
    assert pap["places"]["2"]["external_location_type"] == "out_of_zone_work"


def test_worker_released_at_17_cannot_be_pulled_until_next_timestep(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    work_h = [0] * 24
    other_h = [0] * 24
    other_h[16] = 100
    other_h[17] = 100
    pap = _papdata(["work", "other"], n_people=1, n_homes=1)
    pap["people"]["0"].update({
        "is_worker": True,
        "work_location_type": "poi",
        "work_poi": "0",
        "work_p_inside": 1.0,
    })
    shared = _shared([
        _poi("work", work_h, open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
        _poi("other", other_h, open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
    ])

    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 18, shared_data=shared)

    assert _location_of(out["960"], "0") == ("places", "0")
    assert _location_of(out["1020"], "0") == ("homes", "home-0")
    assert _location_of(out["1080"], "0") == ("places", "1")


def test_release_cooldown_does_not_backfill_from_other_residents(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    work_h = [0] * 24
    other_h = [0] * 24
    other_h[16] = 100
    pap = _papdata(["work", "other"], n_people=200, n_homes=10)
    for person_id in range(100):
        pap["people"][str(person_id)].update({
            "is_worker": True,
            "work_location_type": "poi",
            "work_poi": "0",
            "work_p_inside": 1.0,
        })
    shared = _shared([
        _poi("work", work_h,
             open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
        _poi("other", other_h,
             open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
    ])

    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 17, shared_data=shared)

    visitors = out["1020"].get("places", {}).get("1", [])
    assert 0 < len(visitors) < 100
    assert all(int(person_id) >= 100 for person_id in visitors)


# --- students use their persistent assigned school ---------------------------

def test_assigned_student_goes_to_school_on_weekday(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    school_h = [0] * 24
    other_h = [0] * 24
    pap = _papdata(["school", "other"], n_people=2, n_homes=1)
    pap["people"]["0"].update({
        "is_student": True,
        "school_location_type": "poi",
        "school_poi": "0",
    })
    pap["people"]["1"].update({
        "is_student": True,
        "school_location_type": "out_of_zone",
        "school_poi": None,
    })
    shared = _shared([
        _poi("school", school_h, naics="611110", open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
        _poi("other", other_h, open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
    ])

    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 16, shared_data=shared)

    assert _location_of(out["420"], "0") == ("homes", "home-0")
    assert _location_of(out["480"], "0") == ("places", "0")
    assert _location_of(out["840"], "0") == ("places", "0")
    assert _location_of(out["900"], "0") == ("homes", "home-0")
    assert _location_of(out["960"], "0") == ("homes", "home-0")

    assert _location_of(out["480"], "1") == ("places", "2")
    assert pap["places"]["2"]["label"] == "Out of Zone School"
    assert pap["places"]["2"]["external_location_type"] == "out_of_zone_school"


def test_student_dismissed_at_15_cannot_be_pulled_until_next_timestep(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    school_h = [0] * 24
    other_h = [0] * 24
    other_h[14] = other_h[15] = 100
    pap = _papdata(["school", "other"], n_people=1, n_homes=1)
    pap["people"]["0"].update({
        "is_student": True,
        "school_location_type": "poi",
        "school_poi": "0",
    })
    shared = _shared([
        _poi("school", school_h, naics="611110",
             open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
        _poi("other", other_h,
             open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
    ])

    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 16, shared_data=shared)

    assert _location_of(out["840"], "0") == ("places", "0")
    assert _location_of(out["900"], "0") == ("homes", "home-0")
    assert _location_of(out["960"], "0") == ("places", "1")


@pytest.mark.parametrize(
    ("start_hour", "assignment", "scheduled_place"),
    [
        (7, {
            "is_worker": True,
            "work_location_type": "poi",
            "work_poi": "0",
            "work_p_inside": 1.0,
        }, "work"),
        (6, {
            "is_student": True,
            "school_location_type": "poi",
            "school_poi": "0",
        }, "school"),
    ],
    ids=["worker", "student"],
)
def test_schedule_overrides_generic_expiry_cooldown(
        monkeypatch, start_hour, assignment, scheduled_place):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    assigned_h = [0] * 24
    generic_h = [0] * 24
    generic_h[start_hour] = 100
    pap = _papdata([scheduled_place, "other"], n_people=1, n_homes=1)
    pap["people"]["0"].update(assignment)
    shared = _shared([
        _poi(scheduled_place, assigned_h,
             naics="611110" if scheduled_place == "school" else "445110",
             open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
        _poi("other", generic_h, dwell=1,
             open_hours={a: [["0:00", "24:00"]] for a in ABBR}),
    ])

    out = gen_patterns(pap, datetime(2021, 1, 4, start_hour), 2,
                       shared_data=shared)

    assert _location_of(out["60"], "0") == ("places", "1")
    assert _location_of(out["120"], "0") == ("places", "0")


# --- determinism + bounded occupancy (no dwell-driven ballooning) -------------

def test_deterministic_and_bounded(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    h = [0] * 24
    for k in range(8, 18):
        h[k] = 80
    # a long-dwell POI: occupancy must track the TARGET (~80), not arrivals*dwell.
    shared = _shared([_poi("hotel", h, dwell=480,
                           open_hours={a: [["0:00", "24:00"]] for a in ABBR})])
    pap = _papdata(["hotel"])
    a = _peak_occ(gen_patterns(pap, datetime(2021, 1, 4, 0), 24, shared_data=_shared([_poi("hotel", h, dwell=480, open_hours={x: [["0:00", "24:00"]] for x in ABBR})])))
    b = _peak_occ(gen_patterns(pap, datetime(2021, 1, 4, 0), 24, shared_data=shared))
    assert a == b                                   # deterministic
    assert b.get("0", 0) <= 100                     # bounded near target ~80, not 80*8h


# --- helpers ------------------------------------------------------------------

def test_open_gate_hard_vs_soft():
    gate = _build_open_gate(json.dumps({"Mon": [["8:00", "17:00"]]}), "445110")
    # LISTED day -> hard gate (3am impossible)
    assert gate["Monday"][10] == 1.0 and gate["Monday"][3] == 0.0
    # UNLISTED day (Tuesday) -> NAICS-default soft, NOT a hard zero (don't delete
    # the POI on a day SafeGraph merely omitted; day_factor handles real closures)
    assert gate["Tuesday"][10] == 1.0 and gate["Tuesday"][3] > 0.0
    # no open_hours at all -> NAICS-default soft row for every day
    soft = _build_open_gate(None, "445110")
    assert soft["Monday"][10] == 1.0 and 0.0 < soft["Monday"][3] < 0.01


def test_catchment_fraction():
    cl = {CBG_IN}
    assert _catchment_fraction(json.dumps({CBG_IN: 30, CBG_OUT: 70}), cl) == pytest.approx(0.3)
    assert _catchment_fraction(None, cl) is None        # missing -> caller uses fallback
