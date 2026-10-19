"""Fractional dwell in movement generation (DELINEO_FRACTIONAL_DWELL).

On (default): each visit's length is drawn in minutes from the
POI's bucketed dwell distribution and starts at a random minute of its first
hour; the person is listed at the POI for every hour the visit overlaps, and
hours the visit only partly fills carry the minutes spent there. Off
(DELINEO_FRACTIONAL_DWELL=0): generic visits keep the whole-hour median +/- 1
rule and the output carries no minutes.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
for _m in ("patterns", "patterns_loader", "dwell", "patterns_codec"):
    sys.modules.pop(_m, None)

from dwell import build_dwell_reference, drop_pass_by_bucket, expected_bucket_minutes  # noqa: E402
from patterns import _minutes_or_default, _partial_hour_minutes, gen_patterns  # noqa: E402
from patterns_codec import decode_patterns_binary  # noqa: E402
from server_app.fullstack_client import FullstackClient  # noqa: E402

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CBG_IN = "400010001001"
BUCKETS = ["<5", "5-20", "21-60", "61-240", ">240"]


def _shared(pois):
    df = pd.DataFrame(pois)
    dwell_reference = build_dwell_reference(df)

    class _Stub:
        def is_empty(self):
            return False

        def for_patterns_stats(self, placekeys):
            return df[df["placekey"].isin(placekeys)]

        def for_dwell_reference(self):
            return dwell_reference

    return _Stub()


def _poi(pk, buckets, *, hours=None, naics="713940"):
    """An always-open POI with the given hourly visitor counts and buckets."""
    return {
        "placekey": pk, "median_dwell": 30,
        "popularity_by_hour": hours if hours is not None else [30] * 24,
        "popularity_by_day": {d: 10 for d in WEEK},
        "open_hours": json.dumps({a: [["0:00", "24:00"]] for a in ABBR}),
        "naics_code": naics,
        "visitor_home_cbgs": json.dumps({CBG_IN: 10}),
        "bucketed_dwell_times": json.dumps(dict(zip(BUCKETS, buckets))),
        "raw_visitor_counts": 500, "raw_visit_counts": 500, "region": "OK",
    }


def _world(buckets, n_people=600, n_homes=60, first_hour_only=False):
    """One target POI with ``buckets`` plus 30 identical peers (so the
    empirical-Bayes prior agrees with the target and no shrinkage kicks in).

    ``first_hour_only``: visitors arrive only in the first hour, so each person
    makes at most one visit and consecutive listed hours are exactly a visit.
    The output records minutes per hour, not arrival times, so back-to-back
    visits are otherwise indistinguishable from one long visit."""
    peers = [_poi(f"peer-{i}", buckets, hours=[0] * 24) for i in range(30)]
    hours = [200] + [0] * 23 if first_hour_only else None
    pap = {
        "people": {str(p): {"home": f"home-{p % n_homes}"} for p in range(n_people)},
        "homes": {f"home-{h}": {"cbg": CBG_IN} for h in range(n_homes)},
        "places": {"0": {"placekey": "target"}},
    }
    return pap, _shared(peers + [_poi("target", buckets, hours=hours)])


def _visits(out, place_id="0"):
    """Per person, runs of consecutive hours at the place: list of
    (first_hour_index, [minutes per hour]). Runs touching the last generated
    hour are dropped (they may be cut off by the end of the run). Use with
    first_hour_only worlds, where each run is one visit."""
    keys = sorted(int(k) for k in out)
    last = len(keys) - 1
    runs, open_runs = [], {}
    for h, key in enumerate(keys):
        snap = out[str(key)]
        present = set(snap["places"].get(place_id, []))
        minutes = snap.get("minutes", {})
        for pid in list(open_runs):
            if pid not in present:
                runs.append(open_runs.pop(pid))
        for pid in present:
            open_runs.setdefault(pid, (h, []))[1].append(minutes.get(pid, 60))
    runs.extend(r for r in open_runs.values() if r[0] + len(r[1]) - 1 < last)
    return [r for r in runs if r[0] + len(r[1]) - 1 < last]


def test_on_by_default(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    monkeypatch.delenv("DELINEO_FRACTIONAL_DWELL", raising=False)
    pap, shared = _world([0, 100, 0, 0, 0])
    default = gen_patterns(pap, datetime(2021, 1, 4, 0), 12, shared_data=shared)
    explicit_on = gen_patterns(pap, datetime(2021, 1, 4, 0), 12, shared_data=shared,
                               fractional_dwell=True)
    assert default == explicit_on
    assert any(snap.get("minutes") for snap in default.values())


def test_env_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    monkeypatch.setenv("DELINEO_FRACTIONAL_DWELL", "0")
    pap, shared = _world([0, 100, 0, 0, 0])
    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 12, shared_data=shared)
    explicit_off = gen_patterns(pap, datetime(2021, 1, 4, 0), 12, shared_data=shared,
                                fractional_dwell=False)
    assert out == explicit_off
    assert all("minutes" not in snap for snap in out.values())


@pytest.mark.parametrize("bucket,lo,hi", [(1, 5, 20), (2, 21, 60), (3, 61, 240)])
def test_visit_minutes_match_the_bucket(monkeypatch, bucket, lo, hi):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    buckets = [0, 0, 0, 0, 0]
    buckets[bucket] = 100
    pap, shared = _world(buckets, first_hour_only=True)
    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 24, shared_data=shared,
                       fractional_dwell=True)
    visits = _visits(out)
    assert len(visits) > 100
    for first_hour, minutes in visits:
        assert first_hour == 0
        assert lo <= sum(minutes) <= hi
        assert all(1 <= m <= 60 for m in minutes)
        # Only a visit's first and last hour can be partial.
        assert all(m == 60 for m in minutes[1:-1])


def test_short_visits_are_listed_for_one_or_two_hours(monkeypatch):
    # A 5-20 minute visit starting at a uniform minute crosses into a second
    # hour sometimes, never a third.
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    pap, shared = _world([0, 100, 0, 0, 0], first_hour_only=True)
    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 24, shared_data=shared,
                       fractional_dwell=True)
    lengths = np.bincount([len(m) for _, m in _visits(out)])
    assert lengths[1] > 0 and lengths[2] > 0
    assert len(lengths) == 3


def test_pass_by_visits_are_dropped_and_target_shrinks(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    pap, shared = _world([50, 50, 0, 0, 0], first_hour_only=True)
    on = gen_patterns(pap, datetime(2021, 1, 4, 0), 24, shared_data=shared,
                      fractional_dwell=True)
    # No visit shorter than 5 minutes remains.
    assert min(sum(m) for _, m in _visits(on)) >= 5
    # Each hour's head count shrinks by the dropped bucket's share of the
    # hourly visitor counts (it is ~1 hour-touch per pass-by visit).
    monkeypatch.setenv("DELINEO_DWELL_DROP_PASS_BY", "0")
    kept = gen_patterns(pap, datetime(2021, 1, 4, 0), 24, shared_data=shared,
                        fractional_dwell=True)
    peak_on = max(len(s["places"].get("0", [])) for s in on.values())
    peak_kept = max(len(s["places"].get("0", [])) for s in kept.values())
    assert peak_on < peak_kept
    assert min(sum(m) for _, m in _visits(kept)) < 5


def test_drop_pass_by_bucket_math():
    probs = np.array([[0.5, 0.5, 0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0, 0.0]])
    tail_p = np.array([1.0, 1.0])
    out, keep = drop_pass_by_bucket(probs, tail_p)
    np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0, 0.0, 0.0])
    # Hour-touches per visit: <5 -> 1 + 1.5/60, 5-20 -> 1 + 11.5/60.
    t0, t1 = 1 + 1.5 / 60, 1 + 11.5 / 60
    assert keep[0] == pytest.approx(0.5 * t1 / (0.5 * t0 + 0.5 * t1))
    # A POI whose visits are all <5 minutes is left alone.
    np.testing.assert_allclose(out[1], probs[1])
    assert keep[1] == 1.0
    np.testing.assert_allclose(expected_bucket_minutes(np.array([0.5]))[0],
                               [2.5, 12.5, 40.5, 150.5, 240 + 60 + 30.5])


def test_partial_hour_minutes_add_up_to_the_visit():
    # A 20-minute visit starting at minute 50 of hour 2: 10 + 10 minutes.
    is_home = np.array([False, False, True])
    start = np.array([2 * 60 + 50, -1, -1])
    end = np.array([2 * 60 + 70, -1, -1])
    pids = ["a", "worker", "home"]
    assert _partial_hour_minutes(2, is_home, start, end, pids) == {"a": 10}
    assert _partial_hour_minutes(3, is_home, start, end, pids) == {"a": 10}


def test_scheduled_workers_fill_the_whole_hour(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    pap, shared = _world([0, 100, 0, 0, 0])
    pap["people"]["0"].update({"work_location_type": "poi", "work_poi": "0"})
    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 24, shared_data=shared,
                       fractional_dwell=True)
    # Work runs 9:00-17:00; snapshot keys mark the end of each hour.
    work_hours = [s for k, s in out.items() if 10 * 60 <= int(k) <= 16 * 60]
    assert all("0" in s["places"]["0"] for s in work_hours)
    assert all("0" not in s.get("minutes", {}) for s in work_hours)


def test_binary_upload_carries_the_minutes(monkeypatch):
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    monkeypatch.delenv("DELINEO_PATTERNS_BINARY", raising=False)
    pap, shared = _world([0, 100, 0, 0, 0], n_people=120, n_homes=12)
    pap["people"] = {str(p): v for p, v in pap["people"].items()}
    pap["homes"] = {str(i): v for i, v in enumerate(pap["homes"].values())}
    for person in pap["people"].values():
        person["home"] = str(int(person["home"].split("-")[1]))
    pap["places"] = {"500": {"placekey": "target"}}
    out = gen_patterns(pap, datetime(2021, 1, 4, 0), 6, shared_data=shared,
                       fractional_dwell=True)
    _, body, _ = FullstackClient._patterns_part(pap, out)
    bp = decode_patterns_binary(body.getvalue())
    assert bp.minutes is not None
    assert {k: v.get("minutes") for k, v in bp.items()} == \
        {k: v.get("minutes") for k, v in out.items()}


def test_blank_median_dwell_falls_back_to_an_hour(monkeypatch):
    # Real SafeGraph rows can carry a blank median_dwell; both modes must cope.
    assert _minutes_or_default("") == 60.0
    assert _minutes_or_default(None) == 60.0
    assert _minutes_or_default("nan") == 60.0
    assert _minutes_or_default("0") == 60.0
    assert _minutes_or_default("12.5") == 12.5
    monkeypatch.setenv("DELINEO_MOVEMENT_SCALE", "1")
    pap, shared = _world([0, 100, 0, 0, 0])
    shared.for_patterns_stats({"target"})  # stub sanity
    df = shared.for_patterns_stats({"target"}).copy()
    df["median_dwell"] = ""

    class _Blank:
        def is_empty(self):
            return False

        def for_patterns_stats(self, placekeys):
            return df

        def for_dwell_reference(self):
            return shared.for_dwell_reference()

    for on in (False, True):
        out = gen_patterns(pap, datetime(2021, 1, 4, 0), 3, shared_data=_Blank(),
                           fractional_dwell=on)
        assert len(out) == 3
