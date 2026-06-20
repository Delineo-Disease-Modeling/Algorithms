"""Stage 1 of the movement-model redesign (docs/MOVEMENT_MODEL_REDESIGN.md).

Destination selection now weights POIs by ABSOLUTE hour-of-day occupancy
(popularity_by_hour) gated by open hours, instead of the self-normalized shape.
A 2-visit/month POI can no longer out-attract a high-traffic one during a quiet
hour, which is the root of the over-occupancy pathology (verified on real
Barnsdall data: "Fooshee D Scott Dntst" peak occupancy 1,210 -> 0).

The mega-POI concentration this exposes (a regional airport over-drawing) is the
expected Stage 1 limitation, addressed by the Stage 2 capacity cap and Stage 3
catchment scaling.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

from patterns import (  # noqa: E402
    _build_stats_from_df,
    _overall_busy_factor,
    gen_patterns,
)

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Default (Stage 1) gen_patterns output hash on the shared fixture below.
# Regenerate together with any intentional change to the movement kernel.
STAGE1_GOLDEN = "c61c86876a189c339dc32e9b4ba0544e5ffe30c2e02bc4fec1b148496e977e19"


def _build_shared():
    rows = []
    for i in range(5):
        h = [0] * 24
        for k in range(6, 20):
            h[k] = (i + 1) * ((k % 5) + 1)
        rows.append({
            "placekey": f"pk-{i}",
            "median_dwell": 30 + 10 * i,
            "popularity_by_hour": h,
            "popularity_by_day": {d: (i + 1) * (j + 1) for j, d in enumerate(WEEK)},
            "raw_visit_counts": 100 * (i + 1),
            "raw_visitor_counts": 40 * (i + 1),
            "normalized_visits_by_state_scaling": 1970 * (i + 1),
            "visitor_home_cbgs": json.dumps({"400010001001": 10 * (i + 1)}),
            "open_hours": json.dumps({"Mon": [["8:00", "18:00"]]}),
            "naics_code": "621210",
        })
    # PatternsData-like stub: only is_empty + for_patterns_stats are used.
    df = pd.DataFrame(rows)

    class _Stub:
        def is_empty(self):
            return False

        def for_patterns_stats(self, placekeys):
            return df[df["placekey"].isin(placekeys)]

    return _Stub()


def _build_papdata():
    return {
        "people": {str(p): {"home": f"home-{p % 20}"} for p in range(300)},
        "homes": {f"home-{h}": {"cbg": "400010001001"} for h in range(20)},
        "places": {str(i): {"placekey": f"pk-{i}"} for i in range(5)},
    }


def test_default_movement_golden():
    out = gen_patterns(_build_papdata(), datetime(2021, 1, 4, 0), duration=48,
                       shared_data=_build_shared())
    digest = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    assert digest == STAGE1_GOLDEN, (
        "Stage 1 default movement output changed — regenerate STAGE1_GOLDEN if "
        "the kernel change was intentional."
    )


def _two_poi_stats():
    """A 2-visit noise POI (single 3am blip, dentist) vs a busy 24h POI."""
    noise_h = [0] * 24
    noise_h[3] = 1
    big_h = [50] * 24
    for h in range(8, 20):
        big_h[h] = 250
    df = pd.DataFrame([
        {"placekey": "noise", "median_dwell": 30, "popularity_by_hour": noise_h,
         "popularity_by_day": {d: 1 for d in WEEK},
         "open_hours": None, "naics_code": "621210"},             # dentist, no hours
        {"placekey": "big", "median_dwell": 30, "popularity_by_hour": big_h,
         "popularity_by_day": {d: 100 for d in WEEK},
         "open_hours": json.dumps({a: [["0:00", "24:00"]] for a in ABBR}),  # 24h
         "naics_code": "445110"},
    ])
    return _build_stats_from_df(df, {"noise": "0", "big": "1"})


def test_absolute_weighting_inverts_noise_dominance():
    stats = _two_poi_stats()

    # Stage 1: at 3am the busy 24h POI dwarfs the 2-visit dentist.
    ids, w = _overall_busy_factor(stats, "Monday", 3, use_absolute=True)
    wd = dict(zip(ids, w))
    assert wd.get("1", 0.0) > 50 * wd.get("0", 0.0)

    # Legacy: the self-normalized shape INVERTS this — the dentist wins at 3am,
    # which is the bug Stage 1 fixes.
    lids, lw = _overall_busy_factor(stats, "Monday", 3, use_absolute=False)
    lwd = dict(zip(lids, lw))
    assert lwd.get("0", 0.0) > lwd.get("1", 0.0)


def test_open_hours_hard_gate_excludes_closed_hours():
    df = pd.DataFrame([{
        "placekey": "shop", "median_dwell": 30,
        "popularity_by_hour": [5] * 24,                    # data claims 3am activity
        "popularity_by_day": {d: 10 for d in WEEK},
        "open_hours": json.dumps({a: [["8:00", "17:00"]] for a in ["Mon", "Tue", "Wed", "Thu", "Fri"]}),
        "naics_code": "445110",
    }])
    stats = _build_stats_from_df(df, {"shop": "0"})

    open_ids, _ = _overall_busy_factor(stats, "Monday", 10, use_absolute=True)
    assert "0" in open_ids                                  # open at 10am

    closed_ids, _ = _overall_busy_factor(stats, "Monday", 3, use_absolute=True)
    assert "0" not in closed_ids                            # hard-gated at 3am
