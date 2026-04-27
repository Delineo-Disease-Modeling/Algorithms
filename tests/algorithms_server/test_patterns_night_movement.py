from datetime import datetime

import pandas as pd

from patterns import (
    _aggregate_busyness,
    _compute_peak_busyness,
    _overall_busy_factor,
    gen_patterns,
)


def _stats_with_late_night_activity():
    hour_weights = [0.0] * 24
    hour_weights[23] = 1.0
    raw_hour_counts = [0] * 24
    raw_hour_counts[23] = 50

    return {
        "place-1": {
            "median_dwell_hours": 2,
            "hour_weights": hour_weights,
            "day_weights": {"Monday": 1.0},
            "raw_hour_counts": raw_hour_counts,
            "raw_day_counts": {"Monday": 1},
        }
    }


def test_busy_helpers_include_late_night_hours():
    stats = _stats_with_late_night_activity()

    place_ids, weights = _overall_busy_factor(stats, "Monday", 23)

    assert place_ids == ["place-1"]
    assert weights == [1.0]
    assert _aggregate_busyness(stats, "Monday", 23) == 50
    assert _compute_peak_busyness(stats) == 50


class SharedPatternsData:
    def is_empty(self):
        return False

    def for_patterns_stats(self, placekey_set):
        assert placekey_set == {"late-night-placekey"}
        return pd.DataFrame([
            {
                "placekey": "late-night-placekey",
                "median_dwell": 120,
                "popularity_by_hour": [0] * 22 + [100, 0],
                "popularity_by_day": {"Monday": 100},
            }
        ])


def test_gen_patterns_can_start_new_late_night_trips():
    papdata = {
        "people": {str(i): {"home": "home-1"} for i in range(200)},
        "homes": {"home-1": {}},
        "places": {"place-1": {"placekey": "late-night-placekey"}},
    }

    patterns = gen_patterns(
        papdata,
        datetime(2021, 1, 4, 22),
        duration=1,
        shared_data=SharedPatternsData(),
    )

    assert patterns["60"]["places"].get("place-1")
