"""Category-relative dwell modeling for generic POI visitors."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
sys.modules.pop("dwell", None)

from dwell import (  # noqa: E402
    build_dwell_reference,
    build_dwell_models,
    parse_dwell_bucket_counts,
    sample_dwell_hours,
)


PRIOR = np.array([0.0, 0.8, 0.2, 0.0, 0.0])
FOCAL = np.array([0.0, 0.0, 0.0, 1.0, 0.0])


def _bucket_json(probabilities, total=1000):
    counts = np.asarray(probabilities, dtype=float) * total
    return json.dumps({
        label: float(count)
        for label, count in zip(("<5", "5-20", "21-60", "61-240", ">240"), counts)
    })


def _row(placekey, naics="523930", probabilities=PRIOR, *, median=35,
         visitors=100, visits=120, polygon_class="OWNED_POLYGON", parent=None,
         enclosed="false", distance=1000, region="OK", closed_on=None,
         date_range_start="2021-04-01", **extra):
    row = {
        "placekey": placekey,
        "naics_code": naics,
        "median_dwell": median,
        "bucketed_dwell_times": _bucket_json(probabilities),
        "raw_visitor_counts": visitors,
        "raw_visit_counts": visits,
        "polygon_class": polygon_class,
        "parent_placekey": parent,
        "enclosed": enclosed,
        "distance_from_home": distance,
        "region": region,
        "closed_on": closed_on,
        "date_range_start": date_range_start,
    }
    row.update(extra)
    return row


def _peers(count=30, naics="523930", prefix="peer", probabilities=PRIOR,
           **row_kwargs):
    return [
        _row(f"{prefix}-{index}", naics, probabilities, **row_kwargs)
        for index in range(count)
    ]


def _model(rows, placekey="focal"):
    return build_dwell_models(pd.DataFrame(rows), {placekey})[placekey]


def test_parse_real_double_encoded_and_legacy_buckets():
    modern = {"<5": 8, "5-20": 0, "21-60": 1, "61-240": 5, ">240": 174}
    parsed = parse_dwell_bucket_counts(json.dumps(json.dumps(modern)))
    assert parsed.tolist() == [8, 0, 1, 5, 174]

    legacy = {
        "<5": 1, "5-10": 2, "11-20": 3, "21-60": 4,
        "61-120": 5, "121-240": 6, ">240": 7,
    }
    assert parse_dwell_bucket_counts(legacy).tolist() == [1, 5, 4, 11, 7]
    assert parse_dwell_bucket_counts({"<5": 0, "5-20": 0}) is None
    assert parse_dwell_bucket_counts({"<5": -1, "5-20": 2}) is None
    assert parse_dwell_bucket_counts("not-json") is None


@pytest.mark.parametrize(
    ("visitors", "expected_weight", "expected"),
    [
        (5, 0.2, np.array([0.0, 0.64, 0.16, 0.20, 0.0])),
        (200, 10 / 11, np.array([0.0, 0.8 / 11, 0.2 / 11, 10 / 11, 0.0])),
        (0, 0.0, PRIOR),
        (None, 0.0, PRIOR),
        (np.nan, 0.0, PRIOR),
    ],
)
def test_low_sample_shrinkage_uses_unique_devices(visitors, expected_weight, expected):
    rows = _peers() + [
        _row("focal", probabilities=FOCAL, visitors=visitors, visits=400)
    ]
    model = _model(rows)
    assert model.hard_fallback is False
    assert model.poi_weight == pytest.approx(expected_weight)
    assert model.probabilities == pytest.approx(expected)


@pytest.mark.parametrize(
    ("rows", "expected_level", "expected_count"),
    [
        (
            _peers(30, "523930") + [_row("focal", "523930")],
            "naics6", 30,
        ),
        (
            _peers(29, "523930")
            + _peers(2, "523999", prefix="four")
            + [_row("focal", "523930")],
            "naics4", 31,
        ),
        (
            _peers(29, "523930")
            + _peers(1, "524100", prefix="sector")
            + [_row("focal", "523930")],
            "naics2", 30,
        ),
    ],
)
def test_prior_hierarchy_counts_other_peer_pois(rows, expected_level, expected_count):
    model = _model(rows)
    assert model.category_level == expected_level
    assert model.peer_count == expected_count


def test_compact_reference_keeps_statewide_peers_after_zone_filtering():
    full = pd.DataFrame(_peers() + [_row("focal", probabilities=FOCAL, visitors=5)])
    reference = build_dwell_reference(full)
    zone = full[full["placekey"] == "focal"]
    model = build_dwell_models(zone, {"focal"}, reference=reference)["focal"]
    assert model.category_level == "naics6"
    assert model.peer_count == 30
    assert model.probabilities == pytest.approx(
        np.array([0.0, 0.64, 0.16, 0.20, 0.0])
    )


def test_category_reference_does_not_mix_states():
    rows = (
        _peers(29, region="OK")
        + [_row(f"texas-{index}", region="TX") for index in range(30)]
        + [_row("focal", region="OK")]
    )
    model = _model(rows)
    assert model.category_level is None
    assert model.peer_count == 0


def test_gowin_two_signal_outlier_falls_back_without_contaminating_prior():
    gowin_counts = np.array([8, 0, 1, 5, 174], dtype=float)
    rows = _peers() + [
        _row(
            "focal",
            probabilities=gowin_counts / gowin_counts.sum(),
            median=1327,
            visitors=8,
            visits=188,
            distance=14,
        )
    ]
    model = _model(rows)
    assert model.hard_fallback is True
    assert "statistical_outlier" in model.fallback_reasons
    assert "home_like_repeat_visitors" in model.fallback_reasons
    assert model.probabilities == pytest.approx(PRIOR)
    assert model.poi_weight == 0.0


@pytest.mark.parametrize(
    "focal_row",
    [
        _row("focal", probabilities=PRIOR, median=1327, visitors=200),
        _row(
            "focal",
            probabilities=np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            median=35,
            visitors=200,
        ),
    ],
    ids=["median-only", "shape-only"],
)
def test_statistical_fallback_requires_two_of_three_signals(focal_row):
    model = _model(_peers() + [focal_row])
    assert model.hard_fallback is False
    assert "statistical_outlier" not in model.fallback_reasons


def test_long_share_plus_shape_is_a_two_signal_statistical_fallback():
    focal = _row(
        "focal",
        probabilities=np.array([0.0, 0.0, 0.0, 0.02, 0.98]),
        median=35,
        visitors=200,
    )
    model = _model(_peers() + [focal])
    assert model.hard_fallback is True
    assert "statistical_outlier" in model.fallback_reasons


def test_common_secondary_category_mode_is_not_treated_as_one_percent_tail():
    short_mode = _peers(40, probabilities=PRIOR)
    long_mode_probabilities = np.array([0.0, 0.0, 0.1, 0.4, 0.5])
    long_mode = _peers(
        11,
        prefix="long",
        probabilities=long_mode_probabilities,
        median=300,
    )
    long_mode[-1]["placekey"] = "focal"

    model = _model(short_mode + long_mode)
    assert model.hard_fallback is False
    assert "statistical_outlier" not in model.fallback_reasons


def test_shared_polygon_child_forces_prior_without_statistical_outlier():
    rows = _peers(30, "722513") + [
        _row(
            "focal",
            "722513",
            probabilities=np.array([0.0, 0.7, 0.3, 0.0, 0.0]),
            median=35,
            visitors=400,
            visits=5000,
            polygon_class="SHARED_POLYGON",
            parent="host-placekey",
            enclosed="true",
        )
    ]
    model = _model(rows)
    assert model.hard_fallback is True
    assert model.fallback_reasons == ("shared_child",)
    assert model.probabilities == pytest.approx(PRIOR)


@pytest.mark.parametrize(
    ("polygon_class", "parent"),
    [("SHARED_POLYGON", None), ("OWNED_POLYGON", "host-placekey")],
)
def test_shared_polygon_or_parent_alone_does_not_force_fallback(polygon_class, parent):
    rows = _peers(30, "722513") + [
        _row(
            "focal", "722513",
            probabilities=np.array([0.0, 0.7, 0.3, 0.0, 0.0]),
            visitors=400,
            polygon_class=polygon_class,
            parent=parent,
        )
    ]
    model = _model(rows)
    assert model.hard_fallback is False
    assert model.probabilities != pytest.approx(PRIOR)


def test_enclosed_child_forces_fallback_even_without_shared_polygon():
    focal = _row(
        "focal", "722513",
        probabilities=np.array([0.0, 0.7, 0.3, 0.0, 0.0]),
        polygon_class="OWNED_POLYGON",
        parent="host-placekey",
        enclosed="true",
    )
    model = _model(_peers(30, "722513") + [focal])
    assert model.hard_fallback is True
    assert model.fallback_reasons == ("shared_child",)


def test_closed_before_source_month_forces_category_fallback():
    focal = _row("focal", closed_on="2021-03-15", date_range_start="2021-04-01")
    model = _model(_peers() + [focal])
    assert model.hard_fallback is True
    assert model.fallback_reasons == ("stale_closed",)


def test_low_repeat_frequency_near_home_is_not_a_home_like_signal():
    focal = _row("focal", visitors=100, visits=1, distance=14)
    model = _model(_peers() + [focal])
    assert model.hard_fallback is False
    assert "home_like_repeat_visitors" not in model.fallback_reasons


def test_category_extreme_night_and_weekend_activity_forces_fallback():
    weekday_days = {
        "Monday": 10, "Tuesday": 10, "Wednesday": 10, "Thursday": 10,
        "Friday": 10, "Saturday": 1, "Sunday": 1,
    }
    business_hours = [0] * 24
    for hour in range(9, 17):
        business_hours[hour] = 10
    peers = _peers(
        30,
        popularity_by_hour=business_hours,
        popularity_by_day=weekday_days,
    )

    weekend_days = {
        "Monday": 1, "Tuesday": 1, "Wednesday": 1, "Thursday": 1,
        "Friday": 1, "Saturday": 10, "Sunday": 10,
    }
    night_hours = [0] * 24
    for hour in range(0, 6):
        night_hours[hour] = 10
    focal = _row(
        "focal",
        popularity_by_hour=night_hours,
        popularity_by_day=weekend_days,
    )

    model = _model(peers + [focal])
    assert model.hard_fallback is True
    assert model.fallback_reasons == ("temporal_mismatch",)


def test_sampler_draws_from_finite_buckets_and_decreasing_tail():
    probability_matrix = np.eye(5)
    modeled = np.ones(5, dtype=bool)
    tail_p = np.ones(5)
    legacy = np.ones(5, dtype=np.int64)
    destinations = np.repeat(np.arange(5), 200)

    samples = sample_dwell_hours(
        np.random.default_rng(42), destinations, probability_matrix,
        modeled, tail_p, legacy,
    ).reshape(5, 200)

    assert np.all(samples[0] == 1)
    assert np.all(samples[1] == 1)
    assert np.all(samples[2] == 1)
    assert samples[3].min() >= 2 and samples[3].max() <= 4
    assert np.all(samples[4] == 5)

    tail_samples = sample_dwell_hours(
        np.random.default_rng(7),
        np.zeros(10_000, dtype=np.int64),
        np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]),
        np.array([True]),
        np.array([0.5]),
        np.array([1], dtype=np.int64),
    )
    counts = np.bincount(tail_samples)
    assert counts[5] > counts[6] > counts[7]
    assert tail_samples.max() > 5
