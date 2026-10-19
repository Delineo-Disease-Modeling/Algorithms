"""Data plumbing for the movement-model redesign (docs/MOVEMENT_MODEL_REDESIGN.md).

The patterns loader surfaces the redesign input columns (absolute visit volume,
observed home-CBG catchment, open hours, category) and reports per-field coverage
at load time. These tests pin that the columns are loaded/projected and that the
coverage report is well-formed.
"""
import json
import sys
from pathlib import Path

import pandas as pd

# Import the patterns modules from THIS checkout's server dir, so the test is
# robust to the conftest path hack and to running inside a git worktree.
_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
# Evict any patterns_loader the shared conftest cached from a different checkout.
for _m in ("patterns", "patterns_loader"):
    sys.modules.pop(_m, None)

from patterns_loader import (  # noqa: E402
    ALL_NEEDED_COLUMNS,
    COVERAGE_FIELDS,
    DWELL_REFERENCE_COLUMNS,
    PATTERNS_STATS_COLUMNS,
    PatternsData,
)
from dwell import build_dwell_models  # noqa: E402

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
BASE_FIELDS = {"placekey", "median_dwell", "popularity_by_hour", "popularity_by_day"}
REDESIGN_FIELDS = {
    "raw_visit_counts", "raw_visitor_counts", "normalized_visits_by_state_scaling",
    "visitor_home_cbgs", "open_hours", "naics_code",
}
DWELL_QUALITY_FIELDS = {
    "parent_placekey", "polygon_class", "enclosed", "distance_from_home", "region",
    "closed_on", "date_range_start",
}


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
            "parent_placekey": "parent-pk" if i == 0 else None,
            "polygon_class": "SHARED_POLYGON" if i == 0 else "OWNED_POLYGON",
            "enclosed": "true" if i == 0 else "false",
            "distance_from_home": 100 + i,
            "region": "OK",
            "closed_on": None,
            "date_range_start": "2021-04-01",
        })
    return PatternsData(pd.DataFrame(rows))


def test_redesign_columns_are_loaded_and_projected():
    assert REDESIGN_FIELDS.issubset(set(ALL_NEEDED_COLUMNS))
    assert REDESIGN_FIELDS.issubset(set(PATTERNS_STATS_COLUMNS))
    assert BASE_FIELDS.issubset(set(PATTERNS_STATS_COLUMNS))
    assert DWELL_QUALITY_FIELDS.issubset(set(ALL_NEEDED_COLUMNS))
    assert DWELL_QUALITY_FIELDS.issubset(set(PATTERNS_STATS_COLUMNS))
    assert DWELL_QUALITY_FIELDS.issubset(set(DWELL_REFERENCE_COLUMNS))


def test_for_patterns_stats_surfaces_redesign_columns():
    shared = _build_shared()
    cols = set(shared.for_patterns_stats({f"pk-{i}" for i in range(5)}).columns)
    assert BASE_FIELDS.issubset(cols)
    assert REDESIGN_FIELDS.issubset(cols)
    assert DWELL_QUALITY_FIELDS.issubset(cols)


def test_for_patterns_stats_preserves_dwell_quality_values():
    row = _build_shared().for_patterns_stats({"pk-0"}).iloc[0]
    assert row["parent_placekey"] == "parent-pk"
    assert row["polygon_class"] == "SHARED_POLYGON"
    assert row["enclosed"] == "true"
    assert row["distance_from_home"] == 100
    assert row["region"] == "OK"
    assert row["date_range_start"] == "2021-04-01"


def test_field_coverage_reports_expected_fields():
    cov = _build_shared().field_coverage()
    assert set(cov) == set(COVERAGE_FIELDS)
    assert cov["visitor_home_cbgs"] == 100.0  # present on every fixture row
    assert cov["wkt_area_sq_meters"] == 0.0   # absent from the fixture frame
    # empty frame -> all zero, no crash
    assert PatternsData(pd.DataFrame()).field_coverage() == {c: 0.0 for c in COVERAGE_FIELDS}


def test_parquet_loader_keeps_statewide_dwell_peers_after_zone_filter(tmp_path):
    rows = []
    for index in range(31):
        rows.append({
            "PLACEKEY": "focal" if index == 30 else f"peer-{index}",
            "POI_CBG": "400010001001" if index == 30 else "400010009999",
            "REGION": "OK",
            "NAICS_CODE": "523930",
            "MEDIAN_DWELL": 35,
            "BUCKETED_DWELL_TIMES": json.dumps({
                "<5": 0, "5-20": 80, "21-60": 20, "61-240": 0, ">240": 0,
            }),
            "RAW_VISITOR_COUNTS": 100,
            "RAW_VISIT_COUNTS": 120,
            "POPULARITY_BY_HOUR": [0] * 24,
            "POPULARITY_BY_DAY": {day: 1 for day in WEEK},
        })
    state_dir = tmp_path / "OK"
    state_dir.mkdir()
    path = state_dir / "2021-04-OK.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    shared = PatternsData.load([str(path)], cbg_set={"400010001001"})
    assert shared.df["placekey"].tolist() == ["focal"]

    stats = shared.for_patterns_stats({"focal"})
    model = build_dwell_models(
        stats,
        {"focal"},
        reference=shared.for_dwell_reference(),
    )["focal"]
    assert model.category_level == "naics6"
    assert model.peer_count == 30


def test_csv_loader_normalizes_float_like_postal_codes(tmp_path):
    state_dir = tmp_path / "OK"
    state_dir.mkdir()
    path = state_dir / "2021-04-OK.csv"
    pd.DataFrame([
        {
            "PLACEKEY": "matching-poi",
            "POSTAL_CODE": 74003,
            "REGION": "OK",
            "NAICS_CODE": "722513",
        },
        {
            "PLACEKEY": "missing-zip-poi",
            "POSTAL_CODE": None,
            "REGION": "OK",
            "NAICS_CODE": "722513",
        },
    ]).to_csv(path, index=False)

    shared = PatternsData.load([str(path)], zip_codes=[74003])

    assert shared.df["placekey"].tolist() == ["matching-poi"]
