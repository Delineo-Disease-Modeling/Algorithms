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
    PATTERNS_STATS_COLUMNS,
    PatternsData,
)

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
BASE_FIELDS = {"placekey", "median_dwell", "popularity_by_hour", "popularity_by_day"}
REDESIGN_FIELDS = {
    "raw_visit_counts", "raw_visitor_counts", "normalized_visits_by_state_scaling",
    "visitor_home_cbgs", "open_hours", "naics_code",
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
        })
    return PatternsData(pd.DataFrame(rows))


def test_redesign_columns_are_loaded_and_projected():
    assert REDESIGN_FIELDS.issubset(set(ALL_NEEDED_COLUMNS))
    assert REDESIGN_FIELDS.issubset(set(PATTERNS_STATS_COLUMNS))
    assert BASE_FIELDS.issubset(set(PATTERNS_STATS_COLUMNS))


def test_for_patterns_stats_surfaces_redesign_columns():
    shared = _build_shared()
    cols = set(shared.for_patterns_stats({f"pk-{i}" for i in range(5)}).columns)
    assert BASE_FIELDS.issubset(cols)
    assert REDESIGN_FIELDS.issubset(cols)


def test_field_coverage_reports_expected_fields():
    cov = _build_shared().field_coverage()
    assert set(cov) == set(COVERAGE_FIELDS)
    assert cov["visitor_home_cbgs"] == 100.0  # present on every fixture row
    assert cov["wkt_area_sq_meters"] == 0.0   # absent from the fixture frame
    # empty frame -> all zero, no crash
    assert PatternsData(pd.DataFrame()).field_coverage() == {c: 0.0 for c in COVERAGE_FIELDS}
