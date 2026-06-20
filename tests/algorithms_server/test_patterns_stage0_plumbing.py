"""Stage 0 of the movement-model redesign (docs/MOVEMENT_MODEL_REDESIGN.md).

The patterns loader now surfaces the redesign input columns (absolute visit
volume, observed home-CBG catchment, open hours, category) and reports per-field
coverage at load time. Stage 0 is PLUMBING ONLY: gen_patterns still consumes the
legacy four columns, so its output must be byte-identical.

The golden-hash test pins gen_patterns' output on a fixed, seeded fixture. A
later stage that intentionally changes movement will update GOLDEN_HASH in the
same commit that changes the behavior.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Import the patterns modules from THIS checkout's server dir, so the test is
# robust to the conftest path hack and to running inside a git worktree.
_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

from patterns import gen_patterns  # noqa: E402
from patterns_loader import (  # noqa: E402
    ALL_NEEDED_COLUMNS,
    COVERAGE_FIELDS,
    PATTERNS_STATS_COLUMNS,
    PatternsData,
)

WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LEGACY_FIELDS = {"placekey", "median_dwell", "popularity_by_hour", "popularity_by_day"}
REDESIGN_FIELDS = {
    "raw_visit_counts", "raw_visitor_counts", "normalized_visits_by_state_scaling",
    "visitor_home_cbgs", "open_hours", "naics_code",
}

# gen_patterns output hash on the fixture below, captured from origin/main
# (pre-redesign). Stage 0 is plumbing-only and MUST keep this identical.
GOLDEN_HASH = "fe83aa1a904fad76b1d785f52eb7bb1dca6de0a844c0264d4874a62d32b4a2fe"


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
            # redesign inputs gen_patterns must ignore in Stage 0:
            "raw_visit_counts": 100 * (i + 1),
            "raw_visitor_counts": 40 * (i + 1),
            "normalized_visits_by_state_scaling": 1970 * (i + 1),
            "visitor_home_cbgs": json.dumps({"400010001001": 10 * (i + 1)}),
            "open_hours": json.dumps({"Mon": [["8:00", "18:00"]]}),
            "naics_code": "621210",
        })
    return PatternsData(pd.DataFrame(rows))


def _build_papdata():
    return {
        "people": {str(p): {"home": f"home-{p % 20}"} for p in range(300)},
        "homes": {f"home-{h}": {"cbg": "400010001001"} for h in range(20)},
        "places": {str(i): {"placekey": f"pk-{i}"} for i in range(5)},
    }


def test_redesign_columns_are_loaded_and_projected():
    assert REDESIGN_FIELDS.issubset(set(ALL_NEEDED_COLUMNS))
    assert REDESIGN_FIELDS.issubset(set(PATTERNS_STATS_COLUMNS))
    # legacy columns are untouched
    assert LEGACY_FIELDS.issubset(set(PATTERNS_STATS_COLUMNS))


def test_for_patterns_stats_surfaces_redesign_columns():
    shared = _build_shared()
    cols = set(shared.for_patterns_stats({f"pk-{i}" for i in range(5)}).columns)
    assert LEGACY_FIELDS.issubset(cols)
    assert REDESIGN_FIELDS.issubset(cols)


def test_field_coverage_reports_expected_fields():
    cov = _build_shared().field_coverage()
    assert set(cov) == set(COVERAGE_FIELDS)
    assert cov["visitor_home_cbgs"] == 100.0  # present on every fixture row
    assert cov["wkt_area_sq_meters"] == 0.0   # absent from the fixture frame
    # empty frame -> all zero, no crash
    assert PatternsData(pd.DataFrame()).field_coverage() == {c: 0.0 for c in COVERAGE_FIELDS}


def test_gen_patterns_output_unchanged_golden():
    out = gen_patterns(_build_papdata(), datetime(2021, 1, 4, 0), duration=48,
                       shared_data=_build_shared())
    digest = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    assert digest == GOLDEN_HASH, (
        "gen_patterns output changed — Stage 0 must be behavior-neutral. If a "
        "later stage intentionally changed movement, regenerate GOLDEN_HASH."
    )
