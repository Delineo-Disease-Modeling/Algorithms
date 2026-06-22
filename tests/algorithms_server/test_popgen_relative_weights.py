"""Relative-type sampling weights in popgen (SyntheticPopulationGenerator).

Guards a former operator-precedence bug in ``generate_household`` where the
``parent`` weight was ``parent + parent_in_law / total`` instead of
``(parent + parent_in_law)``. Because ``random.choices`` takes raw (unnormalized)
weights, the un-divided ``parent`` count (hundreds–thousands) dwarfed the other
buckets (fractions < 1), so siblings, grandchildren and other relatives were
almost never drawn. The fix combines parent + parent-in-law into one bucket and
lets ``random.choices`` normalize the raw counts.
"""
import sys
from pathlib import Path

# conftest.py hard-codes (and chdirs to) the MAIN Algorithms checkout, so force
# the local checkout's server/ to the front of the path and evict any popgen the
# shared conftest cached from a different checkout (mirrors test_popgen_catchment_fj).
_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
for _m in ("patterns", "patterns_loader", "popgen"):
    sys.modules.pop(_m, None)

import random  # noqa: E402

from popgen import SyntheticPopulationGenerator  # noqa: E402

# A county where 'parent' is a minority of relatives and siblings/other-relatives
# dominate. Under the precedence bug the parent bucket would still win ~every draw.
COUNTY = {
    "parent": 10,
    "parent-in-law": 5,
    "brother_or_sister": 30,
    "grandchild": 20,
    "other_relative": 35,
    "son-in-law or daughter-in-law": 7,
}
CHOICES = ["parent", "sibling", "grandchild", "other_relative"]


def test_parent_bucket_combines_in_law_counts():
    # parent bucket = parent + parent-in-law = 15; others are their raw counts.
    assert SyntheticPopulationGenerator._relative_type_weights(COUNTY) == [15, 30, 20, 35]


def test_weights_are_proportional_not_a_dominating_parent():
    w = SyntheticPopulationGenerator._relative_type_weights(COUNTY)
    # Regression guard: the parent bucket must be a normal proportional weight,
    # never the near-certain winner the precedence bug produced.
    assert w[0] < w[1]                                   # parents (15) < siblings (30)
    assert w[0] < w[3]                                   # parents (15) < other (35)
    assert max(range(len(w)), key=lambda i: w[i]) == 3   # 'other_relative' most likely


def test_sampling_distribution_reflects_census_counts():
    random.seed(1234)
    w = SyntheticPopulationGenerator._relative_type_weights(COUNTY)
    total = float(sum(w))
    n = 40_000
    draws = random.choices(CHOICES, weights=w, k=n)
    freq = {c: draws.count(c) / n for c in CHOICES}
    expected = {c: wi / total for c, wi in zip(CHOICES, w)}
    for c in CHOICES:
        assert abs(freq[c] - expected[c]) < 0.02, (c, freq[c], expected[c])
    # The bug's signature was parent dominating (~>0.9). It should now sit near 0.15.
    assert freq["parent"] < 0.25
