"""Registry-consistency guards for the clustering-algorithm name lists.

There are three places an algorithm name has to line up, and historically they
drift independently — an algorithm gets added to one list but not another, so it
is "valid" in one layer and a 500/ValueError in the next. The three layers:

  * ``VALID_ALGORITHMS`` (czcode_modules.algorithm_runner) — the canonical set of
    names the runner advertises as valid.
  * the ``AlgorithmRunner._handlers()`` dispatch map — the names the runner can
    actually execute.
  * ``VALID_CLUSTER_ALGORITHMS`` (server_app.constants) — the HTTP gate; the
    subset of algorithms reachable through the ``/cluster-cbgs`` route.

The runner list and the dispatch map must be *exactly* equal: a name advertised
without a handler raises ValueError at run time, and a handler with no advertised
name is unreachable. The HTTP gate is modeled as a *subset* of the runner set —
it may deliberately withhold algorithms from the ``/cluster-cbgs`` route. Today
it withholds none: ``hierarchical_core_satellites`` was retired end-to-end, so
the gate and the runner set are currently identical. We still assert subset +
pin the exclusion set (now empty) rather than hard-coding full equality, so a new
runner algorithm nobody wired into the HTTP gate — or a future intentional
exclusion that isn't recorded here — fails this test and forces a conscious
decision.
"""
from czcode_modules.algorithm_runner import VALID_ALGORITHMS, AlgorithmRunner
from server_app.constants import VALID_CLUSTER_ALGORITHMS


def _dispatch_keys():
    # _handlers() only binds self._run_* methods; it touches none of the
    # constructor deps, so placeholder None args are enough to read the map.
    runner = AlgorithmRunner(None, None, None, None, None, None, None)
    return tuple(runner._handlers().keys())


# Algorithms the runner can dispatch but the HTTP /cluster-cbgs gate
# deliberately withholds. Currently none — hierarchical_core_satellites was
# retired, so the gate equals the runner set. Re-populate if an algorithm is
# ever gated out of the route again (and keep it in sync with the route's
# rejection behavior).
HTTP_GATE_EXCLUDED: set[str] = set()


def test_valid_algorithms_has_no_duplicates():
    assert len(VALID_ALGORITHMS) == len(set(VALID_ALGORITHMS))


def test_dispatch_handlers_have_no_duplicates():
    keys = _dispatch_keys()
    assert len(keys) == len(set(keys))


def test_advertised_algorithms_match_dispatch_handlers():
    # Exact equality: every advertised name is dispatchable and every handler is
    # advertised. This is the core "valid in one list, missing in the other" guard.
    assert set(VALID_ALGORITHMS) == set(_dispatch_keys())


def test_http_gate_is_a_subset_of_runner_algorithms():
    # The HTTP gate must never expose a name the runner can't dispatch (that would
    # be a 500 instead of a clean 400 or a real run).
    assert set(VALID_CLUSTER_ALGORITHMS) <= set(VALID_ALGORITHMS)


def test_http_gate_excludes_exactly_the_known_algorithms():
    # Pin the gate/runner relationship so adding a runner algorithm without wiring
    # it into the HTTP gate (or adding an exclusion without recording it here)
    # trips this test.
    assert set(VALID_ALGORITHMS) - set(VALID_CLUSTER_ALGORITHMS) == HTTP_GATE_EXCLUDED
