"""CensusDataPuller hardening: configurable API key + fail-loud census fetch.

Guards two popgen changes:
  - the Census API key is no longer a hardcoded constructor default; it resolves
    explicit arg -> CENSUS_API_KEY env var -> CENSUS_API_KEY_DEFAULT.
  - pull_counties_census_data re-raises on fetch failure instead of silently
    returning {}, which previously fed a degenerate population into gen_pop.
"""
import sys
from pathlib import Path

# Mirror test_popgen_catchment_fj: force this checkout's server/ to the front and
# evict any popgen the shared conftest cached from a different checkout.
_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
for _m in ("patterns", "patterns_loader", "popgen"):
    sys.modules.pop(_m, None)

import pytest  # noqa: E402

from popgen import CENSUS_API_KEY_DEFAULT, CensusDataPuller  # noqa: E402


def test_api_key_defaults_to_module_constant(monkeypatch):
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    assert CensusDataPuller().api_key == CENSUS_API_KEY_DEFAULT


def test_api_key_reads_env_var(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "env-key-123")
    assert CensusDataPuller().api_key == "env-key-123"


def test_explicit_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "env-key-123")
    assert CensusDataPuller(api_key="explicit-key").api_key == "explicit-key"


def test_pull_counties_reraises_on_fetch_failure(monkeypatch):
    puller = CensusDataPuller(api_key="x")

    def boom(*args, **kwargs):
        raise RuntimeError("census API down")

    monkeypatch.setattr(puller, "fetch_census_data", boom)
    # Must propagate, not return {} — an empty census dict silently degenerates
    # the population downstream in gen_pop.
    with pytest.raises(RuntimeError, match="census API down"):
        puller.pull_counties_census_data("40", ["001"], None)
