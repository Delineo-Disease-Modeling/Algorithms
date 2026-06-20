"""Places-bundle catchment fraction f_j (docs/MOVEMENT_MODEL_REDESIGN.md §10, S1).

popgen.convert_data emits a per-POI ``catchment_fj`` into output['places'] (the
bundle the simulator reads in world.parse_facility), so the external-FOI term has
the in-cluster visitor share. f_j uses the SAME definition + median fallback as
gen_patterns (shared helpers), and is floored so (1 - f_j)/f_j can't explode.
"""
import json
import sys
from pathlib import Path

import pandas as pd

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
# Evict any patterns/popgen the shared conftest cached from a different checkout.
for _m in ("patterns", "patterns_loader", "popgen"):
    sys.modules.pop(_m, None)

import popgen  # noqa: E402
from popgen import CATCHMENT_FJ_FLOOR, convert_data  # noqa: E402

CBG_IN = "400010001001"
CBG_IN2 = "400010001002"
CBG_OUT = "999990000001"
META = ["location_name", "top_category", "latitude", "longitude",
        "street_address", "postal_code", "polygon_wkt", "wkt_area_sq_meters",
        "visitor_home_cbgs"]


def _people_df():
    # one person, one household, home CBG in the cluster
    return pd.DataFrame([{
        "person_id": "0", "household_id": "h0", "gender": "M", "age": 30,
        "cbg": CBG_IN, "household_lat": None, "household_lon": None,
    }])


def _place_row(pk, home_cbgs):
    return {
        "placekey": pk, "location_name": pk, "top_category": "shop",
        "latitude": 36.0, "longitude": -96.0, "street_address": None,
        "postal_code": "74000", "polygon_wkt": None, "wkt_area_sq_meters": 500.0,
        "visitor_home_cbgs": None if home_cbgs is None else json.dumps(home_cbgs),
    }


class _Stub:
    def __init__(self, rows):
        self._df = pd.DataFrame(rows, columns=["placekey"] + META)

    def is_empty(self):
        return False

    def get_placekeys_for_cbgs(self, cbg_set):
        return self._df["placekey"].tolist()

    def for_popgen_places(self, placekeys):
        return self._df[self._df["placekey"].isin(set(placekeys))].reset_index(drop=True)


def _run(rows):
    cz_data = {CBG_IN: 100, CBG_IN2: 100}            # the simulated cluster CBGs
    out = convert_data(_people_df(), cz_data, shared_data=_Stub(rows))
    # map placekey -> emitted catchment_fj
    return {p["placekey"]: p["catchment_fj"] for p in out["places"].values()}


def test_popgen_uses_the_shared_catchment_helpers():
    # popgen imports gen_patterns' helpers (single source of the f_j definition +
    # fallback), so the places-bundle f_j and the movement-target f_j can't drift.
    assert popgen._catchment_fraction(json.dumps({CBG_IN: 30, CBG_OUT: 70}), {CBG_IN}) == 0.3
    assert popgen._catchment_fraction(None, {CBG_IN}) is None          # missing -> fallback
    assert popgen._median_fj_fallback([0.4, 0.6, None]) == 0.5


def test_in_cluster_fraction_is_emitted():
    fj = _run([_place_row("local", {CBG_IN: 30, CBG_OUT: 70})])
    assert fj["local"] == 0.3


def test_all_external_is_floored_not_zero():
    fj = _run([_place_row("airport", {CBG_OUT: 100})])          # f_j = 0.0 observed
    assert fj["airport"] == CATCHMENT_FJ_FLOOR


def test_tiny_fraction_is_floored():
    fj = _run([_place_row("tiny", {CBG_IN: 1, CBG_OUT: 9999})])  # f_j = 1e-4
    assert fj["tiny"] == CATCHMENT_FJ_FLOOR


def test_missing_visitor_cbgs_uses_median_fallback():
    # observed f_j are 0.4 and 0.6 -> median 0.5; the missing one inherits it.
    fj = _run([
        _place_row("a", {CBG_IN: 40, CBG_OUT: 60}),
        _place_row("b", {CBG_IN: 60, CBG_OUT: 40}),
        _place_row("missing", None),
    ])
    assert fj["a"] == 0.4 and fj["b"] == 0.6
    assert fj["missing"] == 0.5


def test_every_place_has_bounded_fj():
    fj = _run([
        _place_row("local", {CBG_IN: 30, CBG_OUT: 70}),
        _place_row("airport", {CBG_OUT: 100}),
        _place_row("missing", None),
    ])
    assert len(fj) == 3
    assert all(CATCHMENT_FJ_FLOOR <= v <= 1.0 for v in fj.values())
