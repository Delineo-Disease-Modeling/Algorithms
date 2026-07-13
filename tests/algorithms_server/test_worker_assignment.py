import sys
from pathlib import Path

import pandas as pd

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
for _m in ("patterns", "patterns_loader", "worker_assignment", "papdata_convert"):
    sys.modules.pop(_m, None)

from papdata_convert import convert_data  # noqa: E402
from worker_assignment import (  # noqa: E402
    assign_workers,
    compute_home_origin_capture,
    effective_workplace_area_weight,
)


class _StubPatterns:
    def __init__(self, rows):
        self._df = pd.DataFrame(rows)

    def is_empty(self):
        return False

    def get_placekeys_for_cbgs(self, cbg_set):
        df = self._df[self._df["poi_cbg"].isin(cbg_set)]
        return df["placekey"].tolist()

    def for_popgen_places(self, placekeys):
        return (
            self._df[self._df["placekey"].isin(placekeys)]
            .drop_duplicates(subset=["placekey"])
            .reset_index(drop=True)
        )


def _people_df():
    return pd.DataFrame([
        {
            "person_id": 1,
            "household_id": 10,
            "gender": "M",
            "age": 41,
            "cbg": "400010001001",
            "household_lat": None,
            "household_lon": None,
        },
        {
            "person_id": 2,
            "household_id": 11,
            "gender": "F",
            "age": 33,
            "cbg": "400010001002",
            "household_lat": None,
            "household_lon": None,
        },
        {
            "person_id": 3,
            "household_id": 12,
            "gender": "F",
            "age": 15,
            "cbg": "400010001001",
            "household_lat": None,
            "household_lon": None,
        },
    ])


def _places():
    return _StubPatterns([
        {
            "placekey": "small",
            "location_name": "Small Shop",
            "top_category": "Retail",
            "latitude": 36.0,
            "longitude": -96.0,
            "street_address": "",
            "postal_code": "74002",
            "poi_cbg": "400010001001",
            "polygon_wkt": None,
            "wkt_area_sq_meters": 80,
            "visitor_home_cbgs": '{"400010001001": 5}',
        },
        {
            "placekey": "large",
            "location_name": "Large Workplace",
            "top_category": "Office",
            "latitude": 36.1,
            "longitude": -96.1,
            "street_address": "",
            "postal_code": "74002",
            "poi_cbg": "400010001001",
            "polygon_wkt": None,
            "wkt_area_sq_meters": 5000,
            "visitor_home_cbgs": '{"400010001002": 5}',
        },
    ])


def _places_with_school():
    rows = _places()._df.to_dict("records")
    rows.append({
        "placekey": "school",
        "location_name": "Local Elementary",
        "top_category": "Elementary and Secondary Schools",
        "latitude": 36.2,
        "longitude": -96.2,
        "street_address": "",
        "postal_code": "74002",
        "poi_cbg": "400010001001",
        "polygon_wkt": None,
        "wkt_area_sq_meters": 3000,
        "visitor_home_cbgs": '{"400010001001": 5}',
        "naics_code": "611110",
    })
    return _StubPatterns(rows)


def test_worker_assignment_emits_persistent_in_zone_pois(monkeypatch):
    monkeypatch.setenv("DELINEO_WORKER_PROB_18_64", "1")
    monkeypatch.setenv("DELINEO_WORKER_PROB_65_PLUS", "0")

    papdata = convert_data(
        _people_df(),
        {"400010001001": 100, "400010001002": 100},
        shared_data=_places(),
        home_origin_capture={
            "400010001001": 1.0,
            "400010001002": 0.0,
        },
    )

    assert papdata["people"]["1"]["is_worker"] is True
    assert papdata["people"]["1"]["work_location_type"] == "poi"
    assert papdata["people"]["1"]["work_poi"] in papdata["places"]

    assert papdata["people"]["2"]["is_worker"] is True
    assert papdata["people"]["2"]["work_location_type"] == "poi"
    assert papdata["people"]["2"]["work_poi"] in papdata["places"]
    assert papdata["people"]["2"]["work_p_inside"] == 0.0
    external_places = [
        place for place in papdata["places"].values()
        if place.get("external_location_type") == "out_of_zone_work"
    ]
    assert len(external_places) == 0

    assert papdata["people"]["3"]["is_worker"] is False
    assert papdata["people"]["3"]["work_location_type"] == "none"
    assert papdata["people"]["3"]["is_student"] is True
    assert papdata["people"]["3"]["school_location_type"] == "out_of_zone"
    school_external_places = [
        place for place in papdata["places"].values()
        if place.get("external_location_type") == "out_of_zone_school"
    ]
    assert len(school_external_places) == 1
    assert school_external_places[0]["label"] == "Out of Zone School"
    assert papdata["worker_assignment"]["worker_count"] == 2
    assert papdata["worker_assignment"]["in_zone_worker_count"] == 2
    assert papdata["worker_assignment"]["out_of_zone_worker_count"] == 0
    assert papdata["school_assignment"]["student_count"] == 1
    assert papdata["school_assignment"]["out_of_zone_student_count"] == 1


def test_school_assignment_uses_elementary_secondary_school_pois(monkeypatch):
    monkeypatch.setenv("DELINEO_WORKER_PROB_18_64", "1")
    papdata = convert_data(
        _people_df(),
        {"400010001001": 100, "400010001002": 100},
        shared_data=_places_with_school(),
        home_origin_capture={
            "400010001001": 1.0,
            "400010001002": 0.0,
        },
    )

    student = papdata["people"]["3"]
    assert student["is_student"] is True
    assert student["school_location_type"] == "poi"
    school = papdata["places"][student["school_poi"]]
    assert school["label"] == "Local Elementary"
    assert school["naics_code"] == "611110"
    assert papdata["school_assignment"]["in_zone_student_count"] == 1


def test_worker_assignment_without_work_pois_does_not_emit_out_of_zone(monkeypatch):
    monkeypatch.setenv("DELINEO_WORKER_PROB_18_64", "1")
    papdata = {
        "people": {
            "1": {
                "age": 41,
                "home_cbg": "400010001001",
            }
        },
        "places": {},
    }

    summary = assign_workers(papdata)

    assert papdata["people"]["1"]["is_worker"] is False
    assert papdata["people"]["1"]["work_location_type"] == "none"
    assert summary["worker_count"] == 0
    assert summary["out_of_zone_worker_count"] == 0
    assert "external_workplace_id" not in papdata


def test_effective_workplace_area_weight_is_capped():
    assert effective_workplace_area_weight(None) == 50
    assert effective_workplace_area_weight(20) == 50
    assert effective_workplace_area_weight(750) == 750
    assert effective_workplace_area_weight(5000) == 2000


def test_compute_home_origin_capture_uses_full_denominator():
    df = pd.DataFrame([
        {
            "poi_cbg": "400010001001",
            "visitor_home_cbgs": '{"400010001001": 30, "400010001002": 10}',
        },
        {
            "poi_cbg": "999990000001",
            "visitor_home_cbgs": '{"400010001001": 70, "400010001002": 10}',
        },
    ])

    capture = compute_home_origin_capture(
        df,
        cluster_cbgs={"400010001001"},
        source_cbgs={"400010001001", "400010001002"},
    )

    assert capture["400010001001"] == 0.3
    assert capture["400010001002"] == 0.5
