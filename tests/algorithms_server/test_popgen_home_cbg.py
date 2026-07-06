import sys
from pathlib import Path

import pandas as pd

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
for _m in ("patterns", "patterns_loader", "popgen", "papdata_convert"):
    sys.modules.pop(_m, None)

from papdata_convert import convert_data  # noqa: E402


def test_convert_data_emits_person_home_cbg_matching_household():
    df = pd.DataFrame([
        {
            "person_id": 1,
            "household_id": 10,
            "gender": "M",
            "age": 42,
            "cbg": "400010001001",
            "household_lat": None,
            "household_lon": None,
        },
        {
            "person_id": 2,
            "household_id": 10,
            "gender": "F",
            "age": 39,
            "cbg": "400010001001",
            "household_lat": None,
            "household_lon": None,
        },
        {
            "person_id": 3,
            "household_id": 11,
            "gender": "F",
            "age": 12,
            "cbg": "400010001002",
            "household_lat": None,
            "household_lon": None,
        },
    ])

    papdata = convert_data(
        df,
        {"400010001001": 100, "400010001002": 50},
        shared_data=None,
    )

    assert papdata["people"]["1"]["home"] == "10"
    assert papdata["people"]["1"]["home_cbg"] == "400010001001"
    assert papdata["people"]["2"]["home_cbg"] == "400010001001"
    assert papdata["people"]["3"]["home_cbg"] == "400010001002"

    for person in papdata["people"].values():
        assert person["home_cbg"] == papdata["homes"][person["home"]]["cbg"]

    assert papdata["homes"]["10"]["members"] == 2
    assert papdata["homes"]["11"]["members"] == 1
