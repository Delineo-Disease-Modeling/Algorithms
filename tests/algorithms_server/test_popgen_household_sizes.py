import random
import sys
from pathlib import Path

import numpy as np

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
for _m in ("cbg_demographics", "patterns", "patterns_loader", "popgen", "census_data"):
    sys.modules.pop(_m, None)

import popgen  # noqa: E402
from census_data import CensusDataPuller  # noqa: E402
from popgen import SyntheticPopulationGenerator  # noqa: E402


CBG = "400010001001"


def _county(**overrides):
    data = {
        "total_households": 10,
        "total_family_households": 0,
        "family_households": 0,
        "avg_household_size": 2.5,
        "householders": 10,
        "male_householders": 0,
        "male_hh_living_alone": 10,
        "female_hh_living_alone": 0,
        "opposite-sex spouse": 0,
        "same-sex spouse": 0,
        "opposite-sex unmarried_partner": 0,
        "same-sex unmarried_partner": 0,
        "with_children_under_18": 0,
        "brother_or_sister": 0,
        "grandchild": 0,
        "parent": 0,
        "parent-in-law": 0,
        "son-in-law or daughter-in-law": 0,
        "other_relative": 0,
        "size_2": 0,
        "size_3": 0,
        "size_4": 0,
        "size_5": 0,
        "size_6": 0,
        "size_7_plus": 0,
        "nonfamily_size_1": 10,
        "nonfamily_size_2": 0,
        "nonfamily_size_3": 0,
        "nonfamily_size_4": 0,
        "nonfamily_size_5": 0,
        "nonfamily_size_6": 0,
        "nonfamily_size_7_plus": 0,
    }
    data.update(overrides)
    return data


def test_census_puller_requests_nonfamily_household_size_buckets():
    variables = CensusDataPuller(api_key="x").variables_detailed

    assert variables["nonfamily_size_1"] == "B11016_010E"
    assert variables["nonfamily_size_2"] == "B11016_011E"
    assert variables["nonfamily_size_7_plus"] == "B11016_016E"


def test_household_size_distribution_uses_nonfamily_one_person_bucket():
    county = _county(nonfamily_size_1=6, nonfamily_size_2=4)

    assert SyntheticPopulationGenerator._household_size_distribution(
        county, is_family=False) == [(1, 6), (2, 4)]


def test_nonfamily_one_person_household_generates_only_head(monkeypatch):
    monkeypatch.setattr(
        popgen.CbgSexAgeSampler,
        "load_default",
        staticmethod(lambda cbgs: None),
    )
    random.seed(1)
    np.random.seed(1)

    county = _county()
    generator = SyntheticPopulationGenerator({"001": county}, {CBG: 100})

    household = generator.generate_household(county, "001", CBG)

    assert len(household) == 1
    assert household[0].relate_head == 1
    assert household[0].gender == "M"
