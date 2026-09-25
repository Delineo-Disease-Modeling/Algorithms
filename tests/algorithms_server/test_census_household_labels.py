"""Census variable labels behind synthetic household composition.

Two codes used to be mislabeled: ``family_households`` pointed at B09019_002E
(population *in households*) and ``with_children_under_18`` at B11003_002E
(married-couple families). That made P(spouse) about 0.18 and P(children) about
0.72 for Oklahoma family households, instead of about 0.72 and 0.43.

These tests pin every code the generator relies on to its official ACS 2023
label, and run the real CensusDataPuller merge + imputation over recorded
Census rows (fixtures/acs2023_oklahoma_household_tables.json) so a wrong code
shows up as an implausible probability, not just a string mismatch.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

_SERVER = Path(__file__).resolve().parents[2] / "server"
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
for _m in ("cbg_demographics", "patterns", "patterns_loader", "popgen", "census_data"):
    sys.modules.pop(_m, None)

import popgen  # noqa: E402
from census_data import (  # noqa: E402
    CENSUS_API_MAX_VARIABLES,
    DERIVED_SUM_FIELDS,
    CensusDataPuller,
)
from popgen import SyntheticPopulationGenerator  # noqa: E402

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "acs2023_oklahoma_household_tables.json").read_text()
)
LABELS = FIXTURE["labels"]

STATE_FIPS = "40"          # Oklahoma
TULSA = "143"              # in ACS 1-year: real county row
OSAGE = "113"              # not in ACS 1-year: imputed from the state row

_OWN_KIDS = "With own children of the householder under 18 years:"

# key -> (code, official ACS 2023 label)
PINNED_DETAILED = {
    "total_households": ("B11001_001E", "Estimate!!Total:"),
    "family_households": ("B11001_002E", "Estimate!!Total:!!Family households:"),
    "total_family_households": ("B11003_001E", "Estimate!!Total:"),
    "married_with_children": (
        "B11003_003E", f"Estimate!!Total:!!Married-couple family:!!{_OWN_KIDS}"),
    "single_father_with_children": (
        "B11003_010E",
        f"Estimate!!Total:!!Other family:!!Male householder, no spouse present:!!{_OWN_KIDS}"),
    "single_mother_with_children": (
        "B11003_016E",
        f"Estimate!!Total:!!Other family:!!Female householder, no spouse present:!!{_OWN_KIDS}"),
    "opposite-sex spouse": ("B09019_010E", "Estimate!!Total:!!In households:!!Opposite-sex spouse"),
    "same-sex spouse": ("B09019_011E", "Estimate!!Total:!!In households:!!Same-sex spouse"),
    "householders": ("B09019_003E", "Estimate!!Total:!!In households:!!Householder:"),
    "male_householders": ("B09019_004E", "Estimate!!Total:!!In households:!!Householder:!!Male:"),
}


def _fake_fetch(base_url, variables, state_fips, county_fips="*"):
    """Serve recorded Census rows in the API's shape for whatever codes are asked.

    A code missing from the fixture raises KeyError, so adding a variable
    forces re-recording the fixture and checking the new code's label.
    """
    source = FIXTURE["acs5_profile"] if "profile" in base_url else FIXTURE["acs1"]
    codes = list(variables.values())
    if county_fips is None:
        rows = [["NAME", *codes, "state"]]
        geos = ["000"]
    else:
        rows = [["NAME", *codes, "state", "county"]]
        geos = [geo for geo in source if geo != "000"]
    for geo in geos:
        recorded = source[geo]
        row = [recorded["NAME"], *(recorded[code] for code in codes), state_fips]
        if county_fips is not None:
            row.append(geo)
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def census_rows():
    puller = CensusDataPuller(api_key="x")
    puller.fetch_census_data = _fake_fetch
    return puller.pull_counties_census_data(STATE_FIPS, [TULSA, OSAGE], None)


@pytest.mark.parametrize("key", sorted(PINNED_DETAILED))
def test_detailed_code_is_pinned_to_its_acs_label(key):
    code, label = PINNED_DETAILED[key]
    assert CensusDataPuller(api_key="x").variables_detailed[key] == code
    assert LABELS[code] == label


def test_every_requested_code_has_a_recorded_label():
    puller = CensusDataPuller(api_key="x")
    requested = set(puller.variables_detailed.values()) | set(puller.variables_base.values())
    assert requested <= set(LABELS), sorted(requested - set(LABELS))


def test_household_count_keys_are_not_person_counts():
    # B09019 lines under "In households:" count people, not households.
    for key, code in CensusDataPuller(api_key="x").variables_detailed.items():
        if key.endswith("households"):
            assert not LABELS[code].startswith("Estimate!!Total:!!In households"), (key, code)


def test_requests_fit_the_census_api_variable_cap():
    puller = CensusDataPuller(api_key="x")
    # +1 for NAME, which every request also asks for.
    assert len(puller.variables_detailed) + 1 <= CENSUS_API_MAX_VARIABLES
    assert len(puller.variables_base) + 1 <= CENSUS_API_MAX_VARIABLES


@pytest.mark.parametrize("geo", ["000", TULSA, OSAGE])
def test_partner_probability_is_plausible(census_rows, geo):
    p_partner = SyntheticPopulationGenerator._partner_probability(census_rows[geo])
    assert 0.6 <= p_partner <= 0.8, p_partner


@pytest.mark.parametrize("geo", ["000", TULSA, OSAGE])
def test_children_probability_is_plausible(census_rows, geo):
    p_children = SyntheticPopulationGenerator._children_probability(census_rows[geo])
    assert 0.35 <= p_children <= 0.5, p_children


@pytest.mark.parametrize("geo", ["000", TULSA, OSAGE])
def test_family_households_equal_b11003_families(census_rows, geo):
    row = census_rows[geo]
    assert row["family_households"] == row["total_family_households"]


@pytest.mark.parametrize("geo", ["000", TULSA, OSAGE])
def test_last_requested_variable_is_merged(census_rows, geo):
    last_key = list(CensusDataPuller(api_key="x").variables_detailed)[-1]
    assert isinstance(census_rows[geo][last_key], int)


def test_imputed_county_gets_new_variables_and_derived_sum(census_rows):
    osage, state = census_rows[OSAGE], census_rows["000"]
    parts = DERIVED_SUM_FIELDS["with_children_under_18"]
    for key in ("family_households", *parts):
        assert 0 < osage[key] < state[key], key
    assert osage["with_children_under_18"] == sum(osage[part] for part in parts)


def test_derived_children_sum_fails_loud_when_a_part_is_missing():
    row = {"married_with_children": 10, "single_mother_with_children": 4}
    with pytest.raises(ValueError, match="single_father_with_children"):
        CensusDataPuller.add_derived_fields({TULSA: row})


def test_generated_family_households_mostly_have_a_spouse(census_rows, monkeypatch):
    monkeypatch.setattr(popgen.CbgSexAgeSampler, "load_default", staticmethod(lambda cbgs: None))
    random.seed(7)
    np.random.seed(7)
    generator = SyntheticPopulationGenerator(census_rows, {"401430001001": 100})

    family_flags = []
    sample_size = generator._sample_household_size

    def record_family_flag(county_data, is_family):
        family_flags.append(is_family)
        return sample_size(county_data, is_family)

    generator._sample_household_size = record_family_flag
    compositions = [generator.determine_household_composition(TULSA) for _ in range(4000)]
    family = [c for c, is_family in zip(compositions, family_flags) if is_family]

    partner_share = sum(c["has_partner"] for c in family) / len(family)
    assert 0.6 <= partner_share <= 0.8, partner_share
