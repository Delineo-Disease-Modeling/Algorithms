import random

import numpy as np

from popgen import SyntheticPopulationGenerator


def _county_data(**overrides):
    data = {
        "total_population": 100,
        "total_households": 40,
        "pop_in_households": 100,
        "avg_household_size": 2.5,
        "total_family_households": 0,
        "family_households": 1,
        "with_children_under_18": 0,
        "householders": 40,
        "male_householders": 20,
        "male_hh_living_alone": 0,
        "female_householders": 20,
        "female_hh_living_alone": 0,
        "opposite-sex spouse": 0,
        "same-sex spouse": 0,
        "opposite-sex unmarried_partner": 0,
        "same-sex unmarried_partner": 0,
        "brother_or_sister": 0,
        "parent": 0,
        "parent-in-law": 0,
        "son-in-law or daughter-in-law": 0,
        "other_relative": 0,
        "grandchild": 0,
        "size_1": 0,
        "size_2": 0,
        "size_3": 0,
        "size_4": 0,
        "size_5": 0,
        "size_6": 0,
        "size_7_plus": 0,
        "nonfamily_size_2": 0,
        "nonfamily_size_3": 0,
        "nonfamily_size_4": 0,
        "nonfamily_size_5": 0,
        "nonfamily_size_6": 0,
        "nonfamily_size_7_plus": 0,
    }
    data.update(overrides)
    return data


def _generator(county_data):
    return SyntheticPopulationGenerator(
        {"001": county_data},
        {"990010000001": county_data["total_population"]},
    )


def test_generate_household_uses_one_person_size_bucket():
    random.seed(1)
    np.random.seed(1)
    county_data = _county_data(
        avg_household_size=4.0,
        total_households=10,
        householders=10,
        male_householders=5,
        female_householders=5,
        size_1=10,
    )
    generator = _generator(county_data)

    household = generator.generate_household(county_data, "001", "990010000001")

    assert len(household) == 1
    assert household[0].relate_head == 1


def test_household_size_distribution_combines_nonfamily_size_buckets():
    county_data = _county_data(size_1=20, size_2=30, nonfamily_size_2=50)
    generator = _generator(county_data)

    distribution = dict(generator._household_size_distribution(county_data))

    assert distribution[1] == 20
    assert distribution[2] == 80


def test_generate_county_population_stops_near_cz_population_target():
    random.seed(2)
    np.random.seed(2)
    county_data = _county_data(
        total_population=20,
        total_households=10,
        avg_household_size=7.0,
        size_7_plus=10,
    )
    generator = _generator(county_data)

    population = generator.generate_county_population("001", target_households=10, cz_population=20)

    assert 20 <= len(population) <= 26
