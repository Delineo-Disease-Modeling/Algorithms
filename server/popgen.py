import json
import random
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from czcode import generate_cz

try:
    from residential import ResidentialCache
    RESIDENTIAL_AVAILABLE = True
except ImportError:
    RESIDENTIAL_AVAILABLE = False
    ResidentialCache = None

# The Census ACS client and the DataFrame -> papdata conversion live in their own
# modules now; re-exported here so existing call sites and tests can keep doing
# `from popgen import CensusDataPuller / convert_data / CENSUS_API_KEY_DEFAULT /
# CATCHMENT_FJ_FLOOR`.
from census_data import CensusDataPuller, CENSUS_API_KEY_DEFAULT  # noqa: F401
from cbg_demographics import CbgSexAgeSampler
from papdata_convert import convert_data, CATCHMENT_FJ_FLOOR  # noqa: F401
# Re-exported so `popgen._catchment_fraction` / `popgen._median_fj_fallback`
# stay the documented single access point for the shared f_j helpers (their f_j
# definition must match gen_patterns' movement targets).
from patterns import _catchment_fraction, _median_fj_fallback  # noqa: F401


@dataclass
class Person:
    person_id: int
    household_id: int
    county_code: str
    cbg: str
    gender: str  # 'M' or 'F'
    age: int
    relate_head: int  # 1: head, 2: partner, 3: child, 4: relative, 5: non-relative
    household_lat: Optional[float] = None  # Latitude of household
    household_lon: Optional[float] = None  # Longitude of household

class SyntheticPopulationGenerator:
    def __init__(self, census_data: dict, cz_data: dict, gdf=None):
        """Initialize with input census data.

        Args:
            census_data: Census demographic data by county
            cz_data: Convenience zone data (CBG -> population mapping)
            gdf: Optional GeoDataFrame with CBG geometries for residential sampling
        """
        self.census_data = census_data
        self.cz_data = cz_data
        self.gdf = gdf
        # Set up counters for IDs
        self.next_person_id = 1
        self.next_household_id = 1

        # Initialize residential cache for sampling home locations
        self.residential_cache = None
        if RESIDENTIAL_AVAILABLE and gdf is not None:
            self.residential_cache = ResidentialCache(gdf, use_buildings=True)
            print("Residential cache initialized - homes will be placed in residential areas")
        else:
            print("Residential sampling unavailable - homes will not have coordinates")

        self.cbg_sex_age_sampler = CbgSexAgeSampler.load_default(cz_data.keys())
        if self.cbg_sex_age_sampler is not None:
            print(f"CBG sex-age demographics loaded from {self.cbg_sex_age_sampler.source_path}")
        else:
            print("CBG sex-age demographics unavailable - using role-based age/sex fallback")

        # Age distributions (simplified)
        # Used as a fallback when CBG-level B01001 sex-by-age data is unavailable
        # or has no matching count for a role's age range.
        self.age_distributions = {
            'householder': {'mean': 50, 'std': 15, 'min': 18, 'max': 95},
            'spouse_partner': {'mean': 48, 'std': 15, 'min': 18, 'max': 95},
            'child': {'mean': 15, 'std': 10, 'min': 0, 'max': 40},
            'grandchild': {'mean': 10, 'std': 7, 'min': 0, 'max': 25},
            'parent': {'mean': 75, 'std': 8, 'min': 55, 'max': 100},
            'sibling': {'mean': 45, 'std': 15, 'min': 18, 'max': 95},
            'other_relative': {'mean': 40, 'std': 20, 'min': 0, 'max': 95},
            'non_relative': {'mean': 35, 'std': 15, 'min': 18, 'max': 95},
            'foster_child': {'mean': 12, 'std': 5, 'min': 0, 'max': 21}
        }

    def generate_age(self, role: str) -> int:
        """Generate an age based on the person's role in the household."""
        dist = self.age_distributions.get(role, self.age_distributions['non_relative'])
        age = int(np.random.normal(dist['mean'], dist['std']))
        return max(min(age, dist['max']), dist['min'])  # Clamp to min/max

    def generate_demographics(self, cbg: str, role: str, gender: Optional[str] = None) -> tuple[str, int]:
        """Generate sex and age for a person in ``cbg``.

        Prefer CBG-level B01001 sex-by-age counts, filtered to the role's existing
        plausible age range. When no CBG row / matching bucket exists, fall back to
        the legacy role-age normal distribution and a supplied or random sex.
        """
        dist = self.age_distributions.get(role, self.age_distributions['non_relative'])
        if self.cbg_sex_age_sampler is not None:
            sampled = self.cbg_sex_age_sampler.sample(
                cbg,
                int(dist['min']),
                int(dist['max']),
                sex=gender,
            )
            if sampled is not None:
                return sampled
        if gender is None:
            gender = 'M' if random.random() < 0.5 else 'F'
        return gender, self.generate_age(role)

    @staticmethod
    def _positive_count(county_data: Dict[str, Any], key: str) -> int:
        try:
            return max(0, int(county_data.get(key, 0) or 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _household_size_distribution(cls, county_data: Dict[str, Any], is_family: bool) -> List[tuple[int, int]]:
        if is_family:
            size_keys = [
                (2, 'size_2'),
                (3, 'size_3'),
                (4, 'size_4'),
                (5, 'size_5'),
                (6, 'size_6'),
                (7, 'size_7_plus'),
            ]
        else:
            size_keys = [
                (1, 'nonfamily_size_1'),
                (2, 'nonfamily_size_2'),
                (3, 'nonfamily_size_3'),
                (4, 'nonfamily_size_4'),
                (5, 'nonfamily_size_5'),
                (6, 'nonfamily_size_6'),
                (7, 'nonfamily_size_7_plus'),
            ]
        return [
            (size, count)
            for size, key in size_keys
            if (count := cls._positive_count(county_data, key)) > 0
        ]

    @classmethod
    def _sample_household_size(cls, county_data: Dict[str, Any], is_family: bool) -> int:
        size_distribution = cls._household_size_distribution(county_data, is_family)
        if not size_distribution:
            avg_size = float(county_data.get("avg_household_size") or 1)
            household_size = max(1, int(np.random.normal(avg_size, 1)))
            return max(2, household_size) if is_family else household_size

        sizes = [size for size, _ in size_distribution]
        weights = [count for _, count in size_distribution]
        size_probs = [count / sum(weights) for count in weights]
        return int(np.random.choice(sizes, p=size_probs))

    @classmethod
    def _sample_head_gender(cls, county_data: Dict[str, Any], is_family: bool, household_size: int) -> str:
        if not is_family and household_size == 1:
            male_alone = cls._positive_count(county_data, "male_hh_living_alone")
            female_alone = cls._positive_count(county_data, "female_hh_living_alone")
            living_alone = male_alone + female_alone
            if living_alone > 0:
                return 'M' if random.random() < (male_alone / living_alone) else 'F'

        householders = cls._positive_count(county_data, "householders")
        male_householders = cls._positive_count(county_data, "male_householders")
        if householders <= 0:
            return 'M' if random.random() < 0.5 else 'F'
        return 'M' if random.random() < (male_householders / householders) else 'F'

    @classmethod
    def _partner_probability(cls, county_data: Dict[str, Any]) -> float:
        """P(a family household has a spouse present).

        Spouses of the householder (B09019) over family households (B11001).
        Each married-couple household has exactly one such spouse, so this is
        the married-couple share of family households.
        """
        spouses = (cls._positive_count(county_data, "opposite-sex spouse")
                   + cls._positive_count(county_data, "same-sex spouse"))
        return spouses / max(1, cls._positive_count(county_data, "family_households"))

    @classmethod
    def _children_probability(cls, county_data: Dict[str, Any]) -> float:
        """P(a family household has own children under 18), from B11003.

        ``with_children_under_18`` is the sum of the "with own children" lines
        across all three family types, derived by CensusDataPuller.
        """
        return (cls._positive_count(county_data, "with_children_under_18")
                / max(1, cls._positive_count(county_data, "total_family_households")))

    def determine_household_composition(self, county_code: str) -> Dict[str, int]:
        """Determine the composition of a household based on census data."""
        county_data = self.census_data[county_code]

        # Determine if it's a family household
        total_households = max(1, self._positive_count(county_data, "total_households"))
        family_households = min(
            total_households,
            self._positive_count(county_data, "total_family_households"),
        )
        is_family = random.random() < (family_households / total_households)

        # Determine household size from the matching B11016 family/nonfamily
        # distribution. This includes 1-person nonfamily households.
        household_size = self._sample_household_size(county_data, is_family)

        # Calculate probabilities for different household types
        has_partner = False
        has_children = 0
        has_relatives = 0
        has_nonrelatives = 0
        head_gender = self._sample_head_gender(county_data, is_family, household_size)

        if is_family:
            # Family households
            if random.random() < self._partner_probability(county_data):  # Most family households have a partner
                has_partner = True
                household_size -= 1  # Account for partner

            # Determine if has children and how many
            if random.random() < self._children_probability(county_data):
                child_count = min(household_size - 1, np.random.geometric(p=0.5))
                has_children = child_count
                household_size -= child_count

            # Determine if has other relatives and how many
            if household_size > 1:
                # TODO: Use percent_other_relatives to determine number of relatives
                relative_count = min(household_size - 1, np.random.poisson(1))
                has_relatives = relative_count
                household_size -= relative_count

            # Remaining slots are for non-relatives
            has_nonrelatives = max(0, household_size - 1)  # -1 for the head
        else:
            # Non-family households
            non_family_households = max(0, total_households - family_households)
            percent_unmarried = (
                (self._positive_count(county_data, "opposite-sex unmarried_partner")
                 + self._positive_count(county_data, "same-sex unmarried_partner"))
                / max(1, non_family_households)
            )
            if household_size > 1 and random.random() < percent_unmarried:  # Some non-family households have unmarried partners
                has_partner = True
                household_size -= 1

            # Non-family households don't have children by definition
            # Remaining slots are for non-relatives
            has_nonrelatives = max(0, household_size - 1)  # -1 for the head

        return {
            'head_gender': head_gender,
            'has_partner': has_partner,
            'num_children': has_children,
            'num_relatives': has_relatives,
            'num_nonrelatives': has_nonrelatives
        }

    @staticmethod
    def _relative_type_weights(county_data) -> List[float]:
        """Sampling weights for a household ``relative``'s type.

        Returns weights aligned to ``['parent', 'sibling', 'grandchild',
        'other_relative']``. The ``parent`` bucket combines parents and
        parents-in-law. Weights are raw Census relative counts; ``random.choices``
        normalizes them internally, so they need not sum to 1.

        Note: this previously had an operator-precedence bug
        (``parent + parent_in_law / total``) that added the raw ``parent`` count
        (hundreds–thousands) unnormalized while the other buckets were fractions
        in ``[0, 1]`` — so the parent bucket won almost every draw and siblings,
        grandchildren and other relatives were effectively never sampled.
        """
        return [
            county_data["parent"] + county_data["parent-in-law"],  # parent (incl. in-law)
            county_data["brother_or_sister"],                      # sibling
            county_data["grandchild"],                             # grandchild
            county_data["other_relative"],                         # other_relative
        ]

    def generate_household(self, county_data, county_code: str, cbg: str) -> List[Person]:
        """Generate all members of a single household."""
        household_id = self.next_household_id
        self.next_household_id += 1

        household_composition = self.determine_household_composition(county_code)
        household_members = []

        # Sample household location from residential areas
        household_lat, household_lon = None, None
        if self.residential_cache is not None:
            household_lat, household_lon = self.residential_cache.sample_home_location(cbg)

        # Create household head
        head_gender = household_composition['head_gender']
        head_gender, head_age = self.generate_demographics(cbg, 'householder', gender=head_gender)
        head = Person(
            person_id=self.next_person_id,
            household_id=household_id,
            county_code=county_code,
            cbg=cbg,
            gender=head_gender,
            age=head_age,
            relate_head=1,  # 1: head
            household_lat=household_lat,
            household_lon=household_lon
        )
        self.next_person_id += 1
        household_members.append(head)

        # Add partner if present
        if household_composition['has_partner']:
            partner_gender = 'F' if head_gender == 'M' else 'M'
            # Same-sex couples exist too
            percent_ss_couples = (county_data["same-sex spouse"] + county_data["same-sex unmarried_partner"]) / (county_data["same-sex spouse"] + county_data["same-sex unmarried_partner"] + county_data["opposite-sex unmarried_partner"] + county_data["opposite-sex spouse"])
            if random.random() < percent_ss_couples:
                partner_gender = head_gender
            partner_gender, partner_age = self.generate_demographics(
                cbg, 'spouse_partner', gender=partner_gender)

            partner = Person(
                person_id=self.next_person_id,
                household_id=household_id,
                county_code=county_code,
                cbg=cbg,
                gender=partner_gender,
                age=partner_age,
                relate_head=2,  # 2: partner
                household_lat=household_lat,
                household_lon=household_lon
            )
            self.next_person_id += 1
            household_members.append(partner)

        # Add children
        for _ in range(household_composition['num_children']):
            child_gender, child_age = self.generate_demographics(cbg, 'child')
            child = Person(
                person_id=self.next_person_id,
                household_id=household_id,
                county_code=county_code,
                cbg=cbg,
                gender=child_gender,
                age=child_age,
                relate_head=3,  # 3: child
                household_lat=household_lat,
                household_lon=household_lon
            )
            self.next_person_id += 1
            household_members.append(child)

        # Add relatives
        for _ in range(household_composition['num_relatives']):
            # Decide which type of relative.
            # TODO: The son-in-law and daughter-in-law categories imply adult children
            # in the home; revise this to better reflect the actual distribution of
            # relatives (they are currently folded into none of the four buckets).
            relative_type = random.choices(
                ['parent', 'sibling', 'grandchild', 'other_relative'],
                weights=self._relative_type_weights(county_data),
            )[0]
            relative_gender, relative_age = self.generate_demographics(cbg, relative_type)

            relative = Person(
                person_id=self.next_person_id,
                household_id=household_id,
                county_code=county_code,
                cbg=cbg,
                gender=relative_gender,
                age=relative_age,
                relate_head=4,  # 4: relative
                household_lat=household_lat,
                household_lon=household_lon
            )
            self.next_person_id += 1
            household_members.append(relative)

        # Add non-relatives
        for _ in range(household_composition['num_nonrelatives']):
            nonrel_type = 'foster_child' if random.random() < 0.1 else 'non_relative'
            nonrel_gender, nonrel_age = self.generate_demographics(cbg, nonrel_type)

            nonrel = Person(
                person_id=self.next_person_id,
                household_id=household_id,
                county_code=county_code,
                cbg=cbg,
                gender=nonrel_gender,
                age=nonrel_age,
                relate_head=5,  # 5: non-relative
                household_lat=household_lat,
                household_lon=household_lon
            )
            self.next_person_id += 1
            household_members.append(nonrel)

        return household_members

    def generate_county_population(self, county_code: str, target_households: int = None, cz_population: int = 0) -> List[Person]:
        """Generate a synthetic population for a specific county."""
        county_data = self.census_data[county_code]

        # If target_households is not specified, use a fraction of the actual number
        if target_households is None:
            target_households = min(10000, county_data["total_households"] // 10)

        # Create households for each cbg in the county
        population = []
        county_cbgs = self.cz_data; county_cbgs = [i for i in county_cbgs if i[2:5] == county_code]
        for cbg in county_cbgs:
            # Determine number of households in this cbg
            cbg_population = self.cz_data[cbg]
            pop_fraction = cbg_population / cz_population
            cbg_households = int(target_households * pop_fraction)
            for _ in range(cbg_households):
                household = self.generate_household(county_data, county_code, cbg)
                population.extend(household)

        return population

    def generate_full_population(self, sample_factor: float = 0.01) -> List[Person]:
        """Generate a synthetic population for all counties in the census data."""
        population = []

        for county_code, county_data in self.census_data.items():
            if county_code == "000":  # Skip the state-wide entry
                continue
            # Calculate the total estimated population for the cbgs (for this county) in the convenience zone by summing the population estimates for the corresponding cbgs
            county_cbgs = self.cz_data; county_cbgs = [i for i in county_cbgs if i[2:5] == county_code]
            county_pop = [self.cz_data[i] for i in county_cbgs]
            cz_population = sum(county_pop)
            sample_factor = cz_population / county_data["total_population"]
            target_households = int(county_data["total_households"] * sample_factor)
            county_population = self.generate_county_population(county_code, target_households, cz_population)
            population.extend(county_population)
            print("Generated", len(county_population), "people in", target_households, "households for county", county_code, "(", str(np.round(100*sample_factor, 2)), "% of population)")
        return population

    def validate_population(self, population: List[Person]) -> Dict[str, Any]:
        """Run validation checks on the generated population."""
        population_df = pd.DataFrame([vars(p) for p in population])

        # Household size distribution
        household_sizes = population_df.groupby('household_id').size()
        avg_household_size = household_sizes.mean()

        # Gender distribution
        gender_dist = population_df['gender'].value_counts(normalize=True).to_dict()

        # Age distribution
        age_stats = {
            'mean': population_df['age'].mean(),
            'median': population_df['age'].median(),
            'min': population_df['age'].min(),
            'max': population_df['age'].max()
        }

        # Relationship distribution
        relation_dist = population_df['relate_head'].value_counts(normalize=True).to_dict()

        return {
            'total_people': len(population),
            'total_households': len(household_sizes),
            'avg_household_size': avg_household_size,
            'gender_distribution': gender_dist,
            'age_statistics': age_stats,
            'relationship_distribution': relation_dist
        }

    def save_population(self, population: List[Person]):
        """Save the generated population to a CSV file."""
        population_df = pd.DataFrame([vars(p) for p in population])
        #population_df.to_csv(output_path, index=False)
        #print(f"Population saved to {output_path}")
        return population_df


def _apply_bench_seed():
    seed = os.getenv('DELINEO_BENCH_SEED')
    if seed is None or seed == '':
        return
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)


def gen_pop(cz_data, gdf=None, shared_data=None, home_origin_capture=None):
    """
    Generate synthetic population for a convenience zone.

    Args:
        cz_data: Dictionary mapping CBG IDs to population counts
        gdf: Optional GeoDataFrame with CBG geometries for residential area sampling
        shared_data: Pre-loaded PatternsData used to derive the places dict.
        home_origin_capture: Optional CBG -> p_inside map used for persistent
            worker inside/outside assignment.

    Returns:
        Dictionary with people, homes, and places data (papdata format)
    """
    _apply_bench_seed()

    # Create data puller
    datapuller = CensusDataPuller()

    # Call datapuller on specified state and counties
    cbgs = list(cz_data.keys())
    states = list(set([i[:2] for i in cbgs]))
    if len(states) > 1:
        print('Warning: Multiple states found in the provided CBGs.')
       #raise ValueError("Multiple states found in the provided CBGs.")

    STATE_FIPS = states[0]
    COUNTIES_FIPS = list(set([i[2:5] for i in cbgs]))

    census_data = datapuller.pull_counties_census_data(STATE_FIPS, COUNTIES_FIPS, None)

    # Create population generator with optional gdf for residential sampling
    generator = SyntheticPopulationGenerator(census_data, cz_data, gdf=gdf)

    # Generate population for all counties (with a small sample factor)
    population = generator.generate_full_population(sample_factor=0.01)

    # Validate the population
    try:
        validation_results = generator.validate_population(population)
        print("\nPopulation Validation:")
        for key, value in validation_results.items():
            print(f"{key}: {value}")
    except Exception as e:
        # Validation is diagnostic (prints stats), so don't make it fatal — but
        # surface the real error instead of swallowing it under a bare except.
        print(f"\nERROR: COULD NOT VALIDATE POPULATION: {e}\n")

    # Save the population to CSV
    population = generator.save_population(population)

    return convert_data(
        population,
        cz_data,
        shared_data=shared_data,
        home_origin_capture=home_origin_capture,
    )

if __name__ == '__main__':
    try:
        geoids, _ = generate_cz('240430006012', 10_0000)
        papdata = gen_pop(geoids)

        with open(r'./output/papdata.json', 'w') as f:
            json.dump(papdata, f)
    except Exception as e:
        print(f'ERROR: could not generate papdata.json: {e}')
        raise
