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

        # Age distributions (simplified)
        # TODO: Create a function to derive age distributions from additional census data
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

    def determine_household_composition(self, county_code: str) -> Dict[str, int]:
        """Determine the composition of a household based on census data."""
        county_data = self.census_data[county_code]

        # Determine if it's a family household
        is_family = random.random() < (county_data["total_family_households"] / county_data["total_households"])

        # Determine household size based on distribution
        size_distribution = []
        size_counts = 0
        for i in range(2, 8):  # Sizes 2 through 7+
            size_key = f"size_{i}" if i < 7 else "size_7_plus"
            if size_key in county_data:
                size_distribution.append((i, county_data[size_key]))
                size_counts += county_data[size_key]

        # If we don't have size distribution data, use average household size
        if size_counts == 0:
            avg_size = county_data["avg_household_size"]
            household_size = max(1, int(np.random.normal(avg_size, 1)))
        else:
            size_probs = [count/size_counts for _, count in size_distribution]
            household_size = np.random.choice([size for size, _ in size_distribution], p=size_probs)

        # Determine gender of household head
        head_is_male = random.random() < (county_data["male_householders"] / county_data["householders"])

        # Calculate probabilities for different household types
        has_partner = False
        has_children = 0
        has_relatives = 0
        has_nonrelatives = 0

        if is_family:
            # Family households
            percent_married = (county_data["opposite-sex spouse"] + county_data["same-sex spouse"]) / county_data["family_households"]
            if random.random() < percent_married:  # Most family households have a partner
                has_partner = True
                household_size -= 1  # Account for partner

            # Determine if has children and how many
            if random.random() < (county_data.get("with_children_under_18", 0) / max(1, county_data.get("total_family_households", 1))):
                child_count = min(household_size - 1, np.random.geometric(p=0.5))
                has_children = child_count
                household_size -= child_count

            # Determine if has other relatives and how many
            if household_size > 1:
                percent_other_relatives = (county_data["brother_or_sister"] + county_data["parent"] + county_data["parent-in-law"] + county_data["son-in-law or daughter-in-law"] + county_data["other_relative"]) / county_data["family_households"]
                # TODO: Use percent_other_relatives to determine number of relatives
                relative_count = min(household_size - 1, np.random.poisson(1))
                has_relatives = relative_count
                household_size -= relative_count

            # Remaining slots are for non-relatives
            has_nonrelatives = max(0, household_size - 1)  # -1 for the head
        else:
            # Non-family households
            non_family_households = county_data["total_households"] - county_data["total_family_households"]
            percent_unmarried = (county_data["opposite-sex unmarried_partner"] + county_data["same-sex unmarried_partner"]) / non_family_households
            if random.random() < percent_unmarried:  # Some non-family households have unmarried partners
                has_partner = True
                household_size -= 1

            # Non-family households don't have children by definition
            # Remaining slots are for non-relatives
            has_nonrelatives = max(0, household_size - 1)  # -1 for the head

        return {
            'head_gender': 'M' if head_is_male else 'F',
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
        head = Person(
            person_id=self.next_person_id,
            household_id=household_id,
            county_code=county_code,
            cbg=cbg,
            gender=head_gender,
            age=self.generate_age('householder'),
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

            partner = Person(
                person_id=self.next_person_id,
                household_id=household_id,
                county_code=county_code,
                cbg=cbg,
                gender=partner_gender,
                age=self.generate_age('spouse_partner'),
                relate_head=2,  # 2: partner
                household_lat=household_lat,
                household_lon=household_lon
            )
            self.next_person_id += 1
            household_members.append(partner)

        # TODO: For all non-head members and partners, determine gender via additional census data
        # Add children
        for _ in range(household_composition['num_children']):
            child_gender = 'M' if random.random() < 0.5 else 'F'
            child = Person(
                person_id=self.next_person_id,
                household_id=household_id,
                county_code=county_code,
                cbg=cbg,
                gender=child_gender,
                age=self.generate_age('child'),
                relate_head=3,  # 3: child
                household_lat=household_lat,
                household_lon=household_lon
            )
            self.next_person_id += 1
            household_members.append(child)

        # Add relatives
        for _ in range(household_composition['num_relatives']):
            relative_gender = 'M' if random.random() < 0.5 else 'F'

            # Decide which type of relative.
            # TODO: The son-in-law and daughter-in-law categories imply adult children
            # in the home; revise this to better reflect the actual distribution of
            # relatives (they are currently folded into none of the four buckets).
            relative_type = random.choices(
                ['parent', 'sibling', 'grandchild', 'other_relative'],
                weights=self._relative_type_weights(county_data),
            )[0]

            relative = Person(
                person_id=self.next_person_id,
                household_id=household_id,
                county_code=county_code,
                cbg=cbg,
                gender=relative_gender,
                age=self.generate_age(relative_type),
                relate_head=4,  # 4: relative
                household_lat=household_lat,
                household_lon=household_lon
            )
            self.next_person_id += 1
            household_members.append(relative)

        # Add non-relatives
        for _ in range(household_composition['num_nonrelatives']):
            nonrel_gender = 'M' if random.random() < 0.5 else 'F'
            nonrel_type = 'foster_child' if random.random() < 0.1 else 'non_relative'

            nonrel = Person(
                person_id=self.next_person_id,
                household_id=household_id,
                county_code=county_code,
                cbg=cbg,
                gender=nonrel_gender,
                age=self.generate_age(nonrel_type),
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


def gen_pop(cz_data, gdf=None, shared_data=None):
    """
    Generate synthetic population for a convenience zone.

    Args:
        cz_data: Dictionary mapping CBG IDs to population counts
        gdf: Optional GeoDataFrame with CBG geometries for residential area sampling
        shared_data: Pre-loaded PatternsData used to derive the places dict.

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

    return convert_data(population, cz_data, shared_data=shared_data)

if __name__ == '__main__':
    try:
        geoids, _ = generate_cz('240430006012', 10_0000)
        papdata = gen_pop(geoids)

        with open(r'./output/papdata.json', 'w') as f:
            json.dump(papdata, f)
    except Exception as e:
        print(f'ERROR: could not generate papdata.json: {e}')
        raise
