import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import requests
import yaml


class CensusDataPuller:
    def __init__(self, api_key: str = "b1cdc56f4855e77fe024c8b2dfa187b7985cbd89"):
        """
        Initialize the CensusDataPuller with the Census API key.
        
        Args:
            api_key: Census API key. Defaults to the one in your original code.
        """
        self.api_key = api_key
        self.detailed_base_url = "https://api.census.gov/data/2023/acs/acs1"
        self.profile_base_url = "https://api.census.gov/data/2023/acs/acs5/profile"
        
        # Variables to retrieve for base profile data
        self.variables_base = {
            "total_households": "DP02_0001E",
            "pop_in_households": "DP02_0018E",
            "avg_household_size": "DP02_0016E",
            "avg_family_size": "DP02_0017E"
        }
        
        # Variables to retrieve from detailed tables
        self.variables_detailed = {
            # From B11001
            "total_households": "B11001_001E",
            # From B11003
            "total_family_households": "B11003_001E",
            "with_children_under_18": "B11003_002E",
            "married_with_children": "B11003_003E",
            "single_mother_with_children": "B11003_004E",
            "single_father_with_children": "B11003_005E",
            # From B11016
            "size_2": "B11016_003E",
            "size_3": "B11016_004E",
            "size_4": "B11016_005E",
            "size_5": "B11016_006E",
            "size_6": "B11016_007E",
            "size_7_plus": "B11016_008E",
            # From B11017
            "multigenerational_households": "B11017_002E",
            # From B09019
            "total_population": "B09019_001E",
            "family_households": "B09019_002E",
            "householders": "B09019_003E",
            "male_householders": "B09019_004E",
            "male_hh_living_alone": "B09019_005E",
            "male_hh_not_living_alone": "B09019_006E",
            "female_householders": "B09019_007E",
            "female_hh_living_alone": "B09019_008E",
            "female_hh_not_living_alone": "B09019_009E",
            "opposite-sex spouse": "B09019_010E",
            "same-sex spouse": "B09019_011E",
            "opposite-sex unmarried_partner": "B09019_012E",
            "same-sex unmarried_partner": "B09019_013E", 
            "child": "B09019_014E",
            "biological child": "B09019_015E",
            "adopted_child": "B09019_016E",
            "stepchild": "B09019_017E",
            "grandchild": "B09019_018E",
            "brother_or_sister": "B09019_019E",
            "parent": "B09019_020E",
            "parent-in-law": "B09019_021E",
            "son-in-law or daughter-in-law": "B09019_022E",
            "other_relative": "B09019_023E",
            "foster child": "B09019_024E",
            "other_nonrelatives": "B09019_025E",
            "in_group_quarters": "B09019_026E",
        }
        
        # Field dictionary for mapping API field names to readable names
        self.field_dict = {v: k for k, v in self.variables_detailed.items()}
    
    @staticmethod
    def safe_int(value: Union[str, int, float, None]) -> Optional[int]:
        """
        Helper function to safely convert a value to integer.
        
        Args:
            value: Value to convert to integer
            
        Returns:
            Integer value or None if conversion fails
        """
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def fetch_census_data(self, base_url: str, variables: Dict[str, str], 
                         state_fips: str, county_fips: str = "*") -> List[List[str]]:
        """
        Fetch data from Census API for specified variables.
        
        Args:
            base_url: Census API base URL
            variables: Dictionary of variable names and codes
            state_fips: State FIPS code
            county_fips: County FIPS code or "*" for all counties
            
        Returns:
            JSON response as a list of lists
        """
        if county_fips is None:
            params = {
                "get": "NAME," + ",".join(variables.values()),
                "for": f"state:{state_fips}",
                "key": self.api_key
            }
        else:
            params = {
                "get": "NAME," + ",".join(variables.values()),
                "for": f"county:{county_fips}",
                "in": f"state:{state_fips}",
                "key": self.api_key
            }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error fetching data: {response.status_code} - {response.text}")
    
    def process_profile_data(self, raw_data: List[List[str]], 
                            variables: Dict[str, str]) -> Dict[str, Dict[str, Union[str, int, float]]]:
        """
        Process profile data from Census API.
        
        Args:
            raw_data: Raw data from Census API
            variables: Dictionary of variable names and codes
            
        Returns:
            Processed data as a dictionary
        """
        headers = raw_data[0]
        rows = raw_data[1:]

        formatted_data = {}
        for row in rows:
            county_name = row[headers.index("NAME")]
            county_code = row[headers.index("county")]

            formatted_data[county_code] = {
                "county_name": county_name,
                "total_households": self.safe_int(row[headers.index(variables["total_households"])]),
                "pop_in_households": self.safe_int(row[headers.index(variables["pop_in_households"])]),
                "avg_household_size": float(row[headers.index(variables["avg_household_size"])]),
                "avg_family_size": float(row[headers.index(variables["avg_family_size"])])
            }

        return formatted_data
    
    def merge_datasets(self, counties_dict: Dict[str, Dict], 
                      data_lists: List[List[str]], 
                      field_dict: Dict[str, str]) -> Dict[str, Dict]:
        """
        Merge base profile data with detailed data.
        
        Args:
            counties_dict: Dictionary of county data
            data_lists: List of data rows from Census API
            field_dict: Dictionary mapping API field names to readable names
            
        Returns:
            Merged dataset
        """
        # Extract headers from the first list
        counties_list = list(counties_dict.keys())
        headers = data_lists[0]
        
        # Process each data row (starting from index 1)
        for data_row in data_lists[1:]:
            # Get the county code from the last element of the data row
            county_code = data_row[-1]
            # Remove county_code from counties_list
            counties_list = [i for i in counties_list if i != county_code]
            
            # Check if this county exists in the counties dictionary
            if county_code in counties_dict:
                # Process each field in the data row
                for i in range(1, len(headers) - 2):  # Skip 'NAME' and state/county at the end
                    field_name = field_dict.get(headers[i], headers[i])
                    field_value = data_row[i]
                    
                    # Convert to integer if the value is a numeric string
                    if field_value and field_value.isdigit():
                        field_value = int(field_value)
                    
                    # Add the field to the county dictionary
                    counties_dict[county_code][field_name] = field_value
        
        # Impute missing data for counties not in the detailed data
        for county_code in counties_list:
            # County population as a percent of state population
            ratio = counties_dict[county_code]["pop_in_households"] / counties_dict["000"]["pop_in_households"] 
            data_row = data_lists[1]
            for i in range(1, len(headers) - 2):  # Skip 'NAME' and state/county at the end
                field_name = field_dict.get(headers[i], headers[i])
                field_value = data_row[i]
                
                # Convert to integer if the value is a numeric string
                if field_value and field_value.isdigit():
                    field_value = int(np.round(int(field_value)*ratio, 0))
                
                # Add the field to the county dictionary
                counties_dict[county_code][field_name] = field_value

        return counties_dict
    
    def pull_counties_census_data(self, state_fips: str, county_fips: List[str], 
                                 output_file: Optional[str] = None) -> Dict[str, Dict]:
        """
        Pull census data for specified counties.
        
        Args:
            state_fips: State FIPS code
            county_fips: List of county FIPS codes
            output_file: Path to output file (optional)
            
        Returns:
            Dictionary of census data for each county
        """
        print("Pulling census data...")
        try:
            # Fetch base profile data for state and counties
            base_state_data = self.fetch_census_data(self.profile_base_url, self.variables_base, state_fips, county_fips=None)
            base_raw_data = self.fetch_census_data(self.profile_base_url, self.variables_base, state_fips)
            base_data = base_state_data + base_raw_data
            del base_data[2]  # Delete duplicate label row
            
            # Add dummy county code '000' to the state-wide data
            base_data[0].append("county")
            base_data[1].append("000")
            
            # Process base profile data
            base_data = self.process_profile_data(base_data, self.variables_base)
            
            # Fetch detailed data for state and counties
            state_data = self.fetch_census_data(self.detailed_base_url, self.variables_detailed, state_fips, county_fips=None)
            raw_data = self.fetch_census_data(self.detailed_base_url, self.variables_detailed, state_fips)
            census_data = state_data + raw_data
            del census_data[2]  # Delete duplicate label row
            
            # Add dummy county code '000' to the state-wide data
            census_data[1].append("000")
            
            # Merge datasets
            census_data = self.merge_datasets(base_data, census_data, self.field_dict)

            # Filter for only the rows with COUNTY FIP in the COUNTY_FIPS list
            relevant_data = {k: v for k, v in census_data.items() if k in county_fips}
            
            # Add the state data to relevant_data and make key "000" the first element
            relevant_data["000"] = census_data["000"]
            relevant_data = {"000": relevant_data["000"], **{k: v for k, v in relevant_data.items() if k != "000"}}

            # Save to JSON if output file is specified
            if output_file:
                with open(output_file, "w") as json_file:
                    json.dump(relevant_data, json_file, indent=4)
                print(f"Census data saved to {output_file}")
            
            return relevant_data
            
        except Exception as e:
            print(f"Error: {e}")
            return {}

@dataclass

class Person:
    person_id: int
    household_id: int
    county_code: str
    cbg: str
    gender: str  # 'M' or 'F'
    age: int
    relate_head: int  # 1: head, 2: partner, 3: child, 4: relative, 5: non-relative

class SyntheticPopulationGenerator:
    def __init__(self, census_data: dict, cz_data: dict ):
        """Initialize with input census data."""
        self.census_data = census_data
        self.cz_data = cz_data
        # Set up counters for IDs
        self.next_person_id = 1
        self.next_household_id = 1
        
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
    
    def generate_household(self, county_data, county_code: str, cbg: str) -> List[Person]:
        """Generate all members of a single household."""
        household_id = self.next_household_id
        self.next_household_id += 1
        
        household_composition = self.determine_household_composition(county_code)
        household_members = []
        
        # Create household head
        head_gender = household_composition['head_gender']
        head = Person(
            person_id=self.next_person_id,
            household_id=household_id,
            county_code=county_code,
            cbg=cbg,
            gender=head_gender,
            age=self.generate_age('householder'),
            relate_head=1  # 1: head
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
                relate_head=2  # 2: partner
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
                relate_head=3  # 3: child
            )
            self.next_person_id += 1
            household_members.append(child)
        
        # Add relatives
        for _ in range(household_composition['num_relatives']):
            relative_gender = 'M' if random.random() < 0.5 else 'F'
            
            # Decide which type of relative
            # TODO: The son-in-law and daughter-in-law categories imply adult children in the home
                # Revise this to better reflect the actual distribution of relatives
            tot_other_relatives = county_data["brother_or_sister"] + county_data["parent"] + county_data["parent-in-law"] + county_data["son-in-law or daughter-in-law"] + county_data["other_relative"]
            relative_type = random.choices(
                ['parent', 'sibling', 'grandchild', 'other_relative'],
                weights=[county_data["parent"]+county_data["parent-in-law"]/tot_other_relatives # parent
                        , county_data["brother_or_sister"]/tot_other_relatives # sibling
                        , county_data["grandchild"]/tot_other_relatives # grandchild
                        , county_data["other_relative"]/tot_other_relatives] # other_relative
            )[0]
            
            relative = Person(
                person_id=self.next_person_id,
                household_id=household_id,
                county_code=county_code,
                cbg=cbg,
                gender=relative_gender,
                age=self.generate_age(relative_type),
                relate_head=4  # 4: relative
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
                relate_head=5  # 5: non-relative
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


def convert_data(df, cz_data):
    """
    Convert data frame with person and household information into a specific dictionary format
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output JSON file
    """
    
    # Initialize output dictionary
    output = {
        "people": {},
        "homes": {},
        "places": {}
    }
    
    # Create mappings
    # For sex: M -> 0, F -> 1
    sex_mapping = {"M": 0, "F": 1}
    
    # Process each row in the dataframe
    for index, row in df.iterrows():
        person_id = str(row['person_id'])
        household_id = str(row['household_id'])
        
        # Add person to people dictionary
        output["people"][person_id] = {
            "sex": sex_mapping.get(row['gender']),
            "age": row['age'],
            "home": household_id
        }
        
        # Add or update household in homes dictionary
        if household_id not in output["homes"]:
            output["homes"][household_id] = {
                "cbg": row['cbg'],
                "members": 1
            }
        else:
            output["homes"][household_id]["members"] += 1
    
    # Get places from CG dataset
    cbgs = list(cz_data.keys())
    placekeys = []
    with pd.read_csv(r'./data/patterns.csv', chunksize=10000, usecols=['poi_cbg', 'placekey']) as reader:
        for chunk in reader:
            for i, row in chunk.iterrows():
                try:
                    cbg = str(int(float(row['poi_cbg'])))
                    if cbg in cbgs:
                        placekeys.append(row['placekey'])
                except:
                    continue

    places = pd.read_csv(r'./data/2021_05_05_03_core_poi.csv', usecols=['placekey', 'location_name', 'top_category', 'latitude', 'longitude', 'postal_code'])
    places = places[places['placekey'].isin(placekeys)].reset_index()

    for i, row in places.iterrows():
        output['places'][str(i)] = {
            'label': row['location_name'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            # Sometimes this is empty
            'top_category': 'None' if pd.isna(row['top_category']) else row['top_category'],
            'postal_code': row['postal_code']
        }
    
    return output

def gen_pop(cz_data):
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

    census_data = datapuller.pull_counties_census_data(STATE_FIPS, COUNTIES_FIPS, "census_data.json")
    
    # Create population generator
    generator = SyntheticPopulationGenerator(census_data, cz_data)
    
    # Generate population for all counties (with a small sample factor)
    population = generator.generate_full_population(sample_factor=0.01)
    
    # Validate the population
    validation_results = generator.validate_population(population)
    print("\nPopulation Validation:")
    for key, value in validation_results.items():
        print(f"{key}: {value}")
    
    # Save the population to CSV
    population = generator.save_population(population)

    return convert_data(population, cz_data)
