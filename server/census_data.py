"""Census ACS data client for synthetic population generation.

`CensusDataPuller` fetches and merges the ACS profile + detailed household
tables for a state and its counties. Extracted from `popgen.py` (which
re-exports it) so the population generator isn't carrying a whole API client.
"""
import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import requests

# Census API key. The committed literal is a free, rate-limited Census key that
# also lives in git history; prefer setting the CENSUS_API_KEY env var and rotate
# the literal out when convenient.
CENSUS_API_KEY_DEFAULT = "b1cdc56f4855e77fe024c8b2dfa187b7985cbd89"


class CensusDataPuller:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the CensusDataPuller with the Census API key.

        Args:
            api_key: Census API key. If omitted, falls back to the CENSUS_API_KEY
                environment variable, then to CENSUS_API_KEY_DEFAULT.
        """
        self.api_key = api_key or os.environ.get("CENSUS_API_KEY") or CENSUS_API_KEY_DEFAULT
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
            # Fail loud: gen_pop is the only caller, and an empty census dict
            # silently produces a degenerate population (wrong/zero households)
            # downstream. Surface the failure instead of returning {} — matches the
            # fail-loud posture in patterns.py.
            print(f"ERROR: Census data fetch failed: {e}")
            raise
