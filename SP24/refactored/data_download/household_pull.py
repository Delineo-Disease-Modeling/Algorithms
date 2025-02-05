import requests
import json

# Census API key
API_KEY = "b1cdc56f4855e77fe024c8b2dfa187b7985cbd89"

# Base URL for the ACS 5-Year Detailed Tables
DETAILED_BASE_URL = "https://api.census.gov/data/2021/acs/acs5"
BASE_URL = "https://api.census.gov/data/2021/acs/acs5/profile"

# State and county FIPS codes
STATE_FIPS = "24"  # Maryland
COUNTY_FIPS = "*"  # All counties

VARIABLES_BASE = {
    "avg_household_size": "DP02_0016E",
    "avg_family_size": "DP02_0017E"
}

# Variables to retrieve from each table
VARIABLES = {
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
    "multigenerational_households": "B11017_002E"
}

# Default data when values are missing
DEFAULT_DATA = {
    "family_households": {
        "total_percentage": 65.4,
        "with_children_under_18": {
            "total_percentage": 28.8,
            "distribution": {
                "married_couple_with_children": {
                    "total_percentage": 60.68,
                    "number_of_children": {
                        "1": 42.9,
                        "2": 36.9,
                        "3": 14.1,
                        "4+": 6.1
                    }
                },
                "single_mother_with_children": {
                    "total_percentage": 33.45,
                    "number_of_children": {
                        "1": 43.5,
                        "2": 36.3,
                        "3": 14.1,
                        "4+": 6.2
                    }
                },
                "single_father_with_children": {
                    "total_percentage": 5.87,
                    "number_of_children": {
                        "1": 42.3,
                        "2": 37.5,
                        "3": 14.1,
                        "4+": 6
                    }
                }
            }
        },
        "with_over_65": {
            "total_percentage": 20.6,
            "multi_generational": 7.2,
        },
        "size_distribution": {
            "2": 44.17,
            "3": 22.54,
            "4": 18.69,
            "5": 8.82,
            "6": 3.48,
            "7+": 2.29
        }
    }
}

# Helper function to safely convert to integer
def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def fetch_profile_data(BASE_URL, api_key, variables, state_fips):
    params = {
        "get": "NAME," + ",".join(variables.values()),
        "for": "county:*",
        "in": f"state:{state_fips}",
        "key": api_key
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")

# Fetch data from Census API
def fetch_census_data(base_url, api_key, variables, state_fips, county_fips):
    params = {
        "get": "NAME," + ",".join(variables.values()),
        "for": f"county:{county_fips}",
        "in": f"state:{state_fips}",
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.status_code} - {response.text}")

def process_profile_data(raw_data, variables):
    headers = raw_data[0]
    rows = raw_data[1:]

    formatted_data = {}
    for row in rows:
        county_name = row[headers.index("NAME")]
        county_code = row[headers.index("county")]

        formatted_data[county_code] = {
            "county_name": county_name,
            "avg_household_size": float(row[headers.index(variables["avg_household_size"])]),
            "avg_family_size": float(row[headers.index(variables["avg_family_size"])])
        }

    return formatted_data

# Process Census data
def process_census_data(raw_data, variables):
    headers = raw_data[0]
    rows = raw_data[1:]

    processed_data = {}
    for row in rows:
        county_name = row[headers.index("NAME")]
        county_code = row[headers.index("county")]

        data = {
            "family_households": DEFAULT_DATA["family_households"].copy()
        }

        # Retrieve total households and family households
        total_households = safe_int(row[headers.index(variables["total_households"])])
        total_family_households = safe_int(row[headers.index(variables["total_family_households"])])

        if total_households and total_family_households:
            # Calculate family households as a percentage of total households
            family_household_percentage = total_family_households / total_households

            data["family_households"]["total_percentage"] = round(family_household_percentage * 100, 1)

            # Scale other percentages by this baseline
            with_children_under_18 = safe_int(row[headers.index(variables["with_children_under_18"])])
            if with_children_under_18:
                data["family_households"]["with_children_under_18"]["total_percentage"] = round(
                    (with_children_under_18 / total_family_households) * family_household_percentage * 100, 1
                )

            # Size distribution
            size_data = {
                key: safe_int(row[headers.index(variables[key])])
                for key in ["size_2", "size_3", "size_4", "size_5", "size_6", "size_7_plus"]
            }
            total_size = sum(filter(None, size_data.values()))
            if total_size > 0:
                data["family_households"]["size_distribution"] = {
                    key: round((value / total_size) * 100, 2) if value else 0
                    for key, value in size_data.items()
                }

        processed_data[county_code] = data

    return processed_data

def merge_data(detailed_data, profile_data):
    merged_data = {}
    for county_code, detailed_entry in detailed_data.items():
        profile_entry = profile_data.get(county_code, {})
        merged_data[county_code] = {**detailed_entry, **profile_entry}
    return merged_data

# Fetch and process data
try:

    profile_raw_data = fetch_profile_data(BASE_URL, API_KEY, VARIABLES_BASE, STATE_FIPS)
    profile_data = process_profile_data(profile_raw_data, VARIABLES_BASE)

    raw_data = fetch_census_data(DETAILED_BASE_URL, API_KEY, VARIABLES, STATE_FIPS, COUNTY_FIPS)
    census_data = process_census_data(raw_data, VARIABLES)

    census_data = merge_data(profile_data, census_data)

    # Save to JSON
    with open("census_data.json", "w") as json_file:
        json.dump(census_data, json_file, indent=4)
    print("Census data saved to census_data.json")
except Exception as e:
    print(f"Error: {e}")
