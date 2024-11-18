import pandas as pd
import json
import re
import os
from collections import OrderedDict

# Directory containing the census data
census_dir = "./SP24/census"

# Initialize an output dictionary
output = {}

# Regex patterns for identifying age/sex and household types
age_sex_pattern = re.compile(r"Estimate!!Total:!!(?P<sex>Male|Female):!!(?P<age_group>.+)")
household_pattern = re.compile(r"Estimate!!Total:!!In households:!!(?P<household_group>[^:]+):?(?P<sex>Male|Female)?:?(?P<status>.+)?")

# Function to process CSV data for age/sex or household type
def process_csv(data, pattern, category_key):
    category_data = {"male": {}, "female": {}} if category_key == "age_sex" else {}

    for column in data.columns:
        match = pattern.search(column)
        if match:
            if category_key == "age_sex":
                sex = match.group('sex').lower()
                age_group = match.group('age_group').strip()
                number = pd.to_numeric(data[column], errors='coerce').fillna(0).sum()
                category_data[sex][age_group] = {
                    "number_of_people": int(number)
                }
            else:  # household_type
                household_group = match.group('household_group').strip()
                sex = match.group('sex')
                status = match.group('status').strip() if match.group('status') else "All"
                number = pd.to_numeric(data[column], errors='coerce').fillna(0).sum()

                # Nested structure for household type
                if household_group not in category_data:
                    category_data[household_group] = {}
                if sex:
                    if sex not in category_data[household_group]:
                        category_data[household_group][sex] = {}
                    category_data[household_group][sex][status] = {
                        "number_of_people": int(number)
                    }
                else:
                    category_data[household_group][status] = {
                        "number_of_people": int(number)
                    }
    
    return category_data

# Iterate over each state folder in the census directory
for state_folder in os.listdir(census_dir):
    state_path = os.path.join(census_dir, state_folder)
    
    # Process only directories
    if not os.path.isdir(state_path):
        continue

    # Initialize the state's entry in the output dictionary
    if state_folder not in output:
        output[state_folder] = OrderedDict()  # Use OrderedDict to control insertion order

    # Initialize accumulators
    total_age_sex = {"male": {}, "female": {}}
    total_household_type = {}

    age_sex_file = os.path.join(state_path, "ACSDT5Y2019.B01001-Data.csv")
    household_file = os.path.join(state_path, "ACSDT5Y2019.B09019-Data.csv")

    if os.path.exists(age_sex_file):
        data = pd.read_csv(age_sex_file, header=1)
        
        for _, row in data.iterrows():
            geo_info = row['NAME'].split(", ")
            county_city = ", ".join(geo_info[:-1])

            if county_city not in output[state_folder]:
                output[state_folder][county_city] = {}

            city_age_sex = process_csv(pd.DataFrame([row]), age_sex_pattern, "age_sex")
            output[state_folder][county_city]["age_sex"] = city_age_sex

            # Aggregate into total_age_sex
            for sex in city_age_sex:
                for age_group, values in city_age_sex[sex].items():
                    if age_group not in total_age_sex[sex]:
                        total_age_sex[sex][age_group] = {"number_of_people": 0}
                    total_age_sex[sex][age_group]["number_of_people"] += values["number_of_people"]

    if os.path.exists(household_file):
        data = pd.read_csv(household_file, header=1)
        
        for _, row in data.iterrows():
            geo_info = row['NAME'].split(", ")
            county_city = ", ".join(geo_info[:-1])

            if county_city not in output[state_folder]:
                output[state_folder][county_city] = {}

            city_household_type = process_csv(pd.DataFrame([row]), household_pattern, "household_type")
            output[state_folder][county_city]["household_type"] = city_household_type

            # Aggregate into total_household_type
            for household_group, values in city_household_type.items():
                if household_group not in total_household_type:
                    total_household_type[household_group] = {}
                for sex, status_values in values.items():
                    if sex not in total_household_type[household_group]:
                        total_household_type[household_group][sex] = {}
                    for status, value in status_values.items():
                        if status not in total_household_type[household_group][sex]:
                            total_household_type[household_group][sex][status] = {"number_of_people": 0}
                        total_household_type[household_group][sex][status]["number_of_people"] += value

    # Add "Total" entry for the state
    output[state_folder] = OrderedDict([("Total", {
        "age_sex": total_age_sex,
        "household_type": total_household_type
    })] + list(output[state_folder].items()))

# Ensure the output directory exists
os.makedirs("./SP24/refactored/data_download", exist_ok=True)

# Save the result to a JSON file in the specified directory
with open("./SP24/refactored/data_download/census_data_summary1.json", "w") as json_file:
    json.dump(output, json_file, indent=4)

print("JSON file './SP24/refactored/data_download/census_data_summary.json' created successfully.")
