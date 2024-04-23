import yaml
import pandas as pd
import os
import csv

def convert_dict_to_string(input_dict):
    # Extracting keys and values from the input dictionary
    keys = list(input_dict.keys())
    values = list(input_dict.values())

    # Convert keys to integers for sorting
    keys_int = []
    for key in keys:
        if key.startswith('<'):
            keys_int.append(-1)
        elif key.startswith('>'):
            keys_int.append(float('inf'))
        else:
            range_values = re.findall(r'\d+', key)
            keys_int.append(sum(map(int, range_values)) / len(range_values))

    # Sorting keys and values based on integer representation of keys
    sorted_indices = sorted(range(len(keys_int)), key=lambda k: keys_int[k])

    # Creating a new dictionary with sorted keys and values
    sorted_dict = {}
    for i in sorted_indices:
        sorted_dict[keys[i]] = values[i]

    # Converting the dictionary to a JSON string
    json_string = json.dumps(sorted_dict)

    return json_string

def generate_yaml(town):
    """
    Generates YAML file for the specified town.
    """

    csv_file = f'./input/{town}.csv'
    csv_poi_file = f'./input/{town}.pois.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_poi_file)

    # Create a dictionary with location_name as key and true/false based on naics_code existence
    location_dict = {}

    for index, row in df.iterrows():
        location_name = row["location_name"]
        naics_code = row["naics_code"]
        if pd.notna(naics_code):
            location_dict[location_name] = True
        else:
            location_dict[location_name] = False

    data = {}

    # Read CSV file and extract data
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if location_dict[row['location_name']]:
                data[row['location_name']] = {
                    'bucketed_dwell_times': row['bucketed_dwell_times'],
                    'popularity_by_day': yaml.safe_load(row['popularity_by_day']),
                    'popularity_by_hour': yaml.safe_load(row['popularity_by_hour']),
                    'raw_visit_counts': int(row['raw_visit_counts']),
                    'related_same_day_brand': yaml.safe_load(row['related_same_day_brand'])
                }

    yaml_file = f'./input/{town}.yaml'

    with open(yaml_file, mode="w") as yamlfile:
        yaml.dump(data, yamlfile, default_flow_style=False)

    if os.path.exists(yaml_file):
        #update_yaml_file(yaml_file)
        print(f"YAML file generated successfully for town '{town}'")
    else:
        print(f"File '{yaml_file}' not found.")
