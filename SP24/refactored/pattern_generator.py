import json

def merge_files(result_hh_file, result_poi_file, patterns_file):
    # Load data from result_hh.json
    with open(result_hh_file, 'r') as hh_file:
        result_hh_data = json.load(hh_file)
    
    # Load data from result_poi.json
    with open(result_poi_file, 'r') as poi_file:
        result_poi_data = json.load(poi_file)
    
    patterns = {}

    # Extract and merge data from result_hh_data
    for timestep, households in result_hh_data.items():
        timestep = timestep.split('_')[1]  # Extract timestep number
        patterns[timestep] = {"homes": {}, "places": {}}
        for household, people in households.items():
            if people:  # If there are people in the household
                patterns[timestep]["homes"][household.split('_')[1]] = [str(person["id"]) for person in people]

    # Extract and merge data from result_poi_data
    for timestep, pois in result_poi_data.items():
        timestep = timestep.split('_')[1]  # Extract timestep number
        for poi_name, people in pois.items():
            if people:  # If there are people at the POI
                if timestep not in patterns:
                    patterns[timestep] = {"homes": {}, "places": {}}
                if poi_name.startswith("id_"):
                    poi_id = poi_name.split('_')[1]
                else:
                    poi_id = poi_name
                patterns[timestep]["places"][poi_id] = [str(person[0]["id"]) for person in people if person]

    # Write aggregated data to patterns.json
    with open(patterns_file, 'w') as patterns_json:
        json.dump(patterns, patterns_json, indent=4)

# Call the function with file names
merge_files("./output/result_hh.json", "./output/result_poi.json", "patterns.json")