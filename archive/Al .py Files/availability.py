import json
import yaml
import csv
import numpy as np

# Load the JSON data from the file
with open('result_poi.json', 'r') as file:
    data = json.load(file)

with open('simul_settings.yaml', mode="r") as settingstream:
    settings = yaml.full_load(settingstream)

time_range = settings['time']

# Initialize sets to store unique person IDs and timestamps
person_ids = set()
timestamps = set()

# Initialize max_person_id
max_person_id = 0

# Extract unique person IDs and timestamps
for timestamp_key, poi_data in data.items():
    timestamp_idx = int(timestamp_key.split('_')[1])  # Extract the timestamp index from the key
    timestamps.add(timestamp_idx)

    for location_key, timestamp_list in poi_data.items():
        for person_list in timestamp_list:
            for person in person_list:
                person_id = person['id']
                person_ids.add(person_id)

                if person_id > max_person_id:
                    max_person_id = person_id

# Convert sets to sorted lists
sorted_person_ids = sorted(list(person_ids))
sorted_timestamps = sorted(list(timestamps))

# Create a 2D array
result_array = np.zeros((max_person_id + 1, len(sorted_timestamps)), dtype=int)
print(len(sorted_person_ids))



for timestamp_key, poi_data in data.items():
    timestamp_idx = int(timestamp_key.split('_')[1])  # Extract the timestamp index from the key

    for location_key, timestamp_list in poi_data.items():
        for person_list in timestamp_list:
            for person in person_list:
                person_ids.add(person['id'])
                result_array[person['id'] - 1, int(timestamp_idx/60) - 1] = 1 

for person_id in range(result_array.shape[0]):
    row = result_array[person_id, :]
    
    # Find indices where 1 appears
    ones_indices = np.where(row == 1)[0]

    # Calculate consecutive zeros
    consecutive_zeros = np.diff(ones_indices) - 1

    # Set the value in the result_array to the number of consecutive zeros
    result_array[person_id, ones_indices[:-1]] = consecutive_zeros


# Create a CSV file
csv_file_path = 'result_matrix.csv'
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write header
    header = ['Person ID'] + [f'Timestamp {ts}' for ts in sorted_timestamps]
    writer.writerow(header)

    # Write data
    for i, person_id in enumerate(sorted_person_ids):
        row = [person_id] + list(result_array[i, :])
        writer.writerow(row)

print(f"CSV file created: {csv_file_path}")
