import pandas as pd
import yaml

# Read CSV file
df = pd.read_csv('./input/hagerstown.pois.csv')

# Initialize dictionary to store data
data = {}

# Loop through each row in the dataframe
for _, row in df.iterrows():
    # Extract relevant data
    location_name = row['location_name']
    bucketed_dwell_times = '{"<5":0,"5-10":0,"11-20":0,"21-60":0,"61-120":0,"121-240":0,">240":0}' # Placeholder, as this data is not available in the CSV
    popularity_by_day = {"Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0, "Friday": 0, "Saturday": 0, "Sunday": 0} # Placeholder, as this data is not available in the CSV
    popularity_by_hour = [0] * 24 # Placeholder, as this data is not available in the CSV
    raw_visit_counts = 0 # Placeholder, as this data is not available in the CSV
    related_same_day_brand = {} # Placeholder, as this data is not available in the CSV
    
    # Store data in the dictionary
    data[location_name] = {
        'bucketed_dwell_times': bucketed_dwell_times,
        'popularity_by_day': popularity_by_day,
        'popularity_by_hour': popularity_by_hour,
        'raw_visit_counts': raw_visit_counts,
        'related_same_day_brand': related_same_day_brand
    }

# Write data to YAML file
with open('./input/hagerstown.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)
