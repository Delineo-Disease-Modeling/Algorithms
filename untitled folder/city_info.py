import yaml
import pandas as pd
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from glob import glob 


'''
    SET CITY NAME HERE
'''

city = "Barnsdall"

# Reading Patterns file
# TODO: For final test, use patterns.csv
files = glob("patterns.csv")

def read_file(file):
    return pd.read_csv(file)
with ThreadPoolExecutor(8) as pool:
    df = pd.concat(pool.map(read_file, files))

df_barnsdall = df.loc[df['city'] == city]

import json 

# Create an empty dictionary to store the results
location_name_dict = {}

# Group the rows by "location_name" column
grouped_location_name = df_barnsdall.groupby("location_name")


# Iterate over the groups and create the dictionary
for location_name, group in grouped_location_name:
    selected_cols = ["placekey", "distance_from_home", 'median_dwell', "related_same_day_brand", "popularity_by_hour", "popularity_by_day"]
    subset = group[selected_cols]

    location_name_dict[location_name] = subset.to_dict(orient="records")[0]

    #parse location dictionary from json format of some data
    for target in ['related_same_day_brand', 'popularity_by_hour', 'popularity_by_day']:
        location_name_dict[location_name][target] = json.loads(location_name_dict[location_name][target])
    

with open('barnsdall.yaml', mode="w") as outstream:
    yaml.dump(location_name_dict)