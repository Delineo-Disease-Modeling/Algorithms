import json
import yaml
import csv
import numpy as np

class Household:
    def __init__(self, cbg, population, total_count):
        self.cbg = cbg
        self.population = population
        self.total_count = total_count

class Person:
    def __init__(self, age, cbg, hh_id, household, id, sex):
        self.age = age
        self.cbg = cbg
        self.hh_id = hh_id
        self.household = household
        self.id = id
        self.sex = sex

def construct_person(loader, node):
    fields = loader.construct_mapping(node)
    return Person(**fields)

def construct_household(loader, node):
    fields = loader.construct_mapping(node)
    return Household(**fields)

yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object:__main__.Person', construct_person)
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object:__main__.Household', construct_household)

def generate_interhousehold_movements(
        availability_matrix, household_info,
        individual_movement_frequency, social_event_frequency,
        regular_visitation_frequency, school_children_frequency
    ):
    num_timestamps = availability_matrix.shape[1]
    num_persons = availability_matrix.shape[0]

    result_array = np.zeros((num_persons, num_timestamps), dtype=int)

    for timestamp in range(num_timestamps):
        for person_id in range(num_persons):
            # Check if the person is available at the given timestamp
            if availability_matrix[person_id, timestamp] != 0:
                if person_id in household_info:
                    household = household_info[person_id]
                    cbg = household.cbg
                    household_members = [p.id for p in household.population]
        
                    # Generate individual movement
                    if np.random.rand() < individual_movement_frequency:
                        result_array[person_id, timestamp] = 1

                    # Generate social event visit
                    if np.random.rand() < social_event_frequency:
                        other_household_members = [
                            p for h in household_info if h.cbg != cbg for p in h.population
                        ]
                        selected_person = np.random.choice(other_household_members)
                        result_array[selected_person.id, timestamp] = 1

                    # Generate regular visitation
                    if np.random.rand() < regular_visitation_frequency:
                        selected_person = np.random.choice(household_members)
                        result_array[selected_person, timestamp] = 1

                    # Generate school children movement
                    if np.random.rand() < school_children_frequency and any(p.age < 19 for p in household.population):
                        result_array[person_id, timestamp] = 1

    return result_array

# Load the JSON data from the file
with open('result_poi.json', 'r') as file:
    data = json.load(file)

# Load the availability matrix
availability_matrix = np.genfromtxt('result_matrix.csv', delimiter=',', skip_header=1)

# Load household info from YAML
with open('households.yaml', 'r') as house_file:
    household_data = yaml.safe_load(house_file)

# Create a list of Household objects
household_info = []

for household_list in household_data:
    for household_dict in household_list:
        print(household_dict.cbg)
        cbg = household_dict.cbg  # Accessing the cbg attribute directly
        population_data = household_dict.population
        population = [
            Person(**person_data.__dict__) for person_data in population_data
        ]
        total_count = household_dict.total_count
        household_info.append(Household(cbg, population, total_count))


# Sample settings, adjust as needed
individual_movement_frequency = 0.2
social_event_frequency = 0.1
regular_visitation_frequency = 0.15
school_children_frequency = 0.3

result_interhousehold = generate_interhousehold_movements(
    availability_matrix, household_info,
    individual_movement_frequency,
    social_event_frequency,
    regular_visitation_frequency,
    school_children_frequency
)

csv_file_path_interhousehold = 'interhousehold_movements.csv'
with open(csv_file_path_interhousehold, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write header
    header = ['Person ID'] + [f'Timestamp {ts}' for ts in range(result_interhousehold.shape[1])]
    writer.writerow(header)

    # Write data
    for i in range(result_interhousehold.shape[0]):
        row = [i] + list(result_interhousehold[i, :])
        writer.writerow(row)

print(f"CSV file created for interhousehold movements: {csv_file_path_interhousehold}")
