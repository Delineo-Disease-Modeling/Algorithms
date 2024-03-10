import json
import csv
import numpy as np
from household import Household, Person
import pandas

class InterHousehold:
    def __init__(self, hh_list:list[Household]):
        self.hh_list = hh_list
        self.people = []
        
        for hh in hh_list:
            self.people += hh.population    
        


        self.individual_movement_frequency = 0.2
        self.school_children_frequency = 0.3
        self.regular_visitation_frequency = 0.15

        self.social_event_frequency = 0.1
        self.social_max_size = 10
        
        

    def random_boolean(self, probability_of_true):
        """
        Generates a random boolean value based on the given probability of True.
        
        :param probability_of_true: A float representing the probability of returning True, between 0 and 1.
        :return: A boolean value, where True occurs with the given probability.
        """
        return np.random.random() < probability_of_true



    def next(self):
        self.individual_movement()
        self.social_event()

    
    def social_event(self):
        number = int(self.social_event_frequency * len(self.hh_list))
        hh_social = np.random.choice(self.hh_list, size=number, replace=False)
        for hh in hh_social:
            hh.social = True
            # randomly choose people from other households to gather
            guests_num = np.random.randint(1, high=self.social_max_size - len(hh.population), size=None, dtype='l')
            guests = np.random.choice(self.people, size=guests_num, replace=False)
            for guest in guests:
                if guest.current_household == guest.household:
                    guest.current_household = hh
                    guest.household.population.remove(guest)
                    hh.population.append(guest)




    def individual_movement(self):
        for person in self.people:
            if person.current_household != person.household: # person not in its hosuehold, he is a guest
                # put the person back to its original household
                person.current_household.population.remove(person)
                person.current_household = person.household
                person.household.population.append(person)
                
            else:
                move = self.random_boolean(self.individual_movement_frequency)
                if move:
                    hh = np.random.choice(self.hh_list, replace=False)
                    while (hh == person.current_household and len(self.hh_list) >= 2):
                        hh = np.random.choice(self.hh_list, replace=False)
                    
                    person.current_household = hh
                    hh.population.append(person)
                    







    

        
        



    def generate_interhousehold_movements(
            self,
            availability_matrix,
            household_info
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
                        if np.random.rand() < self.individual_movement_frequency:
                            result_array[person_id, timestamp] = 1

                        # Generate social event visit
                        if np.random.rand() < self.social_event_frequency:
                            other_household_members = [
                                p for h in household_info if h.cbg != cbg for p in h.population
                            ]
                            selected_person = np.random.choice(other_household_members)
                            result_array[selected_person.id, timestamp] = 1

                        # Generate regular visitation
                        if np.random.rand() < self.regular_visitation_frequency:
                            selected_person = np.random.choice(household_members)
                            result_array[selected_person, timestamp] = 1

                        # Generate school children movement
                        if np.random.rand() < self.school_children_frequency and any(p.age < 19 for p in household.population):
                            result_array[person_id, timestamp] = 1

        return result_array




# # Load the JSON data from the file
# with open('result_poi.json', 'r') as file:
#     data = json.load(file)

# # Load the availability matrix
# availability_matrix = np.genfromtxt('result_matrix.csv', delimiter=',', skip_header=1)

# # Load household info from YAML
# with open('households.yaml', 'r') as house_file:
#     household_data = yaml.safe_load(house_file)

# # Create a list of Household objects
# household_info = []

# for household_list in household_data:
#     for household_dict in household_list:
#         print(household_dict.cbg)
#         cbg = household_dict.cbg  # Accessing the cbg attribute directly
#         population_data = household_dict.population
#         population = [
#             Person(**person_data.__dict__) for person_data in population_data
#         ]
#         total_count = household_dict.total_count
#         household_info.append(Household(cbg, population, total_count))




# csv_file_path_interhousehold = 'interhousehold_movements.csv'
# with open(csv_file_path_interhousehold, 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)

#     # Write header
#     header = ['Person ID'] + [f'Timestamp {ts}' for ts in range(result_interhousehold.shape[1])]
#     writer.writerow(header)

#     # Write data
#     for i in range(result_interhousehold.shape[0]):
#         row = [i] + list(result_interhousehold[i, :])
#         writer.writerow(row)

# print(f"CSV file created for interhousehold movements: {csv_file_path_interhousehold}")