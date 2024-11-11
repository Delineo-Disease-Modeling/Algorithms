import random
import sys
import json
import numpy as np
from household import Person, Household
from inter_hh import InterHousehold
from papdata import Papdata

class Simulate:
    def __init__(self, settings, poi_info, location_info, hh_info):
        self.settings = settings
        self.poi_info = poi_info
        self.location_info = location_info
        self.hh_info = hh_info
        self.interhouse = InterHousehold(hh_info, settings)

        # Adjusted movement probability model parameters
        self.alpha = 0.05
        self.beta = 0.05
        self.gamma = 0.05
        self.delta = 0.05
        self.lambda_smoothing = 0.005

        print(f"Initialized movement probability model with alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")

        # Initialize POI indices and capacities
        self.poi_ids = list(self.poi_info.keys())
        self.poi_indices = {poi_id: i for i, poi_id in enumerate(self.poi_ids)}
        self.num_pois = len(self.poi_ids)

        # Initialize capacity C
        self.C = np.array([self.poi_info[poi_id]['capacity'] for poi_id in self.poi_ids])

        # Initialize S_prev, F_prev, A_prev
        self.S_prev = np.zeros(self.num_pois)
        self.F_prev = np.zeros(self.num_pois)
        self.A_prev = np.zeros(self.num_pois)

        # Initialize person last POI visit times
        self.person_last_poi_visit_time = {}  # Key: person.id, Value: last visit time

    def get_hh_info(self):
        hh_dict = {hh.id: hh for hh in self.hh_info}
        return hh_dict

    def compute_movement_probability(self, alpha, beta, gamma, C, S_prev, F_prev, A_prev, delta, lambda_smoothing):
        # Decay previous visits and apply smoothing
        F_prev_adjusted = delta * F_prev
        A_prev_adjusted = lambda_smoothing * A_prev

        # Calculate movement probability M based on current parameters
        M = np.abs(alpha * (C - S_prev) + beta * F_prev_adjusted + gamma * A_prev_adjusted)
        M_sum = np.sum(M)
        if M_sum > 0:
            M /= M_sum  # Normalize
        else:
            M = np.ones_like(M) / len(M)  # Assign equal probability if sum is zero
        return M

    def day_to_day(self, hh_dict, person, time):
        # Compute movement probabilities M
        M = self.compute_movement_probability(
            self.alpha, self.beta, self.gamma,
            self.C, self.S_prev, self.F_prev, self.A_prev,
            self.delta, self.lambda_smoothing
        )

        # Randomly choose a POI based on M
        poi_index = np.random.choice(self.num_pois, p=M)
        poi_id = self.poi_ids[poi_index]

        # Assign person to the POI
        arrival_time = time
        self.poi_info[poi_id].setdefault('current_people', []).append((person, arrival_time))

        # Update F_prev
        self.F_prev[poi_index] += 1

        # Update A_prev if necessary
        self.A_prev = self.F_prev.copy()

    def timestep(self, clock, poi_dict, hh_dict, time):
        removals = []

        cooldown_period = 10  # Timesteps before a person can go to a POI again

        # Iterate through each household and its population
        for hh_id, curr_hh in hh_dict.items():
            for person in curr_hh.population[:]:  # Use a copy of the list
                is_working = self.move_to_work(clock, poi_dict, person)
                if is_working:
                    # Mark person for removal after processing
                    removals.append((curr_hh, person))
                else:
                    # Get the last visit time from the dictionary; default to negative infinity if not found
                    last_visit_time = self.person_last_poi_visit_time.get(person.id, -float('inf'))
                    if (person.availability and 
                        random.random() <= 0.02 and 
                        person.hh_id == person.location.id and
                        (time - last_visit_time) >= cooldown_period):
                        # 2% chance to go to a POI if cooldown period has passed
                        self.day_to_day(hh_dict, person, time)
                        removals.append((curr_hh, person))

            # Remove persons who have moved out of the household
            for curr_hh, person in removals:
                if person in curr_hh.population:
                    curr_hh.population.remove(person)
            removals.clear()

        # Update POIs by removing people as needed
        self.update_pois(clock, poi_dict, hh_dict, time)

        # Handle inter-household movements
        self.interhouse.next()

        # Update S_prev with current POI populations
        for i, poi_id in enumerate(self.poi_ids):
            poi_population = len(self.poi_info[poi_id].get('current_people', []))
            self.S_prev[i] = poi_population

        # Update A_prev if necessary
        self.A_prev = self.F_prev.copy()

        # Calculate populations
        total_poi_population = sum(len(poi.get('current_people', [])) for poi in poi_dict.values())
        total_hh_population = sum(len(hh.population) for hh in hh_dict.values())

        # Debugging output
        print(f"{clock // 60}:{clock % 60:02d} - Households: {total_hh_population}, POIs: {total_poi_population}")

        return poi_dict, hh_dict

    def update_pois(self, clock, poi_dict, hh_dict, time):
        for poi_id, poi_details in poi_dict.items():
            current_people = poi_details.get('current_people', [])
            updated_people = []
            for person_info in current_people:
                person, arrival_time = person_info
                dwell_time = random.randint(5, 15)  # Random dwell time between 5 and 15 timesteps
                if time - arrival_time >= dwell_time:
                    # Person returns home
                    home_hh = hh_dict.get(person.hh_id)
                    if home_hh:
                        home_hh.population.append(person)
                        # Update last visit time in the dictionary
                        self.person_last_poi_visit_time[person.id] = time
                else:
                    # Person stays at the POI
                    updated_people.append(person_info)
            # Update the 'current_people' list for the POI
            poi_details['current_people'] = updated_people

    def move_to_work(self, clock, poi_dict, person):
        # Placeholder for work movement logic
        return False

    def assign_work_poi(self, occupation):
        # Placeholder for assigning a work POI based on occupation
        return None

    def start(self):
        time = 0
        clock = self.settings['start_time'] * 60
        hh_dict = self.get_hh_info()
        poi_dict = self.poi_info.copy()

        hh_return_dict = {}
        poi_return_dict = {}

        for i in range(self.settings['time']):
            poi_dict, hh_dict = self.timestep(clock, poi_dict, hh_dict, time)
            time += 1
            clock = (self.settings['start_time'] * 60 + time) % 1440

            # Capture data at specified intervals
            if time % 60 == 0 or time == 0:
                self.capture_timestep_data(time, hh_dict, poi_dict, hh_return_dict, poi_return_dict)

        self.write_results(hh_return_dict, poi_return_dict)

    def capture_timestep_data(self, time, hh_dict, poi_dict, hh_return_dict, poi_return_dict):
        hh_ret = {}
        for hh in hh_dict.values():
            pop_list = [person.to_dict() for person in hh.population]
            hh_ret[f"household_{hh.id}"] = pop_list

        poi_ret = {}
        for poi_id, poi_details in poi_dict.items():
            pop_list_poi = [person.to_dict() for person, arrival_time in poi_details.get('current_people', [])]
            poi_ret[f"poi_{poi_id}"] = pop_list_poi

        hh_return_dict[f'timestep_{time}'] = hh_ret
        poi_return_dict[f'timestep_{time}'] = poi_ret

    def write_results(self, hh_return_dict: dict, poi_return_dict: dict):
        with open("output/result_hh.json", "w", encoding='utf-8') as hhstream:
            json.dump(hh_return_dict, hhstream, ensure_ascii=False, indent=4)

        with open("output/result_poi.json", "w", encoding='utf-8') as poistream:
            json.dump(poi_return_dict, poistream, ensure_ascii=False, indent=4)
