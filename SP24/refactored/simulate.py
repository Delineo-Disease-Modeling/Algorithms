import random
import sys
import json
from household import Person, Household
from inter_hh import InterHousehold
from papdata import Papdata

# Assuming Papdata is still relevant; if not, you can remove it.
# If POI-related classes or functions are removed, ensure to adjust or remove related imports and usages.

class Simulate:
    def __init__(self, settings, poi_info, location_info, hh_info):
        self.settings = settings
        self.poi_info = poi_info  # This is now a dictionary from preprocess_data.py
        self.location_info = location_info
        self.hh_info = hh_info
        self.interhouse = InterHousehold(hh_info, settings)

        # Initialize Papdata if still needed
        # Make sure the file path and usage align with your current setup
        # Uncomment if Papdata is still in use
        # papdata = Papdata(self.hh_info, f'input/{settings["town"]}.pois.csv')
        # papdata.generate()

    def get_hh_info(self):
        """
        Convert hh_info list to a dictionary with household IDs as keys.
        Assumes each Household object has a unique 'id' attribute.
        """
        hh_dict = {hh.id: hh for hh in self.hh_info}
        return hh_dict

    def day_to_day(self, hh_dict, person):
        """
        Assign a person to a random POI based on poi_info.
        """
        if not self.poi_info:
            return  # No POIs available

        random_poi_id = random.choice(list(self.poi_info.keys()))
        # Assuming poi_info[random_poi_id] is a dictionary with relevant POI details
        # You can add the person to the POI's population list
        self.poi_info[random_poi_id].setdefault('current_people', []).append(person)
        return False  # Return value can be adjusted based on your logic

    def timestep(self, clock, poi_dict, hh_dict):
        """
        Handle the movement of people for each timestep.
        """
        removals = []

        # Iterate through each household and its population
        for hh_id, curr_hh in hh_dict.items():
            for person in curr_hh.population:
                is_working = self.move_to_work(clock, poi_dict, person)
                if is_working:
                    # Mark person for removal after processing
                    removals.append((curr_hh, person))
                elif person.availability and random.random() <= 0.10 and person.hh_id == person.location.id:
                    # 10% chance to go to a random POI
                    self.day_to_day(hh_dict, person)
                    removals.append((curr_hh, person))

            # Remove persons who have moved out of the household
            for curr_hh, person in removals:
                if person in curr_hh.population:
                    curr_hh.population.remove(person)

        # Update POIs by removing people as needed
        for poi_id, poi_details in poi_dict.items():
            # Implement any POI-specific logic here, such as removing people based on time
            # For example:
            # self.remove_people_from_poi(clock, poi_details)
            pass  # Placeholder for POI update logic

        # Handle inter-household movements
        self.interhouse.next()

        # Calculate populations
        total_poi_population = sum(len(poi.get('current_people', [])) for poi in poi_dict.values())
        total_hh_population = sum(len(hh.population) for hh in hh_dict.values())

        # Debugging output
        print(f"{clock // 60}:{clock % 60:02d} - Households: {total_hh_population}, POIs: {total_poi_population}")

        return poi_dict, hh_dict

    def move_to_work(self, clock, poi_dict, person):
        """
        Placeholder for logic to move a person to work.
        Implement your own logic based on person attributes and POIs.
        """
        # Example logic:
        if person.occupation and random.random() < 0.5:
            work_poi_id = self.assign_work_poi(person.occupation)
            if work_poi_id:
                self.poi_info[work_poi_id].setdefault('current_people', []).append(person)
                return True
        return False

    def assign_work_poi(self, occupation):
        """
        Assign a POI based on occupation. This is a placeholder and should be customized.
        """
        # Filter POIs based on occupation category if applicable
        # For simplicity, assign randomly
        if not self.poi_info:
            return None
        return random.choice(list(self.poi_info.keys()))

    def start(self):
        """
        Run the simulation based on the provided settings.
        """
        time = 0
        clock = self.settings['start_time'] * 60  # Current time in minutes (e.g., 10 AM == 600)
        hh_dict = self.get_hh_info()
        poi_dict = self.poi_info.copy()  # Create a working copy of POI info

        hh_return_dict = {}
        poi_return_dict = {}

        for i in range(self.settings['time']):
            poi_dict, hh_dict = self.timestep(clock, poi_dict, hh_dict)
            time += 1
            clock = (self.settings['start_time'] * 60 + time) % 1440  # Wrap around after 24 hours

            # Capture data at specified intervals (e.g., every 60 timesteps)
            if time % 60 == 0 or time == 0:
                self.capture_timestep_data(time, hh_dict, poi_dict, hh_return_dict, poi_return_dict)

        self.write_results(hh_return_dict, poi_return_dict)

    def capture_timestep_data(self, time, hh_dict, poi_dict, hh_return_dict, poi_return_dict):
        """
        Capture and store the current state of households and POIs.
        """
        hh_ret = {}
        for hh in hh_dict.values():
            pop_list = [person.to_dict() for person in hh.population]
            hh_ret[f"household_{hh.id}"] = pop_list

        poi_ret = {}
        for poi_id, poi_details in poi_dict.items():
            pop_list_poi = [person.to_dict() for person in poi_details.get('current_people', [])]
            poi_ret[f"poi_{poi_id}"] = pop_list_poi

        hh_return_dict[f'timestep_{time}'] = hh_ret
        poi_return_dict[f'timestep_{time}'] = poi_ret

    def write_results(self, hh_return_dict: dict, poi_return_dict: dict):
        """
        Write the simulation results to JSON files.
        """
        with open("output/result_hh.json", "w", encoding='utf-8') as hhstream:
            json.dump(hh_return_dict, hhstream, ensure_ascii=False, indent=4)

        with open("output/result_poi.json", "w", encoding='utf-8') as poistream:
            json.dump(poi_return_dict, poistream, ensure_ascii=False, indent=4)
