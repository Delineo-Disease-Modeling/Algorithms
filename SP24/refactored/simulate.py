from poi import POI
import random
import sys
import json 
import csv
from household import Person, Household, poi_category, age_category
from inter_hh import InterHousehold

category_weight = {
    'Agriculture, Forestry, Fishing and Hunting': 20,
    'Mining, Quarrying, and Oil and Gas Extraction': 50,
    'Utilities': 10,
    'Construction': 50,
    'Manufacturing': 50,
    'Wholesale Trade': 20,
    'Retail Trade': 20,
    'Transportation and Warehousing': 20,
    'Information': 30,
    'Depository Credit Intermediation': 20,
    'Real Estate and Rental and Leasing': 20,
    'Professional, Scientific, and Technical Services': 20,
    'Management of Companies and Enterprises': 10,
    'Administrative and Support and Waste Management and Remediation Services': 10,
    'Education': 50,
    'Medical': 25,
    'Arts, Entertainment, and Recreation': 10,
    'Restaurants and Other Eating Places': 15,
    'Others': 20,
    'Public Administration': 20
}

class Simulate:

    def __init__(self, settings, city_info, hh_info, category_info):
        self.settings = settings
        self.city_info = city_info
        self.hh_info = hh_info
        self.category_info = category_info
        self.interhouse = InterHousehold(hh_info)

    def get_city_info(self):

        poi_dict = {}

        for poi_name in self.city_info:
            visit = self.city_info[poi_name]['raw_visit_counts']
            bucket = self.city_info[poi_name]['bucketed_dwell_times']
            same_day = self.city_info[poi_name]['related_same_day_brand']
            pop_hr = self.city_info[poi_name]['popularity_by_hour']
            pop_day = self.city_info[poi_name]['popularity_by_day']

            cur_poi = POI(poi_name, visit, bucket, same_day, pop_hr, pop_day, self.settings['time'])
            poi_dict[poi_name] = cur_poi

        return poi_dict

    def get_hh_info(self):

        hh_dict = {}
        for hh_id in self.hh_info:
            hh_dict[hh_id] = hh_id

        return hh_dict


    def get_popularity_matrix(self, poi_dict):

        name = list(poi_dict.keys())
        weights = []

        for poi_name in poi_dict.keys():
            weights.append(poi_dict[poi_name].visit)

        return [name, weights]
    
    '''
        Role Based Movement Pattern
    '''
    def analyze_category(self):
        category_dict = {}
        with open(self.category_info, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                id = row['placekey']
                name = row['location_name']
                naics = row['naics_code']
                if naics == '': continue
                category = poi_category[naics[:2]]

                #match category to location
                category_dict[name] = category

        return category_dict
    
    def distribute_occupation(self, hh_dict, poi_dict, category_dict):
        #count each category occurance
        category_count = {}
        occupation_count = {}
        for poi in poi_dict:
            category = category_dict[poi]

            if poi in occupation_count:
                occupation_count[poi] += 1
            else:
                occupation_count[poi] = 1

            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1

        #calculate the number of individuals to assign to each occupation category
        total_population = sum(len(household.population) for household in hh_dict.keys())
        total_count = sum(occupation_count.values())
        total_weight = sum(category_weight.values())

        occupation_population = {}
        for poi in poi_dict:
            category = category_dict[poi]
            weight = category_weight[category]
            #count = occupation_count[poi] #commented since all occupations are not redundunt in this case
            occupation_population[poi] = int(weight / (total_weight / len(category_weight)) * (total_population / total_count))

        hhlist = hh_dict.keys()

        for hh in hhlist:
            for person in hh.population:
                occupation = self.select_occupation(person, category_dict, occupation_population)
                if occupation is None: continue
                occupation_population[occupation] -= 1
                person.set_occupation(occupation)

    def select_occupation(self, person, category_dict, occupation_population):
        occupations = list(occupation_population.keys())
        # Choose from occupations that still have individuals to assign
        available_occupations = [occ for occ in occupations if occupation_population[occ] > 0]

        for category, (start_age, end_age) in age_category.items():
            if start_age <= person.age <= end_age:
                if category == "Adolescent":
                    available_occupations = [occ for occ in available_occupations if category_dict[occ] == 'Education']
                    break
                elif category == "Adult":
                    if random.random() <= (3.9 + 10) / 100: return None #unemployed + virtual worker
                elif category == "Preschool" or category == "Retired":
                    return None

        if available_occupations == []: return None
        occupation = random.choice(available_occupations)

        return occupation
        
    def move_to_work(self, clock, curr_hh, poi_dict, person):
        if person.occupation is not None:
            work_start_time = person.work_time[0] * 60
            work_end_time = (person.work_time[1] if person.work_time[1] > person.work_time[0] else (person.work_time[1] + 24)) * 60
            shouldWork = work_start_time <= clock and clock < work_end_time
            if shouldWork and person.hh_id == person.location.id:
                poi_dict[person.occupation].add_person_to_work(work_end_time - clock, person)
                return False  # should be removed from their household
        return True  # No need to remove person from household

    def day_to_day(self, curr_hh, poi_dict, person):
        random_poi_name = random.choice(list(poi_dict.keys()))
        poi_dict[random_poi_name].add_person_to_none_work(person)
        return False


    def timestep(self, clock, poi_dict, hh_dict, popularity_matrix):
        '''
            Movement of people in each timestep
        '''
        removals = []


        # Role-based Movement
        for hh_id, curr_hh in hh_dict.items():
            for person in curr_hh.population:
                can_go_out = self.move_to_work(clock, curr_hh, poi_dict, person)
                if not can_go_out:
                # Mark person for removal after processing
                    removals.append((curr_hh, person))
                elif person.availability and random.random() <= 0.10 and person.hh_id == person.location.id: #10% chance to go to random POIs
                    self.day_to_day(curr_hh, poi_dict, person)
                    removals.append((curr_hh, person))

            for curr_hh, person in removals:
                if person in curr_hh.population:
                    curr_hh.population.remove(person)


        # POIs
        for poi in poi_dict.keys():
            curr_poi = poi_dict[poi]
            curr_poi.remove_people(clock, poi_dict)
                   


        # Interhouse Movement
        #self.interhouse.next()


        '''
            Get each population for each time step
        '''


        total_poi_population = 0
        each_poi_population = []
        total_hh_population = sum(len(household.population) for household in hh_dict.values())
        for poi in poi_dict.values():
            total_poi_population += poi.population
            each_poi_population.append(poi.population)
        print(clock // 60, ":", f"{clock%60:02d}")
        print("hh:  ", total_hh_population, "poi: ", total_poi_population, " => ", each_poi_population)


        return poi_dict, hh_dict



    def start(self):

        time = 0
        clock = self.settings['start_time'] * 60 #current time in minutes ex) 10AM == 600

        poi_dict = self.get_city_info()
        hh_dict = self.get_hh_info()
        
        category_dict = self.analyze_category()
        self.distribute_occupation(hh_dict, poi_dict, category_dict)

        hh_return_dict = {}
        poi_return_dict = {}

        popularity_matrix = self.get_popularity_matrix(poi_dict)

        for i in range(self.settings['time']):
            poi_dict, hh_dict = self.timestep(clock, poi_dict, hh_dict, popularity_matrix)
            time += 1
            clock = self.settings['start_time'] * 60 + time

            # Print info!
            old_out = sys.stdout

            class PersonEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Person):
                        return {
                            'id': obj.id,
                            'sex': obj.sex,
                            'age': obj.age,
                            'cbg': obj.cbg,
                            'household': obj.household,
                            'hh_id': obj.hh_id,
                        }
                    return super().default(obj)

            class HouseholdEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Household):
                        return {
                            'cbg': obj.cbg,
                            'population': obj.population,
                            'total_count': obj.total_count,
                            'members': tuple([self.default(p) for p in obj.population])
                        }
                    elif isinstance(obj, Person):
                        return {
                            'id': obj.id,
                            'sex': obj.sex,
                            'age': obj.age,
                            'cbg': obj.cbg,
                            'household': obj.household,
                            'hh_id': obj.hh_id,

                        }
                    return super().default(obj)

            class POIEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, POI):
                        return {
                            'current_people': tuple([PersonEncoder().default(p) for p in obj.current_people]),
                        }
                    elif isinstance(obj, Person):
                        return HouseholdEncoder().default(obj)
                    elif isinstance(obj, Household):
                        return HouseholdEncoder().default(obj)
                    return super().default(obj)

            # Print info!
            old_out = sys.stdout
            if time % 60 == 0.0 or time == 0:

                count = 1
                poi_count = 1

                hh_ret = {}
                for hh, pop in hh_dict.items():
                    pop_list = []
                    new_pop = pop.population.copy()
                    for person in new_pop:
                        person_list = vars(person).copy()
                        person_list.pop('household')
                        pop_list.append(person_list)
                        # print(pop_list)
                    hh_ret[f"household_{count}"] = pop_list
                    count += 1

                hh_return_dict[f'timestep_{time}'] = hh_ret

                poi_ret = {}
                for poi, cur_poi in poi_dict.items():
                    pop_list_poi = []
                    new_pop_poi = list(cur_poi.current_people.copy())
                    for spot in new_pop_poi:
                        spot_list = []
                        for person in spot:
                            person_list_poi = vars(person).copy()
                            person_list_poi.pop('household')
                            spot_list.append(person_list_poi)
                        pop_list_poi.append(spot_list)

                    poi_ret[f"id_{poi_count}_{cur_poi.name}"] = pop_list_poi
                    poi_count += 1
                poi_return_dict[f'timestep_{time}'] = poi_ret

        self.write(hh_return_dict, poi_return_dict)
    
    def write(self, hh_return_dict:dict, poi_return_dict:dict):
        # Convert all custom objects to dictionaries using to_dict method
        def convert_to_dict(obj):
            if isinstance(obj, Person) or isinstance(obj, Household):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {key: convert_to_dict(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(element) for element in obj]
            else:
                return obj

        hh_return_dict_converted = convert_to_dict(hh_return_dict)
        poi_return_dict_converted = convert_to_dict(poi_return_dict)

        with open("output/result_hh.json", "w+", encoding='utf-8') as hhstream:
            json.dump(hh_return_dict_converted, hhstream, ensure_ascii=False, indent=4)

        with open("output/result_poi.json", "w+", encoding='utf-8') as poistream:
            json.dump(poi_return_dict_converted, poistream, ensure_ascii=False, indent=4)
