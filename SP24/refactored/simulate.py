from poi import POI
import random
import sys
import json 
import csv
from household import Person, Household

category_weight = {
    'Agriculture, Forestry, Fishing and Hunting': 20,
    'Mining, Quarrying, and Oil and Gas Extraction': 100,
    'Utilities': 5,
    'Construction': 50,
    'Manufacturing': 200,
    'Wholesale Trade': 20,
    'Retail Trade': 20,
    'Transportation and Warehousing': 20,
    'Information': 30,
    'Finance and Insurance': 20,
    'Real Estate and Rental and Leasing': 20,
    'Professional, Scientific, and Technical Services': 20,
    'Management of Companies and Enterprises': 10,
    'Administrative and Support and Waste Management and Remediation Services': 10,
    'Educational Services': 3000,
    'Health Care and Social Assistance': 25,
    'Arts, Entertainment, and Recreation': 10,
    'Accommodation and Food Services': 15,
    'Other Services (except Public Administration)': 20,
    'Public Administration': 20
}

class Simulate:

    def __init__(self, settings, city_info, hh_info, category_info):
        self.settings = settings
        self.city_info = city_info
        self.hh_info = hh_info
        self.category_info = category_info

    def get_city_info(self):

        poi_dict = {}

        for poi_name in self.city_info:
            visit = self.city_info[poi_name]['raw_visit_counts']
            bucket = self.city_info[poi_name]['bucketed_dwell_times']
            same_day = self.city_info[poi_name]['related_same_day_brand']
            pop_hr = self.city_info[poi_name]['popularity_by_hour']
            pop_day = self.city_info[poi_name]['popularity_by_day']

            cur_poi = POI(poi_name, visit, bucket, same_day, pop_hr, pop_day)
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
    def create_category_dictionary(self):
        category_dict = {}
        with open(self.category_info, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                location_name = row['location_name']
                top_category = row['top_category']
                category_dict[location_name] = top_category
        return category_dict

    def distribute_occupation(self, poi_dict, category_dict):
        category_count = {}
        for poi_name in poi_dict.keys():
            category = category_dict[poi_name]
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1

        total_population = sum(len(household.population) for household in self.hh_dict.keys())
        total_count = sum(self.category_count.values())
        total_weight = sum(self.category_weights.values())

        # Calculate the number of individuals to assign to each occupation category
        occupation_counts = {}
        for category, count in self.category_count.items():
            weight = self.category_weights[category]
            occupation_counts[category] = int(count / total_count * total_population * weight / total_weight)

        for hh in self.hh_dict.keys():
            for person in hh.population:
                occupation = self.select_occupation(occupation_counts)
                person.set_occupation(occupation)
                occupation_counts[occupation] -= 1

    def select_occupation(self, occupation_counts):
        occupations = list(occupation_counts.keys())
        # Choose from occupations that still have individuals to assign
        available_occupations = [occ for occ in occupations if occupation_counts[occ] > 0]
        return random.choice(available_occupations)
    


    def timestep(self, poi_dict, hh_dict, popularity_matrix):
        '''
            Calculates Each Timestep
        '''

        '''
            Releasing people from households TODO: categorize people
        '''
        for hh in hh_dict.keys():
            cur_hh = hh_dict[hh]
            for person in cur_hh.population:
                # TODO 집에서 나갈 확률
                # if random.choices([True, False], [1, 10])[0]:
                #     target_poi = random.choices(
                #         popularity_matrix[0], popularity_matrix[1])[0]
                poi_dict[person.occupation].add_person_to_work(person)
                cur_hh.population.remove(person) 


        '''
            Movement of people in each timestep
        '''
        for poi in poi_dict.keys():
            cur_poi = poi_dict[poi]
            cur_poi.current_people.rotate(-1)

            popped_people = cur_poi.current_people[-1]
            cur_poi.current_people[-1] = []

            for person in popped_people:
                person, target = cur_poi.send_person(person, poi_dict)
                if target == "home":
                    person.household.add_member(person)
                elif target == "out of state":
                    person.household.add_member(person)
                else:
                    poi_dict[target].add_person(person)

        return poi_dict, hh_dict



    def start(self):

        time = 0

        poi_dict = self.get_city_info()
        hh_dict = self.get_hh_info()
        
        category_dict = self.create_category_dictionary()
        self.distribute_occupation(poi_dict, category_dict)

        # print(hh_dict)
        # for key, value in hh_dict.items():
        #     print(f"Key: {key}, Value: {value}")

        hh_return_dict = {}
        poi_return_dict = {}

        popularity_matrix = self.get_popularity_matrix(poi_dict)

        for i in range(self.settings['time']):
            print("timestep" + str(time))
            poi_dict, hh_dict = self.timestep(poi_dict, hh_dict, popularity_matrix)
            time += 1

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
    
    def write(self, hh_return_dict, poi_return_dict):
        with open("output/result_hh.json", "w+", encoding='utf-8') as hhstream:
            json.dump(hh_return_dict, hhstream, ensure_ascii=False, indent=4)

        with open("output/result_poi.json", "w+", encoding='utf-8') as poistream:
            json.dump(poi_return_dict, poistream, ensure_ascii=False, indent=4)
