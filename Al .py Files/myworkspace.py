from collections import deque
import numpy as np
import pandas as pd
import yaml
import json
import random
from household import Household, Person, Population


class POI():

    def __init__(self, name, visit, bucket, same_day, pop_hr, pop_day): # time_step field?
        bucket = json.loads(bucket)
        self.name = name
        temp_queue = [[] for i in range(500)]
        self.current_people = deque(temp_queue)
        self.visit = visit
        self.bucketed_dwell_time = bucket
        self.same_day_brands = same_day
        self.pop_hr = pop_hr
        self.pop_day = pop_day

    def add_person(self):
        values = ["<5", "5-10", "11-20", "21-60", "61-120", "121-240", ">240"]
        sum = self.bucketed_dwell_time["<5"] + self.bucketed_dwell_time["5-10"] + self.bucketed_dwell_time["11-20"] + self.bucketed_dwell_time["21-60"] + self.bucketed_dwell_time["61-120"] + self.bucketed_dwell_time["121-240"] + self.bucketed_dwell_time[">240"]
        weights = [self.bucketed_dwell_time["<5"]/sum, self.bucketed_dwell_time["5-10"]/sum, self.bucketed_dwell_time["11-20"]/sum, self.bucketed_dwell_time["21-60"]/sum, self.bucketed_dwell_time["61-120"]/sum, self.bucketed_dwell_time["121-240"]/sum, self.bucketed_dwell_time[">240"]/sum]


        random_string = random.choices(values, weights=weights)[0]
        if(random_string == "<5"):
            random_integer = random.randint(1, 4)
        elif(random_string == "5-10"):
            random_integer = random.randint(5, 10)
        elif(random_string == "11-20"):
            random_integer = random.randint(11, 20)
        elif(random_string == "21-60"):
            random_integer = random.randint(21, 60)
        elif(random_string == "61-120"):
            random_integer = random.randint(61, 120)
        elif(random_string == "121-240"):
            random_integer = random.randint(121, 240)
        elif(random_string == ">240"):
            random_integer = random.randint(241, 500)

        if random_integer > len(self.current_people):
            self.current_people.append(deque())
        
        else:
            self.current_people[random_integer - 1].append(random_integer)
        
        print(random_string)
        print(random_integer)
        print(self.current_people[random_integer - 1])

        #self.current_people.append(person)

    def _calculate_next_poi_weights(self, poi_dict):
        #Ex) If current POI is "American Heritage Bank", 
        #Same_day_brands is a dictionary that has Circle K: 50, Dollar General: 48, Phillips 66: 100
        #The brands can be either in state(within Barnsdall) or out state(outside Barnsdall)
        #If brand name is in poi_dict.keys(), it's instate, if not, it's outstate
        #In this example, only Dollar General is instate and the two others are out state

        #We will now have three possibilities for the next_poi, which is either Dollar General, out of state, or home
        #Out of state is considered one POI from now on as a whole

        #First, we will deal with outstate brands
        #outstate_sum is the sum of the weights of all outstate brands Ex) 50 + 100 = 150
        #outstate_count is the count of how many outstate brands exist Ex) 2 (Circle K and Phillips 66)
        #outstate_avg is the average of all the outstate weights Ex) 150 / 2 = 75
        outstate_sum = sum(self.same_day_brands[brand_name] for brand_name in self.same_day_brands.keys() if brand_name not in poi_dict.keys())
        outstate_count = sum(1 for brand_name in self.same_day_brands.keys() if brand_name not in poi_dict.keys())
        outstate_avg = outstate_sum / outstate_count

        #Now, we will consider instate brands
        #instate_sum is the sum of the weights of all instate brands Ex) 48
        #instate_count is the count of how many instate brands exist Ex) 1 (Dollar General)
        instate_sum = sum(self.same_day_brands[brand_name] for brand_name in self.same_day_brands.keys() if brand_name in poi_dict.keys())
        instate_count = sum(1 for brand_name in self.same_day_brands.keys() if brand_name in poi_dict.keys())

        #next_poi_count is the total of instant_count and the outstate POI Ex) 2 (Dollar General and out state)
        next_poi_count = instate_count + 1

        #next_poi_sum is the sum of instate_sum and the outstate avg Ex) 48 + 75 = 123
        next_poi_sum = outstate_avg + instate_sum

        #home_weight is the weight we will assign to the home POI where a person goes back home after leaving this POI
        #Ex) 123 / 2 = 61.5
        home_weight = next_poi_sum / next_poi_count

        #constant we used for a more realistic home weight 
        home_constant = 2
        #Modified home weight Ex) 61.5 / 2 = 30.75
        home_weight_modified = home_weight / home_constant

        #Total poi sum weight should also include the weight of the home POI Ex) 123 + 30.75 = 153.75
        next_poi_sum += home_weight_modified

        return instate_sum, outstate_avg, home_weight_modified, next_poi_sum

    def send_person(self, person, poi_dict):
        instate_sum, outstate_avg, home_weight_modified, next_poi_sum = self._calculate_next_poi_weights(poi_dict)
        
        # Append in-state POI names to next_poi_list
        next_poi_list = [brand_name for brand_name in self.same_day_brands.keys() if brand_name in poi_dict.keys()]

        # Considers all the POIs that are outside of the scope as “out of state” and append to next_poi_list.
        # (Even if there are multiple out-of-state POIs, we consider them as one POI)
        next_poi_list.extend(["out of state", "home"])

        # Calculate the weights of each of the POI (including “out of state” and “home”)
        # These weights tell how likely a person would go to that POI as their next destination
        # (higher the weight, the more likely a person would choose to go to that POI)
        next_poi_weights = [self.same_day_brands[brand_name] / next_poi_sum for brand_name in next_poi_list if brand_name in poi_dict.keys()]
        # Also calculate the weights of “out of state” and “home”
        next_poi_weights.extend([outstate_avg / next_poi_sum, home_weight_modified / next_poi_sum])

        # Randomly gets the next possible destination based on the weights we calculated.
        next_poi = random.choices(next_poi_list, weights=next_poi_weights)[0]

        print(next_poi_list)
        print(next_poi_weights)
        print(next_poi)

        # Returns the person who will leave this POI and the next destination(POI)
        return [person, next_poi]


def timestep(poi_dict):
    '''
        Calculates Each Timestep
    '''
    #print("timestep 1 result")
    # for poi in poi_dict.keys():
    #     print(poi)
        


def get_info(city_info):

    poi_dict = {}

    for poi_name in city_info:
        visit = city_info[poi_name]['raw_visit_counts']
        bucket = city_info[poi_name]['bucketed_dwell_times']
        same_day = city_info[poi_name]['related_same_day_brand']
        pop_hr = city_info[poi_name]['popularity_by_hour']
        pop_day = city_info[poi_name]['popularity_by_day']

        cur_poi = POI(poi_name, visit, bucket, same_day, pop_hr, pop_day)
        poi_dict[poi_name] = cur_poi

    return poi_dict


def simulation(settings, city_info):

    poi_dict = get_info(city_info)
    a = poi_dict["American Heritage Bank"]
    a.send_person(None, poi_dict)
    #poi all set

    #TODO delete, test code

    # for i in range(10):
    #     timestep(poi_dict)

if __name__=="__main__":

    print("main function loading")

    with open('simul_settings.yaml', mode="r") as settingstream:
        settings = yaml.full_load(settingstream)

    with open('barnsdall.yaml') as citystream:
        city_info = yaml.full_load(citystream)
    
    simulation(settings, city_info)



    