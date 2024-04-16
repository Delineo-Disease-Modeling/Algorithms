import json
from collections import deque
import random

class POI():

    def __init__(self, name, visit, bucket, same_day, pop_hr, pop_day, simul_time = 600):  # time_step field?
        bucket = json.loads(bucket)
        self.name = name
        temp_queue = [[] for i in range(simul_time)]
        self.current_people = deque(temp_queue)
        self.visit = visit
        self.bucketed_dwell_time = bucket
        self.same_day_brands = same_day
        self.pop_hr = pop_hr
        self.pop_day = pop_day
        self.population = 0

    def add_person_to_none_work(self, person):
        values = ["<5", "5-10", "11-20", "21-60", "61-120", "121-240", ">240"]
        sum = self.bucketed_dwell_time["<5"] + self.bucketed_dwell_time["5-10"] + self.bucketed_dwell_time["11-20"] + \
            self.bucketed_dwell_time["21-60"] + self.bucketed_dwell_time["61-120"] + \
            self.bucketed_dwell_time["121-240"] + \
            self.bucketed_dwell_time[">240"]
        weights = [self.bucketed_dwell_time["<5"]/sum, self.bucketed_dwell_time["5-10"]/sum, self.bucketed_dwell_time["11-20"]/sum,
                   self.bucketed_dwell_time["21-60"]/sum, self.bucketed_dwell_time["61-120"]/sum, self.bucketed_dwell_time["121-240"]/sum, self.bucketed_dwell_time[">240"]/sum]

        random_string = random.choices(values, weights=weights)[0]
        if (random_string == "<5"):
            random_integer = random.randint(1, 4)
        elif (random_string == "5-10"):
            random_integer = random.randint(5, 10)
        elif (random_string == "11-20"):
            random_integer = random.randint(11, 20)
        elif (random_string == "21-60"):
            random_integer = random.randint(21, 60)
        elif (random_string == "61-120"):
            random_integer = random.randint(61, 120)
        elif (random_string == "121-240"):
            random_integer = random.randint(121, 240)
        elif (random_string == ">240"):
            random_integer = random.randint(241, 500)

        if random_integer > len(self.current_people):
            self.current_people.append(deque())

        else:
            self.current_people[random_integer - 1].append(person)

        self.population += 1

    def add_person_to_work(self, time, person):
        if time > len(self.current_people):
            self.current_people.append(deque())
        else:
            self.current_people[time - 1].append(person)

        self.population += 1
        person.availability = False

    def remove_people(self, clock, poi_dict):
        self.current_people.rotate(-1)

        popped_people = self.current_people[len(self.current_people) - 1]
        self.current_people[-1] = []

        for person in popped_people:
            if person.occupation != None and person.left_from_work:
                work_end_time = (person.work_time[1] if person.work_time[1] > person.work_time[0] else (person.work_time[1] + 24)) * 60
                poi_dict[person.occupation].add_person_to_work(work_end_time - clock, person) #poi to poi
                person.left_from_work = False
            else:
                person.household.add_member(person) #poi to home
                person.availability = True
            self.population -= 1
        
        if (720 <= clock and clock <= 780) or (1050 <= clock and clock <= 1140): #if lunch time (12 - 13) or dinner time (1730 - 1900) 
            for people in self.current_people:
                for person in people:
                    if not person.availability: self.break_from_work(poi_dict, self, people, person)

    def break_from_work(self, poi_dict, curr_poi, people, person):
        person, next_poi = curr_poi.next_poi(person, poi_dict)
        if next_poi != None:
            person.left_from_work = True
            people.remove(person) #pop from curr_poi
            curr_poi.population -= 1
            poi_dict[next_poi].add_person_to_none_work(person) #add to next_poi

    def next_poi(self, person, poi_dict):
        instate_sum = 0
        #outstate_sum = 0
        #outstate_count = 0
        next_poi_list = []

        for brand_name in self.same_day_brands.keys():
            if brand_name in poi_dict.keys():
                instate_sum += self.same_day_brands[brand_name]
                next_poi_list.append(brand_name)
            #else:
                #outstate_sum += self.same_day_brands[brand_name]
                #outstate_count += 1

        #outstate_avg = outstate_sum / outstate_count if outstate_count != 0 else 1
        #next_poi_sum = outstate_avg + instate_sum

        #next_poi_list.append("out of state")

        next_poi_weights = []
        for brand_name in next_poi_list:
            if brand_name in poi_dict.keys():
                next_poi_weights.append(self.same_day_brands[brand_name])
            else:
                continue

        #next_poi_weights.append(outstate_avg / next_poi_sum)

        if next_poi_list == []:
            next_poi = None
        else:
            next_poi = random.choices(next_poi_list, weights=next_poi_weights)[0]

        return [person, next_poi]
    
    def next_poi_or_home(self, person, poi_dict):
        instate_sum = 0
        next_poi_count = 1  # bc outstate is already a part of next poi list
        outstate_sum = 0
        outstate_count = 0
        home_constant = 2
        next_poi_list = []

        for brand_name in self.same_day_brands.keys():
            if brand_name in poi_dict.keys():
                instate_sum += self.same_day_brands[brand_name]
                next_poi_count += 1
                next_poi_list.append(brand_name)
            else:
                outstate_sum += self.same_day_brands[brand_name]
                outstate_count += 1

        outstate_avg = outstate_sum / outstate_count if outstate_count != 0 else 1
        next_poi_sum = outstate_avg + instate_sum
        home_weight = next_poi_sum / next_poi_count if next_poi_count != 0 else 1
        home_weight_modified = home_weight / home_constant

        next_poi_list.append("out of state")
        next_poi_list.append("home")

        # final total sum
        next_poi_sum += home_weight_modified
        next_poi_weights = []
        for brand_name in next_poi_list:
            if brand_name in poi_dict.keys():
                next_poi_weights.append(
                    self.same_day_brands[brand_name] / next_poi_sum)
            else:
                continue

        next_poi_weights.append(outstate_avg / next_poi_sum)
        next_poi_weights.append(home_weight_modified / next_poi_sum)

        next_poi = random.choices(next_poi_list, weights=next_poi_weights)[0]

        return [person, next_poi]