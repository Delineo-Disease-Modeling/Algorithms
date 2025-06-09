import csv
import json
from collections import deque
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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
        person.availability = False

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
      
class Person:
    def __init__(self, id=None, home=None, sex=None, age=None) -> None:
        self.id = id              # 人员 ID
        self.home = home          # 家庭 ID
        self.sex = sex            # 性别
        self.age = age            # 年龄
        self.is_poi = False       # 是否在 POI
        self.visited = {}         # 访问过的 POI 和次数
        self.total_visited = 0    # 总访问次数
        self.curr_poi = ""        # 当前 POI
        self.hour_stayed = 0      # 在当前 POI 停留的小时数

    def visit(self, poi: str):
        if poi in self.visited:
            self.visited[poi] += 1
        else:
            self.visited[poi] = 1
        self.total_visited += 1
        self.is_poi = True
        self.curr_poi = poi
        self.hour_stayed = 1

    def leave(self):
        self.is_poi = False
        self.hour_stayed = 0
        self.curr_poi = ""

    def stay(self):
        self.hour_stayed += 1

    def at_home(self) -> bool:
        """Check if the person is at home (not in a POI)."""
        return not self.is_poi

    def to_dict(self):
        """Convert Person to dictionary for serialization."""
        return {
            "id": self.id,
            "home": self.home,
            "sex": self.sex,
            "age": self.age,
            "is_poi": self.is_poi,
            "curr_poi": self.curr_poi,
            "hour_stayed": self.hour_stayed,
            "total_visited": self.total_visited,
            "visited": self.visited
        }

    def __repr__(self) -> str:
        return (f"Person(id={self.id}, home={self.home}, is_poi={self.is_poi}, "
                f"curr_poi='{self.curr_poi}', hour_stayed={self.hour_stayed}, "
                f"total_visited={self.total_visited}, visited={self.visited})")
        
class POIs:
    def __init__(self, pois_dict, alpha=0.1, occupancy_weight=1.0, tendency_decay=0.5):
        self.alpha = alpha
        self.occupancy_weight = occupancy_weight
        self.tendency_decay = tendency_decay
        # pois = [poi_id, ...]
        self.pois = list(pois_dict.keys())
        # pois_id_to_index = {poi_id: index}
        self.poi_id_to_index = {poi_id: index for index, poi_id in enumerate(self.pois)}
        # raw_visit_counts = {poi_id: raw_visit_counts}
        self.raw_visit_counts = {poi_id: pois_dict[poi_id]['raw_visit_counts'] for poi_id in pois_dict}
        # raw_visitor_counts = {poi_id: raw_visitor_counts}
        self.raw_visitor_counts = {poi_id: pois_dict[poi_id]['raw_visitor_counts'] for poi_id in pois_dict}
        # capacities = [{poi_id: capacity} for 30 days]
        self.capacities = [{poi_id: pois_dict[poi_id]['visits_by_day'][i] for poi_id in pois_dict} for i in range(30)]
        # probabilities = [{poi_id: probability} for 24 hours]
        self.probabilities = [{poi_id: pois_dict[poi_id]['probability_by_hour'][i] for poi_id in pois_dict} for i in range(24)]
        # {prev_poi_id: {after_poi_id: tendency}}
        self.tendency_probabilities = {poi_id: pois_dict[poi_id]['after_tendency'] for poi_id in pois_dict}
        # {poi_id: occupancy}
        self.occupancies = {poi_id: 0 for poi_id in pois_dict}
        # Dwell times and CDFs
        self.dwell_times = {poi_id: pois_dict[poi_id]['dwell_times'] for poi_id in pois_dict}
        self.dwell_time_cdfs = {poi_id: pois_dict[poi_id]['dwell_time_cdf'] for poi_id in pois_dict}

    def get_capacities_by_day(self, current_time):
        return self.capacities[min(current_time.day, 29)]

    def get_probabilities_by_time(self, current_time):
        return self.probabilities[current_time.hour]
    
    def get_capacities_by_time(self, current_time):
        return {poi_id: self.capacities[min(current_time.day, 29)][poi_id] * self.probabilities[current_time.hour][poi_id] for poi_id in self.capacities[min(current_time.day, 29)]}
    
    def get_after_tendencies(self, prev_poi_id):
        return {after_poi_id: self.tendency_probabilities[prev_poi_id].get(after_poi_id, 0) for after_poi_id in self.pois}
    
    def get_dwell_time_cdf(self, poi_id):
        return self.dwell_times[poi_id], self.dwell_time_cdfs[poi_id]
    
    def capacity_occupancy_diff(self, current_time):
        C = np.array(list(self.get_capacities_by_time(current_time).values()))
        O = np.array(list(self.occupancies.values()))
        return np.maximum(C - O, 0)
    
    def capacity_occupancy_diff_with_tendency(self, current_time, population):
        C = np.array(list(self.get_capacities_by_time(current_time).values()))
        O = np.array(list(self.occupancies.values()))
        A = np.array([list(self.get_after_tendencies(poi_id).values()) for poi_id in self.pois])
        
        # Apply occupancy weight to capacity-occupancy difference
        capacity_term = np.maximum(C - O, 0) * self.occupancy_weight
        
        # Apply tendency decay based on time spent
        tendency_term = A * self.alpha * (1 - self.tendency_decay)
        
        return (tendency_term + capacity_term[:, np.newaxis]) / population
    
    def generate_distribution(self, current_time, population):
        distribution = self.capacity_occupancy_diff(current_time)
        move_probability = sum(distribution) / population
        # normalize distribution
        return move_probability, distribution / np.sum(distribution) if np.sum(distribution) > 0 else np.zeros_like(distribution)
    
    def generate_distributions_with_tendency(self, current_time, population):
        distributions = self.capacity_occupancy_diff_with_tendency(current_time, population)
        move_probabilities = [sum(distribution) / population for distribution in distributions]
        # normalize distributions
        return move_probabilities, [distribution / np.sum(distribution) if np.sum(distribution) > 0 else np.zeros_like(distribution) for distribution in distributions]

    def get_next_poi(self, move_probability, distribution):
        if np.random.random() < move_probability:
            return np.random.choice(self.pois, p=distribution)
        else:
            return None
    
    def leave(self, poi_id):
        self.occupancies[poi_id] -= 1

    def enter(self, poi_id):
        self.occupancies[poi_id] += 1
        
def enter_poi(people, pois, current_time, hagerstown_pop, safegraph_to_place_id, place_id_to_safegraph):
    """
    Modified enter_poi to map SafeGraph IDs to papdata["places"] IDs.
    
    Args:
        people: Dictionary of Person objects.
        pois: POIs object containing POI data.
        current_time: Current time in the simulation.
        hagerstown_pop: Population size of Hagerstown.
        safegraph_to_place_id: Dictionary mapping SafeGraph IDs to papdata["places"] IDs.
        place_id_to_safegraph: Dictionary mapping papdata["places"] IDs to SafeGraph IDs.
    """
    move_probability, distribution = pois.generate_distribution(current_time, hagerstown_pop)
    move_probability_with_tendency, distributions_with_tendency = pois.generate_distributions_with_tendency(current_time, hagerstown_pop)
    for person_id, person in people.items():
        if person.curr_poi == "":
            next_poi_id = pois.get_next_poi(move_probability, distribution)
        else:
            # 将 person.curr_poi 转换回 SafeGraph ID
            safegraph_poi_id = place_id_to_safegraph.get(person.curr_poi, person.curr_poi)
            curr_poi_index = pois.poi_id_to_index[safegraph_poi_id]  # 使用 SafeGraph ID 查找索引
            next_poi_id = pois.get_next_poi(move_probability_with_tendency[curr_poi_index], distributions_with_tendency[curr_poi_index])
        if next_poi_id is not None:
            # 映射 SafeGraph ID 到 papdata["places"] ID
            mapped_poi_id = safegraph_to_place_id.get(next_poi_id, next_poi_id)
            pois.enter(next_poi_id)  # pois.enter 仍使用 SafeGraph ID
            person.visit(mapped_poi_id)  # person.visit 使用映射后的 ID
            
def leave_poi(people, current_time, pois, place_id_to_safegraph):
    """
    Optimized function to simulate leaving a certain POI.
    
    Args:
        people: Dictionary of Person objects.
        current_time: Current time in the simulation.
        pois: POIs object containing POI data.
        place_id_to_safegraph: Dictionary mapping papdata["places"] IDs to SafeGraph IDs.
    """
    for person_id, person in people.items():
        if not person.is_poi:
            continue  

        # 映射 person.curr_poi 回 SafeGraph ID
        poi_id = place_id_to_safegraph.get(person.curr_poi, person.curr_poi)
        hour_stayed = person.hour_stayed

        # Get dwell time CDF for the current POI
        dwell_times, dwell_time_cdf = pois.get_dwell_time_cdf(poi_id)

        # Find the probability of leaving based on the dwell time
        index = next((i for i, dt in enumerate(dwell_times) if dt >= hour_stayed), len(dwell_time_cdf) - 1)
        leave_prob = dwell_time_cdf[index]

        # Adjust leave probability based on occupancy
        expected_capacity = pois.capacities[min(current_time.day, 29)].get(poi_id, 1)
        current_occupancy = pois.occupancies.get(poi_id, 0)
        if expected_capacity > 0:
            occupancy_ratio = current_occupancy / expected_capacity
        else:
            occupancy_ratio = 0

        # Modify leave probability with occupancy
        if occupancy_ratio > 1:  # Over-occupied POI
            leave_prob *= occupancy_ratio
        else:  # Under-occupied POI
            leave_prob *= 0.5

        # Clamp leave probability between 0 and 1
        leave_prob = min(max(leave_prob, 0), 1)

        # Decide if the person leaves
        if random.random() < leave_prob:
            person.leave()
            pois.occupancies[poi_id] = max(0, current_occupancy - 1)  # Decrement occupancy
        else:
            person.stay()

def parse_json_field(field):
    if not field:
        return {}
    try:
        return json.loads(field)
    except json.JSONDecodeError:
        return {}

def compute_dwell_time_cdf(bucketed_dwell_times):
    """
    Compute the cumulative distribution function (CDF) of dwell times,
    grouping 0-60 minutes together and considering 120-240 minutes as covering 120-180 and 180-240.
    """
    # Map dwell time buckets to representative times in hours
    dwell_time_buckets = {
        '<60': 1,          # Group 0-60 minutes together as 1 hour
        '61-120': 1.5,     # Average of 61-120 minutes = 1.5 hours
        '121-240': 3,      # Average of 120-180 and 180-240 minutes = 3 hours
        '>240': 5          # Assume 5 hours for >240 minutes
    }
    
    # Combine counts for 0-60 minutes
    count_under_60 = (
        bucketed_dwell_times.get('<5', 0) +
        bucketed_dwell_times.get('5-10', 0) +
        bucketed_dwell_times.get('11-20', 0) +
        bucketed_dwell_times.get('21-60', 0)
    )
    
    # Combine counts for 121-240 minutes
    count_121_240 = bucketed_dwell_times.get('121-240', 0)
    # If there were separate counts for 120-180 and 180-240, sum them up
    # Since in your data it's '121-240', we use that directly

    # Build the adjusted bucketed dwell times
    adjusted_bucketed_dwell_times = {
        '<60': count_under_60,
        '61-120': bucketed_dwell_times.get('61-120', 0),
        '121-240': count_121_240,
        '>240': bucketed_dwell_times.get('>240', 0)
    }
    
    # Total count of visits
    total_visits = sum(adjusted_bucketed_dwell_times.values())
    
    # Compute probabilities
    dwell_times = []
    probabilities = []
    for bucket in ['<60', '61-120', '121-240', '>240']:
        count = adjusted_bucketed_dwell_times.get(bucket, 0)
        probability = count / total_visits if total_visits > 0 else 0
        dwell_time = dwell_time_buckets.get(bucket, 5)  # Default to 5 hours if not specified
        dwell_times.append(dwell_time)
        probabilities.append(probability)
    
    # Compute CDF
    cdf = []
    cumulative_prob = 0
    for prob in probabilities:
        cumulative_prob += prob
        cdf.append(cumulative_prob)
    
    return dwell_times, cdf

def preprocess_csv(papdata, file_path):
    pois_dict = {}
 
    placekeys = [ pap['placekey'] for pap in list(papdata['places'].values()) ]
    
    with pd.read_csv(file_path, chunksize=10000, usecols=['placekey', 'popularity_by_hour', 'bucketed_dwell_times', 'related_same_month_brand', 'location_name', 'raw_visit_counts', 'raw_visitor_counts', 'visits_by_day', 'related_same_day_brand']) as reader:
        for chunk in reader:
            for _, row in chunk[chunk['placekey'].isin(placekeys)].iterrows():
                poi_id = None

                for id, desc in papdata['places'].items():
                    if desc['placekey'] == row['placekey']:
                        poi_id = id
                        break
                
                if poi_id is None:
                    continue

                '''''
				# 解析 popularity_by_day
				popularity_by_day = parse_json_field(row.get('popularity_by_day', '{}'))

				'''''

				# 其他现有逻辑保持不变
                sum_popularity = sum(parse_json_field(row['popularity_by_hour'])) 
                probability_by_hour = [p / sum_popularity for p in parse_json_field(row['popularity_by_hour'])] if sum_popularity > 0 else []

                bucketed_dwell_times = parse_json_field(row['bucketed_dwell_times'])
                dwell_times, dwell_time_cdf = compute_dwell_time_cdf(bucketed_dwell_times)

                related_same_month_brand = parse_json_field(row['related_same_month_brand'])
                sum_tendency = sum(related_same_month_brand.values())
                after_tendency = {poi_id : related_same_month_brand.get(pois_dict.get(poi_id, {}).get('location_name', ''), 0) / sum_tendency  if sum_tendency > 0 else 0 for poi_id in pois_dict.keys()}

				# 更新 pois_dict，添加 popularity_by_day
                pois_dict[poi_id] = {
                    'location_name': row['location_name'],
                    'raw_visit_counts': int(row['raw_visit_counts']),
                    'raw_visitor_counts': int(row['raw_visitor_counts']),
                    'visits_by_day': parse_json_field(row['visits_by_day']),
                    'probability_by_hour': probability_by_hour,
                    'dwell_times': dwell_times,
                    'dwell_time_cdf': dwell_time_cdf,
                    'related_same_day_brand': parse_json_field(row['related_same_day_brand']),
                    'after_tendency': after_tendency,
                    #'popularity_by_day': popularity_by_day
                } 

    return pois_dict

def gen_patterns(papdata, start_time: datetime, duration=168):
    # Constants
    alpha = 0.16557695315916893
    occupancy_weight = 1.5711109677337263
    tendency_decay = 0.3460627088857086

    people = {}
    for person_id, person_info in papdata["people"].items():
        person = Person()
        person.id = int(person_id)  # 设置 ID
        person.sex = person_info.get("sex")
        person.age = person_info.get("age")
        person.home = person_info.get("home")
        people[int(person_id)] = person
        
    print('processing csv...')
    pois_dict = preprocess_csv(papdata, './data/patterns.csv')
    
    print('processing pois...')
    pois = POIs(pois_dict, alpha=alpha, occupancy_weight=occupancy_weight, tendency_decay=tendency_decay)
    
    print('processing sg to place data...')
    safegraph_to_place_id = {}
    place_id_to_safegraph = {}
    place_ids = sorted(papdata["places"].keys(), key=int)
    safegraph_ids = sorted(pois_dict.keys())
    if len(place_ids) != len(safegraph_ids):
        print("Warning: Number of places in papdata does not match number of POIs in pois_dict")
    for place_id, sg_id in zip(place_ids, safegraph_ids):
        safegraph_to_place_id[sg_id] = place_id
        place_id_to_safegraph[place_id] = sg_id
        
    output = {}
        
    for hour in range(duration):
        current_time = start_time + timedelta(hours=hour)
        current_weekday = current_time.weekday()
        print(f"Simulating hour {hour + 1}/{duration} at {current_time}...")
        leave_poi(people, current_time, pois, place_id_to_safegraph)
        enter_poi(people, pois, current_time, len(people),safegraph_to_place_id, place_id_to_safegraph)

        # 记录当前时间点的状态（以分钟为键）
        current_minutes = (hour + 1) * 60  # hour=0 → 60 分钟, hour=1 → 120 分钟, ...
        updated_homes = {}
        updated_places = {}
        for person in people.values():
            if person.at_home():
                home_id = str(person.home)
                if home_id not in updated_homes:
                    updated_homes[home_id] = []
                updated_homes[home_id].append(str(person.id))
            else:
                poi_id = person.curr_poi
                if poi_id:
                    poi_id = str(poi_id)
                    if poi_id not in updated_places:
                        updated_places[poi_id] = []
                    updated_places[poi_id].append(str(person.id))

        output[str(current_minutes)] = {
            "homes": updated_homes,
            "places": updated_places
        }
        
    return output
    
if __name__ == '__main__':
    try:
        papdata = {}
        with open('output/papdata.json', 'r') as f:
            papdata = json.load(f)
        
        patterns = gen_patterns(papdata, datetime.now(), 168)
        with open('output/patterns.json', 'w') as f:
            json.dump(patterns, f, indent=4)
    except:
        print('ERROR: Could not read papdata.json or generated patterns')
        