import numpy as np
import pandas as pd
import random
import yaml
from enum import Enum  # Kept for potential future extensions

# -----------------------------------------------------------------------------
# Global Variables & Categories
# -----------------------------------------------------------------------------

next_household_id = 0
next_groupquarter_id = 0

# Occupation categories and age ranges
poi_category = {
    '11': 'Agriculture, Forestry, Fishing and Hunting',
    '21': 'Mining, Quarrying, and Oil and Gas Extraction',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale Trade',
    '44': 'Retail Trade',
    '45': 'Retail Trade',
    '48': 'Transportation and Warehousing',
    '49': 'Transportation and Warehousing',
    '51': 'Information',
    '52': 'Depository Credit Intermediation',
    '53': 'Real Estate and Rental and Leasing',
    '54': 'Professional, Scientific, and Technical Services',
    '55': 'Management of Companies and Enterprises',
    '56': 'Administrative and Support and Waste Management and Remediation Services',
    '61': 'Education',
    '62': 'Medical',
    '71': 'Arts, Entertainment, and Recreation',
    '72': 'Restaurants and Other Eating Places',
    '81': 'Others',
    '92': 'Public Administration'
}

age_category = {
    "Preschool": (0, 4),
    "Adolescent": (5, 18),
    "Adult": (19, 64),
    "Retired": (65, 99)
}

# -----------------------------------------------------------------------------
# Class Definitions
# -----------------------------------------------------------------------------

class Person:
    """
    Class representing an individual.
    """
    def __init__(self, id, sex, age, cbg, household, hh_id, tags: dict = None, work_naics=None):
        self.id = id
        self.sex = sex
        self.age = age
        self.cbg = cbg
        self.household = household  # designated home household
        self.hh_id = hh_id         # original household id
        self.tags = tags
        self.occupation = None
        self.occupation_id = 0
        self.work_time = (0, 0)
        self.work_naics = work_naics
        self.availability = True
        self.left_from_work = False
        self.location = household  # current location

    def set_occupation(self, occupation):
        self.occupation = occupation
        self.set_work_time()

    def set_work_time(self):
        if (self.occupation == poi_category['62'] or self.occupation == poi_category['72']) and random.random() <= 0.15:
            self.work_time = (17, 24)
        elif self.occupation == poi_category['21']:
            if random.random() <= 0.10:
                self.work_time = (18, 2)
            else:
                self.work_time = (6, 18)
        elif self.occupation == poi_category['23']:
            start_time = random.randint(6, 8)
            end_time = random.randint(14, 18)
            self.work_time = (start_time, end_time)
        elif self.occupation is not None:
            self.work_time = (9, 17)

    def assign_household(self, hh):
        if self.location.id != hh.id:
            try:
                self.location.population.remove(self)
            except ValueError:
                pass
            if self not in hh.population:
                hh.population.append(self)
            self.location = hh

    def at_home(self) -> bool:
        return self.household.id == self.location.id

    def to_dict(self):
        return {
            'id': self.id,
            'cbg': self.cbg,
            'sex': self.sex,
            'age': self.age,
            'home': self.household.id,
            'availability': self.availability,
            'at_home': self.at_home(),
        }

    def __str__(self):
        return f"Person {self.id}"

    def __repr__(self):
        return self.__str__()

class Population:
    """
    Collection of Person instances.
    """
    def __init__(self):
        self.total_count = 0
        self.population = []

    def populate_indiv(self, person):
        self.population = np.append(self.population, person)

class Household(Population):
    """
    Household class that holds people and supports social events.
    """
    def __init__(self, cbg, total_count=0, population=None):
        global next_household_id
        super().__init__()
        if population is None:
            population = []
        self.total_count = total_count
        self.population = population
        self.cbg = cbg
        self.id = next_household_id
        next_household_id += 1
        self.social_days = 0
        self.social_max_duration = 0
        # New fields from clustering data
        self.movement_ratio = 0.0
        self.estimated_population = 0
        self.social_tendency = 'medium'  # Default social tendency

    def add_member(self, person):
        self.population.append(person)

    def start_social(self, duration: int):
        if self.social_days != 0:
            raise ValueError("Already hosting social")
        if not self.population:
            raise ValueError("No population to host social")
        self.social_days = 1
        self.social_max_duration = duration

    def end_social(self):
        self.social_days = 0
        self.social_max_duration = 0
        for person in list(self.population):
            if person.household.id != self.id:
                person.assign_household(person.household)

    def is_social(self) -> bool:
        return self.social_days > 0

    def has_hosts(self) -> bool:
        return any(person.household.id == self.id for person in self.population)

    def to_dict(self):
        return {
            'id': self.id,
            'cbg': self.cbg,
            'members': len(self.population),
            'movement_ratio': self.movement_ratio,
            'social_tendency': self.social_tendency
        }

    def __str__(self):
        hosts = [p for p in self.population if p.hh_id == self.id]
        guests = [p for p in self.population if p.hh_id != self.id]
        return f"Household {self.id} with hosts {hosts} and guests {guests}"

    def __repr__(self):
        return self.__str__()

class GroupQuarter(Population):
    """
    Group Quarters such as Nursing Homes or Prisons.
    """
    def __init__(self, cbg, total_count=0, population=None, type=None):
        global next_groupquarter_id
        super().__init__()
        if population is None:
            population = []
        self.total_count = total_count
        self.population = population
        self.cbg = cbg
        self.id = next_groupquarter_id
        self.type = type
        next_groupquarter_id += 1

    def add_member(self, person):
        self.population.append(person)

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'cbg': self.cbg,
            'members': len(self.population)
        }

    def __str__(self):
        return f"GroupQuarter {self.id}"

    def __repr__(self):
        return self.__str__()

# -----------------------------------------------------------------------------
# Helper Functions for Household Creation
# -----------------------------------------------------------------------------

def create_households(pop_data, households, cbg):
    """
    Creates households based on census household count and population parameters.
    Returns a list of Household objects.
    """
    count = 1
    result = []
    married, opposite_sex, samesex, female_single, male_single, other = 0, 1, 2, 3, 4, 5

    for _ in range(int(households * pop_data["family_percents"][married] / 100)):
        result.append(create_married_hh(pop_data, cbg, count))
        count += 1
    for _ in range(int(households * pop_data['family_percents'][samesex] / 100)):
        result.append(create_samesex_hh(pop_data, cbg, count))
        count += 1
    for _ in range(int(households * pop_data['family_percents'][opposite_sex] / 100)):
        result.append(create_oppositesex_hh(pop_data, cbg, count))
        count += 1
    for _ in range(int(households * pop_data['family_percents'][female_single] / 100)):
        result.append(create_femsingle_hh(pop_data, cbg, count))
        count += 1
    for _ in range(int(households * pop_data['family_percents'][male_single] / 100)):
        result.append(create_malesingle_hh(pop_data, cbg, count))
        count += 1
    for _ in range(int(households * pop_data['family_percents'][other] / 100)):
        result.append(create_other_hh(pop_data, cbg, count))
        count += 1

    return result

def create_married_hh(pop_data, cbg, count):
    household = Household(cbg)
    age_percent = pop_data['age_percent_married']
    age_group = random.choices(pop_data['age_groups_married'], age_percent)[0]
    
    husband_age = random.choice(range(age_group, age_group + 10))
    household.add_member(Person(pop_data['count'], 0, husband_age, cbg, household, count))
    pop_data['count'] += 1

    wife_age = random.choice(range(age_group, age_group + 10))
    household.add_member(Person(pop_data['count'], 1, wife_age, cbg, household, count))
    pop_data['count'] += 1

    children_percent = pop_data['children_true_percent'] + 0.1 if (15 < age_group < 45) else 0.05
    child_bool = random.choices([True, False], [children_percent, 100 - children_percent])[0]
    if child_bool:
        num_child = random.choices(pop_data['children_groups'], pop_data['children_percent'])[0]
        for _ in range(num_child):
            household.add_member(
                Person(
                    pop_data['count'],
                    random.choices([0, 1], [pop_data['male_percent'], pop_data['female_percent']])[0],
                    random.choice(range(1, 19)),
                    cbg,
                    household,
                    count
                )
            )
            pop_data['count'] += 1
    return household

def create_samesex_hh(pop_data, cbg, count):
    household = Household(cbg)
    age_percent = pop_data['age_percent']
    age_group = random.choices(pop_data['age_groups'], age_percent)[0]
    husband_age = random.choice(range(age_group, age_group + 10))
    wife_age = random.choice(range(age_group, age_group + 10))
    gender = random.choices([0, 1], [pop_data['male_percent'], pop_data['female_percent']])[0]
    household.add_member(Person(pop_data['count'], gender, husband_age, cbg, household, count))
    pop_data['count'] += 1
    household.add_member(Person(pop_data['count'], gender, wife_age, cbg, household, count))
    pop_data['count'] += 1
    return household

def create_oppositesex_hh(pop_data, cbg, count):
    household = Household(cbg)
    age_percent = pop_data['age_percent']
    age_group = random.choices(pop_data['age_groups'], age_percent)[0]
    husband_age = random.choice(range(age_group, age_group + 10))
    wife_age = random.choice(range(age_group, age_group + 10))
    household.add_member(
        Person(pop_data['count'], random.choices([0, 1], [pop_data['male_percent'], pop_data['female_percent']])[0],
               husband_age, cbg, household, count)
    )
    pop_data['count'] += 1
    household.add_member(
        Person(pop_data['count'], random.choices([0, 1], [pop_data['male_percent'], pop_data['female_percent']])[0],
               wife_age, cbg, household, count)
    )
    pop_data['count'] += 1
    return household

def create_femsingle_hh(pop_data, cbg, count):
    household = Household(cbg)
    age_percent = pop_data['age_percent']
    age_group = random.choices(pop_data['age_groups'], age_percent)[0]
    age = random.choice(range(age_group, age_group + 10))
    household.add_member(Person(pop_data['count'], 1, age, cbg, household, count))
    pop_data['count'] += 1
    return household

def create_malesingle_hh(pop_data, cbg, count):
    household = Household(cbg)
    age_percent = pop_data['age_percent']
    age_group = random.choices(pop_data['age_groups'], age_percent)[0]
    age = random.choice(range(age_group, age_group + 10))
    household.add_member(Person(pop_data['count'], 0, age, cbg, household, count))
    pop_data['count'] += 1
    return household

def create_other_hh(pop_data, cbg, count):
    household = Household(cbg)
    age_percent = pop_data['age_percent']
    age_group = random.choices(pop_data['age_groups'], age_percent)[0]
    size_percent = pop_data['size_percent']
    size_group = random.choices(pop_data['size_groups'], size_percent)[0]
    for _ in range(size_group):
        age = random.choice(range(age_group, age_group + 10))
        household.add_member(
            Person(pop_data['count'], random.choices([0, 1], [pop_data['male_percent'], pop_data['female_percent']])[0],
                   age, cbg, household, count)
        )
        pop_data['count'] += 1
    return household

def visualize_household(household):
    print("ANALYZING HOUSEHOLD " + str(household))
    print("The current household has " + str(len(household.population)) + " members.")
    for i, member in enumerate(household.population):
        print(f"member {i}: id: {member.id}, sex: {member.sex}, age: {member.age}")

def create_pop_from_cluster(cluster, census_df):
    populations = 0
    households = 0
    for i in cluster:
        i = int(i)
        populations += int(census_df[census_df.census_block_group == i].values[0][1])
        households += int(census_df[census_df.census_block_group == i].values[0][3])
    return populations, households

# -----------------------------------------------------------------------------
# New prepare_data_for_papdata Function (Fix Applied)
# -----------------------------------------------------------------------------

def prepare_data_for_papdata(household_list):
    """
    Flattens the nested household_list and returns a list of Household objects.
    This ensures Papdata.generate() receives objects with attributes like population.
    """
    prepared = [home for group in household_list for home in group]
    return prepared

# -----------------------------------------------------------------------------
# New enhance_households_with_movement_data Function
# -----------------------------------------------------------------------------

def enhance_households_with_movement_data(household_list, cbg_info):
    """
    Enhances household data with movement patterns from cbg_info.yaml
    """
    # Create a lookup dictionary for quick access to CBG data
    cbg_lookup = {item['GEOID10']: item for item in cbg_info}
    
    # Flatten household list and enrich with movement data
    for group in household_list:
        for home in group:
            cbg = home.cbg
            if cbg in cbg_lookup:
                # Add movement properties to households
                home.movement_ratio = cbg_lookup[cbg].get('ratio', 0)
                home.estimated_population = cbg_lookup[cbg].get('estimated_population', 0)
                
                # You could adjust social tendencies based on movement patterns
                if home.movement_ratio > 0.6:  # High internal connectivity
                    home.social_tendency = 'low'  # They stay within their community
                elif home.movement_ratio < 0.4:  # Low internal connectivity
                    home.social_tendency = 'high'  # They travel more outside their community
                else:
                    home.social_tendency = 'medium'
    
    return household_list

# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Load population info
    with open(r'E:\Dileno Diesease Modelling\Algorithms\population_info.yaml', mode="r", encoding="utf-8") as file:
        pop_data = yaml.full_load(file)
    print(pop_data)  # Debug output; remove if not needed

    # Read census data
    census_df = pd.read_csv(r"E:\Dileno Diesease Modelling\Algorithms\safegraph_cbg_population_estimate.csv")
    
    # Use cbg_info.yaml instead of clusters.csv
    with open(r'E:\Dileno Diesease Modelling\Algorithms\SP24\cbg_info.yaml', mode="r", encoding="utf-8") as file:
        cbg_info = yaml.full_load(file)
    
    # Extract GEOID10 values as clusters
    clusters = [item['GEOID10'] for item in cbg_info]

    household_list = []
    # For each census block group, create households based on census household count
    for cbg in clusters:
        _, households_count = create_pop_from_cluster([cbg], census_df)
        household_list.append(create_households(pop_data, households_count, cbg))
    
    # Enhance households with movement data from clustering
    household_list = enhance_households_with_movement_data(household_list, cbg_info)

    # Dump the household list to a YAML file
    with open(r'E:\Dileno Diesease Modelling\Algorithms\households.yaml', mode="wt", encoding="utf-8") as outstream:
        yaml.dump(household_list, outstream)
    print("Successfully Created Households")

    # Prepare data for Papdata by flattening the nested list to get Household objects
    prepared_data = prepare_data_for_papdata(household_list)
    place_path = r'E:\Dileno Diesease Modelling\Algorithms\SP24\refactored\input\hagerstown.pois.csv'

    import sys
    sys.path.append(r'E:\Dileno Diesease Modelling\Algorithms\SP24\refactored')
    from papdata import Papdata
    papdata_processor = Papdata(prepared_data, place_path)
    papdata_processor.generate()
    print("Papdata JSON generation completed.")