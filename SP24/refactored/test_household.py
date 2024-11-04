import random

'''
Cencus population data:
'''
population_distribution = {
    "Male Children": 14.44,
    "Male Adults": 35.05,
    "Female Children": 13.87,
    "Female Adults": 36.66,
    "total_children_percentage": 28.31,
    "total_adults_percentage": 71.71,
    "total_male_percentage": 49.49,
    "total_female_percentage": 50.53
}

'''
Census hosuehold data:
'''
# DP02 + B11001
household_type_distribution = {
    "family_households": 65.4,
    "nonfamily_households": 34.6,
    "with_children": 28.8,
}
avg_household_size = 2.49

# https://www2.census.gov/programs-surveys/demo/tables/families/2023/cps-2023/taba3.xls
children_distribution = {
    "single_mother_child_distribution": {
        "1": 0.435,
        "2": 0.363,
        "3": 0.141,
        "4+": 0.062
    },
    "single_father_child_distribution": {
        "1": 0.423,
        "2": 0.375,
        "3": 0.141,
        "4+": 0.060
    },
    "two_parent_child_distribution": {
        "1": 0.429,
        "2": 0.369,
        "3": 0.141,
        "4+": 0.061
    }
}
household_w_children_distribution = {
    "married_couple_with_children": 0.6068,
    "single_mother_with_children": 0.3345,
    "single_father_with_children": 0.0587
}
parent_age_distribution = {
    "married_couple_with_children": {
        "15-24": 0.02,
        "25-34": 0.25,
        "35-44": 0.40,
        "45-54": 0.25,
        "55-64": 0.07,
        "65+": 0.01
    },
    "single_mother_with_children": {
        "15-24": 0.08,
        "25-34": 0.30,
        "35-44": 0.35,
        "45-54": 0.20,
        "55-64": 0.06,
        "65+": 0.01
    },
    "single_father_with_children": {
        "15-24": 0.03,
        "25-34": 0.22,
        "35-44": 0.38,
        "45-54": 0.28,
        "55-64": 0.08,
        "65+": 0.01
    }
}

# S11001
nonfamily_percentages = {
    "living_alone": 28.8,
    "not_living_alone": 5.8,
}


class Person:
    _last_id = 0

    def __init__(self, age: int, sex: int, hh_id: int = None, tags: dict = None, cbg: int = None):
        """
        Initializes the Person class
        """
        self.id = Person._last_id
        Person._last_id += 1
        self.age = age
        self.sex = sex
        self.hh_id = hh_id
        self.tags = tags
        self.cbg = cbg

    def to_dict(self):
        return {
            "id": self.id,
            "age": self.age,
            "sex": self.sex,
            "hh_id": self.hh_id,
            "tags": self.tags,
            "cbg": self.cbg
        }


class Household:
    _last_id = 0

    def __init__(self, population: list[Person] = [], cbg: int = None, type: str = None):
        """
        Initializes the Household class
        """
        self.id = Household._last_id
        Household._last_id += 1
        self.population = population
        self.cbg = cbg
        self.type = type

    def to_dict(self):
        return {
            "id": self.id,
            "population": [person.to_dict() for person in self.population],
            "cbg": self.cbg,
            "type": self.type
        }


'''
The idea is to generate the number of people in each of the 4 large category (maybe male/female children/adult) first so that the total population would be close to the actual
distribution. 

Then genreate each household in a specific order based on the composition of the household.

Children is unique in that they can not be alone in a household and can help us determine the age of the parents in the household.

1. We start with the generate 1 or 2 adult households w children, using data on # of children in households. 
    Each household should -1 or -2 adult and -# of children from the population)
    This step should use up all the children in the population.
    
2. Next we generate living alone households. This should be relatively easy since we have all the data.( male/female householder living alone, over 65 living alone, etc)
    Each household should -1 adult from the population.

Now the rest of the population consists of grandparents living with child/grandchild, other relatives, and some special cases.

3. Assign grandparents and other relatives to the existing households.
    Add almost all of the remaining adults to a small portion of the households.

4. The remaining households are the special cases.

5. Iterate through the generated households and refine the age to match the distribution.

This approach should not include any optimization and should be fast in generating the households.

'''


def gen_basic_people_distribution(total_population):
    '''
    returns the number of people in each age and sex group
    '''
    return {
        "Male Children": int(total_population * population_distribution["Male Children"] / 100),
        "Male Adults": int(total_population * population_distribution["Male Adults"] / 100),
        "Female Children": int(total_population * population_distribution["Female Children"] / 100),
        "Female Adults": int(total_population * population_distribution["Female Adults"] / 100),
    }


def gen_households(total_households, total_population):
    ''' Steps
    1. generate 2 adult households w children (since children can help us determine the age of the adults)
        a. 1 child
        b. 2 children
        c. 3 children
        d. 4+ children
    2. generate living alone households (1 person, could be any age range > 18)
        a. male living alone
        b. female living alone
    '''
    import random

    # Initialize people distribution
    people_counts = {
        "Male Children": int(total_population * population_distribution["Male Children"] / 100),
        "Male Adults": int(total_population * population_distribution["Male Adults"] / 100),
        "Female Children": int(total_population * population_distribution["Female Children"] / 100),
        "Female Adults": int(total_population * population_distribution["Female Adults"] / 100),
    }
    print("Initial People Counts:", people_counts)

    # Initialize lists to store people and households
    people = []
    households = []

    # Generate households with children
    number_of_households_with_children = int(total_households * household_type_distribution["with_children"] / 100)
    num_married_couple_with_children = int(number_of_households_with_children * household_w_children_distribution["married_couple_with_children"])
    num_single_mother_with_children = int(number_of_households_with_children * household_w_children_distribution["single_mother_with_children"])
    num_single_father_with_children = number_of_households_with_children - num_married_couple_with_children - num_single_mother_with_children

    # Define age ranges
    age_ranges = {
        "15-24": (15, 24),
        "25-34": (25, 34),
        "35-44": (35, 44),
        "45-54": (45, 54),
        "55-64": (55, 64),
        "65+": (65, 90)
    }

    # Generate married couple households with children
    for _ in range(num_married_couple_with_children):
        # Determine number of children
        number_of_children = random.choices(
            population=[1, 2, 3, 4],
            weights=[
                children_distribution["two_parent_child_distribution"]["1"],
                children_distribution["two_parent_child_distribution"]["2"],
                children_distribution["two_parent_child_distribution"]["3"],
                children_distribution["two_parent_child_distribution"]["4+"]
            ]
        )[0]

        # Check if there are enough adults
        if people_counts["Male Adults"] < 1 or people_counts["Female Adults"] < 1:
            break

        # Assign age group to parents
        age_group = random.choices(
            population=list(parent_age_distribution["married_couple_with_children"].keys()),
            weights=list(parent_age_distribution["married_couple_with_children"].values())
        )[0]
        age_range = age_ranges[age_group]

        # Assign ages to parents
        parent1_age = random.randint(age_range[0], age_range[1])
        parent2_age = random.randint(age_range[0], age_range[1])

        # Create parent Person objects
        parent1 = Person(age=parent1_age, sex=0)  # Male
        parent2 = Person(age=parent2_age, sex=1)  # Female

        # Decrement counts
        people_counts["Male Adults"] -= 1
        people_counts["Female Adults"] -= 1

        household_population = [parent1, parent2]

        # Create children
        for _ in range(number_of_children):
            if people_counts["Male Children"] + people_counts["Female Children"] < 1:
                break
            total_children_remaining = people_counts["Male Children"] + people_counts["Female Children"]
            male_child_prob = people_counts["Male Children"] / total_children_remaining
            child_sex = random.choices(population=[0, 1], weights=[male_child_prob, 1 - male_child_prob])[0]
            child_age = random.randint(0, 17)
            child = Person(age=child_age, sex=child_sex)
            if child_sex == 0:
                people_counts["Male Children"] -= 1
            else:
                people_counts["Female Children"] -= 1
            household_population.append(child)

        # Create Household
        household = Household(population=household_population)
        households.append(household)
        people.extend(household_population)

    # Generate single mother households with children
    for _ in range(num_single_mother_with_children):
        number_of_children = random.choices(
            population=[1, 2, 3, 4],
            weights=[
                children_distribution["single_mother_child_distribution"]["1"],
                children_distribution["single_mother_child_distribution"]["2"],
                children_distribution["single_mother_child_distribution"]["3"],
                children_distribution["single_mother_child_distribution"]["4+"]
            ]
        )[0]

        if people_counts["Female Adults"] < 1:
            break

        age_group = random.choices(
            population=list(parent_age_distribution["single_mother_with_children"].keys()),
            weights=list(parent_age_distribution["single_mother_with_children"].values())
        )[0]
        age_range = age_ranges[age_group]
        mother_age = random.randint(age_range[0], age_range[1])

        mother = Person(age=mother_age, sex=1)  # Female
        people_counts["Female Adults"] -= 1
        household_population = [mother]

        for _ in range(number_of_children):
            if people_counts["Male Children"] + people_counts["Female Children"] < 1:
                break
            total_children_remaining = people_counts["Male Children"] + people_counts["Female Children"]
            male_child_prob = people_counts["Male Children"] / total_children_remaining
            child_sex = random.choices(population=[0, 1], weights=[male_child_prob, 1 - male_child_prob])[0]
            child_age = random.randint(0, 17)
            child = Person(age=child_age, sex=child_sex)
            if child_sex == 0:
                people_counts["Male Children"] -= 1
            else:
                people_counts["Female Children"] -= 1
            household_population.append(child)

        household = Household(population=household_population)
        households.append(household)
        people.extend(household_population)

    # Generate single father households with children
    for _ in range(num_single_father_with_children):
        number_of_children = random.choices(
            population=[1, 2, 3, 4],
            weights=[
                children_distribution["single_father_child_distribution"]["1"],
                children_distribution["single_father_child_distribution"]["2"],
                children_distribution["single_father_child_distribution"]["3"],
                children_distribution["single_father_child_distribution"]["4+"]
            ]
        )[0]

        if people_counts["Male Adults"] < 1:
            break

        age_group = random.choices(
            population=list(parent_age_distribution["single_father_with_children"].keys()),
            weights=list(parent_age_distribution["single_father_with_children"].values())
        )[0]
        age_range = age_ranges[age_group]
        father_age = random.randint(age_range[0], age_range[1])

        father = Person(age=father_age, sex=0)  # Male
        people_counts["Male Adults"] -= 1
        household_population = [father]

        for _ in range(number_of_children):
            if people_counts["Male Children"] + people_counts["Female Children"] < 1:
                break
            total_children_remaining = people_counts["Male Children"] + people_counts["Female Children"]
            male_child_prob = people_counts["Male Children"] / total_children_remaining
            child_sex = random.choices(population=[0, 1], weights=[male_child_prob, 1 - male_child_prob])[0]
            child_age = random.randint(0, 17)
            child = Person(age=child_age, sex=child_sex)
            if child_sex == 0:
                people_counts["Male Children"] -= 1
            else:
                people_counts["Female Children"] -= 1
            household_population.append(child)

        household = Household(population=household_population)
        households.append(household)
        people.extend(household_population)

    # Generate living alone households
    number_of_living_alone_households = int(total_households * nonfamily_percentages["living_alone"] / 100)
    for _ in range(number_of_living_alone_households):
        if people_counts["Male Adults"] + people_counts["Female Adults"] < 1:
            break

        total_adults_remaining = people_counts["Male Adults"] + people_counts["Female Adults"]
        male_adult_prob = people_counts["Male Adults"] / total_adults_remaining
        person_sex = random.choices(population=[0, 1], weights=[male_adult_prob, 1 - male_adult_prob])[0]
        person_age = random.randint(18, 90)
        person = Person(age=person_age, sex=person_sex)
        if person_sex == 0:
            people_counts["Male Adults"] -= 1
        else:
            people_counts["Female Adults"] -= 1

        household = Household(population=[person])
        households.append(household)
        people.append(person)

    print("Remaining People Counts after household generation:", people_counts)
    print(f"Total households generated: {len(households)}")
    print(f"Total people generated: {len(people)}")

    return households, people

def print_households(households):
    for household in households:
        print(f"Household ID: {household.id}")
        for person in household.population:
            sex = 'Male' if person.sex == 0 else 'Female'
            print(f"  Person ID: {person.id}, Age: {person.age}, Sex: {sex}")
        print("-" * 40)


if __name__ == "__main__":
    total_households = 1000
    total_population = total_households * avg_household_size

    hh, people = gen_households(total_households, total_population)
    print_households(hh)
