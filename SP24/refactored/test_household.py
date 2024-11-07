import matplotlib.pyplot as plt
import numpy as np

'''
The idea is to generate the number of people in each of the 4 large categories (male/female children/adults) first so that the total population would be close to the actual
distribution.

Then generate each household in a specific order based on the composition of the household.

Children are unique in that they cannot be alone in a household and can help us determine the age of the parents in the household.

1. We start by generating family households with 1 or 2 adults with children, using data on the number of children in households.
   Each household should subtract 2 or 1 adult and subtract the number of children from the population.
   This step should use up all the children in the population.

2. Family households without children
   Subtract 2 adults from the population.

3. Next, we generate non-family households (living alone and not living alone).
   Each household should subtract 1+ adult from the population.

Now the rest of the population consists of grandparents living with child/grandchild, other relatives, and some special cases.

4. Assign grandparents and other relatives to the existing family households.
   Add almost all of the remaining adults to a small portion of the households.

5. The remaining households are special cases.

5. Iterate through the generated households and refine the age to match the distribution.

This approach should not include any optimization and should be fast in generating the households.

'''

'''
Census population data:
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
Census household data:
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

# Added data for households with children aged 18-25
married_couples_percentages = {
    "with_children_18_25": 13.51,
    "with_children_under_18": 73.16,
    "without_children": 26.84
}

# S11001
nonfamily_percentages = {
    "living_alone": 28.8,
    "not_living_alone": 5.8,
}


'''
Data Hierarchy:
'''

input_data = {
    "total_households": 10000,
    "avg_household_size": 2.49,
    "pop_info": {
        "population_distribution": {
                    "Male Children": 14.44,
                    "Male Adults": 35.05,
                    "Female Children": 13.87,
                    "Female Adults": 36.66
        },
        "parent_age_distribution" :{
            "married_couple_with_children": {
                "15-24": 2,
                "25-34": 25,
                "35-44": 40,
                "45-54": 25,
                "55-64": 7,
                "65+": 1
            },
            "single_mother_with_children": {
                "15-24": 8,
                "25-34": 30,
                "35-44": 35,
                "45-54": 20,
                "55-64": 6,
                "65+": 1
            },
            "single_father_with_children": {
                "15-24": 3,
                "25-34": 22,
                "35-44": 38,
                "45-54": 28,
                "55-64": 8,
                "65+": 1
            }
        },
        "married_couple_without_children_age_distribution" :{
            "15-24": 5,
            "25-34": 20,
            "35-44": 20,
            "45-54": 25,
            "55-64": 20,
            "65+": 10
        }
    },
    "household_info": {
        "family_households": {
            "total_percentage": 65.4,
            "with_children_under_18": {
                "total_percentage": 28.8,
                "distribution": {
                    "married_couple_with_children": {
                        "total_percentage": 60.68,
                        "number_of_children": {
                            "1": 42.9,
                            "2": 36.9,
                            "3": 14.1,
                            "4+": 6.1
                        }
                    },
                    "single_mother_with_children": {
                        "total_percentage": 33.45,
                        "number_of_children": {
                            "1": 43.5,
                            "2": 36.3,
                            "3": 14.1,
                            "4+": 6.2
                        }
                    },
                    "single_father_with_children": {
                        "total_percentage": 5.87,
                        "number_of_children": {
                            "1": 42.3,
                            "2": 37.5,
                            "3": 14.1,
                            "4+": 6
                        }
                    }
                }
            }
        },
        "nonfamily_households": {
            "total_percentage": 34.6,
            "living_alone": 28.8,
            "not_living_alone": 5.8
        },
        "married_couples_percentages" : {
            "with_children_18_25": 13.51,
            "with_children_under_18": 73.16,
            "without_children": 26.84
        }
    }
}


class Person:
    _last_id = 0

    def __init__(self, age: int, sex: int, hh_id: int = None, position_in_hh: str = None, tags: dict = None, cbg: int = None):
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
        self.position_in_hh = position_in_hh  # Added attribute

    def to_dict(self):
        return {
            "id": self.id,
            "age": self.age,
            "sex": self.sex,
            "hh_id": self.hh_id,
            "position_in_hh": self.position_in_hh,
            "tags": self.tags,
            "cbg": self.cbg
        }


class Household:
    _last_id = 0

    def __init__(self, population: list = [], cbg: int = None):
        """
        Initializes the Household class
        """
        self.id = Household._last_id
        Household._last_id += 1
        self.population = population
        self.cbg = cbg

    def to_dict(self):
        return {
            "id": self.id,
            "population": [person.to_dict() for person in self.population],
            "cbg": self.cbg
        }


def generate_number_of_children_4plus():
    """
    Generates a number of children 4 or more, with probabilities decreasing as the number increases.
    """
    number_of_children_options = [4, 5, 6, 7, 8]
    probabilities = [0.5, 0.25, 0.15, 0.07, 0.03]  # Adjusted probabilities
    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()  # Normalize
    number_of_children = np.random.choice(number_of_children_options, p=probabilities)
    return number_of_children


def generate_children_ages(parent_age, number_of_children, max_child_age=17):
    """
    Generates a list of children's ages based on parent's age and number of children.
    """
    min_parental_age_at_birth = 15
    max_parental_age_at_birth = 50

    max_oldest_child_age = min(parent_age - min_parental_age_at_birth, max_child_age)
    min_oldest_child_age = max(parent_age - max_parental_age_at_birth, 0)

    if max_oldest_child_age < min_oldest_child_age:
        # Parent is too young; assign child age 0
        oldest_child_age = 0
    else:
        oldest_child_age = np.random.randint(int(min_oldest_child_age), int(max_oldest_child_age)+1)

    children_ages = [oldest_child_age]

    for _ in range(number_of_children - 1):
        # Age gap between siblings is between 1 and 3 years
        age_gap = np.random.randint(1, 4)
        child_age = children_ages[-1] - age_gap
        if child_age < 0:
            child_age = 0
        children_ages.append(child_age)

    # Ensure all ages are between 0 and max_child_age
    children_ages = [max(min(age, max_child_age), 0) for age in children_ages]

    # Shuffle the ages to avoid ordering from oldest to youngest
    np.random.shuffle(children_ages)

    return children_ages


def get_people_distribution(total_population):
    population_distribution = input_data["pop_info"]["population_distribution"]
    total_percentage = sum(population_distribution.values())

    return {
        "Male Children": int(total_population * population_distribution["Male Children"] / total_percentage),
        "Male Adults": int(total_population * population_distribution["Male Adults"] / total_percentage),
        "Female Children": int(total_population * population_distribution["Female Children"] / total_percentage),
        "Female Adults": int(total_population * population_distribution["Female Adults"] / total_percentage),
    }

def print_all_households(households):
    for household in households:
        print("Household ID:", household.id)
        for person in household.population:
            print(person.to_dict())
        print("**********************************************\n")

def print_all_people(people):
    for person in people:
        print(person.to_dict())
        print("**********************************************\n")



def gen_households(total_households, total_population):
    # Initialize people distribution
    people_counts = get_people_distribution(total_population)
    print("Initial People Counts:", people_counts)

    # Initialize lists to store people and households
    people = []
    households = []

    # Define age ranges
    age_ranges = {
        "15-24": (15, 24),
        "25-34": (25, 34),
        "35-44": (35, 44),
        "45-54": (45, 54),
        "55-64": (55, 64),
        "65+": (65, 90)
    }

    # Part 1: Generate households with children under 18
    household_info = input_data["household_info"]
    family_households_info = household_info["family_households"]
    with_children_under_18_info = family_households_info["with_children_under_18"]
    total_percentage_with_children_under_18 = with_children_under_18_info["total_percentage"] / 100

    number_of_households_with_children = int(total_households * total_percentage_with_children_under_18)

    distribution_with_children = with_children_under_18_info["distribution"]

    num_married_couple_with_children = int(
        number_of_households_with_children * (distribution_with_children["married_couple_with_children"]["total_percentage"] / 100)
    )

    num_single_mother_with_children = int(
        number_of_households_with_children * (distribution_with_children["single_mother_with_children"]["total_percentage"] / 100)
    )

    num_single_father_with_children = int(
        number_of_households_with_children * (distribution_with_children["single_father_with_children"]["total_percentage"] / 100)
    )

    parent_age_distribution = input_data["pop_info"]["parent_age_distribution"]

    # Generate married couple households with children under 18
    for _ in range(num_married_couple_with_children):
        # Determine number of children
        num_children_dist = distribution_with_children["married_couple_with_children"]["number_of_children"]
        prob = [
            num_children_dist["1"],
            num_children_dist["2"],
            num_children_dist["3"],
            num_children_dist["4+"]
        ]
        prob = np.array(prob)
        prob = prob / prob.sum()  # Normalize

        num_child = np.random.choice(
            ['1', '2', '3', '4+'],
            p=prob
        )

        if num_child == '4+':
            number_of_children = generate_number_of_children_4plus()
        else:
            number_of_children = int(num_child)

        # Assign age group to parents
        age_group_probabilities = np.array(list(parent_age_distribution["married_couple_with_children"].values()))
        age_group_probabilities = age_group_probabilities / age_group_probabilities.sum()  # Normalize

        age_group = np.random.choice(
            list(parent_age_distribution["married_couple_with_children"].keys()),
            p=age_group_probabilities
        )
        age_range = age_ranges[age_group]

        # Assign ages to parents
        parent1_age = np.random.randint(age_range[0], age_range[1]+1)
        parent2_age = np.random.randint(age_range[0], age_range[1]+1)

        # Create parent Person objects with position_in_hh
        parent1 = Person(age=parent1_age, sex=0, position_in_hh='parent')  # Male
        parent2 = Person(age=parent2_age, sex=1, position_in_hh='parent')  # Female

        # Decrement counts
        people_counts["Male Adults"] -= 1
        people_counts["Female Adults"] -= 1

        household_population = [parent1, parent2]

        # Generate children's ages (up to 17)
        children_ages = generate_children_ages(min(parent1_age, parent2_age), number_of_children, max_child_age=17)

        for child_age in children_ages:
            total_children_remaining = people_counts["Male Children"] + people_counts["Female Children"]
            if total_children_remaining <= 0:
                break
            male_child_prob = people_counts["Male Children"] / total_children_remaining if total_children_remaining > 0 else 0.5
            child_sex = np.random.choice([0, 1], p=[male_child_prob, 1 - male_child_prob])
            child = Person(age=child_age, sex=child_sex, position_in_hh='child')
            if child_sex == 0:
                people_counts["Male Children"] -= 1
            else:
                people_counts["Female Children"] -= 1
            household_population.append(child)

        # Create Household
        household = Household(population=household_population)
        households.append(household)
        people.extend(household_population)

    # Married couples percentages
    married_couples_percentages = household_info["married_couples_percentages"]

    # Update total number of married couples
    total_number_of_married_couples = int(num_married_couple_with_children / (married_couples_percentages["with_children_under_18"] / 100))

    # Number of married couples with children aged 18-25
    num_married_couple_with_children_18_25 = int(total_number_of_married_couples * (married_couples_percentages["with_children_18_25"] / 100))

    # Generate married couple households with children aged 18-25
    for _ in range(num_married_couple_with_children_18_25):
        # Assume 1 to 3 children aged 18-25 skewed to the lower end
        number_of_children = np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])

        # Assign age group to parents (should be older to have older children)
        possible_age_groups = ["45-54", "55-64", "65+"]
        age_group_probabilities = np.array([0.4, 0.4, 0.2])  # Assumed probabilities
        age_group_probabilities = age_group_probabilities / age_group_probabilities.sum()  # Normalize

        age_group = np.random.choice(
            possible_age_groups,
            p=age_group_probabilities
        )
        age_range = age_ranges[age_group]

        # Assign ages to parents
        parent1_age = np.random.randint(age_range[0], age_range[1]+1)
        parent2_age = np.random.randint(age_range[0], age_range[1]+1)

        # Create parent Person objects with position_in_hh
        parent1 = Person(age=parent1_age, sex=0, position_in_hh='parent')  # Male
        parent2 = Person(age=parent2_age, sex=1, position_in_hh='parent')  # Female

        # Decrement counts
        people_counts["Male Adults"] -= 1
        people_counts["Female Adults"] -= 1

        household_population = [parent1, parent2]

        # Generate children's ages (18 to 25)
        children_ages = []
        for _ in range(number_of_children):
            child_age = np.random.randint(18, 26)
            children_ages.append(child_age)

        for child_age in children_ages:
            total_adults_remaining = people_counts["Male Adults"] + people_counts["Female Adults"]
            if total_adults_remaining <= 0:
                break
            male_adult_prob = people_counts["Male Adults"] / total_adults_remaining if total_adults_remaining > 0 else 0.5
            child_sex = np.random.choice([0, 1], p=[male_adult_prob, 1 - male_adult_prob])
            child = Person(age=child_age, sex=child_sex, position_in_hh='child')
            if child_sex == 0:
                people_counts["Male Adults"] -= 1
            else:
                people_counts["Female Adults"] -= 1
            household_population.append(child)

        # Create Household
        household = Household(population=household_population)
        households.append(household)
        people.extend(household_population)

    # Continue with generating single parent households
    # Generate single mother households with children under 18
    for _ in range(num_single_mother_with_children):
        num_children_dist = distribution_with_children["single_mother_with_children"]["number_of_children"]
        prob = [
            num_children_dist["1"],
            num_children_dist["2"],
            num_children_dist["3"],
            num_children_dist["4+"]
        ]
        prob = np.array(prob)
        prob = prob / prob.sum()  # Normalize

        num_child = np.random.choice(
            ['1', '2', '3', '4+'],
            p=prob
        )

        if num_child == '4+':
            number_of_children = generate_number_of_children_4plus()
        else:
            number_of_children = int(num_child)

        age_group_probabilities = np.array(list(parent_age_distribution["single_mother_with_children"].values()))
        age_group_probabilities = age_group_probabilities / age_group_probabilities.sum()  # Normalize

        age_group = np.random.choice(
            list(parent_age_distribution["single_mother_with_children"].keys()),
            p=age_group_probabilities
        )
        age_range = age_ranges[age_group]
        mother_age = np.random.randint(age_range[0], age_range[1]+1)

        mother = Person(age=mother_age, sex=1, position_in_hh='parent')  # Female
        people_counts["Female Adults"] -= 1
        household_population = [mother]

        # Generate children's ages (up to 17)
        children_ages = generate_children_ages(mother_age, number_of_children, max_child_age=17)

        for child_age in children_ages:
            total_children_remaining = people_counts["Male Children"] + people_counts["Female Children"]
            if total_children_remaining <= 0:
                break
            male_child_prob = people_counts["Male Children"] / total_children_remaining if total_children_remaining > 0 else 0.5
            child_sex = np.random.choice([0, 1], p=[male_child_prob, 1 - male_child_prob])
            child = Person(age=child_age, sex=child_sex, position_in_hh='child')
            if child_sex == 0:
                people_counts["Male Children"] -= 1
            else:
                people_counts["Female Children"] -= 1
            household_population.append(child)

        household = Household(population=household_population)
        households.append(household)
        people.extend(household_population)

    # Generate single father households with children under 18
    for _ in range(num_single_father_with_children):
        num_children_dist = distribution_with_children["single_father_with_children"]["number_of_children"]
        prob = [
            num_children_dist["1"],
            num_children_dist["2"],
            num_children_dist["3"],
            num_children_dist["4+"]
        ]
        prob = np.array(prob)
        prob = prob / prob.sum()  # Normalize

        num_child = np.random.choice(
            ['1', '2', '3', '4+'],
            p=prob
        )

        if num_child == '4+':
            number_of_children = generate_number_of_children_4plus()
        else:
            number_of_children = int(num_child)

        age_group_probabilities = np.array(list(parent_age_distribution["single_father_with_children"].values()))
        age_group_probabilities = age_group_probabilities / age_group_probabilities.sum()  # Normalize

        age_group = np.random.choice(
            list(parent_age_distribution["single_father_with_children"].keys()),
            p=age_group_probabilities
        )
        age_range = age_ranges[age_group]
        father_age = np.random.randint(age_range[0], age_range[1]+1)

        father = Person(age=father_age, sex=0, position_in_hh='parent')  # Male
        people_counts["Male Adults"] -= 1
        household_population = [father]

        # Generate children's ages (up to 17)
        children_ages = generate_children_ages(father_age, number_of_children, max_child_age=17)

        for child_age in children_ages:
            total_children_remaining = people_counts["Male Children"] + people_counts["Female Children"]
            if total_children_remaining <= 0:
                break
            male_child_prob = people_counts["Male Children"] / total_children_remaining if total_children_remaining > 0 else 0.5
            child_sex = np.random.choice([0, 1], p=[male_child_prob, 1 - male_child_prob])
            child = Person(age=child_age, sex=child_sex, position_in_hh='child')
            if child_sex == 0:
                people_counts["Male Children"] -= 1
            else:
                people_counts["Female Children"] -= 1
            household_population.append(child)

        household = Household(population=household_population)
        households.append(household)
        people.extend(household_population)

    # Part 2: Generate family households without children
    total_family_households = int(total_households * (family_households_info["total_percentage"] / 100))
    total_family_households_generated_with_children = (
        num_married_couple_with_children +
        num_single_mother_with_children +
        num_single_father_with_children +
        num_married_couple_with_children_18_25
    )
    number_of_family_households_without_children = total_family_households - total_family_households_generated_with_children

    # Number of married couples without children
    num_married_couple_without_children = int(total_number_of_married_couples * (married_couples_percentages["without_children"] / 100))
    num_married_couple_without_children = min(num_married_couple_without_children, number_of_family_households_without_children)

    married_couple_without_children_age_distribution = input_data["pop_info"]["married_couple_without_children_age_distribution"]

    # Generate married couple households without children
    for _ in range(num_married_couple_without_children):
        # Assign age group to partners
        age_group_probabilities = np.array(list(married_couple_without_children_age_distribution.values()))
        age_group_probabilities = age_group_probabilities / age_group_probabilities.sum()  # Normalize

        age_group = np.random.choice(
            list(married_couple_without_children_age_distribution.keys()),
            p=age_group_probabilities
        )
        age_range = age_ranges[age_group]

        # Assign ages to partners
        partner1_age = np.random.randint(age_range[0], age_range[1]+1)
        partner2_age = np.random.randint(age_range[0], age_range[1]+1)

        # Create partner Person objects with position_in_hh
        partner1 = Person(age=partner1_age, sex=0, position_in_hh='partner')  # Male
        partner2 = Person(age=partner2_age, sex=1, position_in_hh='partner')  # Female

        # Decrement counts
        people_counts["Male Adults"] -= 1
        people_counts["Female Adults"] -= 1

        household_population = [partner1, partner2]

        # Create Household
        household = Household(population=household_population)
        households.append(household)
        people.extend(household_population)

    # Update total family households generated
    total_family_households_generated = total_family_households_generated_with_children + num_married_couple_without_children
    remaining_family_households = total_family_households - total_family_households_generated

    # print households
    print_all_households(households)

    print("Remaining Family Households:", remaining_family_households) #TODO This is not correct, still have 27% family households


    print("Remaining People Counts after Family Households:", people_counts)
    return households, people

if __name__ == "__main__":
    total_households = input_data["total_households"]
    avg_household_size = input_data["avg_household_size"]
    total_population = int(total_households * avg_household_size)

    hh, people = gen_households(total_households, total_population)
