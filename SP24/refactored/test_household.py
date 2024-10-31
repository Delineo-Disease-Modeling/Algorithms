import json
import random
import matplotlib.pyplot as plt
import re

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

    def __init__(self, population: list[Person] = [], cbg: int = None):
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


'''
The idea is to generate the number of people in each catagory first so that the total population would be close to the actual
distribution. 

Then genreate each household in a specific order based on the composition of the household.
Children is unique in that they can not be alone in a household and can help us determine the age of the parents in the household.

1. We start with the generate 2 adult households w children, using data on # of children in households. 
    This step should use up all the children in the population.
    
2. Next we generate living alone households. This should be relatively easy since we have all the data.( male/female householder living alone, over 65 living alone, etc)

3. Now the rest of the population consists of grandparents living with child/grandchild, other relatives, and some special cases.

4. Assign grandparents and other relatives to the existing households.

5. The remaining households are the special cases.

'''



def gen_basic_people_distribution(pop_info):
    '''
    returns the number of people in each age and sex group
    '''


def gen_hh_alone(hh_info, people):



def gen_households(hh_info, people):
    ''' Steps
    1. generate 2 adult households w children (since children can help us determine the age of the adults)
        a. 1 child
        b. 2 children
        c. 3 children
        d. 4+ children
    2. generate living alone households (1 person, could be any age range > 18)
        a. male living alone
        b. female living alone
    3. generate 2 adult married households w/o children (could be any age range > 18)
    4. other households (need to find data, but a small percentage)
    5. add grandparents and other relatives to the existing households (need to find data, but a small percentage)
    '''
    households = []
    people = []

    # generate 2 adult households w children






def read_population_info():
    with open("input/pop_info.json", "r") as file:
        pop_info = json.load(file)
    return pop_info

def read_household_info():
    with open("input/hh_info.json", "r") as file:
        hh_info = json.load(file)
    return hh_info


if __name__ == "__main__":
    # read the parameters
    pop_info = read_population_info()
    hh_info = read_household_info()

    hh = gen_households(hh_info, pop_info)


