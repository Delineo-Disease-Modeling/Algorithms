import numpy as np
import pandas as pd
import random
import yaml

class Person:
    '''
    Class for each individual
    '''

    def __init__(self, id, sex, age, cbg, household, hh_id):
        self.id = id
        self.sex = sex # male 0 female 1
        self.age = age
        self.cbg = cbg
        self.household = household
        self.hh_id = hh_id

class Population:

    '''
    Class for storing population
    '''

    def __init__(self):
        # total population
        self.total_count = 0
        # container for persons in the population
        self.population = []
    
    def populate_indiv(self, person):
        '''
        Populates the population with the person object
        @param person = Person class object to be added to population
        '''
        
        #adding population
        self.population = np.append(self.population, person)

class Household(Population):

    ''' 
    Household class, inheriting Population since its a small population
    '''
    def __init__(self, cbg, total_count=0, population=[]):
        self.total_count = total_count
        self.population = population
        self.cbg = cbg


    def add_member(self, person):
        '''
        Adds member to the household, with sanity rules applied
        @param person = person to be added to household
        '''
        self.population.append(person)

    #TODO: Add more functions for leaving/coming back, etc if needed
    #jiwoo: an idea would be to extend from the population info to create
    #more realistic dataset (combination) of population in a household