import json
import csv
import numpy as np
from household import Household, Person
import pandas

class InterHousehold:
    def __init__(self, hh_list:list[Household]):
        self.iteration = 1
        self.hh_list = hh_list
        self.people:list[Person] = []
        
        for hh in hh_list:
            self.people += hh.population

        self.hh_by_cbg:dict = {}
        self.p_by_cbg:dict = {}

        for hh in hh_list:
            cbg = hh.cbg
            if self.hh_by_cbg.get(cbg) != None:
                self.hh_by_cbg[cbg].append(hh)
                self.p_by_cbg[cbg] += hh.population
            else:
                self.hh_by_cbg[cbg] = list([hh])
                self.p_by_cbg[cbg] = list(hh.population)
        
       

        self.social_hh:set[Household] = set()
        self.movement_people:set[Person] = set()
        


        self.individual_movement_frequency = 0.2

        self.social_event_frequency = 0.1
        self.social_guest_num = 10
        self.social_max_duration = 3

        self.school_children_frequency = 0.3
        self.regular_visitation_frequency = 0.15

        self.prefer_cbg = 0 # possobility that guests come from the same cbg


    def select_guest(self, cbg=None, size:int=1) -> list[Person]:
        if cbg:
            guests:list[Person] = np.random.choice(self.p_by_cbg[cbg], size=size, replace=False)
        else:
            guests:list[Person] = np.random.choice(self.people, size=size, replace=False)
        return [g for g in guests if g.availablility] # only return those who are availiable


    def select_hh(self, cbg=None, size:int=1) -> list[Household]:
        if cbg:
            hh:list[Household] = np.random.choice(self.hh_by_cbg[cbg], size=size, replace=False)
        else:
            hh:list[Household] = np.random.choice(self.hh_list, size=size, replace=False)
        
        return hh
        
    

    def random_boolean(self, probability_of_true):
        """
        Generates a random boolean value based on the given probability of True.
        
        :param probability_of_true: A float representing the probability of returning True, between 0 and 1.
        :return: A boolean value, where True occurs with the given probability.
        """
        return np.random.random() < probability_of_true


    def next(self):
        

        self.individual_movement()
        self.social_event()
        print(f"InterHousehold iteration {self.iteration}")
        self.iteration += 1
        
        # self.children_movement()

    
    def children_movement(self):
        for hh in self.hh_list:
            children = []
            adults = []
            for person in hh.population:
                if person.age < 18:
                    children.append(person)
                else:
                    adults.append(person)

            children_num = len(children) if len(children) < 3 else 3
            adult_num = len(adults) if len(adults) < 2 else 2

            if children_num > 0 and adult_num > 0 and self.random_boolean(self.school_children_frequency):
                children = np.random.choice(children, size=children_num, replace=False)
                adults = np.random.choice(adults, size=adult_num, replace=False)

                hh:Household = np.random.choice(self.hh_list, replace=False)
                for person in children + adults:
                    pass


    def social_event(self):
        # increase the houses already in social by 1 day
        for hh in list(self.social_hh):
            if hh.social_days >= hh.social_max_duration: # if the host day reachees maximum, stop social
                hh.end_social()
                self.social_hh.discard(hh)
            hh.social_days += 1


        number = int(self.social_event_frequency * len(self.hh_list)) # define a certain number of households that will host social events

        hh_social:list[Household] = np.random.choice(self.hh_list, size=number, replace=False) # choose hosueholds

        for hh in hh_social: # iterate through the randomly selected households
            if not hh.is_social(): # if the household is not hosting social
                if hh.population: # the household has its original population
                    self.social_hh.add(hh)
                    hh.start_social(duration=np.random.randint(1, self.social_max_duration + 1))
                    guest_num = np.random.randint(1, self.social_guest_num + 1)
                    guest = self.select_guest(hh.cbg, size=guest_num) if self.random_boolean(self.prefer_cbg) else self.select_guest(size=guest_num)
                    for person in guest:
                        if person.hh_id != hh.id and person not in hh.population: # if the person does not belong to the household member and is not a guest yet
                            person.assign_household(hh)


    def individual_movement(self):
        for person in list(self.movement_people):
            # put the person back to its original household
            person.assign_household(person.household)
            # remove person from movement list
            self.movement_people.discard(person)

        # Generate a random number between 0 and 1 for each person
        # and select the persono if the number is less than or equal to individual_movement_frequency
        selected_person = [person for person, rand in zip(self.people, np.random.rand(len(self.people))) if rand <= self.individual_movement_frequency]

        for person in selected_person:
            if person.location.social_days == 0 and person.availablility: # if move and person is not in social and person is not going to work
                same_cbg = self.random_boolean(self.prefer_cbg)

                hh = self.select_hh(person.cbg, size=1)[0] if same_cbg else self.select_hh(size=1)[0] # choose a random household in the same cbg
                
                while (hh.id == person.location.id and len(self.hh_list) >= 2 and hh.social_days > 0): # if person belongs to its household, there are enough household for selction, and the household is hosting social, reselct household
                    hh = self.select_hh(person.cbg, size=1)[0] if same_cbg else self.select_hh(size=1)[0]
                
                person.assign_household(hh)
                self.movement_people.add(person)
