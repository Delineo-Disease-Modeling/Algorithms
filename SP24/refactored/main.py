import yaml
from household import Household, Person
from simulate import Simulate
from inter_hh import InterHousehold
import numpy as np
# Define a custom constructor for loading Person objects
def person_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return Person(**fields)

# Define a custom constructor for loading Household objects
def household_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return Household(**fields)

def load_household():
    with open('input/households.yaml', 'r', encoding='utf-8') as hhstream:
        hh_info_pre = yaml.load(hhstream, Loader=yaml.SafeLoader)
        # print(hh_info_pre)

    hh_info = []

    for list_hh in hh_info_pre:
        for hh in list_hh:
            hh_info.append(hh)
    return hh_info

def format_hh(hh_list:list[Household]):
    for hh in hh_list:
        hh.total_count = len(hh.population)
        hh.id = hh.population[0].hh_id
    return hh_list

if __name__ == "__main__":
    with open('input/simul_settings.yaml', mode="r", encoding='utf-8') as settingstream:
        settings = yaml.safe_load(settingstream)

    with open('input/barnsdall.yaml', mode="r", encoding='utf-8') as citystream:
        city_info = yaml.safe_load(citystream)

    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object:__main__.Person', person_constructor)
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object:__main__.Household', household_constructor)
    
    hh_info = load_household()

    hh_info = format_hh(hh_info)
    
    simulate = Simulate(settings, city_info, hh_info)
    print("Starting simulation")
    simulate.start()
    print("Simulation has ended")