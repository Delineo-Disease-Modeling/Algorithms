import yaml
from household import Household, Person
from simulate import Simulate
from inter_hh import InterHousehold
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

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
    person_id_counter = 1
    for i in range(0, len(hh_list)):
        hh = hh_list[i]
        hh.total_count = len(hh.population)
        hh.id = i + 1
        for person in hh.population:
            person.hh_id = i + 1
            person.location = hh
            person.household = hh
            person.id = person_id_counter
            person_id_counter += 1
    return hh_list


def visualize_simulation_results():
    with open('output/result_hh.json') as f:
        hh_data = json.load(f)
    with open('output/result_poi.json') as f:
        poi_data = json.load(f)
    
    timesteps = sorted([int(step.split('_')[1]) for step in hh_data.keys()])
    timestep_keys = [f'timestep_{step}' for step in timesteps]
    
    hh_counts = [sum(len(household) for household in hh_data[step].values()) for step in timestep_keys]
    poi_counts = [sum(sum(len(person) for person in poi) for poi in poi_data[step].values()) for step in timestep_keys]
    
    
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(timesteps))
    
    plt.bar(index, hh_counts, bar_width, label='Households')
    plt.bar(index + bar_width, poi_counts, bar_width, label='POIs')
    
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title('Distribution of People in Households vs. POIs')
    plt.xticks(index + bar_width / 2, timesteps)
    plt.legend()
    plt.tight_layout()
    plt.show()



def animate(i, timesteps, hh_counts, poi_counts, bars):
    
    bars[0].set_height(hh_counts[i])
    bars[1].set_height(poi_counts[i])
    
    
    plt.title(f'Distribution of People at Timestep {timesteps[i]}')

def visualize_simulation_results_dynamic():
    with open('output/result_hh.json') as f:
        hh_data = json.load(f)
    with open('output/result_poi.json') as f:
        poi_data = json.load(f)
    
    
    timesteps = sorted([int(step.split('_')[1]) for step in hh_data.keys()])
    hh_counts = [sum(len(household) for household in hh_data[f'timestep_{step}'].values()) for step in timesteps]
    poi_counts = [sum(sum(len(person) for person in poi) for poi in poi_data[f'timestep_{step}'].values()) for step in timesteps]

    fig, ax = plt.subplots()
    bars = plt.bar(['Households', 'POIs'], [hh_counts[0], poi_counts[0]])

    
    ani = FuncAnimation(fig, animate, frames=len(timesteps), fargs=(timesteps, hh_counts, poi_counts, bars), repeat=False)
    
    plt.xlabel('Location Type')
    plt.ylabel('Count')
    plt.tight_layout()

    ani.save('simulation_distribution.gif', writer='pillow', fps=1)

def visualize_occupation_distribution():
    
    with open('output/result_hh.json') as f:
        hh_data = json.load(f)

    
    timestep_60_data = hh_data.get("timestep_60", {})

    
    occupation_counts = {}
    for household in timestep_60_data.values():
        for person in household:
            occupation = person.get("occupation")
            if occupation:
                occupation_counts[occupation] = occupation_counts.get(occupation, 0) + 1


    occupations = list(occupation_counts.keys())
    counts = list(occupation_counts.values())

   
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=occupations, autopct='%1.1f%%', startangle=140)
    plt.title('Occupation Distribution at Timestep 60')
    plt.axis('equal') 
    plt.show()

if __name__ == "__main__":
    with open('input/simul_settings.yaml', mode="r", encoding='utf-8') as settingstream:
        settings = yaml.safe_load(settingstream)

    with open('input/barnsdall.yaml', mode="r", encoding='utf-8') as citystream:
        city_info = yaml.safe_load(citystream)

    category_info = 'input/barnsdall.pois.csv'

    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object:__main__.Person', person_constructor)
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object:__main__.Household', household_constructor)
    
    hh_info = load_household()

    hh_info = format_hh(hh_info)

    simulate = Simulate(settings, city_info, hh_info, category_info)
    print("Starting simulation")
    simulate.start()
    print("Simulation has ended")

    #comment this out if you don't want make the animated gif
    #visualize_simulation_results_dynamic()

    visualize_simulation_results()

    visualize_occupation_distribution()

    