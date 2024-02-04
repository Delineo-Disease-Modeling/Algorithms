from household import Population, Person, Household
import numpy as np
import pandas as pd
import yaml

def simulation(settings):
    # Runs for designated timesteps (minutes)
    for i in range(settings['time']):
        

if __name__=="__main__":
    print("main function loading")
    with open('simul_settings.yaml', mode="r") as settingstream:
        settings = yaml.full_load(settingstream)
    
    