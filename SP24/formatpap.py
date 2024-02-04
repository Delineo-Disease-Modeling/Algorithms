import json
import pandas as pd

class Person:
    '''
    Class for each individual
    '''
    def __init__(self, id, sex, age, household):
        self.id = id
        self.sex = sex # male 0 female 1
        self.age = age
        self.household = household
        self.location = household


if __name__ == '__main__':
    '''
    
    '''
    with open('result_hh.json') as file:
        result_hh = json.load(file)
    
    known_people = []
    new_ppl = []
    
    for households in result_hh.values():
        for house, people in households.items():
            for person in people:
                if person['id'] not in known_people:
                    known_people.append(person['id'])
                    new_ppl.append(Person(person['id'], person['sex'], person['age'], person['cbg'], house))
    
    known_people.sort()
    print(new_ppl)
        
    '''
    Population Data Conversion
    '''
    
    # pop_data = pd.read_csv('safegraph_cbg_population_estimate.csv', usecols=['census_block_group', 'B00001e1', 'B00002e1'])
    # pop_data.columns = ['census_block_group', 'population', 'households']
    # pop_data.dropna(inplace=True)
    # pop_data = pop_data.astype({'population': int, 'households': int})
    # pop_data.to_csv('cbg_populations.csv', index=False)

    '''
    result_poi data to papdata
    '''
    
    with open('result_poi.json') as file:
        result_poi = json.load(file)
        
    known_ids = []
    poi_data = {'places': {}}

    for timestep, poidata in result_poi.items():
        for label in poidata.keys():
            poi_id = label[3:(label[3:].index('_')+3)]
            poi_name = label[label[3:].index('_')+4:]

            poi_data['places'][poi_id] = {'label': poi_name, 'cbg': '-1'}
    
    with open('papdata_pois.json', 'w', encoding='utf-8') as f:
        json.dump(poi_data, f, ensure_ascii=False, indent=4)
        
        
