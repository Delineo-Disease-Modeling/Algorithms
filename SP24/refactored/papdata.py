import json
import pandas as pd
from household import Household, Person

class Papdata:
    """
    A class to manage and process data for people, homes, and places.
    """


    def __init__(self, hh_info: list[Household], place_path: str, out_path='output/papdata.json'):
        """
        Initializes the Papdata class with household information, places data path, and output path.
        
        Parameters:
            hh_info (list): A list of all households.
            place_path (str): The path to a CSV file containing data about various places.
            out_path (str): The path to save the output JSON file. Defaults to 'output/papdata.json'.
        """
        self.hh_info = hh_info
        self.place_path = place_path
        self.out_path = out_path
        self.pap_dict:dict = {"people":{}, "homes":{}, "places":{}}

    def read_place(self) -> pd.DataFrame:
        """
        Reads the places data from a CSV file specified by the place_path attribute.
        
        Returns:
            pd.DataFrame: A DataFrame containing the places data.
        """
        df = pd.read_csv(self.place_path)
        
        return df

    def generate(self):
        """
        Processes the household information and places data to populate the pap_dict.
        """
        # add homes and persons
        '''
        for home in self.hh_info:
            home_id = home.population[0].hh_id
            self.pap_dict["homes"][home_id] = {"cbg":home.cbg, "members":len(home.population)}
            for person in home.population:
                self.pap_dict["people"][person.id] = {
                    "sex": person.sex, 
                    "age":person.age, 
                    "home":person.hh_id, 
                    "availability":person.availability, 
                    "work place":person.work_naics, #adjusted 
                    "work start time":person.work_time[0], 
                    "work end time":person.work_time[1]}
        '''

        # add homes and persons
        for h in self.hh_info:
            self.pap_dict["homes"][int(h.id)] = h.to_dict()
            for p in h.population:
                self.pap_dict["people"][int(p.id)] = p.to_dict()



        # add places
        places = self.read_place()
        places_dict = {}

        for index, row in places.iterrows():
            places_dict[index] = {"label":row['location_name'],
                            "cbg":-1,
                            "latitude":row['latitude'],
                            "longitude":row['longitude'],
                            "capacity":0,
                            }
        self.pap_dict['places'] = places_dict
        
        self.write()
    
    def write(self):
        """
        Writes the processed data stored in pap_dict to a JSON file specified by out_path.
        """
        with open(self.out_path, 'w', encoding='utf-8') as file:
            json.dump(self.pap_dict, file, indent=4)