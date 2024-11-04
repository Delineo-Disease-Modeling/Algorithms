import numpy as np
from collections import defaultdict

class POIs:
    def __init__(self, poi_dict, location_dict, start_time):
        poi_dict.sort()
        self.poi_dict = poi_dict
        self.start_time = start_time
        self.CP_matrix = np.array([0] * len(poi_dict))
        self.RD_matrix = np.array([0] * len(poi_dict))
        self.RM_matrix = np.array([0] * len(poi_dict))
        for poi in poi_dict:
            current_poi = self.poi_dict[poi]
            current_poi["related_same_day_brand"]
            current_poi["related_same_month_brand"] 

    def get_parameters(self, time_step):
        for poi in self.poi_dict:
            current_poi = self.poi_dict[poi]
            current_time = self.start_time + time_step
            current_hour = (current_time // 60) // 24
            self.C = np.append(self.C, current_poi["raw_visitor_counts"] / current_poi["popularity_by_hour"][current_hour])
