import numpy as np
from collections import defaultdict

class POIs:
    def __init__(self, poi_dict, location_dict, start_time):
        self.poi_dict = poi_dict
        self.start_time = start_time
        
        # Get number of POIs
        num_pois = len(poi_dict)
        
        # Initialize matrices with proper dimensions
        self.CP_matrix = np.zeros(num_pois)  # Changed from len(poi_dict) for consistency
        self.RD_matrix = np.zeros((num_pois, num_pois))  # Changed from eye() to zeros()
        self.RM_matrix = np.zeros((num_pois, num_pois))  # Changed from eye() to zeros()

        # Add diagonal 1s for POIs with no related brands
        np.fill_diagonal(self.RD_matrix, 1)
        np.fill_diagonal(self.RM_matrix, 1)

        for poi_id, poi_info in poi_dict.items():
            # Process related brands
            rd_dict = poi_info.get("related_same_day_brand", {})
            rm_dict = poi_info.get("related_same_month_brand", {})

            # Only process if there are related brands
            if rd_dict or rm_dict:
                # Remove diagonal 1 when we have related brands
                self.RD_matrix[poi_id][poi_id] = 0
                self.RM_matrix[poi_id][poi_id] = 0

                # Update matrices
                for matrix, related_dict in [(self.RD_matrix, rd_dict), (self.RM_matrix, rm_dict)]:
                    total = sum(related_dict.values())
                    if total > 0:  # Prevent division by zero
                        for related_poi_name, visit_count in related_dict.items():
                            related_poi_ids = location_dict.get(related_poi_name, [])
                            # Handle case where a business has multiple locations
                            for related_poi_id in related_poi_ids:
                                matrix[poi_id][related_poi_id] = visit_count / total

        # Ensure matrices are row-stochastic (each row sums to 1)
        row_sums = self.RD_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        self.RD_matrix = self.RD_matrix / row_sums

        row_sums = self.RM_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        self.RM_matrix = self.RM_matrix / row_sums

    def get_parameters(self, time_step):
        for poi in self.poi_dict:
            current_poi = self.poi_dict[poi]
            current_time = self.start_time + time_step
            current_hour = (current_time // 60) // 24
            self.C = np.append(self.C, current_poi["raw_visitor_counts"] / current_poi["popularity_by_hour"][current_hour])
