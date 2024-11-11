import csv
import json

def parse_json_field(field):
    """
    Safely parse a JSON field. If parsing fails, return an empty dictionary or list based on expected type.
    """
    if not field:
        return {}
    try:
        return json.loads(field)
    except json.JSONDecodeError:
        return {}

def preprocess_csv(file_path):
    """
    Preprocess the CSV file and create two dictionaries:
    - `poi_dict`: Safegraph_place_id as keys with details as values.
    - `location_dict`: Location_name as keys with lists of safegraph_place_id as values.

    :param file_path: Path to the hagerstown.csv file
    :return: A tuple (poi_dict, location_dict)
    """
    poi_dict = {}
    location_dict = {}

    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            safegraph_place_id = row['safegraph_place_id']
            location_name = row['location_name']

            if not safegraph_place_id:
                continue  # Skip rows without a safegraph_place_id

            # Parse JSON fields
            visits_by_day = parse_json_field(row['visits_by_day'])
            poi_cbg = parse_json_field(row['poi_cbg'])
            bucketed_dwell_times = parse_json_field(row['bucketed_dwell_times'])
            related_same_day_brand = parse_json_field(row['related_same_day_brand'])
            related_same_month_brand = parse_json_field(row['related_same_month_brand'])
            popularity_by_hour = parse_json_field(row['popularity_by_hour'])
            popularity_by_day = parse_json_field(row['popularity_by_day'])

            # Construct the inner dictionary for poi_dict
            poi_dict[safegraph_place_id] = {
                'location_name': location_name,
                'raw_visit_counts': int(row['raw_visit_counts']) if row['raw_visit_counts'] else 0,
                'raw_visitor_counts': int(row['raw_visitor_counts']) if row['raw_visitor_counts'] else 0,
                'visits_by_day': visits_by_day,
                'poi_cbg': poi_cbg,
                'distance_from_home': float(row['distance_from_home']) if row['distance_from_home'] else None,
                'bucketed_dwell_times': bucketed_dwell_times,
                'related_same_day_brand': related_same_day_brand,
                'related_same_month_brand': related_same_month_brand,
                'popularity_by_hour': popularity_by_hour,
                'popularity_by_day': popularity_by_day,
                'capacity': int(row['raw_visit_counts']) if row['raw_visit_counts'] else 0  # Added capacity field
            }

            # Add to location_dict
            if location_name not in location_dict:
                location_dict[location_name] = []
            location_dict[location_name].append(safegraph_place_id)
    sorted(poi_dict)
    sorted(location_dict)
    return poi_dict, location_dict
