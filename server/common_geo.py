import json
import os
from math import atan2, cos, pi, sin, sqrt


SERVER_DIR = os.path.dirname(__file__)


STATE_ABBR_TO_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
    'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
    'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
    'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
    'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
    'WY': '56', 'PR': '72', 'VI': '78', 'GU': '66', 'MP': '69',
    'AS': '60',
}
STATE_FIPS_TO_ABBR = {fips: abbr for abbr, fips in STATE_ABBR_TO_FIPS.items()}


def normalize_cbg(raw):
    """Convert any CBG representation to a zero-padded 12-digit string."""
    try:
        cbg_str = str(int(float(raw)))
    except (TypeError, ValueError):
        cbg_str = str(raw).strip()
    if len(cbg_str) == 11:
        cbg_str = cbg_str.zfill(12)
    if len(cbg_str) == 12 and cbg_str.isdigit():
        return cbg_str
    return None


def get_neighboring_states(states):
    try:
        neighbors = []
        path = os.path.join(SERVER_DIR, 'data', 'neighbor-states.json')
        with open(path, 'r', encoding='utf-8') as f:
            neighbor_list = json.load(f)

        for state in states:
            for item in neighbor_list:
                if item.get('code') == state:
                    neighbors.extend(item.get('Neighborcodes', []))
                    neighbors = list(set(neighbors))
                    break
        return neighbors
    except Exception:
        return []


def distance(lat1, long1, lat2, long2):
    """Calculate haversine distance between two coordinates in kilometers."""
    lat1, long1 = lat1 * pi / 180, long1 * pi / 180
    lat2, long2 = lat2 * pi / 180, long2 * pi / 180
    radius = 6371
    haversine = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((long2 - long1) / 2) ** 2
    c = 2 * atan2(sqrt(haversine), sqrt(1 - haversine))
    return radius * c


def build_cbg_centers(gdf):
    """Build a lookup of CBG -> (lat, lon) using representative points."""
    centers = {}
    if gdf is None or len(gdf) == 0:
        return centers

    cbg_col = None
    if 'CensusBlockGroup' in gdf.columns:
        cbg_col = 'CensusBlockGroup'
    elif 'GEOID' in gdf.columns:
        cbg_col = 'GEOID'
    if cbg_col is None:
        return centers

    try:
        gdf_wgs = gdf.to_crs("EPSG:4326")
    except Exception:
        gdf_wgs = gdf

    reps = gdf_wgs.representative_point()
    for idx, raw_cbg in gdf_wgs[cbg_col].items():
        cbg = normalize_cbg(raw_cbg) or str(raw_cbg).strip()
        if not cbg:
            continue
        point = reps.loc[idx]
        centers[cbg] = (float(point.y), float(point.x))

    return centers
