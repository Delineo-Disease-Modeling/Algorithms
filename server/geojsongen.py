"""
GeoJSON utilities for Census Block Groups (CBGs).
- Original: Generate individual state shapefiles
- Added: get_cbg_geojson for static map visualization
"""

import os
import json
import geopandas as gpd
import pandas as pd
from functools import lru_cache
from shapely.geometry import Point


# State FIPS code to abbreviation mapping
STATE_FIPS_TO_ABBR = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
    '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
    '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
    '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
    '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
    '56': 'WY', '72': 'PR', '78': 'VI', '66': 'GU', '69': 'MP',
    '60': 'AS',
}


# Cache loaded shapefiles to avoid reloading
@lru_cache(maxsize=10)
def load_state_shapefile(state_fips):
    """Load shapefile for a specific state."""
    shapefile_dir = r"./data/shapefiles/"
    
    # First try FIPS-named shapefile format (tl_2020_XX_bg)
    shapefile_path = os.path.join(shapefile_dir, f"tl_2020_{state_fips}_bg", f"tl_2020_{state_fips}_bg.shp")
    
    if os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        if 'GEOID' not in gdf.columns and 'CensusBlockGroup' in gdf.columns:
            gdf['GEOID'] = gdf['CensusBlockGroup']
        return gdf
    
    # Try state abbreviation geojson format
    state_abbr = STATE_FIPS_TO_ABBR.get(state_fips)
    if state_abbr:
        geojson_path = os.path.join(shapefile_dir, f"{state_abbr}.geojson")
        if os.path.exists(geojson_path):
            gdf = gpd.read_file(geojson_path)
            # Normalize column names
            if 'GEOID' not in gdf.columns and 'CensusBlockGroup' in gdf.columns:
                gdf['GEOID'] = gdf['CensusBlockGroup']
            return gdf
    
    # Try FIPS-named geojson as fallback
    geojson_path = os.path.join(shapefile_dir, f"{state_fips}.geojson")
    if os.path.exists(geojson_path):
        gdf = gpd.read_file(geojson_path)
        if 'GEOID' not in gdf.columns and 'CensusBlockGroup' in gdf.columns:
            gdf['GEOID'] = gdf['CensusBlockGroup']
        return gdf
    
    print(f"No shapefile found for state FIPS {state_fips} (abbr: {state_abbr})")
    return None


def load_population_data():
    """Load population data for CBGs."""
    pop_file = r"./data/cbg_b01.csv"
    if os.path.exists(pop_file):
        try:
            df = pd.read_csv(pop_file, dtype={'census_block_group': str})
            return dict(zip(df['census_block_group'].astype(str), df['B01001e1'].fillna(0).astype(int)))
        except Exception as e:
            print(f"Error loading population data: {e}")
    return {}


def get_cbg_geojson(cbg_list, include_neighbors=False):
    """
    Generate GeoJSON for the specified CBGs.
    
    Args:
        cbg_list: List of CBG IDs (12-digit GEOIDs)
        include_neighbors: If True, also include neighboring CBGs
        
    Returns:
        GeoJSON FeatureCollection
    """
    # Group CBGs by state FIPS (first 2 digits)
    cbgs_by_state = {}
    for cbg in cbg_list:
        cbg_str = str(cbg).zfill(12)
        state_fips = cbg_str[:2]
        if state_fips not in cbgs_by_state:
            cbgs_by_state[state_fips] = set()
        cbgs_by_state[state_fips].add(cbg_str)
    
    # Load population data
    pop_data = load_population_data()
    
    features = []
    all_loaded_gdfs = []
    
    # Load shapefiles and extract features
    for state_fips, state_cbgs in cbgs_by_state.items():
        gdf = load_state_shapefile(state_fips)
        if gdf is None:
            print(f"Shapefile not found for state FIPS {state_fips}")
            continue
        
        # Ensure GEOID column exists
        if 'GEOID' not in gdf.columns:
            if 'CensusBlockGroup' in gdf.columns:
                gdf['GEOID'] = gdf['CensusBlockGroup']
            else:
                print(f"No GEOID column in shapefile for state {state_fips}")
                continue
        
        all_loaded_gdfs.append(gdf)
        
        # Filter to requested CBGs
        gdf_filtered = gdf[gdf['GEOID'].isin(state_cbgs)].copy()
        
        if gdf_filtered.empty:
            continue
        
        # Convert to WGS84 for web mapping
        if gdf_filtered.crs and gdf_filtered.crs != "EPSG:4326":
            gdf_filtered = gdf_filtered.to_crs("EPSG:4326")
        
        # Add properties and convert to features
        for idx, row in gdf_filtered.iterrows():
            geoid = row['GEOID']
            geom = row.geometry.__geo_interface__
            
            population = pop_data.get(geoid, 0)
            
            feature = {
                'type': 'Feature',
                'geometry': geom,
                'properties': {
                    'GEOID': geoid,
                    'CensusBlockGroup': geoid,
                    'population': population,
                    'ratio': 0.5,
                    'in_cluster': True
                }
            }
            features.append(feature)
    
    # If include_neighbors, find adjacent CBGs
    if include_neighbors and all_loaded_gdfs:
        neighbor_cbgs = set()
        selected_set = set(str(cbg).zfill(12) for cbg in cbg_list)
        
        for gdf in all_loaded_gdfs:
            if gdf.crs and gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
                
            gdf_selected = gdf[gdf['GEOID'].isin(selected_set)]
            
            for idx, selected_row in gdf_selected.iterrows():
                # Find CBGs that touch this one
                touches = gdf[gdf.geometry.touches(selected_row.geometry)]
                for _, neighbor_row in touches.iterrows():
                    neighbor_geoid = neighbor_row['GEOID']
                    if neighbor_geoid not in selected_set and neighbor_geoid not in neighbor_cbgs:
                        neighbor_cbgs.add(neighbor_geoid)
                        
                        population = pop_data.get(neighbor_geoid, 0)
                        
                        feature = {
                            'type': 'Feature',
                            'geometry': neighbor_row.geometry.__geo_interface__,
                            'properties': {
                                'GEOID': neighbor_geoid,
                                'CensusBlockGroup': neighbor_geoid,
                                'population': population,
                                'ratio': 0.0,
                                'in_cluster': False
                            }
                        }
                        features.append(feature)
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }


def get_cbg_at_point(latitude, longitude, state_fips=None):
    """
    Resolve a latitude/longitude to a containing CBG GEOID.

    Args:
        latitude: point latitude
        longitude: point longitude
        state_fips: optional two-digit state FIPS hint to limit search

    Returns:
        Dict with GEOID and population, or None if no containing CBG is found.
    """
    try:
        lat = float(latitude)
        lng = float(longitude)
    except (TypeError, ValueError):
        return None

    if state_fips is not None:
        candidate_states = [str(state_fips).zfill(2)]
    else:
        candidate_states = list(STATE_FIPS_TO_ABBR.keys())

    point = Point(lng, lat)
    pop_data = load_population_data()

    for cur_state in candidate_states:
        gdf = load_state_shapefile(cur_state)
        if gdf is None:
            continue

        if 'GEOID' not in gdf.columns:
            if 'CensusBlockGroup' in gdf.columns:
                gdf = gdf.copy()
                gdf['GEOID'] = gdf['CensusBlockGroup']
            else:
                continue

        if gdf.crs and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # Use spatial index when available so we only test nearby polygons.
        try:
            candidate_idx = list(gdf.sindex.intersection((lng, lat, lng, lat)))
            if not candidate_idx:
                continue
            candidates = gdf.iloc[candidate_idx]
        except Exception:
            candidates = gdf

        matches = candidates[candidates.geometry.intersects(point)]
        if matches.empty:
            continue

        match = matches.iloc[0]
        geoid = str(match['GEOID']).zfill(12)
        return {
            'GEOID': geoid,
            'population': int(pop_data.get(geoid, 0)),
        }

    return None


# Original script for generating individual state shapefiles
def generate_state_shapefiles():
    import time
    import itertools
    from pyogrio import read_dataframe
    from uszipcode import SearchEngine

    print('Generating individual shapefiles...')

    engine = SearchEngine()
    geof = read_dataframe(r'./data/cbg_2020.geojson', columns=['State', 'CensusBlockGroup', 'geometry'])

    print('Read US shapefile data...')

    states = [ engine.find_state(state, best_match=False) for state in engine.state_list ]
    states = sorted(list(set(itertools.chain.from_iterable(states))))

    success = 0

    for state in states:
        try:
            start = time.time()
            geof[geof['State'] == state].to_file(f'./data/shapefiles/{state}.geojson')  
            end = time.time()
            
            success += 1
            print(f'Processed \'{state}\' in {end - start:.2f}')
        except:
            print(f'Error processing \'{state}\'')

    print(f'Generated {success}/{len(states)} shape files')


if __name__ == "__main__":
    generate_state_shapefiles()
