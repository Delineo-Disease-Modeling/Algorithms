import os
import json
import pickle
import logging
from math import sin, cos, atan2, pi, sqrt

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import folium
import yaml
from folium import plugins, Marker
from uszipcode import SearchEngine

# ----------------------------
# Configuration Module
# ----------------------------
class Config:
    def __init__(self, location_name=None, core_cbg=None, min_cluster_pop=5000, states=None, custom_black_cbgs=None):
        """
        Initialize a configuration for CBG clustering.
        
        Args:
            location_name: Name of the location (e.g., "baltimore", "hagerstown")
            core_cbg: Census Block Group ID to use as the core
            min_cluster_pop: Minimum population for the cluster
            states: List of state names to include
            custom_black_cbgs: List of CBGs to highlight in black
        """
        self.location_name = location_name or "hagerstown"
        self.states = states or ["Maryland", "Pennsylvania"]
        self.core_cbg = core_cbg or "240430006012"
        self.min_cluster_pop = min_cluster_pop
        
        # Set up paths to match where data files are stored
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, "output")
        
        # Set paths based on states
        self.paths = self._setup_paths()
        
        # Default map settings - will be updated based on location
        self.map = {
            "default_location": [39.6418, -77.7199], # Default Hagerstown coordinates
            "zoom_start": 12
        }
        
        # Color scheme for movement ratios
        self.ratio_colors = {
            0.8: "#0000FF",  # Blue
            0.6: "#008000",  # Green
            0.4: "#FFFF00",  # Yellow
            0.2: "#FFA500",  # Orange
            0.0: "#FF0000",  # Red
        }
        
        # Black CBGs - either custom or default for Hagerstown
        if custom_black_cbgs is not None:
            self.black_cbgs = custom_black_cbgs
        elif self.location_name == "hagerstown" and self.core_cbg == "240430006012":
            self.black_cbgs = [
                "240430002003", "240430003021", "240430001002", "240430001001",
                "240430008001", "240430008002", "240430008003", "240430007003",
                "240430010014", "240430010012", "240430010021", "240430102003"
            ]
        else:
            self.black_cbgs = []
            
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_paths(self):
        """Set up file paths based on configuration"""
        data_dir = os.path.join(self.base_dir, "Data-Files")
        
        paths = {
            "patterns_csv": os.path.join(data_dir, "patterns.csv"),
            "poi_csv": os.path.join(data_dir, "2021_05_05_03_core_poi.csv"),
            "population_csv": os.path.join(data_dir, "safegraph_cbg_population_estimate.csv"),
            "output_yaml": f"{self.location_name}_cbg_info.yaml",
            "output_html": f"{self.location_name}_map.html"
        }
        
        # Add shapefile paths based on states
        self._add_state_shapefiles(paths, data_dir)
        
        return paths
        
    def _add_state_shapefiles(self, paths, data_dir):
        """Add shapefile paths for each state in the configuration"""
        # Map of state names to FIPS codes
        state_to_fips = {
            "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05",
            "California": "06", "Colorado": "08", "Connecticut": "09", "Delaware": "10",
            "District of Columbia": "11", "Florida": "12", "Georgia": "13", "Hawaii": "15",
            "Idaho": "16", "Illinois": "17", "Indiana": "18", "Iowa": "19",
            "Kansas": "20", "Kentucky": "21", "Louisiana": "22", "Maine": "23",
            "Maryland": "24", "Massachusetts": "25", "Michigan": "26", "Minnesota": "27",
            "Mississippi": "28", "Missouri": "29", "Montana": "30", "Nebraska": "31",
            "Nevada": "32", "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35",
            "New York": "36", "North Carolina": "37", "North Dakota": "38", "Ohio": "39",
            "Oklahoma": "40", "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44",
            "South Carolina": "45", "South Dakota": "46", "Tennessee": "47", "Texas": "48",
            "Utah": "49", "Vermont": "50", "Virginia": "51", "Washington": "53",
            "West Virginia": "54", "Wisconsin": "55", "Wyoming": "56"
        }
        
        # Add shapefile paths for each state
        for state in self.states:
            state_abbr = state[:2].lower()
            if state in state_to_fips:
                fips = state_to_fips[state]
                shapefile_path = os.path.join(data_dir, f"tl_2010_{fips}_bg10", f"tl_2010_{fips}_bg10.shp")
                
                # Just log that the shapefile needs to be available
                if not os.path.exists(shapefile_path):
                    print(f"WARNING: Shapefile for {state} not found at {shapefile_path}")
                    print(f"You need to place the Census Block Group shapefile for {state} at this location.")
                
                paths[f"shapefile_{state_abbr}"] = shapefile_path
            else:
                print(f"WARNING: No FIPS code found for state: {state}")
    
    @classmethod
    def from_location(cls, location_name, seed_zip=None):
        """
        Create a Config instance for a specific location
        
        Args:
            location_name: Name of the location (city, town)
            seed_zip: Optional ZIP code to help identify the state and area
            
        Returns:
            Config: Configuration instance for the location
        """
        from uszipcode import SearchEngine
        
        search = SearchEngine()
        
        # If we have a seed ZIP code, use it to find location details
        if seed_zip:
            location = search.by_zipcode(seed_zip)
            if not location:
                raise ValueError(f"Could not find location for ZIP code {seed_zip}")
            
            state = location.state
            state_fips = location.state_fips
            county_fips = location.county_fips
            lat = location.lat
            lng = location.lng
        else:
            # Try to find the location by name
            results = search.by_city(city=location_name)
            if not results:
                raise ValueError(f"Could not find location: {location_name}")
            
            # Use the first result
            location = results[0]
            state = location.state
            state_fips = location.state_fips
            county_fips = location.county_fips
            lat = location.lat
            lng = location.lng
        
        # Generate a core CBG based on the location
        # Format: State FIPS (2) + County FIPS (3) + Tract (6) + Block Group (1)
        # We'll use a simplified approach for the tract and block group parts
        tract = "001001"  # Default tract (will be replaced with actual data if available)
        block_group = "1"  # Default block group
        
        core_cbg = f"{state_fips}{county_fips}{tract}{block_group}"
        
        # Create configuration
        config = cls(
            location_name=location_name.lower().replace(" ", "_"),
            core_cbg=core_cbg,
            states=[state],
            min_cluster_pop=5000  # Default minimum population
        )
        
        # Update map center based on location
        if lat and lng:
            config.map["default_location"] = [lat, lng]
            
        return config
    
    @classmethod
    def from_cbg(cls, cbg, min_pop=5000, custom_black_cbgs=None):
        """
        Create a Config instance based on a specific CBG
        
        Args:
            cbg: Census Block Group ID
            min_pop: Minimum population for clustering
            custom_black_cbgs: List of CBGs to highlight in black
            
        Returns:
            Config: Configuration instance for the CBG area
        """
        # Extract state FIPS from CBG (first 2 digits)
        state_fips = cbg[:2]
        
        # Map of FIPS codes to state names
        fips_to_state = {
            "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
            "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
            "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
            "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
            "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
            "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
            "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
            "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
            "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
            "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
            "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
            "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
            "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"
        }
        
        if state_fips not in fips_to_state:
            raise ValueError(f"Invalid state FIPS code in CBG: {state_fips}")
        
        state = fips_to_state[state_fips]
        location_name = f"cbg_{cbg}"  # Default location name based on CBG
        
        # Create configuration
        return cls(
            location_name=location_name,
            core_cbg=cbg,
            min_cluster_pop=min_pop,
            states=[state],
            custom_black_cbgs=custom_black_cbgs
        )


# ----------------------------
# Logging Setup
# ----------------------------
def setup_logging(config: Config):
    log_path = os.path.join(config.output_dir, "clustering.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("cbg_clustering")


# ----------------------------
# Data Loader Module
# ----------------------------
class DataLoader:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def get_zip_codes(self):
        """
        Retrieve zip codes for specified states.
        """
        search = SearchEngine()
        zip_codes = []
        for state in self.config.states:
            self.logger.info(f"Retrieving zip codes for {state}")
            data = search.by_state(state, returns=0)
            state_zips = [int(item.zipcode) for item in data]
            zip_codes.extend(state_zips)
        return zip_codes

    def load_safegraph_data(self, zip_codes):
        """
        Load SafeGraph patterns data, filtering by zip codes.
        """
        filename = f"{self.config.location_name}.csv"
        # Use os.path.join properly for the current platform
        full_filename = os.path.join(self.config.output_dir, filename)
        
        # Debug logging to check path
        self.logger.info(f"Looking for SafeGraph data at: {full_filename}")
        
        try:
            self.logger.info(f"Loading SafeGraph data from {full_filename}")
            df = pd.read_csv(full_filename)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.logger.info(f"File {full_filename} not found. Processing raw data.")
            datalist = []
            # Check if patterns file exists before trying to read it
            if not os.path.exists(self.config.paths["patterns_csv"]):
                self.logger.error(f"Raw patterns file not found at {self.config.paths['patterns_csv']}")
                raise FileNotFoundError(f"SafeGraph patterns file not found at {self.config.paths['patterns_csv']}")
                
            self.logger.info(f"Reading patterns from {self.config.paths['patterns_csv']}")
            with pd.read_csv(self.config.paths["patterns_csv"], chunksize=10000) as reader:
                for chunk in reader:
                    datalist.append(chunk[chunk['postal_code'].isin(zip_codes)])
            
            df = pd.concat(datalist, axis=0)
            try:
                df['poi_cbg'] = df['poi_cbg'].astype('int64')
            except (ValueError, TypeError):
                self.logger.warning("Unable to convert poi_cbg to int64")
            df.to_csv(full_filename)
            self.logger.info(f"Saved processed data to {full_filename}")
        return df

    def load_poi_data(self, zip_codes, df):
        """
        Load Points of Interest (POI) data, filtering by zip codes.
        """
        filename = f"{self.config.location_name}.pois.csv"
        full_filename = os.path.join(self.config.output_dir, filename)
        try:
            self.logger.info(f"Loading POI data from {full_filename}")
            poif = pd.read_csv(full_filename)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.logger.info(f"File {full_filename} not found. Processing raw POI data.")
            expanded_zip_codes = zip_codes.copy()
            for _, row in df.iterrows():
                if row['visitor_daytime_cbgs']:
                    for visitor_cbg in json.loads(row['visitor_daytime_cbgs']).keys():
                        if visitor_cbg not in expanded_zip_codes:
                            expanded_zip_codes.append(visitor_cbg)
            datalist = []
            with pd.read_csv(self.config.paths["poi_csv"], chunksize=10000) as reader:
                for chunk in reader:
                    datalist.append(chunk[chunk['postal_code'].isin(expanded_zip_codes)])
            poif = pd.concat(datalist, axis=0)
            poif.to_csv(full_filename)
            self.logger.info(f"Saved processed POI data to {full_filename}")
        return poif

    def load_shapefiles(self):
        """
        Load and merge shapefiles for the specified states.
        """
        self.logger.info("Loading shapefiles for states: " + ", ".join(self.config.states))
        
        # Get all shapefile paths from config
        shapefile_keys = [k for k in self.config.paths.keys() if k.startswith("shapefile_")]
        
        if not shapefile_keys:
            self.logger.error("No shapefiles configured for the specified states")
            raise ValueError("No shapefiles found in configuration")
        
        gdfs = []
        for key in shapefile_keys:
            shapefile_path = self.config.paths[key]
            if os.path.exists(shapefile_path):
                self.logger.info(f"Loading shapefile: {shapefile_path}")
                try:
                    gdf = gpd.read_file(shapefile_path)
                    gdfs.append(gdf)
                except Exception as e:
                    self.logger.error(f"Error loading shapefile {shapefile_path}: {e}")
            else:
                self.logger.warning(f"Shapefile not found: {shapefile_path}")
                
        if not gdfs:
            self.logger.error("No shapefiles could be loaded")
            raise FileNotFoundError("No shapefiles could be loaded")
            
        # Merge all geodataframes
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        
        # Process coordinates for mapping
        self.logger.info("Processing shapefile coordinates for mapping")
        gdf['coords'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
        gdf['coords'] = [coords[0] for coords in gdf['coords']]
        gdf['longitude'] = gdf['coords'].apply(lambda x: x[0])
        gdf['latitude'] = gdf['coords'].apply(lambda x: x[1])
        
        return gdf

    def get_population_data(self):
        """
        Load census population data for CBGs.
        """
        return pd.read_csv(self.config.paths["population_csv"], index_col='census_block_group')


# ----------------------------
# Utility Functions & Helpers
# ----------------------------
def distance(lat1, long1, lat2, long2):
    """
    Calculate haversine distance between two coordinates in kilometers.
    """
    lat1, long1 = lat1 * pi/180, long1 * pi/180
    lat2, long2 = lat2 * pi/180, long2 * pi/180
    radius = 6371  # km
    haversine = sin((lat2 - lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((long2 - long1)/2)**2
    c = 2 * atan2(sqrt(haversine), sqrt(1 - haversine))
    return radius * c

# For caching population data once loaded.
_population_cache = None
def cbg_population(cbg, config: Config, logger: logging.Logger):
    global _population_cache
    if (_population_cache is None):
        try:
            _population_cache = pd.read_csv(config.paths["population_csv"], index_col='census_block_group')
        except Exception as e:
            logger.error(f"Error loading population data: {e}")
            return 0
    try:
        cbg_int = int(cbg)
        if cbg_int in _population_cache.index:
            return int(_population_cache.loc[cbg_int].B00002e1)
        else:
            logger.warning(f"CBG {cbg} not found in population data")
            return 0
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(f"Error retrieving population for CBG {cbg}: {e}")
        return 0

class Helpers:
    @staticmethod
    def calculate_cbg_ratio(G, cbg, cluster_cbgs):
        movement_in = 0
        movement_out = 0
        for neighbor in G.adj[cbg]:
            if neighbor in cluster_cbgs:
                movement_in += G.adj[cbg][neighbor]['weight'] / 2
            else:
                movement_out += G.adj[cbg][neighbor]['weight']
        total_movement = movement_in + movement_out
        return movement_in / total_movement if total_movement > 0 else 0

    @staticmethod
    def calculate_movement_stats(G, cluster_cbgs):
        movement_in = 0
        movement_out = 0
        for cbg in cluster_cbgs:
            for neighbor in G.adj[cbg]:
                if neighbor in cluster_cbgs:
                    movement_in += G.adj[cbg][neighbor]['weight'] / 2
                else:
                    movement_out += G.adj[cbg][neighbor]['weight']
        total = movement_in + movement_out
        return {'in': movement_in, 'out': movement_out, 'ratio': movement_in / total if total > 0 else 0}


# ----------------------------
# Graph Builder Module
# ----------------------------
class GraphBuilder:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def gen_graph(self, df):
        """
        Generate a graph representing CBG connectivity based on movement data.
        """
        self.logger.info("Generating graph from movement data")
        G = nx.Graph()
        skipped_count = 0
        cbgs = set()
        # Collect all CBG IDs from patterns data
        for _, row in df.iterrows():
            cbgs.add(row['poi_cbg'])
            if isinstance(row['visitor_daytime_cbgs'], str):
                for visitor_cbg in json.loads(row['visitor_daytime_cbgs']).keys():
                    try:
                        cbgs.add(int(visitor_cbg))
                    except ValueError:
                        skipped_count += 1
                        cbgs.add(str(visitor_cbg))
                        self.logger.debug(f"Non-numeric visitor_cbg: {visitor_cbg}")
        if skipped_count > 0:
            self.logger.info(f"Skipped {skipped_count} non-numeric CBG IDs")
        # Add nodes to graph
        for cbg in cbgs:
            G.add_node(str(cbg))
        # Add edges based on visitor movement
        for _, row in df.iterrows():
            if isinstance(row['visitor_daytime_cbgs'], str):
                visitor_dict = json.loads(row['visitor_daytime_cbgs'])
                for visitor_cbg, count in visitor_dict.items():
                    try:
                        src_cbg = str(int(visitor_cbg)) if visitor_cbg.isdigit() else visitor_cbg
                        dst_cbg = str(row['poi_cbg'])
                        if src_cbg == dst_cbg:
                            continue  # skip self-loop
                        if G.has_edge(src_cbg, dst_cbg):
                            G[src_cbg][dst_cbg]['weight'] += count
                        else:
                            G.add_edge(src_cbg, dst_cbg, weight=count)
                    except ValueError:
                        continue
        self.logger.info(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G


# ----------------------------
# Clustering Algorithms Module
# ----------------------------
class Clustering:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def greedy_weight(self, G, u0, min_pop):
        """
        Greedy algorithm to build a cluster of CBGs based on movement weights.
        """
        self.logger.info(f"Starting greedy_weight algorithm with seed CBG {u0}")
        cluster = [u0]
        population = cbg_population(u0, self.config, self.logger)
        it = 1
        self.logger.info(f"Seed CBG population: {population}")
        while population < min_pop:
            all_adj_cbgs = []
            for i in cluster:
                try:
                    for j in list(G.adj[i]):
                        if j not in all_adj_cbgs and j not in cluster:
                            all_adj_cbgs.append(j)
                except KeyError:
                    self.logger.warning(f"CBG {i} not found in graph")
                    continue
            if not all_adj_cbgs:
                self.logger.warning(f"No adjacent CBGs found after {it} iterations. Cannot reach target population.")
                break
            max_movement = 0
            cbg_to_add = all_adj_cbgs[0]
            for candidate in all_adj_cbgs:
                current_movement = 0
                for member in cluster:
                    try:
                        current_movement += G.adj[candidate][member]['weight']
                    except (KeyError, ZeroDivisionError):
                        continue
                if current_movement > max_movement:
                    max_movement = current_movement
                    cbg_to_add = candidate
            cluster.append(cbg_to_add)
            cbg_pop = cbg_population(cbg_to_add, self.config, self.logger)
            population += cbg_pop
            self.logger.info(f"Iteration {it}: Added CBG {cbg_to_add} with pop {cbg_pop}. New total: {population}")
            it += 1
            if it > 1000:
                self.logger.warning("Reached maximum iterations (1000). Stopping algorithm.")
                break
        return cluster, population

    def greedy_ratio(self, G, u0, min_pop):
        """
        Greedy algorithm to build a cluster of CBGs based on movement ratio.
        """
        self.logger.info(f"Starting greedy_ratio algorithm with seed CBG {u0}")
        cluster = [u0]
        population = cbg_population(u0, self.config, self.logger)
        it = 1
        self.logger.info(f"Seed CBG population: {population}")
        while population < min_pop:
            all_adj_cbgs = []
            for i in cluster:
                try:
                    for j in list(G.adj[i]):
                        if j not in all_adj_cbgs and j not in cluster:
                            all_adj_cbgs.append(j)
                except KeyError:
                    self.logger.warning(f"CBG {i} not found in graph")
                    continue
            if not all_adj_cbgs:
                self.logger.warning(f"No adjacent CBGs found after {it} iterations. Cannot reach target population.")
                break
            max_ratio = 0
            cbg_to_add = all_adj_cbgs[0]
            for candidate in all_adj_cbgs:
                movement_in = 0
                movement_out = 0
                for j in G.adj[candidate]:
                    if j in cluster:
                        movement_in += G.adj[candidate][j]['weight']
                    else:
                        movement_out += G.adj[candidate][j]['weight']
                total_movement = movement_in + movement_out
                if total_movement > 0:
                    ratio = movement_in / total_movement
                    if ratio > max_ratio:
                        max_ratio = ratio
                        cbg_to_add = candidate
            cluster.append(cbg_to_add)
            cbg_pop = cbg_population(cbg_to_add, self.config, self.logger)
            population += cbg_pop
            self.logger.info(f"Iteration {it}: Added CBG {cbg_to_add} with pop {cbg_pop}. New total: {population}")
            it += 1
            if it > 1000:
                self.logger.warning("Reached maximum iterations (1000). Stopping algorithm.")
                break
        return cluster, population


# ----------------------------
# Visualization Module
# ----------------------------
class Visualizer:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def get_color_for_ratio(self, ratio):
        if ratio >= 0.8:
            return "#0000FF"
        elif ratio >= 0.6:
            return "#008000"
        elif ratio >= 0.4:
            return "#FFFF00"
        elif ratio >= 0.2:
            return "#FFA500"
        else:
            return "#FF0000"

    @staticmethod
    def cbg_geocode(cbg_id, df, poif, gdf=None):
        """
        Find geographical coordinates for a Census Block Group (CBG).
        
        Priority is given to the geodataframe if available; otherwise, falls back to POI data.
        
        Args:
            cbg_id: The CBG ID to locate.
            df: SafeGraph patterns dataframe.
            poif: Points of Interest dataframe.
            gdf: Optional geodataframe with CBG shapes.
            
        Returns:
            Dictionary with keys 'latitude' and 'longitude'.
        """
        # First try to find in geodataframe if available.
        if gdf is not None:
            cbg_data = gdf[gdf['GEOID10'] == cbg_id]
            if not cbg_data.empty and not pd.isna(cbg_data['latitude'].iloc[0]):
                return {
                    'latitude': cbg_data['latitude'].iloc[0],
                    'longitude': cbg_data['longitude'].iloc[0]
                }
        # Then try to find in POI data.
        poi_locations = poif[poif['census_block_group'] == int(cbg_id)]
        if not poi_locations.empty and not pd.isna(poi_locations['latitude'].iloc[0]):
            return {
                'latitude': poi_locations['latitude'].iloc[0],
                'longitude': poi_locations['longitude'].iloc[0]
            }
        # If not found, return default (None in this case).
        return {'latitude': None, 'longitude': None}

    def generate_maps(self, G, gdf, algorithm_result, poif, df):
        """
        Generate and save map visualizations.
        """
        def safe_center():
            try:
                seed = Visualizer.cbg_geocode(self.config.core_cbg, df, poif, gdf)
                if seed['latitude'] is None or seed['longitude'] is None:
                    return self.config.map["default_location"]
                return [seed['latitude'], seed['longitude']]
            except Exception:
                self.logger.warning("Error getting center coordinates, using default", exc_info=True)
                return self.config.map["default_location"]

        center = safe_center()
        map_obj = folium.Map(location=center, zoom_start=self.config.map["zoom_start"])
        features = []
        for i, cbg in enumerate(algorithm_result[0]):
            try:
                ratio = Helpers.calculate_cbg_ratio(G, cbg, algorithm_result[0])
                color = self.get_color_for_ratio(ratio)
                shape = gdf[gdf['GEOID10'] == cbg]
                if shape.empty:
                    continue
                shape = shape.to_crs("EPSG:4326")
                geojson = json.loads(shape.to_json())
                feature = geojson['features'][0]
                feature['properties']['times'] = [(pd.Timestamp('today') + pd.Timedelta(i, 'D')).isoformat()]
                feature['properties']['style'] = {'fillColor': color, 'color': color, 'fillOpacity': 0.7}
                features.append(feature)
            except Exception:
                self.logger.error(f"Error processing CBG {cbg} for map", exc_info=True)
        map_obj.add_child(plugins.TimestampedGeoJson(
            {'type': 'FeatureCollection', 'features': features},
            period='PT6H',
            add_last_point=True,
            auto_play=False,
            loop=False
        ))
        for cbg in self.config.black_cbgs:
            shape = gdf[gdf['GEOID10'] == cbg]
            if not shape.empty:
                shape = shape.to_crs("EPSG:4326")
                folium.GeoJson(
                    json.loads(shape.to_json()),
                    style_function=lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.7}
                ).add_to(map_obj)
        output_map_path = os.path.join(self.config.output_dir, self.config.paths["output_html"])
        map_obj.save(output_map_path)
        self.logger.info(f"Map saved to {output_map_path}")


# ----------------------------
# Export Module
# ----------------------------
class Exporter:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def generate_yaml_output(self, G, algorithm_result):
        cbg_info_list = []
        for cbg in algorithm_result[0]:
            try:
                cbg_str = str(cbg)
                pop_est = cbg_population(cbg, self.config, self.logger)
                movement_in_S, movement_out_S = 0, 0
                for neighbor in G.adj[cbg]:
                    if neighbor in algorithm_result[0]:
                        movement_in_S += G.adj[cbg][neighbor]['weight'] / 2
                    else:
                        movement_out_S += G.adj[cbg][neighbor]['weight']
                total_movement = movement_in_S + movement_out_S
                ratio = movement_in_S / total_movement if total_movement > 0 else None
                cbg_info_list.append({
                    "GEOID10": cbg_str,
                    "movement_in": movement_in_S,
                    "movement_out": movement_out_S,
                    "ratio": ratio,
                    "estimated_population": pop_est
                })
            except Exception:
                self.logger.error(f"Error processing CBG {cbg} for YAML output", exc_info=True)
        output_yaml_path = os.path.join(self.config.output_dir, self.config.paths["output_yaml"])
        with open(output_yaml_path, "w", encoding="utf-8") as outfile:
            yaml.dump(cbg_info_list, outfile)
        self.logger.info(f"YAML output saved to {output_yaml_path}")


# ----------------------------
# Main Execution Function
# ----------------------------
def main():
    config = Config()
    logger = setup_logging(config)
    logger.info("Starting clustering analysis")

    # Data loading
    data_loader = DataLoader(config, logger)
    zip_codes = data_loader.get_zip_codes()
    logger.info(f"Retrieved {len(zip_codes)} zip codes")
    df = data_loader.load_safegraph_data(zip_codes)
    poif = data_loader.load_poi_data(zip_codes, df)
    gdf = data_loader.load_shapefiles()
    _ = data_loader.get_population_data()  # Population data is cached in cbg_population

    # Graph generation
    graph_builder = GraphBuilder(logger)
    G = graph_builder.gen_graph(df)

    # Run clustering algorithm (using greedy_weight here)
    clustering_algo = Clustering(config, logger)
    algorithm_result = clustering_algo.greedy_weight(G, config.core_cbg, config.min_cluster_pop)
    logger.info(f"Clustering complete: {len(algorithm_result[0])} CBGs, population: {algorithm_result[1]}")
    movement_stats = Helpers.calculate_movement_stats(G, algorithm_result[0])
    logger.info(f"Movement stats: IN {movement_stats['in']}, OUT {movement_stats['out']}, Ratio {movement_stats['ratio']:.4f}")

    # Generate map visualizations
    visualizer = Visualizer(config, logger)
    visualizer.generate_maps(G, gdf, algorithm_result, poif, df)

    # Generate YAML output
    exporter = Exporter(config, logger)
    exporter.generate_yaml_output(G, algorithm_result)

    logger.info("Processing complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        main_logger = logging.getLogger("cbg_clustering")
        main_logger.critical("Fatal error occurred", exc_info=True)
        raise

# ----------------------------
# Server Integration Function
# ----------------------------
def generate_cz(cbg, zip_code, name, min_pop):
    """
    Generate a convenience zone starting from a core CBG.
    
    This function is called by server.py to generate a convenience zone
    for display in the web frontend.
    
    Args:
        cbg (str): Core Census Block Group ID
        zip_code (str or list): ZIP code(s) for location identification
        name (str): Location name
        min_pop (int): Minimum population for the cluster
        
    Returns:
        tuple: (cluster_result, map_object)
            - cluster_result is a tuple (list_of_cbgs, total_population)
            - map_object is a folium.Map instance
    """
    # Initialize custom black CBGs for specific locations
    custom_black_cbgs = None
    
    # Special handling for Hagerstown (preserve known black CBGs)
    if name.lower() == "hagerstown" and cbg == "240430006012":
        custom_black_cbgs = [
            "240430002003", "240430003021", "240430001002", "240430001001",
            "240430008001", "240430008002", "240430008003", "240430007003",
            "240430010014", "240430010012", "240430010021", "240430102003"
        ]
    
    # If we have a seed ZIP code, use it to create config
    if zip_code:
        if isinstance(zip_code, list) and zip_code:
            seed_zip = str(zip_code[0])
        else:
            seed_zip = str(zip_code)
            
        # Create config from location name and ZIP
        try:
            config = Config.from_location(name, seed_zip=seed_zip)
            # Override with provided core CBG
            config.core_cbg = cbg
            config.min_cluster_pop = min_pop
            config.black_cbgs = custom_black_cbgs or []
        except Exception as e:
            print(f"Error creating config from location: {e}")
            # Fallback to basic configuration if location lookup fails
            config = Config(
                location_name=name, 
                core_cbg=cbg, 
                min_cluster_pop=min_pop,
                custom_black_cbgs=custom_black_cbgs
            )
    else:
        # Create config directly from CBG
        try:
            config = Config.from_cbg(cbg, min_pop=min_pop, custom_black_cbgs=custom_black_cbgs)
            config.location_name = name
        except Exception as e:
            print(f"Error creating config from CBG: {e}")
            # Fallback to basic configuration
            config = Config(
                location_name=name, 
                core_cbg=cbg, 
                min_cluster_pop=min_pop,
                custom_black_cbgs=custom_black_cbgs
            )
    
    # Set up logging
    logger = setup_logging(config)
    logger.info(f"Starting clustering analysis for {name}, core CBG: {cbg}")
    
    # Process ZIP codes
    data_loader = DataLoader(config, logger)
    if zip_code:
        if isinstance(zip_code, list):
            zip_codes = [int(z) for z in zip_code if z]
        else:
            zip_codes = [int(zip_code)]
    else:
        zip_codes = data_loader.get_zip_codes()
    
    logger.info(f"Using {len(zip_codes)} zip codes")
    
    # Load data
    df = data_loader.load_safegraph_data(zip_codes)
    poif = data_loader.load_poi_data(zip_codes, df)
    gdf = data_loader.load_shapefiles()
    
    # Build graph
    graph_builder = GraphBuilder(logger)
    G = graph_builder.gen_graph(df)
    
    # Run clustering algorithm
    clustering_algo = Clustering(config, logger)
    algorithm_result = clustering_algo.greedy_weight(G, cbg, min_pop)
    
    # Create visualization map
    def safe_center():
        try:
            seed = Visualizer.cbg_geocode(config.core_cbg, df, poif, gdf)
            if seed['latitude'] is None or seed['longitude'] is None:
                return config.map["default_location"]
            return [seed['latitude'], seed['longitude']]
        except Exception as e:
            logger.warning(f"Error getting center coordinates, using default: {e}")
            return config.map["default_location"]

    center = safe_center()
    map_obj = folium.Map(location=center, zoom_start=config.map["zoom_start"])
    
    # Add cluster CBGs to map
    visualizer = Visualizer(config, logger)
    features = []
    for i, cbg in enumerate(algorithm_result[0]):
        try:
            ratio = Helpers.calculate_cbg_ratio(G, cbg, algorithm_result[0])
            color = visualizer.get_color_for_ratio(ratio)
            shape = gdf[gdf['GEOID10'] == cbg]
            if shape.empty:
                # Try alternative column name that might be used in shapefiles
                shape = gdf[gdf['GEOID'] == cbg]
                if shape.empty:
                    logger.warning(f"Could not find geometry for CBG {cbg}")
                    continue
            
            shape = shape.to_crs("EPSG:4326")
            geojson = json.loads(shape.to_json())
            feature = geojson['features'][0]
            feature['properties']['times'] = [(pd.Timestamp('today') + pd.Timedelta(i, 'D')).isoformat()]
            feature['properties']['style'] = {'fillColor': color, 'color': color, 'fillOpacity': 0.7}
            features.append(feature)
        except Exception as e:
            logger.error(f"Error processing CBG {cbg} for map: {e}")
            
    map_obj.add_child(plugins.TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period='PT6H',
        add_last_point=True,
        auto_play=False,
        loop=False
    ))
    
    # Add black CBGs
    logger.info(f"Adding {len(config.black_cbgs)} black CBGs to map")
    for cbg in config.black_cbgs:
        try:
            # Try with GEOID10 first (standard for 2010 CBG shapefiles)
            shape = gdf[gdf['GEOID10'] == cbg]
            
            # If not found, try with GEOID (used in some shapefiles)
            if shape.empty:
                shape = gdf[gdf['GEOID'] == cbg]
                
            if shape.empty:
                logger.warning(f"Could not find geometry for black CBG {cbg}")
                continue
                
            shape = shape.to_crs("EPSG:4326")
            geojson_data = json.loads(shape.to_json())
            
            # Add black CBG directly to map with popup showing its ID
            folium.GeoJson(
                geojson_data,
                name=f"Black CBG {cbg}",
                style_function=lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.7}
            ).add_child(folium.Popup(cbg)).add_to(map_obj)
            
            logger.info(f"Added black CBG {cbg} to map")
        except Exception as e:
            logger.error(f"Error adding black CBG {cbg}: {e}")
    
    # Generate YAML output if needed
    output_yaml_path = os.path.join(config.output_dir, config.paths["output_yaml"])
    exporter = Exporter(config, logger)
    exporter.generate_yaml_output(G, algorithm_result)
    logger.info(f"YAML output saved to {output_yaml_path}")
    
    # Return the algorithm result and map
    return algorithm_result, map_obj