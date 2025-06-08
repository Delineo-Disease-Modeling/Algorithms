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
    def __init__(self):
        self.location_name = "hagerstown"
        self.states = ["Maryland", "Pennsylvania"]
        self.core_cbg = "240430006012"
        self.min_cluster_pop = 5000
        self.output_dir = r"./output"
        self.paths = {
            "shapefile_md": r"./data/tl_2010_24_bg10/tl_2010_24_bg10.shp",
            "shapefile_pa": r"./data/tl_2010_42_bg10/tl_2010_42_bg10.shp",
            "patterns_csv": r"./data/patterns.csv",
            "poi_csv": r"./data/2021_05_05_03_core_poi.csv",
            "population_csv": r"./data/safegraph_cbg_population_estimate.csv",
            "output_yaml": "cbg_info.yaml",
            "output_html": "map_with_black_clusters.html"
        }
        self.map = {
            "default_location": [39.6418, -77.7199],
            "zoom_start": 12
        }
        self.ratio_colors = {
            0.8: "#0000FF",  # Blue
            0.6: "#008000",  # Green
            0.4: "#FFFF00",  # Yellow
            0.2: "#FFA500",  # Orange
            0.0: "#FF0000",  # Red
        }
        self.black_cbgs = [
            "240430002003", "240430003021", "240430001002", "240430001001",
            "240430008001", "240430008002", "240430008003", "240430007003",
            "240430010014", "240430010012", "240430010021", "240430102003"
        ]
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def __init__(self, cbg, zip_code, name, min_pop):
        search = SearchEngine()
        if isinstance(zip_code, list):
            self.states = [ search.by_zipcode(zip).state for zip in zip_code ]
        else:
            self.states = [ search.by_zipcode(zip_code).state ]
        
        self.location_name = name
        self.core_cbg = cbg
        self.min_cluster_pop = min_pop
        self.output_dir = r"./output"
        self.paths = {
            "shapefile_md": r"./data/tl_2010_24_bg10/tl_2010_24_bg10.shp",
            "shapefile_pa": r"./data/tl_2010_42_bg10/tl_2010_42_bg10.shp",
            "patterns_csv": r"./data/patterns.csv",
            "poi_csv": r"./data/2021_05_05_03_core_poi.csv",
            "population_csv": r"./data/safegraph_cbg_population_estimate.csv",
            "output_yaml": "cbg_info.yaml",
            "output_html": "map_with_black_clusters.html"
        }
        self.map = {
            "default_location": [0.0, 0.0],
            "zoom_start": 12
        }
        self.ratio_colors = {
            0.8: "#0000FF",  # Blue
            0.6: "#008000",  # Green
            0.4: "#FFFF00",  # Yellow
            0.2: "#FFA500",  # Orange
            0.0: "#FF0000",  # Red
        }
        self.black_cbgs = [ ]
        os.makedirs(self.output_dir, exist_ok=True)

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
        full_filename = os.path.join(self.config.output_dir, filename)
        try:
            self.logger.info(f"Loading SafeGraph data from {full_filename}")
            df = pd.read_csv(full_filename)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.logger.info(f"File {full_filename} not found. Processing raw data.")
            datalist = []
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
        self.logger.info("Loading shapefiles")
        gdf1 = gpd.read_file(self.config.paths["shapefile_md"])
        gdf2 = gpd.read_file(self.config.paths["shapefile_pa"])
        gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))
        # Process coordinates for mapping
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
    if _population_cache is None:
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
            try:
                for neighbor in G.adj[cbg]:
                    if neighbor in cluster_cbgs:
                        movement_in += G.adj[cbg][neighbor]['weight'] / 2
                    else:
                        movement_out += G.adj[cbg][neighbor]['weight']
            except:
                pass
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
        self.map_obj = folium.Map(location=center, zoom_start=self.config.map["zoom_start"])
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
        self.map_obj.add_child(plugins.TimestampedGeoJson(
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
                ).add_to(self.map_obj)
        output_map_path = os.path.join(self.config.output_dir, self.config.paths["output_html"])
        self.map_obj.save(output_map_path)
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
    

def generate_cz(cbg, zip, name, min_pop):
    config = Config(cbg, zip, name, min_pop)

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
    
    # Get per-cbg population data (hh generator needs this)
    geoids = {str(int(float(cbg))):cbg_population(str(int(float(cbg))), config, logger) for cbg in algorithm_result[0]}

    logger.info("Processing complete")
    
    return geoids, visualizer.map_obj

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        main_logger = logging.getLogger("cbg_clustering")
        main_logger.critical("Fatal error occurred", exc_info=True)
        raise
