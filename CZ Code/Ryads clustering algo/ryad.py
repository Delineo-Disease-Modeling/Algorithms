#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
production_notebook.py

This script consolidates an entire Jupyter Notebook into one production-level Python file.
It includes data setup, safegraph file processing, graph generation, clustering algorithm,
visualizations with folium, and YAML export.

Note: The code logic and algorithms have not been modified.
"""

# Standard library imports
import json
import pickle
from math import sin, cos, atan2, pi, sqrt
import yaml
import os

# Third-party imports
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import folium
from folium import plugins, Marker
import matplotlib.pyplot as plt
from uszipcode import SearchEngine

# Define output directory path globally
OUTPUT_DIR = r"E:\Dileno Diesease Modelling\Algorithms\CZ Code\Ryads clustering algo\outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# Function Definitions
# ================================

def cbg_geocode(cbg, df, poif):
    """
    Generates the latitude and longitude coordinates for a CBG

    Args:
        cbg (str): CBG ID

    Returns:
        pd.Series: ID, Latitude, & Longitude coords for a CBG
    """
    lat = []
    long = []
    for _, poi in df.loc[df['poi_cbg'] == int(cbg)].iterrows():
        poi_info = poif.loc[poif['safegraph_place_id'] == poi['safegraph_place_id']]
        lat.append(poi_info.iloc[0]['latitude'])
        long.append(poi_info.iloc[0]['longitude'])
    return pd.Series(data={
        'label': str(cbg),
        'latitude': np.mean(lat),
        'longitude': np.mean(long)
    })


def poi_geocode(poi):
    """
    Finds the latitude and longitude coordinates for a POI

    Args:
        poi (pd.Series): Series that contains the SafeGraph data for a specific POI

    Returns:
        pd.Series: Name, Latitude, & Longitude coords for a POI
    """
    place_id = poi if type(poi) == str else poi['safegraph_place_id']
    poi_data = poif.loc[poif['safegraph_place_id'] == place_id]
    return pd.Series(data={
        'label': place_id,
        'latitude': poi_data['latitude'],
        'longitude': poi_data['longitude']
    })


def distance(lat1, long1, lat2, long2):
    """
    Finds the distance between two coordinates in kilometers.

    Args:
        lat1, long1: Latitude and Longitude of point 1.
        lat2, long2: Latitude and Longitude of point 2.

    Returns:
        float: Distance in kilometers.
    """
    lat1 = lat1 * pi / 180
    long1 = long1 * pi / 180
    lat2 = lat2 * pi / 180
    long2 = long2 * pi / 180
    Radius = 6371
    haversine = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((long2 - long1) / 2)**2
    c = 2 * atan2(sqrt(haversine), sqrt(1 - haversine))
    dist = Radius * c
    return dist


def gen_graph(df, weighted=True):
    """
    Reads from a SafeGraph dataframe of a certain location and creates an undirected graph.

    Args:
        df (pd.DataFrame): The SafeGraph dataset of a particular location
        weighted (bool, optional): Whether the result should have weighted edges. Defaults to True.

    Returns:
        nx.Graph: Graph where nodes are the CBGs and the edges and weights are determined by visitor counts.
    """
    G = nx.Graph()
    regg_count = 0
    for _, row in df.iterrows():
        poi_cbg = str(row['poi_cbg'])
        try:
            int_poi = int(float(poi_cbg))
            poi_cbg = str(int_poi)
        except:
            regg_count += 1
            continue

        G.add_node(poi_cbg, pois=[])
        G.nodes[poi_cbg]['pois'].append(row['safegraph_place_id'])
        weight = row['median_dwell'] if weighted else 1

        for visitor_cbg, num_visitors in json.loads(row['visitor_daytime_cbgs']).items():
            if visitor_cbg == poi_cbg:
                continue  # Ignore edges that connect and come from the same node
            if G.has_edge(visitor_cbg, poi_cbg):
                try:
                    G[visitor_cbg][poi_cbg]['weight'] += int(num_visitors * weight)
                except ZeroDivisionError:
                    continue
            else:
                try:
                    G.add_weighted_edges_from([(visitor_cbg, poi_cbg, int(num_visitors * weight))])
                except ZeroDivisionError:
                    continue

        # Remove nodes without any edges
        if G.degree[poi_cbg] == 0:
            G.remove_node(poi_cbg)

    UG = G.to_undirected()
    for node in G:
        for node2 in nx.neighbors(G, node):
            if node in nx.neighbors(G, node2):
                UG.edges[node, node2]['weight'] = G.edges[node, node2]['weight'] + G.edges[node2, node]['weight']
    print('G has %d nodes and %d edges.' % (nx.number_of_nodes(UG), nx.number_of_edges(UG)))
    print("bad data ", regg_count)
    return UG


def gen_d_graph(df, gdf, weighted=True):
    """
    Reads from a SafeGraph dataframe of a certain location and creates a distance-weighted undirected graph.

    Args:
        df (pd.DataFrame): The SafeGraph dataset of a particular location
        gdf (GeoDataFrame): Geospatial data for CBGs.
        weighted (bool, optional): Whether the result should have weighted edges. Defaults to True.

    Returns:
        nx.Graph: Graph where nodes are the CBGs and the edges and weights are determined by visitors and distance.
    """
    G = nx.Graph()
    seed_row = gdf[gdf['GEOID'] == '240430006012']
    seed_latitiude = seed_row['latitude'].iloc[0]
    seed_long = seed_row['longitude'].iloc[0]
    regg_count = 0
    for _, row in df.iterrows():
        poi_cbg = str(row['poi_cbg'])
        try:
            int_poi = int(float(poi_cbg))
            poi_cbg = str(int_poi)
        except:
            regg_count += 1
            continue

        G.add_node(poi_cbg, pois=[])
        G.nodes[poi_cbg]['pois'].append(row['safegraph_place_id'])
        weight = row['median_dwell'] if weighted else 1

        for visitor_cbg, num_visitors in json.loads(row['visitor_daytime_cbgs']).items():
            if visitor_cbg == poi_cbg:
                continue  # Ignore edges that connect and come from the same node
            try:
                poi_row = gdf[gdf['GEOID'] == poi_cbg]
                poi_latitude = poi_row['latitude'].iloc[0]
                poi_long = poi_row['longitude'].iloc[0]
            except IndexError:
                regg_count += 1
                continue

            dist = int(distance(seed_latitiude, seed_long, poi_latitude, poi_long))
            if G.has_edge(visitor_cbg, poi_cbg):
                try:
                    G[visitor_cbg][poi_cbg]['weight'] += int(num_visitors * weight) / dist
                except ZeroDivisionError:
                    G[visitor_cbg][poi_cbg]['weight'] += int(num_visitors * weight)
            else:
                try:
                    G.add_weighted_edges_from([(visitor_cbg, poi_cbg, int(num_visitors * weight) / dist)])
                except ZeroDivisionError:
                    G.add_weighted_edges_from([(visitor_cbg, poi_cbg, int(num_visitors * weight))])

        # Remove nodes without any edges
        if G.degree[poi_cbg] == 0:
            G.remove_node(poi_cbg)

    # Change to undirected graph
    UG = G.to_undirected()
    for node in G:
        for node2 in nx.neighbors(G, node):
            if node in nx.neighbors(G, node2):
                UG.edges[node, node2]['weight'] = G.edges[node, node2]['weight'] + G.edges[node2, node]['weight']
    print('G has %d nodes and %d edges.' % (nx.number_of_nodes(UG), nx.number_of_edges(UG)))
    print("bad data ", regg_count)
    return UG


def cbg_population(cbg):
    """
    Get population of a CBG from the census data.
    
    Args:
        cbg (str): Census Block Group ID.
    
    Returns:
        int: Population of the CBG or 0 if not available.
    """
    try:
        return int(cbg_pops.loc[int(cbg)].B00002e1)
    except (ValueError, TypeError):
        return 0


def graph_to_csv(UG, filename='adj_graph.csv'):
    """
    Saves the adjacency matrix of the graph to a CSV file.
    """
    # If filename doesn't have a directory, add the output directory
    if os.path.dirname(filename) == '':
        filename = os.path.join(OUTPUT_DIR, filename)
        
    # Get adjacency matrix
    adj_matrix = nx.adjacency_matrix(UG)
    array_adj_matrix = adj_matrix.toarray()
    # Create the dataframe
    nodes = list(UG.nodes())
    df_adj_matrix = pd.DataFrame(array_adj_matrix, index=nodes, columns=nodes)
    df_adj_matrix.to_csv(filename)
    print(f'Graph written to {filename}')


def greedy_weight(C, u0, min_pop):
    """
    Greedy algorithm to expand a cluster until a minimum population is reached.

    Args:
        C (nx.Graph): The graph.
        u0 (str): Seed CBG.
        min_pop (int): Minimum population threshold.

    Returns:
        tuple: (cluster (list of CBGs), total population)
    """
    cluster = [u0]
    population = cbg_population(u0)
    it = 1
    while population < min_pop:
        # Get all adjacent CBGs of the cluster
        all_adj_cbgs = []
        for i in cluster:
            adj_cbgs = list(C.adj[i])
            for j in adj_cbgs:
                if j not in all_adj_cbgs and j not in cluster:
                    all_adj_cbgs.append(j)
        movement_in = 0
        movement_out = 0
        max_movement = 0
        cbg_to_add = 0

        # Calculate the movement from S to all adjacent CBGs
        for i in all_adj_cbgs:
            current_movement = 0
            for j in cluster:
                try:
                    current_movement += C.adj[i][j]['weight']
                except:
                    pass
            # Find the CBG with the greatest movement and add it to S
            if current_movement > max_movement:
                max_movement = current_movement
                cbg_to_add = i

        cluster.append(cbg_to_add)
        print("  Prev pop  ", population, end="", flush=True)
        population += cbg_population(cbg_to_add)
        print("  New pop  ", population, end="", flush=True)
        print("")
    return cluster, population


# ================================
# Main Execution
# ================================

def main():
    # -------------------------
    # Setup and Data Preparation
    # -------------------------
    # Change these variables to the intended area
    location_name = 'hagerstown'

    # Get zipcodes of intended state/s
    search = SearchEngine()
    data = search.by_state("Maryland", returns=0)
    data2 = search.by_state("Pennsylvania", returns=0)
    zip_codes1 = [i.zipcode for i in data]
    zip_codes1 = [int(zipcode) for zipcode in zip_codes1]
    zip_codes2 = [i.zipcode for i in data2]
    zip_codes2 = [int(zipcode) for zipcode in zip_codes2]
    zip_codes = zip_codes1 + zip_codes2
    # print(zip_codes)

    # Set CBG to cluster on and the minimum population (USER INPUT)
    core_cbg = '240430006012'
    min_cluster_pop = 5000

    # (Note: The following duplicate block from the original notebook has been consolidated.)
    # -------------------------
    # Read SafeGraph Files - Try cached first, then source
    # -------------------------
    # First try to read processed files
    filename = os.path.join(OUTPUT_DIR, f"{location_name}.csv")
    try:
        print(f"Trying to read cached data from {filename}")
        df = pd.read_csv(filename)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Cached file not found, reading from source...")
        datalist = []
        with pd.read_csv(r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\patterns.csv', chunksize=10000) as reader:
            for chunk in reader:
                datalist.append(chunk[chunk['postal_code'].isin(zip_codes)])
        
        df = pd.concat(datalist, axis=0)
        del datalist
        try:
            df['poi_cbg'] = df['poi_cbg'].astype('int64')
        except:
            print("Warning: Some poi_cbg values couldn't be converted to int64")
            pass
        
        # Save processed data to output directory
        print(f"Saving processed patterns data to {filename}")
        df.to_csv(filename)

    # Process POI data similarly
    cols = ['location_name', 'safegraph_place_id', 'latitude', 'longitude']

    # Add visitor CBGs to zip_codes list (same as before)
    for _, row in df.iterrows():
        if not row['visitor_daytime_cbgs']:
            continue
        for visitor_cbg in json.loads(row['visitor_daytime_cbgs']).keys():
            if visitor_cbg not in zip_codes:
                zip_codes.append(visitor_cbg)

    # Try to read cached POI data first
    filename = os.path.join(OUTPUT_DIR, f"{location_name}.pois.csv")
    try:
        print(f"Trying to read cached POI data from {filename}")
        poif = pd.read_csv(filename)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Cached POI file not found, reading from source...")
        datalist = []
        with pd.read_csv(r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\2021_05_05_03_core_poi.csv', chunksize=10000) as reader:
            for chunk in reader:
                datalist.append(chunk[chunk['postal_code'].isin(zip_codes)])
        
        poif = pd.concat(datalist, axis=0)
        del datalist
        
        # Save processed POI data
        print(f"Saving processed POI data to {filename}")
        poif.to_csv(filename)

    # -------------------------
    # Load CBG Shapefiles
    # -------------------------
    # 2010 Shapefiles used for SafeGraph (SafeGraph data is from 2021 but they use 2010 CBGs)
    gdf1 = gpd.read_file(r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\tl_2010_24_bg10\tl_2010_24_bg10.shp')
    gdf2 = gpd.read_file(r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\tl_2010_42_bg10\tl_2010_42_bg10.shp')
    gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))

    # -------------------------
    # Generate Graph and Population Data
    # -------------------------
    # Get population of a CBG from census data
    global cbg_pops
    cbg_pops = pd.read_csv(r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\safegraph_cbg_population_estimate.csv', index_col='census_block_group')

    # Get coordinates for each CBG
    gdf['coords'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
    gdf['coords'] = [coords[0] for coords in gdf['coords']]
    gdf['longitude'] = gdf['coords'].apply(lambda x: x[0])
    gdf['latitude'] = gdf['coords'].apply(lambda x: x[1])
    print('creating graph')

    # Create the regular graph
    global G
    G = gen_graph(df)
    # For distance graph, you could alternatively call:
    # G = gen_d_graph(df, gdf)

    # (Optional) Save or load the graph with pickle
    # with open(os.path.join(OUTPUT_DIR, "dist_graph.pkl"), "wb") as file:
    #     pickle.dump(G, file)
    # with open("dist_graph.pkl", "rb") as file:
    #     G = pickle.load(file)

    # (Optional) Save the graph as CSV
    # graph_to_csv(G)

    # Test the cbg_geocode function
    cbg_geocode('240430005001', df, poif)

    # Get total population of CBGs in the dataframe
    total_cbgs_population = 0
    for cbg in df.poi_cbg.drop_duplicates().tolist():
        total_cbgs_population += cbg_population(cbg)
    print("Total CBG population:", total_cbgs_population)

    # -------------------------
    # Run the Greedy Clustering Algorithm
    # -------------------------
    print('starting algo')
    algorithm_result = greedy_weight(G, core_cbg, min_cluster_pop)
    print('algo done')
    print('cluster population:', algorithm_result[1], '# of cbgs:', len(algorithm_result[0]))
    print(algorithm_result[0])
    movement_in_cbgs = 0
    movement_out_cbgs = 0

    for i in algorithm_result[0]:
        adj = list(G.adj[i])
        for j in adj:
            if j in algorithm_result[0]:
                movement_in_cbgs += G.adj[i][j]['weight'] / 2
            else:
                movement_out_cbgs += G.adj[i][j]['weight']

    print("IN ", movement_in_cbgs, "OUT", movement_out_cbgs, " Ratio ", movement_in_cbgs / (movement_in_cbgs + movement_out_cbgs))

    # -------------------------
    # Create Map Visualizations using Folium
    # -------------------------
    # First Map: Initial CBG clustering visualization
    map_obj = folium.Map(location=[39.6418, -77.7199], zoom_start=12)  # Adjust as needed

    # Surrounded CBGs
    cbg_n = {}
    algo_copy = algorithm_result[0].copy()
    for cbg in algorithm_result[0]:
        try:
            row = gdf[gdf['GEOID10'] == cbg].iloc[0]
        except IndexError:
            continue
        # Find all neighboring CBGs
        current_neighbours = gdf[gdf.geometry.touches(row['geometry'])]
        print("N ", current_neighbours['GEOID10'])
        for i in current_neighbours['GEOID10']:
            if i in algorithm_result[0]:
                continue
            if i not in cbg_n:
                cbg_n[i] = 1
            else:
                cbg_n[i] += 1

    for key in cbg_n:
        if cbg_n[key] >= 3:
            print('found')
            shape = gdf[gdf['GEOID10'] == key]
            layer = folium.GeoJson(shape, style_function=lambda feature: {
                'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.7
            }).add_to(map_obj)
            layer.add_child(folium.Popup(key))
            algo_copy.append(key)

    print('saving')
    map_obj.save(os.path.join(OUTPUT_DIR, "pagerank_10000.html"))

    # Second Map: Animated visualization of final CBG states
    movement_in_S = 0
    movement_out_S = 0
    seed = cbg_geocode('240430006012', df, poif)
    center = (seed['latitude'], seed['longitude'])
    map_anim = folium.Map(location=[center[0], center[1]], zoom_start=13)

    features = []
    for i, cbg in enumerate(algorithm_result[0]):
        movement_in_S = 0
        movement_out_S = 0
        shape = gdf[gdf['GEOID10'] == cbg]
        edges = G.edges(cbg)
        for j in edges:
            if j[1] in algorithm_result[0]:
                movement_in_S += G[cbg][j[1]]['weight'] / 2
            else:
                movement_out_S += G[cbg][j[1]]['weight']
        ratio = movement_in_S / (movement_in_S + movement_out_S)
        if ratio >= 0.8:
            fillcolor = '#0000FF'
            color = '#0000FF'
        elif ratio >= 0.6:
            fillcolor = '#008000'
            color = '#008000'
        elif ratio >= 0.4:
            fillcolor = '#FFFF00'
            color = '#FFFF00'
        elif ratio >= 0.2:
            fillcolor = '#FFA500'
            color = '#FFA500'
        elif ratio >= 0:
            fillcolor = '#FF0000'
            color = '#FF0000'
        if not shape.empty:
            shape = shape.to_crs("EPSG:4326")
            geojson = json.loads(shape.to_json())
            feature = geojson['features'][0]
            feature['properties']['times'] = [(pd.Timestamp('today') + pd.Timedelta(i, 'D')).isoformat()]
            feature['properties']['style'] = {'fillColor': fillcolor, 'color': color, 'fillOpacity': 0.7}
            features.append(feature)

    existing_cluster_layer = plugins.TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period='PT6H',
        add_last_point=True,
        auto_play=False,
        loop=False
    )
    map_anim.add_child(existing_cluster_layer)

    # Add black clusters to the map
    black_style = {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.7}
    black_cbgs = [
        "240430002003", "240430003021", "240430001002", "240430001001",
        "240430008001", "240430008002", "240430008003", "240430007003",
        "240430010014", "240430010012", "240430010021", "240430102003"
    ]
    for cbg in black_cbgs:
        shape = gdf[gdf['GEOID10'] == cbg]
        if not shape.empty:
            shape = shape.to_crs(epsg='4326')
            geojson = json.loads(shape.to_json())
            folium.GeoJson(
                geojson,
                style_function=lambda x: black_style
            ).add_to(map_anim)

    map_anim.save(os.path.join(OUTPUT_DIR, "map_with_black_clusters.html"))

    # -------------------------
    # Export CBG Information to YAML
    # -------------------------
    cbg_info_list = []
    for cbg in algorithm_result[0]:
        cbg_str = str(cbg)
        pop_est = cbg_population(cbg)
        movement_in_S = 0
        movement_out_S = 0
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

    with open(os.path.join(OUTPUT_DIR, "cbg_info.yaml"), "w", encoding="utf-8") as outfile:
        yaml.dump(cbg_info_list, outfile)

    print(f"All outputs saved to {OUTPUT_DIR}")


# ================================
# Entry Point
# ================================
if __name__ == '__main__':
    main()
