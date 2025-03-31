"""
Note: The code logic and algorithms have not been modified.
This revised version fixes minor issues and removes unnecessary parts.

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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Configuration Class
# ----------------------------
class ClusteringConfig:
    def __init__(self, 
                 output_dir=None,
                 location_name='hagerstown',
                 core_cbg='240430006012',
                 min_cluster_pop=10000,
                 states=None):
        self.output_dir = output_dir or '.'
        self.location_name = location_name
        self.core_cbg = core_cbg
        self.min_cluster_pop = min_cluster_pop
        self.states = states or ["Maryland", "Pennsylvania"]
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)


# ================================
# Function Definitions
# ================================

def cbg_geocode(cbg, df, poif):
    '''Generates the latitude and longitude coordinates for a CBG

    Args:
        cbg (str): CBG ID
        df (pd.DataFrame): DataFrame containing SafeGraph data.
        poif (pd.DataFrame): DataFrame containing POI data.
        
    Returns:
        pd.Series: Label, Latitude, and Longitude for the given CBG.
    '''
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
    Finds the latitude and longitude coordinates for a POI.

    Args:
        poi (pd.Series or str): Contains SafeGraph data for a POI or its place ID.

    Returns:
        pd.Series: Label, Latitude, and Longitude for the POI.
    """
    # Assumes 'poif' is available as a global variable.
    place_id = poi if isinstance(poi, str) else poi['safegraph_place_id']
    poi_data = poif.loc[poif['safegraph_place_id'] == place_id]
    return pd.Series(data={
        'label': place_id,
        'latitude': poi_data.iloc[0]['latitude'],
        'longitude': poi_data.iloc[0]['longitude']
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
    lat1, long1, lat2, long2 = lat1 * pi / 180, long1 * pi / 180, lat2 * pi / 180, long2 * pi / 180
    Radius = 6371
    haversine = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((long2 - long1) / 2)**2
    c = 2 * atan2(sqrt(haversine), sqrt(1 - haversine))
    return Radius * c


def gen_graph(df, weighted=True):
    """
    Creates an undirected graph from a SafeGraph dataframe.
    
    Args:
        df (pd.DataFrame): SafeGraph dataset.
        weighted (bool, optional): Whether edges are weighted by visitor counts.
        
    Returns:
        nx.Graph: Graph where nodes represent CBGs.
    """
    G = nx.Graph()
    regg_count = 0
    for _, row in df.iterrows():
        poi_cbg = str(row['poi_cbg'])
        try:
            poi_cbg = str(int(float(poi_cbg)))
        except:
            regg_count += 1
            continue

        G.add_node(poi_cbg, pois=[])
        G.nodes[poi_cbg]['pois'].append(row['safegraph_place_id'])
        wt = row['median_dwell'] if weighted else 1

        for visitor_cbg, num_visitors in json.loads(row['visitor_daytime_cbgs']).items():
            if visitor_cbg == poi_cbg:
                continue
            if G.has_edge(visitor_cbg, poi_cbg):
                try:
                    G[visitor_cbg][poi_cbg]['weight'] += int(num_visitors * wt)
                except ZeroDivisionError:
                    continue
            else:
                try:
                    G.add_weighted_edges_from([(visitor_cbg, poi_cbg, int(num_visitors * wt))])
                except ZeroDivisionError:
                    continue

        if G.degree[poi_cbg] == 0:
            G.remove_node(poi_cbg)

    UG = G.to_undirected()
    for node in G:
        for node2 in nx.neighbors(G, node):
            if node in nx.neighbors(G, node2):
                UG.edges[node, node2]['weight'] = G.edges[node, node2]['weight'] + G.edges[node2, node]['weight']
    print('G has %d nodes and %d edges.' % (nx.number_of_nodes(UG), nx.number_of_edges(UG)))
    print("Bad data count:", regg_count)
    return UG


def gen_d_graph(df, gdf, weighted=True):
    """
    Creates a distance-weighted undirected graph from the SafeGraph data.
    
    Args:
        df (pd.DataFrame): SafeGraph dataset.
        gdf (GeoDataFrame): Geospatial data for CBGs.
        weighted (bool, optional): Whether edges are weighted by visitor counts.
        
    Returns:
        nx.Graph: Graph with distance-adjusted weights.
    """
    G = nx.Graph()
    seed_row = gdf[gdf['GEOID'] == '240430006012']
    seed_latitiude = seed_row['latitude'].iloc[0]
    seed_long = seed_row['longitude'].iloc[0]
    regg_count = 0
    for _, row in df.iterrows():
        poi_cbg = str(row['poi_cbg'])
        try:
            poi_cbg = str(int(float(poi_cbg)))
        except:
            regg_count += 1
            continue

        G.add_node(poi_cbg, pois=[])
        G.nodes[poi_cbg]['pois'].append(row['safegraph_place_id'])
        wt = row['median_dwell'] if weighted else 1

        for visitor_cbg, num_visitors in json.loads(row['visitor_daytime_cbgs']).items():
            if visitor_cbg == poi_cbg:
                continue
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
                    G[visitor_cbg][poi_cbg]['weight'] += int(num_visitors * wt) / dist
                except ZeroDivisionError:
                    G[visitor_cbg][poi_cbg]['weight'] += int(num_visitors * wt)
            else:
                try:
                    G.add_weighted_edges_from([(visitor_cbg, poi_cbg, int(num_visitors * wt) / dist)])
                except ZeroDivisionError:
                    G.add_weighted_edges_from([(visitor_cbg, poi_cbg, int(num_visitors * wt))])

        if G.degree[poi_cbg] == 0:
            G.remove_node(poi_cbg)

    UG = G.to_undirected()
    for node in G:
        for node2 in nx.neighbors(G, node):
            if node in nx.neighbors(G, node2):
                UG.edges[node, node2]['weight'] = G.edges[node, node2]['weight'] + G.edges[node2, node]['weight']
    print('G has %d nodes and %d edges.' % (nx.number_of_nodes(UG), nx.number_of_edges(UG)))
    print("Bad data count:", regg_count)
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
        # Assumes 'cbg_pops' is defined as a global variable.
        return int(cbg_pops.loc[int(cbg)].B00002e1)
    except (ValueError, TypeError):
        return 0


def graph_to_csv(UG, filename='adj_graph.csv'):
    """
    Saves the adjacency matrix of the graph to a CSV file.
    """
    if os.path.dirname(filename) == '':
        filename = os.path.join(OUTPUT_DIR, filename)
        
    adj_matrix = nx.adjacency_matrix(UG)
    array_adj_matrix = adj_matrix.toarray()
    nodes = list(UG.nodes())
    df_adj_matrix = pd.DataFrame(array_adj_matrix, index=nodes, columns=nodes)
    df_adj_matrix.to_csv(filename)
    print(f'Graph written to {filename}')


def greedy_weight(C, u0, min_pop):
    """
    Greedy algorithm to expand a cluster until a minimum population is reached.
    
    Args:
        C (nx.Graph): Graph.
        u0 (str): Seed CBG.
        min_pop (int): Minimum population threshold.
        
    Returns:
        tuple: (list of CBGs in cluster, total population)
    """
    cluster = [u0]
    population = cbg_population(u0)
    it = 1
    while population < min_pop:
        all_adj_cbgs = []
        for i in cluster:
            for j in list(C.adj[i]):
                if j not in all_adj_cbgs and j not in cluster:
                    all_adj_cbgs.append(j)
        max_movement = 0
        cbg_to_add = None

        for candidate in all_adj_cbgs:
            current_movement = 0
            for member in cluster:
                try:
                    current_movement += C.adj[candidate][member]['weight']
                except KeyError:
                    continue
            if current_movement > max_movement:
                max_movement = current_movement
                cbg_to_add = candidate

        if cbg_to_add is None:
            break

        cluster.append(cbg_to_add)
        print("  Prev pop ", population, end="", flush=True)
        population += cbg_population(cbg_to_add)
        print("  New pop ", population)
        it += 1
    return cluster, population


def pagerank_nibble(G, u0, alpha=0.85, eps=1.0e-4):
    '''Pagerank Nibble algorithm to find a cluster.
    
    Args:
        G (nx.Graph): Graph.
        u0 (str): Seed CBG.
        alpha (float): Reset probability.
        eps (float): Tolerance for stopping.
        
    Returns:
        tuple: (set of CBGs in cluster, dummy value 1)
    '''
    pr = nx.pagerank(G, alpha=alpha, personalization={u0: 1}, tol=1.0e-6)
    pr_items = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    cluster = {u0}
    min_conductance = np.inf
    for count, (cbg, p_score) in enumerate(pr_items, start=1):
        print(count)
        if p_score < eps:
            break
        cluster.add(cbg)
        conductance = nx.conductance(G, cluster)
        if conductance < min_conductance:
            min_conductance = conductance
            best_cluster = cluster.copy()
    return best_cluster, 1


def greedy_ratio(C, u0, min_pop):
    '''Greedy algorithm that maximizes the ratio of internal to external movement.
    
    Args:
        C (nx.Graph): Graph.
        u0 (str): Seed CBG.
        min_pop (int): Minimum population threshold.
        
    Returns:
        tuple: (list of CBGs in cluster, total population)
    '''
    cluster = [u0]
    population = cbg_population(u0)
    it = 1
    while population < min_pop:
        print(population)
        all_adj_cbgs = []
        for i in cluster:
            for j in list(C.adj[i]):
                if j not in all_adj_cbgs and j not in cluster:
                    all_adj_cbgs.append(j)
        max_ratio = 0
        cbg_to_add = None
        for candidate in all_adj_cbgs:
            cluster.append(candidate)
            movement_in = 0
            movement_out = 0
            for member in cluster:
                for neighbor in list(C.adj[member]):
                    if neighbor in cluster:
                        movement_in += C.adj[member][neighbor]['weight']
                    else:
                        movement_out += C.adj[member][neighbor]['weight']
            ratio = movement_in / (movement_in + movement_out) if (movement_in + movement_out) else 0
            if ratio > max_ratio:
                max_ratio = ratio
                cbg_to_add = candidate
            cluster.remove(candidate)
        
        if cbg_to_add is None:
            break
        cluster.append(cbg_to_add)
        print("Iteration #", it, " Prev pop", population, end="", flush=True)
        population += cbg_population(cbg_to_add)
        print(" New pop", population, " Movement in", movement_in, " Movement out", movement_out, " Ratio", ratio)
        it += 1
    
    return cluster, population


# The following functions are placeholders for further modularization.
# If you wish to integrate them into a frontend, you can implement them as needed.

def run_clustering(config, algorithm="greedy_weight"):
    """
    Placeholder for running the specified clustering algorithm and returning results.
    (Not fully implemented; logic is already present in main().)
    """
    # You can encapsulate your main() clustering logic here.
    pass


def create_cluster_map(result, gdf, config):
    """
    Placeholder for creating and saving a cluster visualization map.
    (Not fully implemented; map creation logic is in main().)
    """
    pass


# ================================
# Main Execution
# ================================

def main():
    # -------------------------
    # Setup and Data Preparation
    # -------------------------
    location_name = 'hagerstown'
    core_cbg = '240430006012'
    min_cluster_pop = 10000

    # Get zip codes from desired states
    search = SearchEngine()
    data_md = search.by_state("Maryland", returns=0)
    data_pa = search.by_state("Pennsylvania", returns=0)
    zip_codes = [int(i.zipcode) for i in data_md] + [int(i.zipcode) for i in data_pa]

    # Read SafeGraph data - try cached file first
    filename = os.path.join(OUTPUT_DIR, f"{location_name}.csv")
    try:
        print(f"Trying to read cached data from {filename}")
        df = pd.read_csv(filename)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Cached file not found, reading from source...")
        datalist = []
        source_file = r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\patterns.csv'
        with pd.read_csv(source_file, chunksize=10000) as reader:
            for chunk in reader:
                datalist.append(chunk[chunk['postal_code'].isin(zip_codes)])
        df = pd.concat(datalist, axis=0)
        try:
            df['poi_cbg'] = df['poi_cbg'].astype('int64')
        except Exception:
            print("Warning: Some poi_cbg values couldn't be converted to int64")
        print(f"Saving processed patterns data to {filename}")
        df.to_csv(filename)

    # Process POI data similarly
    for _, row in df.iterrows():
        if not row['visitor_daytime_cbgs']:
            continue
        for visitor_cbg in json.loads(row['visitor_daytime_cbgs']).keys():
            if visitor_cbg not in zip_codes:
                zip_codes.append(visitor_cbg)

    filename_poi = os.path.join(OUTPUT_DIR, f"{location_name}.pois.csv")
    try:
        print(f"Trying to read cached POI data from {filename_poi}")
        global poif
        poif = pd.read_csv(filename_poi)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Cached POI file not found, reading from source...")
        datalist = []
        source_file_poi = r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\2021_05_05_03_core_poi.csv'
        with pd.read_csv(source_file_poi, chunksize=10000) as reader:
            for chunk in reader:
                datalist.append(chunk[chunk['postal_code'].isin(zip_codes)])
        poif = pd.concat(datalist, axis=0)
        print(f"Saving processed POI data to {filename_poi}")
        poif.to_csv(filename_poi)

    # -------------------------
    # Load CBG Shapefiles
    # -------------------------
    gdf1 = gpd.read_file(r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\tl_2010_24_bg10\tl_2010_24_bg10.shp')
    gdf2 = gpd.read_file(r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\tl_2010_42_bg10\tl_2010_42_bg10.shp')
    gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2], ignore_index=True))

    # -------------------------
    # Generate Graph and Population Data
    # -------------------------
    global cbg_pops
    cbg_pops = pd.read_csv(r'E:\Dileno Diesease Modelling\Algorithms\Data-Files-1\safegraph_cbg_population_estimate.csv', index_col='census_block_group')

    # Compute representative coordinates for each CBG in the shapefile
    gdf['coords'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
    gdf['coords'] = [coords[0] for coords in gdf['coords']]
    gdf['longitude'] = gdf['coords'].apply(lambda x: x[0])
    gdf['latitude'] = gdf['coords'].apply(lambda x: x[1])
    print('Creating graph...')

    # Create the regular graph (or uncomment for distance graph)
    global G
    G = gen_graph(df)
    # Alternatively:
    # G = gen_d_graph(df, gdf)

    # (Optional) Save the graph as CSV
    # graph_to_csv(G)

    # Test the cbg_geocode function
    print("Test geocoding for CBG 240430005001:")
    print(cbg_geocode('240430005001', df, poif))

    total_cbgs_population = sum(cbg_population(cbg) for cbg in df.poi_cbg.drop_duplicates().tolist())
    print("Total CBG population:", total_cbgs_population)

    # -------------------------
    # Run the Greedy Clustering Algorithm
    # -------------------------
    print('Starting Greedy Weight algorithm...')
    algorithm_result = greedy_weight(G, core_cbg, min_cluster_pop)
    print('Algorithm done.')
    print('Cluster population:', algorithm_result[1], '# of CBGs:', len(algorithm_result[0]))
    print("Cluster CBGs:", algorithm_result[0])
    
    movement_in_cbgs = 0
    movement_out_cbgs = 0
    for i in algorithm_result[0]:
        for j in list(G.adj[i]):
            if j in algorithm_result[0]:
                movement_in_cbgs += G.adj[i][j]['weight'] / 2
            else:
                movement_out_cbgs += G.adj[i][j]['weight']
    print("IN", movement_in_cbgs, "OUT", movement_out_cbgs, "Ratio", movement_in_cbgs / (movement_in_cbgs + movement_out_cbgs))

    # Run PageRank Nibble algorithm
    print('Running PageRank Nibble algorithm...')
    pagerank_result = pagerank_nibble(G, core_cbg, 0.6)
    print('PageRank cluster size:', len(pagerank_result[0]))
    pagerank_pop = sum(cbg_population(i) for i in pagerank_result[0])
    print("PageRank population:", pagerank_pop)
    movement_in_pagerank = 0
    movement_out_pagerank = 0
    for i in pagerank_result[0]:
        for j in list(G.adj[i]):
            if j in pagerank_result[0]:
                movement_in_pagerank += G.adj[i][j]['weight'] / 2
            else:
                movement_out_pagerank += G.adj[i][j]['weight']
    print("PageRank IN:", movement_in_pagerank, "OUT:", movement_out_pagerank,
          "Ratio:", movement_in_pagerank / (movement_in_pagerank + movement_out_pagerank))

    # -------------------------
    # Create Map Visualizations using Folium
    # -------------------------
    # First Map: Initial CBG clustering visualization
    map_obj = folium.Map(location=[39.6418, -77.7199], zoom_start=12)
    cbg_n = {}
    algo_copy = algorithm_result[0].copy()
    for cbg in algorithm_result[0]:
        try:
            row = gdf[gdf['GEOID10'] == cbg].iloc[0]
        except IndexError:
            continue
        # Find neighboring CBGs (those touching the current geometry)
        current_neighbours = gdf[gdf.geometry.touches(row['geometry'])]
        for i in current_neighbours['GEOID10']:
            if i in algorithm_result[0]:
                continue
            cbg_n[i] = cbg_n.get(i, 0) + 1

    for key in cbg_n:
        if cbg_n[key] >= 3:
            shape = gdf[gdf['GEOID10'] == key]
            layer = folium.GeoJson(shape, style_function=lambda feature: {
                'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.7
            }).add_to(map_obj)
            layer.add_child(folium.Popup(key))
            algo_copy.append(key)
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
        for edge in G.edges(cbg):
            if edge[1] in algorithm_result[0]:
                movement_in_S += G[cbg][edge[1]]['weight'] / 2
            else:
                movement_out_S += G[cbg][edge[1]]['weight']
        ratio = movement_in_S / (movement_in_S + movement_out_S)
        if ratio >= 0.8:
            fillcolor, color = '#0000FF', '#0000FF'
        elif ratio >= 0.6:
            fillcolor, color = '#008000', '#008000'
        elif ratio >= 0.4:
            fillcolor, color = '#FFFF00', '#FFFF00'
        elif ratio >= 0.2:
            fillcolor, color = '#FFA500', '#FFA500'
        else:
            fillcolor, color = '#FF0000', '#FF0000'
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
            folium.GeoJson(geojson, style_function=lambda x: black_style).add_to(map_anim)

    map_anim.save(os.path.join(OUTPUT_DIR, "map_with_black_clusters.html"))

    # -------------------------
    # Export CBG Information to YAML
    # -------------------------
    cbg_info_list = []
    for cbg in algorithm_result[0]:
        pop_est = cbg_population(cbg)
        movement_in_S = 0
        movement_out_S = 0
        for neighbor in list(G.adj[cbg]):
            if neighbor in algorithm_result[0]:
                movement_in_S += G.adj[cbg][neighbor]['weight'] / 2
            else:
                movement_out_S += G.adj[cbg][neighbor]['weight']
        total_movement = movement_in_S + movement_out_S
        ratio = movement_in_S / total_movement if total_movement > 0 else None
        cbg_info_list.append({
            "GEOID10": str(cbg),
            "movement_in": movement_in_S,
            "movement_out": movement_out_S,
            "ratio": ratio,
            "estimated_population": pop_est
        })

    with open(os.path.join(OUTPUT_DIR, "cbg_info.yaml"), "w", encoding="utf-8") as outfile:
        yaml.dump(cbg_info_list, outfile)

    print(f"All outputs saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
