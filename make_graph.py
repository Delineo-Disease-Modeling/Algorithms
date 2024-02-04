"""
make_graph:
    Helper methods for Delineo project
    INCLUDES:
        reading Safegraph data to generate mobility network
        geocoding of the CBGs
        interpreting and the clustering results
    DOES NOT INCLUDE: 
        clustering methods
    
Author:
    Weicheng Hu (whu14)
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import random
import json
import sys
from geopy import Nominatim


def genGraph(df, zipCodes=None, weighted=True):   
    """
    Reads a file name and outputs a networkX 

    Params
    ------
        df: the DataFrame containing all the safegraph data
        indexList: list of zip codes to be included
            default = None: does not filter out any
        weighted: if the edges are weighted 
            default: True

    Return
    ------
        G: the networkX graph
            node: all the CBG(POI)'s and CBG's
            edge: edge exists if visitors at the POI is from the CBG
            weight: the number of visitors between the POI and the CBG
    """
    #Create the networkX graph
    G = nx.Graph()
    
    #If no zip codes provided, use all rows
    zipCode = (zipCodes != None)

    for i in range(len(df)):
        if zipCode == True and int(df['postal_code'][i]) not in zipCodes:
            continue
            
        text = df['poi_cbg'][i]
        poi = str(int(text))
        loop = 0
        
        G.add_node(poi, pois = [])
        G.nodes[poi]['pois'].append(df['safegraph_place_id'][i])
                
        if weighted:
            weight = df.iloc[i]['median_dwell']
        else:
            weight = 1
        # comment: for the processing, this is json format, use pase here: https://www.w3schools.com/python/python_json.asp
        for visitor_cbg, num_visitor in json.loads(df.iloc[i]['visitor_daytime_cbgs']).items():
            visitor_cbg = str(int(visitor_cbg))
            #avoid loops in the graph
            if visitor_cbg == poi:
                loop += 2
                
            try:
                G[visitor_cbg][poi] += int(num_visitor)
            except:
                G.add_weighted_edges_from([(visitor_cbg, poi, int(num_visitor)*weight)])
                
        if G.degree[poi] - loop == 0 or G.degree[poi] == 0:
            G.remove_node(poi)
            
    print('G has %d nodes and %d edges.' %(nx.number_of_nodes(G),nx.number_of_edges(G)))
                
    #Return the graph
    return G


def getLabels(G):
    """
    Returns all the labels of the nodes in G in the order of nodes
    
    Params
    ------
        G: the graph
    """
    labels = []
    for i in range(len(nodes)):
        labels.append(G.nodes[nodes[i]]['label'])
    return labels


def plotSizes(results):
    '''
    Plot the number of CBGs in each cluster
    
    Params
    ------
        results: the clustering results in fixed format
    '''    
    size = [len(cluster) for cluster in results]
    plt.plot(size,'o')
    plt.xlabel('Label')
    plt.ylabel('Count of CBGs')
    
    
def labelNodes(G, results):
    '''
    Label nodes with the labels 
    
    Params
    ------
        G: the graph
        results: the clustering results in fixed format
    '''
    for i in range(len(results)):
        for CBG in results[i]:
            G.nodes[CBG]['label'] = i
            
        
def calcIntertia(G):
    '''
    Calculate intra and inter-cluster inertia
    
    Params
    ------
        G: the graph with nodes labelled
        
    Return
    ------
        intra: the sum of edges within clusters
        inter: the sum of edges between clusters
    '''
    intra = 0
    inter = 0
    nodes = list(G.nodes)

    for i in range(len(nodes)):
        for j in range(i,len(nodes)):
            node1 = nodes[i]
            node2 = nodes[j]
            try:
                weight = int(G[node1][node2]['weight'])
                if G.nodes[node1]['label'] == G.nodes[node2]['label']:
                    intra += weight
                elif G.nodes[node1]['label'] == '' or G.nodes[node2]['label'] == '':
                    continue
                else:
                    inter += weight
            except:
                pass

    return intra, inter


def clusterData(location_data, results):
    '''
    Create the cluster data DataFrame from clustering results and location data
    
    Params
    ------
        location_data: pd.DataFrame that contains the location data
        results: the clustering results in fixed format
    
    Return
    ------
        cluster_data: pd.DataFrame that contains the cluster data (group + long + lat)
    '''
    cbg_id = []
    cbg_cluster = []
    cbg_latitude = []
    cbg_longitude = []
    for i in range(len(results)):
        for cbg in results[i]: 
            try:
                cbg_latitude.append(location_data['latitude'][cbg])
                cbg_id.append(cbg)
                cbg_cluster.append(i)
            except:
                pass
                #cbg_cluster.append(-1)
            try:
                cbg_longitude.append(location_data['longitude'][cbg])
            except:
                pass
        
    cluster_data = pd.DataFrame()
    cluster_data['cbg_id'] = cbg_id
    cluster_data['cbg_cluster'] = cbg_cluster
    cluster_data['cbg_latitude'] = cbg_latitude
    cluster_data['cbg_longitude'] = cbg_longitude
    cluster_data.set_index('cbg_id', inplace=True, drop=True)
    
    return cluster_data


def geocode(df, zipCodes):
    '''
    Geocoding method for the CBGs
    
    param:
        df: the pd dataframe that stores the list of CBG's and street address\
        
    return:
        location_data: a new dataframe with CBG's coordinates, index are the CBG IDs
    '''
    N = df.shape[0]
    # set up the geocoding tool
    geolocator = Nominatim(user_agent="example app")
    
    # loop over all CBGs, store the coordinates in a dictionary
    location_dict = dict()
    for i in range(N):
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %f%%; coordinates collected: %d" \
                         % ('='*int(i/N*50), round(i/N*100, ndigits=3), len(location_dict)))
        sys.stdout.flush()
        if df['postal_code'][i] in zipCodes:
            cbg = df['poi_cbg'][i]
            address = ','.join([df['street_address'][i],df['city'][i],df['region'][i],'USA'])
            try:
                address = tuple(geolocator.geocode(address).point)
            except:
                continue
            address = (address[0], address[1])
            try:
                location_dict[cbg].append(address)
            except:
                location_dict[cbg] = [address]
    
    cbg_id = []           
    cbg_latitude = []
    cbg_longitude = []
    for cbg in list(location_dict.keys()):
        locs = location_dict[cbg]
        coor = np.mean(locs,axis=0)
        cbg_latitude.append(coor[0])
        cbg_longitude.append(coor[1])
        cbg_id.append(str(int(cbg)))

    location_data = pd.DataFrame()
    location_data['latitude'] = cbg_latitude
    location_data['longitude'] = cbg_longitude
    location_data['census_block_group'] = cbg_id
    
    location_data = location_data.set_index('census_block_group')
    return location_data

def geocode(df, G, CBGs):
    '''
    Geocoding method for the CBGs
    
    param:
        df: the pd dataframe that stores the list of CBG's and street address\
        
    return:
        location_data: a new dataframe with CBG's coordinates, index are the CBG IDs
    '''
    found = False
    N = df.shape[0]
    # set up the geocoding tool
    geolocator = Nominatim(user_agent="example app")
    
    # loop over all CBGs, store the coordinates in a dictionary
    location_dict = dict()
    for i in range(N):
        if found:
            break
        if df['poi_cbg'][i] == CBG:
            address = ','.join([df['street_address'][i],df['city'][i],df['region'][i],'USA'])
            try:
                address = tuple(geolocator.geocode(address).point)
                location_dict[cbg] = (address[0], address[1])
                found = True
            except:
                continue
                
    return location_dict

def genGraphBipartite(df, zipCodes=None, weighted=True):   
    """
    Reads a file name and outputs a networkX 

    Params
    ------
        df: the DataFrame containing all the safegraph data
        indexList: list of zip codes to be included
            default = None: does not filter out any
        weighted: if the edges are weighted 
            default: True

    Return
    ------
        G: the networkX graph
            node: all the CBG(POI)'s and CBG's
            edge: edge exists if visitors at the POI is from the CBG
            weight: the number of visitors between the POI and the CBG
    """
    #Create the networkX graph
    G = nx.Graph()
    POIs, CBGs = set(), set()
    
    #If no zip codes provided, use all rows
    zipCode = (zipCodes != None)

    for i in range(len(df)):
        if zipCode == True and int(df['postal_code'][i]) not in zipCodes:
            continue
            
        poi = df['safegraph_place_id'][i]
        loop = 0
        #POIs.add(poi)
        
        G.add_node(poi)
        
        if weighted:
            weight = df.iloc[i]['median_dwell']
        else:
            weight = 1
        for visitor_cbg, num_visitor in json.loads(df.iloc[i]['visitor_daytime_cbgs']).items():
            visitor_cbg = str(int(visitor_cbg))
            #avoid loops in the graph
            if visitor_cbg == poi:
                loop += 2
                
            try:
                G[visitor_cbg][poi] += int(num_visitor)
            except:
                G.add_weighted_edges_from([(visitor_cbg, poi, int(num_visitor)*weight)])
                CBGs.add(visitor_cbg)
                
        if G.degree[poi] - loop == 0 or G.degree[poi] == 0:
            G.remove_node(poi)
        else: 
            POIs.add(poi)
            
    print('G has %d nodes and %d edges.' %(nx.number_of_nodes(G),nx.number_of_edges(G)))
                
    #Return the graph
    return G, POIs, CBGs