import json

import networkx as nx

from common_geo import normalize_cbg


class GraphBuilder:
    def __init__(self, logger):
        self.logger = logger

    def gen_graph(self, df):
        self.logger.info("Generating graph from movement data")
        G = nx.Graph()

        for _, row in df.iterrows():
            val = row['visitor_daytime_cbgs']
            if not isinstance(val, str) or not val.strip():
                continue

            try:
                dst_cbg = normalize_cbg(row['poi_cbg'])
            except (TypeError, ValueError):
                continue

            visitor_dict = json.loads(val)
            if isinstance(visitor_dict, str):
                visitor_dict = json.loads(visitor_dict)
            for visitor_cbg, count in visitor_dict.items():
                try:
                    src_cbg = normalize_cbg(visitor_cbg)
                    weight = float(count)
                    if weight <= 0:
                        continue

                    if src_cbg == dst_cbg:
                        if dst_cbg not in G:
                            G.add_node(dst_cbg, self_weight=0)
                        G.nodes[dst_cbg]['self_weight'] = G.nodes[dst_cbg].get('self_weight', 0) + weight
                        continue

                    if G.has_edge(src_cbg, dst_cbg):
                        G[src_cbg][dst_cbg]['weight'] += weight
                    else:
                        G.add_edge(src_cbg, dst_cbg, weight=weight)
                except (TypeError, ValueError):
                    continue

        self.logger.info(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def gen_digraph(self, df):
        self.logger.info("Generating directed graph from movement data")
        DG = nx.DiGraph()

        for _, row in df.iterrows():
            val = row['visitor_daytime_cbgs']
            if not isinstance(val, str) or not val.strip():
                continue

            try:
                dst_cbg = normalize_cbg(row['poi_cbg'])
            except (TypeError, ValueError):
                continue

            visitor_dict = json.loads(val)
            if isinstance(visitor_dict, str):
                visitor_dict = json.loads(visitor_dict)
            for visitor_cbg, count in visitor_dict.items():
                try:
                    src_cbg = normalize_cbg(visitor_cbg)
                    weight = float(count)
                    if weight <= 0:
                        continue

                    if src_cbg == dst_cbg:
                        if dst_cbg not in DG:
                            DG.add_node(dst_cbg, self_weight=0)
                        DG.nodes[dst_cbg]['self_weight'] = DG.nodes[dst_cbg].get('self_weight', 0) + weight
                        continue

                    if DG.has_edge(src_cbg, dst_cbg):
                        DG[src_cbg][dst_cbg]['weight'] += weight
                    else:
                        DG.add_edge(src_cbg, dst_cbg, weight=weight)
                except (TypeError, ValueError):
                    continue

        self.logger.info(f"Generated directed graph with {DG.number_of_nodes()} nodes and {DG.number_of_edges()} edges")
        return DG
