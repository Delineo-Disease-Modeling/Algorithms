import json

import networkx as nx

from common_geo import normalize_cbg


class GraphBuilder:
    def __init__(self, logger):
        self.logger = logger

    def _movement_records(self, df):
        for _, row in df.iterrows():
            val = row['visitor_daytime_cbgs']
            if not isinstance(val, str) or not val.strip():
                continue

            dst_cbg = normalize_cbg(row['poi_cbg'])
            if not dst_cbg:
                continue

            try:
                visitor_dict = json.loads(val)
                if isinstance(visitor_dict, str):
                    visitor_dict = json.loads(visitor_dict)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(visitor_dict, dict):
                continue

            for visitor_cbg, count in visitor_dict.items():
                try:
                    src_cbg = normalize_cbg(visitor_cbg)
                    weight = float(count)
                    if not src_cbg or weight <= 0:
                        continue
                    yield src_cbg, dst_cbg, weight
                except (TypeError, ValueError):
                    continue

    def gen_graph(self, df):
        self.logger.info("Generating graph from movement data")
        G = nx.Graph()

        for src_cbg, dst_cbg, weight in self._movement_records(df):
            if src_cbg == dst_cbg:
                if dst_cbg not in G:
                    G.add_node(dst_cbg, self_weight=0)
                G.nodes[dst_cbg]['self_weight'] = G.nodes[dst_cbg].get('self_weight', 0) + weight
                continue

            if G.has_edge(src_cbg, dst_cbg):
                G[src_cbg][dst_cbg]['weight'] += weight
            else:
                G.add_edge(src_cbg, dst_cbg, weight=weight)

        self.logger.info(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def gen_digraph(self, df):
        self.logger.info("Generating directed graph from movement data")
        DG = nx.DiGraph()

        for src_cbg, dst_cbg, weight in self._movement_records(df):
            if src_cbg == dst_cbg:
                if dst_cbg not in DG:
                    DG.add_node(dst_cbg, self_weight=0)
                DG.nodes[dst_cbg]['self_weight'] = DG.nodes[dst_cbg].get('self_weight', 0) + weight
                continue

            if DG.has_edge(src_cbg, dst_cbg):
                DG[src_cbg][dst_cbg]['weight'] += weight
            else:
                DG.add_edge(src_cbg, dst_cbg, weight=weight)

        self.logger.info(f"Generated directed graph with {DG.number_of_nodes()} nodes and {DG.number_of_edges()} edges")
        return DG
