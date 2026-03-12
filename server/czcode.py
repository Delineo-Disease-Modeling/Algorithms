import os
import json
import logging
import folium
import yaml
import math

import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np

try:
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    SCIPY_MILP_AVAILABLE = True
except Exception:
    SCIPY_MILP_AVAILABLE = False

from folium import plugins
from uszipcode import SearchEngine
from math import sin, cos, atan2, pi, sqrt
from datetime import datetime as _datetime
from typing import Optional, List

from patterns_loader import (
    PatternsData, resolve_patterns_files, states_from_cbgs, PATTERNS_BASE_DIR
)

# USPS state abbreviation -> 2-digit state FIPS.
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

# ----------------------------
# Configuration Module
# ----------------------------
class Config:
    def __init__(self, cbg, min_pop, patterns_file=None, patterns_folder=None,
                 month=None, start_date: Optional[_datetime] = None):
        """
        Initialize configuration for CZ generation.

        Args:
            cbg: Census Block Group GEOID
            min_pop: Minimum population for the cluster
            patterns_file: Optional specific patterns CSV file path
            patterns_folder: Optional folder containing monthly pattern files (e.g., 2019-01-OK.csv)
            month: Optional month key (YYYY-MM) when using patterns_folder
            start_date: Optional simulation start date (used to auto-resolve monthly file)
        """
        zip_codes = []
        with open(r'./data/zip_to_cbg.json', 'r') as f:
            zip_to_cbg = json.load(f)
            for zip, cbgs in zip_to_cbg.items():
                if cbg in cbgs:
                    zip_codes.append(zip)

        search = SearchEngine()
        self.states = list(set([ search.by_zipcode(zip).state for zip in zip_codes ]))
        self.states = list(set(self.states) | set(get_neighboring_states(self.states)))

        self.location_name = f'{cbg}'
        self.core_cbg = cbg
        self.min_cluster_pop = min_pop
        self.output_dir = r"./output"
        self.start_date = start_date

        # Auto-derive month from start_date if not provided
        if month is None and start_date is not None:
            month = start_date.strftime('%Y-%m')

        # Determine which patterns file to use
        self.patterns_folder = patterns_folder
        self.month = month
        resolved_patterns_csv = self._resolve_patterns_file(patterns_file, patterns_folder, month)

        self.paths = {
            "shapefiles_dir": r"./data/shapefiles/",
            "patterns_csv": resolved_patterns_csv,
            # "population_csv": r"./data/safegraph_cbg_population_estimate.csv",
            "population_csv": r"./data/cbg_b01.csv",
            "output_yaml": "cbg_info.yaml",
            "output_html": "map.html"
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

    def _resolve_patterns_file(self, patterns_file, patterns_folder, month):
        """
        Resolve which patterns CSV file to use.

        Priority:
        1. Explicit patterns_file if provided
        2. Auto-resolve from state + month using patterns_loader (new folder structure)
        3. State-scoped monthly file (converted preferred) when month is provided
        """
        # Priority 1: Explicit file
        if patterns_file:
            if os.path.exists(patterns_file):
                return patterns_file
            raise FileNotFoundError(f"Patterns file not found: {patterns_file}")

        # Priority 2: Auto-resolve from state + month using patterns_loader
        if month and self.states:
            resolved = resolve_patterns_files(self.states,
                                              self.start_date or _datetime.now(),
                                              PATTERNS_BASE_DIR)
            if resolved:
                return resolved[0]

        search_root = patterns_folder or os.path.join(os.path.dirname(__file__), "data")
        month_key = str(month or "").strip()

        if month_key:
            state_candidates = []
            core = str(self.core_cbg or "").strip()
            if len(core) >= 2:
                state_hint = STATE_FIPS_TO_ABBR.get(core[:2])
                if state_hint:
                    state_candidates.append(state_hint)

            for state in self.states:
                state_code = str(state or "").strip().upper()
                if state_code and state_code not in state_candidates:
                    state_candidates.append(state_code)

            candidate_paths = []
            for state in state_candidates:
                stem = f"{month_key}-{state}"
                candidate_paths.extend([
                    os.path.join(search_root, state, f"{stem}.csv.gz"),
                    os.path.join(search_root, state, f"{stem}.converted.csv"),
                    os.path.join(search_root, state, f"{stem}.csv"),
                    os.path.join(search_root, f"{stem}.csv.gz"),
                    os.path.join(search_root, f"{stem}.converted.csv"),
                    os.path.join(search_root, f"{stem}.csv"),
                ])

            import glob
            for state in state_candidates:
                stem = f"{month_key}-{state}"
                candidate_paths.extend(sorted(glob.glob(
                    os.path.join(search_root, "**", f"{stem}.csv.gz"),
                    recursive=True
                )))
                candidate_paths.extend(sorted(glob.glob(
                    os.path.join(search_root, "**", f"{stem}.converted.csv"),
                    recursive=True
                )))
                candidate_paths.extend(sorted(glob.glob(
                    os.path.join(search_root, "**", f"{stem}.csv"),
                    recursive=True
                )))

            seen = set()
            ordered_paths = []
            for path in candidate_paths:
                norm = os.path.abspath(path)
                if norm in seen:
                    continue
                seen.add(norm)
                ordered_paths.append(path)

            for path in ordered_paths:
                if os.path.exists(path):
                    return path

            # Exact month not found — try closest available month
            from patterns_loader import closest_month, _available_months_for_state
            for state in state_candidates:
                available = _available_months_for_state(state, search_root)
                nearest = closest_month(month_key, available)
                if nearest and nearest != month_key:
                    nearest_stem = f"{nearest}-{state}"
                    fallback_candidates = [
                        os.path.join(search_root, state, f"{nearest_stem}.csv.gz"),
                        os.path.join(search_root, state, f"{nearest_stem}.converted.csv"),
                        os.path.join(search_root, state, f"{nearest_stem}.csv"),
                    ]
                    for path in fallback_candidates:
                        if os.path.exists(path):
                            import logging
                            logging.getLogger('cbg_clustering').info(
                                f"No data for {month_key}, using closest month: {nearest}"
                            )
                            return path

            states_msg = ", ".join(state_candidates) if state_candidates else "unknown state"
            raise FileNotFoundError(
                f"No state-scoped monthly patterns file found for month '{month_key}' "
                f"under {search_root} for {states_msg}. "
                "Expected files like '<DATA>/<STATE>/YYYY-MM-<STATE>.csv.gz' "
                "or '<DATA>/<STATE>/YYYY-MM-<STATE>.csv'."
            )

        raise FileNotFoundError(
            "No patterns_file was provided and no month was specified. "
            "Global default patterns fallback is disabled; provide a state-scoped monthly file."
        )

# ----------------------------
# Logging Setup
# ----------------------------
def setup_logging(config: Config):
    # log_path = os.path.join(config.output_dir, "clustering.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler(log_path),
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

    def load_safegraph_data(self, zip_codes, shared_data: 'PatternsData' = None):
        """
        Load SafeGraph patterns data, filtering by zip codes.
        Uses month-aware caching when monthly patterns are configured.

        Args:
            zip_codes: List of zip codes to filter by
            shared_data: Optional pre-loaded PatternsData to avoid re-reading CSV
        """
        # If we have pre-loaded shared data, use it directly
        if shared_data is not None and not shared_data.is_empty():
            self.logger.info("Using pre-loaded shared patterns data for clustering")
            df = shared_data.for_clustering()
            # Normalize column names to lowercase
            df.columns = [str(c).strip().lower() for c in df.columns]
            if 'poi_cbg' in df.columns:
                df['poi_cbg'] = pd.to_numeric(df['poi_cbg'], errors='coerce')
                df.dropna(subset=['poi_cbg'], inplace=True)
                df['poi_cbg'] = df['poi_cbg'].astype('int64').astype('string')
            return df

        patterns_csv = self.config.paths["patterns_csv"]
        source_stem = os.path.splitext(os.path.basename(patterns_csv))[0]
        source_mtime = int(os.path.getmtime(patterns_csv)) if os.path.exists(patterns_csv) else 0
        month_part = self.config.month if self.config.month else "nomonth"
        # Cache key includes source file + mtime so stale extracts are not reused.
        cache_version = "v3"
        filename = f"{self.config.location_name}_{month_part}_{source_stem}_{source_mtime}_{cache_version}.csv"

        full_filename = os.path.join(self.config.output_dir, filename)
        try:
            self.logger.info(f"Loading SafeGraph data from {full_filename}")
            df = pd.read_csv(full_filename)
            df.columns = [str(c).strip().lower() for c in df.columns]
            required_cols = {'poi_cbg', 'visitor_daytime_cbgs'}
            if not required_cols.issubset(df.columns):
                missing = sorted(required_cols - set(df.columns))
                raise ValueError(f"Cached file missing required columns: {missing}")
            # Ensure that CBGs are read as strings
            df['poi_cbg'] = pd.to_numeric(df['poi_cbg'], errors='coerce')
            df.dropna(subset=['poi_cbg'], inplace=True)
            df['poi_cbg'] = df['poi_cbg'].astype('int64').astype('string')
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError):
            self.logger.info(f"File {full_filename} not found. Processing raw data from {patterns_csv}")
            allowed_state_fips = {
                STATE_ABBR_TO_FIPS[state]
                for state in self.config.states
                if state in STATE_ABBR_TO_FIPS
            }
            datalist = []
            with pd.read_csv(patterns_csv, chunksize=10000) as reader:
                for chunk in reader:
                    chunk = chunk.copy()
                    chunk.columns = [str(c).strip().lower() for c in chunk.columns]
                    if 'poi_cbg' in chunk.columns:
                        chunk['poi_cbg'] = pd.to_numeric(chunk['poi_cbg'], errors='coerce')
                        chunk.dropna(subset=['poi_cbg'], inplace=True)
                        chunk['poi_cbg'] = chunk['poi_cbg'].astype('int64').astype('string')

                    # Prefer CBG state-prefix filtering (robust to ZIP DB gaps).
                    if allowed_state_fips and 'poi_cbg' in chunk.columns:
                        chunk = chunk[chunk['poi_cbg'].str.slice(0, 2).isin(allowed_state_fips)]
                    # Fallback for fixtures/special files.
                    elif 'postal_code' in chunk.columns and zip_codes:
                        chunk['postal_code'] = pd.to_numeric(chunk['postal_code'], errors='coerce')
                        datalist.append(chunk[chunk['postal_code'].isin(zip_codes)])
                        continue

                    datalist.append(chunk)
            if not datalist:
                raise ValueError(f"No data rows were loaded from {patterns_csv}")

            df = pd.concat(datalist, axis=0, ignore_index=True)
            df.columns = [str(c).strip().lower() for c in df.columns]
            required_cols = {'poi_cbg', 'visitor_daytime_cbgs'}
            if not required_cols.issubset(df.columns):
                missing = sorted(required_cols - set(df.columns))
                available = ', '.join(sorted(df.columns))
                raise ValueError(
                    f"Patterns file is missing required columns: {missing}. "
                    f"Available columns: {available}"
                )

            # Convert poi_cbg to proper strings
            df['poi_cbg'] = pd.to_numeric(df['poi_cbg'], errors='coerce')
            df.dropna(subset=['poi_cbg'], inplace=True)
            # Now force truncation
            df['poi_cbg'] = df['poi_cbg'].astype('int64').astype('string')

            df.to_csv(full_filename, index=False)
            self.logger.info(f"Saved processed data to {full_filename}")
        return df

    def load_shapefiles(self):
        """
        Load and merge shapefiles for the specified states.
        """
        self.logger.info("Loading shapefiles")
        gds = []

        for state in self.config.states:
            state = str(state)
            state_fips = STATE_ABBR_TO_FIPS.get(state)
            if not state_fips:
                self.logger.warning(f"Could not resolve state FIPS for {state}; skipping shapefile load.")
                continue

            shapefile_2016_path = (
                f"./data/shapefiles_2016/tl_2016_{state_fips}_bg/"
                f"tl_2016_{state_fips}_bg.shp"
            )
            if not os.path.exists(shapefile_2016_path):
                self.logger.warning(
                    "Missing 2016 TIGER/Line shapefile for state %s (FIPS: %s) at %s. "
                    "Legacy fallback is disabled.",
                    state,
                    state_fips,
                    shapefile_2016_path,
                )
                continue

            gdf_state = gpd.read_file(shapefile_2016_path)

            # Normalize column names/types from 2016 TIGER/Line shapefiles.
            if 'GEOID' not in gdf_state.columns and 'CensusBlockGroup' in gdf_state.columns:
                gdf_state['GEOID'] = gdf_state['CensusBlockGroup']
            if 'CensusBlockGroup' not in gdf_state.columns and 'GEOID' in gdf_state.columns:
                gdf_state['CensusBlockGroup'] = gdf_state['GEOID']

            if 'GEOID' in gdf_state.columns:
                gdf_state['GEOID'] = (
                    gdf_state['GEOID']
                    .astype(str)
                    .str.replace(r'\.0$', '', regex=True)
                    .str.zfill(12)
                )
            if 'CensusBlockGroup' in gdf_state.columns:
                gdf_state['CensusBlockGroup'] = (
                    gdf_state['CensusBlockGroup']
                    .astype(str)
                    .str.replace(r'\.0$', '', regex=True)
                    .str.zfill(12)
                )
            if 'State' not in gdf_state.columns:
                gdf_state['State'] = state

            # Normalize CRS so GeoPandas can concatenate state geometries safely.
            if gdf_state.crs is None:
                gdf_state = gdf_state.set_crs("EPSG:4326", allow_override=True)
            else:
                gdf_state = gdf_state.to_crs("EPSG:4326")

            self.logger.info(f"Loaded {state} geometry from 2016 TIGER/Line shapefile")
            gds.append(gdf_state)

        if not gds:
            raise FileNotFoundError(
                "No 2016 state shapefiles could be loaded. "
                "Ensure files exist under ./data/shapefiles_2016/tl_2016_<FIPS>_bg/."
            )

        gdf = gpd.GeoDataFrame(pd.concat(gds, ignore_index=True), crs="EPSG:4326")
        return gdf

    def get_population_data(self):
        """
        Load census population data for CBGs.
        """
        return pd.read_csv(self.config.paths["population_csv"], index_col='census_block_group')


# ----------------------------
# Utility Functions & Helpers
# ----------------------------
def get_neighboring_states(states):
    try:
        neighbors = []
        with open(r'data/neighbor-states.json', 'r') as f:

            neighborlist = json.load(f)

            for state in states:
                for n in neighborlist:
                    if n['code'] == state:
                        neighbors.extend(n['Neighborcodes'])
                        neighbors = list(set(neighbors))
                        break
        return neighbors
    except:
        return []


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


def build_cbg_centers(gdf):
    """
    Build a lookup of CBG -> (lat, lon) using representative points.
    """
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
        try:
            cbg = str(int(float(raw_cbg)))
        except (TypeError, ValueError):
            cbg = str(raw_cbg).strip()
        if not cbg:
            continue

        point = reps.loc[idx]
        centers[cbg] = (float(point.y), float(point.x))

    return centers

# For caching population data once loaded.
_population_cache = None
def cbg_population(cbg, config: Config, logger: logging.Logger):
    global _population_cache
    if _population_cache is None:
        try:
            _population_cache = pd.read_csv(config.paths["population_csv"], index_col='census_block_group', usecols=['census_block_group', 'B01003e1'])
        except Exception as e:
            logger.error(f"Error loading population data: {e}")
            return 0
    try:
        cbg_int = int(float(cbg))
        if cbg_int in _population_cache.index:
            return int(_population_cache.loc[cbg_int].B01003e1)
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
        missing_cbgs = []
        for cbg in cluster_cbgs:
            if cbg not in G:
                missing_cbgs.append(cbg)
                continue

            movement_in += float(G.nodes[cbg].get('self_weight', 0))

            for neighbor in G.adj[cbg]:
                if neighbor in cluster_cbgs:
                    movement_in += G.adj[cbg][neighbor]['weight'] / 2
                else:
                    movement_out += G.adj[cbg][neighbor]['weight']

        if missing_cbgs:
            missing_sorted = sorted(set(missing_cbgs))
            raise ValueError(
                f"CBGs not present in mobility graph: {', '.join(missing_sorted)}"
            )

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

        for _, row in df.iterrows():
            if not isinstance(row['visitor_daytime_cbgs'], str):
                continue

            try:
                dst_cbg = str(int(float(row['poi_cbg'])))
            except (TypeError, ValueError):
                continue

            visitor_dict = json.loads(row['visitor_daytime_cbgs'])
            for visitor_cbg, count in visitor_dict.items():
                try:
                    src_cbg = str(int(float(visitor_cbg)))
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


# ----------------------------
# Clustering Algorithms Module
# ----------------------------
class Clustering:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    @staticmethod
    def _record_trace_step(
        trace_collector,
        iteration,
        cluster_before,
        population_before,
        candidates,
        selected_cbg,
        selected_population,
        cluster_after,
        population_after,
        metrics_after=None
    ):
        if trace_collector is None:
            return

        sorted_candidates = sorted(
            candidates,
            key=lambda item: (float(item.get('score', 0.0)), item.get('cbg', '')),
            reverse=True
        )
        for idx, candidate in enumerate(sorted_candidates):
            candidate['rank'] = idx + 1
            candidate['selected'] = (candidate.get('cbg') == selected_cbg)

        step = {
            'iteration': int(iteration),
            'cluster_before': list(cluster_before),
            'population_before': int(population_before),
            'candidates': sorted_candidates,
            'selected_cbg': selected_cbg,
            'selected_population': int(selected_population),
            'cluster_after': list(cluster_after),
            'population_after': int(population_after),
        }
        if metrics_after:
            step['metrics_after'] = metrics_after
        trace_collector.append(step)

    @staticmethod
    def _movement_outside_cluster(G: nx.Graph, candidate, cluster_set):
        movement_out = 0.0
        if candidate not in G:
            return movement_out
        for neighbor in G.adj[candidate]:
            if neighbor in cluster_set:
                continue
            movement_out += float(G.adj[candidate][neighbor].get('weight', 0))
        return movement_out

    @staticmethod
    def _seed_distance_km(seed_cbg, candidate_cbg, cbg_centers):
        if not cbg_centers:
            return None

        seed_center = cbg_centers.get(seed_cbg)
        cand_center = cbg_centers.get(candidate_cbg)
        if seed_center is None or cand_center is None:
            return None

        try:
            return float(distance(
                seed_center[0],
                seed_center[1],
                cand_center[0],
                cand_center[1],
            ))
        except Exception:
            return None

    @classmethod
    def _seed_guard_membership(cls, seed_cbg, candidate_cbg, cbg_centers, seed_guard_distance_km):
        if candidate_cbg == seed_cbg:
            return True, 0.0

        seed_distance = cls._seed_distance_km(seed_cbg, candidate_cbg, cbg_centers)
        try:
            threshold_km = float(seed_guard_distance_km)
        except (TypeError, ValueError):
            threshold_km = 0.0

        if threshold_km <= 0 or seed_distance is None:
            return True, seed_distance
        return seed_distance <= threshold_km, seed_distance

    @classmethod
    def _seed_guard_contributor_sets(cls, seed_cbg, cluster_cbgs, cbg_centers, seed_guard_distance_km):
        contributor_set = set()
        excluded_set = set()

        for cbg in cluster_cbgs:
            contributes, _ = cls._seed_guard_membership(
                seed_cbg,
                cbg,
                cbg_centers,
                seed_guard_distance_km,
            )
            if contributes:
                contributor_set.add(cbg)
            else:
                excluded_set.add(cbg)

        if seed_cbg in cluster_cbgs:
            contributor_set.add(seed_cbg)
            excluded_set.discard(seed_cbg)

        return contributor_set, excluded_set

    def greedy_fast(self, G: nx.Graph, u0: str, min_pop: int, trace_collector=None):
        self.logger.info(f"Starting greedy_fast algorithm with seed CBG {u0}")
        population = cbg_population(u0, self.config, self.logger)

        cluster = [u0]
        cluster_set = {u0}
        surround = list(set([j for j in list(G.adj[u0]) if j not in cluster_set]))

        # Find all surrounding cbgs
        for j in list(G.adj[u0]):
            if j not in surround and j not in cluster_set:
                surround.append(j)

        if len(surround) == 0:
            self.logger.warning(f"No adjacent CBGs found. Cannot reach target population.")
            return cluster, population

        itr = 0

        while population < min_pop:
            max_weight = 0
            best_cbg = surround[0]
            best_pop = 0
            candidate_details = []

            for candidate in surround:
                if candidate in cluster_set:
                    continue

                cur_pop = cbg_population(candidate, self.config, self.logger)
                if cur_pop == 0:
                    continue

                weight = sum([G.get_edge_data(candidate, cbg, {}).get('weight', 0) for cbg in cluster])
                weight = float(weight)
                movement_outside = self._movement_outside_cluster(G, candidate, cluster_set)
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(cur_pop),
                    'score': weight,
                    'movement_to_cluster': weight,
                    'movement_to_outside': movement_outside,
                })

                if weight > max_weight:
                    max_weight = weight
                    best_cbg = candidate
                    best_pop = cur_pop

            prev_cluster = list(cluster)
            prev_population = population

            # Add best CBG to cluster
            surround.remove(best_cbg)
            cluster.append(best_cbg)
            cluster_set.add(best_cbg)

            # Update population
            population += best_pop

            self._record_trace_step(
                trace_collector,
                iteration=itr,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=best_cbg,
                selected_population=best_pop,
                cluster_after=cluster,
                population_after=population
            )

            self.logger.info(f"Iteration {itr}: Added CBG {best_cbg} with pop {best_pop}. New total: {population}")

            # Add the new CBG's neighbors
            surround.extend([j for j in list(G.adj[best_cbg]) if j not in cluster_set])
            surround = list(set(surround))

            itr += 1
            if itr > 500:
                self.logger.warning(f"Max iterations exceeded (500). Cannot reach target population.")
                break

        return cluster, population


    def greedy_weight(self, G, u0, min_pop, trace_collector=None):
        """
        Greedy algorithm to build a cluster of CBGs based on movement weights.
        """
        self.logger.info(f"Starting greedy_weight algorithm with seed CBG {u0}")
        cluster = [u0]
        cluster_set = {u0}
        population = cbg_population(u0, self.config, self.logger)
        it = 1
        self.logger.info(f"Seed CBG population: {population}")
        while population < min_pop:
            all_adj_cbgs = []
            for i in cluster:
                try:
                    for j in list(G.adj[i]):
                        if j not in all_adj_cbgs and j not in cluster_set:
                            all_adj_cbgs.append(j)
                except KeyError:
                    self.logger.warning(f"CBG {i} not found in graph")
                    continue
            if not all_adj_cbgs:
                self.logger.warning(f"No adjacent CBGs found after {it} iterations. Cannot reach target population.")
                break
            max_movement = 0
            cbg_to_add = all_adj_cbgs[0]
            candidate_details = []
            for candidate in all_adj_cbgs:
                current_movement = 0
                for member in cluster:
                    try:
                        current_movement += G.adj[candidate][member]['weight']
                    except (KeyError, ZeroDivisionError):
                        continue
                movement_to_cluster = float(current_movement)
                movement_to_outside = self._movement_outside_cluster(G, candidate, cluster_set)
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(cbg_population(candidate, self.config, self.logger)),
                    'score': movement_to_cluster,
                    'movement_to_cluster': movement_to_cluster,
                    'movement_to_outside': movement_to_outside,
                })
                if current_movement > max_movement:
                    max_movement = current_movement
                    cbg_to_add = candidate

            prev_cluster = list(cluster)
            prev_population = population
            cluster.append(cbg_to_add)
            cluster_set.add(cbg_to_add)
            cbg_pop = cbg_population(cbg_to_add, self.config, self.logger)
            population += cbg_pop

            self._record_trace_step(
                trace_collector,
                iteration=it - 1,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=cbg_to_add,
                selected_population=cbg_pop,
                cluster_after=cluster,
                population_after=population
            )

            self.logger.info(f"Iteration {it}: Added CBG {cbg_to_add} with pop {cbg_pop}. New total: {population}")
            it += 1
            if it > 1000:
                self.logger.warning("Reached maximum iterations (1000). Stopping algorithm.")
                break
        return cluster, population

    def greedy_weight_seed_guard(
        self,
        G,
        u0,
        min_pop,
        seed_guard_distance_km=20.0,
        cbg_centers=None,
        trace_collector=None,
    ):
        """
        Greedy weight variant that excludes seed-distant CBGs from future movement scoring.

        The frontier still includes neighbors of every selected CBG so the zone can continue
        expanding, but movement scores only accumulate through the subset of cluster members
        that remain within `seed_guard_distance_km` of the seed.
        """
        self.logger.info(
            "Starting greedy_weight_seed_guard algorithm with seed CBG %s (distance %.2f km)",
            u0,
            float(seed_guard_distance_km),
        )
        cluster = [u0]
        cluster_set = {u0}
        contributor_set = {u0}
        excluded_set = set()
        population = cbg_population(u0, self.config, self.logger)
        cbg_centers = cbg_centers or {}
        it = 1

        self.logger.info(f"Seed CBG population: {population}")
        while population < min_pop:
            all_adj_cbgs = []
            for i in cluster:
                try:
                    for j in list(G.adj[i]):
                        if j not in all_adj_cbgs and j not in cluster_set:
                            all_adj_cbgs.append(j)
                except KeyError:
                    self.logger.warning(f"CBG {i} not found in graph")
                    continue

            if not all_adj_cbgs:
                self.logger.warning(
                    f"No adjacent CBGs found after {it} iterations. Cannot reach target population."
                )
                break

            best_choice = None
            candidate_details = []
            for candidate in all_adj_cbgs:
                movement_to_cluster = 0.0
                movement_to_full_cluster = 0.0
                for member in cluster:
                    try:
                        edge_weight = float(G.adj[candidate][member]['weight'])
                    except (KeyError, ZeroDivisionError):
                        continue
                    movement_to_full_cluster += edge_weight
                    if member in contributor_set:
                        movement_to_cluster += edge_weight

                movement_to_outside = self._movement_outside_cluster(G, candidate, cluster_set)
                candidate_pop = int(cbg_population(candidate, self.config, self.logger))
                contributes_after, seed_distance = self._seed_guard_membership(
                    u0,
                    candidate,
                    cbg_centers,
                    seed_guard_distance_km,
                )
                distance_tiebreak = 0.0 if seed_distance is None else -float(seed_distance)
                candidate_tuple = (
                    float(movement_to_cluster),
                    int(contributes_after),
                    distance_tiebreak,
                    float(candidate_pop),
                    str(candidate),
                )
                candidate_details.append({
                    'cbg': candidate,
                    'population': candidate_pop,
                    'score': float(movement_to_cluster),
                    'movement_to_cluster': float(movement_to_cluster),
                    'movement_to_full_cluster': float(movement_to_full_cluster),
                    'movement_to_outside': float(movement_to_outside),
                    'seed_distance_km': (
                        float(seed_distance) if seed_distance is not None else None
                    ),
                    'movement_contributes_after_selection': bool(contributes_after),
                })
                if best_choice is None or candidate_tuple > best_choice[0]:
                    best_choice = (
                        candidate_tuple,
                        candidate,
                        candidate_pop,
                        contributes_after,
                        seed_distance,
                    )

            if best_choice is None:
                self.logger.warning(
                    f"No valid candidate CBGs found after {it} iterations. Cannot reach target population."
                )
                break

            _, cbg_to_add, cbg_pop, contributes_after, seed_distance = best_choice
            prev_cluster = list(cluster)
            prev_population = population
            cluster.append(cbg_to_add)
            cluster_set.add(cbg_to_add)
            population += cbg_pop

            if contributes_after:
                contributor_set.add(cbg_to_add)
            else:
                excluded_set.add(cbg_to_add)

            metrics_after = {
                'seed_guard_distance_km': float(seed_guard_distance_km),
                'movement_contributor_count': len(contributor_set),
                'movement_excluded_count': len(excluded_set),
            }
            if excluded_set:
                metrics_after['movement_excluded_cbgs'] = list(excluded_set)

            self._record_trace_step(
                trace_collector,
                iteration=it - 1,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=cbg_to_add,
                selected_population=cbg_pop,
                cluster_after=cluster,
                population_after=population,
                metrics_after=metrics_after,
            )

            movement_note = "counts toward future scoring"
            if not contributes_after:
                movement_note = "excluded from future scoring"
            self.logger.info(
                "Iteration %d: Added CBG %s with pop %d. New total: %d (%s, distance=%s km)",
                it,
                cbg_to_add,
                cbg_pop,
                population,
                movement_note,
                (
                    f"{float(seed_distance):.2f}"
                    if seed_distance is not None
                    else "unknown"
                ),
            )
            it += 1
            if it > 1000:
                self.logger.warning("Reached maximum iterations (1000). Stopping algorithm.")
                break
        return cluster, population

    def greedy_ratio(self, G, u0, min_pop, trace_collector=None):
        """
        Greedy algorithm to build a cluster of CBGs based on movement ratio.
        """
        self.logger.info(f"Starting greedy_ratio algorithm with seed CBG {u0}")
        cluster = [u0]
        cluster_set = {u0}
        population = cbg_population(u0, self.config, self.logger)
        it = 1
        self.logger.info(f"Seed CBG population: {population}")
        while population < min_pop:
            all_adj_cbgs = []
            for i in cluster:
                try:
                    for j in list(G.adj[i]):
                        if j not in all_adj_cbgs and j not in cluster_set:
                            all_adj_cbgs.append(j)
                except KeyError:
                    self.logger.warning(f"CBG {i} not found in graph")
                    continue
            if not all_adj_cbgs:
                self.logger.warning(f"No adjacent CBGs found after {it} iterations. Cannot reach target population.")
                break
            max_ratio = 0
            cbg_to_add = all_adj_cbgs[0]
            candidate_details = []
            for candidate in all_adj_cbgs:
                movement_in = 0
                movement_out = 0
                for j in G.adj[candidate]:
                    if j in cluster:
                        movement_in += G.adj[candidate][j]['weight']
                    else:
                        movement_out += G.adj[candidate][j]['weight']
                total_movement = movement_in + movement_out
                ratio = 0.0
                if total_movement > 0:
                    ratio = movement_in / total_movement
                    if ratio > max_ratio:
                        max_ratio = ratio
                        cbg_to_add = candidate
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(cbg_population(candidate, self.config, self.logger)),
                    'score': float(ratio),
                    'movement_to_cluster': float(movement_in),
                    'movement_to_outside': float(movement_out),
                    'movement_total': float(total_movement),
                })

            prev_cluster = list(cluster)
            prev_population = population
            cluster.append(cbg_to_add)
            cluster_set.add(cbg_to_add)
            cbg_pop = cbg_population(cbg_to_add, self.config, self.logger)
            population += cbg_pop

            self._record_trace_step(
                trace_collector,
                iteration=it - 1,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=cbg_to_add,
                selected_population=cbg_pop,
                cluster_after=cluster,
                population_after=population
            )

            self.logger.info(f"Iteration {it}: Added CBG {cbg_to_add} with pop {cbg_pop}. New total: {population}")
            it += 1
            if it > 1000:
                self.logger.warning("Reached maximum iterations (1000). Stopping algorithm.")
                break
        return cluster, population

    def _select_candidate_nodes(self, G: nx.Graph, u0: str, candidate_limit: int):
        """
        Build a connected candidate set around the seed for MILP optimization.
        """
        if u0 not in G:
            return [u0]

        try:
            limit = int(candidate_limit)
        except (TypeError, ValueError):
            limit = 120
        limit = max(10, limit)

        selected = [u0]
        selected_set = {u0}
        frontier = set(G.adj[u0])

        while frontier and len(selected) < limit:
            best_node = None
            best_score = None

            for node in list(frontier):
                if node in selected_set:
                    continue

                flow_to_selected = 0.0
                for nb in G.adj[node]:
                    if nb in selected_set:
                        flow_to_selected += float(G.adj[node][nb].get('weight', 0))

                self_w = float(G.nodes[node].get('self_weight', 0))
                deg = float(G.degree[node])
                score = (flow_to_selected, self_w, deg, node)

                if best_score is None or score > best_score:
                    best_score = score
                    best_node = node

            if best_node is None:
                break

            frontier.discard(best_node)
            if best_node in selected_set:
                continue

            selected.append(best_node)
            selected_set.add(best_node)
            for nb in G.adj[best_node]:
                if nb not in selected_set:
                    frontier.add(nb)

        return selected

    def _optimize_czi_subproblem(
        self,
        lam: float,
        nodes,
        edges,
        edge_weights,
        directed_arcs,
        node_to_idx,
        populations,
        self_weights,
        external_boundary,
        seed_idx,
        pop_floor: int,
        pop_cap: int,
        mip_rel_gap: float,
        time_limit_sec: float
    ):
        n = len(nodes)
        m = len(edges)
        d = len(directed_arcs)
        if n == 0:
            return None

        # Variable layout:
        # [x(0..n-1), y(0..m-1), z(0..m-1), f(0..d-1)]
        x_start = 0
        y_start = n
        z_start = n + m
        f_start = n + (2 * m)
        num_vars = n + (2 * m) + d
        M = max(1, n - 1)

        c = np.zeros(num_vars, dtype=float)
        for node, idx in node_to_idx.items():
            coeff = ((1.0 - lam) * float(self_weights.get(node, 0.0))) - (
                lam * float(external_boundary.get(node, 0.0))
            )
            c[x_start + idx] = -coeff

        for e_idx, w in enumerate(edge_weights):
            c[y_start + e_idx] = -((1.0 - lam) * float(w))
            c[z_start + e_idx] = lam * float(w)

        rows = []
        cols = []
        vals = []
        lb = []
        ub = []
        row_idx = 0

        def add_row(coeffs, row_lb=-np.inf, row_ub=np.inf):
            nonlocal row_idx
            for col, val in coeffs.items():
                if val == 0:
                    continue
                rows.append(row_idx)
                cols.append(col)
                vals.append(float(val))
            lb.append(float(row_lb))
            ub.append(float(row_ub))
            row_idx += 1

        # y_e linearization: y = x_u AND x_v
        for e_idx, (u, v) in enumerate(edges):
            xu = x_start + node_to_idx[u]
            xv = x_start + node_to_idx[v]
            ye = y_start + e_idx
            add_row({ye: 1, xu: -1}, row_ub=0)                 # y <= x_u
            add_row({ye: 1, xv: -1}, row_ub=0)                 # y <= x_v
            add_row({ye: -1, xu: 1, xv: 1}, row_ub=1)          # y >= x_u + x_v - 1

        # z_e linearization: z = |x_u - x_v|
        for e_idx, (u, v) in enumerate(edges):
            xu = x_start + node_to_idx[u]
            xv = x_start + node_to_idx[v]
            ze = z_start + e_idx
            add_row({ze: 1, xu: -1, xv: 1}, row_lb=0)          # z >= x_u - x_v
            add_row({ze: 1, xu: 1, xv: -1}, row_lb=0)          # z >= x_v - x_u
            add_row({ze: 1, xu: -1, xv: -1}, row_ub=0)         # z <= x_u + x_v
            add_row({ze: 1, xu: 1, xv: 1}, row_ub=2)           # z <= 2 - x_u - x_v

        # Population band: floor <= sum(pop_i x_i) <= cap
        pop_coeff = {}
        for node, idx in node_to_idx.items():
            pop_coeff[x_start + idx] = float(populations.get(node, 0))
        add_row(pop_coeff, row_lb=float(pop_floor), row_ub=float(pop_cap))

        # Force seed inclusion.
        add_row({x_start + seed_idx: 1}, row_lb=1, row_ub=1)

        # Flow capacities and adjacency index maps for connectivity.
        out_arcs = {node: [] for node in nodes}
        in_arcs = {node: [] for node in nodes}
        for a_idx, (u, v) in enumerate(directed_arcs):
            fa = f_start + a_idx
            xu = x_start + node_to_idx[u]
            xv = x_start + node_to_idx[v]
            add_row({fa: 1, xu: -M}, row_ub=0)                 # f_uv <= M * x_u
            add_row({fa: 1, xv: -M}, row_ub=0)                 # f_uv <= M * x_v
            out_arcs[u].append(a_idx)
            in_arcs[v].append(a_idx)

        seed_node = nodes[seed_idx]

        # Conservation for non-seed nodes: inflow - outflow = x_i
        for node in nodes:
            if node == seed_node:
                continue

            coeff = {x_start + node_to_idx[node]: -1}
            for a_idx in in_arcs[node]:
                coeff[f_start + a_idx] = coeff.get(f_start + a_idx, 0) + 1
            for a_idx in out_arcs[node]:
                coeff[f_start + a_idx] = coeff.get(f_start + a_idx, 0) - 1
            add_row(coeff, row_lb=0, row_ub=0)

        # Seed conservation: outflow - inflow = sum_{i != seed} x_i
        seed_coeff = {}
        for a_idx in out_arcs[seed_node]:
            seed_coeff[f_start + a_idx] = seed_coeff.get(f_start + a_idx, 0) + 1
        for a_idx in in_arcs[seed_node]:
            seed_coeff[f_start + a_idx] = seed_coeff.get(f_start + a_idx, 0) - 1
        for node in nodes:
            if node == seed_node:
                continue
            seed_coeff[x_start + node_to_idx[node]] = -1
        add_row(seed_coeff, row_lb=0, row_ub=0)

        A = coo_matrix((vals, (rows, cols)), shape=(row_idx, num_vars)).tocsr()
        constraints = LinearConstraint(A, np.array(lb), np.array(ub))

        lower = np.zeros(num_vars, dtype=float)
        upper = np.concatenate([
            np.ones(n + (2 * m), dtype=float),
            np.full(d, float(M), dtype=float),
        ])
        bounds = Bounds(lower, upper)

        integrality = np.zeros(num_vars, dtype=int)
        integrality[: n + (2 * m)] = 1

        options = {
            'disp': False,
            'mip_rel_gap': float(max(0.0, mip_rel_gap)),
            'time_limit': float(max(1.0, time_limit_sec)),
        }

        try:
            return milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options=options,
            )
        except Exception:
            self.logger.exception("MILP subproblem failed unexpectedly")
            return None

    def czi_optimal_cap(
        self,
        G: nx.Graph,
        u0: str,
        max_pop: int,
        candidate_limit: int = 120,
        population_floor_ratio: float = 0.9,
        mip_rel_gap: float = 0.02,
        time_limit_sec: float = 20.0,
        max_dinkelbach_iters: int = 8
    ):
        """
        Approximate rigorous optimization:
        maximize CZI = inside / (inside + boundary)
        subject to:
          - seed included
          - connected cluster
          - population_floor <= population <= max_pop

        Notes:
          - Uses Dinkelbach iterations over MILP subproblems.
          - Operates on a connected candidate neighborhood around the seed.
          - Falls back to greedy_czi_balanced if MILP is unavailable or fails.
        """
        if u0 not in G:
            pop = cbg_population(u0, self.config, self.logger)
            self.logger.warning(f"Seed CBG {u0} not found in graph for czi_optimal_cap")
            return [u0], pop

        if not SCIPY_MILP_AVAILABLE:
            self.logger.warning(
                "SciPy MILP is unavailable; falling back to greedy_czi_balanced."
            )
            return self.greedy_czi_balanced(G, u0, max_pop)

        try:
            pop_cap = int(max_pop)
        except (TypeError, ValueError):
            pop_cap = 0
        seed_pop = cbg_population(u0, self.config, self.logger)
        pop_cap = max(pop_cap, seed_pop)

        floor_ratio = float(population_floor_ratio)
        floor_ratio = min(1.0, max(0.0, floor_ratio))
        pop_floor = max(seed_pop, int(math.floor(pop_cap * floor_ratio)))

        candidates = self._select_candidate_nodes(G, u0, candidate_limit)
        node_set = set(candidates)
        if u0 not in node_set:
            candidates = [u0] + candidates
            node_set = set(candidates)

        populations = {}
        self_weights = {}
        external_boundary = {}
        for node in candidates:
            populations[node] = max(0, int(cbg_population(node, self.config, self.logger)))
            self_weights[node] = float(G.nodes[node].get('self_weight', 0.0))

            boundary_w = 0.0
            for nb in G.adj[node]:
                w = float(G.adj[node][nb].get('weight', 0))
                if w <= 0:
                    continue
                if nb not in node_set:
                    boundary_w += w
            external_boundary[node] = boundary_w

        edges = []
        edge_weights = []
        seen_edges = set()
        for u in candidates:
            for v in G.adj[u]:
                if v not in node_set:
                    continue
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                w = float(G.adj[u][v].get('weight', 0))
                if w <= 0:
                    continue
                edges.append(key)
                edge_weights.append(w)

        directed_arcs = []
        for u, v in edges:
            directed_arcs.append((u, v))
            directed_arcs.append((v, u))

        node_to_idx = {node: i for i, node in enumerate(candidates)}
        seed_idx = node_to_idx[u0]

        total_candidate_pop = sum(populations.values())
        pop_floor = min(pop_floor, total_candidate_pop)
        pop_cap = min(pop_cap, total_candidate_pop)
        if pop_floor > pop_cap:
            pop_floor = pop_cap

        if pop_cap <= 0:
            self.logger.warning(
                "Candidate graph has zero population for czi_optimal_cap; falling back to greedy."
            )
            return self.greedy_czi_balanced(G, u0, max_pop)

        self.logger.info(
            "Starting czi_optimal_cap with %d candidates, cap=%d, floor=%d",
            len(candidates), pop_cap, pop_floor
        )

        best_cluster = None
        best_score = -1.0
        lam = 0.5
        dinkelbach_iters = max(1, int(max_dinkelbach_iters))

        floor_attempts = [pop_floor]
        while floor_attempts[-1] > seed_pop:
            next_floor = max(seed_pop, int(math.floor(floor_attempts[-1] * 0.9)))
            if next_floor == floor_attempts[-1]:
                break
            floor_attempts.append(next_floor)
        if seed_pop not in floor_attempts:
            floor_attempts.append(seed_pop)

        for floor_idx, attempt_floor in enumerate(floor_attempts):
            attempt_found = False
            current_lam = lam

            if floor_idx > 0:
                self.logger.info(
                    "Relaxing population floor to %d for czi_optimal_cap",
                    attempt_floor
                )

            for it in range(dinkelbach_iters):
                result = self._optimize_czi_subproblem(
                    lam=current_lam,
                    nodes=candidates,
                    edges=edges,
                    edge_weights=edge_weights,
                    directed_arcs=directed_arcs,
                    node_to_idx=node_to_idx,
                    populations=populations,
                    self_weights=self_weights,
                    external_boundary=external_boundary,
                    seed_idx=seed_idx,
                    pop_floor=attempt_floor,
                    pop_cap=pop_cap,
                    mip_rel_gap=mip_rel_gap,
                    time_limit_sec=time_limit_sec,
                )

                if result is None or result.x is None:
                    self.logger.warning(
                        "MILP solve returned no solution at Dinkelbach iteration %d (floor=%d)",
                        it,
                        attempt_floor
                    )
                    break

                x = result.x[: len(candidates)]
                chosen = [node for node, idx in node_to_idx.items() if x[idx] >= 0.5]
                if u0 not in chosen:
                    chosen.append(u0)

                # Ensure stable order with seed first and then candidate order.
                chosen_set = set(chosen)
                chosen = [node for node in candidates if node in chosen_set]
                if not chosen:
                    break

                try:
                    stats = Helpers.calculate_movement_stats(G, chosen)
                except ValueError:
                    self.logger.warning(
                        "Candidate solution had nodes missing from graph; falling back."
                    )
                    break

                inside = float(stats.get('in', 0.0))
                boundary = float(stats.get('out', 0.0))
                denom = inside + boundary
                czi = (inside / denom) if denom > 0 else 0.0
                residual = inside - current_lam * denom

                chosen_pop = sum(populations.get(node, 0) for node in chosen)
                if czi > best_score or (
                    abs(czi - best_score) < 1e-9 and best_cluster is not None and chosen_pop > sum(populations.get(node, 0) for node in best_cluster)
                ):
                    best_score = czi
                    best_cluster = chosen

                self.logger.info(
                    "Dinkelbach iter %d (floor=%d): CZI=%.5f, inside=%.1f, boundary=%.1f, pop=%d, status=%s",
                    it, attempt_floor, czi, inside, boundary, chosen_pop, str(result.status)
                )

                attempt_found = True
                next_lam = czi
                if abs(next_lam - current_lam) <= 1e-4 or abs(residual) <= 1e-3:
                    current_lam = next_lam
                    break
                current_lam = next_lam

            if attempt_found:
                lam = current_lam
                break

        if not best_cluster:
            self.logger.warning(
                "czi_optimal_cap could not produce a valid MILP cluster; using greedy_czi_balanced fallback."
            )
            return self.greedy_czi_balanced(G, u0, max_pop)

        best_population = sum(cbg_population(node, self.config, self.logger) for node in best_cluster)
        self.logger.info(
            "czi_optimal_cap selected %d CBGs, population=%d, CZI=%.5f",
            len(best_cluster), best_population, best_score
        )
        return best_cluster, best_population

    def greedy_czi_balanced(self, G: nx.Graph, u0: str, min_pop: int,
                            alpha: float = 0.75, overshoot_penalty: float = 0.25,
                            distance_penalty_weight: float = 0.02,
                            distance_scale_km: float = 20.0,
                            cbg_centers=None,
                            trace_collector=None):
        """
        Greedy clustering with CZI-aware utility.

        Utility per candidate:
          CZI_after
          - distance_penalty_weight * normalized_seed_distance

        Notes:
          `alpha` and `overshoot_penalty` are retained in the signature for backward
          compatibility, but no longer affect ranking.
        """
        self.logger.info(f"Starting greedy_czi_balanced algorithm with seed CBG {u0}")
        population = cbg_population(u0, self.config, self.logger)
        cluster = [u0]
        cluster_set = {u0}
        cbg_centers = cbg_centers or {}
        seed_center = cbg_centers.get(u0)

        if u0 not in G:
            self.logger.warning(f"Seed CBG {u0} not found in graph")
            return cluster, population

        base_stats = Helpers.calculate_movement_stats(G, cluster)
        movement_in = float(base_stats.get('in', 0))
        movement_out = float(base_stats.get('out', 0))
        surround = set(G.adj[u0])

        itr = 0
        while population < min_pop:
            if not surround:
                self.logger.warning(
                    f"No adjacent CBGs found after {itr} iterations. Cannot reach target population."
                )
                break

            best = None
            candidate_details = []

            for candidate in list(surround):
                if candidate in cluster_set or candidate not in G:
                    continue

                cand_pop = cbg_population(candidate, self.config, self.logger)
                if cand_pop <= 0:
                    continue

                self_weight = float(G.nodes[candidate].get('self_weight', 0))
                in_to_cluster = 0.0
                out_to_outside = 0.0
                for neighbor in G.adj[candidate]:
                    weight = float(G.adj[candidate][neighbor].get('weight', 0))
                    if weight <= 0:
                        continue
                    if neighbor in cluster_set:
                        in_to_cluster += weight
                    else:
                        out_to_outside += weight

                inside_after = movement_in + self_weight + in_to_cluster
                boundary_after = movement_out - in_to_cluster + out_to_outside
                if boundary_after < 0 and abs(boundary_after) < 1e-9:
                    boundary_after = 0.0
                total_after = inside_after + boundary_after
                czi_after = (inside_after / total_after) if total_after > 0 else 0.0

                distance_penalty = 0.0
                if seed_center and distance_scale_km > 0:
                    cand_center = cbg_centers.get(candidate)
                    if cand_center:
                        dist_km = distance(
                            seed_center[0], seed_center[1],
                            cand_center[0], cand_center[1]
                        )
                        distance_penalty = dist_km / (dist_km + distance_scale_km)

                score = (
                    czi_after
                    - distance_penalty_weight * distance_penalty
                )

                candidate_tuple = (
                    score,
                    czi_after,
                    -distance_penalty,
                    in_to_cluster,
                    candidate,
                    cand_pop,
                    inside_after,
                    boundary_after
                )
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(cand_pop),
                    'score': float(score),
                    'movement_to_cluster': float(in_to_cluster),
                    'movement_to_outside': float(out_to_outside),
                    'czi_after': float(czi_after),
                    'distance_penalty': float(distance_penalty),
                    'movement_inside_after': float(inside_after),
                    'movement_boundary_after': float(boundary_after),
                })
                if best is None or candidate_tuple[:4] > best[:4]:
                    best = candidate_tuple

            if best is None:
                self.logger.warning(
                    f"No valid candidate CBGs found after {itr} iterations. Cannot reach target population."
                )
                break

            _, czi_after, _, _, best_cbg, best_pop, inside_after, boundary_after = best
            prev_cluster = list(cluster)
            prev_population = population
            cluster.append(best_cbg)
            cluster_set.add(best_cbg)
            population += best_pop
            movement_in = inside_after
            movement_out = boundary_after

            self._record_trace_step(
                trace_collector,
                iteration=itr,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=best_cbg,
                selected_population=best_pop,
                cluster_after=cluster,
                population_after=population,
                metrics_after={
                    'czi': float(czi_after),
                    'movement_inside': float(movement_in),
                    'movement_boundary': float(movement_out),
                }
            )

            surround.remove(best_cbg)
            surround.update([j for j in G.adj[best_cbg] if j not in cluster_set])

            self.logger.info(
                f"Iteration {itr}: Added CBG {best_cbg} with pop {best_pop}. "
                f"New total: {population}. CZI: {czi_after:.4f}"
            )

            itr += 1
            if itr > 500:
                self.logger.warning("Max iterations exceeded (500). Cannot reach target population.")
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
    def cbg_geocode(cbg_id, gdf=None):
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
        try:
            point = gdf[gdf['CensusBlockGroup'] == str(int(float(cbg_id)))].representative_point()
            center = point.iloc[0]

            return {
                'latitude': center.y,
                'longitude': center.x
            }
        except:
            return { 'latitude': None, 'longitude': None }


    def generate_maps(self, G, gdf, algorithm_result):
        """
        Generate and save map visualizations.
        """
        def safe_center():
            try:
                seed = Visualizer.cbg_geocode(self.config.core_cbg, gdf)
                if seed['latitude'] is None or seed['longitude'] is None:
                    for cbg in algorithm_result[0]:
                        pos = Visualizer.cbg_geocode(cbg, gdf)
                        if pos['latitude'] is not None and pos['longitude'] is not None:
                            return [ pos['latitude'], pos['longitude'] ]

                    return self.config.map["default_location"]

                return [ seed['latitude'], seed['longitude'] ]
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
                shape = gdf[gdf['CensusBlockGroup'] == cbg]
                if shape.empty:
                    continue
                shape = shape.to_crs("EPSG:4326")
                geojson = json.loads(shape.to_json())
                feature = geojson['features'][0]
                feature['properties']['times'] = [(pd.Timestamp('today') + pd.Timedelta(i, 'D')).isoformat()]
                feature['properties']['style'] = {'fillColor': color, 'color': color, 'fillOpacity': 0.7}
                features.append(feature)

                loc = shape.representative_point().iloc[0]
                folium.Marker(location=[loc.y, loc.x], popup=f'{cbg} - Population: {cbg_population(cbg, self.config, self.logger)}').add_to(self.map_obj)
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
            shape = gdf[gdf['CensusBlockGroup'] == cbg]
            if not shape.empty:
                shape = shape.to_crs("EPSG:4326")
                folium.GeoJson(
                    json.loads(shape.to_json()),
                    style_function=lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.7}
                ).add_to(self.map_obj)

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
    seed_cbg = '240430002001'
    min_pop = 150_000

    config = Config(seed_cbg, min_pop)
    logger = setup_logging(config)
    logger.info("Starting clustering analysis")

    # Data loading
    data_loader = DataLoader(config, logger)
    zip_codes = data_loader.get_zip_codes()
    logger.info(f"Retrieved {len(zip_codes)} zip codes")
    df = data_loader.load_safegraph_data(zip_codes)
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
    visualizer.generate_maps(G, gdf, algorithm_result)

    # Save map to file
    output_map_path = os.path.join(config.output_dir, config.paths["output_html"])
    visualizer.map_obj.save(output_map_path)
    logger.info(f"Map saved to {output_map_path}")

    # Generate YAML output
    exporter = Exporter(config, logger)
    exporter.generate_yaml_output(G, algorithm_result)

    with open(r'output/algorithm_result.json', 'w') as f:
        json.dump(algorithm_result, f, indent=2)

    # request from dahbura
    with open(r'output/cbglistpop.json', 'w') as f:
        cbglistpop = {
            'meta': {
                'name': 'Hagerstown, MD',
                'seed_cbg': seed_cbg,
                'min_pop': min_pop,
                'total_pop': algorithm_result[1]
            }
        }

        for cbg in algorithm_result[0]:
            cbglistpop[cbg] = cbg_population(cbg, config, logger)
        json.dump(cbglistpop, f, indent=2)

    logger.info("Processing complete")


def generate_cz(cbg, min_pop, patterns_file=None, patterns_folder=None, month=None,
                start_date: Optional[_datetime] = None,
                shared_data: Optional['PatternsData'] = None,
                algorithm='czi_balanced', distance_penalty_weight=None,
                distance_scale_km=None, optimal_candidate_limit=None,
                optimal_population_floor_ratio=None, optimal_mip_rel_gap=None,
                optimal_time_limit_sec=None, optimal_max_iters=None,
                seed_guard_distance_km=None,
                include_trace=False):
    """
    Generate a convenience zone cluster starting from a seed CBG.

    Args:
        cbg: Seed Census Block Group GEOID
        min_pop: Minimum population for the cluster
        patterns_file: Optional specific patterns CSV file path
        patterns_folder: Optional folder containing monthly pattern files
        month: Optional month key (YYYY-MM) when using patterns_folder
        start_date: Optional simulation start date for auto-resolving monthly file
        shared_data: Optional pre-loaded PatternsData to avoid re-reading CSV

    Returns:
        Tuple of (geoids dict, map object, gdf GeoDataFrame)
        If include_trace=True, returns (geoids, map, gdf, trace_payload).
    """
    config = Config(cbg, min_pop, patterns_file=patterns_file,
                    patterns_folder=patterns_folder, month=month,
                    start_date=start_date)

    logger = setup_logging(config)
    logger.info("Starting clustering analysis")
    if config.month:
        logger.info(f"Using monthly patterns: {config.month} (file: {config.paths['patterns_csv']})")

    # Data loading
    data_loader = DataLoader(config, logger)
    zip_codes = data_loader.get_zip_codes()
    logger.info(f"Retrieved {len(zip_codes)} zip codes")
    df = data_loader.load_safegraph_data(zip_codes, shared_data=shared_data)
    gdf = data_loader.load_shapefiles()
    _ = data_loader.get_population_data()  # Population data is cached in cbg_population

    # Graph generation
    graph_builder = GraphBuilder(logger)
    G = graph_builder.gen_graph(df)
    cbg_centers = build_cbg_centers(gdf)

    if G.number_of_nodes() == 0:
        raise ValueError(
            "No mobility graph could be built from the selected patterns data. "
            "Try a different start date or location."
        )
    if config.core_cbg not in G:
        raise ValueError(
            f"Seed CBG {config.core_cbg} is not present in the mobility graph for "
            f"{config.paths['patterns_csv']}. Try a different start date or location."
        )

    # Run clustering algorithm
    clustering_algo = Clustering(config, logger)
    algorithm_key = str(algorithm or 'czi_balanced').strip().lower()
    algorithm_map = {
        'czi_balanced': clustering_algo.greedy_czi_balanced,
        'czi_optimal_cap': clustering_algo.czi_optimal_cap,
        'greedy_fast': clustering_algo.greedy_fast,
        'greedy_weight': clustering_algo.greedy_weight,
        'greedy_weight_seed_guard': clustering_algo.greedy_weight_seed_guard,
        'greedy_ratio': clustering_algo.greedy_ratio,
    }
    if algorithm_key not in algorithm_map:
        raise ValueError(
            f"Invalid clustering algorithm '{algorithm}'. "
            "Valid options: czi_balanced, czi_optimal_cap, greedy_fast, greedy_weight, "
            "greedy_weight_seed_guard, greedy_ratio"
        )

    logger.info(f"Using clustering algorithm: {algorithm_key}")
    trace_steps = [] if include_trace else None
    if algorithm_key == 'czi_balanced':
        czi_kwargs = {'cbg_centers': cbg_centers}
        if distance_penalty_weight is not None:
            czi_kwargs['distance_penalty_weight'] = float(distance_penalty_weight)
        if distance_scale_km is not None:
            czi_kwargs['distance_scale_km'] = float(distance_scale_km)
        if trace_steps is not None:
            czi_kwargs['trace_collector'] = trace_steps
        algorithm_result = clustering_algo.greedy_czi_balanced(
            G,
            config.core_cbg,
            config.min_cluster_pop,
            **czi_kwargs
        )
    elif algorithm_key == 'czi_optimal_cap':
        optimal_kwargs = {}
        if optimal_candidate_limit is not None:
            optimal_kwargs['candidate_limit'] = int(optimal_candidate_limit)
        if optimal_population_floor_ratio is not None:
            optimal_kwargs['population_floor_ratio'] = float(optimal_population_floor_ratio)
        if optimal_mip_rel_gap is not None:
            optimal_kwargs['mip_rel_gap'] = float(optimal_mip_rel_gap)
        if optimal_time_limit_sec is not None:
            optimal_kwargs['time_limit_sec'] = float(optimal_time_limit_sec)
        if optimal_max_iters is not None:
            optimal_kwargs['max_dinkelbach_iters'] = int(optimal_max_iters)
        algorithm_result = clustering_algo.czi_optimal_cap(
            G,
            config.core_cbg,
            config.min_cluster_pop,
            **optimal_kwargs
        )
    elif algorithm_key == 'greedy_weight_seed_guard':
        seed_guard_kwargs = {
            'cbg_centers': cbg_centers,
        }
        if seed_guard_distance_km is not None:
            seed_guard_kwargs['seed_guard_distance_km'] = float(seed_guard_distance_km)
        if trace_steps is not None:
            seed_guard_kwargs['trace_collector'] = trace_steps
        algorithm_result = clustering_algo.greedy_weight_seed_guard(
            G,
            config.core_cbg,
            config.min_cluster_pop,
            **seed_guard_kwargs
        )
    else:
        algorithm_result = algorithm_map[algorithm_key](
            G,
            config.core_cbg,
            config.min_cluster_pop,
            trace_collector=trace_steps
        )
    logger.info(f"Clustering complete: {len(algorithm_result[0])} CBGs, population: {algorithm_result[1]}")

    # Generate map visualizations
    visualizer = Visualizer(config, logger)
    visualizer.generate_maps(G, gdf, algorithm_result)

    # Get per-cbg population data (hh generator needs this)
    geoids = { cbg:cbg_population(cbg, config, logger) for cbg in algorithm_result[0] }

    logger.info("Processing complete")

    if include_trace:
        trace_payload = {
            'algorithm': algorithm_key,
            'seed_cbg': config.core_cbg,
            'supports_stepwise': algorithm_key != 'czi_optimal_cap',
            'steps': trace_steps or [],
        }
        if algorithm_key == 'czi_optimal_cap':
            trace_payload['note'] = (
                "czi_optimal_cap is solved as a global optimization and does not have a "
                "single greedy add-one expansion sequence."
            )
        return geoids, visualizer.map_obj, gdf, trace_payload

    return geoids, visualizer.map_obj, gdf

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        main_logger = logging.getLogger("cbg_clustering")
        main_logger.critical("Fatal error occurred", exc_info=True)
        raise
