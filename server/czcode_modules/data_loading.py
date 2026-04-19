import os

import geopandas as gpd
import pandas as pd

from common_geo import STATE_ABBR_TO_FIPS
from patterns_loader import PatternsData

from .cache_service import DEFAULT_ALGORITHM_CACHE


class DataLoader:
    def __init__(self, config, logger, cache_service=DEFAULT_ALGORITHM_CACHE):
        self.config = config
        self.logger = logger
        self.cache_service = cache_service

    def get_zip_codes(self):
        search = self.cache_service.get_search_engine()
        zip_codes = []
        for state in self.config.states:
            self.logger.info(f"Retrieving zip codes for {state}")
            data = search.by_state(state, returns=0)
            state_zips = [int(item.zipcode) for item in data]
            zip_codes.extend(state_zips)
        return zip_codes

    def load_safegraph_data(self, zip_codes, shared_data: 'PatternsData' = None):
        if shared_data is not None and not shared_data.is_empty():
            self.logger.info("Using pre-loaded shared patterns data for clustering")
            df = shared_data.for_clustering()
            df.columns = [str(c).strip().lower() for c in df.columns]
            if 'poi_cbg' in df.columns:
                df['poi_cbg'] = pd.to_numeric(df['poi_cbg'], errors='coerce')
                df.dropna(subset=['poi_cbg'], inplace=True)
                df['poi_cbg'] = df['poi_cbg'].astype('int64').astype('string').str.zfill(12)
            return df

        patterns_file = self.config.paths["patterns_csv"]

        if patterns_file.endswith('.parquet'):
            self.logger.info(f"Loading parquet patterns from {patterns_file}")
            df = pd.read_parquet(patterns_file)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if 'poi_cbg' in df.columns:
                df['poi_cbg'] = df['poi_cbg'].astype(str).str.strip().str.zfill(12)
            return df

        self.logger.info(f"Loading CSV patterns from {patterns_file}")
        df = pd.read_csv(patterns_file)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'poi_cbg' in df.columns:
            df['poi_cbg'] = pd.to_numeric(df['poi_cbg'], errors='coerce')
            df.dropna(subset=['poi_cbg'], inplace=True)
            df['poi_cbg'] = df['poi_cbg'].astype('int64').astype('string').str.zfill(12)
        return df

    def load_shapefiles(self):
        return self.cache_service.get_or_load_shapefiles(
            self.config.states,
            self._load_shapefiles_uncached,
        )

    def _load_shapefiles_uncached(self):
        self.logger.info("Loading shapefiles (first request, will be cached)")
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

        return gpd.GeoDataFrame(pd.concat(gds, ignore_index=True), crs="EPSG:4326")

    def get_population_data(self):
        return self.cache_service.get_or_load_population(
            self.config.paths["population_csv"],
            lambda: pd.read_csv(self.config.paths["population_csv"], index_col='census_block_group')
        )
