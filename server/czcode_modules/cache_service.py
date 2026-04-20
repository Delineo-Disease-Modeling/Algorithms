import threading

import geopandas as gpd
import pandas as pd

try:
    from uszipcode import SearchEngine
except ImportError:  # pragma: no cover - depends on local runtime packaging.
    SearchEngine = None


class AlgorithmCacheService:
    """Explicit cache container for heavyweight algorithm resources."""

    def __init__(self):
        self._search_engines_by_thread = {}
        self._shapefile_cache = {}
        self._population_cache = {}
        self._graph_cache = {}

    def get_search_engine(self):
        if SearchEngine is None:
            raise RuntimeError('uszipcode is not installed in this Python environment')
        thread_id = threading.get_ident()
        if thread_id not in self._search_engines_by_thread:
            self._search_engines_by_thread[thread_id] = SearchEngine()
        return self._search_engines_by_thread[thread_id]

    def get_or_load_shapefiles(self, states, loader_fn) -> gpd.GeoDataFrame:
        key = frozenset(states)
        if key not in self._shapefile_cache:
            self._shapefile_cache[key] = loader_fn()
        return self._shapefile_cache[key]

    def get_or_load_population(self, csv_path: str, loader_fn) -> pd.DataFrame:
        if csv_path not in self._population_cache:
            self._population_cache[csv_path] = loader_fn()
        return self._population_cache[csv_path]

    def get_or_build_graph(self, cache_key: tuple, build_fn):
        if cache_key not in self._graph_cache:
            self._graph_cache[cache_key] = build_fn()
        return self._graph_cache[cache_key]


DEFAULT_ALGORITHM_CACHE = AlgorithmCacheService()
