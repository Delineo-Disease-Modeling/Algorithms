from czcode import Config, DataLoader, GraphBuilder, build_cbg_centers, setup_logging


class AnalysisResourceCache:
    def __init__(self):
        self._mobility_graphs = {}
        self._directed_mobility_graphs = {}
        self._cbg_centers = {}

    def get_mobility_graph(self, seed_cbg, patterns_file=None, patterns_folder=None, month=None, cache_tag='v3'):
        key = (seed_cbg, patterns_file, patterns_folder, month, cache_tag)
        if key in self._mobility_graphs:
            return self._mobility_graphs[key]

        config = Config(
            seed_cbg,
            0,
            patterns_file=patterns_file,
            patterns_folder=patterns_folder,
            month=month
        )
        logger = setup_logging(config)

        data_loader = DataLoader(config, logger)
        zip_codes = data_loader.get_zip_codes()
        df = data_loader.load_safegraph_data(zip_codes)

        graph = GraphBuilder(logger).gen_graph(df)
        self._mobility_graphs[key] = graph
        return graph

    def get_directed_mobility_graph(self, seed_cbg, patterns_file=None, patterns_folder=None, month=None, cache_tag='v3'):
        key = (seed_cbg, patterns_file, patterns_folder, month, cache_tag)
        if key in self._directed_mobility_graphs:
            return self._directed_mobility_graphs[key]

        config = Config(
            seed_cbg,
            0,
            patterns_file=patterns_file,
            patterns_folder=patterns_folder,
            month=month
        )
        logger = setup_logging(config)

        data_loader = DataLoader(config, logger)
        zip_codes = data_loader.get_zip_codes()
        df = data_loader.load_safegraph_data(zip_codes)

        graph = GraphBuilder(logger).gen_digraph(df)
        self._directed_mobility_graphs[key] = graph
        return graph

    def get_cbg_centers(self, seed_cbg, patterns_file=None, patterns_folder=None, month=None, cache_tag='v1'):
        key = (seed_cbg, patterns_file, patterns_folder, month, cache_tag)
        if key in self._cbg_centers:
            return self._cbg_centers[key]

        config = Config(
            seed_cbg,
            0,
            patterns_file=patterns_file,
            patterns_folder=patterns_folder,
            month=month
        )
        logger = setup_logging(config)
        data_loader = DataLoader(config, logger)
        gdf = data_loader.load_shapefiles()
        centers = build_cbg_centers(gdf)
        self._cbg_centers[key] = centers
        return centers
