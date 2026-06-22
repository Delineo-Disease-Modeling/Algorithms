import json
import logging

from common_geo import build_cbg_centers

from .algorithm_runner import (
    AlgorithmRunner,
    build_trace_payload,
    normalize_algorithm_key,
    normalize_seed_cbgs,
)
from .cache_service import DEFAULT_ALGORITHM_CACHE
from .clustering import Clustering
from .config import Config
from .data_loading import DataLoader
from .export import Exporter
from .graph import GraphBuilder
from .logging_utils import setup_logging
from .metrics import Helpers, cbg_population
from .visualization import Visualizer


def main():
    seed_cbg = '240430002001'
    min_pop = 150_000

    config = Config(seed_cbg, min_pop)
    logger = setup_logging(config)
    logger.info("Starting clustering analysis")

    data_loader = DataLoader(config, logger)
    zip_codes = data_loader.get_zip_codes()
    logger.info(f"Retrieved {len(zip_codes)} zip codes")
    df = data_loader.load_safegraph_data(zip_codes)
    gdf = data_loader.load_shapefiles()
    _ = data_loader.get_population_data()

    graph_builder = GraphBuilder(logger)
    G = graph_builder.gen_graph(df)

    clustering_algo = Clustering(config, logger)
    algorithm_result = clustering_algo.greedy_weight(G, config.core_cbg, config.min_cluster_pop)
    logger.info(f"Clustering complete: {len(algorithm_result[0])} CBGs, population: {algorithm_result[1]}")
    movement_stats = Helpers.calculate_movement_stats(G, algorithm_result[0])
    logger.info(f"Movement stats: IN {movement_stats['in']}, OUT {movement_stats['out']}, Ratio {movement_stats['ratio']:.4f}")

    visualizer = Visualizer(config, logger)
    visualizer.generate_maps(G, gdf, algorithm_result)

    output_map_path = f"{config.output_dir}/{config.paths['output_html']}"
    visualizer.map_obj.save(output_map_path)
    logger.info(f"Map saved to {output_map_path}")

    exporter = Exporter(config, logger)
    exporter.generate_yaml_output(G, algorithm_result)

    with open(r'output/algorithm_result.json', 'w', encoding='utf-8') as f:
        json.dump(algorithm_result, f, indent=2)

    with open(r'output/cbglistpop.json', 'w', encoding='utf-8') as f:
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
                start_date=None, shared_data=None, algorithm='czi_balanced',
                distance_penalty_weight=None, distance_scale_km=None,
                optimal_candidate_limit=None, optimal_population_floor_ratio=None,
                optimal_mip_rel_gap=None, optimal_time_limit_sec=None,
                optimal_max_iters=None, seed_guard_distance_km=None,
                seed_cbgs=None,
                mobility_prune_min_seed_capture=None,
                containment_threshold=None, include_trace=False,
                progress_callback=None,
                cache_service=DEFAULT_ALGORITHM_CACHE):
    prog = progress_callback or (lambda msg, pct: None)

    prog('Resolving location...', 5)
    config = Config(cbg, min_pop, patterns_file=patterns_file,
                    patterns_folder=patterns_folder, month=month,
                    start_date=start_date)

    logger = setup_logging(config)
    logger.info("Starting clustering analysis")
    if config.month:
        logger.info(f"Using monthly patterns: {config.month} (file: {config.paths['patterns_csv']})")

    prog('Loading zip codes...', 10)
    data_loader = DataLoader(config, logger, cache_service=cache_service)
    zip_codes = data_loader.get_zip_codes()
    logger.info(f"Retrieved {len(zip_codes)} zip codes")

    prog('Loading patterns data...', 20)
    df = data_loader.load_safegraph_data(zip_codes, shared_data=shared_data)

    prog('Loading shapefiles...', 40)
    gdf = data_loader.load_shapefiles()
    _ = data_loader.get_population_data()

    prog('Building mobility graph...', 55)
    graph_cache_key = (config.paths['patterns_csv'], frozenset(config.states))
    G = cache_service.get_or_build_graph(
        graph_cache_key,
        lambda: GraphBuilder(logger).gen_graph(df),
    )
    cbg_centers = build_cbg_centers(gdf)

    clustering_algo = Clustering(config, logger)
    algorithm_key = normalize_algorithm_key(algorithm)

    if G.number_of_nodes() == 0:
        raise ValueError(
            "No mobility graph could be built from the selected patterns data. "
            "Try a different start date or location."
        )
    normalized_seed_cbgs = normalize_seed_cbgs(seed_cbgs, config.core_cbg)

    prog('Running clustering algorithm...', 75)
    logger.info(f"Using clustering algorithm: {algorithm_key}")
    trace_steps = [] if include_trace else None
    algorithm_run = AlgorithmRunner(
        clustering_algo=clustering_algo,
        config=config,
        logger=logger,
        graph=G,
        patterns_df=df,
        cbg_centers=cbg_centers,
        cache_service=cache_service,
    ).run(
        algorithm_key,
        normalized_seed_cbgs,
        trace_steps=trace_steps,
        algorithm=algorithm,
        distance_penalty_weight=distance_penalty_weight,
        distance_scale_km=distance_scale_km,
        optimal_candidate_limit=optimal_candidate_limit,
        optimal_population_floor_ratio=optimal_population_floor_ratio,
        optimal_mip_rel_gap=optimal_mip_rel_gap,
        optimal_time_limit_sec=optimal_time_limit_sec,
        optimal_max_iters=optimal_max_iters,
        seed_guard_distance_km=seed_guard_distance_km,
        mobility_prune_min_seed_capture=mobility_prune_min_seed_capture,
        containment_threshold=containment_threshold,
    )
    algorithm_result = algorithm_run.result
    logger.info(f"Clustering complete: {len(algorithm_result[0])} CBGs, population: {algorithm_result[1]}")

    prog('Generating map...', 90)
    visualizer = Visualizer(config, logger)
    visualizer.generate_maps(G, gdf, algorithm_result)

    geoids = {cbg: cbg_population(cbg, config, logger) for cbg in algorithm_result[0]}

    prog('Done', 100)
    logger.info("Processing complete")

    if include_trace:
        return geoids, visualizer.map_obj, gdf, build_trace_payload(
            algorithm_key,
            config.core_cbg,
            trace_steps,
            algorithm_run.metadata,
        )

    return geoids, visualizer.map_obj, gdf


if __name__ == "__main__":
    try:
        main()
    except Exception:
        main_logger = logging.getLogger("cbg_clustering")
        main_logger.critical("Fatal error occurred", exc_info=True)
        raise
