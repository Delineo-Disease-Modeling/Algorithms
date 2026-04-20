import json
import logging

from common_geo import build_cbg_centers

from .cache_service import DEFAULT_ALGORITHM_CACHE
from .clustering import Clustering
from .config import Config
from .data_loading import DataLoader
from .export import Exporter
from .graph import GraphBuilder
from .logging_utils import setup_logging
from .metrics import Helpers, cbg_population
from server_app.seed_regions import get_cbg_to_zip_map, get_zip_to_cbgs_map
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
                seed_cbgs=None, local_radius_km=None,
                core_containment_threshold=None, core_improvement_epsilon=None,
                satellite_flow_threshold=None, max_satellites=None,
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
    algorithm_key = str(algorithm or 'czi_balanced').strip().lower()
    algorithm_map = {
        'czi_balanced': clustering_algo.greedy_czi_balanced,
        'czi_optimal_cap': clustering_algo.czi_optimal_cap,
        'greedy_fast': clustering_algo.greedy_fast,
        'greedy_weight': clustering_algo.greedy_weight,
        'greedy_weight_seed_guard': clustering_algo.greedy_weight_seed_guard,
        'greedy_ratio': clustering_algo.greedy_ratio,
        'greedy_ttwa': clustering_algo.greedy_ttwa,
        'hierarchical_core_satellites': clustering_algo.hierarchical_core_satellites,
    }

    if G.number_of_nodes() == 0:
        raise ValueError(
            "No mobility graph could be built from the selected patterns data. "
            "Try a different start date or location."
        )
    normalized_seed_cbgs = []
    seen_seed_cbgs = set()
    for seed_cbg in seed_cbgs or [config.core_cbg]:
        if not seed_cbg or seed_cbg in seen_seed_cbgs:
            continue
        seen_seed_cbgs.add(seed_cbg)
        normalized_seed_cbgs.append(seed_cbg)

    if algorithm_key == 'hierarchical_core_satellites':
        if not any(seed_cbg in G for seed_cbg in normalized_seed_cbgs):
            raise ValueError(
                "None of the resolved seed-region CBGs are present in the mobility graph. "
                "Try a different location or date."
            )
    elif config.core_cbg not in G:
        raise ValueError(
            f"Seed CBG {config.core_cbg} is not present in the mobility graph for "
            f"{config.paths['patterns_csv']}. Try a different start date or location."
        )

    if algorithm_key not in algorithm_map:
        raise ValueError(
            f"Invalid clustering algorithm '{algorithm}'. "
            "Valid options: czi_balanced, czi_optimal_cap, greedy_fast, greedy_weight, "
            "greedy_weight_seed_guard, greedy_ratio, greedy_ttwa, hierarchical_core_satellites"
        )

    prog('Running clustering algorithm...', 75)
    logger.info(f"Using clustering algorithm: {algorithm_key}")
    trace_steps = [] if include_trace else None
    algorithm_metadata = {}
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
        seed_guard_kwargs = {'cbg_centers': cbg_centers}
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
    elif algorithm_key == 'greedy_ttwa':
        digraph_cache_key = (config.paths['patterns_csv'], frozenset(config.states), 'directed')
        DG = cache_service.get_or_build_graph(
            digraph_cache_key,
            lambda: GraphBuilder(logger).gen_digraph(df),
        )
        if DG.number_of_nodes() == 0:
            raise ValueError(
                "No directed mobility graph could be built from the selected patterns data."
            )
        if config.core_cbg not in DG:
            raise ValueError(
                f"Seed CBG {config.core_cbg} is not present in the directed mobility graph for "
                f"{config.paths['patterns_csv']}. Try a different start date or location."
            )
        ttwa_kwargs = {}
        if containment_threshold is not None:
            ttwa_kwargs['containment_threshold'] = float(containment_threshold)
        if trace_steps is not None:
            ttwa_kwargs['trace_collector'] = trace_steps
        algorithm_result = clustering_algo.greedy_ttwa(
            DG,
            config.core_cbg,
            config.min_cluster_pop,
            **ttwa_kwargs
        )
    elif algorithm_key == 'hierarchical_core_satellites':
        digraph_cache_key = (config.paths['patterns_csv'], frozenset(config.states), 'directed')
        DG = cache_service.get_or_build_graph(
            digraph_cache_key,
            lambda: GraphBuilder(logger).gen_digraph(df),
        )
        hierarchy_kwargs = {
            'cbg_to_zip': get_cbg_to_zip_map(),
            'zip_to_cbgs': get_zip_to_cbgs_map(),
            'cbg_centers': cbg_centers,
        }
        if local_radius_km is not None:
            hierarchy_kwargs['local_radius_km'] = float(local_radius_km)
        if core_containment_threshold is not None:
            hierarchy_kwargs['core_containment_threshold'] = float(core_containment_threshold)
        if core_improvement_epsilon is not None:
            hierarchy_kwargs['core_improvement_epsilon'] = float(core_improvement_epsilon)
        if satellite_flow_threshold is not None:
            hierarchy_kwargs['satellite_flow_threshold'] = float(satellite_flow_threshold)
        if max_satellites is not None:
            hierarchy_kwargs['max_satellites'] = int(max_satellites)
        if trace_steps is not None:
            hierarchy_kwargs['trace_collector'] = trace_steps
        algorithm_result = clustering_algo.hierarchical_core_satellites(
            DG,
            normalized_seed_cbgs,
            config.min_cluster_pop,
            **hierarchy_kwargs
        )
        if len(algorithm_result) > 2 and isinstance(algorithm_result[2], dict):
            algorithm_metadata = algorithm_result[2]
    else:
        algorithm_result = algorithm_map[algorithm_key](
            G,
            config.core_cbg,
            config.min_cluster_pop,
            trace_collector=trace_steps
        )
    logger.info(f"Clustering complete: {len(algorithm_result[0])} CBGs, population: {algorithm_result[1]}")

    prog('Generating map...', 90)
    visualizer = Visualizer(config, logger)
    visualizer.generate_maps(G, gdf, algorithm_result)

    geoids = {cbg: cbg_population(cbg, config, logger) for cbg in algorithm_result[0]}

    prog('Done', 100)
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
        elif algorithm_key == 'hierarchical_core_satellites':
            trace_payload['note'] = (
                "Trace steps show local core growth. ZIP-level satellite selection is "
                "applied after the local core stabilizes."
            )
            trace_payload['algorithm_metadata'] = algorithm_metadata
        return geoids, visualizer.map_obj, gdf, trace_payload

    return geoids, visualizer.map_obj, gdf


if __name__ == "__main__":
    try:
        main()
    except Exception:
        main_logger = logging.getLogger("cbg_clustering")
        main_logger.critical("Fatal error occurred", exc_info=True)
        raise
