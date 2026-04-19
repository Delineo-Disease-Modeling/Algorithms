import threading

from czcode import (
    Clustering,
    Config,
    DataLoader,
    GraphBuilder,
    Helpers,
    build_cbg_centers,
    cbg_population,
    distance,
    generate_cz,
    setup_logging,
)
from geojsongen import get_cbg_geojson

from .analysis_helpers import compute_top_candidate_pois
from .constants import (
    DEFAULT_DISTANCE_PENALTY_WEIGHT,
    DEFAULT_DISTANCE_SCALE_KM,
    DEFAULT_SEED_GUARD_DISTANCE_KM,
)
from .logging_utils import log_candidate_pois
from .pattern_resolution import resolve_localized_patterns_extract


class AnalysisResourceCache:
    def __init__(self):
        self._mobility_graphs = {}
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


class PreviewClusteringService:
    def __init__(self, clustering_store, resources=None):
        self.clustering_store = clustering_store
        self.resources = resources or AnalysisResourceCache()

    @staticmethod
    def graph_key(cbg):
        try:
            return str(int(float(cbg)))
        except (TypeError, ValueError):
            return str(cbg).strip()

    def cluster_cbgs(self, cbg, min_pop, patterns_file=None, patterns_folder=None, month=None,
                     algorithm='czi_balanced', czi_params=None, optimal_params=None,
                     seed_guard_params=None, ttwa_params=None,
                     include_trace=False, progress_callback=None):
        czi_params = czi_params or {}
        optimal_params = optimal_params or {}
        seed_guard_params = seed_guard_params or {}
        ttwa_params = ttwa_params or {}
        result = generate_cz(
            cbg,
            min_pop,
            patterns_file=patterns_file,
            patterns_folder=patterns_folder,
            month=month,
            algorithm=algorithm,
            distance_penalty_weight=czi_params.get('distance_penalty_weight'),
            distance_scale_km=czi_params.get('distance_scale_km'),
            optimal_candidate_limit=optimal_params.get('optimal_candidate_limit'),
            optimal_population_floor_ratio=optimal_params.get('optimal_population_floor_ratio'),
            optimal_mip_rel_gap=optimal_params.get('optimal_mip_rel_gap'),
            optimal_time_limit_sec=optimal_params.get('optimal_time_limit_sec'),
            optimal_max_iters=optimal_params.get('optimal_max_iters'),
            seed_guard_distance_km=seed_guard_params.get('seed_guard_distance_km'),
            containment_threshold=ttwa_params.get('containment_threshold'),
            include_trace=include_trace,
            progress_callback=progress_callback,
        )
        if include_trace:
            geoids, map_obj, _gdf, trace_payload = result
            return geoids, [map_obj.location[0], map_obj.location[1]], trace_payload

        geoids, map_obj, _gdf = result
        return geoids, [map_obj.location[0], map_obj.location[1]], None

    def rank_frontier_candidates_for_cluster(
        self,
        graph,
        seed_cbg,
        cluster_cbgs,
        algorithm,
        min_pop=0,
        czi_params=None,
        seed_guard_params=None,
        patterns_file=None,
        month=None,
        limit=None,
    ):
        czi_params = czi_params or {}
        seed_guard_params = seed_guard_params or {}

        config = Config(seed_cbg, 0, patterns_file=patterns_file, month=month)
        logger = setup_logging(config)
        clustering = Clustering(config, logger)

        def normalize_output_cbg(cbg):
            from common_geo import normalize_cbg
            return normalize_cbg(cbg) or str(cbg)

        seed_graph_cbg = self.graph_key(seed_cbg)

        cluster_graph_ids = []
        seen_graph_ids = set()
        missing_cluster_cbgs = []
        for cbg in cluster_cbgs:
            gkey = self.graph_key(cbg)
            if not gkey or gkey in seen_graph_ids:
                continue
            seen_graph_ids.add(gkey)
            if gkey not in graph:
                missing_cluster_cbgs.append(normalize_output_cbg(cbg))
                continue
            cluster_graph_ids.append(gkey)

        cluster_set = set(cluster_graph_ids)
        if not cluster_graph_ids:
            return [], sorted(set(missing_cluster_cbgs))

        frontier = set()
        for member in cluster_graph_ids:
            try:
                for neighbor in graph.adj[member]:
                    if neighbor not in cluster_set:
                        frontier.add(neighbor)
            except KeyError:
                continue

        if not frontier:
            return [], sorted(set(missing_cluster_cbgs))

        current_population = 0
        for cbg in cluster_cbgs:
            current_population += int(cbg_population(cbg, config, logger) or 0)

        candidate_details = []

        if algorithm == 'czi_balanced':
            distance_penalty_weight = (
                czi_params.get('distance_penalty_weight')
                if czi_params.get('distance_penalty_weight') is not None
                else DEFAULT_DISTANCE_PENALTY_WEIGHT
            )
            distance_scale_km = (
                czi_params.get('distance_scale_km')
                if czi_params.get('distance_scale_km') is not None
                else DEFAULT_DISTANCE_SCALE_KM
            )
            _ = min_pop

            movement_stats = Helpers.calculate_movement_stats(graph, cluster_graph_ids)
            movement_in = float(movement_stats.get('in', 0))
            movement_out = float(movement_stats.get('out', 0))

            cbg_centers = self.resources.get_cbg_centers(seed_cbg, patterns_file=patterns_file, month=month, cache_tag='v1')
            seed_center = cbg_centers.get(self.graph_key(seed_cbg)) or cbg_centers.get(seed_cbg)

            for candidate in frontier:
                if candidate in cluster_set or candidate not in graph:
                    continue

                cand_pop = int(cbg_population(candidate, config, logger) or 0)
                if cand_pop <= 0:
                    continue

                self_weight = float(graph.nodes[candidate].get('self_weight', 0))
                in_to_cluster = 0.0
                out_to_outside = 0.0
                for neighbor in graph.adj[candidate]:
                    weight = float(graph.adj[candidate][neighbor].get('weight', 0))
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
                if seed_center and distance_scale_km and distance_scale_km > 0:
                    cand_center = cbg_centers.get(candidate) or cbg_centers.get(normalize_output_cbg(candidate))
                    if cand_center:
                        dist_km = distance(
                            seed_center[0], seed_center[1],
                            cand_center[0], cand_center[1]
                        )
                        distance_penalty = dist_km / (dist_km + distance_scale_km)

                score = czi_after - distance_penalty_weight * distance_penalty
                candidate_details.append({
                    'cbg': normalize_output_cbg(candidate),
                    'population': int(cand_pop),
                    'score': float(score),
                    'movement_to_cluster': float(in_to_cluster),
                    'movement_to_outside': float(out_to_outside),
                    'czi_after': float(czi_after),
                    'distance_penalty': float(distance_penalty),
                    'movement_inside_after': float(inside_after),
                    'movement_boundary_after': float(boundary_after),
                })
        elif algorithm == 'greedy_weight_seed_guard':
            seed_guard_distance_km = (
                seed_guard_params.get('seed_guard_distance_km')
                if seed_guard_params.get('seed_guard_distance_km') is not None
                else DEFAULT_SEED_GUARD_DISTANCE_KM
            )
            cbg_centers = self.resources.get_cbg_centers(
                seed_cbg,
                patterns_file=patterns_file,
                month=month,
                cache_tag='v1',
            )
            contributor_set, _excluded_set = clustering._seed_guard_contributor_sets(
                seed_graph_cbg,
                cluster_graph_ids,
                cbg_centers,
                seed_guard_distance_km,
            )

            for candidate in frontier:
                if candidate in cluster_set or candidate not in graph:
                    continue

                cand_pop = int(cbg_population(candidate, config, logger) or 0)
                movement_to_cluster = 0.0
                movement_to_full_cluster = 0.0
                movement_to_outside = 0.0

                for neighbor in graph.adj[candidate]:
                    weight = float(graph.adj[candidate][neighbor].get('weight', 0))
                    if neighbor in cluster_set:
                        movement_to_full_cluster += weight
                        if neighbor in contributor_set:
                            movement_to_cluster += weight
                    else:
                        movement_to_outside += weight

                contributes_after, seed_distance = clustering._seed_guard_membership(
                    seed_graph_cbg,
                    candidate,
                    cbg_centers,
                    seed_guard_distance_km,
                )
                candidate_details.append({
                    'cbg': normalize_output_cbg(candidate),
                    'population': int(cand_pop),
                    'score': float(movement_to_cluster),
                    'movement_to_cluster': float(movement_to_cluster),
                    'movement_to_full_cluster': float(movement_to_full_cluster),
                    'movement_to_outside': float(movement_to_outside),
                    'seed_distance_km': float(seed_distance) if seed_distance is not None else None,
                    'movement_contributes_after_selection': bool(contributes_after),
                })
        else:
            for candidate in frontier:
                if candidate in cluster_set or candidate not in graph:
                    continue

                cand_pop = int(cbg_population(candidate, config, logger) or 0)
                movement_to_cluster = 0.0
                movement_to_outside = 0.0

                for neighbor in graph.adj[candidate]:
                    weight = float(graph.adj[candidate][neighbor].get('weight', 0))
                    if neighbor in cluster_set:
                        movement_to_cluster += weight
                    else:
                        movement_to_outside += weight

                if algorithm == 'greedy_ratio':
                    total_movement = movement_to_cluster + movement_to_outside
                    score = (movement_to_cluster / total_movement) if total_movement > 0 else 0.0
                    candidate_details.append({
                        'cbg': normalize_output_cbg(candidate),
                        'population': int(cand_pop),
                        'score': float(score),
                        'movement_to_cluster': float(movement_to_cluster),
                        'movement_to_outside': float(movement_to_outside),
                        'movement_total': float(total_movement),
                    })
                else:
                    candidate_details.append({
                        'cbg': normalize_output_cbg(candidate),
                        'population': int(cand_pop),
                        'score': float(movement_to_cluster),
                        'movement_to_cluster': float(movement_to_cluster),
                        'movement_to_outside': float(movement_to_outside),
                    })

        sorted_candidates = sorted(
            candidate_details,
            key=lambda item: (float(item.get('score', 0.0)), item.get('cbg', '')),
            reverse=True
        )

        if isinstance(limit, int) and limit > 0:
            sorted_candidates = sorted_candidates[:limit]

        for idx, candidate in enumerate(sorted_candidates):
            candidate['rank'] = idx + 1
            candidate['selected'] = False

        return sorted_candidates, sorted(set(missing_cluster_cbgs))

    def start_cluster_job(self, cbg_str, min_pop, pattern_selection, algorithm_config, include_trace):
        cid = self.clustering_store.next_id()

        def run():
            try:
                def prog(msg, pct):
                    self.clustering_store.update(cid, msg, pct)

                geoids, center, trace_payload = self.cluster_cbgs(
                    cbg_str,
                    min_pop,
                    patterns_file=pattern_selection.file_path,
                    month=pattern_selection.month,
                    algorithm=algorithm_config['algorithm'],
                    czi_params=algorithm_config['effective_czi_params'],
                    optimal_params=algorithm_config['effective_optimal_params'],
                    seed_guard_params=algorithm_config['effective_seed_guard_params'],
                    ttwa_params=algorithm_config['effective_ttwa_params'],
                    include_trace=include_trace,
                    progress_callback=prog
                )
                cluster = list(geoids.keys())
                size = sum(list(geoids.values()))

                self.clustering_store.update(cid, 'Generating GeoJSON...', 95)
                geojson = get_cbg_geojson(cluster, include_neighbors=True)
                trace_geojson = None

                if include_trace and trace_payload and trace_payload.get('steps'):
                    trace_cbgs_set = set(cluster)
                    for step in trace_payload.get('steps', []):
                        trace_cbgs_set.update(step.get('cluster_before', []))
                        trace_cbgs_set.update(step.get('cluster_after', []))
                        for candidate in step.get('candidates', []):
                            candidate_cbg = candidate.get('cbg')
                            if candidate_cbg:
                                trace_cbgs_set.add(candidate_cbg)
                    if trace_cbgs_set:
                        trace_geojson = get_cbg_geojson(list(trace_cbgs_set), include_neighbors=False)

                response_data = {
                    'cluster': cluster,
                    'seed_cbg': cbg_str,
                    'size': size,
                    'center': center,
                    'geojson': geojson,
                    'algorithm': algorithm_config['algorithm'],
                    'clustering_params': (
                        algorithm_config['effective_czi_params']
                        if algorithm_config['algorithm'] == 'czi_balanced'
                        else algorithm_config['effective_optimal_params'] if algorithm_config['algorithm'] == 'czi_optimal_cap'
                        else algorithm_config['effective_seed_guard_params'] if algorithm_config['algorithm'] == 'greedy_weight_seed_guard'
                        else algorithm_config['effective_ttwa_params'] if algorithm_config['algorithm'] == 'greedy_ttwa'
                        else {}
                    ),
                    'patterns_file_used': pattern_selection.file_path,
                    'patterns_source': pattern_selection.source,
                    'patterns_month': pattern_selection.month,
                    'use_test_data': pattern_selection.use_test_data,
                }
                if include_trace:
                    response_data['trace'] = trace_payload
                    response_data['trace_geojson'] = trace_geojson

                self.clustering_store.update(cid, 'Done', 100, done=True, result=response_data)
            except ValueError as exc:
                self.clustering_store.update(cid, str(exc), 0, error=True)
            except Exception as exc:
                import traceback
                print(f'Error clustering CBGs: {exc}')
                traceback.print_exc()
                self.clustering_store.update(cid, str(exc), 0, error=True)

        threading.Thread(target=run, daemon=True).start()
        return cid

    def compute_cz_metrics(self, seed_cbg, normalized_cbgs, pattern_selection):
        graph = self.resources.get_mobility_graph(seed_cbg, patterns_file=pattern_selection.file_path, cache_tag='v3')
        graph_cluster_cbgs = []
        seen_graph_ids = set()
        for cbg in normalized_cbgs:
            gkey = self.graph_key(cbg)
            if not gkey or gkey in seen_graph_ids:
                continue
            seen_graph_ids.add(gkey)
            graph_cluster_cbgs.append(gkey)

        movement_stats = Helpers.calculate_movement_stats(graph, graph_cluster_cbgs)
        return {
            'movement_inside': float(movement_stats.get('in', 0)),
            'movement_boundary': float(movement_stats.get('out', 0)),
            'czi': float(movement_stats.get('ratio', 0)),
            'containment_ratio': float(movement_stats.get('ratio', 0)),
            'cbg_count': len(normalized_cbgs),
            'patterns_file_used': pattern_selection.file_path,
            'patterns_source': pattern_selection.source,
            'patterns_month': pattern_selection.month,
            'use_test_data': pattern_selection.use_test_data,
        }

    def compute_frontier_candidates(self, seed_cbg, normalized_cluster, min_pop, limit, pattern_selection, algorithm_config):
        graph = self.resources.get_mobility_graph(seed_cbg, patterns_file=pattern_selection.file_path, cache_tag='v3')
        candidates, missing_cluster_cbgs = self.rank_frontier_candidates_for_cluster(
            graph=graph,
            seed_cbg=seed_cbg,
            cluster_cbgs=normalized_cluster,
            algorithm=algorithm_config['algorithm'],
            min_pop=min_pop,
            czi_params=algorithm_config['czi_params'],
            seed_guard_params=algorithm_config['seed_guard_params'],
            patterns_file=pattern_selection.file_path,
            month=pattern_selection.month,
            limit=limit,
        )
        return {
            'seed_cbg': seed_cbg,
            'cluster_size': len(normalized_cluster),
            'candidate_count': len(candidates),
            'candidates': candidates,
            'algorithm': algorithm_config['algorithm'],
            'patterns_file_used': pattern_selection.file_path,
            'patterns_source': pattern_selection.source,
            'patterns_month': pattern_selection.month,
            'missing_cluster_cbgs': missing_cluster_cbgs,
            'use_test_data': pattern_selection.use_test_data,
        }

    def compute_candidate_pois(self, seed_cbg, candidate_cbg, normalized_cluster, limit, pattern_selection):
        analysis_patterns_file, analysis_patterns_mode = resolve_localized_patterns_extract(
            seed_cbg,
            pattern_selection.file_path,
            month=pattern_selection.month,
            cache_tag='v3'
        )
        log_candidate_pois(
            "candidate-pois seed=%s candidate=%s cluster_size=%s source=%s analysis_mode=%s file=%s month=%s",
            seed_cbg,
            candidate_cbg,
            len(normalized_cluster),
            pattern_selection.source,
            analysis_patterns_mode,
            analysis_patterns_file,
            pattern_selection.month,
        )
        pois = compute_top_candidate_pois(
            analysis_patterns_file,
            candidate_cbg=candidate_cbg,
            cluster_cbgs=normalized_cluster,
            limit=limit
        )
        return {
            'candidate_cbg': candidate_cbg,
            'cluster_size': len(normalized_cluster),
            'pois': pois,
            'patterns_file_used': analysis_patterns_file,
            'patterns_source': pattern_selection.source,
            'patterns_analysis_mode': analysis_patterns_mode,
            'patterns_month': pattern_selection.month,
            'use_test_data': pattern_selection.use_test_data,
        }
