import threading

from czcode import Helpers, generate_cz
from geojsongen import get_cbg_geojson

from .analysis_config import effective_params_for_algorithm
from .analysis_helpers import compute_top_candidate_pois
from .analysis_resources import AnalysisResourceCache
from .frontier_candidates import FrontierCandidateAnalyzer
from .logging_utils import log_candidate_pois
from .pattern_resolution import resolve_localized_patterns_extract
from .second_order_destinations import SecondOrderDestinationAnalyzer


class PreviewClusteringService:
    def __init__(self, clustering_store, resources=None):
        self.clustering_store = clustering_store
        self.resources = resources or AnalysisResourceCache()
        self.frontier_candidates = FrontierCandidateAnalyzer(self.resources)
        self.second_order_destinations = SecondOrderDestinationAnalyzer(self.resources)

    @staticmethod
    def graph_key(cbg):
        try:
            return str(int(float(cbg)))
        except (TypeError, ValueError):
            return str(cbg).strip()

    @staticmethod
    def normalize_output_cbg(cbg):
        from common_geo import normalize_cbg
        return normalize_cbg(cbg) or str(cbg)

    def cluster_cbgs(self, cbg, min_pop, patterns_file=None, patterns_folder=None, month=None,
                     algorithm='czi_balanced', czi_params=None, optimal_params=None,
                     seed_guard_params=None, ttwa_params=None, hierarchical_params=None,
                     mobility_prune_params=None,
                     seed_cbgs=None,
                     include_trace=False, progress_callback=None):
        czi_params = czi_params or {}
        optimal_params = optimal_params or {}
        seed_guard_params = seed_guard_params or {}
        ttwa_params = ttwa_params or {}
        hierarchical_params = hierarchical_params or {}
        mobility_prune_params = mobility_prune_params or {}
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
            seed_cbgs=seed_cbgs,
            local_radius_km=hierarchical_params.get('local_radius_km'),
            core_containment_threshold=hierarchical_params.get('core_containment_threshold'),
            core_improvement_epsilon=hierarchical_params.get('core_improvement_epsilon'),
            satellite_flow_threshold=hierarchical_params.get('satellite_flow_threshold'),
            max_satellites=hierarchical_params.get('max_satellites'),
            mobility_prune_min_seed_capture=mobility_prune_params.get('min_seed_capture'),
            containment_threshold=ttwa_params.get('containment_threshold'),
            include_trace=include_trace,
            progress_callback=progress_callback,
        )
        if include_trace:
            geoids, map_obj, _gdf, trace_payload = result
            return geoids, [map_obj.location[0], map_obj.location[1]], trace_payload

        geoids, map_obj, _gdf = result
        return geoids, [map_obj.location[0], map_obj.location[1]], None

    def rank_frontier_candidates_for_cluster(self, *args, **kwargs):
        return self.frontier_candidates.rank_frontier_candidates_for_cluster(*args, **kwargs)

    def compute_second_order_destinations(self, *args, **kwargs):
        return self.second_order_destinations.compute_second_order_destinations(*args, **kwargs)

    def start_cluster_job(self, cbg_str, min_pop, pattern_selection, algorithm_config, include_trace, seed_cbgs=None):
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
                    hierarchical_params=algorithm_config['effective_hierarchical_params'],
                    mobility_prune_params=algorithm_config['effective_mobility_prune_params'],
                    seed_cbgs=seed_cbgs,
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
                    'clustering_params': effective_params_for_algorithm(algorithm_config),
                    'patterns_file_used': pattern_selection.file_path,
                    'patterns_source': pattern_selection.source,
                    'patterns_month': pattern_selection.month,
                    'use_test_data': pattern_selection.use_test_data,
                }
                if include_trace:
                    response_data['trace'] = trace_payload
                    response_data['trace_geojson'] = trace_geojson
                    if isinstance(trace_payload, dict) and trace_payload.get('algorithm_metadata'):
                        response_data['algorithm_metadata'] = trace_payload.get('algorithm_metadata')

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
