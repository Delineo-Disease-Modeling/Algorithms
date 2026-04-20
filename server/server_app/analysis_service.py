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
from .seed_regions import (
    describe_city_approximation_for_zip,
    get_cbg_to_zip_map,
    get_zip_to_cbgs_map,
)


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

    @staticmethod
    def normalize_output_cbg(cbg):
        from common_geo import normalize_cbg
        return normalize_cbg(cbg) or str(cbg)

    def cluster_cbgs(self, cbg, min_pop, patterns_file=None, patterns_folder=None, month=None,
                     algorithm='czi_balanced', czi_params=None, optimal_params=None,
                     seed_guard_params=None, ttwa_params=None, hierarchical_params=None,
                     seed_cbgs=None,
                     include_trace=False, progress_callback=None):
        czi_params = czi_params or {}
        optimal_params = optimal_params or {}
        seed_guard_params = seed_guard_params or {}
        ttwa_params = ttwa_params or {}
        hierarchical_params = hierarchical_params or {}
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
                missing_cluster_cbgs.append(self.normalize_output_cbg(cbg))
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
                    'cbg': self.normalize_output_cbg(candidate),
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
                    'cbg': self.normalize_output_cbg(candidate),
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
                        'cbg': self.normalize_output_cbg(candidate),
                        'population': int(cand_pop),
                        'score': float(score),
                        'movement_to_cluster': float(movement_to_cluster),
                        'movement_to_outside': float(movement_to_outside),
                        'movement_total': float(total_movement),
                    })
                else:
                    candidate_details.append({
                        'cbg': self.normalize_output_cbg(candidate),
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

    def compute_second_order_destinations(
        self,
        seed_cbg,
        seed_cbgs,
        limit,
        pattern_selection,
        max_recommended=4,
    ):
        max_gateway_cbgs_per_city = 4
        gateway_flow_capture_target = 0.85
        max_gateway_population_per_city = 12000
        recommended_explicit_population_cap = 25000

        digraph = self.resources.get_directed_mobility_graph(
            seed_cbg,
            patterns_file=pattern_selection.file_path,
            cache_tag='v3',
        )
        config = Config(seed_cbg, 0, patterns_file=pattern_selection.file_path, month=pattern_selection.month)
        logger = setup_logging(config)

        normalized_seed_cbgs = []
        missing_seed_cbgs = []
        seen = set()
        for cbg in seed_cbgs:
            graph_cbg = self.graph_key(cbg)
            if not graph_cbg or graph_cbg in seen:
                continue
            seen.add(graph_cbg)
            if graph_cbg in digraph:
                normalized_seed_cbgs.append(graph_cbg)
            else:
                missing_seed_cbgs.append(self.normalize_output_cbg(cbg))

        if not normalized_seed_cbgs:
            raise ValueError('None of the seed region CBGs are present in the directed mobility graph')

        seed_set = set(normalized_seed_cbgs)
        cbg_centers = {}
        get_cbg_centers = getattr(self.resources, 'get_cbg_centers', None)
        if callable(get_cbg_centers):
            try:
                cbg_centers = get_cbg_centers(
                    seed_cbg,
                    patterns_file=pattern_selection.file_path,
                    month=pattern_selection.month,
                    cache_tag='v1',
                ) or {}
            except Exception:
                cbg_centers = {}

        def lookup_center(cbg):
            if not cbg_centers:
                return None
            normalized = self.normalize_output_cbg(cbg)
            graph_cbg = self.graph_key(cbg)
            return cbg_centers.get(graph_cbg) or cbg_centers.get(normalized)

        def average_center(cbgs):
            coords = [lookup_center(cbg) for cbg in cbgs]
            coords = [coord for coord in coords if coord]
            if not coords:
                return None
            lat = sum(float(coord[0]) for coord in coords) / len(coords)
            lon = sum(float(coord[1]) for coord in coords) / len(coords)
            return lat, lon

        def distance_from_seed(cbgs):
            if not seed_center:
                return None
            unit_center = average_center(cbgs)
            if not unit_center:
                return None
            return float(distance(
                seed_center[0],
                seed_center[1],
                unit_center[0],
                unit_center[1],
            ))

        def coupling_score(flow_value, distance_km):
            flow_value = float(flow_value or 0.0)
            if distance_km is None or DEFAULT_DISTANCE_SCALE_KM <= 0:
                return flow_value
            return float(flow_value / (1.0 + (distance_km / DEFAULT_DISTANCE_SCALE_KM)))

        seed_center = average_center(normalized_seed_cbgs)
        cbg_to_zip = get_cbg_to_zip_map()
        zip_to_cbgs = get_zip_to_cbgs_map()
        seed_zip_codes = sorted({
            zip_code for zip_code in (cbg_to_zip.get(cbg) for cbg in normalized_seed_cbgs) if zip_code
        })
        seed_destination_unit_ids = {
            city_info['unit_id']
            for city_info in (
                describe_city_approximation_for_zip(zip_code)
                for zip_code in seed_zip_codes
            )
            if city_info and city_info.get('unit_id')
        }
        seed_city_labels = sorted({
            str(city_info['label'])
            for city_info in (
                describe_city_approximation_for_zip(zip_code)
                for zip_code in seed_zip_codes
            )
            if city_info and city_info.get('unit_type') == 'city_approximation' and city_info.get('label')
        })

        seed_population = 0
        for cbg in normalized_seed_cbgs:
            seed_population += int(cbg_population(cbg, config, logger) or 0)

        total_seed_movement = 0.0
        total_seed_internal_movement = 0.0
        total_seed_external_outbound_flow = 0.0
        total_seed_external_inbound_flow = 0.0
        destination_stats = {}

        for seed_node in normalized_seed_cbgs:
            self_weight = float(digraph.nodes[seed_node].get('self_weight', 0) or 0)
            total_seed_movement += self_weight
            total_seed_internal_movement += self_weight

            for _, dst, data in digraph.out_edges(seed_node, data=True):
                weight = float(data.get('weight', 0) or 0)
                if weight <= 0:
                    continue
                total_seed_movement += weight
                if dst in seed_set:
                    total_seed_internal_movement += weight
                    continue

                total_seed_external_outbound_flow += weight
                destination_zip = cbg_to_zip.get(dst)
                if not destination_zip:
                    continue
                city_info = describe_city_approximation_for_zip(destination_zip)
                unit_id = city_info['unit_id'] if city_info else None
                if not unit_id or unit_id in seed_destination_unit_ids:
                    continue

                stats = destination_stats.setdefault(unit_id, {
                    'label': city_info['label'] if city_info else f'ZIP {destination_zip}',
                    'unit_type': city_info['unit_type'] if city_info else 'zip_fallback',
                    'outbound_flow': 0.0,
                    'inbound_flow': 0.0,
                    'member_zips': set(),
                    'member_cbgs': set(),
                    'member_flow_by_cbg': {},
                })
                stats['outbound_flow'] += weight
                stats['member_zips'].add(destination_zip)
                stats['member_cbgs'].add(dst)
                member_stats = stats['member_flow_by_cbg'].setdefault(dst, {
                    'outbound_flow': 0.0,
                    'inbound_flow': 0.0,
                })
                member_stats['outbound_flow'] += weight

            for src, _, data in digraph.in_edges(seed_node, data=True):
                weight = float(data.get('weight', 0) or 0)
                if weight <= 0 or src in seed_set:
                    continue

                total_seed_external_inbound_flow += weight
                source_zip = cbg_to_zip.get(src)
                if not source_zip:
                    continue
                city_info = describe_city_approximation_for_zip(source_zip)
                unit_id = city_info['unit_id'] if city_info else None
                if not unit_id or unit_id in seed_destination_unit_ids:
                    continue

                stats = destination_stats.setdefault(unit_id, {
                    'label': city_info['label'] if city_info else f'ZIP {source_zip}',
                    'unit_type': city_info['unit_type'] if city_info else 'zip_fallback',
                    'outbound_flow': 0.0,
                    'inbound_flow': 0.0,
                    'member_zips': set(),
                    'member_cbgs': set(),
                    'member_flow_by_cbg': {},
                })
                stats['inbound_flow'] += weight
                stats['member_zips'].add(source_zip)
                stats['member_cbgs'].add(src)
                member_stats = stats['member_flow_by_cbg'].setdefault(src, {
                    'outbound_flow': 0.0,
                    'inbound_flow': 0.0,
                })
                member_stats['inbound_flow'] += weight

        destinations = []
        for unit_id, stats in destination_stats.items():
            outbound_flow = float(stats['outbound_flow'])
            inbound_flow = float(stats['inbound_flow'])
            bidirectional_flow = outbound_flow + inbound_flow
            if bidirectional_flow <= 0:
                continue

            unit_cbgs = sorted({
                cbg
                for zip_code in stats['member_zips']
                for cbg in zip_to_cbgs.get(zip_code, [])
                if cbg not in seed_set and cbg in digraph
            })
            if not unit_cbgs:
                unit_cbgs = sorted(
                    cbg for cbg in stats['member_cbgs']
                    if cbg not in seed_set and cbg in digraph
                )
            if not unit_cbgs:
                continue

            city_population = 0
            for cbg in unit_cbgs:
                city_population += int(cbg_population(cbg, config, logger) or 0)

            member_details = []
            for cbg in unit_cbgs:
                member_flow = stats['member_flow_by_cbg'].get(cbg, {})
                member_outbound_flow = float(member_flow.get('outbound_flow', 0.0) or 0.0)
                member_inbound_flow = float(member_flow.get('inbound_flow', 0.0) or 0.0)
                member_bidirectional_flow = member_outbound_flow + member_inbound_flow
                member_distance_km = None
                member_center = lookup_center(cbg)
                if seed_center and member_center:
                    member_distance_km = float(distance(
                        seed_center[0],
                        seed_center[1],
                        member_center[0],
                        member_center[1],
                    ))
                member_details.append({
                    'cbg': self.normalize_output_cbg(cbg),
                    'population': int(cbg_population(cbg, config, logger) or 0),
                    'seed_outbound_flow': member_outbound_flow,
                    'seed_inbound_flow': member_inbound_flow,
                    'seed_bidirectional_flow': member_bidirectional_flow,
                    'distance_km': member_distance_km,
                    'gateway_score': coupling_score(member_bidirectional_flow, member_distance_km),
                })

            member_details.sort(
                key=lambda item: (
                    -float(item['gateway_score']),
                    -float(item['seed_bidirectional_flow']),
                    -float(item['seed_outbound_flow']),
                    float(item['distance_km']) if item.get('distance_km') is not None else float('inf'),
                    int(item['population']),
                    str(item['cbg']),
                ),
            )

            selected_gateway_cbgs = []
            selected_gateway_population = 0
            captured_gateway_bidirectional_flow = 0.0
            gateway_target_flow = bidirectional_flow * gateway_flow_capture_target
            for detail in member_details:
                member_flow = float(detail['seed_bidirectional_flow'])
                if not selected_gateway_cbgs:
                    selected_gateway_cbgs.append(detail)
                    selected_gateway_population += int(detail['population'])
                    captured_gateway_bidirectional_flow += member_flow
                    continue

                if member_flow <= 0:
                    continue
                if len(selected_gateway_cbgs) >= max_gateway_cbgs_per_city:
                    break
                if selected_gateway_population + int(detail['population']) > max_gateway_population_per_city:
                    continue
                if captured_gateway_bidirectional_flow >= gateway_target_flow:
                    break

                selected_gateway_cbgs.append(detail)
                selected_gateway_population += int(detail['population'])
                captured_gateway_bidirectional_flow += member_flow

            if not selected_gateway_cbgs:
                selected_gateway_cbgs = member_details[:1]
                selected_gateway_population = sum(int(item['population']) for item in selected_gateway_cbgs)
                captured_gateway_bidirectional_flow = sum(
                    float(item['seed_bidirectional_flow']) for item in selected_gateway_cbgs
                )

            explicit_cbgs = [item['cbg'] for item in selected_gateway_cbgs]
            distance_km = distance_from_seed(explicit_cbgs)
            coupling = coupling_score(bidirectional_flow, distance_km)
            destinations.append({
                'unit_id': unit_id,
                'label': str(stats.get('label') or unit_id),
                'unit_type': str(stats.get('unit_type') or 'city_approximation'),
                'cbgs': explicit_cbgs,
                'cbg_count': len(explicit_cbgs),
                'city_cbg_count': len(unit_cbgs),
                'zip_codes': sorted(stats['member_zips']),
                'zip_count': len(stats['member_zips']),
                'population': int(selected_gateway_population),
                'city_population': int(city_population),
                'outbound_flow': outbound_flow,
                'inbound_flow': inbound_flow,
                'bidirectional_flow': bidirectional_flow,
                'share_of_seed_external_bidirectional': 0.0,
                'coupling': float(coupling),
                'distance_km': distance_km,
                'captured_bidirectional_flow': float(captured_gateway_bidirectional_flow),
                'captured_bidirectional_flow_share': (
                    float(captured_gateway_bidirectional_flow / bidirectional_flow)
                    if bidirectional_flow > 0
                    else 0.0
                ),
                'gateway_cbgs': selected_gateway_cbgs,
                'share_of_seed_total_movement': (
                    float(outbound_flow / total_seed_movement)
                    if total_seed_movement > 0
                    else 0.0
                ),
                'share_of_seed_external_outbound': (
                    float(outbound_flow / total_seed_external_outbound_flow)
                    if total_seed_external_outbound_flow > 0
                    else 0.0
                ),
            })

        total_seed_external_bidirectional_flow = sum(
            float(item['bidirectional_flow']) for item in destinations
        )
        for destination in destinations:
            destination['share_of_seed_external_bidirectional'] = (
                float(destination['bidirectional_flow'] / total_seed_external_bidirectional_flow)
                if total_seed_external_bidirectional_flow > 0
                else 0.0
            )

        destinations.sort(
            key=lambda item: (
                -float(item['coupling']),
                -float(item['bidirectional_flow']),
                -float(item['outbound_flow']),
                float(item['distance_km']) if item.get('distance_km') is not None else float('inf'),
                int(item['population']),
                str(item['unit_id']),
            ),
        )

        if isinstance(limit, int) and limit > 0:
            destinations = destinations[:limit]

        cumulative_external_outbound_share = 0.0
        cumulative_total_movement_share = 0.0
        cumulative_bidirectional_share = 0.0
        recommended_unit_ids = []
        recommended_external_outbound_share = 0.0
        recommended_total_movement_share = 0.0
        recommended_bidirectional_share = 0.0
        recommended_explicit_population = int(seed_population)
        for destination in destinations:
            cumulative_external_outbound_share += float(destination['share_of_seed_external_outbound'])
            cumulative_total_movement_share += float(destination['share_of_seed_total_movement'])
            cumulative_bidirectional_share += float(destination['share_of_seed_external_bidirectional'])
            destination['cumulative_external_outbound_share'] = float(min(1.0, cumulative_external_outbound_share))
            destination['cumulative_seed_total_movement_share'] = float(min(1.0, cumulative_total_movement_share))
            destination['cumulative_external_bidirectional_share'] = float(min(1.0, cumulative_bidirectional_share))

            candidate_external_share = float(destination['share_of_seed_external_outbound'])
            candidate_total_share = float(destination['share_of_seed_total_movement'])
            candidate_bidirectional_share = float(destination['share_of_seed_external_bidirectional'])
            candidate_population = int(destination['population'])
            next_recommended_population = recommended_explicit_population + candidate_population
            should_recommend = (
                len(recommended_unit_ids) < max(0, int(max_recommended))
                and next_recommended_population <= recommended_explicit_population_cap
                and (
                    candidate_bidirectional_share >= 0.10
                    or (
                        recommended_bidirectional_share < 0.80
                        and candidate_bidirectional_share >= 0.03
                    )
                    or (
                        not recommended_unit_ids
                        and (
                            candidate_total_share >= 0.01
                            or candidate_external_share >= 0.01
                        )
                    )
                )
            )
            destination['recommended'] = bool(should_recommend)
            if should_recommend:
                recommended_unit_ids.append(str(destination['unit_id']))
                recommended_external_outbound_share += candidate_external_share
                recommended_total_movement_share += candidate_total_share
                recommended_bidirectional_share += candidate_bidirectional_share
                recommended_explicit_population = next_recommended_population

        return {
            'seed_cbg': seed_cbg,
            'seed_cbgs': [self.normalize_output_cbg(cbg) for cbg in normalized_seed_cbgs],
            'missing_seed_cbgs': sorted(set(missing_seed_cbgs)),
            'seed_zip_codes': seed_zip_codes,
            'seed_city_labels': seed_city_labels,
            'seed_population': int(seed_population),
            'total_seed_movement': float(total_seed_movement),
            'total_seed_internal_movement': float(total_seed_internal_movement),
            'total_seed_external_outbound_flow': float(total_seed_external_outbound_flow),
            'total_seed_external_inbound_flow': float(total_seed_external_inbound_flow),
            'total_seed_external_bidirectional_flow': float(total_seed_external_bidirectional_flow),
            'unit_type': 'city_approximation',
            'approximation_note': 'City units are approximated by grouping ZIP codes with the same city/state metadata. Guided mode keeps only the highest-coupling gateway CBGs from each selected city in the explicit simulation zone.',
            'destination_count': len(destinations),
            'destinations': destinations,
            'recommended_unit_ids': recommended_unit_ids,
            'recommended_captured_external_outbound_share': float(min(1.0, recommended_external_outbound_share)),
            'recommended_captured_seed_total_movement_share': float(min(1.0, recommended_total_movement_share)),
            'recommended_captured_external_bidirectional_share': float(min(1.0, recommended_bidirectional_share)),
            'recommended_explicit_population': int(recommended_explicit_population),
            'recommended_explicit_population_cap': int(recommended_explicit_population_cap),
            'patterns_file_used': pattern_selection.file_path,
            'patterns_source': pattern_selection.source,
            'patterns_month': pattern_selection.month,
            'use_test_data': pattern_selection.use_test_data,
        }

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
                    'clustering_params': (
                        algorithm_config['effective_czi_params']
                        if algorithm_config['algorithm'] == 'czi_balanced'
                        else algorithm_config['effective_optimal_params'] if algorithm_config['algorithm'] == 'czi_optimal_cap'
                        else algorithm_config['effective_seed_guard_params'] if algorithm_config['algorithm'] == 'greedy_weight_seed_guard'
                        else algorithm_config['effective_ttwa_params'] if algorithm_config['algorithm'] == 'greedy_ttwa'
                        else algorithm_config['effective_hierarchical_params'] if algorithm_config['algorithm'] == 'hierarchical_core_satellites'
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
