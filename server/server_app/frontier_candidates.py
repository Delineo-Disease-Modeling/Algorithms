from common_geo import normalize_cbg
from czcode import Clustering, Helpers, cbg_population, distance, setup_logging

from .analysis_config import build_analysis_config
from .constants import (
    DEFAULT_DISTANCE_PENALTY_WEIGHT,
    DEFAULT_DISTANCE_SCALE_KM,
    DEFAULT_SEED_GUARD_DISTANCE_KM,
)


class FrontierCandidateAnalyzer:
    def __init__(self, resources):
        self.resources = resources

    @staticmethod
    def graph_key(cbg):
        try:
            return str(int(float(cbg)))
        except (TypeError, ValueError):
            return str(cbg).strip()

    @staticmethod
    def normalize_output_cbg(cbg):
        return normalize_cbg(cbg) or str(cbg)

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

        config = build_analysis_config(seed_cbg, 0, patterns_file=patterns_file, month=month)
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
                    cand_center = cbg_centers.get(candidate) or cbg_centers.get(self.normalize_output_cbg(candidate))
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
