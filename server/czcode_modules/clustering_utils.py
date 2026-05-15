import networkx as nx

from common_geo import distance


class ClusteringUtilsMixin:
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
        metrics_after=None,
        higher_score_better=True
    ):
        if trace_collector is None:
            return

        sorted_candidates = sorted(
            candidates,
            key=lambda item: (float(item.get('score', 0.0)), item.get('cbg', '')),
            reverse=bool(higher_score_better)
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

    @staticmethod
    def _seed_centroid(seed_cbgs, cbg_centers):
        if not cbg_centers:
            return None

        coords = []
        for cbg in seed_cbgs or []:
            center = cbg_centers.get(cbg)
            if center is None:
                continue
            coords.append((float(center[0]), float(center[1])))

        if not coords:
            return None

        lat = sum(item[0] for item in coords) / len(coords)
        lon = sum(item[1] for item in coords) / len(coords)
        return (lat, lon)

    @staticmethod
    def _directed_containment(DG, cluster_set):
        internal = 0.0
        out_origin = 0.0
        out_dest = 0.0

        for node in cluster_set:
            if node not in DG:
                continue

            internal += float(DG.nodes[node].get('self_weight', 0) or 0)

            for _, v, data in DG.out_edges(node, data=True):
                if v == node:
                    continue
                w = float(data.get('weight', 0) or 0)
                if v in cluster_set:
                    internal += w
                else:
                    out_origin += w

            for u, _, data in DG.in_edges(node, data=True):
                if u == node or u in cluster_set:
                    continue
                out_dest += float(data.get('weight', 0) or 0)

        orig_total = internal + out_origin
        dest_total = internal + out_dest
        origin_containment = internal / orig_total if orig_total > 0 else 0.0
        destination_containment = internal / dest_total if dest_total > 0 else 0.0
        zone_containment = min(origin_containment, destination_containment)

        return {
            'internal': internal,
            'out_origin': out_origin,
            'out_dest': out_dest,
            'origin': origin_containment,
            'destination': destination_containment,
            'zone': zone_containment,
        }

    @staticmethod
    def _within_local_radius(seed_centroid, candidate_cbg, cbg_centers, local_radius_km):
        if seed_centroid is None or not cbg_centers:
            return True

        try:
            radius_km = float(local_radius_km)
        except (TypeError, ValueError):
            radius_km = 0.0

        if radius_km <= 0:
            return True

        candidate_center = cbg_centers.get(candidate_cbg)
        if candidate_center is None:
            return False

        try:
            dist_km = float(distance(
                seed_centroid[0],
                seed_centroid[1],
                candidate_center[0],
                candidate_center[1],
            ))
        except Exception:
            return False

        return dist_km <= radius_km

    @staticmethod
    def _node_total_directed_flow(DG, node):
        if node not in DG:
            return 0.0

        total = float(DG.nodes[node].get('self_weight', 0) or 0)
        for _, v, data in DG.out_edges(node, data=True):
            if v == node:
                continue
            total += float(data.get('weight', 0) or 0)
        for u, _, data in DG.in_edges(node, data=True):
            if u == node:
                continue
            total += float(data.get('weight', 0) or 0)
        return total

