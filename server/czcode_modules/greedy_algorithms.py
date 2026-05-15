import networkx as nx

from common_geo import distance

from .metrics import Helpers, cbg_population


class GreedyClusteringMixin:
    def greedy_fast(self, G: nx.Graph, u0: str, min_pop: int, trace_collector=None):
        self.logger.info(f"Starting greedy_fast algorithm with seed CBG {u0}")
        population = cbg_population(u0, self.config, self.logger)

        cluster = [u0]
        cluster_set = {u0}
        surround = list(set([j for j in list(G.adj[u0]) if j not in cluster_set]))

        for j in list(G.adj[u0]):
            if j not in surround and j not in cluster_set:
                surround.append(j)

        if len(surround) == 0:
            self.logger.warning("No adjacent CBGs found. Cannot reach target population.")
            return cluster, population

        itr = 0

        while population < min_pop:
            max_weight = 0
            best_cbg = surround[0]
            best_pop = 0
            candidate_details = []

            for candidate in surround:
                if candidate in cluster_set:
                    continue

                cur_pop = cbg_population(candidate, self.config, self.logger)
                if cur_pop == 0:
                    continue

                weight = sum([G.get_edge_data(candidate, cbg, {}).get('weight', 0) for cbg in cluster])
                weight = float(weight)
                movement_outside = self._movement_outside_cluster(G, candidate, cluster_set)
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(cur_pop),
                    'score': weight,
                    'movement_to_cluster': weight,
                    'movement_to_outside': movement_outside,
                })

                if weight > max_weight:
                    max_weight = weight
                    best_cbg = candidate
                    best_pop = cur_pop

            prev_cluster = list(cluster)
            prev_population = population

            surround.remove(best_cbg)
            cluster.append(best_cbg)
            cluster_set.add(best_cbg)
            population += best_pop

            self._record_trace_step(
                trace_collector,
                iteration=itr,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=best_cbg,
                selected_population=best_pop,
                cluster_after=cluster,
                population_after=population
            )

            self.logger.info(f"Iteration {itr}: Added CBG {best_cbg} with pop {best_pop}. New total: {population}")

            surround.extend([j for j in list(G.adj[best_cbg]) if j not in cluster_set])
            surround = list(set(surround))

            itr += 1
            if itr > 500:
                self.logger.warning("Max iterations exceeded (500). Cannot reach target population.")
                break

        return cluster, population

    def greedy_weight(self, G, u0, min_pop, trace_collector=None):
        self.logger.info(f"Starting greedy_weight algorithm with seed CBG {u0}")
        cluster = [u0]
        cluster_set = {u0}
        population = cbg_population(u0, self.config, self.logger)
        it = 1
        self.logger.info(f"Seed CBG population: {population}")
        while population < min_pop:
            all_adj_cbgs = []
            for i in cluster:
                try:
                    for j in list(G.adj[i]):
                        if j not in all_adj_cbgs and j not in cluster_set:
                            all_adj_cbgs.append(j)
                except KeyError:
                    self.logger.warning(f"CBG {i} not found in graph")
                    continue
            if not all_adj_cbgs:
                self.logger.warning(f"No adjacent CBGs found after {it} iterations. Cannot reach target population.")
                break
            max_movement = 0
            cbg_to_add = all_adj_cbgs[0]
            candidate_details = []
            for candidate in all_adj_cbgs:
                current_movement = 0
                for member in cluster:
                    try:
                        current_movement += G.adj[candidate][member]['weight']
                    except (KeyError, ZeroDivisionError):
                        continue
                movement_to_cluster = float(current_movement)
                movement_to_outside = self._movement_outside_cluster(G, candidate, cluster_set)
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(cbg_population(candidate, self.config, self.logger)),
                    'score': movement_to_cluster,
                    'movement_to_cluster': movement_to_cluster,
                    'movement_to_outside': movement_to_outside,
                })
                if current_movement > max_movement:
                    max_movement = current_movement
                    cbg_to_add = candidate

            prev_cluster = list(cluster)
            prev_population = population
            cluster.append(cbg_to_add)
            cluster_set.add(cbg_to_add)
            cbg_pop = cbg_population(cbg_to_add, self.config, self.logger)
            population += cbg_pop

            self._record_trace_step(
                trace_collector,
                iteration=it - 1,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=cbg_to_add,
                selected_population=cbg_pop,
                cluster_after=cluster,
                population_after=population
            )

            self.logger.info(f"Iteration {it}: Added CBG {cbg_to_add} with pop {cbg_pop}. New total: {population}")
            it += 1
            if it > 1000:
                self.logger.warning("Reached maximum iterations (1000). Stopping algorithm.")
                break
        return cluster, population

    def greedy_weight_seed_guard(
        self,
        G,
        u0,
        min_pop,
        seed_guard_distance_km=20.0,
        cbg_centers=None,
        trace_collector=None,
    ):
        self.logger.info(
            "Starting greedy_weight_seed_guard algorithm with seed CBG %s (distance %.2f km)",
            u0,
            float(seed_guard_distance_km),
        )
        cluster = [u0]
        cluster_set = {u0}
        contributor_set = {u0}
        excluded_set = set()
        population = cbg_population(u0, self.config, self.logger)
        cbg_centers = cbg_centers or {}
        it = 1

        self.logger.info(f"Seed CBG population: {population}")
        while population < min_pop:
            all_adj_cbgs = []
            for i in cluster:
                try:
                    for j in list(G.adj[i]):
                        if j not in all_adj_cbgs and j not in cluster_set:
                            all_adj_cbgs.append(j)
                except KeyError:
                    self.logger.warning(f"CBG {i} not found in graph")
                    continue

            if not all_adj_cbgs:
                self.logger.warning(
                    f"No adjacent CBGs found after {it} iterations. Cannot reach target population."
                )
                break

            best_choice = None
            candidate_details = []
            for candidate in all_adj_cbgs:
                movement_to_cluster = 0.0
                movement_to_full_cluster = 0.0
                for member in cluster:
                    try:
                        edge_weight = float(G.adj[candidate][member]['weight'])
                    except (KeyError, ZeroDivisionError):
                        continue
                    movement_to_full_cluster += edge_weight
                    if member in contributor_set:
                        movement_to_cluster += edge_weight

                movement_to_outside = self._movement_outside_cluster(G, candidate, cluster_set)
                candidate_pop = int(cbg_population(candidate, self.config, self.logger))
                contributes_after, seed_distance = self._seed_guard_membership(
                    u0,
                    candidate,
                    cbg_centers,
                    seed_guard_distance_km,
                )
                distance_tiebreak = 0.0 if seed_distance is None else -float(seed_distance)
                candidate_tuple = (
                    float(movement_to_cluster),
                    int(contributes_after),
                    distance_tiebreak,
                    float(candidate_pop),
                    str(candidate),
                )
                candidate_details.append({
                    'cbg': candidate,
                    'population': candidate_pop,
                    'score': float(movement_to_cluster),
                    'movement_to_cluster': float(movement_to_cluster),
                    'movement_to_full_cluster': float(movement_to_full_cluster),
                    'movement_to_outside': float(movement_to_outside),
                    'seed_distance_km': (
                        float(seed_distance) if seed_distance is not None else None
                    ),
                    'movement_contributes_after_selection': bool(contributes_after),
                })
                if best_choice is None or candidate_tuple > best_choice[0]:
                    best_choice = (
                        candidate_tuple,
                        candidate,
                        candidate_pop,
                        contributes_after,
                        seed_distance,
                    )

            if best_choice is None:
                self.logger.warning(
                    f"No valid candidate CBGs found after {it} iterations. Cannot reach target population."
                )
                break

            _, cbg_to_add, cbg_pop, contributes_after, seed_distance = best_choice
            prev_cluster = list(cluster)
            prev_population = population
            cluster.append(cbg_to_add)
            cluster_set.add(cbg_to_add)
            population += cbg_pop

            if contributes_after:
                contributor_set.add(cbg_to_add)
            else:
                excluded_set.add(cbg_to_add)

            metrics_after = {
                'seed_guard_distance_km': float(seed_guard_distance_km),
                'movement_contributor_count': len(contributor_set),
                'movement_excluded_count': len(excluded_set),
            }
            if excluded_set:
                metrics_after['movement_excluded_cbgs'] = list(excluded_set)

            self._record_trace_step(
                trace_collector,
                iteration=it - 1,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=cbg_to_add,
                selected_population=cbg_pop,
                cluster_after=cluster,
                population_after=population,
                metrics_after=metrics_after,
            )

            movement_note = "counts toward future scoring"
            if not contributes_after:
                movement_note = "excluded from future scoring"
            self.logger.info(
                "Iteration %d: Added CBG %s with pop %d. New total: %d (%s, distance=%s km)",
                it,
                cbg_to_add,
                cbg_pop,
                population,
                movement_note,
                (
                    f"{float(seed_distance):.2f}"
                    if seed_distance is not None
                    else "unknown"
                ),
            )
            it += 1
            if it > 1000:
                self.logger.warning("Reached maximum iterations (1000). Stopping algorithm.")
                break
        return cluster, population

    def greedy_ratio(self, G, u0, min_pop, trace_collector=None):
        self.logger.info(f"Starting greedy_ratio algorithm with seed CBG {u0}")
        cluster = [u0]
        cluster_set = {u0}
        population = cbg_population(u0, self.config, self.logger)
        it = 1
        self.logger.info(f"Seed CBG population: {population}")
        while population < min_pop:
            all_adj_cbgs = []
            for i in cluster:
                try:
                    for j in list(G.adj[i]):
                        if j not in all_adj_cbgs and j not in cluster_set:
                            all_adj_cbgs.append(j)
                except KeyError:
                    self.logger.warning(f"CBG {i} not found in graph")
                    continue
            if not all_adj_cbgs:
                self.logger.warning(f"No adjacent CBGs found after {it} iterations. Cannot reach target population.")
                break
            max_ratio = 0
            cbg_to_add = all_adj_cbgs[0]
            candidate_details = []
            for candidate in all_adj_cbgs:
                movement_in = 0
                movement_out = 0
                for j in G.adj[candidate]:
                    if j in cluster:
                        movement_in += G.adj[candidate][j]['weight']
                    else:
                        movement_out += G.adj[candidate][j]['weight']
                total_movement = movement_in + movement_out
                ratio = 0.0
                if total_movement > 0:
                    ratio = movement_in / total_movement
                    if ratio > max_ratio:
                        max_ratio = ratio
                        cbg_to_add = candidate
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(cbg_population(candidate, self.config, self.logger)),
                    'score': float(ratio),
                    'movement_to_cluster': float(movement_in),
                    'movement_to_outside': float(movement_out),
                    'movement_total': float(total_movement),
                })

            prev_cluster = list(cluster)
            prev_population = population
            cluster.append(cbg_to_add)
            cluster_set.add(cbg_to_add)
            cbg_pop = cbg_population(cbg_to_add, self.config, self.logger)
            population += cbg_pop

            self._record_trace_step(
                trace_collector,
                iteration=it - 1,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=cbg_to_add,
                selected_population=cbg_pop,
                cluster_after=cluster,
                population_after=population
            )

            self.logger.info(f"Iteration {it}: Added CBG {cbg_to_add} with pop {cbg_pop}. New total: {population}")
            it += 1
            if it > 1000:
                self.logger.warning("Reached maximum iterations (1000). Stopping algorithm.")
                break
        return cluster, population
    def greedy_czi_balanced(self, G: nx.Graph, u0: str, min_pop: int,
                            alpha: float = 0.75, overshoot_penalty: float = 0.25,
                            distance_penalty_weight: float = 0.02,
                            distance_scale_km: float = 20.0,
                            cbg_centers=None,
                            trace_collector=None):
        self.logger.info(f"Starting greedy_czi_balanced algorithm with seed CBG {u0}")
        population = cbg_population(u0, self.config, self.logger)
        cluster = [u0]
        cluster_set = {u0}
        cbg_centers = cbg_centers or {}
        seed_center = cbg_centers.get(u0)

        if u0 not in G:
            self.logger.warning(f"Seed CBG {u0} not found in graph")
            return cluster, population

        base_stats = Helpers.calculate_movement_stats(G, cluster)
        movement_in = float(base_stats.get('in', 0))
        movement_out = float(base_stats.get('out', 0))
        surround = set(G.adj[u0])

        itr = 0
        while population < min_pop:
            if not surround:
                self.logger.warning(
                    f"No adjacent CBGs found after {itr} iterations. Cannot reach target population."
                )
                break

            best = None
            candidate_details = []

            for candidate in list(surround):
                if candidate in cluster_set or candidate not in G:
                    continue

                cand_pop = cbg_population(candidate, self.config, self.logger)
                if cand_pop <= 0:
                    continue

                self_weight = float(G.nodes[candidate].get('self_weight', 0))
                in_to_cluster = 0.0
                out_to_outside = 0.0
                for neighbor in G.adj[candidate]:
                    weight = float(G.adj[candidate][neighbor].get('weight', 0))
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
                if seed_center and distance_scale_km > 0:
                    cand_center = cbg_centers.get(candidate)
                    if cand_center:
                        dist_km = distance(
                            seed_center[0], seed_center[1],
                            cand_center[0], cand_center[1]
                        )
                        distance_penalty = dist_km / (dist_km + distance_scale_km)

                score = (
                    czi_after
                    - distance_penalty_weight * distance_penalty
                )

                candidate_tuple = (
                    score,
                    czi_after,
                    -distance_penalty,
                    in_to_cluster,
                    candidate,
                    cand_pop,
                    inside_after,
                    boundary_after
                )
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(cand_pop),
                    'score': float(score),
                    'movement_to_cluster': float(in_to_cluster),
                    'movement_to_outside': float(out_to_outside),
                    'czi_after': float(czi_after),
                    'distance_penalty': float(distance_penalty),
                    'movement_inside_after': float(inside_after),
                    'movement_boundary_after': float(boundary_after),
                })
                if best is None or candidate_tuple[:4] > best[:4]:
                    best = candidate_tuple

            if best is None:
                self.logger.warning(
                    f"No valid candidate CBGs found after {itr} iterations. Cannot reach target population."
                )
                break

            _, czi_after, _, _, best_cbg, best_pop, inside_after, boundary_after = best
            prev_cluster = list(cluster)
            prev_population = population
            cluster.append(best_cbg)
            cluster_set.add(best_cbg)
            population += best_pop
            movement_in = inside_after
            movement_out = boundary_after

            self._record_trace_step(
                trace_collector,
                iteration=itr,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=best_cbg,
                selected_population=best_pop,
                cluster_after=cluster,
                population_after=population,
                metrics_after={
                    'czi': float(czi_after),
                    'movement_inside': float(movement_in),
                    'movement_boundary': float(movement_out),
                }
            )

            surround.remove(best_cbg)
            surround.update([j for j in G.adj[best_cbg] if j not in cluster_set])

            self.logger.info(
                f"Iteration {itr}: Added CBG {best_cbg} with pop {best_pop}. "
                f"New total: {population}. CZI: {czi_after:.4f}"
            )

            itr += 1
            if itr > 500:
                self.logger.warning("Max iterations exceeded (500). Cannot reach target population.")
                break

        return cluster, population

