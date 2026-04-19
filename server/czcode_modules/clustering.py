import math

import networkx as nx
import numpy as np

from common_geo import distance

from .metrics import Helpers, cbg_population

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    SCIPY_MILP_AVAILABLE = True
except Exception:
    SCIPY_MILP_AVAILABLE = False


class Clustering:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

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
        metrics_after=None
    ):
        if trace_collector is None:
            return

        sorted_candidates = sorted(
            candidates,
            key=lambda item: (float(item.get('score', 0.0)), item.get('cbg', '')),
            reverse=True
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

    def greedy_ttwa(
        self,
        DG,
        u0,
        min_pop,
        containment_threshold=0.70,
        max_iter=1000,
        trace_collector=None,
    ):
        self.logger.info(
            f"Starting greedy_ttwa with seed CBG {u0} "
            f"(threshold={containment_threshold}, min_pop={min_pop})"
        )

        if u0 not in DG:
            raise ValueError(f"Seed CBG {u0} not present in directed mobility graph")

        def _self_weight(node):
            return float(DG.nodes[node].get('self_weight', 0) or 0)

        cluster = [u0]
        cluster_set = {u0}
        population = cbg_population(u0, self.config, self.logger)

        internal = _self_weight(u0)
        out_origin = 0.0
        out_dest = 0.0
        for _, v, data in DG.out_edges(u0, data=True):
            if v == u0:
                continue
            out_origin += float(data.get('weight', 0))
        for u, _, data in DG.in_edges(u0, data=True):
            if u == u0:
                continue
            out_dest += float(data.get('weight', 0))

        def _containment(internal_, out_origin_, out_dest_):
            orig_total = internal_ + out_origin_
            dest_total = internal_ + out_dest_
            orig_c = internal_ / orig_total if orig_total > 0 else 0.0
            dest_c = internal_ / dest_total if dest_total > 0 else 0.0
            return orig_c, dest_c, min(orig_c, dest_c)

        orig_c, dest_c, zone_c = _containment(internal, out_origin, out_dest)
        self.logger.info(
            f"Seed containment: origin={orig_c:.4f} dest={dest_c:.4f} "
            f"zone={zone_c:.4f} pop={population}"
        )

        it = 1
        while it <= max_iter:
            candidates = set()
            for node in cluster:
                candidates.update(DG.successors(node))
                candidates.update(DG.predecessors(node))
            candidates -= cluster_set
            if not candidates:
                self.logger.info("No further candidates; stopping.")
                break

            best = None
            best_score = None
            best_deltas = None
            candidate_details = []
            for cand in candidates:
                delta_internal = _self_weight(cand)
                delta_out_origin = 0.0
                delta_out_dest = 0.0

                for _, y, data in DG.out_edges(cand, data=True):
                    if y == cand:
                        continue
                    w = float(data.get('weight', 0))
                    if y in cluster_set:
                        delta_internal += w
                        delta_out_dest -= w
                    else:
                        delta_out_origin += w

                for y, _, data in DG.in_edges(cand, data=True):
                    if y == cand:
                        continue
                    w = float(data.get('weight', 0))
                    if y in cluster_set:
                        delta_internal += w
                        delta_out_origin -= w
                    else:
                        delta_out_dest += w

                new_internal = internal + delta_internal
                new_out_origin = out_origin + delta_out_origin
                new_out_dest = out_dest + delta_out_dest
                _, _, new_zone_c = _containment(new_internal, new_out_origin, new_out_dest)
                score = new_zone_c - zone_c

                candidate_details.append({
                    'cbg': cand,
                    'population': int(cbg_population(cand, self.config, self.logger)),
                    'score': float(score),
                    'new_zone_containment': float(new_zone_c),
                    'delta_internal': float(delta_internal),
                    'delta_out_origin': float(delta_out_origin),
                    'delta_out_dest': float(delta_out_dest),
                })

                if (
                    best_score is None
                    or score > best_score
                    or (score == best_score and (best is None or cand < best))
                ):
                    best_score = score
                    best = cand
                    best_deltas = (delta_internal, delta_out_origin, delta_out_dest)

            if best is None:
                break

            if population >= min_pop and best_score <= 0:
                self.logger.info(
                    f"No candidate improves zone containment (best Δ={best_score:.4g}); "
                    f"stopping with zone_c={zone_c:.4f} pop={population}"
                )
                break

            chosen_pop = cbg_population(best, self.config, self.logger)
            prev_cluster = list(cluster)
            prev_population = population

            cluster.append(best)
            cluster_set.add(best)
            d_int, d_oo, d_od = best_deltas
            internal += d_int
            out_origin += d_oo
            out_dest += d_od
            orig_c, dest_c, zone_c = _containment(internal, out_origin, out_dest)
            population += chosen_pop

            self._record_trace_step(
                trace_collector,
                iteration=it - 1,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=best,
                selected_population=chosen_pop,
                cluster_after=cluster,
                population_after=population,
            )

            self.logger.info(
                f"Iteration {it}: added {best} (pop {chosen_pop}); "
                f"zone_c={zone_c:.4f} (orig={orig_c:.4f}, dest={dest_c:.4f}) "
                f"pop={population}"
            )
            it += 1

        if it > max_iter:
            self.logger.warning(f"Reached max_iter={max_iter}; stopping.")

        if zone_c < containment_threshold:
            self.logger.warning(
                f"Final zone_c={zone_c:.4f} below threshold "
                f"{containment_threshold}; zone may be weakly self-contained."
            )

        return cluster, population

    def _select_candidate_nodes(self, G: nx.Graph, u0: str, candidate_limit: int):
        if u0 not in G:
            return [u0]

        try:
            limit = int(candidate_limit)
        except (TypeError, ValueError):
            limit = 120
        limit = max(10, limit)

        selected = [u0]
        selected_set = {u0}
        frontier = set(G.adj[u0])

        while frontier and len(selected) < limit:
            best_node = None
            best_score = None

            for node in list(frontier):
                if node in selected_set:
                    continue

                flow_to_selected = 0.0
                for nb in G.adj[node]:
                    if nb in selected_set:
                        flow_to_selected += float(G.adj[node][nb].get('weight', 0))

                self_w = float(G.nodes[node].get('self_weight', 0))
                deg = float(G.degree[node])
                score = (flow_to_selected, self_w, deg, node)

                if best_score is None or score > best_score:
                    best_score = score
                    best_node = node

            if best_node is None:
                break

            frontier.discard(best_node)
            if best_node in selected_set:
                continue

            selected.append(best_node)
            selected_set.add(best_node)
            for nb in G.adj[best_node]:
                if nb not in selected_set:
                    frontier.add(nb)

        return selected

    def _optimize_czi_subproblem(
        self,
        lam: float,
        nodes,
        edges,
        edge_weights,
        directed_arcs,
        node_to_idx,
        populations,
        self_weights,
        external_boundary,
        seed_idx,
        pop_floor: int,
        pop_cap: int,
        mip_rel_gap: float,
        time_limit_sec: float
    ):
        n = len(nodes)
        m = len(edges)
        d = len(directed_arcs)
        if n == 0:
            return None

        x_start = 0
        y_start = n
        z_start = n + m
        f_start = n + (2 * m)
        num_vars = n + (2 * m) + d
        M = max(1, n - 1)

        c = np.zeros(num_vars, dtype=float)
        for node, idx in node_to_idx.items():
            coeff = ((1.0 - lam) * float(self_weights.get(node, 0.0))) - (
                lam * float(external_boundary.get(node, 0.0))
            )
            c[x_start + idx] = -coeff

        for e_idx, w in enumerate(edge_weights):
            c[y_start + e_idx] = -((1.0 - lam) * float(w))
            c[z_start + e_idx] = lam * float(w)

        rows = []
        cols = []
        vals = []
        lb = []
        ub = []
        row_idx = 0

        def add_row(coeffs, row_lb=-np.inf, row_ub=np.inf):
            nonlocal row_idx
            for col, val in coeffs.items():
                if val == 0:
                    continue
                rows.append(row_idx)
                cols.append(col)
                vals.append(float(val))
            lb.append(float(row_lb))
            ub.append(float(row_ub))
            row_idx += 1

        for e_idx, (u, v) in enumerate(edges):
            xu = x_start + node_to_idx[u]
            xv = x_start + node_to_idx[v]
            ye = y_start + e_idx
            add_row({ye: 1, xu: -1}, row_ub=0)
            add_row({ye: 1, xv: -1}, row_ub=0)
            add_row({ye: -1, xu: 1, xv: 1}, row_ub=1)

        for e_idx, (u, v) in enumerate(edges):
            xu = x_start + node_to_idx[u]
            xv = x_start + node_to_idx[v]
            ze = z_start + e_idx
            add_row({ze: 1, xu: -1, xv: 1}, row_lb=0)
            add_row({ze: 1, xu: 1, xv: -1}, row_lb=0)
            add_row({ze: 1, xu: -1, xv: -1}, row_ub=0)
            add_row({ze: 1, xu: 1, xv: 1}, row_ub=2)

        pop_coeff = {}
        for node, idx in node_to_idx.items():
            pop_coeff[x_start + idx] = float(populations.get(node, 0))
        add_row(pop_coeff, row_lb=float(pop_floor), row_ub=float(pop_cap))

        add_row({x_start + seed_idx: 1}, row_lb=1, row_ub=1)

        out_arcs = {node: [] for node in nodes}
        in_arcs = {node: [] for node in nodes}
        for a_idx, (u, v) in enumerate(directed_arcs):
            fa = f_start + a_idx
            xu = x_start + node_to_idx[u]
            xv = x_start + node_to_idx[v]
            add_row({fa: 1, xu: -M}, row_ub=0)
            add_row({fa: 1, xv: -M}, row_ub=0)
            out_arcs[u].append(a_idx)
            in_arcs[v].append(a_idx)

        seed_node = nodes[seed_idx]

        for node in nodes:
            if node == seed_node:
                continue

            coeff = {x_start + node_to_idx[node]: -1}
            for a_idx in in_arcs[node]:
                coeff[f_start + a_idx] = coeff.get(f_start + a_idx, 0) + 1
            for a_idx in out_arcs[node]:
                coeff[f_start + a_idx] = coeff.get(f_start + a_idx, 0) - 1
            add_row(coeff, row_lb=0, row_ub=0)

        seed_coeff = {}
        for a_idx in out_arcs[seed_node]:
            seed_coeff[f_start + a_idx] = seed_coeff.get(f_start + a_idx, 0) + 1
        for a_idx in in_arcs[seed_node]:
            seed_coeff[f_start + a_idx] = seed_coeff.get(f_start + a_idx, 0) - 1
        for node in nodes:
            if node == seed_node:
                continue
            seed_coeff[x_start + node_to_idx[node]] = -1
        add_row(seed_coeff, row_lb=0, row_ub=0)

        A = coo_matrix((vals, (rows, cols)), shape=(row_idx, num_vars)).tocsr()
        constraints = LinearConstraint(A, np.array(lb), np.array(ub))

        lower = np.zeros(num_vars, dtype=float)
        upper = np.concatenate([
            np.ones(n + (2 * m), dtype=float),
            np.full(d, float(M), dtype=float),
        ])
        bounds = Bounds(lower, upper)

        integrality = np.zeros(num_vars, dtype=int)
        integrality[: n + (2 * m)] = 1

        options = {
            'disp': False,
            'mip_rel_gap': float(max(0.0, mip_rel_gap)),
            'time_limit': float(max(1.0, time_limit_sec)),
        }

        try:
            return milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options=options,
            )
        except Exception:
            self.logger.exception("MILP subproblem failed unexpectedly")
            return None

    def czi_optimal_cap(
        self,
        G: nx.Graph,
        u0: str,
        max_pop: int,
        candidate_limit: int = 120,
        population_floor_ratio: float = 0.9,
        mip_rel_gap: float = 0.02,
        time_limit_sec: float = 20.0,
        max_dinkelbach_iters: int = 8
    ):
        if u0 not in G:
            pop = cbg_population(u0, self.config, self.logger)
            self.logger.warning(f"Seed CBG {u0} not found in graph for czi_optimal_cap")
            return [u0], pop

        if not SCIPY_MILP_AVAILABLE:
            self.logger.warning(
                "SciPy MILP is unavailable; falling back to greedy_czi_balanced."
            )
            return self.greedy_czi_balanced(G, u0, max_pop)

        try:
            pop_cap = int(max_pop)
        except (TypeError, ValueError):
            pop_cap = 0
        seed_pop = cbg_population(u0, self.config, self.logger)
        pop_cap = max(pop_cap, seed_pop)

        floor_ratio = float(population_floor_ratio)
        floor_ratio = min(1.0, max(0.0, floor_ratio))
        pop_floor = max(seed_pop, int(math.floor(pop_cap * floor_ratio)))

        candidates = self._select_candidate_nodes(G, u0, candidate_limit)
        node_set = set(candidates)
        if u0 not in node_set:
            candidates = [u0] + candidates
            node_set = set(candidates)

        populations = {}
        self_weights = {}
        external_boundary = {}
        for node in candidates:
            populations[node] = max(0, int(cbg_population(node, self.config, self.logger)))
            self_weights[node] = float(G.nodes[node].get('self_weight', 0.0))

            boundary_w = 0.0
            for nb in G.adj[node]:
                w = float(G.adj[node][nb].get('weight', 0))
                if w <= 0:
                    continue
                if nb not in node_set:
                    boundary_w += w
            external_boundary[node] = boundary_w

        edges = []
        edge_weights = []
        seen_edges = set()
        for u in candidates:
            for v in G.adj[u]:
                if v not in node_set or u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                w = float(G.adj[u][v].get('weight', 0))
                if w <= 0:
                    continue
                edges.append(key)
                edge_weights.append(w)

        directed_arcs = []
        for u, v in edges:
            directed_arcs.append((u, v))
            directed_arcs.append((v, u))

        node_to_idx = {node: i for i, node in enumerate(candidates)}
        seed_idx = node_to_idx[u0]

        total_candidate_pop = sum(populations.values())
        pop_floor = min(pop_floor, total_candidate_pop)
        pop_cap = min(pop_cap, total_candidate_pop)
        if pop_floor > pop_cap:
            pop_floor = pop_cap

        if pop_cap <= 0:
            self.logger.warning(
                "Candidate graph has zero population for czi_optimal_cap; falling back to greedy."
            )
            return self.greedy_czi_balanced(G, u0, max_pop)

        self.logger.info(
            "Starting czi_optimal_cap with %d candidates, cap=%d, floor=%d",
            len(candidates), pop_cap, pop_floor
        )

        best_cluster = None
        best_score = -1.0
        lam = 0.5
        dinkelbach_iters = max(1, int(max_dinkelbach_iters))

        floor_attempts = [pop_floor]
        while floor_attempts[-1] > seed_pop:
            next_floor = max(seed_pop, int(math.floor(floor_attempts[-1] * 0.9)))
            if next_floor == floor_attempts[-1]:
                break
            floor_attempts.append(next_floor)
        if seed_pop not in floor_attempts:
            floor_attempts.append(seed_pop)

        for floor_idx, attempt_floor in enumerate(floor_attempts):
            attempt_found = False
            current_lam = lam

            if floor_idx > 0:
                self.logger.info(
                    "Relaxing population floor to %d for czi_optimal_cap",
                    attempt_floor
                )

            for it in range(dinkelbach_iters):
                result = self._optimize_czi_subproblem(
                    lam=current_lam,
                    nodes=candidates,
                    edges=edges,
                    edge_weights=edge_weights,
                    directed_arcs=directed_arcs,
                    node_to_idx=node_to_idx,
                    populations=populations,
                    self_weights=self_weights,
                    external_boundary=external_boundary,
                    seed_idx=seed_idx,
                    pop_floor=attempt_floor,
                    pop_cap=pop_cap,
                    mip_rel_gap=mip_rel_gap,
                    time_limit_sec=time_limit_sec,
                )

                if result is None or result.x is None:
                    self.logger.warning(
                        "MILP solve returned no solution at Dinkelbach iteration %d (floor=%d)",
                        it,
                        attempt_floor
                    )
                    break

                x = result.x[: len(candidates)]
                chosen = [node for node, idx in node_to_idx.items() if x[idx] >= 0.5]
                if u0 not in chosen:
                    chosen.append(u0)

                chosen_set = set(chosen)
                chosen = [node for node in candidates if node in chosen_set]
                if not chosen:
                    break

                try:
                    stats = Helpers.calculate_movement_stats(G, chosen)
                except ValueError:
                    self.logger.warning(
                        "Candidate solution had nodes missing from graph; falling back."
                    )
                    break

                inside = float(stats.get('in', 0.0))
                boundary = float(stats.get('out', 0.0))
                denom = inside + boundary
                czi = (inside / denom) if denom > 0 else 0.0
                residual = inside - current_lam * denom

                chosen_pop = sum(populations.get(node, 0) for node in chosen)
                if czi > best_score or (
                    abs(czi - best_score) < 1e-9 and best_cluster is not None and chosen_pop > sum(populations.get(node, 0) for node in best_cluster)
                ):
                    best_score = czi
                    best_cluster = chosen

                self.logger.info(
                    "Dinkelbach iter %d (floor=%d): CZI=%.5f, inside=%.1f, boundary=%.1f, pop=%d, status=%s",
                    it, attempt_floor, czi, inside, boundary, chosen_pop, str(result.status)
                )

                attempt_found = True
                next_lam = czi
                if abs(next_lam - current_lam) <= 1e-4 or abs(residual) <= 1e-3:
                    current_lam = next_lam
                    break
                current_lam = next_lam

            if attempt_found:
                lam = current_lam
                break

        if not best_cluster:
            self.logger.warning(
                "czi_optimal_cap could not produce a valid MILP cluster; using greedy_czi_balanced fallback."
            )
            return self.greedy_czi_balanced(G, u0, max_pop)

        best_population = sum(cbg_population(node, self.config, self.logger) for node in best_cluster)
        self.logger.info(
            "czi_optimal_cap selected %d CBGs, population=%d, CZI=%.5f",
            len(best_cluster), best_population, best_score
        )
        return best_cluster, best_population

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
