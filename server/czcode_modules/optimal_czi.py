import math

import networkx as nx
import numpy as np

from .metrics import Helpers, cbg_population

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    SCIPY_MILP_AVAILABLE = True
except Exception:
    SCIPY_MILP_AVAILABLE = False


class OptimalCziMixin:
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

