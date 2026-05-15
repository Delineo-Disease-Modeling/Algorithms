import math

import networkx as nx

from .metrics import Helpers, cbg_population


class MobilityPruneMixin:
    def mobility_prune(
        self,
        G: nx.Graph,
        seed_cbgs,
        min_pop: int,
        max_iter: int = 1000,
        trace_collector=None,
        envelope_population_multiplier: float = 2.0,
        envelope_population_floor: int = 100000,
        envelope_max_cbgs: int = 250,
        min_seed_capture: float = 0.80,
        trace_candidate_limit: int = 50,
    ):
        seed_cluster = []
        missing_seed_cbgs = []
        seen = set()
        for cbg in seed_cbgs or []:
            if cbg in seen:
                continue
            seen.add(cbg)
            if cbg in G:
                seed_cluster.append(cbg)
            else:
                missing_seed_cbgs.append(cbg)

        if not seed_cluster:
            raise ValueError("None of the seed region CBGs are present in the mobility graph")

        self.logger.info(
            "Starting mobility_prune with %d seed CBGs",
            len(seed_cluster),
        )

        seed_set = set(seed_cluster)
        cluster = list(seed_cluster)
        cluster_set = set(cluster)
        population_by_cbg = {}

        def get_population(cbg):
            if cbg not in population_by_cbg:
                population_by_cbg[cbg] = max(
                    0,
                    int(cbg_population(cbg, self.config, self.logger) or 0),
                )
            return population_by_cbg[cbg]

        population = sum(get_population(cbg) for cbg in cluster)
        seed_population = int(population)

        try:
            legacy_min_population = max(0, int(min_pop))
        except (TypeError, ValueError):
            legacy_min_population = 0

        try:
            multiplier = float(envelope_population_multiplier)
        except (TypeError, ValueError):
            multiplier = 2.0
        if multiplier < 1.0:
            multiplier = 1.0

        try:
            population_floor = max(0, int(envelope_population_floor))
        except (TypeError, ValueError):
            population_floor = 100000

        envelope_population_target = max(
            int(population),
            int(math.ceil(population * multiplier)),
            int(population_floor),
        )

        try:
            max_envelope_cbgs = int(envelope_max_cbgs)
        except (TypeError, ValueError):
            max_envelope_cbgs = 250
        if max_envelope_cbgs <= 0:
            max_envelope_cbgs = 250

        try:
            max_trace_candidates = int(trace_candidate_limit)
        except (TypeError, ValueError):
            max_trace_candidates = 50
        if max_trace_candidates <= 0:
            max_trace_candidates = 50

        try:
            min_seed_capture_threshold = float(min_seed_capture)
        except (TypeError, ValueError):
            min_seed_capture_threshold = 0.80
        min_seed_capture_threshold = min(1.0, max(0.0, min_seed_capture_threshold))

        seed_movement_total = 0.0
        seed_movement_always_captured = 0.0
        seed_movement_by_cbg = {}
        counted_seed_edges = set()
        for seed in seed_set:
            self_weight = float(G.nodes[seed].get('self_weight', 0) or 0)
            seed_movement_total += self_weight
            seed_movement_always_captured += self_weight
            for neighbor in G.adj[seed]:
                edge_key = frozenset((seed, neighbor))
                if edge_key in counted_seed_edges:
                    continue
                counted_seed_edges.add(edge_key)
                weight = float(G.adj[seed][neighbor].get('weight', 0) or 0)
                seed_movement_total += weight
                if neighbor in seed_set:
                    seed_movement_always_captured += weight
                else:
                    seed_movement_by_cbg[neighbor] = (
                        seed_movement_by_cbg.get(neighbor, 0.0) + weight
                    )

        seed_movement_captured = seed_movement_always_captured

        def seed_capture_share(captured):
            return captured / seed_movement_total if seed_movement_total > 0 else 1.0

        def movement_to_cluster(candidate):
            total = 0.0
            for neighbor in G.adj[candidate]:
                if neighbor in cluster_set:
                    total += float(G.adj[candidate][neighbor].get('weight', 0) or 0)
            return total

        def cap_trace_candidates(candidates, selected_cbg, higher_score_better=True):
            if trace_collector is None or len(candidates) <= max_trace_candidates:
                return candidates

            sorted_candidates = sorted(
                candidates,
                key=lambda item: (float(item.get('score', 0.0)), item.get('cbg', '')),
                reverse=bool(higher_score_better),
            )
            capped = sorted_candidates[:max_trace_candidates]
            if selected_cbg and not any(item.get('cbg') == selected_cbg for item in capped):
                selected = next(
                    (item for item in sorted_candidates if item.get('cbg') == selected_cbg),
                    None,
                )
                if selected is not None:
                    capped.append(selected)
            return capped

        frontier = set()
        for seed in seed_cluster:
            frontier.update(neighbor for neighbor in G.adj[seed] if neighbor not in cluster_set)

        growth_iteration = 0
        envelope_limited_by_cbg_cap = False
        while (
            (
                population < envelope_population_target
                or seed_capture_share(seed_movement_captured) < min_seed_capture_threshold
            )
            and frontier
            and growth_iteration < max_iter
        ):
            if len(cluster_set) >= max_envelope_cbgs:
                envelope_limited_by_cbg_cap = True
                break

            best = None
            candidate_details = []
            stale_candidates = []
            needs_seed_capture = (
                seed_capture_share(seed_movement_captured) < min_seed_capture_threshold
            )
            for candidate in sorted(frontier):
                if candidate in cluster_set or candidate not in G:
                    stale_candidates.append(candidate)
                    continue

                candidate_pop = get_population(candidate)
                if candidate_pop <= 0:
                    continue

                candidate_movement = movement_to_cluster(candidate)
                if candidate_movement <= 0:
                    continue

                movement_outside = self._movement_outside_cluster(G, candidate, cluster_set)
                population_after = int(population + candidate_pop)
                envelope_overshoot = max(0, population_after - envelope_population_target)
                movement_per_person = candidate_movement / candidate_pop
                seed_movement_gain = float(seed_movement_by_cbg.get(candidate, 0.0))
                seed_capture_after = seed_movement_captured + seed_movement_gain
                seed_capture_share_after = seed_capture_share(seed_capture_after)
                seed_movement_gain_per_person = seed_movement_gain / candidate_pop
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(candidate_pop),
                    'score': (
                        float(seed_capture_share_after)
                        if needs_seed_capture
                        else float(candidate_movement)
                    ),
                    'movement_to_cluster': float(candidate_movement),
                    'movement_to_outside': float(movement_outside),
                    'movement_to_cluster_per_person': float(movement_per_person),
                    'seed_movement_gain': float(seed_movement_gain),
                    'seed_movement_gain_per_person': float(seed_movement_gain_per_person),
                    'seed_capture_after': float(seed_capture_share_after),
                    'population_after': int(population_after),
                    'envelope_overshoot': int(envelope_overshoot),
                })

                if needs_seed_capture:
                    candidate_key = (
                        seed_movement_gain,
                        seed_movement_gain_per_person,
                        candidate_movement,
                        movement_per_person,
                        -envelope_overshoot,
                        -movement_outside,
                        -candidate_pop,
                    )
                else:
                    candidate_key = (
                        candidate_movement,
                        movement_per_person,
                        -envelope_overshoot,
                        -movement_outside,
                        -candidate_pop,
                    )
                if best is None or candidate_key > best[0]:
                    best = (
                        candidate_key,
                        candidate,
                        candidate_pop,
                        candidate_movement,
                        movement_outside,
                        movement_per_person,
                        population_after,
                        envelope_overshoot,
                        seed_capture_after,
                        seed_capture_share_after,
                        seed_movement_gain,
                    )

            for candidate in stale_candidates:
                frontier.discard(candidate)

            if best is None:
                self.logger.info(
                    "No positive-population mobility frontier remains for bounded prune envelope."
                )
                break

            (
                _,
                selected_cbg,
                selected_pop,
                selected_movement,
                selected_movement_outside,
                selected_movement_per_person,
                population_after,
                selected_overshoot,
                selected_seed_capture_after,
                selected_seed_capture_share_after,
                selected_seed_movement_gain,
            ) = best
            prev_cluster = list(cluster)
            prev_population = int(population)

            frontier.discard(selected_cbg)
            cluster.append(selected_cbg)
            cluster_set.add(selected_cbg)
            population = int(population_after)
            seed_movement_captured = selected_seed_capture_after
            for neighbor in G.adj[selected_cbg]:
                if neighbor not in cluster_set:
                    frontier.add(neighbor)

            self._record_trace_step(
                trace_collector,
                iteration=growth_iteration,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=cap_trace_candidates(candidate_details, selected_cbg),
                selected_cbg=selected_cbg,
                selected_population=selected_pop,
                cluster_after=cluster,
                population_after=population,
                metrics_after={
                    'stage': 'bounded_envelope_growth',
                    'movement_to_cluster': float(selected_movement),
                    'movement_to_outside': float(selected_movement_outside),
                    'movement_to_cluster_per_person': float(selected_movement_per_person),
                    'envelope_population_target': int(envelope_population_target),
                    'envelope_overshoot': int(selected_overshoot),
                    'seed_movement_gain': float(selected_seed_movement_gain),
                    'seed_capture': float(selected_seed_capture_share_after),
                },
            )

            self.logger.info(
                "Envelope growth iteration %d: added %s pop=%d population=%d seed_capture=%.4f movement_to_cluster=%.2f",
                growth_iteration,
                selected_cbg,
                int(selected_pop),
                int(population),
                float(selected_seed_capture_share_after),
                float(selected_movement),
            )
            growth_iteration += 1

        if growth_iteration >= max_iter and population < envelope_population_target:
            self.logger.warning(
                "Bounded mobility-prune envelope hit max iterations before target: pop=%d target=%d",
                int(population),
                int(envelope_population_target),
            )

        cluster = sorted(cluster_set)
        initial_envelope_cbg_count = len(cluster_set)
        initial_envelope_population = int(population)

        try:
            full_stats = Helpers.calculate_movement_stats(G, cluster)
            movement_inside = float(full_stats.get('in', 0.0))
            movement_outside = float(full_stats.get('out', 0.0))
        except ValueError:
            movement_inside = 0.0
            movement_outside = 0.0

        initial_czi = (
            movement_inside / (movement_inside + movement_outside)
            if movement_inside + movement_outside > 0
            else 0.0
        )
        initial_movement_inside = float(movement_inside)
        initial_movement_outside = float(movement_outside)

        initial_seed_capture_share = seed_capture_share(seed_movement_captured)

        self.logger.info(
            "Initial bounded mobility-prune envelope: CBGs=%d population=%d target=%d seed_capture=%.4f CZI=%.4f movement_inside=%.2f movement_boundary=%.2f",
            len(cluster),
            int(population),
            int(envelope_population_target),
            float(initial_seed_capture_share),
            float(initial_czi),
            float(movement_inside),
            float(movement_outside),
        )

        def movement_effect_if_removed(candidate):
            self_weight = float(G.nodes[candidate].get('self_weight', 0) or 0)
            movement_to_remaining = 0.0
            movement_to_current_outside = 0.0
            for neighbor in G.adj[candidate]:
                weight = float(G.adj[candidate][neighbor].get('weight', 0) or 0)
                if neighbor in cluster_set:
                    movement_to_remaining += weight
                else:
                    movement_to_current_outside += weight

            movement_loss = self_weight + movement_to_remaining
            movement_inside_after = max(0.0, movement_inside - movement_loss)
            movement_outside_after = max(
                0.0,
                movement_outside - movement_to_current_outside + movement_to_remaining,
            )
            total_after = movement_inside_after + movement_outside_after
            czi_after = movement_inside_after / total_after if total_after > 0 else 0.0
            return (
                movement_loss,
                movement_to_current_outside,
                movement_inside_after,
                movement_outside_after,
                czi_after,
            )

        def remains_seed_connected(candidate):
            next_set = cluster_set - {candidate}
            if not next_set:
                return False
            remaining_seeds = seed_set & next_set
            if not remaining_seeds:
                return False

            seen_nodes = set()
            stack = list(remaining_seeds)
            while stack:
                node = stack.pop()
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                for neighbor in G.adj[node]:
                    if neighbor in next_set and neighbor not in seen_nodes:
                        stack.append(neighbor)
            return len(seen_nodes) == len(next_set)

        iteration = 0
        stopped_by_seed_capture = False
        while iteration < max_iter:
            best = None
            candidate_details = []
            current_seed_capture_share = (
                seed_movement_captured / seed_movement_total
                if seed_movement_total > 0
                else 1.0
            )

            for candidate in sorted(cluster_set):
                if candidate in seed_set:
                    continue

                candidate_pop = population_by_cbg.get(candidate, 0)
                if candidate_pop <= 0:
                    continue

                population_after = int(population - candidate_pop)

                if not remains_seed_connected(candidate):
                    continue

                (
                    movement_loss,
                    movement_to_current_outside,
                    movement_inside_after,
                    movement_outside_after,
                    czi_after,
                ) = movement_effect_if_removed(candidate)
                movement_loss_per_person = movement_loss / candidate_pop
                seed_movement_loss = float(seed_movement_by_cbg.get(candidate, 0.0))
                seed_capture_after = max(0.0, seed_movement_captured - seed_movement_loss)
                seed_capture_share_after = (
                    seed_capture_after / seed_movement_total
                    if seed_movement_total > 0
                    else 1.0
                )
                seed_movement_loss_per_person = seed_movement_loss / candidate_pop
                would_violate_seed_capture = (
                    seed_capture_share_after < min_seed_capture_threshold
                    and seed_movement_loss > 1e-12
                )

                candidate_details.append({
                    'cbg': candidate,
                    'population': int(candidate_pop),
                    'score': float(seed_capture_share_after),
                    'movement_loss': float(movement_loss),
                    'movement_loss_per_person': float(movement_loss_per_person),
                    'movement_to_current_outside': float(movement_to_current_outside),
                    'movement_inside_after': float(movement_inside_after),
                    'movement_boundary_after': float(movement_outside_after),
                    'seed_movement_loss': float(seed_movement_loss),
                    'seed_movement_loss_per_person': float(seed_movement_loss_per_person),
                    'seed_movement_captured_after': float(seed_capture_after),
                    'seed_capture_after': float(seed_capture_share_after),
                    'current_seed_capture': float(current_seed_capture_share),
                    'czi_after': float(czi_after),
                    'would_violate_min_seed_capture': bool(would_violate_seed_capture),
                    'population_after': int(population_after),
                })

                if would_violate_seed_capture:
                    continue

                candidate_key = (
                    seed_movement_loss_per_person,
                    seed_movement_loss,
                    movement_loss_per_person,
                    movement_loss,
                    -candidate_pop,
                    candidate,
                )
                if best is None or candidate_key < best[0]:
                    best = (
                        candidate_key,
                        candidate,
                        candidate_pop,
                        movement_loss,
                        movement_inside_after,
                        movement_outside_after,
                        czi_after,
                        seed_movement_loss,
                        seed_capture_after,
                        seed_capture_share_after,
                    )

            if best is None:
                if candidate_details and all(
                    bool(candidate.get('would_violate_min_seed_capture'))
                    for candidate in candidate_details
                ):
                    stopped_by_seed_capture = True
                    self.logger.info(
                        "Stopping mobility_prune before crossing minimum seed capture %.4f",
                        float(min_seed_capture_threshold),
                    )
                    break
                self.logger.info(
                    "No removable non-seed CBG remains without disconnecting the retained zone."
                )
                break

            (
                _,
                selected_cbg,
                removed_pop,
                movement_loss,
                movement_inside_after,
                movement_outside_after,
                czi_after,
                seed_movement_loss,
                seed_capture_after,
                seed_capture_share_after,
            ) = best
            prev_cluster = list(cluster)
            prev_population = int(population)

            cluster_set.remove(selected_cbg)
            cluster = [cbg for cbg in cluster if cbg != selected_cbg]
            population -= int(removed_pop)
            movement_inside = movement_inside_after
            movement_outside = movement_outside_after
            seed_movement_captured = seed_capture_after

            self._record_trace_step(
                trace_collector,
                iteration=iteration,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=cap_trace_candidates(
                    candidate_details,
                    selected_cbg,
                    higher_score_better=True,
                ),
                selected_cbg=selected_cbg,
                selected_population=removed_pop,
                cluster_after=cluster,
                population_after=population,
                metrics_after={
                    'stage': 'reverse_prune',
                    'removed_population': int(removed_pop),
                    'movement_loss': float(movement_loss),
                    'movement_inside': float(movement_inside),
                    'movement_boundary': float(movement_outside),
                    'seed_movement_loss': float(seed_movement_loss),
                    'seed_movement_captured': float(seed_movement_captured),
                    'seed_capture': float(seed_capture_share_after),
                    'czi': float(czi_after),
                },
                higher_score_better=True,
            )

            self.logger.info(
                "Prune iteration %d: removed %s pop=%d population=%d seed_capture=%.4f CZI=%.4f",
                iteration,
                selected_cbg,
                int(removed_pop),
                int(population),
                float(seed_capture_share_after),
                float(czi_after),
            )
            iteration += 1

        final_czi = (
            movement_inside / (movement_inside + movement_outside)
            if movement_inside + movement_outside > 0
            else 0.0
        )
        final_seed_capture_share = (
            seed_movement_captured / seed_movement_total
            if seed_movement_total > 0
            else 1.0
        )
        metadata = {
            'seed_cbgs': list(seed_cluster),
            'missing_seed_cbgs': list(missing_seed_cbgs),
            'seed_population': int(seed_population),
            'bounded_envelope': True,
            'envelope_population_target': int(envelope_population_target),
            'envelope_population_multiplier': float(multiplier),
            'envelope_population_floor': int(population_floor),
            'envelope_max_cbgs': int(max_envelope_cbgs),
            'min_seed_capture': float(min_seed_capture_threshold),
            'envelope_growth_iterations': int(growth_iteration),
            'envelope_limited_by_cbg_cap': bool(envelope_limited_by_cbg_cap),
            'stopped_by_seed_capture_floor': bool(stopped_by_seed_capture),
            'initial_cbg_count': int(initial_envelope_cbg_count),
            'initial_population': int(initial_envelope_population),
            'initial_movement_inside': float(initial_movement_inside),
            'initial_movement_boundary': float(initial_movement_outside),
            'initial_czi': float(initial_czi),
            'seed_movement_total': float(seed_movement_total),
            'initial_seed_movement_captured': float(
                initial_seed_capture_share * seed_movement_total
            ),
            'initial_seed_capture_share': float(initial_seed_capture_share),
            'final_seed_movement_captured': float(seed_movement_captured),
            'final_seed_capture_share': float(final_seed_capture_share),
            'final_movement_inside': float(movement_inside),
            'final_movement_boundary': float(movement_outside),
            'final_czi': float(final_czi),
            'minimum_population_used': False,
            'legacy_min_population': int(legacy_min_population),
            'population_target_met': True,
            'population_reduced': int(initial_envelope_population - population),
            'removed_cbg_count': int(initial_envelope_cbg_count - len(cluster_set)),
        }

        return cluster, int(population), metadata

