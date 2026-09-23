"""Seed prune: take every seed-linked CBG, then trim it.

This is the simpler convenience-zone rule described in the walkthrough deck
("Take every CBG linked to the seed, then trim it"). It sits next to
``mobility_prune`` and does not replace it.

Universe
    The seed CBGs present in the graph, plus every CBG with a direct edge to a
    seed and population > 0. This set holds all of the seed's movement except
    edges to zero-population CBGs, which could never hold simulated residents.
    Every non-seed member touches a seed, so the retained zone stays attached to
    the seed region whatever is removed; no connectivity check is needed.

Prune
    Repeatedly remove the non-seed CBG with the lowest seed movement per
    resident whose removal keeps seed capture at or above ``min_seed_capture``.
    A candidate whose removal would cross the floor is skipped, and the prune
    stops when no candidate can be removed. Seed CBGs are never removed.

Why one pass is enough
    A candidate's seed movement depends only on its own edges to the seeds, so
    its score never changes as other CBGs leave. Seed capture only falls as CBGs
    are removed, so a candidate that is blocked once stays blocked. The
    iterative rule is therefore identical to one ascending pass over a static
    ordering, which is O(n log n) instead of O(n^2).

Seed capture uses the same definition as ``mobility_prune``: captured = seed
self weight + seed-to-seed edges + seed edges to non-seed members; total = seed
self weight + every edge touching a seed.
"""

import networkx as nx

from .metrics import Helpers, cbg_population


SEED_PRUNE_TOLERANCE = 1e-12


def _edge_weight(G, a, b):
    return float(G.adj[a][b].get('weight', 0) or 0)


def seed_movement_accounting(G: nx.Graph, seed_set):
    """Return (total, always_captured, seed_movement_by_non_seed_cbg).

    Each undirected edge touching a seed is counted once, including seed-seed
    edges, which are always captured because seeds are never removed.
    """
    total = 0.0
    always_captured = 0.0
    by_cbg = {}
    counted_edges = set()
    for seed in seed_set:
        self_weight = float(G.nodes[seed].get('self_weight', 0) or 0)
        total += self_weight
        always_captured += self_weight
        for neighbor in G.adj[seed]:
            edge_key = frozenset((seed, neighbor))
            if edge_key in counted_edges:
                continue
            counted_edges.add(edge_key)
            weight = _edge_weight(G, seed, neighbor)
            total += weight
            if neighbor in seed_set:
                always_captured += weight
            else:
                by_cbg[neighbor] = by_cbg.get(neighbor, 0.0) + weight
    return total, always_captured, by_cbg


class SeedPruneMixin:
    def seed_prune(
        self,
        G: nx.Graph,
        seed_cbgs,
        min_pop: int,
        trace_collector=None,
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

        try:
            legacy_min_population = max(0, int(min_pop))
        except (TypeError, ValueError):
            legacy_min_population = 0

        try:
            threshold = float(min_seed_capture)
        except (TypeError, ValueError):
            threshold = 0.80
        threshold = min(1.0, max(0.0, threshold))

        try:
            max_trace_candidates = int(trace_candidate_limit)
        except (TypeError, ValueError):
            max_trace_candidates = 50
        if max_trace_candidates <= 0:
            max_trace_candidates = 50

        seed_set = set(seed_cluster)
        population_by_cbg = {}

        def get_population(cbg):
            if cbg not in population_by_cbg:
                population_by_cbg[cbg] = max(
                    0,
                    int(cbg_population(cbg, self.config, self.logger) or 0),
                )
            return population_by_cbg[cbg]

        seed_movement_total, seed_movement_always_captured, seed_movement_by_cbg = (
            seed_movement_accounting(G, seed_set)
        )

        def capture_share(captured):
            return captured / seed_movement_total if seed_movement_total > 0 else 1.0

        # Universe: seeds + every direct seed neighbour with residents.
        excluded_zero_population = []
        members = []
        for cbg in sorted(seed_movement_by_cbg):
            if get_population(cbg) > 0:
                members.append(cbg)
            else:
                excluded_zero_population.append(cbg)
        excluded_seed_movement = sum(
            seed_movement_by_cbg[cbg] for cbg in excluded_zero_population
        )

        cluster_set = set(seed_cluster) | set(members)
        seed_population = int(sum(get_population(cbg) for cbg in seed_cluster))
        population = int(seed_population + sum(population_by_cbg[cbg] for cbg in members))
        seed_movement_captured = seed_movement_always_captured + sum(
            seed_movement_by_cbg[cbg] for cbg in members
        )

        stats = Helpers.calculate_movement_stats(G, cluster_set)
        movement_inside = float(stats.get('in', 0.0))
        movement_outside = float(stats.get('out', 0.0))

        initial_cbg_count = len(cluster_set)
        initial_population = int(population)
        initial_movement_inside = movement_inside
        initial_movement_outside = movement_outside
        initial_czi = (
            movement_inside / (movement_inside + movement_outside)
            if movement_inside + movement_outside > 0
            else 0.0
        )
        initial_seed_movement_captured = float(seed_movement_captured)
        initial_seed_capture_share = capture_share(seed_movement_captured)

        self.logger.info(
            "Starting seed_prune: seeds=%d universe=%d CBGs population=%d "
            "seed_capture=%.4f (excluded %d zero-population neighbours) target=%.4f",
            len(seed_cluster),
            initial_cbg_count,
            initial_population,
            float(initial_seed_capture_share),
            len(excluded_zero_population),
            float(threshold),
        )

        def score_key(cbg):
            loss = seed_movement_by_cbg[cbg]
            pop = population_by_cbg[cbg]
            return (loss / pop, loss, -pop, cbg)

        ordered = sorted(members, key=score_key)

        def removal_effect(candidate):
            self_weight = float(G.nodes[candidate].get('self_weight', 0) or 0)
            to_remaining = 0.0
            to_outside = 0.0
            for neighbor in G.adj[candidate]:
                weight = _edge_weight(G, candidate, neighbor)
                if neighbor in cluster_set:
                    to_remaining += weight
                else:
                    to_outside += weight
            loss = self_weight + to_remaining
            inside_after = max(0.0, movement_inside - loss)
            outside_after = max(0.0, movement_outside - to_outside + to_remaining)
            total_after = inside_after + outside_after
            czi_after = inside_after / total_after if total_after > 0 else 0.0
            return loss, to_outside, inside_after, outside_after, czi_after

        def blocked_by_floor(seed_loss, captured):
            share_after = capture_share(max(0.0, captured - seed_loss))
            return share_after < threshold and seed_loss > SEED_PRUNE_TOLERANCE

        def trace_candidates(position, captured, current_share):
            details = []
            for cbg in ordered[position:position + max_trace_candidates]:
                pop = population_by_cbg[cbg]
                seed_loss = float(seed_movement_by_cbg[cbg])
                captured_after = max(0.0, captured - seed_loss)
                share_after = capture_share(captured_after)
                details.append({
                    'cbg': cbg,
                    'population': int(pop),
                    'score': float(share_after),
                    'seed_movement_loss': seed_loss,
                    'seed_movement_loss_per_person': seed_loss / pop,
                    'seed_movement_captured_after': float(captured_after),
                    'seed_capture_after': float(share_after),
                    'current_seed_capture': float(current_share),
                    'would_violate_min_seed_capture': bool(
                        blocked_by_floor(seed_loss, captured)
                    ),
                    'population_after': int(population - pop),
                })
            return details

        cluster = sorted(cluster_set)
        blocked = []
        iteration = 0
        for position, candidate in enumerate(ordered):
            seed_loss = float(seed_movement_by_cbg[candidate])
            if blocked_by_floor(seed_loss, seed_movement_captured):
                blocked.append(candidate)
                continue

            removed_pop = population_by_cbg[candidate]
            movement_loss, _to_outside, inside_after, outside_after, czi_after = (
                removal_effect(candidate)
            )
            captured_after = max(0.0, seed_movement_captured - seed_loss)
            share_after = capture_share(captured_after)

            candidates = None
            prev_cluster = None
            prev_population = int(population)
            if trace_collector is not None:
                candidates = trace_candidates(
                    position,
                    seed_movement_captured,
                    capture_share(seed_movement_captured),
                )
                candidates[0]['movement_loss'] = float(movement_loss)
                candidates[0]['czi_after'] = float(czi_after)
                prev_cluster = list(cluster)

            cluster_set.discard(candidate)
            cluster.remove(candidate)
            population -= int(removed_pop)
            movement_inside = inside_after
            movement_outside = outside_after
            seed_movement_captured = captured_after

            if trace_collector is not None:
                self._record_trace_step(
                    trace_collector,
                    iteration=iteration,
                    cluster_before=prev_cluster,
                    population_before=prev_population,
                    candidates=candidates,
                    selected_cbg=candidate,
                    selected_population=removed_pop,
                    cluster_after=cluster,
                    population_after=population,
                    metrics_after={
                        'stage': 'reverse_prune',
                        'removed_population': int(removed_pop),
                        'movement_loss': float(movement_loss),
                        'movement_inside': float(movement_inside),
                        'movement_boundary': float(movement_outside),
                        'seed_movement_loss': float(seed_loss),
                        'seed_movement_captured': float(seed_movement_captured),
                        'seed_capture': float(share_after),
                        'czi': float(czi_after),
                    },
                    higher_score_better=True,
                )
            iteration += 1

        final_seed_capture_share = capture_share(seed_movement_captured)
        final_czi = (
            movement_inside / (movement_inside + movement_outside)
            if movement_inside + movement_outside > 0
            else 0.0
        )

        self.logger.info(
            "seed_prune finished: removed=%d blocked=%d CBGs=%d population=%d "
            "seed_capture=%.4f CZI=%.4f",
            iteration,
            len(blocked),
            len(cluster),
            int(population),
            float(final_seed_capture_share),
            float(final_czi),
        )

        metadata = {
            'seed_cbgs': list(seed_cluster),
            'missing_seed_cbgs': list(missing_seed_cbgs),
            'seed_population': int(seed_population),
            'bounded_envelope': False,
            'universe_rule': 'seed_neighbors',
            'min_seed_capture': float(threshold),
            'seed_capture_target_met': bool(
                final_seed_capture_share + SEED_PRUNE_TOLERANCE >= threshold
            ),
            'stopped_by_seed_capture_floor': bool(blocked),
            'blocked_cbg_count': int(len(blocked)),
            'excluded_zero_population_cbg_count': int(len(excluded_zero_population)),
            'excluded_zero_population_seed_movement': float(excluded_seed_movement),
            'initial_cbg_count': int(initial_cbg_count),
            'initial_population': int(initial_population),
            'initial_movement_inside': float(initial_movement_inside),
            'initial_movement_boundary': float(initial_movement_outside),
            'initial_czi': float(initial_czi),
            'seed_movement_total': float(seed_movement_total),
            'initial_seed_movement_captured': float(initial_seed_movement_captured),
            'initial_seed_capture_share': float(initial_seed_capture_share),
            'final_seed_movement_captured': float(seed_movement_captured),
            'final_seed_capture_share': float(final_seed_capture_share),
            'final_movement_inside': float(movement_inside),
            'final_movement_boundary': float(movement_outside),
            'final_czi': float(final_czi),
            'minimum_population_used': False,
            'legacy_min_population': int(legacy_min_population),
            'population_reduced': int(initial_population - population),
            'removed_cbg_count': int(initial_cbg_count - len(cluster_set)),
        }

        return cluster, int(population), metadata
