from .metrics import cbg_population


class DirectedClusteringMixin:
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
