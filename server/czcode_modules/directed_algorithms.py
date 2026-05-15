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

    def hierarchical_core_satellites(
        self,
        DG,
        seed_cbgs,
        min_pop,
        cbg_to_zip=None,
        zip_to_cbgs=None,
        cbg_centers=None,
        local_radius_km=20.0,
        core_containment_threshold=0.70,
        core_improvement_epsilon=0.0025,
        satellite_flow_threshold=0.02,
        max_satellites=4,
        max_iter=1000,
        trace_collector=None,
    ):
        cbg_to_zip = cbg_to_zip or {}
        zip_to_cbgs = zip_to_cbgs or {}
        cbg_centers = cbg_centers or {}

        seed_cluster = []
        missing_seed_cbgs = []
        seen = set()
        for cbg in seed_cbgs or []:
            if cbg in seen:
                continue
            seen.add(cbg)
            if cbg in DG:
                seed_cluster.append(cbg)
            else:
                missing_seed_cbgs.append(cbg)

        if not seed_cluster:
            raise ValueError("None of the seed region CBGs are present in the directed mobility graph")

        self.logger.info(
            "Starting hierarchical_core_satellites with %d seed CBGs, min_pop=%s, local_radius_km=%s",
            len(seed_cluster),
            int(min_pop),
            float(local_radius_km),
        )

        core_cluster = list(seed_cluster)
        core_set = set(core_cluster)
        seed_centroid = self._seed_centroid(seed_cluster, cbg_centers)
        seed_zip_codes = sorted({
            zip_code for zip_code in (cbg_to_zip.get(cbg) for cbg in seed_cluster) if zip_code
        })

        core_population = sum(cbg_population(cbg, self.config, self.logger) for cbg in core_cluster)
        core_metrics = self._directed_containment(DG, core_set)

        self.logger.info(
            "Initial seed region: pop=%d origin=%.4f dest=%.4f zone=%.4f",
            core_population,
            core_metrics['origin'],
            core_metrics['destination'],
            core_metrics['zone'],
        )

        iteration = 1
        while iteration <= max_iter:
            local_candidates = set()
            for node in core_cluster:
                if node not in DG:
                    continue
                for candidate in DG.successors(node):
                    if candidate in core_set:
                        continue
                    if not self._within_local_radius(seed_centroid, candidate, cbg_centers, local_radius_km):
                        continue
                    local_candidates.add(candidate)
                for candidate in DG.predecessors(node):
                    if candidate in core_set:
                        continue
                    if not self._within_local_radius(seed_centroid, candidate, cbg_centers, local_radius_km):
                        continue
                    local_candidates.add(candidate)

            if not local_candidates:
                self.logger.info("No local-core candidates remain; stopping local growth.")
                break

            best_candidate = None
            best_metrics = None
            best_score = None
            candidate_details = []

            for candidate in sorted(local_candidates):
                next_set = set(core_set)
                next_set.add(candidate)
                next_metrics = self._directed_containment(DG, next_set)
                score = float(next_metrics['zone'] - core_metrics['zone'])
                candidate_zip = cbg_to_zip.get(candidate)
                candidate_details.append({
                    'cbg': candidate,
                    'population': int(cbg_population(candidate, self.config, self.logger)),
                    'score': score,
                    'new_zone_containment': float(next_metrics['zone']),
                    'origin_containment': float(next_metrics['origin']),
                    'destination_containment': float(next_metrics['destination']),
                    'seed_zip': candidate_zip,
                })

                if (
                    best_score is None
                    or score > best_score
                    or (
                        abs(score - best_score) < 1e-12
                        and (best_candidate is None or candidate < best_candidate)
                    )
                ):
                    best_candidate = candidate
                    best_metrics = next_metrics
                    best_score = score

            if best_candidate is None:
                break

            if best_score <= 0:
                self.logger.info(
                    "Local core stabilized with no improving candidate (best delta=%.4g).",
                    float(best_score),
                )
                break

            if (
                core_metrics['zone'] >= float(core_containment_threshold)
                and best_score <= float(core_improvement_epsilon)
            ):
                self.logger.info(
                    "Local core reached containment target %.4f with marginal gain %.4g; stopping.",
                    float(core_metrics['zone']),
                    float(best_score),
                )
                break

            chosen_population = int(cbg_population(best_candidate, self.config, self.logger))
            prev_cluster = list(core_cluster)
            prev_population = int(core_population)

            core_cluster.append(best_candidate)
            core_set.add(best_candidate)
            core_population += chosen_population
            core_metrics = best_metrics

            self._record_trace_step(
                trace_collector,
                iteration=iteration - 1,
                cluster_before=prev_cluster,
                population_before=prev_population,
                candidates=candidate_details,
                selected_cbg=best_candidate,
                selected_population=chosen_population,
                cluster_after=core_cluster,
                population_after=core_population,
                metrics_after={
                    'stage': 'local_core',
                    'origin_containment': float(core_metrics['origin']),
                    'destination_containment': float(core_metrics['destination']),
                    'zone_containment': float(core_metrics['zone']),
                },
            )

            self.logger.info(
                "Local core iteration %d: added %s pop=%d zone=%.4f",
                iteration,
                best_candidate,
                chosen_population,
                core_metrics['zone'],
            )
            iteration += 1

        final_cluster = list(core_cluster)
        final_set = set(core_set)
        total_population = int(core_population)

        satellite_candidates = []
        satellite_units = {}
        for node in core_cluster:
            if node not in DG:
                continue
            neighbor_iter = list(DG.successors(node)) + list(DG.predecessors(node))
            for candidate in neighbor_iter:
                if candidate in final_set:
                    continue
                unit_id = cbg_to_zip.get(candidate)
                if not unit_id or unit_id in seed_zip_codes:
                    continue
                satellite_units.setdefault(unit_id, set())

        for unit_id in list(satellite_units.keys()):
            unit_cbgs = []
            for cbg in zip_to_cbgs.get(unit_id, []):
                if cbg in final_set or cbg not in DG:
                    continue
                unit_cbgs.append(cbg)

            if not unit_cbgs:
                continue

            shared_flow = 0.0
            total_flow = 0.0
            unit_population = 0
            for cbg in unit_cbgs:
                unit_population += int(cbg_population(cbg, self.config, self.logger))
                total_flow += self._node_total_directed_flow(DG, cbg)

                for _, v, data in DG.out_edges(cbg, data=True):
                    if v in core_set:
                        shared_flow += float(data.get('weight', 0) or 0)
                for u, _, data in DG.in_edges(cbg, data=True):
                    if u in core_set:
                        shared_flow += float(data.get('weight', 0) or 0)

            if unit_population <= 0 or shared_flow <= 0 or total_flow <= 0:
                continue

            coupling = float(shared_flow / total_flow)
            satellite_candidates.append({
                'unit_id': unit_id,
                'label': f"ZIP {unit_id}",
                'cbgs': sorted(unit_cbgs),
                'population': int(unit_population),
                'shared_flow': float(shared_flow),
                'total_flow': float(total_flow),
                'coupling': coupling,
                'cbg_count': len(unit_cbgs),
            })

        satellite_candidates.sort(
            key=lambda item: (
                float(item['coupling']),
                float(item['shared_flow']),
                int(item['population']),
                str(item['unit_id']),
            ),
            reverse=True,
        )

        selected_satellites = []
        try:
            max_satellites_int = max(0, int(max_satellites))
        except (TypeError, ValueError):
            max_satellites_int = 0

        target_population = int(min_pop)
        for candidate in satellite_candidates:
            if selected_satellites and total_population >= target_population:
                break
            if max_satellites_int == 0:
                break
            if max_satellites_int and len(selected_satellites) >= max_satellites_int:
                break
            if candidate['coupling'] <= 0:
                break
            if total_population >= target_population and candidate['coupling'] < float(satellite_flow_threshold):
                break

            added_cbgs = [cbg for cbg in candidate['cbgs'] if cbg not in final_set]
            if not added_cbgs:
                continue

            final_cluster.extend(added_cbgs)
            final_set.update(added_cbgs)
            total_population += int(candidate['population'])
            selected_satellites.append({
                'unit_id': candidate['unit_id'],
                'label': candidate['label'],
                'population': int(candidate['population']),
                'coupling': float(candidate['coupling']),
                'shared_flow': float(candidate['shared_flow']),
                'cbg_count': int(candidate['cbg_count']),
            })

        final_metrics = self._directed_containment(DG, final_set)
        metadata = {
            'seed_cbgs': list(seed_cluster),
            'missing_seed_cbgs': list(missing_seed_cbgs),
            'seed_zip_codes': seed_zip_codes,
            'core_cluster': list(core_cluster),
            'core_population': int(core_population),
            'core_containment': {
                'origin': float(core_metrics['origin']),
                'destination': float(core_metrics['destination']),
                'zone': float(core_metrics['zone']),
            },
            'selected_satellites': selected_satellites,
            'satellite_candidate_count': len(satellite_candidates),
            'final_containment': {
                'origin': float(final_metrics['origin']),
                'destination': float(final_metrics['destination']),
                'zone': float(final_metrics['zone']),
            },
            'external_pressure_share': float(max(0.0, 1.0 - final_metrics['zone'])),
            'population_target_met': bool(total_population >= int(min_pop)),
        }

        if total_population < int(min_pop):
            self.logger.warning(
                "Hierarchical cluster ended below target population: pop=%d target=%d",
                total_population,
                int(min_pop),
            )

        return final_cluster, total_population, metadata

