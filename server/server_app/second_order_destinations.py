from common_geo import normalize_cbg
from czcode import cbg_population, distance, setup_logging

from .analysis_config import build_analysis_config
from .constants import DEFAULT_DISTANCE_SCALE_KM
from .seed_regions import (
    describe_city_approximation_for_zip,
    get_cbg_to_zip_map,
    get_zip_to_cbgs_map,
)


class SecondOrderDestinationAnalyzer:
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
        config = build_analysis_config(
            seed_cbg,
            0,
            patterns_file=pattern_selection.file_path,
            month=pattern_selection.month,
        )
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
        use_city_approximation = normalize_cbg(seed_cbg) is not None
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

        def destination_unit_for_zip(zip_code):
            if not zip_code:
                return None
            city_info = (
                describe_city_approximation_for_zip(zip_code)
                if use_city_approximation
                else None
            )
            if city_info and city_info.get('unit_id'):
                return city_info
            return {
                'unit_id': str(zip_code),
                'label': f'ZIP {zip_code}',
                'unit_type': 'zip_fallback',
            }

        seed_zip_codes = sorted({
            zip_code for zip_code in (cbg_to_zip.get(cbg) for cbg in normalized_seed_cbgs) if zip_code
        })
        seed_destination_unit_ids = {
            city_info['unit_id']
            for city_info in (
                destination_unit_for_zip(zip_code)
                for zip_code in seed_zip_codes
            )
            if city_info and city_info.get('unit_id')
        }
        seed_city_labels = sorted({
            str(city_info['label'])
            for city_info in (
                destination_unit_for_zip(zip_code)
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
                city_info = destination_unit_for_zip(destination_zip)
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
                city_info = destination_unit_for_zip(source_zip)
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
