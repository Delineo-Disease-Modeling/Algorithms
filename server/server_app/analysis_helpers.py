import json

import pandas as pd

from common_geo import normalize_cbg

from .logging_utils import log_candidate_pois


def iter_geometry_lng_lat_pairs(coords):
    if isinstance(coords, (list, tuple)):
        if len(coords) == 2 and all(isinstance(v, (int, float)) for v in coords):
            yield float(coords[0]), float(coords[1])
            return
        for item in coords:
            yield from iter_geometry_lng_lat_pairs(item)


def compute_geojson_bounds(feature_collection):
    min_lng = float('inf')
    min_lat = float('inf')
    max_lng = float('-inf')
    max_lat = float('-inf')
    found = False

    for feature in (feature_collection.get('features') or []):
        geom = feature.get('geometry') if isinstance(feature, dict) else None
        coords = geom.get('coordinates') if isinstance(geom, dict) else None
        for lng, lat in iter_geometry_lng_lat_pairs(coords):
            found = True
            min_lng = min(min_lng, lng)
            min_lat = min(min_lat, lat)
            max_lng = max(max_lng, lng)
            max_lat = max(max_lat, lat)

    if not found:
        return None

    return {
        'min_lng': min_lng,
        'min_lat': min_lat,
        'max_lng': max_lng,
        'max_lat': max_lat,
        'center_lat': (min_lat + max_lat) / 2.0,
        'center_lng': (min_lng + max_lng) / 2.0,
    }


def safe_float(raw, default=0.0):
    try:
        value = float(raw)
        if value != value:
            return default
        return value
    except (TypeError, ValueError):
        return default


def parse_visitor_daytime_cbgs(raw):
    parsed = raw
    for _ in range(3):
        if isinstance(parsed, dict):
            break

        text = str(parsed or '').strip()
        if not text:
            return {}

        next_value = None
        for candidate in (text, text.replace("'", '"')):
            try:
                next_value = json.loads(candidate)
                break
            except Exception:
                continue
        if next_value is None:
            return {}
        parsed = next_value

    if not isinstance(parsed, dict):
        return {}

    normalized = {}
    for cbg, count in parsed.items():
        cbg_norm = normalize_cbg(cbg)
        if not cbg_norm:
            continue
        flow = safe_float(count, 0.0)
        if flow <= 0:
            continue
        normalized[cbg_norm] = normalized.get(cbg_norm, 0.0) + flow
    return normalized


def build_candidate_poi_parquet_filter(ds, field_name, field_type, candidate_cbg):
    try:
        candidate_int = int(str(candidate_cbg).strip())
    except (TypeError, ValueError):
        candidate_int = None

    type_name = str(field_type)
    if candidate_int is None:
        return None, type_name

    try:
        import pyarrow.types as patypes

        if patypes.is_integer(field_type):
            return ds.field(field_name) == candidate_int, type_name
        if patypes.is_floating(field_type):
            return ds.field(field_name) == float(candidate_int), type_name
        if patypes.is_string(field_type) or patypes.is_large_string(field_type):
            return (
                (ds.field(field_name) == str(candidate_int))
                | (ds.field(field_name) == f"{candidate_int}.0")
                | (ds.field(field_name) == str(candidate_cbg).strip())
            ), type_name
        if patypes.is_dictionary(field_type):
            value_type = field_type.value_type
            value_type_name = f"{type_name}->{value_type}"
            if patypes.is_string(value_type) or patypes.is_large_string(value_type):
                return (
                    (ds.field(field_name) == str(candidate_int))
                    | (ds.field(field_name) == f"{candidate_int}.0")
                    | (ds.field(field_name) == str(candidate_cbg).strip())
                ), value_type_name
            if patypes.is_integer(value_type):
                return ds.field(field_name) == candidate_int, value_type_name
            if patypes.is_floating(value_type):
                return ds.field(field_name) == float(candidate_int), value_type_name
            return None, value_type_name
    except Exception:
        pass

    return None, type_name


def iter_candidate_poi_pattern_chunks(patterns_file, usecols_lower, candidate_cbg=None, chunksize=10000):
    path = str(patterns_file or '')
    lower_path = path.lower()
    diagnostics = {
        'path': path,
        'format': 'parquet' if lower_path.endswith('.parquet') else 'csv',
        'selected_cols': [],
        'poi_column': None,
        'poi_column_type': None,
        'pushdown_filter': False,
    }

    if lower_path.endswith('.parquet'):
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format='parquet')
        selected_cols = [
            name for name in dataset.schema.names
            if str(name).strip().lower() in usecols_lower
        ]
        diagnostics['selected_cols'] = selected_cols
        if not selected_cols:
            yield None, diagnostics
            return

        filter_expr = None
        actual_poi_col = next(
            (name for name in dataset.schema.names if str(name).strip().lower() == 'poi_cbg'),
            None
        )
        diagnostics['poi_column'] = actual_poi_col
        if actual_poi_col:
            field = dataset.schema.field(actual_poi_col)
            filter_expr, poi_type = build_candidate_poi_parquet_filter(
                ds,
                actual_poi_col,
                field.type,
                candidate_cbg
            )
            diagnostics['poi_column_type'] = poi_type
            diagnostics['pushdown_filter'] = filter_expr is not None

        scanner = dataset.scanner(
            columns=selected_cols,
            filter=filter_expr,
            batch_size=chunksize
        )

        for batch in scanner.to_batches():
            chunk = batch.to_pandas()
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            yield chunk, diagnostics
        return

    diagnostics['selected_cols'] = sorted(usecols_lower)
    with pd.read_csv(
        path,
        chunksize=chunksize,
        dtype=str,
        usecols=lambda c: str(c).strip().lower() in usecols_lower
    ) as reader:
        for chunk in reader:
            chunk = chunk.copy()
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            yield chunk, diagnostics


def compute_top_candidate_pois(patterns_file, candidate_cbg, cluster_cbgs, limit=8):
    cluster_set = {normalize_cbg(cbg) for cbg in cluster_cbgs}
    cluster_set.discard(None)
    if not cluster_set:
        return []

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 8
    limit = max(1, min(limit, 20))

    aggregated = {}
    required_cols = ['poi_cbg', 'visitor_daytime_cbgs']
    metadata_cols = [
        'placekey', 'location_name', 'top_category', 'sub_category',
        'street_address', 'city', 'region', 'postal_code',
        'raw_visit_counts', 'raw_visitor_counts'
    ]
    usecols = required_cols + metadata_cols
    usecols_lower = {str(c).strip().lower() for c in usecols}
    diagnostics = {
        'chunks': 0,
        'candidate_rows': 0,
        'rows_with_visitors': 0,
        'rows_with_overlap': 0,
        'sample_visitor_type': None,
        'sample_visitor_excerpt': None,
        'iter': None,
    }

    for chunk, iter_diagnostics in iter_candidate_poi_pattern_chunks(
        patterns_file,
        usecols_lower,
        candidate_cbg=candidate_cbg,
        chunksize=10000
    ):
        diagnostics['iter'] = iter_diagnostics
        if chunk is None:
            break

        diagnostics['chunks'] += 1
        if 'poi_cbg' not in chunk.columns or 'visitor_daytime_cbgs' not in chunk.columns:
            continue

        chunk['poi_cbg_norm'] = pd.to_numeric(chunk['poi_cbg'], errors='coerce')
        chunk = chunk[chunk['poi_cbg_norm'].notna()]
        if chunk.empty:
            continue

        chunk['poi_cbg_norm'] = (
            chunk['poi_cbg_norm']
            .astype('int64')
            .astype('string')
            .str.zfill(12)
        )
        chunk = chunk[chunk['poi_cbg_norm'] == candidate_cbg]
        if chunk.empty:
            continue
        diagnostics['candidate_rows'] += len(chunk)

        for idx, row in chunk.iterrows():
            raw_visitor_value = row.get('visitor_daytime_cbgs')
            if diagnostics['sample_visitor_type'] is None and raw_visitor_value not in (None, ''):
                diagnostics['sample_visitor_type'] = type(raw_visitor_value).__name__
                diagnostics['sample_visitor_excerpt'] = repr(raw_visitor_value)[:240]

            visitors = parse_visitor_daytime_cbgs(row.get('visitor_daytime_cbgs'))
            if not visitors:
                continue
            diagnostics['rows_with_visitors'] += 1

            cluster_flow = sum(visitors.get(cbg, 0.0) for cbg in cluster_set)
            if cluster_flow <= 0:
                continue
            diagnostics['rows_with_overlap'] += 1

            placekey = str(row.get('placekey') or '').strip()
            location_name = str(row.get('location_name') or '').strip()
            item_id = placekey or f"{candidate_cbg}:{location_name or idx}"

            existing = aggregated.get(item_id)
            if existing is None:
                existing = {
                    'placekey': placekey or None,
                    'location_name': location_name or 'Unknown POI',
                    'top_category': str(row.get('top_category') or '').strip() or None,
                    'sub_category': str(row.get('sub_category') or '').strip() or None,
                    'street_address': str(row.get('street_address') or '').strip() or None,
                    'city': str(row.get('city') or '').strip() or None,
                    'region': str(row.get('region') or '').strip() or None,
                    'postal_code': str(row.get('postal_code') or '').strip() or None,
                    'raw_visit_counts': 0.0,
                    'raw_visitor_counts': 0.0,
                    'cluster_flow': 0.0,
                    'source_rows': 0,
                }
                aggregated[item_id] = existing

            existing['cluster_flow'] += cluster_flow
            existing['raw_visit_counts'] += safe_float(row.get('raw_visit_counts'), 0.0)
            existing['raw_visitor_counts'] += safe_float(row.get('raw_visitor_counts'), 0.0)
            existing['source_rows'] += 1

    iter_info = diagnostics.get('iter') or {}
    log_candidate_pois(
        "candidate-pois file=%s format=%s pushdown=%s poi_col=%s poi_type=%s "
        "chunks=%s candidate_rows=%s visitor_rows=%s overlap_rows=%s sample_type=%s sample=%s",
        iter_info.get('path') or patterns_file,
        iter_info.get('format'),
        iter_info.get('pushdown_filter'),
        iter_info.get('poi_column'),
        iter_info.get('poi_column_type'),
        diagnostics['chunks'],
        diagnostics['candidate_rows'],
        diagnostics['rows_with_visitors'],
        diagnostics['rows_with_overlap'],
        diagnostics['sample_visitor_type'],
        diagnostics['sample_visitor_excerpt'],
    )
    if not aggregated:
        return []

    ranked = sorted(
        aggregated.values(),
        key=lambda item: (
            float(item.get('cluster_flow', 0.0)),
            float(item.get('raw_visit_counts', 0.0)),
            str(item.get('location_name') or '')
        ),
        reverse=True
    )[:limit]

    total_flow = sum(float(item.get('cluster_flow', 0.0)) for item in ranked)
    for idx, item in enumerate(ranked):
        flow = float(item.get('cluster_flow', 0.0))
        item['rank'] = idx + 1
        item['cluster_flow'] = round(flow, 2)
        item['flow_share'] = round((flow / total_flow), 4) if total_flow > 0 else 0.0
        item['raw_visit_counts'] = round(float(item.get('raw_visit_counts', 0.0)), 2)
        item['raw_visitor_counts'] = round(float(item.get('raw_visitor_counts', 0.0)), 2)

    return ranked
