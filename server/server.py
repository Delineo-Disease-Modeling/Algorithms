from flask import Flask, request, jsonify, make_response, after_this_request
from flask_cors import CORS, cross_origin
from czcode import (
  generate_cz,
  Config,
  DataLoader,
  GraphBuilder,
  Helpers,
  Clustering,
  cbg_population,
  build_cbg_centers,
  distance,
  setup_logging,
)
from popgen import gen_pop
from patterns import gen_patterns
from geojsongen import get_cbg_geojson, get_cbg_at_point
from datetime import datetime
import os
import csv
import glob
import requests
import json
import pandas as pd
from jsonschema import validate
from schema import gen_cz_schema
from io import BytesIO
import re
from functools import lru_cache
from run_report import RunReport

app = Flask(__name__)
CORS(app,
  origins=['http://localhost:5173', 'https://coviddev.isi.jhu.edu', 'http://coviddev.isi.jhu.edu', 'https://covidweb.isi.jhu.edu', 'http://covidweb.isi.jhu.edu'],
  methods=['GET', 'HEAD', 'PUT', 'PATCH', 'POST', 'DELETE'],
  allow_headers=['Content-Type', 'Authorization'],
  expose_headers=['Set-Cookie'],
  supports_credentials=True
)

TEST_PATTERNS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'TEST', 'test.csv')
TEST_CLUSTER_COLUMNS = ['poi_cbg', 'visitor_daytime_cbgs']
TEST_SIM_COLUMNS = ['placekey', 'median_dwell', 'popularity_by_hour', 'popularity_by_day']
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
VALID_CLUSTER_ALGORITHMS = {
  'czi_balanced',
  'czi_optimal_cap',
  'greedy_fast',
  'greedy_weight',
  'greedy_ratio'
}
DEFAULT_DISTANCE_PENALTY_WEIGHT = 0.02
DEFAULT_DISTANCE_SCALE_KM = 20.0
DEFAULT_OPTIMAL_CANDIDATE_LIMIT = 120
DEFAULT_OPTIMAL_POP_FLOOR_RATIO = 0.9
DEFAULT_OPTIMAL_MIP_REL_GAP = 0.02
DEFAULT_OPTIMAL_TIME_LIMIT_SEC = 20.0
DEFAULT_OPTIMAL_MAX_ITERS = 8

# Keep this aligned with czcode.py for state lookup from CBG FIPS prefix.
STATE_ABBR_TO_FIPS = {
  'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
  'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
  'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
  'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
  'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
  'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
  'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
  'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
  'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
  'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
  'WY': '56', 'PR': '72', 'VI': '78', 'GU': '66', 'MP': '69',
  'AS': '60',
}
STATE_FIPS_TO_ABBR = {fips: abbr for abbr, fips in STATE_ABBR_TO_FIPS.items()}

def _normalize_cbg(cbg):
  try:
    cbg_str = str(int(float(cbg)))
  except (TypeError, ValueError):
    cbg_str = str(cbg).strip()
  if len(cbg_str) == 11:
    cbg_str = cbg_str.zfill(12)
  if len(cbg_str) == 12 and re.fullmatch(r"\d{12}", cbg_str):
    return cbg_str
  return None

def _read_csv_headers(csv_path):
  with open(csv_path, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    headers = next(reader, [])
  return [h.strip() for h in headers if h is not None]

def _validate_csv_columns(csv_path, required_columns):
  if not os.path.exists(csv_path):
    return False, list(required_columns), []
  headers = _read_csv_headers(csv_path)
  missing = [c for c in required_columns if c not in headers]
  return len(missing) == 0, missing, headers

def _get_test_seed_cbg(csv_path):
  if not os.path.exists(csv_path):
    return None
  with open(csv_path, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
      cbg = _normalize_cbg(row.get('poi_cbg'))
      if cbg:
        return cbg
  return None


def _extract_month_key(start_date_raw):
  if start_date_raw is None:
    return None
  raw = str(start_date_raw).strip()
  if not raw:
    return None

  # Fast path: "YYYY-MM-DD" or "YYYY-MM..."
  m = re.match(r'^(\d{4})-(\d{2})', raw)
  if m:
    return f"{m.group(1)}-{m.group(2)}"

  try:
    parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    return parsed.strftime('%Y-%m')
  except ValueError:
    return None


def _default_patterns_file():
  # Legacy defaults used by clustering if month/state-specific files are unavailable.
  candidates = [
    os.path.join(DATA_DIR, 'patterns.csv'),
    os.path.join(DATA_DIR, 'patterns_o.csv'),
  ]
  for path in candidates:
    if os.path.exists(path):
      return path
  return None


def _resolve_monthly_patterns_file(cbg_str, month_key):
  """
  Resolve month-specific pattern CSV for preview clustering.
  Preferred format: data/{STATE}/{YYYY-MM}-{STATE}.csv
  """
  if not cbg_str or not month_key:
    return None

  state_fips = str(cbg_str)[:2]
  state_abbr = STATE_FIPS_TO_ABBR.get(state_fips)

  candidates = []
  if state_abbr:
    candidates.extend([
      os.path.join(DATA_DIR, state_abbr, f'{month_key}-{state_abbr}.csv'),
      os.path.join(DATA_DIR, f'{month_key}-{state_abbr}.csv'),
    ])
    candidates.extend(sorted(glob.glob(
      os.path.join(DATA_DIR, '**', f'{month_key}-{state_abbr}.csv'),
      recursive=True
    )))

  for path in candidates:
    if path and os.path.exists(path):
      return path

  # Do not cross state boundaries when the seed state is known.
  if state_abbr:
    return None

  # Fallback only when state cannot be inferred: if exactly one monthly file exists, use it.
  generic = sorted(glob.glob(
    os.path.join(DATA_DIR, '**', f'{month_key}-*.csv'),
    recursive=True
  ))
  if len(generic) == 1 and os.path.exists(generic[0]):
    return generic[0]
  return None


def _resolve_patterns_file_for_request(seed_cbg, start_date_raw=None, use_test_data=False):
  patterns_file = None
  patterns_source = 'default'
  patterns_month = None

  if use_test_data:
    ok, missing, _headers = _validate_csv_columns(TEST_PATTERNS_FILE, TEST_CLUSTER_COLUMNS)
    if not ok:
      raise ValueError(
        f"TEST data is missing required columns for clustering: {', '.join(missing)}"
      )
    return TEST_PATTERNS_FILE, 'test', None

  patterns_month = _extract_month_key(start_date_raw)
  if patterns_month:
    monthly_file = _resolve_monthly_patterns_file(seed_cbg, patterns_month)
    if monthly_file:
      patterns_file = monthly_file
      patterns_source = 'monthly'

  if not patterns_file:
    default_file = _default_patterns_file()
    if default_file:
      patterns_file = default_file
      patterns_source = 'default'

  if not patterns_file:
    raise ValueError(
      "No patterns file available for this request (monthly and default lookups failed)."
    )

  return patterns_file, patterns_source, patterns_month


def _safe_float(raw, default=0.0):
  try:
    value = float(raw)
    if value != value:  # NaN
      return default
    return value
  except (TypeError, ValueError):
    return default


def _parse_visitor_daytime_cbgs(raw):
  if isinstance(raw, dict):
    parsed = raw
  else:
    text = str(raw or '').strip()
    if not text:
      return {}
    parsed = None
    for candidate in (text, text.replace("'", '"')):
      try:
        parsed = json.loads(candidate)
        break
      except Exception:
        continue
    if not isinstance(parsed, dict):
      return {}

  normalized = {}
  for cbg, count in parsed.items():
    cbg_norm = _normalize_cbg(cbg)
    if not cbg_norm:
      continue
    flow = _safe_float(count, 0.0)
    if flow <= 0:
      continue
    normalized[cbg_norm] = normalized.get(cbg_norm, 0.0) + flow
  return normalized


def _compute_top_candidate_pois(patterns_file, candidate_cbg, cluster_cbgs, limit=8):
  cluster_set = {_normalize_cbg(cbg) for cbg in cluster_cbgs}
  cluster_set.discard(None)
  if not cluster_set:
    return []

  try:
    limit = int(limit)
  except (TypeError, ValueError):
    limit = 8
  limit = max(1, min(limit, 20))

  aggregated = {}
  source_rows = 0
  required_cols = ['poi_cbg', 'visitor_daytime_cbgs']
  metadata_cols = [
    'placekey', 'location_name', 'top_category', 'sub_category',
    'street_address', 'city', 'region', 'postal_code',
    'raw_visit_counts', 'raw_visitor_counts'
  ]
  usecols = required_cols + metadata_cols

  with pd.read_csv(patterns_file, chunksize=10000, dtype=str, usecols=lambda c: c in usecols) as reader:
    for chunk in reader:
      if 'poi_cbg' not in chunk.columns or 'visitor_daytime_cbgs' not in chunk.columns:
        continue

      chunk = chunk.copy()
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

      for idx, row in chunk.iterrows():
        visitors = _parse_visitor_daytime_cbgs(row.get('visitor_daytime_cbgs'))
        if not visitors:
          continue

        cluster_flow = sum(visitors.get(cbg, 0.0) for cbg in cluster_set)
        if cluster_flow <= 0:
          continue

        source_rows += 1
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
        existing['raw_visit_counts'] += _safe_float(row.get('raw_visit_counts'), 0.0)
        existing['raw_visitor_counts'] += _safe_float(row.get('raw_visitor_counts'), 0.0)
        existing['source_rows'] += 1

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

def _normalize_cluster_algorithm(algorithm):
  if algorithm is None:
    return 'czi_balanced'

  alg = str(algorithm).strip().lower()
  aliases = {
    'balanced': 'czi_balanced',
    'czi': 'czi_balanced',
    'optimal': 'czi_optimal_cap',
    'rigorous': 'czi_optimal_cap',
    'milp': 'czi_optimal_cap',
    'fast': 'greedy_fast',
    'weight': 'greedy_weight',
    'ratio': 'greedy_ratio',
  }
  alg = aliases.get(alg, alg)
  return alg if alg in VALID_CLUSTER_ALGORITHMS else None

def _parse_czi_balanced_params(payload):
  payload = payload or {}

  def parse_float(name, min_value=None, max_value=None, strictly_positive=False):
    raw = payload.get(name)
    if raw is None or raw == '':
      return None, None
    try:
      value = float(raw)
    except (TypeError, ValueError):
      return None, f"Invalid '{name}': expected a number"
    if strictly_positive and value <= 0:
      return None, f"Invalid '{name}': must be > 0"
    if min_value is not None and value < min_value:
      return None, f"Invalid '{name}': must be >= {min_value}"
    if max_value is not None and value > max_value:
      return None, f"Invalid '{name}': must be <= {max_value}"
    return value, None

  distance_penalty_weight, err = parse_float(
    'distance_penalty_weight', min_value=0.0, max_value=1.0
  )
  if err:
    return None, err

  distance_scale_km, err = parse_float(
    'distance_scale_km', min_value=0.1, max_value=500.0, strictly_positive=True
  )
  if err:
    return None, err

  return {
    'distance_penalty_weight': distance_penalty_weight,
    'distance_scale_km': distance_scale_km,
  }, None


def _parse_czi_optimal_params(payload):
  payload = payload or {}

  def parse_float(name, min_value=None, max_value=None, strictly_positive=False):
    raw = payload.get(name)
    if raw is None or raw == '':
      return None, None
    try:
      value = float(raw)
    except (TypeError, ValueError):
      return None, f"Invalid '{name}': expected a number"
    if strictly_positive and value <= 0:
      return None, f"Invalid '{name}': must be > 0"
    if min_value is not None and value < min_value:
      return None, f"Invalid '{name}': must be >= {min_value}"
    if max_value is not None and value > max_value:
      return None, f"Invalid '{name}': must be <= {max_value}"
    return value, None

  def parse_int(name, min_value=None, max_value=None, strictly_positive=False):
    raw = payload.get(name)
    if raw is None or raw == '':
      return None, None
    try:
      value = int(float(raw))
    except (TypeError, ValueError):
      return None, f"Invalid '{name}': expected an integer"
    if strictly_positive and value <= 0:
      return None, f"Invalid '{name}': must be > 0"
    if min_value is not None and value < min_value:
      return None, f"Invalid '{name}': must be >= {min_value}"
    if max_value is not None and value > max_value:
      return None, f"Invalid '{name}': must be <= {max_value}"
    return value, None

  candidate_limit, err = parse_int(
    'optimal_candidate_limit', min_value=20, max_value=400, strictly_positive=True
  )
  if err:
    return None, err

  population_floor_ratio, err = parse_float(
    'optimal_population_floor_ratio', min_value=0.0, max_value=1.0
  )
  if err:
    return None, err

  mip_rel_gap, err = parse_float(
    'optimal_mip_rel_gap', min_value=0.0, max_value=0.2
  )
  if err:
    return None, err

  time_limit_sec, err = parse_float(
    'optimal_time_limit_sec', min_value=1.0, max_value=300.0, strictly_positive=True
  )
  if err:
    return None, err

  max_iters, err = parse_int(
    'optimal_max_iters', min_value=1, max_value=30, strictly_positive=True
  )
  if err:
    return None, err

  return {
    'optimal_candidate_limit': candidate_limit,
    'optimal_population_floor_ratio': population_floor_ratio,
    'optimal_mip_rel_gap': mip_rel_gap,
    'optimal_time_limit_sec': time_limit_sec,
    'optimal_max_iters': max_iters,
  }, None

def gen_and_upload_data(geoids, czone_id, start_date, length, report, gdf=None,
                        patterns_file=None, patterns_folder=None):
  """Generate papdata and patterns, upload to DB API.
  
  Args:
    geoids: Dictionary of CBG -> population
    czone_id: Convenience zone ID
    start_date: Start datetime for patterns
    length: Length in hours
    report: RunReport for logging
    gdf: Optional GeoDataFrame with CBG geometries for residential sampling
  """
  try:
    # Generate People, Households, Places data
    report.info('Generating synthetic population (papdata)...')
    papdata = gen_pop(geoids, gdf=gdf)
    people_count = len(papdata.get('people', {}))
    homes_count = len(papdata.get('homes', {}))
    places_count = len(papdata.get('places', {}))
    
    # Check if homes have coordinates
    homes_with_coords = sum(1 for h in papdata.get('homes', {}).values() 
                           if h.get('latitude') is not None)
    if homes_with_coords > 0:
      report.info(f'Generated papdata: {people_count} people, {homes_count} homes ({homes_with_coords} with coordinates), {places_count} places')
    else:
      report.info(f'Generated papdata: {people_count} people, {homes_count} homes, {places_count} places')
    
    # Generate movement patterns
    report.info('Generating movement patterns...')
    patterns = gen_patterns(
      papdata,
      start_date,
      length,
      patterns_file=patterns_file,
      patterns_folder=patterns_folder
    )
    patterns_count = len(patterns)
    report.info(f'Generated {patterns_count} timestep patterns')
        
    report.info('Uploading data to DB API...')
    resp = requests.post('http://localhost:1890/patterns', data={
      'czone_id': int(czone_id),
    }, files={
      'papdata': ('papdata.json', BytesIO(json.dumps(papdata).encode()), 'text/plain'),
      'patterns': ('patterns.json', BytesIO(json.dumps(patterns).encode()), 'text/plain')
    })
    
    if resp.ok:
      report.info('Data uploaded successfully!')
      report.complete(summary={
        'people_count': people_count,
        'homes_count': homes_count,
        'places_count': places_count,
        'patterns_count': patterns_count,
        'cbg_count': len(geoids),
      })
    else:
      report.fail(f'Error uploading data: HTTP {resp.status_code}')
  except Exception as e:
    report.capture_exception()


def cluster_cbgs(cbg, min_pop, patterns_file=None, patterns_folder=None, month=None,
                 algorithm='czi_balanced', czi_params=None, optimal_params=None,
                 include_trace=False):
  """Just cluster CBGs without generating patterns. Returns geoids dict and map center."""
  czi_params = czi_params or {}
  optimal_params = optimal_params or {}
  result = generate_cz(
    cbg,
    min_pop,
    patterns_file=patterns_file,
    patterns_folder=patterns_folder,
    month=month,
    algorithm=algorithm,
    distance_penalty_weight=czi_params.get('distance_penalty_weight'),
    distance_scale_km=czi_params.get('distance_scale_km'),
    optimal_candidate_limit=optimal_params.get('optimal_candidate_limit'),
    optimal_population_floor_ratio=optimal_params.get('optimal_population_floor_ratio'),
    optimal_mip_rel_gap=optimal_params.get('optimal_mip_rel_gap'),
    optimal_time_limit_sec=optimal_params.get('optimal_time_limit_sec'),
    optimal_max_iters=optimal_params.get('optimal_max_iters'),
    include_trace=include_trace
  )
  if include_trace:
    geoids, map_obj, _gdf, trace_payload = result
    return geoids, [map_obj.location[0], map_obj.location[1]], trace_payload

  geoids, map_obj, _gdf = result
  return geoids, [map_obj.location[0], map_obj.location[1]], None

@lru_cache(maxsize=8)
def get_cached_mobility_graph(seed_cbg, patterns_file=None, patterns_folder=None, month=None, cache_tag='v3'):
  """
  Build and cache the movement graph used by CZ clustering for a seed CBG.
  This allows fast repeated CZI updates while the user edits a zone.
  """
  _ = cache_tag  # Included in cache key to invalidate old cached graphs on logic changes.
  config = Config(
    seed_cbg,
    0,
    patterns_file=patterns_file,
    patterns_folder=patterns_folder,
    month=month
  )
  logger = setup_logging(config)

  data_loader = DataLoader(config, logger)
  zip_codes = data_loader.get_zip_codes()
  df = data_loader.load_safegraph_data(zip_codes)

  graph_builder = GraphBuilder(logger)
  return graph_builder.gen_graph(df)


@lru_cache(maxsize=8)
def get_cached_cbg_centers(seed_cbg, patterns_file=None, patterns_folder=None, month=None, cache_tag='v1'):
  """
  Build and cache representative-point centroids for CBGs in the seed region.
  Used by balanced-CZI frontier scoring in edit mode.
  """
  _ = cache_tag
  config = Config(
    seed_cbg,
    0,
    patterns_file=patterns_file,
    patterns_folder=patterns_folder,
    month=month
  )
  logger = setup_logging(config)
  data_loader = DataLoader(config, logger)
  gdf = data_loader.load_shapefiles()
  return build_cbg_centers(gdf)


def _rank_frontier_candidates_for_cluster(
  graph,
  seed_cbg,
  cluster_cbgs,
  algorithm,
  min_pop=0,
  czi_params=None,
  patterns_file=None,
  month=None,
  limit=None,
):
  """
  Score the current frontier (neighbors of cluster not already selected) using the selected
  clustering algorithm's scoring heuristic. Returns (candidates, missing_cluster_cbgs).
  """
  czi_params = czi_params or {}

  config = Config(seed_cbg, 0, patterns_file=patterns_file, month=month)
  logger = setup_logging(config)
  clustering = Clustering(config, logger)

  # Mobility graph keys are typically stored without leading zeros because they are parsed
  # through int(float(...)) during graph construction.
  def graph_key(cbg):
    try:
      return str(int(float(cbg)))
    except (TypeError, ValueError):
      return str(cbg).strip()

  def normalize_output_cbg(cbg):
    return _normalize_cbg(cbg) or str(cbg)

  cluster_graph_ids = []
  seen_graph_ids = set()
  missing_cluster_cbgs = []
  for cbg in cluster_cbgs:
    gkey = graph_key(cbg)
    if not gkey or gkey in seen_graph_ids:
      continue
    seen_graph_ids.add(gkey)
    if gkey not in graph:
      missing_cluster_cbgs.append(normalize_output_cbg(cbg))
      continue
    cluster_graph_ids.append(gkey)

  cluster_set = set(cluster_graph_ids)
  if not cluster_graph_ids:
    return [], sorted(set(missing_cluster_cbgs))

  frontier = set()
  for member in cluster_graph_ids:
    try:
      for neighbor in graph.adj[member]:
        if neighbor not in cluster_set:
          frontier.add(neighbor)
    except KeyError:
      continue

  if not frontier:
    return [], sorted(set(missing_cluster_cbgs))

  current_population = 0
  for cbg in cluster_cbgs:
    current_population += int(cbg_population(cbg, config, logger) or 0)

  algorithm = _normalize_cluster_algorithm(algorithm) or algorithm or 'greedy_weight'
  candidate_details = []

  if algorithm == 'czi_balanced':
    distance_penalty_weight = (
      czi_params.get('distance_penalty_weight')
      if czi_params.get('distance_penalty_weight') is not None
      else DEFAULT_DISTANCE_PENALTY_WEIGHT
    )
    distance_scale_km = (
      czi_params.get('distance_scale_km')
      if czi_params.get('distance_scale_km') is not None
      else DEFAULT_DISTANCE_SCALE_KM
    )
    alpha = 0.75
    overshoot_penalty = 0.25
    gap = max(1, int(_safe_float(min_pop, current_population)) - int(current_population))

    movement_stats = Helpers.calculate_movement_stats(graph, cluster_graph_ids)
    movement_in = float(movement_stats.get('in', 0))
    movement_out = float(movement_stats.get('out', 0))

    cbg_centers = get_cached_cbg_centers(seed_cbg, patterns_file=patterns_file, month=month, cache_tag='v1')
    seed_center = cbg_centers.get(graph_key(seed_cbg)) or cbg_centers.get(seed_cbg)

    for candidate in frontier:
      if candidate in cluster_set or candidate not in graph:
        continue

      cand_pop = int(cbg_population(candidate, config, logger) or 0)
      if cand_pop <= 0:
        continue

      self_weight = float(graph.nodes[candidate].get('self_weight', 0))
      in_to_cluster = 0.0
      out_to_outside = 0.0
      for neighbor in graph.adj[candidate]:
        weight = float(graph.adj[candidate][neighbor].get('weight', 0))
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

      progress = min(cand_pop, gap) / gap
      overshoot = max(0, cand_pop - gap) / gap
      distance_penalty = 0.0
      if seed_center and distance_scale_km and distance_scale_km > 0:
        cand_center = cbg_centers.get(candidate) or cbg_centers.get(normalize_output_cbg(candidate))
        if cand_center:
          dist_km = distance(
            seed_center[0], seed_center[1],
            cand_center[0], cand_center[1]
          )
          distance_penalty = dist_km / (dist_km + distance_scale_km)

      score = (
        alpha * czi_after
        + (1 - alpha) * progress
        - overshoot_penalty * overshoot
        - distance_penalty_weight * distance_penalty
      )

      candidate_details.append({
        'cbg': normalize_output_cbg(candidate),
        'population': int(cand_pop),
        'score': float(score),
        'movement_to_cluster': float(in_to_cluster),
        'movement_to_outside': float(out_to_outside),
        'czi_after': float(czi_after),
        'progress': float(progress),
        'overshoot': float(overshoot),
        'distance_penalty': float(distance_penalty),
        'movement_inside_after': float(inside_after),
        'movement_boundary_after': float(boundary_after),
      })

  else:
    # For czi_optimal_cap (MILP) there is no single-step greedy frontier score. We expose a
    # consistent heuristic ranking by movement-to-cluster so the edit UI remains usable.
    for candidate in frontier:
      if candidate in cluster_set or candidate not in graph:
        continue

      cand_pop = int(cbg_population(candidate, config, logger) or 0)
      movement_to_cluster = 0.0
      movement_to_outside = 0.0

      for neighbor in graph.adj[candidate]:
        weight = float(graph.adj[candidate][neighbor].get('weight', 0))
        if neighbor in cluster_set:
          movement_to_cluster += weight
        else:
          movement_to_outside += weight

      if algorithm == 'greedy_ratio':
        total_movement = movement_to_cluster + movement_to_outside
        score = (movement_to_cluster / total_movement) if total_movement > 0 else 0.0
        candidate_details.append({
          'cbg': normalize_output_cbg(candidate),
          'population': int(cand_pop),
          'score': float(score),
          'movement_to_cluster': float(movement_to_cluster),
          'movement_to_outside': float(movement_to_outside),
          'movement_total': float(total_movement),
        })
      else:
        # greedy_fast, greedy_weight, and czi_optimal_cap fallback all use movement-to-cluster.
        candidate_details.append({
          'cbg': normalize_output_cbg(candidate),
          'population': int(cand_pop),
          'score': float(movement_to_cluster),
          'movement_to_cluster': float(movement_to_cluster),
          'movement_to_outside': float(movement_to_outside),
        })

  sorted_candidates = sorted(
    candidate_details,
    key=lambda item: (float(item.get('score', 0.0)), item.get('cbg', '')),
    reverse=True
  )

  if isinstance(limit, int) and limit > 0:
    sorted_candidates = sorted_candidates[:limit]

  for idx, candidate in enumerate(sorted_candidates):
    candidate['rank'] = idx + 1
    candidate['selected'] = False

  return sorted_candidates, sorted(set(missing_cluster_cbgs))


def create_cz(data, report):
  """Create convenience zone and schedule data generation."""
  """Create convenience zone and schedule data generation."""
  report.info(f"Generating CZ from CBG: {data['cbg']}")
  report.info(f"Target minimum population: {data['min_pop']}")
  
  algorithm = _normalize_cluster_algorithm(data.get('algorithm'))
  if not algorithm:
    raise ValueError(
      f"Invalid clustering algorithm '{data.get('algorithm')}'. "
      f"Valid options: {', '.join(sorted(VALID_CLUSTER_ALGORITHMS))}"
    )

  czi_params = {}
  optimal_params = {}
  if algorithm == 'czi_balanced':
    czi_params, err = _parse_czi_balanced_params(data)
    if err:
      raise ValueError(err)
  elif algorithm == 'czi_optimal_cap':
    optimal_params, err = _parse_czi_optimal_params(data)
    if err:
      raise ValueError(err)

  geoids, map, gdf = generate_cz(
    data['cbg'],
    data['min_pop'],
    algorithm=algorithm,
    distance_penalty_weight=czi_params.get('distance_penalty_weight'),
    distance_scale_km=czi_params.get('distance_scale_km'),
    optimal_candidate_limit=optimal_params.get('optimal_candidate_limit'),
    optimal_population_floor_ratio=optimal_params.get('optimal_population_floor_ratio'),
    optimal_mip_rel_gap=optimal_params.get('optimal_mip_rel_gap'),
    optimal_time_limit_sec=optimal_params.get('optimal_time_limit_sec'),
    optimal_max_iters=optimal_params.get('optimal_max_iters')
  )

  cluster = list(geoids.keys())
  size = sum(list(geoids.values()))
  
  report.info(f'Clustered {len(cluster)} CBGs with total population {size}')
  report.debug(f'Cluster CBGs: {cluster}')
    
  resp = requests.post('http://localhost:1890/convenience-zones', json={
    'name': data['name'],
    'description': data['description'],
    'latitude': map.location[0],
    'longitude': map.location[1],
    'cbg_list': cluster,
    'start_date': data['start_date'],
    'length': data['length'],
    'size': size,
    'user_id': data['user_id']
  })
    
  if not resp.ok:
    detail = ''
    try:
      detail = resp.text[:500]
    except Exception:
      detail = ''
    report.fail(f'Error creating CZ record: HTTP {resp.status_code}')
    return make_response(jsonify({
      'message': f'Error creating CZ record (HTTP {resp.status_code})',
      'detail': detail
    }), 500)
  
  czone_id = resp.json()['data']['id']
  report.info(f'Created CZ record with ID: {czone_id}')
  
  @after_this_request
  def call_after_request(response):
    start_date = data['start_date'].replace("Z", "+00:00")
    start_date = datetime.fromisoformat(start_date)
    gen_and_upload_data(geoids, czone_id, start_date, data.get('length', 168), report, gdf=gdf)
    return response
    
  return json.dumps({
    'id': czone_id,
    'cluster': cluster,
    'size': size,
    'map': map._repr_html_()
  })

@app.route('/generate-cz', methods=['POST'])
@cross_origin()
def route_generate_cz():
  try:
    request.get_json(force=True)
  except:
    return make_response(jsonify({'message': 'Bad Request'}), 400)
  
  if not request.json:
    return make_response(jsonify({'message': 'Please supply adequate JSON data'}), 400)

  # Normalize and validate CBG early so we can return clearer errors.
  cbg = request.json.get('cbg')
  if cbg is None:
    return make_response(jsonify({'message': "Missing required field: 'cbg'"}), 400)

  cbg_str = str(cbg).strip()
  if not cbg_str.isdigit():
    return make_response(
      jsonify({'message': "Invalid 'cbg': must be digits only (12-digit Census Block Group GEOID)"}),
      400
    )

  # Some GEOIDs may be provided without leading zeros. If 11 digits, pad to 12.
  if len(cbg_str) == 11:
    cbg_str = cbg_str.zfill(12)
    request.json['cbg'] = cbg_str

  if len(cbg_str) != 12 or not re.fullmatch(r"\d{12}", cbg_str):
    return make_response(
      jsonify({
        'message': "Invalid 'cbg': expected exactly 12 digits (e.g., '060590117212')",
        'cbg_received': str(cbg),
        'cbg_normalized': cbg_str
      }),
      400
    )

  algorithm = _normalize_cluster_algorithm(request.json.get('algorithm'))
  if not algorithm:
    return make_response(
      jsonify({
        'message': f"Invalid 'algorithm'. Valid options: {', '.join(sorted(VALID_CLUSTER_ALGORITHMS))}"
      }),
      400
    )
  request.json['algorithm'] = algorithm

  if algorithm == 'czi_balanced':
    czi_params, err = _parse_czi_balanced_params(request.json)
    if err:
      return make_response(jsonify({'message': err}), 400)
    request.json.update({k: v for k, v in czi_params.items() if v is not None})
  elif algorithm == 'czi_optimal_cap':
    optimal_params, err = _parse_czi_optimal_params(request.json)
    if err:
      return make_response(jsonify({'message': err}), 400)
    request.json.update({k: v for k, v in optimal_params.items() if v is not None})
  
  try:
    validate(instance=request.json, schema=gen_cz_schema)
  except Exception as e:
    print(f'Validation error: {e}')
    print(f'Received JSON: {request.json}')
    return make_response(
      jsonify({
        'message': f'JSON data not valid: {str(e)}',
        'hint': "Check 'cbg' is a 12-digit Census Block Group GEOID (leading zeros matter)."
      }),
      400
    )
  
  # Create run report
  data = request.json.copy()
  report = RunReport(
    run_type="cz_generation",
    name=f"CZ Generation: {data.get('name', data['cbg'])}",
    parameters={
      'cbg': data['cbg'],
      'name': data.get('name'),
      'description': data.get('description'),
      'min_pop': data.get('min_pop'),
      'algorithm': _normalize_cluster_algorithm(data.get('algorithm')) or data.get('algorithm'),
      'distance_penalty_weight': data.get('distance_penalty_weight'),
      'distance_scale_km': data.get('distance_scale_km'),
      'optimal_candidate_limit': data.get('optimal_candidate_limit'),
      'optimal_population_floor_ratio': data.get('optimal_population_floor_ratio'),
      'optimal_mip_rel_gap': data.get('optimal_mip_rel_gap'),
      'optimal_time_limit_sec': data.get('optimal_time_limit_sec'),
      'optimal_max_iters': data.get('optimal_max_iters'),
      'length': data.get('length'),
    },
    user_id=data.get('user_id'),
  )
  
  try:
    return create_cz(data, report)
  except Exception as e:
    report.capture_exception()
    return make_response(jsonify({'message': str(e)}), 500)


@app.route('/cluster-cbgs', methods=['POST'])
@cross_origin()
def route_cluster_cbgs():
  """
  Phase 1: Just cluster CBGs based on a seed CBG and min population.
  Returns the cluster for user to edit before pattern generation.
  Does NOT create a DB record or generate patterns yet.
  """
  try:
    request.get_json(force=True)
  except:
    return make_response(jsonify({'message': 'Bad Request'}), 400)
  
  if not request.json:
    return make_response(jsonify({'message': 'Please supply adequate JSON data'}), 400)

  use_test_data = bool(request.json.get('use_test_data'))
  cbg = request.json.get('cbg')
  min_pop = request.json.get('min_pop', 5000)
  algorithm = _normalize_cluster_algorithm(request.json.get('algorithm'))

  if not algorithm:
    return make_response(jsonify({
      'message': f"Invalid 'algorithm'. Valid options: {', '.join(sorted(VALID_CLUSTER_ALGORITHMS))}"
    }), 400)

  czi_params = {}
  optimal_params = {}
  if algorithm == 'czi_balanced':
    czi_params, err = _parse_czi_balanced_params(request.json)
    if err:
      return make_response(jsonify({'message': err}), 400)
  elif algorithm == 'czi_optimal_cap':
    optimal_params, err = _parse_czi_optimal_params(request.json)
    if err:
      return make_response(jsonify({'message': err}), 400)

  effective_czi_params = {}
  effective_optimal_params = {}
  if algorithm == 'czi_balanced':
    effective_czi_params = {
      'distance_penalty_weight': (
        czi_params.get('distance_penalty_weight')
        if czi_params.get('distance_penalty_weight') is not None
        else DEFAULT_DISTANCE_PENALTY_WEIGHT
      ),
      'distance_scale_km': (
        czi_params.get('distance_scale_km')
        if czi_params.get('distance_scale_km') is not None
        else DEFAULT_DISTANCE_SCALE_KM
      ),
    }
  elif algorithm == 'czi_optimal_cap':
    effective_optimal_params = {
      'optimal_candidate_limit': (
        optimal_params.get('optimal_candidate_limit')
        if optimal_params.get('optimal_candidate_limit') is not None
        else DEFAULT_OPTIMAL_CANDIDATE_LIMIT
      ),
      'optimal_population_floor_ratio': (
        optimal_params.get('optimal_population_floor_ratio')
        if optimal_params.get('optimal_population_floor_ratio') is not None
        else DEFAULT_OPTIMAL_POP_FLOOR_RATIO
      ),
      'optimal_mip_rel_gap': (
        optimal_params.get('optimal_mip_rel_gap')
        if optimal_params.get('optimal_mip_rel_gap') is not None
        else DEFAULT_OPTIMAL_MIP_REL_GAP
      ),
      'optimal_time_limit_sec': (
        optimal_params.get('optimal_time_limit_sec')
        if optimal_params.get('optimal_time_limit_sec') is not None
        else DEFAULT_OPTIMAL_TIME_LIMIT_SEC
      ),
      'optimal_max_iters': (
        optimal_params.get('optimal_max_iters')
        if optimal_params.get('optimal_max_iters') is not None
        else DEFAULT_OPTIMAL_MAX_ITERS
      ),
    }

  patterns_file = None
  patterns_source = 'default'
  patterns_month = None
  if use_test_data:
    ok, missing, _headers = _validate_csv_columns(TEST_PATTERNS_FILE, TEST_CLUSTER_COLUMNS)
    if not ok:
      return make_response(jsonify({
        'message': f"TEST data is missing required columns for clustering: {', '.join(missing)}",
        'required_columns': TEST_CLUSTER_COLUMNS,
        'test_file': TEST_PATTERNS_FILE
      }), 400)
    patterns_file = TEST_PATTERNS_FILE
    if cbg is None:
      cbg = _get_test_seed_cbg(TEST_PATTERNS_FILE)
      if cbg is None:
        return make_response(jsonify({
          'message': "TEST data must include at least one valid 12-digit 'poi_cbg' value to infer the seed CBG",
          'test_file': TEST_PATTERNS_FILE
        }), 400)

  if cbg is None:
    return make_response(jsonify({'message': "Missing required field: 'cbg'"}), 400)

  cbg_str = _normalize_cbg(cbg)
  if not cbg_str:
    return make_response(jsonify({'message': "Invalid 'cbg': expected exactly 12 digits"}), 400)

  if not use_test_data:
    patterns_month = _extract_month_key(request.json.get('start_date'))
    if patterns_month:
      monthly_file = _resolve_monthly_patterns_file(cbg_str, patterns_month)
      if monthly_file:
        patterns_file = monthly_file
        patterns_source = 'monthly'
    if not patterns_file:
      default_file = _default_patterns_file()
      if default_file:
        patterns_file = default_file
        patterns_source = 'default'
      else:
        if patterns_month:
          return make_response(jsonify({
            'message': (
              f"No monthly patterns file found for month '{patterns_month}' and no default "
              f"patterns file is available. Expected something like "
              f"'{DATA_DIR}/<STATE>/{patterns_month}-<STATE>.csv'."
            ),
            'month': patterns_month,
            'cbg': cbg_str
          }), 400)
        return make_response(jsonify({
          'message': (
            "No default patterns file found. Provide a start date with available monthly files "
            f"or add '{DATA_DIR}/patterns_o.csv'."
          )
        }), 400)
  
  try:
    geoids, center, trace_payload = cluster_cbgs(
      cbg_str,
      min_pop,
      patterns_file=patterns_file,
      month=patterns_month,
      algorithm=algorithm,
      czi_params=effective_czi_params,
      optimal_params=effective_optimal_params,
      include_trace=True
    )
    cluster = list(geoids.keys())
    size = sum(list(geoids.values()))
    
    # Also return GeoJSON for the map
    geojson = get_cbg_geojson(cluster, include_neighbors=True)
    trace_geojson = None

    if trace_payload and trace_payload.get('steps'):
      trace_cbgs = set(cluster)
      for step in trace_payload.get('steps', []):
        trace_cbgs.update(step.get('cluster_before', []))
        trace_cbgs.update(step.get('cluster_after', []))
        for candidate in step.get('candidates', []):
          candidate_cbg = candidate.get('cbg')
          if candidate_cbg:
            trace_cbgs.add(candidate_cbg)
      if trace_cbgs:
        trace_geojson = get_cbg_geojson(list(trace_cbgs), include_neighbors=False)
    
    return jsonify({
      'cluster': cluster,
      'seed_cbg': cbg_str,
      'size': size,
      'center': center,
      'geojson': geojson,
      'algorithm': algorithm,
      'clustering_params': (
        effective_czi_params
        if algorithm == 'czi_balanced'
        else effective_optimal_params if algorithm == 'czi_optimal_cap' else {}
      ),
      'patterns_file_used': patterns_file,
      'patterns_source': patterns_source,
      'patterns_month': patterns_month,
      'use_test_data': use_test_data,
      'trace': trace_payload,
      'trace_geojson': trace_geojson
    })
  except ValueError as e:
    return make_response(jsonify({'message': str(e)}), 400)
  except Exception as e:
    print(f'Error clustering CBGs: {e}')
    import traceback
    traceback.print_exc()
    return make_response(jsonify({'message': str(e)}), 500)


@app.route('/finalize-cz', methods=['POST'])
@cross_origin()
def route_finalize_cz():
  """
  Phase 2: Create the CZ record and generate patterns for the final CBG list.
  Called after user has edited their CBG selection.
  """
  try:
    request.get_json(force=True)
  except:
    return make_response(jsonify({'message': 'Bad Request'}), 400)
  
  if not request.json:
    return make_response(jsonify({'message': 'Please supply adequate JSON data'}), 400)

  data = request.json
  use_test_data = bool(data.get('use_test_data'))
  cbg_list = data.get('cbg_list', [])
  
  if not cbg_list or not isinstance(cbg_list, list):
    return make_response(jsonify({'message': "Missing or invalid 'cbg_list'"}), 400)

  patterns_file = None
  if use_test_data:
    ok, missing, _headers = _validate_csv_columns(TEST_PATTERNS_FILE, TEST_SIM_COLUMNS)
    if not ok:
      return make_response(jsonify({
        'message': f"TEST data is missing required columns for simulation patterns: {', '.join(missing)}",
        'required_columns': TEST_SIM_COLUMNS,
        'test_file': TEST_PATTERNS_FILE
      }), 400)
    patterns_file = TEST_PATTERNS_FILE
  
  # Create run report
  report = RunReport(
    run_type="cz_generation",
    name=f"CZ Generation: {data.get('name', 'unnamed')}",
    parameters={
      'cbg_list': cbg_list,
      'name': data.get('name'),
      'description': data.get('description'),
      'length': data.get('length'),
    },
    user_id=data.get('user_id'),
  )
  
  try:
    # Get population for each CBG
    from czcode import Config, cbg_population, setup_logging
    # Use first CBG to initialize config (just for population lookup)
    config = Config(cbg_list[0], 0)
    logger = setup_logging(config)
    geoids = { cbg: cbg_population(cbg, config, logger) for cbg in cbg_list }
    
    cluster = list(geoids.keys())
    size = sum(list(geoids.values()))
    
    report.info(f'Finalizing CZ with {len(cluster)} CBGs, total population {size}')
    
    # Get center from first CBG's location (approximate)
    latitude = data.get('latitude', 0)
    longitude = data.get('longitude', 0)
    
    # Create DB record
    resp = requests.post('http://localhost:1890/convenience-zones', json={
      'name': data.get('name', ''),
      'description': data.get('description', ''),
      'latitude': latitude,
      'longitude': longitude,
      'cbg_list': cluster,
      'start_date': data.get('start_date'),
      'length': data.get('length', 168),
      'size': size,
      'user_id': data.get('user_id')
    })
    
    if not resp.ok:
      report.fail(f'Error creating CZ record: HTTP {resp.status_code}')
      detail = ''
      try:
        detail = resp.text[:500]
      except Exception:
        detail = ''
      return make_response(jsonify({
        'message': f'Error creating CZ record (HTTP {resp.status_code})',
        'detail': detail
      }), 500)
    
    czone_id = resp.json()['data']['id']
    report.info(f'Created CZ record with ID: {czone_id}')
    
    # Schedule pattern generation
    @after_this_request
    def call_after_request(response):
      start_date = data['start_date'].replace("Z", "+00:00")
      start_date = datetime.fromisoformat(start_date)
      gen_and_upload_data(
        geoids,
        czone_id,
        start_date,
        data.get('length', 168),
        report,
        patterns_file=patterns_file
      )
      return response
    
    return jsonify({
      'id': czone_id,
      'cluster': cluster,
      'size': size
    })
  except Exception as e:
    report.capture_exception()
    import traceback
    traceback.print_exc()
    return make_response(jsonify({'message': str(e)}), 500)


@app.route('/cz-metrics', methods=['POST'])
@cross_origin()
def route_cz_metrics():
  """
  Compute movement-based CZ metrics, including CZI:
    CZI = movement_inside_zone / (movement_inside_zone + movement_crossing_zone_boundary)
  """
  try:
    request.get_json(force=True)
  except:
    return make_response(jsonify({'message': 'Bad Request'}), 400)

  if not request.json:
    return make_response(jsonify({'message': 'Please supply adequate JSON data'}), 400)

  use_test_data = bool(request.json.get('use_test_data'))
  seed_cbg = request.json.get('seed_cbg')
  cbg_list = request.json.get('cbg_list', [])

  if seed_cbg is None:
    return make_response(jsonify({'message': "Missing required field: 'seed_cbg'"}), 400)

  if not isinstance(cbg_list, list) or len(cbg_list) == 0:
    return make_response(jsonify({'message': "Missing or invalid 'cbg_list'"}), 400)

  seed_cbg = _normalize_cbg(seed_cbg)
  if not seed_cbg:
    return make_response(jsonify({'message': "Invalid 'seed_cbg': expected exactly 12 digits"}), 400)

  patterns_file = None
  if use_test_data:
    ok, missing, _headers = _validate_csv_columns(TEST_PATTERNS_FILE, TEST_CLUSTER_COLUMNS)
    if not ok:
      return make_response(jsonify({
        'message': f"TEST data is missing required columns for clustering metrics: {', '.join(missing)}",
        'required_columns': TEST_CLUSTER_COLUMNS,
        'test_file': TEST_PATTERNS_FILE
      }), 400)
    patterns_file = TEST_PATTERNS_FILE

  normalized_cbgs = []
  seen = set()
  for cbg in cbg_list:
    cbg_str = _normalize_cbg(cbg)
    if not cbg_str:
      continue
    if cbg_str in seen:
      continue
    seen.add(cbg_str)
    normalized_cbgs.append(cbg_str)

  if not normalized_cbgs:
    return make_response(jsonify({'message': "No valid CBGs in 'cbg_list'"}), 400)

  try:
    graph = get_cached_mobility_graph(seed_cbg, patterns_file=patterns_file, cache_tag='v3')
    graph_cluster_cbgs = []
    seen_graph_ids = set()
    for cbg in normalized_cbgs:
      try:
        gkey = str(int(float(cbg)))
      except (TypeError, ValueError):
        gkey = str(cbg).strip()
      if not gkey or gkey in seen_graph_ids:
        continue
      seen_graph_ids.add(gkey)
      graph_cluster_cbgs.append(gkey)

    movement_stats = Helpers.calculate_movement_stats(graph, graph_cluster_cbgs)
    movement_inside = float(movement_stats.get('in', 0))
    movement_boundary = float(movement_stats.get('out', 0))
    czi = float(movement_stats.get('ratio', 0))

    return jsonify({
      'movement_inside': movement_inside,
      'movement_boundary': movement_boundary,
      'czi': czi,
      # Backward-compatible alias of CZI.
      'containment_ratio': float(movement_stats.get('ratio', 0)),
      'cbg_count': len(normalized_cbgs),
    })
  except ValueError as e:
    return make_response(jsonify({'message': str(e)}), 400)
  except Exception as e:
    print(f'Error computing CZ metrics: {e}')
    return make_response(jsonify({'message': f'Error computing CZ metrics: {str(e)}'}), 500)


@app.route('/frontier-candidates', methods=['POST'])
@cross_origin()
def route_frontier_candidates():
  """
  Compute and rank the current frontier candidates for an arbitrary edited cluster.
  This powers the manual "Edit Zone" UI after the initial preview generation.
  """
  try:
    request.get_json(force=True)
  except:
    return make_response(jsonify({'message': 'Bad Request'}), 400)

  if not request.json:
    return make_response(jsonify({'message': 'Please supply adequate JSON data'}), 400)

  use_test_data = bool(request.json.get('use_test_data'))
  seed_cbg = _normalize_cbg(request.json.get('seed_cbg'))
  cbg_list = request.json.get('cbg_list', [])
  algorithm = _normalize_cluster_algorithm(request.json.get('algorithm')) or 'greedy_weight'
  start_date = request.json.get('start_date')
  min_pop = request.json.get('min_pop', 0)
  limit = request.json.get('limit')

  if not seed_cbg:
    return make_response(jsonify({'message': "Invalid or missing 'seed_cbg'"}), 400)
  if not isinstance(cbg_list, list) or len(cbg_list) == 0:
    return make_response(jsonify({'message': "Missing or invalid 'cbg_list'"}), 400)

  normalized_cluster = []
  seen = set()
  for cbg in cbg_list:
    cbg_norm = _normalize_cbg(cbg)
    if not cbg_norm or cbg_norm in seen:
      continue
    seen.add(cbg_norm)
    normalized_cluster.append(cbg_norm)
  if not normalized_cluster:
    return make_response(jsonify({'message': "No valid CBGs in 'cbg_list'"}), 400)

  parsed_limit = None
  if limit is not None:
    try:
      parsed_limit = int(limit)
    except (TypeError, ValueError):
      return make_response(jsonify({'message': "Invalid 'limit': expected integer"}), 400)
    if parsed_limit <= 0:
      parsed_limit = None

  czi_params = {}
  if algorithm == 'czi_balanced':
    czi_params, err = _parse_czi_balanced_params(request.json)
    if err:
      return make_response(jsonify({'message': err}), 400)

  try:
    patterns_file, patterns_source, patterns_month = _resolve_patterns_file_for_request(
      seed_cbg,
      start_date_raw=start_date,
      use_test_data=use_test_data
    )
  except ValueError as e:
    return make_response(jsonify({'message': str(e)}), 400)

  try:
    graph = get_cached_mobility_graph(seed_cbg, patterns_file=patterns_file, cache_tag='v3')
    candidates, missing_cluster_cbgs = _rank_frontier_candidates_for_cluster(
      graph=graph,
      seed_cbg=seed_cbg,
      cluster_cbgs=normalized_cluster,
      algorithm=algorithm,
      min_pop=_safe_float(min_pop, 0),
      czi_params=czi_params,
      patterns_file=patterns_file,
      month=patterns_month,
      limit=parsed_limit,
    )
    return jsonify({
      'seed_cbg': seed_cbg,
      'cluster_size': len(normalized_cluster),
      'candidate_count': len(candidates),
      'candidates': candidates,
      'algorithm': algorithm,
      'patterns_file_used': patterns_file,
      'patterns_source': patterns_source,
      'patterns_month': patterns_month,
      'missing_cluster_cbgs': missing_cluster_cbgs,
      'use_test_data': use_test_data,
    })
  except ValueError as e:
    return make_response(jsonify({'message': str(e)}), 400)
  except Exception as e:
    print(f'Error computing frontier candidates: {e}')
    return make_response(jsonify({'message': f'Error computing frontier candidates: {str(e)}'}), 500)


@app.route('/candidate-pois', methods=['POST'])
@cross_origin()
def route_candidate_pois():
  """
  Return top POIs in a candidate CBG that receive visits from the current cluster.
  """
  try:
    request.get_json(force=True)
  except:
    return make_response(jsonify({'message': 'Bad Request'}), 400)

  if not request.json:
    return make_response(jsonify({'message': 'Please supply adequate JSON data'}), 400)

  use_test_data = bool(request.json.get('use_test_data'))
  seed_cbg = _normalize_cbg(request.json.get('seed_cbg'))
  candidate_cbg = _normalize_cbg(request.json.get('candidate_cbg'))
  cluster_cbgs = request.json.get('cluster_cbgs', [])
  start_date = request.json.get('start_date')
  limit = request.json.get('limit', 8)

  if not seed_cbg:
    return make_response(jsonify({'message': "Invalid or missing 'seed_cbg'"}), 400)
  if not candidate_cbg:
    return make_response(jsonify({'message': "Invalid or missing 'candidate_cbg'"}), 400)
  if not isinstance(cluster_cbgs, list) or len(cluster_cbgs) == 0:
    return make_response(jsonify({'message': "Missing or invalid 'cluster_cbgs'"}), 400)

  normalized_cluster = []
  seen = set()
  for cbg in cluster_cbgs:
    cbg_norm = _normalize_cbg(cbg)
    if not cbg_norm or cbg_norm in seen:
      continue
    seen.add(cbg_norm)
    normalized_cluster.append(cbg_norm)

  if not normalized_cluster:
    return make_response(jsonify({'message': "No valid CBGs in 'cluster_cbgs'"}), 400)

  try:
    patterns_file, patterns_source, patterns_month = _resolve_patterns_file_for_request(
      seed_cbg,
      start_date_raw=start_date,
      use_test_data=use_test_data
    )
  except ValueError as e:
    return make_response(jsonify({'message': str(e)}), 400)

  try:
    pois = _compute_top_candidate_pois(
      patterns_file,
      candidate_cbg=candidate_cbg,
      cluster_cbgs=normalized_cluster,
      limit=limit
    )
    return jsonify({
      'candidate_cbg': candidate_cbg,
      'cluster_size': len(normalized_cluster),
      'pois': pois,
      'patterns_file_used': patterns_file,
      'patterns_source': patterns_source,
      'patterns_month': patterns_month,
      'use_test_data': use_test_data,
    })
  except Exception as e:
    print(f'Error computing candidate POIs: {e}')
    return make_response(jsonify({'message': f'Error computing candidate POIs: {str(e)}'}), 500)


@app.route('/cbg-geojson', methods=['GET'])
@cross_origin()
def route_cbg_geojson():
  """
  Returns GeoJSON for the specified CBGs.
  Query params:
    - cbgs: comma-separated list of CBG IDs
    - include_neighbors: if 'true', also include neighboring CBGs (optional)
  """
  cbgs_param = request.args.get('cbgs', '')
  include_neighbors = request.args.get('include_neighbors', 'false').lower() == 'true'
  
  if not cbgs_param:
    return make_response(jsonify({'message': 'Missing cbgs parameter'}), 400)
  
  cbg_list = [cbg.strip() for cbg in cbgs_param.split(',') if cbg.strip()]
  
  if not cbg_list:
    return make_response(jsonify({'message': 'No valid CBGs provided'}), 400)
  
  try:
    geojson = get_cbg_geojson(cbg_list, include_neighbors=include_neighbors)
    return jsonify(geojson)
  except Exception as e:
    print(f'Error generating GeoJSON: {e}')
    return make_response(jsonify({'message': f'Error generating GeoJSON: {str(e)}'}), 500)


@app.route('/cbg-at-point', methods=['GET'])
@cross_origin()
def route_cbg_at_point():
  """
  Returns the CBG containing a latitude/longitude.
  Query params:
    - latitude: required
    - longitude: required
    - state_fips: optional 2-digit state FIPS hint to speed lookup
  """
  latitude = request.args.get('latitude')
  longitude = request.args.get('longitude')
  state_fips = request.args.get('state_fips')

  if latitude is None or longitude is None:
    return make_response(jsonify({'message': 'Missing latitude/longitude parameters'}), 400)

  try:
    lat = float(latitude)
    lng = float(longitude)
  except (TypeError, ValueError):
    return make_response(jsonify({'message': 'Invalid latitude/longitude'}), 400)

  if lat < -90 or lat > 90 or lng < -180 or lng > 180:
    return make_response(jsonify({'message': 'Latitude/longitude out of range'}), 400)

  if state_fips:
    state_fips = str(state_fips).strip().zfill(2)

  try:
    result = get_cbg_at_point(lat, lng, state_fips=state_fips)
    if not result:
      return make_response(jsonify({'message': 'No CBG found at that location'}), 404)

    return jsonify({
      'cbg': result['GEOID'],
      'population': result['population'],
    })
  except Exception as e:
    print(f'Error resolving CBG at point: {e}')
    return make_response(jsonify({'message': f'Error resolving CBG at point: {str(e)}'}), 500)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=1880)
