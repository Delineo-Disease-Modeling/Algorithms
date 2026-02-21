from flask import Flask, request, jsonify, make_response, after_this_request
from flask_cors import CORS, cross_origin
from czcode import generate_cz, Config, DataLoader, GraphBuilder, Helpers, setup_logging
from popgen import gen_pop
from patterns import gen_patterns
from geojsongen import get_cbg_geojson, get_cbg_at_point
from datetime import datetime
import os
import csv
import requests
import json
from jsonschema import validate
from schema import gen_cz_schema
from io import BytesIO
import re
from functools import lru_cache
from run_report import RunReport

app = Flask(__name__)
CORS(app,
  origins=['http://localhost:3000', 'http://localhost:5173', 'https://coviddev.isi.jhu.edu', 'http://coviddev.isi.jhu.edu', 'https://covidweb.isi.jhu.edu', 'http://covidweb.isi.jhu.edu', 'https://covidmod.isi.jhu.edu', 'http://covidweb.isi.jhu.edu'],
  methods=['GET', 'HEAD', 'PUT', 'PATCH', 'POST', 'DELETE'],
  allow_headers=['Content-Type', 'Authorization'],
  expose_headers=['Set-Cookie'],
  supports_credentials=True
)

TEST_PATTERNS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'TEST', 'test.csv')
TEST_CLUSTER_COLUMNS = ['poi_cbg', 'visitor_daytime_cbgs']
TEST_SIM_COLUMNS = ['placekey', 'median_dwell', 'popularity_by_hour', 'popularity_by_day']

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
    resp = requests.post('http://localhost:3000/api/patterns', data={
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


def cluster_cbgs(cbg, min_pop, patterns_file=None, patterns_folder=None, month=None):
  """Just cluster CBGs without generating patterns. Returns geoids dict and map center."""
  geoids, map_obj, _gdf = generate_cz(
    cbg,
    min_pop,
    patterns_file=patterns_file,
    patterns_folder=patterns_folder,
    month=month
  )
  return geoids, [map_obj.location[0], map_obj.location[1]]

@lru_cache(maxsize=8)
def get_cached_mobility_graph(seed_cbg, patterns_file=None, patterns_folder=None, month=None):
  """
  Build and cache the movement graph used by CZ clustering for a seed CBG.
  This allows fast repeated CZI updates while the user edits a zone.
  """
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


def create_cz(data, report):
  """Create convenience zone and schedule data generation."""
  """Create convenience zone and schedule data generation."""
  report.info(f"Generating CZ from CBG: {data['cbg']}")
  report.info(f"Target minimum population: {data['min_pop']}")
  
  geoids, map, gdf = generate_cz(data['cbg'], data['min_pop'])

  cluster = list(geoids.keys())
  size = sum(list(geoids.values()))
  
  report.info(f'Clustered {len(cluster)} CBGs with total population {size}')
  report.debug(f'Cluster CBGs: {cluster}')
    
  resp = requests.post('http://localhost:3000/api/convenience-zones', json={
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

  patterns_file = None
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
  
  try:
    geoids, center = cluster_cbgs(cbg_str, min_pop, patterns_file=patterns_file)
    cluster = list(geoids.keys())
    size = sum(list(geoids.values()))
    
    # Also return GeoJSON for the map
    geojson = get_cbg_geojson(cluster, include_neighbors=True)
    
    return jsonify({
      'cluster': cluster,
      'seed_cbg': cbg_str,
      'size': size,
      'center': center,
      'geojson': geojson,
      'use_test_data': use_test_data
    })
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
    resp = requests.post('http://localhost:3000/api/convenience-zones', json={
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
    CZI = movement_inside_zone / movement_crossing_zone_boundary
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
    graph = get_cached_mobility_graph(seed_cbg, patterns_file=patterns_file)
    movement_stats = Helpers.calculate_movement_stats(graph, normalized_cbgs)
    movement_inside = float(movement_stats.get('in', 0))
    movement_boundary = float(movement_stats.get('out', 0))
    czi = movement_inside / movement_boundary if movement_boundary > 0 else None

    return jsonify({
      'movement_inside': movement_inside,
      'movement_boundary': movement_boundary,
      'czi': czi,
      # Additional bounded metric (inside / total) for stable comparison.
      'containment_ratio': float(movement_stats.get('ratio', 0)),
      'cbg_count': len(normalized_cbgs),
    })
  except Exception as e:
    print(f'Error computing CZ metrics: {e}')
    return make_response(jsonify({'message': f'Error computing CZ metrics: {str(e)}'}), 500)


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
