from flask import Flask, request, jsonify, make_response, after_this_request
from flask_cors import CORS, cross_origin
from czcode import generate_cz
from popgen import gen_pop
from patterns import gen_patterns
from datetime import datetime
import requests
import json
from jsonschema import validate
from schema import gen_cz_schema
from io import BytesIO

app = Flask(__name__)
CORS(app,
  origins=['http://localhost:5173', 'https://coviddev.isi.jhu.edu', 'http://coviddev.isi.jhu.edu', 'https://covidweb.isi.jhu.edu', 'http://covidweb.isi.jhu.edu'],
  methods=['GET', 'HEAD', 'PUT', 'PATCH', 'POST', 'DELETE'],
  allow_headers=['Content-Type', 'Authorization'],
  expose_headers=['Set-Cookie'],
  supports_credentials=True
)

def gen_and_upload_data(geoids, czone_id, start_date, length):
  # Generate People, Households, Places data
  print('generating papdata...')
  papdata = gen_pop(geoids)
  
  # Generate movement patterns
  print('generating patterns...')
  patterns = gen_patterns(papdata, start_date, length)
      
  print('sending data...')

  resp = requests.post('http://localhost:1890/patterns', data={
    'czone_id': int(czone_id),
  }, files={
    'papdata': ('papdata.json', BytesIO(json.dumps(papdata).encode()), 'text/plain'),
    'patterns': ('patterns.json', BytesIO(json.dumps(patterns).encode()), 'text/plain')
  })
  
  if resp.ok:
    print('sent!')
  else:
    print(f'error sending data... {resp.status_code}')

def create_cz(data):
  geoids, map = generate_cz(data['cbg'], data['min_pop'])

  cluster = list(geoids.keys())
  size = sum(list(geoids.values()))
  
  print('CLUSTER LIST:')
  print(cluster)
    
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
    print(f'ERROR SENDING DATA - {resp.status_code}')
    return '{}'
  
  czone_id = resp.json()['data']['id']
  
  @after_this_request
  def call_after_request(response):
    start_date = data['start_date'].replace("Z", "+00:00")
    start_date = datetime.fromisoformat(start_date)
    gen_and_upload_data(geoids, czone_id, start_date, data.get('length', 168))
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
  
  try:
    validate(instance=request.json, schema=gen_cz_schema)
  except:
    return make_response(jsonify({'message': 'JSON data not valid'}), 400)
  
  return create_cz(request.json.copy())

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=1880)
