from flask import Flask, request, jsonify, make_response, Response
from flask_cors import CORS, cross_origin
from czcode import generate_cz
from popgen import gen_pop
from patterns import gen_patterns
from datetime import datetime
import threading
import requests
import json

app = Flask(__name__)
CORS(app)

def gen_and_upload_data(geoids, czone_id, start_date):
  # Generate People, Households, Places data
  print('generating papdata...')
  papdata = gen_pop(geoids)
  
  # Generate movement patterns
  print('generating patterns...')
  patterns = gen_patterns(papdata, start_date, 168)
      
  print('sending data...')
    
  resp = requests.post('http://localhost:1890/patterns', json={
    'czone_id': czone_id,
    'papdata': papdata,
    'patterns': patterns
  })
  
  if resp.ok:
    print('sent!')
  else:
    print('error sending data...')

def create_cz(data):
  yield ''
  
  geoids, map = generate_cz(data['cbg'], data['min_pop'])

  cluster = list(geoids.keys())
  size = sum(list(geoids.values()))
    
  resp = requests.post('http://localhost:1890/convenience-zones', json={
    'name': data['name'],
    'latitude': map.location[0],
    'longitude': map.location[1],
    'cbg_list': cluster,
    'start_date': data['start_date'],
    'size': size
  })
    
  if not resp.ok:
    return '{}'
  
  czone_id = resp.json()['data']['id']
  
  thread = threading.Thread(target=gen_and_upload_data, args=(geoids, czone_id, datetime.fromisoformat(data['start_date'])))
  thread.daemon = True
  thread.start()
  
  yield json.dumps({
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
    return make_response(jsonify({'message': 'Please specify a CBG, location name, start date, and minimum population'}), 400)
  
  return Response(create_cz(request.json.copy()), mimetype='text/plain')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=1880)
