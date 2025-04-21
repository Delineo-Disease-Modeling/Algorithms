from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
from czcode import generate_cz
from popgen import gen_pop
from patterns import gen_patterns
from datetime import datetime
import json
import threading
import requests

app = Flask(__name__)
CORS(app)

def gen_and_upload_data(geoids, czone_id):
  # Generate People, Households, Places data
  print('generating papdata...')
  papdata = gen_pop(geoids)
  with open(r'./output/papdata.json', 'w') as f:
    json.dump(papdata, f, indent=4)
  
  # Generate movement patterns
  print('generating patterns...')
  patterns = gen_patterns(papdata, datetime.now(), 168)
  
  with open(r'./output/patterns.json', 'w') as f:
    json.dump(patterns, f, indent=4)
    
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

@app.route('/generate-cz', methods=['POST'])
@cross_origin()
def route_generate_cz():
  try:
    request.get_json(force=True)
  except:
    return make_response(jsonify({'message': 'Bad Request'}), 400)
  
  if not request.json:
    return make_response(jsonify({'message': 'Please specify a CBG, location name, start date, and minimum population'}), 400)
  
  geoids, map = generate_cz(request.json['cbg'], request.json['min_pop'])

  cluster = list(geoids.keys())
  size = sum(list(geoids.values()))
    
  resp = requests.post('http://localhost:1890/convenience-zones', json={
    'name': request.json['name'],
    'latitude': map.location[0],
    'longitude': map.location[1],
    'cbg_list': cluster,
    'start_date': request.json['start_date'],
    'size': size
  })
    
  if not resp.ok:
    return make_response(jsonify({
      'message': 'Could not upload cluster to database'  
    }), 500)
  
  czone_id = resp.json()['data']['id']
  
  thread = threading.Thread(target=gen_and_upload_data, args=(geoids, czone_id))
  thread.daemon = True
  thread.start()
  
  return jsonify({
    'id': czone_id,
    'cluster': cluster,
    'size': size,
    'map': map._repr_html_()
  })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=1880)
