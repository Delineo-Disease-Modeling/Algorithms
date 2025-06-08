from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import requests
from czcode import generate_cz
from household import Household, Person
from inter_hh import InterHousehold
import yaml

app = Flask(__name__)
CORS(app)

@app.route('/generate-cz', methods=['POST'])
@cross_origin()
def route_generate_cz():
  try:
    request.get_json(force=True)
  except:
    return make_response(jsonify({'message': 'Bad Request'}), 400)
  
  if not request.json:
    return make_response(jsonify({'message': 'Please specify a CBG, location name, and minimum population'}), 400)
  
  geoids, map = generate_cz(request.json['cbg'], request.json['zip_code'], request.json['name'], request.json['min_pop'])

  cluster = list(geoids.keys())
  size = sum(list(geoids.values()))
    
  resp = requests.post('http://localhost:1890/convenience-zones', json={
    'name': request.json['name'],
    'label': request.json['name'],
    'latitude': map.location[0],
    'longitude': map.location[1],
    'cbg_list': cluster,
    'size': size
  })
  
  if not resp.ok:
    return make_response(jsonify({
      'message': 'Could not upload cluster to database'  
    }), 500)
  
  return jsonify({
    'id': resp.json()['data']['id'],
    'cluster': cluster,
    'size': size,
    'map': map._repr_html_()
  })

@app.route('/generate-household-patterns', methods=['POST'])
@cross_origin()
def route_generate_household_patterns():
    try:
        request.get_json(force=True)
    except:
        return make_response(jsonify({'message': 'Bad Request'}), 400)
    
    if not request.json:
        return make_response(jsonify({'message': 'Please specify household data and config'}), 400)
    
    household_data = request.json.get('household_data', [])
    config = request.json.get('config', {})
    
    if not household_data or not config:
        return make_response(jsonify({'message': 'No household data or config provided'}), 400)
    
    households = []
    for hh_data in household_data:
        household = Household(
            cbg=hh_data.get('cbg'),
            total_count=hh_data.get('total_count', 0),
            population=[Person(**p_data) for p_data in hh_data.get('population', [])]
        )
        households.append(household)
    
    inter_household = InterHousehold(households, config)
    
    inter_household.next()
    
    patterns = {
        'households': [hh.to_dict() for hh in households],
        'people': [p.to_dict() for hh in households for p in hh.population]
    }
    
    resp = requests.post('http://localhost:1890/household-patterns', json=patterns)
    
    if not resp.ok:
        return make_response(jsonify({
            'message': 'Could not upload household patterns to database'
        }), 500)
    
    return jsonify({
        'id': resp.json()['data']['id'],
        'patterns': patterns
    })

@app.route('/generate-movement-patterns', methods=['POST'])
@cross_origin()
def route_generate_movement_patterns():
    try:
        request.get_json(force=True)
    except:
        return make_response(jsonify({'message': 'Bad Request'}), 400)
    
    if not request.json:
        return make_response(jsonify({'message': 'Please specify movement data'}), 400)
    
    movement_data = request.json.get('movement_data', {})
    if not movement_data:
        return make_response(jsonify({'message': 'No movement data provided'}), 400)
    
    patterns = {
        'movement_in': movement_data.get('movement_in', 0),
        'movement_out': movement_data.get('movement_out', 0),
        'ratio': movement_data.get('ratio', 0),
        'estimated_population': movement_data.get('estimated_population', 0),
        'social_tendency': 'low' if movement_data.get('ratio', 0) > 0.6 else 
                          'high' if movement_data.get('ratio', 0) < 0.4 else 
                          'medium'
    }
    
    resp = requests.post('http://localhost:1890/movement-patterns', json=patterns)
    
    if not resp.ok:
        return make_response(jsonify({
            'message': 'Could not upload movement patterns to database'
        }), 500)
    
    return jsonify({
        'id': resp.json()['data']['id'],
        'patterns': patterns
    })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=1880)
