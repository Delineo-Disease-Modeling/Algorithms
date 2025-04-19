from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import json
from czcode import generate_cz
from popgen import gen_pop

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
    
  # resp = requests.post('http://localhost:1890/convenience-zones', json={
  #   'name': request.json['name'],
  #   'label': request.json['name'],
  #   'latitude': map.location[0],
  #   'longitude': map.location[1],
  #   'cbg_list': cluster,
  #   'size': size
  # })
  
  # if not resp.ok:
  #   return make_response(jsonify({
  #     'message': 'Could not upload cluster to database'  
  #   }), 500)
  
  # Generate People, Households, Places data
  papdata = gen_pop(geoids)
  
  # Generate movement patterns
  
  
  return jsonify({
    #'id': resp.json()['data']['id'],
    'cluster': cluster,
    'size': size,
    'map': map._repr_html_()
  })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=1880)
