import time
import itertools
from pyogrio import read_dataframe
from uszipcode import SearchEngine

print('Generating individual shapefiles...')

engine = SearchEngine()
geof = read_dataframe(r'./data/cbg_2020.geojson', columns=['State', 'CensusBlockGroup', 'geometry'])

print('Read US shapefile data...')

states = [ engine.find_state(state, best_match=False) for state in engine.state_list ]
states = sorted(list(set(itertools.chain.from_iterable(states))))

success = 0

for state in states:
  try:
    start = time.time()
    geof[geof['State'] == state].to_file(f'./data/shapefiles/{state}.geojson')  
    end = time.time()
    
    success += 1
    print(f'Processed \'{state}\' in {end - start:.2f}')
  except:
    print(f'Error processing \'{state}\'')
  

print(f'Generated {success}/{len(states)} shape files')
