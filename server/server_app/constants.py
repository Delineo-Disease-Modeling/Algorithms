import os


SERVER_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(SERVER_DIR, 'data')
FULLSTACK_URL = os.environ.get('FULLSTACK_URL', 'http://localhost:3000')

CORS_ORIGINS = [
    'http://localhost:3000',
    'http://localhost:5173',
    'https://coviddev.isi.jhu.edu',
    'http://coviddev.isi.jhu.edu',
    'https://covidweb.isi.jhu.edu',
    'http://covidweb.isi.jhu.edu',
    'https://covidmod.isi.jhu.edu',
]

TEST_PATTERNS_FILE = os.path.join(DATA_DIR, 'TEST', 'test.csv')
TEST_CLUSTER_COLUMNS = ['poi_cbg', 'visitor_daytime_cbgs']
TEST_SIM_COLUMNS = ['placekey', 'median_dwell', 'popularity_by_hour', 'popularity_by_day']

VALID_CLUSTER_ALGORITHMS = {
    'czi_balanced',
    'czi_optimal_cap',
    'greedy_fast',
    'greedy_weight',
    'greedy_weight_seed_guard',
    'greedy_ratio',
    'greedy_ttwa',
}

DEFAULT_DISTANCE_PENALTY_WEIGHT = 0.02
DEFAULT_DISTANCE_SCALE_KM = 20.0
DEFAULT_SEED_GUARD_DISTANCE_KM = 20.0
DEFAULT_OPTIMAL_CANDIDATE_LIMIT = 120
DEFAULT_OPTIMAL_POP_FLOOR_RATIO = 0.9
DEFAULT_OPTIMAL_MIP_REL_GAP = 0.02
DEFAULT_OPTIMAL_TIME_LIMIT_SEC = 20.0
DEFAULT_OPTIMAL_MAX_ITERS = 8
DEFAULT_CONTAINMENT_THRESHOLD = 0.70
