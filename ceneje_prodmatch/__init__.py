import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, 'src', 'config')
DATA_DIR = os.path.join(BASE_DIR, 'ceneje_data')
UNSPLITTED_DATA_DIR = os.path.join(DATA_DIR, 'unsplitted')
DEEPMATCH_DIR = os.path.join(BASE_DIR, 'deepmatcher_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')