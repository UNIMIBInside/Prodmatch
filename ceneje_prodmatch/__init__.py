import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, 'src', 'config')
if not os.path.exists(os.path.join(BASE_DIR, 'ceneje_data')):
    os.makedirs(os.path.join(BASE_DIR, 'ceneje_data'))
DATA_DIR = os.path.join(BASE_DIR, 'ceneje_data')
if not os.path.exists(os.path.join(DATA_DIR, 'unsplitted')):
    os.makedirs(os.path.join(DATA_DIR, 'unsplitted'))
UNSPLITTED_DATA_DIR = os.path.join(DATA_DIR, 'unsplitted')
if not os.path.exists(os.path.join(BASE_DIR, 'deepmatcher_data')):
    os.makedirs(os.path.join(BASE_DIR, 'deepmatcher_data'))
DEEPMATCH_DIR = os.path.join(BASE_DIR, 'deepmatcher_data')
if not os.path.exists(os.path.join(BASE_DIR, 'results')):
    os.makedirs(os.path.join(BASE_DIR, 'results'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
if not os.path.exists(os.path.join(RESULTS_DIR, 'models')):
    os.makedirs(os.path.join(RESULTS_DIR, 'models'))
if not os.path.exists(os.path.join(BASE_DIR, 'cache')):
    os.makedirs(os.path.join(BASE_DIR, 'cache'))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')