import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ceneje_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')