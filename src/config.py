METHOD = 'api'
DATASET = 'cora'

MODEL = "gpt-4o"
FILE_NAME = f"/home/msun415/induction/data/{METHOD}_{DATASET}.txt"
IMG_DIR = f"/home/msun415/induction/data/{METHOD}_{DATASET}/"
CACHE_DIR = f"/home/msun415/induction/cache/{METHOD}_{DATASET}/"

NONTERMS = ['gray','black']
SEED = 0
MAX_ATTEMPTS = 5
SCALE = 10
NODE_SIZE = 5000
FONT_SIZE = 20
MAX_SIZE = 3000*3000*4
LAYOUT = 'kamada_kawai_layout'

RADIUS = 1
# CMAP_NAME = 'twilight'
CMAP_NAME = 'twilight'
