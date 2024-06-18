METHOD = 'api'
DATASET = 'test'
GRAMMAR = 'nlc'

MODEL = "gpt-4o"
FILE_NAME = f"/home/msun415/induction/data/{METHOD}_{DATASET}_{GRAMMAR}.txt"
IMG_DIR = f"/home/msun415/induction/data/{METHOD}_{DATASET}_{GRAMMAR}/"
CACHE_DIR = f"/home/msun415/induction/cache/{METHOD}_{DATASET}_{GRAMMAR}/"

NONTERMS = ['gray','black']
SEED = 0
NUM_ATTEMPTS = 3

# VISUALIZATION
SCALE = 50
NODE_SIZE = 5000
FONT_SIZE = 20
MAX_SIZE = 3000*3000*4
PARTITION_SIZE = 50
LAYOUT = 'kamada_kawai_layout'
# LAYOUT = 'spring_layout'

RADIUS = 1
# CMAP_NAME = 'twilight'
CMAP_NAME = 'twilight'
