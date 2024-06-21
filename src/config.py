METHOD = 'api'
DATASET = 'house'
GRAMMAR = 'ednce'

MODEL = "gpt-4o"
FILE_NAME = f"/home/msun415/induction/data/{METHOD}_{DATASET}_{GRAMMAR}.txt"
IMG_DIR = f"/home/msun415/induction/data/{METHOD}_{DATASET}_{GRAMMAR}/"
CACHE_DIR = f"/home/msun415/induction/cache/{METHOD}_{DATASET}_{GRAMMAR}/"

TERMS = ['cyan']
NONTERMS = ['gray','black']
NONFINAL = ['gray','black']
FINAL = ['red','blue','green']
SEED = 0
NUM_ATTEMPTS = 1

# VISUALIZATION
SCALE = 10
NODE_SIZE = 10000
FONT_SIZE = 20
MAX_SIZE = 3000*3000*4
LAYOUT = 'kamada_kawai_layout'
# LAYOUT = 'spring_layout'
PARTITION_SIZE = 50
PARTITION_SCALE = 5
PARTITION_NODE_SIZE = 500
MAX_IMAGES = 5

RADIUS = 1
# CMAP_NAME = 'twilight'
# CMAP_NAME = 'twilight'
