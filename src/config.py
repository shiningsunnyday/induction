import os
import logging


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


wd = os.getcwd()

METHOD = "api"
DATASET = "enas"
GRAMMAR = "ednce"
SUFFIX = "" # suffix for log filepath, for concurrent runs

MODEL = "gpt-4o"
FILE_NAME = f"{wd}/data/{METHOD}_{DATASET}_{GRAMMAR}.txt"
IMG_DIR = f"{wd}/data/{METHOD}_{DATASET}_{GRAMMAR}/"
CACHE_DIR = f"{wd}/cache/{METHOD}_{DATASET}_{GRAMMAR}/"

NUM_THREADS = 1
NUM_PROCS = 1
VISUALIZE = True
VERBOSE = True
CACHE = False
MAX_TRIES = 10

# EDNCE parameters
CACHE_SUBG = True
LINEAR = True # at most 1 nt per comp, and subgraph must contain nt if present
MIN_EMBEDDING = False # take the minimum or maximum embedding for each rule

TOP_DIFF = 30
SEED = 5
NUM_ATTEMPTS = 1

# VISUALIZATION
SCALE = 5
DIM_SCALE = 7.5
NODE_SIZE = 4000
EDGE_THICKNESS = 2
ARROW_SIZE = 20
RULE_SCALE = 5
RULE_NODE_SIZE = 500
RULE_FONT_SIZE = 10
FONT_SIZE = 20
MAX_SIZE = 3000 * 3000 * 4
LAYOUT = "kamada_kawai_layout"
# LAYOUT = 'spring_layout'
TITLE_FONT_SIZE = 50
PARTITION_SIZE = 50
PARTITION_SCALE = 5
POS_EPSILON = 1
PARTITION_NODE_SIZE = 500
MAX_IMAGES = 5
NUM_COMPONENT_SAMPLES_FOR_MOTIFS = 5
NUM_PARTITON_SAMPLES_FOR_MOTIFS = 5

RADIUS = 1
# CMAP_NAME = 'twilight'
# CMAP_NAME = 'twilight'

if DATASET == "house":
    ### HOUSE
    TERMS = ["cyan"]
    NONTERMS = ["gray", "black"]
    NONFINAL = ["gray", "black"]
    FINAL = ["red", "blue", "green"]
elif DATASET == "ckt":
    ### CKT (with basis)
    TERMS = ['orchid','pink','yellow','lawngreen','greenyellow','yellowgreen','cyan','lightblue','deepskyblue','dodgerblue','lime','seagreen','springgreen','limegreen','lightcoral','coral','salmon','red','darkorange','bisque','navajowhite','orange','plum','violet','mediumpurple','blueviolet']
    NONTERMS = ['gray','black']
    NONFINAL = ['gray']
    FINAL = ['black']
    LOOKUP = {'input': 'orchid',
        'output': 'pink',
        'R': 'yellow',
        'C': 'lawngreen',
        'R serie C': 'greenyellow',
        'R paral C': 'yellowgreen',
        '+gm+': 'cyan',
        '-gm+': 'lightblue',
        '+gm-': 'deepskyblue',
        '-gm-': 'dodgerblue',
        'C paral +gm+': 'lime',
        'C paral -gm+': 'seagreen',
        'C paral +gm-': 'springgreen',
        'C paral -gm-': 'limegreen',
        'R paral +gm+': 'lightcoral',
        'R paral -gm+': 'coral',
        'R paral +gm-': 'salmon',
        'R paral gm-': 'red',
        'R paral C paral +gm+': 'darkorange',
        'R paral C paral -gm+': 'bisque',
        'R paral C paral +gm-': 'navajowhite',
        'R paral C paral -gm-': 'orange',
        'R serie C serie +gm+': 'plum',
        'R serie C serie -gm+': 'violet',
        'R serie C serie +gm-': 'mediumpurple',
        'R serie C serie -gm-': 'blueviolet'
    }
    INVERSE_LOOKUP = {v:k for (k, v) in LOOKUP.items()}
    ### CKT
    TERMS = [
        "yellow",
        "lawngreen",
        "cyan",
        "lightblue",
        "deepskyblue",
        "dodgerblue",
        "silver",
        "lightgrey",
        "orchid",
        "pink",
    ]
    NONTERMS = ["gray", "black"]  # assumes last one is init symbol S
    NONFINAL = ["gray"]
    FINAL = ["black"]
    LOOKUP = {
        "R": "yellow",
        "C": "lawngreen",
        "+gm+": "cyan",
        "-gm+": "lightblue",
        "+gm-": "deepskyblue",
        "-gm-": "dodgerblue",
        "sudo_in": "silver",
        "sudo_out": "lightgrey",
        "input": "orchid",
        "output": "pink",
    }
    INVERSE_LOOKUP = {v:k for (k, v) in LOOKUP.items()}
elif DATASET == "enas":
    NONTERMS = ["gray", "black"]
    NONFINAL = ["gray"]
    FINAL = ["black"]
    TERMS = [
        "skyblue",
        "pink",
        "yellow",
        "orange",
        "greenyellow",
        "seagreen",
        "azure",
        "beige"
    ]
    LOOKUP = {
        "input": "skyblue",
        "output": "pink",
        "conv3": "yellow",
        "sep3": "orange",
        "conv5": "greenyellow",
        "sep5": "seagreen",
        "avg3": "azure",
        "max3": "beige"
    }
    INVERSE_LOOKUP = {v:k for (k, v) in LOOKUP.items()}
