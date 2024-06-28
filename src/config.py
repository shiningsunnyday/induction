METHOD = 'api'
DATASET = 'ckt'
GRAMMAR = 'ednce'

MODEL = "gpt-4o"
FILE_NAME = f"/home/msun415/induction/data/{METHOD}_{DATASET}_{GRAMMAR}.txt"
IMG_DIR = f"/home/msun415/induction/data/{METHOD}_{DATASET}_{GRAMMAR}/"
CACHE_DIR = f"/home/msun415/induction/cache/{METHOD}_{DATASET}_{GRAMMAR}/"

### HOUSE
# TERMS = ['cyan']
# NONTERMS = ['gray','black']
# NONFINAL = ['gray','black']
# FINAL = ['red','blue','green']

### CKT (with basis)
# TERMS = ['orchid','pink','yellow','lawngreen','greenyellow','yellowgreen','cyan','lightblue','deepskyblue','dodgerblue','lime','seagreen','springgreen','limegreen','lightcoral','coral','salmon','red','darkorange','bisque','navajowhite','orange','plum','violet','mediumpurple','blueviolet']
# NONTERMS = ['gray','black']
# NONFINAL = ['gray']
# FINAL = ['black']
# CKT_LOOKUP = {'input': 'orchid',
#     'output': 'pink',
#     'R': 'yellow',
#     'C': 'lawngreen',
#     'R serie C': 'greenyellow',
#     'R paral C': 'yellowgreen',
#     '+gm+': 'cyan',
#     '-gm+': 'lightblue',
#     '+gm-': 'deepskyblue',
#     '-gm-': 'dodgerblue',
#     'C paral +gm+': 'lime',
#     'C paral -gm+': 'seagreen',
#     'C paral +gm-': 'springgreen',
#     'C paral -gm-': 'limegreen',
#     'R paral +gm+': 'lightcoral',
#     'R paral -gm+': 'coral',
#     'R paral +gm-': 'salmon',
#     'R paral gm-': 'red',
#     'R paral C paral +gm+': 'darkorange',
#     'R paral C paral -gm+': 'bisque',
#     'R paral C paral +gm-': 'navajowhite',
#     'R paral C paral -gm-': 'orange',
#     'R serie C serie +gm+': 'plum',
#     'R serie C serie -gm+': 'violet',
#     'R serie C serie +gm-': 'mediumpurple',
#     'R serie C serie -gm-': 'blueviolet'
# }
# INVERSE_LOOKUP = {
#     'orchid': 'input',
#     'pink': 'output',
#     'yellow': 'R',
#     'lawngreen': 'C',
#     'greenyellow': 'R serie C',
#     'yellowgreen': 'R paral C',
#     'cyan': '+gm+',
#     'lightblue': '-gm+',
#     'deepskyblue': '+gm-',
#     'dodgerblue': '-gm-',
#     'lime': 'C paral +gm+',
#     'seagreen': 'C paral -gm+',
#     'springgreen': 'C paral +gm-',
#     'limegreen': 'C paral -gm-',
#     'lightcoral': 'R paral +gm+',
#     'coral': 'R paral -gm+',
#     'salmon': 'R paral +gm-',
#     'red': 'R paral gm-',
#     'darkorange': 'R paral C paral +gm+',
#     'bisque': 'R paral C paral -gm+',
#     'navajowhite': 'R paral C paral +gm-',
#     'orange': 'R paral C paral -gm-',
#     'plum': 'R serie C serie +gm+',
#     'violet': 'R serie C serie -gm+',
#     'mediumpurple': 'R serie C serie +gm-',
#     'blueviolet': 'R serie C serie -gm-',
# }


### CKT
TERMS = ['yellow', 'lawngreen', 'cyan', 'lightblue', 'deepskyblue', 'dodgerblue', 'silver', 'light_grey', 'orchid', 'pink']
NONTERMS = ['gray','black']
NONFINAL = ['gray']
FINAL = ['black']
CKT_LOOKUP = {
    'R': 'yellow',
    'C': 'lawngreen',
    '+gm+': 'cyan',
    '-gm+': 'lightblue',
    '+gm-': 'deepskyblue',
    '-gm-': 'dodgerblue',
    'sudo_in': 'silver',
    'sudo_out': 'light_grey',
    'input': 'orchid',
    'output': 'pink'
}
INVERSE_LOOKUP = {
    'yellow': 'R',
    'lawngreen': 'C',
    'cyan': '+gm+',
    'lightblue': '-gm+',
    'deepskyblue': '+gm-',
    'dodgerblue': '-gm-',
    'silver': 'sudo_in',
    'light_grey': 'sudo_out',
    'orchid': 'input',
    'pink': 'output'
}


SEED = 0
NUM_ATTEMPTS = 1

# VISUALIZATION
SCALE = 7.5
NODE_SIZE = 50000
RULE_SCALE = 5
RULE_NODE_SIZE = 500
RULE_FONT_SIZE = 10
FONT_SIZE = 20
MAX_SIZE = 3000*3000*4
LAYOUT = 'kamada_kawai_layout'
# LAYOUT = 'spring_layout'
TITLE_FONT_SIZE = 50
PARTITION_SIZE = 50
PARTITION_SCALE = 5
POS_EPSILON = 1
PARTITION_NODE_SIZE = 500
MAX_IMAGES = 5

RADIUS = 1
# CMAP_NAME = 'twilight'
# CMAP_NAME = 'twilight'
