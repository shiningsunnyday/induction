from src.config import IMG_DIR, CACHE_DIR
from pathlib import Path
import subprocess
import re
import networkx as nx
from src.draw.graph import *
import hashlib
import json


def prepare_subdue_file(g, tmp_path):
    is_directed = 'd' if GRAMMAR == "ednce" else 'u'
    file_content = ""
    for n in g:
        label = g.nodes[n]["label"]
        line = f"v {n} {label}\n"
        file_content += line
    file_content += "\n"
    for u, v in g.edges:
        line = f"{is_directed} {u} {v} on\n"
        file_content += line
    with open(tmp_path, "w+") as f:
        f.write(file_content)        



def hash_graph(g):
    assert set(g) == set(map(str, range(1,len(g)+1))), "make sure graph nodes are '1', '2', ..."
    json_data = nx.node_link_data(g)
    ans = hashlib.md5(json.dumps(json_data, sort_keys=True).encode()).hexdigest()
    return ans



def _build_dreadnaut_file(g):
    """Prepare file to pass to dreadnaut.
    Warning
    -------
    Assumes that the nodes are represented by the 0 indexed integers.
    """
    # dreadnaut options
    file_content = ["As"]  # sparse mode
    file_content.append("-a")  # do not print out automorphisms
    file_content.append("-m")  # do not print out level markers
    file_content.append("c") # compute canon order
    file_content.append("d") # directed
    labels = set()
    for n in g:
        labels.add(g.nodes[n]['label'])
    labels = sorted(list(labels))
    partitions = [[] for _ in labels]
    for n in g:
        index = labels.index(g.nodes[n]['label'])
        partitions[index].append(str(n))
    partitions = ' | '.join([','.join(l) for l in partitions])        
    # specify graph structure
    file_content.append("n=" + str(g.number_of_nodes()) + " g")
    file_content.append(f"f=[{partitions}]")
    for v in sorted(g.nodes()):
        line = " " + str(v) + " : "
        for nb in g.neighbors(v):
            line += str(nb) + " "
        line += ";"
        file_content.append(line)
    # add nauty command    
    file_content.append(".")    
    file_content.append("x")
    file_content.append("o")
    file_content.append("b")
    return file_content



def compute_canon_order(g, tmp_path, dreadnaut_call="/home/msun415/nauty2_8_8/dreadnaut"):
    # get dreadnaut command file
    file_content = _build_dreadnaut_file(g)
    # write to tmp_path
    with open(tmp_path, 'w') as f:
        print("\n".join(file_content), file=f)
    # call dreadnaut    
    proc = subprocess.run([dreadnaut_call],
                          input=b"< " + tmp_path.encode(),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    [info, _, orbits] = proc.stdout.decode().strip().split("\n", 2)
    canon_order = orbits.split('\n')[1]
    return canon_order



def run_subdue(tmp_path, subdue_call="../subdue-5.2.2/bin/subdue"):
    is_directed = 'd' if GRAMMAR == "ednce" else 'u'
    out_path = str(Path(tmp_path).with_suffix(".out"))
    print("out_path:", out_path)
    command = [
        subdue_call,
        "-out",
        out_path
    ]
    # subdue motif params
    if "MOTIF_MIN_SIZE" in globals():
        command += ["-minsize", f"{MOTIF_MIN_SIZE}"]
    if "MOTIF_MAX_SIZE" in globals():
        command += ["-maxsize", f"{MOTIF_MAX_SIZE}"]
    if "NUM_MOTIFS" in globals():
        command += ["-nsubs", f"{NUM_MOTIFS}"]
    if "MOTIF_NUM_BEAMS" in globals():
        command += ["-beam", f"{MOTIF_NUM_BEAMS}"]
    command += [tmp_path]
    
    result = subprocess.run(command, capture_output=True, text=True)
    if is_directed == 'd':
        pat = r"((?:\s{4}v \w+ \S+\n)+(?:\s{4}d \w+ \w+ on\n)+)"
    else:
        pat = r"((?:\s{4}v \w+ \S+\n)+(?:\s{4}u \w+ \w+ on\n)+)"
    regex = re.compile(pat)
    matches = regex.findall(result.stdout)
    out = []
    for m in matches:
        lines = m.rstrip("\n").split("\n")
        g = nx.DiGraph() if is_directed == 'd' else nx.Graph()
        for line in lines:
            line = line.strip(" ")
            if line[0] == "v":
                _, n, label = line.split()
                g.add_node(n, label=label)
            else:
                _, u, v, _ = line.split()
                g.add_edge(u, v)
        out.append(g)
    return out


def setup(suffix, cache_root=None):
    logger = logging.getLogger('global_logger')
    if cache_root is not None:
        img_dir = os.path.join(cache_root, IMG_DIR)
        cache_dir = os.path.join(cache_root, CACHE_DIR)
    else:
        img_dir = IMG_DIR
        cache_dir = CACHE_DIR
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = None
    cache_iter = 0
    for f in os.listdir(cache_dir):
        stem = Path(f).stem
        if suffix not in stem:
            continue
        if suffix:
            suffix_stem = stem.split(suffix)[0]
        else:
            suffix_stem = stem
        if not suffix_stem.isdigit():
            continue
        if int(suffix_stem) > cache_iter:
            cache_iter = int(suffix_stem)
            cache_path = os.path.join(cache_dir, f"{cache_iter}{suffix}.pkl")
    if cache_iter == 0:
        logger.info(f"init grammar")
    else:
        logger.info(f"loading grammar from iter {cache_iter}{suffix} from {cache_path}")
    return cache_iter, cache_path
