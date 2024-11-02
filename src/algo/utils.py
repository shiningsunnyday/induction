from src.config import IMG_DIR, CACHE_DIR
from pathlib import Path
import subprocess
import re
import networkx as nx
from src.draw.graph import *
import hashlib
import json


def prepare_subdue_file(g, tmp_path):
    is_directed = 'e' if GRAMMAR == "ednce" else 'u'
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



def run_subdue(tmp_path, subdue_call="/home/msun415/subdue-5.2.2/bin/subdue"):
    is_directed = 'e' if GRAMMAR == "ednce" else 'u'
    out_path = str(Path(tmp_path).with_suffix(".out"))
    command = [
        subdue_call,
        "-out",
        out_path,
        "-minsize",
        "2",
        "-maxsize",
        "5",
        "-nsubs",
        "10",
        tmp_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if is_directed:
        pat = r"((?:\s{4}v \w+ \S+\n)+(?:\s{4}d \w+ \w+ on\n)+)"
    else:
        pat = r"((?:\s{4}v \w+ \S+\n)+(?:\s{4}u \w+ \w+ on\n)+)"
    regex = re.compile(pat)
    matches = regex.findall(result.stdout)
    out = []
    for m in matches:
        lines = m.rstrip("\n").split("\n")
        g = nx.DiGraph() if is_directed else nx.Graph()
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


def setup():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = None
    cache_iter = 0
    for f in os.listdir(CACHE_DIR):
        if int(f.split(".pkl")[0]) > cache_iter:
            cache_iter = int(f.split(".pkl")[0])
            cache_path = os.path.join(CACHE_DIR, f"{cache_iter}.pkl")
    return cache_iter, cache_path
