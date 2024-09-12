from src.config import IMG_DIR, CACHE_DIR
from pathlib import Path
import subprocess
import re
import networkx as nx
from src.draw.graph import *


def prepare_subdue_file(g, tmp_path):
    file_content = ""
    for n in g:
        label = g.nodes[n]["label"]
        line = f"v {n} {label}\n"
        file_content += line
    file_content += "\n"
    for u, v in g.edges:
        line = f"u {u} {v} on\n"
        file_content += line
    with open(tmp_path, "w+") as f:
        f.write(file_content)


def run_subdue(tmp_path, subdue_call="/home/msun415/subdue-5.2.2/bin/subdue"):
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
    pat = r"((?:\s{4}v \w+ \S+\n)+(?:\s{4}u \w+ \w+ on\n)+)"
    regex = re.compile(pat)
    matches = regex.findall(result.stdout)
    out = []
    for m in matches:
        lines = m.rstrip("\n").split("\n")
        g = nx.Graph()
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
