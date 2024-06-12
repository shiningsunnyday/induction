import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.config import SEED, SCALE, NODE_SIZE, FONT_SIZE, LAYOUT
from src.draw.utils import hierarchy_pos

def draw_graph(g, path, scale=SCALE, node_size=NODE_SIZE, font_size=FONT_SIZE, layout=LAYOUT):
    if layout == 'spring_layout':
        pos = getattr(nx, layout)(g, seed=SEED)
    else:
        pos = getattr(nx, layout)(g)
    pos_np = np.array([v for v in pos.values()])
    w, l = pos_np.max(axis=0)-pos_np.min(axis=0)
    w = max(w, 1)
    l = max(l, 1)
    fig = plt.Figure(figsize=(scale*w, scale*l))
    ax = fig.add_subplot(1,1,1)    
    colors = [g.nodes[n]['label'] if 'label' in g.nodes[n] else 'r' for n in g]
    labels = {n: n for n in g}
    nx.draw(g, ax=ax, pos=pos, labels=labels, node_color=colors, with_labels=True, node_size=node_size, font_size=font_size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    print(os.path.abspath(path))


def draw_tree(node, path):
    g = nx.Graph()
    g.add_node(node.id, label=node.attrs['rule'])
    bfs = [node]
    while bfs:
        cur = bfs.pop(0)        
        for nei in cur.children:
            g.add_node(nei.id, label=nei.attrs['rule'])
            g.add_edge(cur.id, nei.id)
            bfs.append(nei)
    pos = hierarchy_pos(g, node.id)
    fig = plt.Figure()
    ax = fig.add_subplot(1,1,1)      
    labels = {n: g.nodes[n]['label'] for n in g}
    nx.draw(g, ax=ax, pos=pos, labels=labels, with_labels=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    print(os.path.abspath(path))
    