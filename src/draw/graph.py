import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.config import SEED

def draw_graph(g, path, scale=20, node_size=100):
    pos = nx.spring_layout(g, seed=SEED)
    pos_np = np.array([v for v in pos.values()])
    w, l = pos_np.max(axis=0)-pos_np.min(axis=0)
    w = max(w, 1)
    l = max(l, 1)
    fig = plt.Figure(figsize=(scale*w, scale*l))
    ax = fig.add_subplot(1,1,1)    
    colors = [g.nodes[n]['label'] if 'label' in g.nodes[n] else 'r' for n in g]
    labels = {n: n for n in g}
    nx.draw(g, ax=ax, pos=pos, labels=labels, node_color=colors, with_labels=True, node_size=node_size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    print(os.path.abspath(path))