import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
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
    colors = []
    for n in g:
        if 'label' in g.nodes[n]:
            c = to_rgba(g.nodes[n]['label'])
        else:
            c = to_rgba('r')
        if 'alpha' in g.nodes[n]:
            c = c[:-1] + (g.nodes[n]['alpha'],)
        colors.append(c)
    labels = {n: n for n in g}
    nx.draw_networkx_nodes(g, 
                           ax=ax, 
                           pos=pos, 
                           node_color=colors, 
                           node_size=node_size)
    nx.draw_networkx_labels(g,
                            ax=ax,
                            pos=pos,
                            labels=labels,
                            font_size=font_size)
    dashed_edges = []
    for u, v, dic in g.edges(data=True):    
        if 'style' in dic and dic['style'] == 'dashed':
            dashed_edges.append((u, v))
    nx.draw_networkx_edges(g, 
                           ax=ax,
                           pos=pos,
                           edgelist=set(g.edges()) - set(dashed_edges))
    nx.draw_networkx_edges(g,
                           ax=ax,
                           pos=pos,
                           edgelist=dashed_edges,
                           style='dashed')
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
    