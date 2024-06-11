from src.config import LABELS, SEED, RADIUS
import networkx as nx
import numpy as np
import pygsp as gsp
from pygsp import graphs
import json
from src.draw.color import to_hex, CMAP

def create_random_graph(labels=LABELS):
    g = nx.random_regular_graph(3, 20, seed=SEED)
    labels = np.random.choice(labels, size=(len(g),))
    for n, label in zip(g, labels):
        g.nodes[n]['label'] = label
    return g


def create_test_graph(num):
    if num == 1:
        g = nx.Graph()
        labels = ['r','g','b','c','b','r','g','b','c','b','g','r','c','b']
        edges = [(0,1),(1,2),(2,3),(3,0),(1,4),(4,5),(5,6),(6,7),(7,8),(8,5),(6,9),(9,10),(10,11),(11,12),(12,13),(13,10)]
        for i in range(14):
            g.add_node(i, label=labels[i])
        for e in edges:
            g.add_edge(*e)
    else:
        pass
    return g


def create_minnesota():
    minn = graphs.Minnesota()
    g = minn.to_networkx()
    for n in g:
        g.nodes[n]['label'] = 'r'
    # draw_graph(g, os.path.join(IMG_DIR, 'base.png'))
    return g


def load_cora():
    g = nx.node_link_graph(json.load(open('/home/msun415/induction/data/nx/cora.json')))
    # labels = list(set([g.nodes[n]['label'] for n in g]))
    # assert len(labels) == len(LABELS)
    # lookup = dict(zip(labels, LABELS))
    
    conn = list(nx.connected_components(g))[0]
    print(len(conn), "nodes")
    g = nx.Graph(nx.induced_subgraph(g, conn))    
    lookup = {}        
    for n in g:
        ego_g = nx.ego_graph(g, n, radius=RADIUS)
        val = nx.weisfeiler_lehman_graph_hash(ego_g, iterations=2)
        if val not in lookup:
            lookup[val] = len(lookup)    
        g.nodes[n]['label'] = to_hex(CMAP(lookup[val]))    
    assert len(lookup) <= CMAP.N
    return g    


def debug():
    g = nx.Graph()
    g.add_node(0, label='#e1d8e1ff')
    g.add_node(1, label='#97b6c7ff')
    g.add_node(2, label='#97b6c7ff')
    g.add_node(3, label='#e1d8e1ff')
    g.add_edge(0,1)
    g.add_edge(2,1)
    g.add_edge(2,3)    
    return g
