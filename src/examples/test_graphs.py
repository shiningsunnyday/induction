from src.config import SEED, RADIUS
import networkx as nx
import numpy as np
import pygsp as gsp
from pygsp import graphs
import json
from src.draw.color import to_hex, CMAP

LABELS = ['r','g','b','c']

def create_random_graph(labels=LABELS):
    g = nx.random_regular_graph(3, 20, seed=SEED)
    labels = np.random.choice(labels, size=(len(g),))
    for n, label in zip(g, labels):
        g.nodes[n]['label'] = label
    return g


def create_test_graph(num):
    if num == 1:
        g = nx.Graph()
        labels = ['r','g','b','c','r','r','g','b','c','b','g','r','c','b']
        labels[5], labels[6] = labels[6], labels[5]
        labels[8], labels[7] = labels[7], labels[8]
        labels[12], labels[10] = labels[10], labels[12]
        labels[11], labels[10], labels[13], labels[12] = labels[12], labels[11], labels[10], labels[13]
        edges = [(1, 2), (2, 3), (3, 4), (4, 1), (2, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 6), (7, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 11)]
        for i in range(14):
            g.add_node(i+1, label=labels[i])
        for e in edges:
            g.add_edge(*e)
        g = nx.relabel_nodes(g, {n: n for n in g})
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
    for n in list(sorted(g)):
        ego_g = nx.ego_graph(g, n, radius=RADIUS)
        val = nx.weisfeiler_lehman_graph_hash(ego_g, iterations=2)
        # val = g.nodes[n]['label']
        # nei_labels = [g.nodes[n]['label'] for n in ego_g]
        # nei_labels, counts = np.unique(nei_labels, return_counts=True)
        # nei_labels = [nei_label for (nei_label, count) in zip(nei_labels, counts) if count > 1]
        # labels = sorted(list(set(nei_labels)))
        # labels = ','.join(labels)
        # val = f"{val}_{labels}"
        if val not in lookup:
            lookup[val] = len(lookup)    
        g.nodes[n]['label'] = to_hex(CMAP(lookup[val]))    
    assert len(lookup) <= CMAP.N, f"{len(lookup)} exceeds {CMAP.N} colors"
    g = nx.relabel_nodes(g, {n: str(i+1) for i, n in enumerate(list(g))})
    print(len(lookup), "labels")
    return g


def create_house_graph():    
    # g = nx.DiGraph()
    # edge_list = [(0,1),(0,2,'red'),
    #              (1,2),
    #              (2,3),(2,4,'red'),
    #              (3,4),
    #              (4,5),(4,6,'red'),
    #              (5,6),
    #              (7,6),(7,8,'red'),
    #              (8,4),(8,9,'red'),
    #              (9,2),(9,10,'red'),
    #              (10,0,'blue'),
    #              (11,10),(11,12,'red'),
    #              (12,13,'red'),
    #              (13,14,'red'),
    #              (14,7,'blue')]
    # for a, b, *e in edge_list:
    #     if len(e) == 1:
    #         e = e[0]
    #     else:
    #         e = 'green'
    #     g.add_edge(a, b, label=e)
    # for n in g:
    #     g.nodes[n]['label'] = 'cyan'
    # g = nx.relabel_nodes(g, {n: str(i+1) for i, n in enumerate(list(g))})
    # return g
    EDNCEGrammar()



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
