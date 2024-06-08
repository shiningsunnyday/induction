from src.config import LABELS
import networkx as nx
import numpy as np

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
