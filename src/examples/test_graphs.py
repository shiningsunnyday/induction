from src.config import SEED, RADIUS, CKT_LOOKUP
import networkx as nx
import numpy as np
import pygsp as gsp
from pygsp import graphs
import json
from src.draw.color import to_hex, CMAP
from src.grammar.ednce import EDNCEGrammar, EDNCERule
from src.draw.graph import draw_graph
from networkx.readwrite import json_graph
import os


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
    def construct_house(grammar, K):
        rule1 = grammar.rules[0]
        rule2 = grammar.rules[1]
        rule3 = grammar.rules[2]
        g = nx.DiGraph()
        g.add_node('0', label='black')
        K = 2
        g = rule1(g, '0')    
        for k in range(K):
            nt = grammar.search_nts(g, ['gray'])[0]
            g = rule2(g, nt)
        nt = grammar.search_nts(g, ['gray'])[0]
        g = rule3(g, nt)        
        return g
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
    # in the textbook, edge labels are {h,r,a,b,*} where {h,r} are non-final
    # in the textbook, node labels are {S,X,#} where {S,X} are non-terminal
    # we do the mapping {h,r,a,b,*} -> {black,gray,red,blue,green}
    # {S,X,#} -> {black,gray,cyan}
    # (0,3,h)
    grammar = EDNCEGrammar()
    subg1 = nx.DiGraph()
    subg1.add_node(0, label='cyan')
    subg1.add_node(1, label='cyan')
    subg1.add_node(2, label='cyan')
    subg1.add_node(3, label='gray')
    subg1.add_edge(0, 3, label='black')
    subg1.add_edge(1, 0, label='blue')
    subg1.add_edge(2, 1, label='green')
    subg1.add_edge(2, 3, label='gray')
    subg1.add_edge(3, 1, label='black')
    rule1 = EDNCERule('black', subg1, set())
    grammar.add_rule(rule1)
    subg2 = nx.DiGraph()
    subg2.add_node(0, label='cyan')
    subg2.add_node(1, label='cyan')
    subg2.add_node(2, label='cyan')
    subg2.add_node(3, label='cyan')
    subg2.add_node(4, label='gray')
    subg2.add_edge(0, 1, label='green')
    subg2.add_edge(1, 4, label='black')
    subg2.add_edge(2, 1, label='green')
    subg2.add_edge(3, 4, label='gray')
    subg2.add_edge(4, 2, label='black')
    emb2 = set()
    emb2.add(('cyan', 'black', 'green', 0, 'in', 'in'))
    emb2.add(('cyan', 'black', 'red', 1, 'in', 'in'))
    emb2.add(('cyan', 'black', 'red', 2, 'out', 'out'))
    emb2.add(('cyan', 'gray', 'red', 3, 'in', 'in'))
    subg3 = nx.DiGraph()
    subg3.add_node(0, label='cyan')
    subg3.add_node(1, label='cyan')
    subg3.add_node(2, label='cyan')
    subg3.add_node(3, label='cyan')
    subg3.add_edge(0, 1, label='green')
    subg3.add_edge(2, 1, label='green')
    subg3.add_edge(3, 2, label='blue')
    emb3 = emb2
    rule2 = EDNCERule('gray', subg2, emb2)
    rule3 = EDNCERule('gray', subg3, emb3)
    grammar.add_rule(rule2)
    grammar.add_rule(rule3)
    g = nx.DiGraph()
    for size in [4]:
        house = construct_house(grammar, size)
        g = nx.disjoint_union(g, house)
    g = nx.relabel_nodes(g, {n: str(i+1) for i, n in enumerate(list(g))})
    return g



def load_ckt():
    data_dir = '/home/msun415/induction/data/nx/ckt/'
    for i in range(9000):
        fpath = os.path.join(data_dir, f"{i}.json")
        data = json.load(open(fpath))
        g = json_graph.node_link_graph(data)
        breakpoint()
        lookup = CKT_LOOKUP    
        for n in g:        
            g.nodes[n]['type'] = list(lookup)[g.nodes[n]['type']]
            g.nodes[n]['label'] = lookup[g.nodes[n]['type']]
        for e in g.edges:
            g.edges[e]['label'] = 'black'

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
