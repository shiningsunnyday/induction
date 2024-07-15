from src.config import *
from src.grammar.common import *
from src.api.get_motifs import *
import os
import networkx as nx
from src.draw.graph import draw_graph
from copy import deepcopy
import pickle

def term(g):
    return min([g.nodes[n]['label'] in NONTERMS for n in g])


def obtain_motifs(g, img_paths):
    all_subgraphs = []
    for _ in range(NUM_ATTEMPTS):
        content = get_motifs(img_paths)
        groups = get_groups(content, dtype=type(list(g)[0]))        
        # groups = [['6', '7', '8', '15']]
        # groups = [['4','5','9','14']]
        # groups = [['16','10','3']]
        induced_subgraphs = []
        for group in groups:
            if isinstance(group, dict):
                assert list(group.keys()) == ['index', 'group']
                index = group['index']
                group = [f"{index}:{n}" for n in group['group']]
                if np.any([n not in g for n in group]):
                    continue
            subgraph = copy_graph(g, group)
            if len(subgraph):
                induced_subgraphs.append(subgraph)        
        subgraphs = []
        for subgraph in induced_subgraphs:
            undirected_subgraph = nx.Graph(subgraph)
            comps = list(nx.connected_components(undirected_subgraph))
            subgraphs += [copy_graph(subgraph, comp) for comp in comps]             
        all_subgraphs += subgraphs
    return all_subgraphs


def partition_graph(g, iter):
    """
    Takes a graph g, splits into partition based on PARTITON_SIZE
    Visualizes one image per partition, and outputs the paths
    If g is already a set of connected components, recursively call
    each component separately
    """
    node_sets = list(nx.connected_components(nx.Graph(g)))
    if len(node_sets) > 1: # separate graphs
        res = []
        for i, node_set in enumerate(node_sets):
            conn_g = copy_graph(g, node_set)
            conn_g = conn_g.__class__(conn_g)    
            assert len(set([n.split(':')[0] for n in conn_g])) == 1
            index = list(conn_g)[0].split(':')[0]
            conn_g.graph['title'] = f"Graph #{index}"
            res_g = partition_graph(conn_g, f"{iter}_comp={i}")
            res += res_g
        return res
        
    img_paths = []
    all_nodes = list(g)
    for i in range((len(g)+PARTITION_SIZE-1)//PARTITION_SIZE):
        start = PARTITION_SIZE*i
        nodes = all_nodes[start:start+PARTITION_SIZE]        
        one_hop_neis = neis(g, nodes)
        subgraph = copy_graph(g, nodes+one_hop_neis)
        subgraph = subgraph.__class__(subgraph)
        for n in one_hop_neis:
            subgraph.nodes[n]['node_size'] = PARTITION_NODE_SIZE
        for n in nodes:
            subgraph.nodes[n]['node_size'] = PARTITION_NODE_SIZE
        subgraph.graph['scale'] = PARTITION_SCALE
        subgraph.graph['font_size'] = 20
        for a in one_hop_neis:
            for b in one_hop_neis:
                if subgraph.has_edge(a, b):
                    subgraph.remove_edge(a, b)
        img_path = os.path.join(IMG_DIR, f'{METHOD}_iter={iter}_partition={i}.png')
        draw_graph(subgraph, img_path)
        img_paths.append(img_path)
    return img_paths



def rewire_graph(g, new_n, nodes, anno):
    subneis = [n for n in neis(g, nodes)]
    for n in nodes:
        label = g.nodes[n]['label']
        g.remove_node(n)
        # if n is annotated, add edge to model
        if n in anno:
            if label in NONTERMS:
                anno[new_n].add_child(anno[n])                
                print(f"new edge between {new_n} and {n}")    
    for subnei in subneis:
        g.add_edge(new_n, subnei)


def init_grammar(g, cache_iter, cache_path, grammar_class):
    if cache_iter == 0:
        g = deepcopy(g)
        # wipe clean IMG_DIR
        for f in os.listdir(IMG_DIR):
            if os.path.isfile(os.path.join(IMG_DIR, f)):
                os.remove(os.path.join(IMG_DIR, f))        
        grammar = grammar_class()
        # path = os.path.join(IMG_DIR, f'{METHOD}_0.png')
        # draw_graph(g, path)
        anno = {} # annotation for model 
        iter = 0
    else:
        grammar, anno, g = pickle.load(open(cache_path, 'rb'))
        iter = cache_iter  
        # path = os.path.join(IMG_DIR, f'{METHOD}_0.png')
    return g, grammar, anno, iter