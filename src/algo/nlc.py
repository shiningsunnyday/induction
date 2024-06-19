from src.config import *
import os
import networkx as nx
import numpy as np
import pickle
from copy import deepcopy
from functools import reduce
from src.grammar.nlc import *
from src.draw.graph import *
from src.api.get_motifs import *
from src.algo.utils import *


def term(g):
    return min([g.nodes[n]['label'] == 'gray' for n in g])


def obtain_motifs(g, img_paths):
    all_subgraphs = []
    for _ in range(NUM_ATTEMPTS):
        content = get_motifs(img_paths)
        groups = get_groups(content)
        induced_subgraphs = [nx.induced_subgraph(g, group) for group in groups]
        subgraphs = []
        for subgraph in induced_subgraphs:
            comps = nx.connected_components(subgraph)
            subgraphs += [nx.induced_subgraph(subgraph, comp) for comp in comps]             
        all_subgraphs += subgraphs
    best_ism, best_clique = find_embedding(all_subgraphs, g)        
    return best_ism, best_clique


def partition_graph(g, iter):
    """
    Takes a graph g, splits into partition based on PARTITON_SIZE
    Visualizes one image per partition, and outputs the paths
    """    
    img_paths = []
    all_nodes = list(g)
    for i in range((len(g)+PARTITION_SIZE-1)//PARTITION_SIZE):
        start = PARTITION_SIZE*i
        nodes = all_nodes[start:start+PARTITION_SIZE]        
        one_hop_neis = neis(g, nodes)
        subgraph = nx.Graph(nx.induced_subgraph(g, nodes+one_hop_neis))
        for n in one_hop_neis:
            subgraph.nodes[n]['node_size'] = 1000
        for n in nodes:
            subgraph.nodes[n]['node_size'] = 1000
        subgraph.graph['scale'] = 12
        subgraph.graph['font_size'] = 20
        for a in one_hop_neis:
            for b in one_hop_neis:
                if subgraph.has_edge(a, b):
                    subgraph.remove_edge(a, b)
        img_path = os.path.join(IMG_DIR, f'{METHOD}_{iter}_{i}.png')
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


def init_grammar(g, cache_iter, cache_path):
    if cache_iter == 0:
        g = deepcopy(g)
        # wipe clean IMG_DIR
        for f in os.listdir(IMG_DIR):
            if os.path.isfile(os.path.join(IMG_DIR, f)):
                os.remove(os.path.join(IMG_DIR, f))        
        grammar = NLCGrammar()
        # path = os.path.join(IMG_DIR, f'{METHOD}_0.png')
        # draw_graph(g, path)
        anno = {} # annotation for model 
        iter = 0
    else:
        grammar, anno, g = pickle.load(open(cache_path, 'rb'))
        iter = cache_iter  
        # path = os.path.join(IMG_DIR, f'{METHOD}_0.png')
    return g, grammar, anno, iter



def terminate(g, grammar, anno, iter):
    nodes = list(g)
    new_n = find_next(g)
    rule_no = len(grammar.rules)
    anno[new_n] = NLCNode(new_n, attrs={'rule': rule_no})    
    rule = NLCRule('black', deepcopy(g), None)                        
    rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))            
    grammar.add_rule(rule)
    rewire_graph(g, new_n, nodes, anno)
    model = anno[new_n]    
    cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")                
    pickle.dump((grammar, anno, g), open(cache_path, 'wb+'))        
    return g, grammar, model


def extract_rule(g, best_ism, best_clique, grammar):    
    compats = [best_ism.nodes[n] for n in best_clique]
    lower = reduce(lambda x,y: x|y, [compat['ins'] for compat in compats])
    # L2 = set([(x,y) for x, y in product(LABELS, LABELS)])
    # ous = reduce(lambda x,y: x&y, [compat['out'] for compat in compats])
    # upper = L2 - ous
    nodes_induce = best_ism.nodes[list(best_ism)[0]]['ism']
    rhs_graph = deepcopy(nx.induced_subgraph(g, nodes_induce))
    rule = NLCRule('gray', deepcopy(rhs_graph), lower)   
    rule_no = len(grammar.rules)     
    rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
    grammar.add_rule(rule) 


def update_graph(g, anno, best_ism, best_clique, grammar):
    change = False
    for c in best_clique:
        g_cache = deepcopy(g)
        anno_cache = deepcopy(anno)
        ism = best_ism.nodes[c]
        nodes = ism['ism']    
        new_n = max(list(g))+1
        g.add_node(new_n, label='gray') # annotate which rule was applied to model
        anno[new_n] = NLCNode(new_n, attrs={'rule': len(grammar.rules)-1})        
        print(f"{new_n} new node")                   
        rewire_graph(g, new_n, nodes, anno)           
        if boundary(g):
            g = g_cache
            anno = anno_cache
            continue
        else:
            change = True  
    return g, change



def learn_grammar(g):
    cache_iter, cache_path = setup()    
    g, grammar, anno, iter = init_grammar(g, cache_iter, cache_path)    
    while len(g) > 1:
        iter += 1
        img_paths = partition_graph(g, iter)        
        if term(g):
            g, grammar, model = terminate(g, grammar, anno, iter)
            break        
        best_ism, best_clique = obtain_motifs(g, img_paths)
        if best_ism is None:
            g, grammar, model = terminate(g, grammar, anno, iter)
            break                
        extract_rule(g, best_ism, best_clique, grammar)              
        g, change = update_graph(g, anno, best_ism, best_clique, grammar)        
        if not change:
            g, grammar, model = terminate(g, grammar, anno, iter)
            break
        path = os.path.join(IMG_DIR, f'{METHOD}_{iter}.png')
        draw_graph(g, path)   
        cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")                
        pickle.dump((grammar, anno, g), open(cache_path, 'wb+'))    
    model = anno[find_max(anno)]
    draw_tree(model, os.path.join(IMG_DIR, f"model_{iter}.png"))
    model = NLCModel(model)
    return grammar, model
