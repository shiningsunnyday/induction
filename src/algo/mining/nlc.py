from src.config import IMG_DIR, CACHE_DIR, MAX_ATTEMPTS
import os
import networkx as nx
import numpy as np
import pickle
from copy import deepcopy
from functools import reduce
from src.grammar.nlc import *
from src.draw.graph import *
from Subdue import nx_subdue


def prepare_subdue(g):
    g = nx.relabel_nodes(g, {n: str(i) for i, n in enumerate(list(g))})    
    return g


def learn_grammar(g):
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)    
    g = prepare_subdue(g)
    cache_path = None
    cache_iter = 0
    for f in os.listdir(CACHE_DIR):
        if int(f.split('.pkl')[0]) > cache_iter:
            cache_iter = int(f.split('.pkl')[0])
            cache_path = os.path.join(CACHE_DIR, f"{cache_iter}.pkl")
    if cache_iter == 0:
        g = deepcopy(g)        
        # wipe clean IMG_DIR
        for f in os.listdir(IMG_DIR):
            if os.path.isfile(os.path.join(IMG_DIR, f)):
                os.remove(os.path.join(IMG_DIR, f))        
        grammar = NLCGrammar()
        iter = 0
        path = os.path.join(IMG_DIR, "base.png")
        draw_graph(g, path)        
    else:
        grammar, g = pickle.load(open(cache_path, 'rb'))
        iter = cache_iter  
        path = os.path.join(IMG_DIR, f'api_{iter}.png')
        assert os.path.exists(path)    
    while len(g) > 1:
        iter += 1
        good = False        
        out = nx_subdue(g, verbose=True)        
        if out is None:
            rule = NLCRule('black', g, None)
            rule_no = len(grammar.rules)
            rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))            
            grammar.add_rule(rule)
            break
        groups = [out[i][0]['nodes'] for i in range(len(out))]
        subgraphs = [nx.induced_subgraph(g, group) for group in groups]
        assert min([nx.is_connected(subgraph) for subgraph in subgraphs])
        for i, subgraph in enumerate(subgraphs):
            draw_graph(subgraph, path.replace(".png", f"{i}_subgraph.png"))
        best_ism, best_clique = find_embedding(subgraphs, g)
        if best_ism is not None:
            good = True
        if not good:
            rule = NLCRule('black', g, None)
            rule_no = len(grammar.rules)
            rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))            
            grammar.add_rule(rule)
            break        
        compats = [best_ism.nodes[n] for n in best_clique]
        lower = reduce(lambda x,y: x|y, [compat['ins'] for compat in compats])
        # L2 = set([(x,y) for x, y in product(LABELS, LABELS)])
        # ous = reduce(lambda x,y: x&y, [compat['out'] for compat in compats])
        # upper = L2 - ous
        nodes_induce = best_ism.nodes[list(best_ism)[0]]['ism']
        rhs_graph = deepcopy(nx.induced_subgraph(g, nodes_induce))
        for c in best_clique:
            ism = best_ism.nodes[c]
            nodes = ism['ism']   
            new_n = str(max(list(map(int, g)))+1)
            print(new_n)
            subneis = [n for n in neis(g, nodes)]
            for n in subneis:
                if int(n) >= int(new_n):
                    breakpoint()
            print(subneis)
            g.add_node(new_n, label='gray')
            for n in nodes:
                g.remove_node(n)
            for subnei in subneis:
                g.add_edge(new_n, subnei)    
        path = os.path.join(IMG_DIR, f'api_{iter}.png')
        draw_graph(g, path)    
        rule = NLCRule('gray', rhs_graph, lower)
        rule_no = len(grammar.rules)
        rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
        grammar.add_rule(rule)
        cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")
        pickle.dump((grammar, g), open(cache_path, 'wb+'))
    return grammar
    