from src.config import IMG_DIR, CACHE_DIR, MAX_ATTEMPTS, METHOD
import os
import numpy as np
import pickle
from copy import deepcopy
from functools import reduce
from src.grammar.nce import *
from src.draw.graph import *
from src.algo.utils import *
from Subdue import nx_subdue


def learn_grammar(g):
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)    
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
        grammar = NCEGrammar()
        iter = 0
        path = os.path.join(IMG_DIR, "base.png")
        # draw_graph(g, path)        
    else:
        grammar, g = pickle.load(open(cache_path, 'rb'))
        iter = cache_iter  
        path = os.path.join(IMG_DIR, f'{METHOD}_{iter}.png')        
        assert os.path.exists(path)
    while len(g) > 1:
        print(f"{len(g)} nodes, {len(g.edges)} edges")
        iter += 1
        good = False
    
        ## USE PYTHON SUBDUE
        # out = nx_subdue(g, verbose=True)
        # if out is None:
        #     rule = NCERule('black', g, None)
        #     rule_no = len(grammar.rules)
        #     rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))            
        #     grammar.add_rule(rule)
        #     break
        # groups = [out[i][0]['nodes'] for i in range(len(out))]
        # subgraphs = [copy_graph(g, group) for group in groups]

        ## USE C SUBDUE
        tmp_path = os.path.join(IMG_DIR, f'g_{iter}.g')
        mapping = {n: str(i+1) for i, n in enumerate(g)}
        inv_mapping = {v: k for k, v in mapping.items()}
        g = nx.relabel_nodes(g, mapping) # subdue assumes nodes are consecutive 1,2,...
        prepare_subdue_file(g, tmp_path)
        subgraphs = run_subdue(tmp_path)
        g = nx.relabel_nodes(g, inv_mapping)

        if len(subgraphs) == 0:
            print("no subgraphs found, terminating")
            rule = NCERule('black', g, None)
            rule_no = len(grammar.rules)
            rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))            
            grammar.add_rule(rule)
            break

        assert min([nx.is_connected(subgraph) for subgraph in subgraphs])
        for i, subgraph in enumerate(subgraphs):
            draw_graph(subgraph, path.replace(".png", f"{i}_subgraph.png"))        
        best_ism, best_clique = find_embedding(subgraphs, g)
        if best_ism is not None:
            good = True
        if not good:
            print("no cliques found, terminating")
            rule = NCERule('black', g, None)
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
        rhs_graph = copy_graph(g, nodes_induce)
        rhs_graph = nx.relabel_nodes(rhs_graph, {n: nodes_induce.index(n) for n in rhs_graph})
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
        path = os.path.join(IMG_DIR, f'{METHOD}_{iter}.png')
        draw_graph(g, path)    
        rule = NCERule('gray', rhs_graph, lower)
        rule_no = len(grammar.rules)
        rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
        grammar.add_rule(rule)
        cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")
        pickle.dump((grammar, g), open(cache_path, 'wb+'))
    draw_tree(model, os.path.join(IMG_DIR, f"model_{iter}.png"))        
    model = NCEModel(model)
    return grammar, model
    