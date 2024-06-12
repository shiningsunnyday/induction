from src.config import IMG_DIR, CACHE_DIR, MAX_ATTEMPTS, NONTERMS
import os
import networkx as nx
import numpy as np
import pickle
from copy import deepcopy
from functools import reduce
from src.grammar.nlc import *
from src.draw.graph import *
from src.api.get_motifs import *


def term(g):
    return min([g.nodes[n]['label'] == 'gray' for n in g])


def obtain_motifs(g, path, IMAGE_PATHS):
    good = False
    for k in range(MAX_ATTEMPTS):
        content = get_motifs(IMAGE_PATHS)
        groups = get_groups(content)
        if len(groups) == 0:
            continue
        subgraphs = [nx.induced_subgraph(g, group) for group in groups]
        subgraphs = sum([[nx.induced_subgraph(subgraph, comp) for comp in nx.connected_components(subgraph)] for subgraph in subgraphs], [])
        for i, subgraph in enumerate(subgraphs):
            draw_graph(subgraph, path.replace(".png", f"{i}_subgraph.png"))
        best_ism, best_clique = find_embedding(subgraphs, g)
        if best_ism is not None:
            good = True
            break  

    return best_ism, best_clique, good  


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
        grammar = NLCGrammar()
        iter = 0
        path = os.path.join(IMG_DIR, "base.png")
        draw_graph(g, path)       
        anno = {} # annotation for model 
    else:
        grammar, anno, g = pickle.load(open(cache_path, 'rb'))
        iter = cache_iter  
        path = os.path.join(IMG_DIR, f'api_{iter}.png')
        assert os.path.exists(path)    
    while len(g) > 1:
        iter += 1
        IMAGE_PATHS = [path]
        end = term(g)
        if not end:
            best_ism, best_clique, good = obtain_motifs(g, path, IMAGE_PATHS)
            end = end or not good
        if end:
            nodes = list(g)
            new_n = max(list(g))+1            
            rule_no = len(grammar.rules)
            anno[new_n] = NLCNode(new_n, attrs={'rule':rule_no})
            rule = NLCRule('black', deepcopy(g), None)                        
            rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))            
            grammar.add_rule(rule)
            rewire_graph(g, new_n, nodes, anno)
            model = anno[new_n]
            break        
        try:
            compats = [best_ism.nodes[n] for n in best_clique]
        except:
            print("failed")
            print(best_clique)
            print(best_ism)
        lower = reduce(lambda x,y: x|y, [compat['ins'] for compat in compats])
        # L2 = set([(x,y) for x, y in product(LABELS, LABELS)])
        # ous = reduce(lambda x,y: x&y, [compat['out'] for compat in compats])
        # upper = L2 - ous
        nodes_induce = best_ism.nodes[list(best_ism)[0]]['ism']
        rhs_graph = deepcopy(nx.induced_subgraph(g, nodes_induce))
        rule_no = len(grammar.rules)
        change = False
        for c in best_clique:
            g_cache = deepcopy(g)
            anno_cache = deepcopy(anno)
            ism = best_ism.nodes[c]
            nodes = ism['ism']    
            new_n = max(list(g))+1
            g.add_node(new_n, label='gray') # annotate which rule was applied to model
            anno[new_n] = NLCNode(new_n, attrs={'rule': rule_no})
            model = anno[new_n]
            print(f"{new_n} new node")                   
            rewire_graph(g, new_n, nodes, anno)           
            if boundary(g):
                g = g_cache
                anno = anno_cache
                continue       
            else:
                change = True
        if not change:
            nodes = list(g)
            new_n = max(list(g))+1            
            rule_no = len(grammar.rules)
            anno[new_n] = NLCNode(new_n, attrs={'rule':rule_no})
            rule = NLCRule('black', deepcopy(g), None)                        
            rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))            
            grammar.add_rule(rule)
            rewire_graph(g, new_n, nodes, anno)
            model = anno[new_n]
            break               
        path = os.path.join(IMG_DIR, f'api_{iter}.png')
        draw_graph(g, path)   
        rule = NLCRule('gray', deepcopy(rhs_graph), lower)        
        rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
        grammar.add_rule(rule)
        cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")                
        pickle.dump((grammar, anno, g), open(cache_path, 'wb+'))    
    draw_tree(model, os.path.join(IMG_DIR, f"model_{iter}.png"))
    model = NLCModel(model)
    return grammar, model
