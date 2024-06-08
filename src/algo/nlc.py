from src.config import IMG_DIR, MAX_ATTEMPTS
import os
import networkx as nx
import numpy as np
from copy import deepcopy
from functools import reduce
from src.grammar.nlc import *
from src.draw.graph import *
from src.api.get_motifs import *


def learn_grammar(g):
    os.makedirs(IMG_DIR, exist_ok=True)
    # wipe clean IMG_DIR
    for f in os.listdir(IMG_DIR):
        if 'base.png' in f:
            continue
        if os.path.isfile(os.path.join(IMG_DIR, f)):
            os.remove(os.path.join(IMG_DIR, f))
    g = deepcopy(g)
    grammar = NLCGrammar()
    iter = 0
    path = os.path.join(IMG_DIR, "base.png")
    # draw_graph(g, path)    
    while len(g) > 1:
        iter += 1
        IMAGE_PATHS = [path]
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
        if not good:
            rule = NLCRule('black', g, None)
            rule_no = len(grammar.rules)
            rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))            
            grammar.add_rule(rule)
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
        nodes_induce = list(best_ism.nodes[0]['ism'])
        rhs_graph = deepcopy(nx.induced_subgraph(g, nodes_induce))
        for c in best_clique:
            ism = best_ism.nodes[c]
            nodes = ism['ism']    
            new_n = max(list(g))+1
            print(new_n)
            subneis = [n for n in neis(g, nodes) if n < new_n]
            print(subneis)
            g.add_node(new_n, label='gray')
            for n in nodes:
                g.remove_node(n)
            for subnei in subneis:
                g.add_edge(new_n, subnei)    
        path = os.path.join(IMG_DIR, f'api_{iter}.png')
        draw_graph(g, path)    
        print(best_ism.nodes[0])
        rule = NLCRule('gray', rhs_graph, lower)
        rule_no = len(grammar.rules)
        rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
        grammar.add_rule(rule)
    return grammar
    

