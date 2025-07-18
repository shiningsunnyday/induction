from src.config import *
import os
import networkx as nx
import numpy as np
import pickle
from copy import deepcopy
from functools import reduce
from src.grammar.nce import *
from src.draw.graph import *
from src.api.get_motifs import *
from src.algo.utils import *
from src.algo.common import *
from src.grammar.common import *
from src.grammar.utils import *


def terminate(g, grammar, anno, iter):
    nodes = list(g)
    new_n = find_next(g)
    rule_no = len(grammar.rules)
    anno[new_n] = NCENode(new_n, attrs={"rule": rule_no})
    rule = NCERule("black", deepcopy(g), None)
    rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
    grammar.add_rule(rule)
    g.add_node(new_n, label="black")  # annotate which rule was applied to model
    rewire_graph(g, new_n, nodes, anno)
    model = anno[new_n]
    cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")
    pickle.dump((grammar, anno, g), open(cache_path, "wb+"))
    return g, grammar, model


def extract_rule(g, best_ism, best_clique, grammar):
    compats = [best_ism.nodes[n] for n in best_clique]
    lower = reduce(lambda x, y: x | y, [compat["ins"] for compat in compats])
    # L2 = set([(x,y) for x, y in product(LABELS, LABELS)])
    # ous = reduce(lambda x,y: x&y, [compat['out'] for compat in compats])
    # upper = L2 - ous
    nodes_induce = best_ism.nodes[list(best_ism)[0]]["ism"]
    rhs_graph = copy_graph(g, nodes_induce)
    color = "gray"
    if len(rhs_graph) == len(g):
        color = "black"
    rule = NCERule(color, deepcopy(rhs_graph), lower)
    rule_no = len(grammar.rules)
    rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
    grammar.add_rule(rule)


def update_graph(g, anno, best_ism, best_clique, grammar):
    change = False
    for c in best_clique:
        g_cache = deepcopy(g)
        anno_cache = deepcopy(anno)
        ism = best_ism.nodes[c]
        nodes = ism["ism"]
        new_n = find_next(g)
        g.add_node(new_n, label="gray")  # annotate which rule was applied to model
        anno[new_n] = NCENode(new_n, attrs={"rule": len(grammar.rules) - 1})
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
    g, grammar, anno, iter = init_grammar(g, cache_iter, cache_path, NCEGrammar)
    breakpoint()
    while not term(g):
        iter += 1
        img_paths = partition_graph(g, iter)
        if term(g):
            g, grammar, model = terminate(g, grammar, anno, iter)
            break
        all_subgraphs = obtain_motifs(g, img_paths)
        best_ism, best_clique = find_embedding(all_subgraphs, g, find_iso=find_iso)
        if best_ism is None:
            g, grammar, model = terminate(g, grammar, anno, iter)
            break
        extract_rule(g, best_ism, best_clique, grammar)
        g, change = update_graph(g, anno, best_ism, best_clique, grammar)
        if not change:
            g, grammar, model = terminate(g, grammar, anno, iter)
            break
        path = os.path.join(IMG_DIR, f"{METHOD}_{iter}.png")
        draw_graph(g, path)
        cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")
        pickle.dump((grammar, anno, g), open(cache_path, "wb+"))
    model = anno[find_max(anno)]
    draw_tree(model, os.path.join(IMG_DIR, f"model_{iter}.png"))
    model = NCEModel(model)
    return grammar, model
