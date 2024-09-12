from src.config import *
import os
import networkx as nx
import numpy as np
import pickle
from copy import deepcopy
from src.grammar.ednce import *
from src.draw.graph import *
from src.api.get_motifs import *
from src.algo.utils import *
from src.algo.common import *
from src.grammar.common import *
from src.grammar.utils import *


def terminate(g, grammar, anno, iter):
    # create a rule for each connected component
    conns = list(nx.connected_components(nx.Graph(g)))
    if len(conns) == 1:
        nodes = list(g)
        new_n = find_next(g)
        rule_no = len(grammar.rules)
        anno[new_n] = EDNCENode(new_n, attrs={"rule": rule_no})
        g_copy = g.__class__(g)
        rule = EDNCERule("black", g_copy, None)
        rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
        grammar.add_rule(rule)
        g.add_node(new_n, label="black")  # annotate which rule was applied to model
        rewire_graph(g, new_n, nodes, anno)
        model = anno[new_n]
        cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")
        pickle.dump((grammar, anno, g), open(cache_path, "wb+"))
    else:
        model = []
        while conns:
            conn = conns.pop(-1)
            prefixes = [n.split(":")[0] for n in conn]
            assert len(set(prefixes)) == 1
            prefix = prefixes[0] + ":"
            conn_g = copy_graph(g, conn)
            new_n = find_next(conn_g, prefix)
            conn_g_copy = conn_g.__class__(conn_g)
            rule = EDNCERule("black", conn_g_copy, None, None)
            exist = grammar.check_exists(rule)
            if exist is None:
                rule_no = len(grammar.rules)
                rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
                grammar.add_rule(rule)
            else:
                rule_no = exist
            anno[new_n] = EDNCENode(new_n, attrs={"rule": rule_no})
            g.add_node(new_n, label="black")  # annotate which rule was applied to model
            nodes = conn
            rewire_graph(g, new_n, nodes, anno)
            model.append(anno[new_n])
    cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")
    pickle.dump((grammar, anno, g), open(cache_path, "wb+"))
    return g, grammar, model


def spec(u):
    # ignore rules of a certain form
    if "ckt" in DATASET:
        return (
            u[1] in NONFINAL
            and u[2] in FINAL
            and u[0] not in ["silver", "light_grey", "gray", "black"]
        )
    else:
        raise NotImplementedError


def gpt_add_diff(rhs_graph, diff):
    # for the purpose of textual representation, relabel the nodes in topological sort
    # won't cause side effects outside this function
    assert np.all([r[1] == "gray" for r in diff])
    assert np.all([r[2] == "black" for r in diff])
    rename = {
        "C": "capacitor",
        "R": "resistor",
        "+gm+": "+gm+",
        "-gm+": "-gm+",
        "+gm-": "+gm-",
        "-gm-": "-gm-",
        "input": "input",
        "output": "output",
    }
    inv_rename = {v: k for (k, v) in rename.items()}
    N = len(rhs_graph)
    lookup = {}
    comps = []
    for i, n in enumerate(rhs_graph):
        lookup[n] = i + 1
        t = rhs_graph.nodes[n]["type"]
        t = rename[t]
        comps.append(f"({i+1}) {t}")
    comp_str = ", ".join(comps)
    conns = []
    for src, tgt in rhs_graph.edges:
        t1 = rhs_graph.nodes[src]["type"]
        t2 = rhs_graph.nodes[tgt]["type"]
        t1, t2 = rename[t1], rename[t2]
        n1, n2 = lookup[src], lookup[tgt]
        conns.append(f"({n1}) {t1} to ({n2}) {t2}")
    conn_str = ", ".join(conns)

    # prompt = f"""
    #     Please design an op-amp with me. I represent the op-amp as a flow network, with current going from input to output. I want an op-amp with high Figure of Merit. My basic building blocks are resistor, capacitor, +gm+, -gm+, +gm-, -gm-, input and output.
    #     My design includes a sub-circuit with {N} components: {comp_str}. Current flows from {conn_str}. Please rank among the best building blocks to place next, such that current flows from component ({i+1}) {n} to it.
    #     DO NOT OUTPUT ANY REASONING. JUST OUTPUT THE ANSWER. Output your answer as a comma-separated sorted list of the possible components: BEST_COMPONENT,SECOND_BEST_COMPONENT,...,WORST_COMPONENT
    # """
    prompt = f"""
        Please design an op-amp with me. I represent the op-amp as a flow network, with signal going from input to output. I want an op-amp with high Figure of Merit. My basic building blocks are resistor, capacitor, +gm+, -gm+, +gm-, -gm-, input and output.
        My design includes a sub-circuit with {N} components: {comp_str}. Current flows from {conn_str}. Please rank the possible ways to add a new component, so signal flows into or out of an existing component to the new component.
        Output your answer over multiple lines, beginning with the best addition. For each line, output the id of an existing component, the building block type of the new component, and either "out" if signal should flow out from the existing component, or "in" if the signal flows into the existing component.
        ID,BUILDING_BLOCK,IN_OR_OUT
        ID,BUILDING_BLOCK,IN_OR_OUT
        ...
        ID,BUILDING_BLOCK,IN_OR_OUT
        
        DO NOT OUTPUT ANY REASONING. JUST OUTPUT THE ANSWER. 
    """
    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    res = completion.choices[0].message.content
    pat = r"(\d+),(capacitor|resistor|\+gm\+|-gm\+|\+gm-|-gm-|input|output),(in|out)"
    rank = []
    for line in res.split("\n"):
        mat = re.match(pat, line)
        if mat:
            comp, type_, dir = mat.groups()
            type_name = inv_rename[type_]
            mu = CKT_LOOKUP[type_name]
            x = int(comp) - 1
            for d in ["in", "out"]:
                rank.append((mu, "gray", "black", x, "in", dir))
    rank = [r for r in rank if r in diff]
    return rank


def extract_rule(g, best_ism, best_clique, grammar):
    compats = [best_ism.nodes[n] for n in best_clique]
    lower, ous = reduce_to_bounds(compats)
    nodes_induce = best_ism.nodes[list(best_clique)[0]]["ism"]
    rhs_graph = copy_graph(g, nodes_induce)
    L2 = set(
        list(
            product(
                TERMS + NONTERMS,
                FINAL + NONFINAL,
                FINAL + NONFINAL,
                range(len(rhs_graph)),
                ["in", "out"],
                ["in", "out"],
            )
        )
    )
    upper = L2 - ous
    # don't add unneccessary non-final edges
    diff = set(u for u in upper if u not in lower and spec(u))
    # ask gpt if any of the ones in diff should be added
    emb = lower
    # if 'ckt' in DATASET:
    #     if np.all(['type' in rhs_graph.nodes[n] for n in rhs_graph]):
    #         emb_diff = gpt_add_diff(rhs_graph, diff)
    #         emb = set(emb_diff[:TOP_DIFF]) | emb
    # ask gpt to rank the choices
    # TODO: implement this
    color = "gray"
    rule = EDNCERule(color, rhs_graph, emb, upper)
    # rule = EDNCERule(color, rhs_graph, upper)
    rule_no = len(grammar.rules)
    rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
    grammar.add_rule(rule)


def rewire_graph_ednce(g, new_n, nodes, dirs, ps, anno):
    for n in nodes:
        label = g.nodes[n]["label"]
        g.remove_node(n)
        # if n is annotated, add edge to model
        if n in anno:
            if label in NONTERMS:
                anno[new_n].add_child(anno[n])
    assert list(dirs) == list(ps)
    for k in dirs:
        label = ps[k]
        if dirs[k] == "out":
            g.add_edge(new_n, k, label=label)
        else:
            g.add_edge(k, new_n, label=label)


def update_graph(g, anno, best_ism, best_clique, grammar, index=-1):
    change = False
    for c in best_clique:
        g_cache = deepcopy(g)
        anno_cache = deepcopy(anno)
        ism = best_ism.nodes[c]
        nodes = ism["ism"]
        dirs = ism["dirs"]
        ps = ism["ps"]
        assert len(set([n.split(":")[0] for n in nodes])) == 1
        no = nodes[0].split(":")[0]
        prefix = f"{no}:"
        new_n = find_next(g, prefix=prefix)
        g.add_node(new_n, label="gray")  # annotate which rule was applied to model
        anno[new_n] = EDNCENode(
            new_n, attrs={"rule": len(grammar.rules) - 1 if index == -1 else index}
        )
        print(f"{new_n} new node")
        rewire_graph_ednce(g, new_n, nodes, dirs, ps, anno)
        change = True
        # if boundary(g):
        #     g = g_cache
        #     anno = anno_cache
        #     continue
        # else:
        #     change = True
    return g, change


def compress(g, grammar, anno):
    changed = True
    while changed:
        changed = False
        best_ism = None
        best_clique = None
        best_rule_idx = None
        max_len = 0
        for index, rule in enumerate(grammar.rules):
            ism_subgraph = find_iso(rule.subgraph, g, rule=rule)
            if len(ism_subgraph):
                max_clique = approximate_best_clique(ism_subgraph)
                if len(max_clique) * len(rule.subgraph) > max_len:
                    max_len = len(max_clique) * len(rule.subgraph)
                    best_ism = ism_subgraph
                    best_clique = max_clique
                    best_rule_idx = index
        if best_clique is not None:
            # update grammar
            lower_best, ous_best = reduce_to_bounds(
                [best_ism.nodes[n] for n in best_clique]
            )
            rule = grammar.rules[best_rule_idx]
            rule.embedding = rule.embedding | lower_best
            rule.upper = rule.upper & ous_best
            assert not (rule.embedding & rule.upper)
            grammar.rules[best_rule_idx] = rule
            g, _ = update_graph(
                g, anno, best_ism, best_clique, grammar, index=best_rule_idx
            )
    return g, anno


def learn_grammar(g):
    cache_iter, cache_path = setup()
    g, grammar, anno, iter = init_grammar(g, cache_iter, cache_path, EDNCEGrammar)
    while not term(g):
        iter += 1
        img_paths = partition_graph(g, iter, NUM_PARTITON_SAMPLES_FOR_MOTIFS)
        all_subgraphs = obtain_motifs(g, img_paths)
        g, anno = compress(g, grammar, anno)
        # path = os.path.join(IMG_DIR, f'{METHOD}_{iter-1}_compress.png')
        # draw_graph(g, path)
        best_ism, best_clique = find_embedding(
            all_subgraphs, g, find_iso=find_iso, edges=True
        )
        if best_ism is None:
            break
        extract_rule(g, best_ism, best_clique, grammar)
        g, change = update_graph(g, anno, best_ism, best_clique, grammar)
        if not change:
            break
        path = os.path.join(IMG_DIR, f"{METHOD}_{iter}.png")
        draw_graph(g, path)
        cache_path = os.path.join(CACHE_DIR, f"{iter}.pkl")
        pickle.dump((grammar, anno, g), open(cache_path, "wb+"))
    g, grammar, model = terminate(g, grammar, anno, iter)
    if isinstance(model, list):
        for j, m in enumerate(model):
            draw_tree(m, os.path.join(IMG_DIR, f"model_{iter}_{j}.png"))
    else:
        model = anno[find_max(anno)]
        draw_tree(model, os.path.join(IMG_DIR, f"model_{iter}.png"))
    model = EDNCEModel(anno)
    return grammar, model
