from src.config import *
import os
import networkx as nx
import numpy as np
import pickle
import json
from copy import deepcopy
from src.grammar.ednce import *
from src.draw.graph import *
from src.api.get_motifs import *
from src.algo.utils import *
from src.algo.common import *
from src.grammar.common import *
from src.grammar.utils import *


def edit_grammar(grammar, conn_g):
    conn_g_copy = conn_g.__class__(conn_g)
    rule = EDNCERule("black", conn_g_copy, None, None)
    exist = grammar.check_exists(rule)
    if exist is None:
        rule_no = len(grammar.rules)
        rule.visualize(os.path.join(IMG_DIR, f"rule_{rule_no}.png"))
        grammar.add_rule(rule)
    else:
        rule_no = exist
    return rule_no


def terminate(g, grammar, anno, iter):
    def is_init_graph(g_init):
        return len(g_init) == 1 and g_init.nodes[list(g_init)[0]]['label'] == 'black'
    # create a rule for each connected component
    conns = list(nx.connected_components(nx.Graph(g)))
    conns = sorted(conns, key=lambda c: get_prefix(list(c)[0]))
    if len(conns) == 1:
        nodes = list(g)
        if is_init_graph(g):
            model = anno[list(anno)[-1]]
        else:
            new_n = find_next(g)        
            rule_no = edit_grammar(grammar, g)
            anno[new_n] = EDNCENode(new_n, attrs={"rule": rule_no, "nodes": nodes, "feats": [g.nodes[n]['feat'] if 'feat' in g.nodes[n] else 0.0 for n in nodes]})
            g.add_node(new_n, label="black")  # annotate which rule was applied to model
            rewire_graph(g, new_n, nodes, anno)
            model = anno[new_n]
    else:
        model = []
        while conns:
            conn = conns.pop(-1)
            nodes = conn
            prefixes = [n.split(":")[0] for n in conn]
            assert len(set(prefixes)) == 1
            prefix = prefixes[0] + ":"
            conn_g = copy_graph(g, conn)
            if is_init_graph(conn_g):            
                model.append(anno[list(conn_g)[-1]])
                continue
            new_n = find_next(conn_g, prefix)
            rule_no = edit_grammar(grammar, conn_g)            
            anno[new_n] = EDNCENode(new_n, attrs={"rule": rule_no, "nodes": nodes, "feats": [conn_g.nodes[n]['feat'] if 'feat' in conn_g.nodes[n] else 0.0 for n in nodes]})
            g.add_node(new_n, label="black")  # annotate which rule was applied to model            
            rewire_graph(g, new_n, nodes, anno)
            model.append(anno[new_n])
    return grammar, model, anno, g


def find_indices(graphs, query):
    ans = []
    for i in range(len(graphs)):
        if nx.is_isomorphic(graphs[i], query, node_match=node_match):
            ans.append(i)
    return ans  


def find_partial(graphs, query):
    ans = []
    # query can be a (possibly disconnected) directed graph
    # query_und = nx.Graph(query)
    for i in range(len(graphs)):
        bad = False
        if len(query) > len(graphs[i]):
            continue
        # for conn in nx.connected_components(query_und):
            # conn_g = copy_graph(query, conn)
        gm = DiGraphMatcher(graphs[i], query, node_match=node_match)
        ism_iter = list(gm.subgraph_isomorphisms_iter())
        if len(ism_iter) == 0:
            break
        if not bad:
            ans.append(i)
    return ans


def worker(shared_queue, grammar, graphs, found, lock):
    while True:
        with lock:
            if shared_queue.empty():
                print("process done")
                break
            print(f"len(interms): {shared_queue.qsize()}")
            interm, deriv, poss = shared_queue.get()
        print(f"deriv: {deriv}")
        nts = grammar.search_nts(interm, NONTERMS)
        if len(nts) == 0:
            if nx.is_isomorphic(interm, graphs[poss], node_match=node_match):
                with lock:
                    found[poss].append(deriv)
                    print(f"found {deriv} graph {poss}, count: {len(found[poss])}")
        for j, nt in enumerate(nts):
            for i, rule in enumerate(grammar.rules):                      
                nt_label = interm.nodes[nt]['label']
                if rule.nt == nt_label:
                    c = deepcopy(interm)
                    c = rule(c, nt)
                    if nx.is_connected(nx.Graph(c)):
                        # if poss == 0 and i == 62:
                        #     pdb.set_trace()                     
                        ts = [x for x in c if c.nodes[x]['label'] in TERMS]
                        c_t = copy_graph(c, ts)
                        exist = find_partial([graphs[poss]], c_t)
                        if exist:
                            with lock:
                                shared_queue.put((c, deriv+[i], poss))       


def resolve_ambiguous(model, grammar):
    logger = logging.getLogger('global_logger')
    graphs = []
    for j in range(len(model)):
        deriv = [model[j].graph[n].attrs['rule'] for n in model[j].seq[::-1]]
        t2r = {i:i for i in range(len(grammar.rules))}
        deriv_g = grammar.derive(deriv, t2r)
        # draw_graph(deriv_g, '/home/msun415/test.png')
        graphs.append(deriv_g)    

    NUM_PROCS = 50
    N = len(graphs)
    manager = mp.Manager()
    shared_queue = manager.Queue()
    found = manager.list()
    g = nx.DiGraph()
    g.add_node('0', label='black')
    for j in range(N):
        shared_queue.put((deepcopy(g), [], j))
        found.append(manager.list())
    lock = manager.Lock()    
    processes = []
    for _ in range(NUM_PROCS):
        p = mp.Process(target=worker, args=(shared_queue, grammar, graphs, found, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_derivs = list(map(list, found))
    sets_of_sets = []
    for derivs in all_derivs:
        sets = []
        for i in range(len(derivs)):
            # keep this deriv
            for j in range(len(derivs)):
                if j == i:
                    continue
                sets.append(set(derivs[j])-set(derivs[i]))
        sets_of_sets.append(sets)

    poss_elims = []
    args = [sets for sets in sets_of_sets if sets]  
    if len(args) == 0:
        logger.info("no ambiguity, done")
        return
    for chosen in product(*args):
        elim = set.union(*chosen)
        exist = False
        for p in poss_elims:
            if p == elim:
                exist = True
                break
        if not exist:
            poss_elims.append(elim)

    poss_elims = sorted(poss_elims, key=len)
    for i in range(len(poss_elims)):
        if poss_elims[i] is None:
            continue
        for j in range(i+1, len(poss_elims)):
            if poss_elims[j] is None:
                continue
            if not (poss_elims[i]-poss_elims[j]):
                poss_elims[j] = None

    min_poss_elims = list(filter(lambda x: x is not None, poss_elims))
    best_e = None
    best_counter = None
    for e in min_poss_elims:
        counter = []
        for i, derivs in enumerate(all_derivs):
            inters = [bool(set(derivs[j]) & e) for j in range(len(derivs))]
            if np.all(inters):
                counter.append(i)
        if best_counter is None or len(counter) < len(best_counter):
            best_counter = counter
            best_e = e

    
    data = {'rules': best_e,
            'redo': best_counter}
    json.dump(data, open(args.ambiguous_ckt_file, 'w+'))
    logger.info(f"{len(best_counter)} ambiguous: {best_counter} with rules: {best_e}")


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
    emb = lower if MIN_EMBEDDING else L2-upper
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


def update_graph_mp(g, best_clique, shared_data, lock, index=-1):
    grammar = shared_data['grammar']
    best_ism = shared_data['best_ism']
    anno = shared_data['anno']
    change = False
    for c in best_clique:
        # g_cache = deepcopy(g)
        # anno_cache = deepcopy(anno)
        ism = best_ism.nodes[c]
        nodes = ism["ism"]
        dirs = ism["dirs"]
        ps = ism["ps"]
        assert len(set([get_prefix(n) for n in nodes])) == 1
        no = get_prefix(nodes[0])
        prefix = f"{no}:"
        new_n = find_next(g, prefix=prefix)
        g.add_node(new_n, label="gray")  # annotate which rule was applied to model        
        with lock:
            anno[new_n] = EDNCENode(
                new_n, attrs={
                    "rule": len(grammar.rules)-1 if index == -1 else index,
                    "nodes": nodes,
                    "feats": [g.nodes[n]['feat'] if 'feat' in g.nodes[n] else 0.0 for n in nodes]
                    }
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


def update_graph(g, anno, best_ism, best_clique, grammar, index=-1):
    conns = list(nx.connected_components(nx.Graph(g)))        
    if len(conns) > 1 and NUM_PROCS > 1:
        by_no = {}
        for c in best_clique:
            ism = best_ism.nodes[c]
            nodes = ism["ism"]
            no = get_prefix(nodes[0])            
            by_no[no] = by_no.get(no, []) + [c]
        args = []        
        manager = mp.Manager()
        lock = manager.Lock()
        shared_data = manager.dict(anno=anno, best_ism=best_ism, grammar=grammar)        
        for no in by_no:
            conn = copy_graph(g, [n for n in g if get_prefix(n)==no])
            args.append((conn, by_no[no], shared_data, lock))
        with mp.Pool(NUM_PROCS) as p:
            conns = p.starmap(update_graph_mp, tqdm(args, desc="updating graph"))
        change = False
        for i, no in enumerate(by_no):
            # remove conn
            conn = args[i][0]
            for c in conn:
                g.remove_node(c)
            # add new conn
            conn, change_i = conns[i]
            change |= change_i
            for n, data in conn.nodes(data=True):
                g.add_node(n, **data)
            for e0, e1, data in conn.edges(data=True):
                g.add_edge(e0, e1, **data)
        return g, change
    else:
        change = False
        for c in best_clique:
            # g_cache = deepcopy(g)
            # anno_cache = deepcopy(anno)
            ism = best_ism.nodes[c]
            nodes = ism["ism"]
            dirs = ism["dirs"]
            ps = ism["ps"]
            assert len(set([get_prefix(n) for n in nodes])) == 1
            no = get_prefix(nodes[0])
            prefix = f"{no}:"
            new_n = find_next(g, prefix=prefix)
            g.add_node(new_n, label="gray")  # annotate which rule was applied to model        
            anno[new_n] = EDNCENode(
                new_n, attrs={
                    "rule": len(grammar.rules) - 1 if index == -1 else index,
                    "nodes": nodes,
                    "feats": [g.nodes[n]['feat'] if 'feat' in g.nodes[n] else 0.0 for n in nodes]
                    }
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
    logger = logging.getLogger('global_logger')
    logger.info("begin compress grammar")
    changed = True
    while changed:
        changed = False
        best_i = -1
        best_ism = None
        best_clique = None
        best_rule_idx = None
        max_len = 0
        for index, rule in enumerate(grammar.rules):            
            ism_subgraph = find_iso(rule.subgraph, g, rule=rule)
            if len(ism_subgraph):                
                max_clique = approximate_best_clique(ism_subgraph)
                if len(max_clique) * len(rule.subgraph) > max_len:
                    best_i = index
                    max_len = len(max_clique) * len(rule.subgraph)
                    best_ism = ism_subgraph
                    best_clique = max_clique
                    best_rule_idx = index
        if best_clique is not None:
            best_comps = list(set([get_prefix(best_ism.nodes[c]['ism'][0]) for c in best_clique]))                
            lower_best, ous_best = reduce_to_bounds([best_ism.nodes[n] for n in best_clique])
            ## update grammar
            # rule = grammar.rules[best_rule_idx]
            # logger.info(f"revising rule {best_rule_idx}")
            # rule.embedding = rule.embedding | lower_best
            # rule.upper = rule.upper & ous_best
            # assert not (rule.embedding & rule.upper)
            # grammar.rules[best_rule_idx] = rule
            ## conserve grammar
            if (lower_best-rule.embedding): # extra instructions
                continue
            if ous_best & rule.embedding: # conflict
                continue
            logger.info(f"subgraph {best_i} occurred {len(best_clique)} times across components {sorted(best_comps)}")            
            g, _ = update_graph(
                g, anno, best_ism, best_clique, grammar, index=best_rule_idx
            )
            changed = True
    logger.info("done compress grammar")
    return g, anno


def dfs(anno, k):
    def _dfs(anno, k, anno_k):            
        for c in anno[k].children:
            _dfs(anno, c.id, anno_k)
        anno_k[k] = anno[k]
    anno_k = {}
    _dfs(anno, k, anno_k)
    return anno_k
        


def learn_grammar(g, args):
    orig = deepcopy(g)
    logger = create_logger(
        "global_logger",
        f"{wd}/data/{METHOD}_{DATASET}_{GRAMMAR}{SUFFIX}.log",
    )
    cache_iter, cache_path = setup()
    g, grammar, anno, iter = init_grammar(g, cache_iter, cache_path, EDNCEGrammar)
    while not term(g):
        iter += 1
        # img_paths = partition_graph(g, iter, NUM_PARTITON_SAMPLES_FOR_MOTIFS)
        # all_subgraphs = obtain_motifs(g, img_paths)
        all_subgraphs = obtain_motifs(g, [])
        for i, subgraph in enumerate(all_subgraphs):
            path = os.path.join(IMG_DIR, f"iter_{iter}_motif_{i}.png")
            draw_graph(subgraph, path)
            logger.info(f"subgraph {i} {path}")        
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
        logger.info(f"graph at iter {iter} has {len(g)} nodes")        
        if VISUALIZE:
            draw_graph(g, path)
        suffix = ('_' + Path(args.ambiguous_ckt_file).stem) if args.ambiguous_ckt_file is not None and os.path.exists(args.ambiguous_ckt_file) else  ''
        cache_path = os.path.join(CACHE_DIR, f"{iter}{suffix}.pkl")
        pickle.dump((grammar, anno, g), open(cache_path, "wb+"))

    grammar, model, anno, g = terminate(g, grammar, anno, iter)
    suffix = ('_' + Path(args.ambiguous_ckt_file).stem) if args.ambiguous_ckt_file is not None and os.path.exists(args.ambiguous_ckt_file) else  ''
    cache_path = os.path.join(CACHE_DIR, f"{iter}{suffix}.pkl")
    pickle.dump((grammar, anno, g), open(cache_path, "wb+"))        
    if isinstance(model, list):
        for j, m in enumerate(model):
            pre = get_prefix(m.id)
            # draw_tree(m, os.path.join(IMG_DIR, f"model_{iter}_{pre}.png"))
            model[j] = EDNCEModel(dfs(anno, m.id))
        # Debug
        # revised = lambda rid: re.search(f'revising rule {rid}\n', open('data/api_ckt_ednce.log').read())
        # revised_seq = [not np.any([revised(anno[rid].attrs['rule']) for rid in m.seq]) for m in model]
        # model_ids = [get_prefix(m.seq[0]) for m in model]
        # safe_seqs = [idx for b, idx in zip(revised_seq, model_ids) if b]
    else:
        model = anno[find_max(anno)]
        draw_tree(model, os.path.join(IMG_DIR, f"model_{iter}.png"))
        model = EDNCEModel(anno)
    if args.ambiguous_ckt_file:
        resolve_ambiguous(model, grammar)        
    ## Debug
    if isinstance(model, list):
        for m in list(model)[::-1]:
            res = m.generate(grammar)            
            match = False
            for i in range(len(res)):
                p = get_prefix(list(res[i])[0])
                nodes = list(filter(lambda x:get_prefix(x)==p, list(orig)))
                g_sub = copy_graph(orig, nodes)
                if nx.is_isomorphic(g_sub, res[i], node_match=node_match):
                    match = True
                    break
            if not match:
                breakpoint()
    else:
        model.generate(grammar) # verify logic is correct        
    return grammar, model
