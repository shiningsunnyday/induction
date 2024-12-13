from typing import List, Dict, Any
from src.draw.graph import draw_graph, draw_circuit
from src.config import *
import os
from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import uuid
from copy import deepcopy
from itertools import product
from .nlc import NLCGrammar, NLCModel
from src.grammar.common import *
from src.grammar.utils import *
import random
from collections import Counter
import sys

sys.path.append(os.path.join(os.getcwd(), "../CktGNN/"))
from utils import is_valid_Circuit, is_valid_DAG
import igraph
from tqdm import tqdm

graph_proxy = None

def specification(graph):
    if "ckt" in DATASET:
        type_count = Counter([graph.nodes[n]["type"] for n in graph])
        return type_count["input"] == 1 and type_count["output"] == 1
    return True
    # if 'ckt' in


def find_start_node(graph):
    in_n_ind = [graph.nodes[n]["type"] for n in graph].index("input")
    in_n = list(graph)[in_n_ind]
    return in_n


def node_match(d1, d2):
    return d1.get("label", "#") == d2.get("label", "#")  


def postprocess(graph):
    def dfs(n, vis, edges):
        vis[n] = -1
        out = False
        for nei in graph[n]:
            if vis[nei]:
                continue
            if graph.nodes[nei]["type"] == "output":
                out = True
                edges.append((n, nei))
            elif dfs(nei, vis, edges):
                out = True
                edges.append((n, nei))
        vis[n] = 1
        return out

    in_n = find_start_node(graph)
    edges = []
    vis = {n: 0 for n in graph}
    dfs(in_n, vis, edges)
    graph = nx.edge_subgraph(graph, edges)
    return graph


class EDNCEGrammar(NLCGrammar):
    def __sample__(self):
        # find the initial rule
        rules = self.search_rules("gray")
        init_rules = self.search_rules("black")
        start_rule = random.choice(init_rules)
        cur = start_rule.subgraph
        assert not check_input_xor_output(cur)
        num_nts = len(self.search_nts(cur, NONTERMS))
        iters = 0
        gen_dir = os.path.join(IMG_DIR, "generate/")
        while num_nts > 0:
            if iters >= 100:
                return None
            gray_nodes = self.search_nts(cur, ["gray"])
            # attempt to keep graph connected
            updated = False
            for node in np.random.permutation(gray_nodes):
                for rule in np.random.permutation(rules):
                    res = rule(cur, node)
                    if nx.is_connected(nx.Graph(res)):
                        updated = True
                        cur = res
                        break
                if updated:
                    break
            num_nts = len(self.search_nts(cur, "gray"))
            iters += 1
        return cur

    
    def derive(self, seq, token2rule):
        # find the initial rule
        seq = [token2rule[idx] for idx in seq]
        start_rule = self.rules[seq[0]]
        cur = start_rule.subgraph
        assert not check_input_xor_output(cur)
        for idx in seq[1:]:
            nt_nodes = self.search_nts(cur, NONTERMS)            
            if len(nt_nodes) == 0:
                break
            assert len(nt_nodes) == 1
            node = nt_nodes[0]
            rule = self.rules[idx]
            cur = rule(cur, node)
        return cur        


    def induce_recurse(self, node, model, g):
        fig, ax = plt.subplots()
        nx.draw(g, ax=ax, with_labels=True)                
        rule_id = model.graph[node].attrs['rule']
        if 'nodes' not in model.graph[node].attrs:
            breakpoint()
        nodes = model.graph[node].attrs['nodes']
        rhs = self.rules[rule_id].subgraph        
        rhs = copy_graph(rhs, list(rhs))       
        name_map = dict(zip(rhs, nodes))
        rhs = nx.relabel_nodes(rhs, name_map)
        # set the feats
        for n in rhs:
            rhs.nodes[n]['feat'] = model.feat_lookup[n]        
        for a, b in rhs.edges:
            rhs.edges[(a,b)]['level'] = 1
        g = nx.union(g, rhs)
        for n in rhs:
            g.add_edge(node, n, level=2, label='black')
        for c in model.graph[node].children:
            g = self.induce_recurse(c.id, model, g)            
        return g
        

    def induce(self, model):
        if isinstance(model, list):
            return [self.induce(m) for m in model]
        g = nx.DiGraph()
        root = list(model.graph)[-1]
        nt = self.rules[model.graph[root].attrs['rule']].nt
        g.add_node(root, label=nt)
        g = self.induce_recurse(root, model, g)        
        return g

    def search_rules(self, nt):
        rules = []
        for i, rule in enumerate(self.rules):
            if rule.nt == nt:
                rules.append(rule)
        return rules

    def check_exists(self, rule):
        for i, r in enumerate(self.rules):
            if not nx.is_isomorphic(rule.subgraph, r.subgraph, node_match=node_match):
                continue
            if rule.embedding != r.embedding:
                continue
            if rule.upper != r.upper:
                continue
            return i
        return None

    @staticmethod
    def search_nts(cur, nts):
        res = list(filter(lambda x: cur.nodes[x]["label"] in nts, cur))
        return res

    def generate(self, num_samples=10):
        count = 0
        random.seed(SEED)
        np.random.seed(SEED)
        gen_dir = os.path.join(IMG_DIR, "generate/")
        os.makedirs(gen_dir, exist_ok=True)
        # metrics
        is_valid_dag = []
        is_valid_circuit = []
        res = []
        while len(res) < num_samples:
            print(len(res))
            sample = self.__sample__()
            if sample is None:
                print("sample is None")
                continue
            bad = False
            for e in sample.edges:
                if sample.edges[e]["label"] in NONFINAL:
                    bad = True
            if bad:
                print("non-final nodes or edges")
                continue
            exist = False
            # for circuits
            if not nx.is_connected(nx.Graph(sample)):
                print("not connected")
                continue
            if "ckt" in DATASET:
                try:
                    sample = postprocess(sample)
                except:
                    continue
            if len(sample) == 0:
                continue
            for r in res:
                if nx.is_isomorphic(sample, r, node_match=node_match):
                    exist = True
                    break
            if exist:
                print("isomorphic to existing")
                continue
            draw_graph(
                sample,
                os.path.join(gen_dir, f"graph_{len(res)}.png"),
                node_size=NODE_SIZE,
            )
            isample = nx_to_igraph(sample)
            is_valid_dag.append(is_valid_DAG(isample, subg=False))
            is_valid_circuit.append(is_valid_Circuit(isample, subg=False))
            # draw_circuit(sample, os.path.join(gen_dir, f'graph_{len(res)}.png'))
            if is_valid_dag[-1] and is_valid_circuit[-1]:
                res.append(sample)
            count += 1
            print(f"is_valid_dag: {sum(is_valid_dag)}/{count}")
            print(f"is_valid_circuit: {sum(is_valid_circuit)}/{count}")
        return res


class EDNCERule:
    def __init__(self, nt, subgraph, embedding, upper):
        self.nt = nt
        self.subgraph = subgraph
        self.embedding = embedding
        self.upper = upper

    def __call__(self, cur, node):
        rhs = nx.DiGraph(self.subgraph)
        if ":" in node:
            start = find_next(cur, node[: node.index(":") + 1])
        else:
            start = find_next(cur)
        node_map = {}
        for n in rhs:
            node_map[n] = start
            start = next(start)
        inv_node_map = {v: k for k, v in node_map.items()}
        rhs = nx.relabel_nodes(rhs, node_map)
        cur = nx.union(cur, rhs)
        cur_neis = neis(cur, [node], direction=["in", "out"])
        for cur_nei in cur_neis:
            mu = cur.nodes[cur_nei]["label"]
            for i, cur_node in enumerate(rhs):
                if cur.has_edge(node, cur_nei):
                    d = "out"
                    p = cur[node][cur_nei]["label"]
                else:
                    d = "in"
                    p = cur[cur_nei][node]["label"]
                if self.embedding is not None:
                    for emb in self.embedding:
                        mu_e, p_e, q_e, i_e, d_e, d__e = emb
                        if mu_e == mu and p_e == p and i_e == i and d_e == d:
                            if d__e == "in":
                                cur.add_edge(cur_nei, cur_node, label=q_e)
                            else:
                                cur.add_edge(cur_node, cur_nei, label=q_e)
        cur.remove_node(node)
        return cur

    def visualize(self, path):
        g = nx.Graph()
        g.add_node(0, label=self.nt)
        lhs_path = os.path.join(IMG_DIR, f"{uuid.uuid4()}.png")
        draw_graph(g, lhs_path, scale=5)
        rhs_path = os.path.join(IMG_DIR, f"{uuid.uuid4()}.png")
        g = copy_graph(self.subgraph, list(self.subgraph))
        # g.graph['layout'] = 'spring_layout'
        if self.embedding is not None:
            for e in self.embedding:
                mu, p, q, x, d, d_ = e
                n = list(self.subgraph)[x]
                if "INVERSE_LOOKUP" in globals() and mu in INVERSE_LOOKUP:
                    type_ = INVERSE_LOOKUP[mu]
                    name = f"{x},{type_},{p}/{q},{d}/{d_}"
                    g.add_node(name, label=mu, type=type_, alpha=0.5)
                else:
                    name = f"{x},{mu},{p}/{q},{d}/{d_}"
                    g.add_node(name, label=mu, alpha=0.5)
                g.add_edge(
                    n,
                    name,
                    style="dashed",
                    reverse1=d_ == "in",
                    reverse2=d == "in",
                    label1=q,
                    loc1=1 / 4,
                    label2=p,
                    loc2=3 / 4,
                )
        draw_graph(
            g,
            rhs_path,
            scale=RULE_SCALE,
            node_size=RULE_NODE_SIZE,
            font_size=RULE_FONT_SIZE,
        )
        self.draw_fig(lhs_path, rhs_path, path)

    def draw_fig(self, lhs_path, rhs_path, path):
        # Load PNG images
        lhs_image = Image.open(
            lhs_path
        )  # Replace 'path_to_lhs_image.png' with your file path
        rhs_image = Image.open(
            rhs_path
        )  # Replace 'path_to_rhs_image.png' with your file path

        # Convert images to arrays if necessary
        lhs_array = np.array(lhs_image)
        rhs_array = np.array(rhs_image)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Display images on the left and right axes
        ax1.imshow(lhs_array)
        ax1.set_title("LHS")
        ax1.axis("off")  # Turn off axis for a clean look

        ax3.imshow(rhs_array)
        ax3.set_title("RHS")
        ax3.axis("off")

        # Middle axis for arrow, set axis limits and turn off axis
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

        # Create an arrow that spans horizontally across the middle axis
        arrow = FancyArrowPatch(
            (0, 0.5),
            (1, 0.5),
            mutation_scale=20,
            arrowstyle="-|>",
            color="black",
            lw=3,  # Set line width to make the arrow thicker
            transform=ax2.transAxes,
        )  # Ensure the arrow uses the axis coordinates
        ax2.add_patch(arrow)
        fig.savefig(path, bbox_inches="tight", dpi=300)


class EDNCENode:
    def __init__(
        self, id: int, attrs: Dict[str, Any] = None, children: List["NCENode"] = None
    ):
        self.id = id
        self.attrs = attrs if attrs is not None else {}
        self.children = children if children is not None else []

    def add_child(self, node):
        print("add edge", self.id, node.id)
        self.children.append(node)


class EDNCEModel(NLCModel):
    def __init__(self, anno):
        super().__init__(anno)
        self.seq = list(anno)

    def __generate__(self, node, grammar, res):
        rule_no = node.attrs["rule"]
        rule = grammar.rules[rule_no]
        new_res = []
        for g in res:
            # try every non-terminal
            for n in g:
                if g.nodes[n]["label"] == rule.nt:
                    new_res.append(rule(g, n))
        res = new_res
        res = [r for r in res if nx.is_connected(nx.Graph(r))]
        # try every order
        from itertools import permutations

        res_all = []
        for child_order in permutations(node.children):
            res_cur = deepcopy(res)
            for c in child_order:
                res_cur = self.__generate__(c, grammar, res_cur)
            res_all += res_cur
        return res_all

    def generate(self, grammar):
        g = nx.DiGraph()
        n = self.seq[-1]
        g.add_node(n, label="black")
        res = [g]
        res = self.__generate__(self.graph[n], grammar, res)
        # prune unique
        new_res = []
        for r_new in res:
            if not nx.is_connected(nx.Graph(r_new)):
                continue
            exist = False
            for r_old in new_res:
                if nx.is_isomorphic(
                    r_new, r_old, node_match=node_match
                ):
                    exist = True
                    break
            if not exist:
                new_res.append(r_new)

        gen_dir = os.path.join(IMG_DIR, "generate/")
        os.makedirs(gen_dir, exist_ok=True)
        prefix = get_prefix(list(g.nodes)[0])
        for i, g in enumerate(new_res):
            draw_graph(g, os.path.join(gen_dir, f"{prefix}_{i}.png"))
        return new_res


def equiv_class(graph, nodes, out_ns):
    def label_edge(a, b):
        if b not in graph[a]:
            return "-"
        return graph[a][b]["label"]

    lookup = {}
    for n in out_ns:
        key = []
        for node in nodes:
            e = label_edge(n, node) + "_" + label_edge(node, n)
            key.append(e)
        key = "__".join(key)
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(n)
    equiv = []
    new_lookup = {}
    for index, key in enumerate(lookup):
        equiv.append([])
        for n in lookup[key]:
            equiv[-1].append(n)
            new_lookup[n] = index
    return equiv, new_lookup


def insets_and_outsets(graph, nodes):
    """
    Output a set of MUST connection instructions (inset)
    or IMPOSSIBLE connection instructions (outset)
    Each instruction is (mu, p/q, x, d, d')
        mu: a neighbor's node-label
        p: placeholder for edge-label after contraction
        d: placeholder for edge-direction after contraction
        d': direction of edge
        q: an edge-label of a neighbor
        x: node of daughter graph
    A edNCE instruction is (mu, p/q, x, d, d')  means: establish an edge with label q to node x of D
    from each mu-labeled p-(d) neighbor of m, where d is 'in' or 'out'. If d' != d, reverse the edge.
    Using nodes' (in & out) neighbors, we can infer a set of (mu, ?/q, x, ?, d')
    that HAS to be in the instruction set
    Using nodes' non-(in & out) neighbors, we can infer a set of (mu, ?/q, x, ?, d')
    that CANNOT be in the instruction set
    What about '?'? We enumerate every possible realization. Each realization has:
        1. An edge label p between the mother node and each neighbor y
        defined using vocabulary and current state of graph.
        2. A direction of the edge d between the mother node and y
    For each realization, we return the inset/outset and information to identify the realization.
    """
    # if sorted(list(nodes)) == sorted(['11', '1', '2', '12']):
    #     breakpoint()    
    out_ns = neis(graph, nodes, direction=["in", "out"])
    # # compute direction of mother-daughter
    # dirs = {}
    # for y in out_ns:
    #     num_in = 0
    #     num_out = 0
    #     for x in nodes:
    #         if graph.has_edge(x, y):
    #             num_out += 1
    #         if graph.has_edge(y, x):
    #             num_in += 1
    #     dirs[y] = 'in' if num_in > num_out else 'out'
    # d = 'in' if num_in > num_out else 'out'

    # find equivalent neighbors
    # lookup = {out_n: i for (i, out_n) in enumerate(out_ns)}
    # poss_dirs = list(product(*[['in','out'] for _ in out_ns])) # try every direction
    equiv, lookup = equiv_class(graph, nodes, out_ns)
    poss_dirs = list(product(*[["in", "out"] for _ in equiv]))
    # naively, we need to enumerate, for every neighbor, the direction and edge label
    # however, let's say there are two (out-)neighbors n1, n2 with same node label and same edge label+directions to n (in nodes)
    # then it is always better for the poss dir of n2 to be the same as n1
    # because if n2 is different, then this creates a redundant instruction in the inset
    # similarly, let's say there are two neighbors n1, n2 with same node label that is not connected to n
    # then it is always better for the poss dir of n2 to be the same as n1
    # because if n2 is different, then this creates redundant instructions in the outset
    poss_ps = list(
        product(*[NONFINAL for _ in out_ns])
    )  # try every edge non-final label

    if "ckt" in DATASET:
        ### for CKT ONLY
        # we can further reduce poss_dirs if all nodes in equiv class "precede" nodes or "succeed" nodes on the input-output path
        # do dfs from input to each node in equiv class
        # if no path crosses nodes, then it's true
        # TODO: implement this
        poss_ps = list(
            product(*[FINAL for _ in out_ns])
        )
        # find source        
        prefix = list(nodes)[0][0]
        assert graph.nodes[f"{prefix}:0"]['type'] == 'input'
        # single source shortest paths
        equiv_dir = []
        for e in equiv:
            try:
                has_paths = [nx.has_path(graph, n, ei) for n in nodes for ei in e]
            except:
                breakpoint()
            if np.all(has_paths):
                equiv_dir.append(['out'])
            elif not np.any(has_paths):
                equiv_dir.append(['in'])
            else:
                equiv_dir.append(["in", "out"])
        poss_dirs = list(product(*equiv_dir))

    poss = {}
    for a, dirs in enumerate(poss_dirs):  # for each poss dirs
        for b, ps in enumerate(poss_ps):  # for each poss edge labels
            res_inset = set()
            res_outset = set()
            for i, x in enumerate(nodes):  # for each node
                for j, y in enumerate(out_ns):  # for each out nei
                    e = lookup[y]
                    try:
                        d = dirs[e]
                    except:
                        breakpoint()
                    p = ps[e]
                    label_y = graph.nodes[y]["label"]
                    mu = label_y
                    if graph.has_edge(x, y):
                        q = graph[x][y]["label"]
                        d_ = "out"
                        res_inset.add((mu, p, q, i, d, d_))
                    if graph.has_edge(y, x):
                        q = graph[y][x]["label"]
                        d_ = "in"
                        res_inset.add((mu, p, q, i, d, d_))
                    if not graph.has_edge(x, y):
                        d_ = "out"
                        for q in FINAL + NONFINAL:
                            res_outset.add((mu, p, q, i, d, d_))
                    if not graph.has_edge(y, x):
                        d_ = "in"
                        for q in FINAL + NONFINAL:
                            res_outset.add((mu, p, q, i, d, d_))
            # (a, b) stores the index of realizations
            res_dirs = {y: dirs[lookup[y]] for y in out_ns}
            res_ps = {y: ps[lookup[y]] for y in out_ns}
            poss[(a, b)] = (res_inset, res_outset, res_dirs, res_ps)
    return poss


def add_edge_mp(pair_list, graph_proxy):    
    graph, ism_graph, isms = graph_proxy['graph'], graph_proxy['ism_graph'], graph_proxy['isms']
    res = []
    for i, j in pair_list:
        if int(i.split("_")[0]) == int(j.split("_")[0]):
            res.append(False)
            continue
        ismA = isms[int(i.split("_")[0])]
        ismB = isms[int(j.split("_")[0])]
        # if ednce is linear
            # if i, j in same component
                # don't add edge    
        if LINEAR:
            if get_prefix(list(ismA)[0]) == get_prefix(list(ismB)[0]):
                res.append(False)
                continue
        touch = touching(graph, ismA, ismB)
        if touch:
            res.append(False)    
            continue
        inset_i = ism_graph.nodes[i]["ins"]
        outset_i = ism_graph.nodes[i]["out"]
        inset_j = ism_graph.nodes[j]["ins"]
        outset_j = ism_graph.nodes[j]["out"]
        overlap = (inset_i | inset_j) & (outset_i | outset_j)
        res.append(not overlap)
    print(f"{len(pair_list)} done")
    return res


def add_edge(i, j, graph, ism_graph, isms):    
    if int(i.split("_")[0]) == int(j.split("_")[0]):
        return False
    ismA = isms[int(i.split("_")[0])]
    ismB = isms[int(j.split("_")[0])]
    # if ednce is linear
        # if i, j in same component
            # don't add edge    
    if LINEAR:
        if get_prefix(list(ismA)[0]) == get_prefix(list(ismB)[0]):
            return False
    touch = touching(graph, ismA, ismB)
    if touch:
        return False
    inset_i = ism_graph.nodes[i]["ins"]
    outset_i = ism_graph.nodes[i]["out"]
    inset_j = ism_graph.nodes[j]["ins"]
    outset_j = ism_graph.nodes[j]["out"]
    overlap = (inset_i | inset_j) & (outset_i | outset_j)
    return not overlap


def touching(graph, ismA, ismB):
    nodesA = set(ismA)
    neisA = set(neis(graph, ismA))
    nodesB = set(ismB)
    neisB = set(neis(graph, ismB))
    touch = bool((nodesA | neisA) & nodesB) | bool(nodesA & (nodesB | neisB))
    return touch
    



def initialize(path):
    global graph
    global ism_graph
    global isms
    graph, ism_graph, isms = pickle.load(open(path, 'rb'))



def find_iso(subgraph, graph, rule=None):
    global graph_proxy
    isms = fast_subgraph_isomorphism(graph, subgraph)
    # if ednce is linear
        # if subgraph is terminal-only
            # remove all subgraph instances in comps with nt
    if LINEAR:
        isms_copy = []
        for ism in isms:
            term_only = True
            for n in subgraph:
                if subgraph.nodes[n]['label'] in NONTERMS:
                    term_only = False
            if not term_only:
                isms_copy.append(ism)
                continue
            try:
                pre = get_prefix(list(ism)[0])
            except:
                breakpoint()
            has_nt = False
            for n in graph:
                if get_prefix(n) == pre:
                    if graph.nodes[n]['label'] in NONTERMS:
                        has_nt = True
                        break
            if not has_nt:
                isms_copy.append(ism)
        isms = isms_copy
            
    ism_graph = nx.Graph()
    for i, ismA in enumerate(isms):
        poss = insets_and_outsets(graph, ismA)        
        for a, b in poss: # each one a possible contraction
            inset, outset, dirs, ps = poss[(a, b)]
            if rule is not None:
                inset = inset | rule.embedding
                outset = outset & rule.upper
                # if inset-rule.embedding:
                #     continue
                # if outset-rule.upper:
                #     continue
            if inset & outset:
                continue
            ism_graph.add_node(
                f"{i}_{a}_{b}", ins=inset, out=outset, ism=list(ismA), dirs=dirs, ps=ps
            )
    
    if NUM_PROCS == 1:
        all_args = list(product(list(ism_graph), list(ism_graph)))
        res = tqdm(
            [
                add_edge(i, j, graph, ism_graph, isms)
                for (i, j) in all_args
            ],
            desc="looping over pairs",
        )        
    else:
        with mp.Manager() as manager:
            graph_proxy = manager.dict(graph=graph, ism_graph=ism_graph, isms=isms)
            batch_size = 10000
            all_args = list(product(list(ism_graph), list(ism_graph)))
            num_batches = (len(all_args)+batch_size-1)//batch_size
            print(f"{num_batches} batches")
            args_batch_list = [(all_args[k*batch_size:(k+1)*batch_size], graph_proxy) for k in range(num_batches)]        
            with mp.Pool(NUM_PROCS) as p:
                res = p.starmap(add_edge, tqdm(args_batch_list, desc="looping over pairs"))
        res = sum(res, [])
    for (i, j), should_add in tqdm(zip(all_args, res), "adding edges"):
        if should_add:
            ism_graph.add_edge(i, j)
    return ism_graph
