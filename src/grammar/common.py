import networkx as nx
from src.config import *
import igraph
from copy import deepcopy
import re
from functools import reduce
from tqdm import tqdm


def check_input_xor_output(subgraph):
    inp_outp = []
    for n in subgraph:
        if "type" in subgraph.nodes[n]:
            type_ = subgraph.nodes[n]["type"]
            inp_outp.append(type_ in ["input", "output"])
        else:
            assert subgraph.nodes[n]["label"] in NONTERMS
    return sum(inp_outp) == 1


def nx_to_igraph(g):
    g = deepcopy(g)
    for n in g:
        g.nodes[n]["type"] = list(CKT_LOOKUP).index(g.nodes[n]["type"])
    for e in g.edges:
        if "_igraph_index" in g.edges[e]:
            g.edges[e].pop("_igraph_index")
    for n in g:
        if "_igraph_index" in g.nodes[n]:
            g.nodes[n].pop("_igraph_index")
    ig = igraph.Graph(directed=True)
    ig = ig.from_networkx(g)
    return ig


def copy_graph(g, nodes):
    g_copy = g.__class__()
    for k in g.graph:
        g_copy.graph[k] = g.graph[k]
    for n in nodes:
        g_copy.add_node(n, **g.nodes[n])
    for e in g.edges(data=True):
        if e[0] not in nodes or e[1] not in nodes:
            continue
        g_copy.add_edge(e[0], e[1], **e[2])
    return g_copy


def boundary(g):
    bad = False
    for a, b in g.edges:
        if g.nodes[a]["label"] in NONTERMS and g.nodes[b]["label"] in NONTERMS:
            bad = True
            break
    return bad


def neis(graph, nodes, direction=["out"]):
    res = []
    if "out" in direction:
        ns = sum([list(graph[n]) for n in nodes], [])
        out_neis = list(set([n for n in ns if n not in nodes]))
        res += out_neis
    if "in" in direction:
        ns = sum([list(graph.predecessors(n)) for n in nodes], [])
        in_neis = list(set([n for n in ns if n not in nodes]))
        res += in_neis
    res = list(set(res))
    return res


def reduce_to_bounds(compats):
    lower = reduce(lambda x, y: x | y, [compat["ins"] for compat in compats])
    ous = reduce(lambda x, y: x & y, [compat["out"] for compat in compats])
    return lower, ous


def get_groups(content, dtype=int):
    groups = []
    for l in content.split():
        l = l.replace(" ", "")
        pat = r"(?:(\d+):)?((?:\d+,)*\d+)"  # 2422:2,3,4 or 2,3,4
        mat = re.match(pat, l)
        if mat is None:
            continue
        if ":" in l:
            index, l_arr = mat.groups()
        else:
            index = -1
            _, l_arr = mat.groups()
        l_arr = l_arr.split(",")
        l_arr = list(map(dtype, l_arr))
        if index != -1:
            groups.append({"index": index, "group": l_arr})
        else:
            groups.append(l_arr)
    return groups


def greedy_max_clique(graph):
    # Sort nodes by degree (high to low)
    nodes_sorted_by_degree = sorted(
        graph.nodes, key=lambda x: graph.degree[x], reverse=True
    )
    # Start with an empty clique
    max_clique = []
    # Iterate over the sorted nodes
    for node in nodes_sorted_by_degree:
        # Check if this node can be added to the current clique
        if all(graph.has_edge(node, clique_node) for clique_node in max_clique):
            max_clique.append(node)
    return max_clique


def approximate_best_clique(ism_subgraph):
    max_clique = []
    for ism_conn_subgraph in nx.connected_components(ism_subgraph):
        conn_subgraph = copy_graph(ism_subgraph, ism_conn_subgraph)
        print(
            f"approx max clique {len(conn_subgraph)} nodes {len(conn_subgraph.edges)} edges"
        )
        if len(conn_subgraph) > 1000:
            clique = greedy_max_clique(conn_subgraph)
        else:
            clique = list(nx.approximation.max_clique(conn_subgraph))
        if len(clique) > len(max_clique):
            max_clique = clique
        elif len(clique) == len(max_clique):
            lower_best, ous_best = reduce_to_bounds(
                [ism_subgraph.nodes[n] for n in max_clique]
            )
            lower, ous = reduce_to_bounds([ism_subgraph.nodes[n] for n in clique])
            if len(lower) + len(ous) < len(lower_best) + len(ous_best):
                print("better clique")
                max_clique = clique
    return max_clique


def non_isomorphic(all_subgraphs):
    subgraphs = []
    for s in all_subgraphs:
        exist = False
        for s_ in subgraphs:
            if nx.is_isomorphic(
                s,
                s_,
                node_match=lambda x, y: x["label"] == y["label"],
                edge_match=lambda x, y: x["label"] == y["label"],
            ):
                exist = True
                break
        if not exist:
            subgraphs.append(s)
    return subgraphs


def find_embedding(subgraphs, graph, find_iso, edges=False):
    # find_iso: a custom function that constructs the compat graph
    best_ism = None
    best_clique = None
    max_len = 0
    # eliminate common subgraphs
    subgraphs = non_isomorphic(subgraphs)
    for subgraph in tqdm(subgraphs, desc="looping over subgraphs"):
        # general concerns
        if len(subgraph) == 1:
            continue
        if boundary(subgraph):
            continue
        # domain-specific concerns
        if "ckt" in DATASET:
            if check_input_xor_output(subgraph):
                continue
        ism_subgraph = find_iso(subgraph, graph)
        if len(ism_subgraph) == 0:
            continue
        print(subgraph.nodes, ism_subgraph.nodes)
        max_clique = approximate_best_clique(ism_subgraph)
        # max_clique = list(nx.find_cliques(ism_subgraph))
        better = False
        if edges:
            expr = len(max_clique) * len(subgraph.edges)
        else:
            expr = len(max_clique) * len(subgraph)
        better = expr > max_len
        if better:
            max_len = expr
            best_ism = ism_subgraph
            best_clique = max_clique
    # ism_subgraph: compatibility graph
    # best_ism: best subgraph
    # best cliques: best clique in ism_subgraph for best_ism
    # return best_ism, best_clique
    return best_ism, best_clique
