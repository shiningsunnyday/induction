import networkx as nx
from src.config import NONTERMS

def boundary(g):
    bad = False
    for a, b in g.edges:
        if g.nodes[a]['label'] in NONTERMS and g.nodes[b]['label'] in NONTERMS:
            bad = True
            break
    return bad


def neis(graph, nodes, direction=['out']):
    res = []
    if 'out' in direction:
        ns = sum([list(graph[n]) for n in nodes], [])
        out_neis = list(set([n for n in ns if n not in nodes]))
        res += out_neis
    if 'in' in direction:
        ns = sum([list(graph.predecessors(n)) for n in nodes], [])
        in_neis = list(set([n for n in ns if n not in nodes]))
        res += in_neis
    res = list(set(res))
    return res


def get_groups(content, dtype=int):
    groups = []
    for l in content.split():
        l_str = l.replace(' ','').split(',')
        try:
            l_arr = list(map(dtype, l_str))
        except:
            continue
        groups.append(l_arr)
    return groups


def find_embedding(subgraphs, graph, find_iso, edges=False):
    # find_iso: a custom function that constructs the compat graph
    best_ism = None
    best_clique = None
    max_len = 0
    for subgraph in subgraphs:
        if len(subgraph) == 1:
            continue
        if boundary(subgraph):
            continue
        ism_subgraph = find_iso(subgraph, graph)        
        if len(ism_subgraph) == 0:
            continue
        print(subgraph.nodes, ism_subgraph.nodes)
        max_clique = []
        for ism_conn_subgraph in nx.connected_components(ism_subgraph):
            conn_subgraph = nx.induced_subgraph(ism_subgraph, ism_conn_subgraph)
            clique = list(nx.approximation.max_clique(conn_subgraph))
            if len(clique) > len(max_clique):
                max_clique = clique
        # max_clique = list(nx.find_cliques(ism_subgraph))
        better = False
        if edges:
            expr = len(max_clique)*len(subgraph.edges)
        else:
            expr = len(max_clique)*len(subgraph)
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