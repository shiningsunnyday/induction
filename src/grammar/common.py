import networkx as nx
from src.config import *
from src.grammar.utils import *
import igraph
from copy import deepcopy
import re
from functools import reduce
from tqdm import tqdm
from argparse import ArgumentParser
from networkx.algorithms.isomorphism import DiGraphMatcher
import multiprocessing as mp
import time
import random
import json

def get_parser():
    parser = ArgumentParser()
    # folder params
    parser.add_argument("--cache_root", help="if given, save/load all cached data from here")
    # global args
    parser.add_argument("--visualize", dest="global_visualize", action='store_true')
    parser.add_argument("--cache", dest="global_cache", action='store_true')    
    parser.add_argument("--num_threads", dest="global_num_threads", type=int)
    parser.add_argument("--num_procs", dest="global_num_procs", type=int)        
    # hparams
    parser.add_argument("--scheme", choices=['one','zero'], help='whether to index from 0 or 1', default='zero')    
    # ablations
    parser.add_argument("--ablate_tree", action='store_true') 
    parser.add_argument("--ablate_merge", action='store_true') 
    parser.add_argument("--ablate_root", action='store_true') 
    parser.add_argument("--text_only", action='store_true') 
    # task params
    parser.add_argument("--task", nargs='+', choices=["learn","generate","prediction"])
    parser.add_argument("--seed")
    parser.add_argument("--grammar_ckpt")
    # mol dataset args
    parser.add_argument(
        "--mol-dataset",
        choices=["ptc","hopv","polymers_117", "isocyanates", "chain_extenders", "acrylates","moses", "molqa"]+[f"molqa_{i}" for i in range(6)],
    )
    parser.add_argument(
        "--num-data-samples", type=int
    )
    parser.add_argument(
        "--num-samples", type=int
    )
    parser.add_argument("--ambiguous-file", help='if given and exists, load data from this file to learn grammar and save any ambiguity to the next version; if given and not exist, save ambiguous data to this file after learn grammar')
    # bo args
    parser.add_argument("--checkpoint", help="which ckpt to load", type=int)                     
    return parser    


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    if args.ambiguous_file is not None:
        assert args.ambiguous_file[-5:] == '.json'
    return args


# import pygraphviz as pgv
class MyGraph(nx.DiGraph):
    def __init__(self, graph=None):
        if graph is None:
            graph = nx.DiGraph()
        self.comps = {}
        for n in graph:
            try:
                pre = get_prefix(n)
            except:
                continue
            if pre not in self.comps:
                self.comps[pre] = []
            self.comps[pre].append(n)
        super().__init__(graph)


    def remove_node(self, n):
        try:
            pre = get_prefix(n)
        except ValueError:
            super().remove_node(n)
            return
        # while True:
        #     try:
        #         self.comps[pre].remove(n)
        #     except ValueError:
        #         break
        try:
            self.comps[pre].remove(n)  
        except:
            breakpoint()
        if len(self.comps[pre]) == 0:
            self.comps.pop(pre)
        super().remove_node(n)


    def add_node(self, n, **kwargs):
        try:
            pre = get_prefix(n)
        except ValueError:
            super().add_node(n, **kwargs)
            return
        if pre not in self.comps:
            self.comps[pre] = []
        self.comps[pre].append(n)
        super().add_node(n, **kwargs)
    

    def add_edge(self, u, v, **kwargs):
        if u not in self:
            self.add_node(u)
        if v not in self:
            self.add_node(v)
        super().add_edge(u, v, **kwargs)


def set_global_args(args):
    res = {}
    idx = 1  
    for k, v in args.__dict__.items():
        if v is None:
            continue
        if k[:7] == 'global_':
            res[k[7:].upper()] = v
    return res


def check_input_xor_output(subgraph):
    inp_outp = []
    for n in subgraph:
        if "type" in subgraph.nodes[n]:
            type_ = subgraph.nodes[n]["type"]
            inp_outp.append(type_ in ["input", "output"])
        else:
            assert subgraph.nodes[n]["label"] in NONTERMS
    return sum(inp_outp) == 1


def get_node_by_label(g, label, attr='label'):
    return next(filter(lambda n: g.nodes[n][attr] == label, g))


def nx_to_igraph(g):
    g = deepcopy(g)
    # for n in g:
    #     g.nodes[n]["type"] = list(LOOKUP).index(g.nodes[n]["type"])
    for e in g.edges:
        if "_igraph_index" in g.edges[e]:
            g.edges[e].pop("_igraph_index")
    for n in g:
        if "_igraph_index" in g.nodes[n]:
            g.nodes[n].pop("_igraph_index")
    ig = igraph.Graph.from_networkx(g)
    return ig


def copy_graph(g, nodes, copy_attrs=True):
    g_copy = g.__class__()
    if copy_attrs:
        for k in g.graph:
            g_copy.graph[k] = g.graph[k]
    for n in nodes:
        g_copy.add_node(n, **g.nodes[n])
    if isinstance(g, nx.DiGraph):
        for e in g.in_edges(nodes, data=True):
            if e[0] in nodes:
                g_copy.add_edge(e[0], e[1], **e[2])
        for e in g.out_edges(nodes, data=True):
            if e[1] in nodes:
                g_copy.add_edge(e[0], e[1], **e[2])
    else:
        for e in g.edges(nodes, data=True):
            if e[1] in nodes:
                g_copy.add_edge(e[0], e[1], **e[2])
    return g_copy
    

def copy_graph_mp(g, list_of_nodes, copy_attrs=True):
    res = []
    for nodes in list_of_nodes:
        res.append(copy_graph(g, nodes, copy_attrs=copy_attrs))
    return res


def boundary(g):
    # checks for any non-term connected nodes in g
    bad = False
    for a, b in g.edges:
        if g.nodes[a]["label"] in NONTERMS and g.nodes[b]["label"] in NONTERMS:
            bad = True
            break
    return bad


def count_num_terms(subgraph):
    ct = 0
    for n in subgraph:
        if subgraph.nodes[n]['label'] not in NONTERMS:
            ct += 1
    return ct


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
    ous = reduce(lambda x, y: x | y, [compat["out"] for compat in compats])
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


def random_search_single(g, should_add_edge):
    add_edge, gg, ism_graph, isms = should_add_edge
    should_add_edge = lambda i, j: add_edge(i, j, gg, ism_graph, isms)
    graph = MyGraph(g) # copy, for comps
    n = random.choice(list(graph))
    clique = [n]
    add_back = graph.comps[get_prefix(n)]
    graph.comps.pop(get_prefix(n))
    for c in tqdm(graph.comps, "iterating through comps"):
        for m in graph.comps[c]:
            violate = False
            for c in clique:
                if not should_add_edge(c, m):
                    violate = True
                    break
            if not violate:
                clique.append(m)      
                added = True
                break            
    graph.comps[get_prefix(n)] = add_back
    return clique


def random_search_max_clique(g, should_add_edge):
    max_clique = []
    # Check if the current process is the main process
    if mp.current_process().name == 'MainProcess':
        with mp.Pool(10) as p:
            cliques = p.starmap(random_search_single, tqdm([(g, should_add_edge) for i in range(NUM_RANDOM_SEARCH_TRIES)], desc="random_search_max_clique mp"))
    else:
        cliques = [random_search_single(g, should_add_edge) for i in tqdm(range(NUM_RANDOM_SEARCH_TRIES), desc="random_search_max_clique")]
    max_clique = max(cliques, key=len)
    return max_clique
        

def approximate_best_clique(graph):    
    logger = logging.getLogger('global_logger')
    start_time = time.time()
    logger.info("begin approximate_best_clique")
    if isinstance(graph, tuple):
        graph, should_add_edge = graph
        # a lightweight randomized search algorithm
        logger.info("begin random_search_max_clique")
        max_clique = random_search_max_clique(graph, should_add_edge)
    else:
        max_clique = []
        for ism_conn_subgraph in nx.connected_components(graph):
            conn_subgraph = copy_graph(graph, ism_conn_subgraph)
            print(
                f"approx max clique {len(conn_subgraph)} nodes {len(conn_subgraph.edges)} edges"
            )
            if len(conn_subgraph) > LIMIT_FOR_GREEDY:
                clique = greedy_max_clique(conn_subgraph)
            else:
                try:
                    clique = list(nx.approximation.max_clique(conn_subgraph))
                except:
                    clique = greedy_max_clique(conn_subgraph)
            if len(clique) > len(max_clique):
                max_clique = clique
            elif len(clique) == len(max_clique):
                lower_best, ous_best = reduce_to_bounds(
                    [graph.nodes[n] for n in max_clique]
                )
                lower, ous = reduce_to_bounds([graph.nodes[n] for n in clique])
                if len(lower) + len(ous) < len(lower_best) + len(ous_best):
                    print("better clique")
                    max_clique = clique
    logger.info(f"approximate_best_clique took {time.time()-start_time}")
    return max_clique


def non_isomorphic(all_subgraphs):
    subgraphs = []
    subgraph_ids = []
    for i, s in enumerate(all_subgraphs):
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
            subgraph_ids.append(i)
    return subgraph_ids, subgraphs


def find_embedding(subgraphs, graph, find_iso, edges=False):
    logger = logging.getLogger('global_logger')
    logger.info("begin find embedding")
    # find_iso: a custom function that constructs the compat graph
    best_i = -1
    best_ism = None
    best_clique = None
    max_len = 0
    # eliminate common subgraphs
    subgraph_ids, subgraphs = non_isomorphic(subgraphs)
    for i, subgraph in tqdm(zip(subgraph_ids, subgraphs), desc="looping over subgraphs"):
        # general concerns
        if len(subgraph) == 1:
            continue
        # if boundary(subgraph): # if so, will violate conformity
        #     continue
        # domain-specific concerns
        if DATASET in ["ckt", "enas", "bn"]:
            if check_input_xor_output(subgraph):
                continue
        ism_subgraph = find_iso(subgraph, graph)
        if isinstance(ism_subgraph, tuple):
            print(subgraph.nodes, ism_subgraph[0].nodes)
            if len(ism_subgraph[0]) == 0:
                continue
        else:
            print(subgraph.nodes, ism_subgraph.nodes)
            if len(ism_subgraph) == 0:
                continue        
        max_clique = approximate_best_clique(ism_subgraph)
        # max_clique = list(nx.find_cliques(ism_subgraph))
        # if edges:
        #     expr = len(max_clique) * len(subgraph.edges)
        # else:                
        expr = len(max_clique) * count_num_terms(subgraph)
        better = expr > max_len
        if better:
            best_i = i
            max_len = expr
            best_ism = ism_subgraph
            best_clique = max_clique
    # ism_subgraph: compatibility graph
    # best_ism: best subgraph
    # best cliques: best clique in ism_subgraph for best_ism
    # return best_ism, best_clique    
    if isinstance(best_ism, tuple):
        best_ism = best_ism[0]    
    if best_clique is not None:
        best_comps = list(set([best_ism.nodes[c]['ism'][0].split(':')[0] for c in best_clique]))
        logger.info("done find embedding")
        logger.info(f"subgraph {best_i} occurred {len(best_clique)} times across components {sorted(best_comps)}")
    return best_ism, best_clique

def subgraphs_isomorphism_mp(batch):
    res = []
    for conn, subgraph in batch:        
        gm = DiGraphMatcher(
            conn,
            subgraph,
            node_match=lambda d1, d2: d1.get("label", "#") == d2.get("label", "#"),
            edge_match=lambda d1, d2: d1.get("label", "#") == d2.get("label", "#"),
        )
        isms = list(gm.subgraph_isomorphisms_iter())
        res.append(isms)
    return res

def subgraphs_isomorphism(graph, subgraph):
    gm = DiGraphMatcher(
        graph,
        subgraph,
        node_match=lambda d1, d2: d1.get("label", "#") == d2.get("label", "#"),
        edge_match=lambda d1, d2: d1.get("label", "#") == d2.get("label", "#"),
    )
    isms = list(gm.subgraph_isomorphisms_iter())
    return isms


def fast_subgraph_isomorphism_mp(graph, conns, subgraph):
    # with mp.Manager() as manager:
        # graph_proxy = manager.dict(graph=graph)
    batch_size = SUBG_ISO_BATCH_SIZE
    num_batches = (len(conns)+batch_size-1)//batch_size
    args = []
    conn_batches = [conns[k*batch_size:(k+1)*batch_size] for k in range(num_batches)]
    with mp.Pool(NUM_PROCS) as p:
        copied = p.starmap(copy_graph_mp, tqdm([(graph, conn_batch) for conn_batch in conn_batches], "preparing args"))
    copied = sum(copied, [])
    # for conn in tqdm(conns, desc="preparing args"):
    #     args.append((copy_graph(graph, conn), subgraph))
    args = [(copy, subgraph) for copy in copied]
    num_batches = (len(args)+batch_size-1)//batch_size
    print(f"{num_batches} batches")                
    args_batch_list = [(args[k*batch_size:(k+1)*batch_size],) for k in range(num_batches)]        
    with mp.Pool(NUM_PROCS) as p:
        ans = p.starmap(
            subgraphs_isomorphism_mp,
            tqdm(
                args_batch_list,
                desc="subgraph isomorphism mp",
            ),
        )
        ans = sum(ans, [])    
    return ans


def fast_subgraph_isomorphism(graph, subgraph):
    logger = logging.getLogger('global_logger')
    assert nx.is_connected(nx.Graph(subgraph))
    conns = list(nx.connected_components(nx.Graph(graph)))
    # if conn is not None:
    #     graph = copy_graph(graph, conn)
    #     conns = [conn]
    if len(conns) == 1:
        return subgraphs_isomorphism(graph, subgraph)    
    if NUM_PROCS == 1:
        args = []
        for conn in tqdm(conns, desc="preparing args"):
            args.append((copy_graph(graph, conn), subgraph))        
        ans = [subgraphs_isomorphism(*arg) for arg in tqdm(args, desc="subgraph isomorphism")]
        ans = sum(ans, [])
    else:
        # t0 = time.time()
        # ans = fast_subgraph_isomorphism_mp(graph, conns, subgraph)
        # ans = sum(ans, [])
        # dt = time.time()-t0
        # logger.info(f"fast_subgraph_isomorphism_mp took {dt}")
        t0 = time.time()
        gm = DiGraphMatcher(
            graph,
            subgraph,
            node_match=lambda d1, d2: d1.get("label", "#") == d2.get("label", "#"),
            edge_match=lambda d1, d2: d1.get("label", "#") == d2.get("label", "#"),
        )
        dt = time.time()-t0
        logger.info(f"digraphmatcher took {dt}")
        ans = list(gm.subgraph_isomorphisms_iter())
        # if sorted(ans, key=lambda dic: json.dumps(dic, sort_keys=True)) != sorted(isms, key=lambda dic: json.dumps(dic, sort_keys=True)):
        #     breakpoint()
    return ans
