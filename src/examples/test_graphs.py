from src.config import *
import networkx as nx
import numpy as np

# import pygsp as gsp
# from pygsp import graphs
import json
from src.draw.color import to_hex, CMAP
import importlib
import igraph
import argparse
import os

# from ogb.graphproppred import PygGraphPropPredDataset

if "api" in METHOD:
    grammar = importlib.import_module(f"src.algo.{GRAMMAR}")
else:
    grammar = importlib.import_module(f"src.algo.mining.{GRAMMAR}")
from src.draw.graph import draw_graph, MyGraph
from networkx.readwrite import json_graph
from tqdm import tqdm
from src.grammar.common import copy_graph
import rdkit.Chem as Chem

LABELS = ["r", "g", "b", "c"]


def create_random_graph(labels=LABELS):
    g = nx.random_regular_graph(3, 20, seed=SEED)
    labels = np.random.choice(labels, size=(len(g),))
    for n, label in zip(g, labels):
        g.nodes[n]["label"] = label
    return g


def create_test_graph(num):
    if num == 1:
        g = nx.Graph()
        labels = ["r", "g", "b", "c", "r", "r", "g", "b", "c", "b", "g", "r", "c", "b"]
        labels[5], labels[6] = labels[6], labels[5]
        labels[8], labels[7] = labels[7], labels[8]
        labels[12], labels[10] = labels[10], labels[12]
        labels[11], labels[10], labels[13], labels[12] = (
            labels[12],
            labels[11],
            labels[10],
            labels[13],
        )
        edges = [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            (2, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 6),
            (7, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 11),
        ]
        for i in range(14):
            g.add_node(i + 1, label=labels[i])
        for e in edges:
            g.add_edge(*e)
        g = nx.relabel_nodes(g, {n: n for n in g})
    else:
        pass
    return g


def create_minnesota():
    minn = graphs.Minnesota()
    g = minn.to_networkx()
    for n in g:
        g.nodes[n]["label"] = "r"
    # draw_graph(g, os.path.join(IMG_DIR, 'base.png'))
    return g


def load_cora():
    cwd = os.getcwd()
    g = nx.node_link_graph(json.load(open(f"{cwd}/data/nx/cora.json")))
    # labels = list(set([g.nodes[n]['label'] for n in g]))
    # assert len(labels) == len(LABELS)
    # lookup = dict(zip(labels, LABELS))

    conn = list(nx.connected_components(g))[0]
    print(len(conn), "nodes")
    g = copy_graph(g, conn)
    lookup = {}
    for n in list(sorted(g)):
        ego_g = nx.ego_graph(g, n, radius=RADIUS)
        val = nx.weisfeiler_lehman_graph_hash(ego_g, iterations=2)
        # val = g.nodes[n]['label']
        # nei_labels = [g.nodes[n]['label'] for n in ego_g]
        # nei_labels, counts = np.unique(nei_labels, return_counts=True)
        # nei_labels = [nei_label for (nei_label, count) in zip(nei_labels, counts) if count > 1]
        # labels = sorted(list(set(nei_labels)))
        # labels = ','.join(labels)
        # val = f"{val}_{labels}"
        if val not in lookup:
            lookup[val] = len(lookup)
        g.nodes[n]["label"] = to_hex(CMAP(lookup[val]))
    assert len(lookup) <= CMAP.N, f"{len(lookup)} exceeds {CMAP.N} colors"
    g = nx.relabel_nodes(g, {n: str(i + 1) for i, n in enumerate(list(g))})
    print(len(lookup), "labels")
    return g


def create_house_graph():
    def construct_house(grammar, K):
        rule1 = grammar.rules[0]
        rule2 = grammar.rules[1]
        rule3 = grammar.rules[2]
        g = nx.DiGraph()
        g.add_node("0", label="black")
        K = 2
        g = rule1(g, "0")
        for k in range(K):
            nt = grammar.search_nts(g, ["gray"])[0]
            g = rule2(g, nt)
        nt = grammar.search_nts(g, ["gray"])[0]
        g = rule3(g, nt)
        return g

    # g = nx.DiGraph()
    # edge_list = [(0,1),(0,2,'red'),
    #              (1,2),
    #              (2,3),(2,4,'red'),
    #              (3,4),
    #              (4,5),(4,6,'red'),
    #              (5,6),
    #              (7,6),(7,8,'red'),
    #              (8,4),(8,9,'red'),
    #              (9,2),(9,10,'red'),
    #              (10,0,'blue'),
    #              (11,10),(11,12,'red'),
    #              (12,13,'red'),
    #              (13,14,'red'),
    #              (14,7,'blue')]
    # for a, b, *e in edge_list:
    #     if len(e) == 1:
    #         e = e[0]
    #     else:
    #         e = 'green'
    #     g.add_edge(a, b, label=e)
    # for n in g:
    #     g.nodes[n]['label'] = 'cyan'
    # in the textbook, edge labels are {h,r,a,b,*} where {h,r} are non-final
    # in the textbook, node labels are {S,X,#} where {S,X} are non-terminal
    # we do the mapping {h,r,a,b,*} -> {black,gray,red,blue,green}
    # {S,X,#} -> {black,gray,cyan}
    # (0,3,h)
    grammar = grammar.EDNCEGrammar()
    subg1 = nx.DiGraph()
    subg1.add_node(0, label="cyan")
    subg1.add_node(1, label="cyan")
    subg1.add_node(2, label="cyan")
    subg1.add_node(3, label="gray")
    subg1.add_edge(0, 3, label="black")
    subg1.add_edge(1, 0, label="blue")
    subg1.add_edge(2, 1, label="green")
    subg1.add_edge(2, 3, label="gray")
    subg1.add_edge(3, 1, label="black")
    rule1 = grammar.EDNCERule("black", subg1, set())
    grammar.add_rule(rule1)
    subg2 = nx.DiGraph()
    subg2.add_node(0, label="cyan")
    subg2.add_node(1, label="cyan")
    subg2.add_node(2, label="cyan")
    subg2.add_node(3, label="cyan")
    subg2.add_node(4, label="gray")
    subg2.add_edge(0, 1, label="green")
    subg2.add_edge(1, 4, label="black")
    subg2.add_edge(2, 1, label="green")
    subg2.add_edge(3, 4, label="gray")
    subg2.add_edge(4, 2, label="black")
    emb2 = set()
    emb2.add(("cyan", "black", "green", 0, "in", "in"))
    emb2.add(("cyan", "black", "red", 1, "in", "in"))
    emb2.add(("cyan", "black", "red", 2, "out", "out"))
    emb2.add(("cyan", "gray", "red", 3, "in", "in"))
    subg3 = nx.DiGraph()
    subg3.add_node(0, label="cyan")
    subg3.add_node(1, label="cyan")
    subg3.add_node(2, label="cyan")
    subg3.add_node(3, label="cyan")
    subg3.add_edge(0, 1, label="green")
    subg3.add_edge(2, 1, label="green")
    subg3.add_edge(3, 2, label="blue")
    emb3 = emb2
    rule2 = grammar.EDNCERule("gray", subg2, emb2)
    rule3 = grammar.EDNCERule("gray", subg3, emb3)
    grammar.add_rule(rule2)
    grammar.add_rule(rule3)
    g = nx.DiGraph()
    for size in [4]:
        house = construct_house(grammar, size)
        g = nx.disjoint_union(g, house)
    g = nx.relabel_nodes(g, {n: str(i + 1) for i, n in enumerate(list(g))})
    return g


def union(gs, gs_dict={}):
    whole_g = gs[0].__class__()
    for g in gs:
        for n in g:
            whole_g.add_node(n, **g.nodes[n])
        for e in g.edges:
            whole_g.add_edge(e[0], e[1], **g.edges[e])
    for k in gs_dict:
        whole_g.graph[k] = gs_dict[k]
    return whole_g


def decode_ENAS_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    g.add_vertices(n+2)
    g.vs[0]['type'] = 0  # input node
    for i, node in enumerate(row):
        g.vs[i+1]['type'] = node[0] + 2  # assign 2, 3, ... to other types
        g.add_edge(i, i+1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i+1)
    g.vs[n+1]['type'] = 1  # output node
    g.add_edge(n, n+1)
    # note that the nodes 0, 1, ... n+1 are in a topological order
    return g, n+2


def decode_BN_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    g.add_vertices(n+2)
    g.vs[0]['type'] = 0  # input node
    for i, node in enumerate(row):
        cur_type = node[0] + 2
        g.vs[i+1]['type'] = cur_type
        if sum(node[1:]) == 0:  # if no connections from previous nodes, connect from input
            g.add_edge(0, i+1)
        else:
            for j, edge in enumerate(node[1:]):
                if edge == 1:
                    g.add_edge(j+1, i+1)
    g.vs[n+1]['type'] = 1  # output node
    end_vertices = [v.index for v in g.vs.select(_outdegree_eq=0) if v.index != n+1]
    for j in end_vertices:  # connect all loose-end vertices to the output node
        g.add_edge(j, n+1)
    return g, n+2


def load_ENAS_graphs(path, num_samples, n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000):
    graph_args = argparse.Namespace()
    # load ENAS format NNs to igraphs or tensors
    g_list = []
    max_n = 0  # maximum number of nodes
    with open(path, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                g, n = decode_ENAS_to_igraph(row)
            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)
            max_n = max(max_n, n)
            g_list.append((g, y)) 
            if len(g_list) == num_samples:
                break
    graph_args.num_vertex_type = n_types + 2  # original types + start/end types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1 # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    # return g_list[:int(ng*0.9)], g_list[int(ng*0.9):], graph_args
    return g_list, graph_args

    
def load_BN_graphs(path, num_samples, n_types=8, fmt='igraph', rand_seed=0, with_y=True):
    # load raw Bayesian network strings to igraphs or tensors
    graph_args = argparse.Namespace()
    g_list = []
    max_n = 0
    with open(path, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                g, n = decode_BN_to_igraph(row)
            elif fmt == 'string':
                g, n = decode_BN_to_tensor(row, n_types)
            max_n = max(max_n, n)
            assert(max_n == n)  # all BNs should have the same node number
            g_list.append((g, y)) 
            if len(g_list) == num_samples:
                break

    graph_args.num_class = 1  # how many classes of graphs
    graph_args.num_vertex_type = n_types + 2  # how many vertex types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1 # predefined end vertex type
    ng = len(g_list)
    print('# classes: %d' % graph_args.num_class)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    # return g_list[:int(ng*0.9)], g_list[int(ng*0.9):], graph_args    
    return g_list, graph_args


def load_enas(args):
    num_samples = args.num_data_samples
    data, graph_args = load_ENAS_graphs('D-VAE/data/final_structures6.txt', num_samples, n_types=6, fmt='igraph')
    whole_g = nx.DiGraph()
    gs = []
    for i in range(len(data)):
        g = data[i][0].to_networkx()
        for n in g:
            g.nodes[n]["type"] = list(LOOKUP)[g.nodes[n]["type"]]
            g.nodes[n]["label"] = LOOKUP[g.nodes[n]["type"]]
        for e in g.edges:
            g.edges[e]["label"] = "black"        
        node_map = {n: f"{i}:{n}" for n in g}
        g = nx.relabel_nodes(g, node_map)
        gs.append(g)
    whole_g = union(gs)
    whole_g = MyGraph(whole_g)
    return whole_g


def load_bn(args):
    num_samples = args.num_data_samples
    data, graph_args = load_BN_graphs('D-VAE/data/asia_200k.txt', num_samples, n_types=8, fmt='igraph')    
    whole_g = nx.DiGraph()
    gs = []
    gs_dict = {}
    for i in range(len(data)):
        g = data[i][0].to_networkx()
        y = data[i][1]        
        for n in g:
            g.nodes[n]["type"] = list(LOOKUP)[g.nodes[n]["type"]]
            g.nodes[n]["label"] = LOOKUP[g.nodes[n]["type"]]
        for e in g.edges:
            g.edges[e]["label"] = "black"        
        gs_dict[f"{i}:bic"] = y
        node_map = {n: f"{i}:{n}" for n in g}
        g = nx.relabel_nodes(g, node_map)
        gs.append(g)
    whole_g = union(gs, gs_dict)    
    whole_g = MyGraph(whole_g)
    return whole_g   


def load_ast(args):    
    dataset = PygGraphPropPredDataset(name="ogbg-code2")
    breakpoint()
    

def load_finance(args):
    breakpoint()


def load_ckt(args, load_all=False):
    """
    Load all ckts, and do union over all the graphs
    Combine graph-level attrs of individual graphs into a graph-level attr lookup

    The cktbench is comprised of pairs of ckts, 2*i and 2*i+1 have the same topology but different node-level attrs (device parameters)

    load_all: whether to load all ckts or just half
    """
    ambiguous_file = args.ambiguous_file
    cwd = os.getcwd()
    data_dir = f"{cwd}/data/nx/ckt/"
    if args.num_data_samples is not None:
        num_graphs = args.num_data_samples
    else:
        num_graphs = 0
        while True:
            expr = num_graphs if load_all else 2*num_graphs
            if os.path.exists(os.path.join(data_dir, f"{expr}.json")):
                num_graphs += 1
            else:
                break    
    whole_g = nx.DiGraph()
    # best_i = 0
    # max_size = 0
    # for i in range(9000):
    #     fpath = os.path.join(data_dir, f"{i}.json")
    #     data = json.load(open(fpath))
    #     g = json_graph.node_link_graph(data)
    #     if len(g) > max_size:
    #         max_size = len(g)
    #         best_i = i
    # print(best_i)
    gs = []
    gs_dict = {}
    if ambiguous_file is None or not os.path.exists(ambiguous_file):
        graph_no_iter = range(num_graphs)
    else:
        assert GRAMMAR == "ednce"
        graph_no_iter = json.load(open(ambiguous_file))['redo']
    i = 0
    for i in tqdm(graph_no_iter, desc="loading graphs"):
        expr = i if load_all else 2*i
        fpath = os.path.join(data_dir, f"{expr}.json")
        data = json.load(open(fpath))
        g = json_graph.node_link_graph(data)        
        for n in g:
            g.nodes[n]["type"] = list(LOOKUP)[g.nodes[n]["type"]]
            g.nodes[n]["label"] = LOOKUP[g.nodes[n]["type"]]
        for e in g.edges:
            g.edges[e]["label"] = "black"
        for attr in g.graph:
            if attr == "index":
                continue
            gs_dict[f"{i}:{attr}"] = g.graph[attr]
        node_map = {n: f"{i}:{n}" for n in g}
        g = nx.relabel_nodes(g, node_map)
        gs.append(g)
    whole_g = union(gs, gs_dict)
    whole_g = MyGraph(whole_g)
    return whole_g


def load_mols(args, num_samples=-1):
    filename = f"data/api_mol_hg/{args.mol_dataset}_smiles.txt"
    # debug ptc
    # smiles_list = ['COP(=O)(OC)OC=C(Cl)Cl']
    # return smiles_list
    # single chain extender
    # smiles_list = ['OCCNC(=O)NCCCCCCNC(=O)NCCO']
    # return smiles_list
    # top-10 ptc
    # debug_smiles_list = ['COc1c(N)cccc1', 'CC(C)CCCC(C)C1CCC2C3CC=C4CC(OC(=O)Cc5ccc(N(CCCl)CCCl)cc5)CCC4(C)C3CCC12C', 'Nc1nc(=O)n(C2OC(CO)C(O)C2O)cn1', 'CC(=O)C(N=Nc1ccc(-c2ccc(N=NC(C(C)=O)C(=O)Nc3ccccc3)c(Cl)c2)cc1Cl)C(=O)Nc1ccccc1', 'ClCC(Br)CBr', 'ClC1=C(Cl)C2(Cl)C3C4CC(C5OC45)C3C1(Cl)C2(Cl)Cl', 'ClC1=C(Cl)C2(Cl)C3C(Cl)C(Cl)CC3C1(Cl)C2(Cl)Cl', 'CNC(=O)CSP(=S)(OC)OC', 'ClC=C(Cl)Cl', 'COP(=O)(OC)OC=C(Cl)Cl']
    # top-10 hopv
    # debug_smiles_list = ['Cc1c(-c2cccs2)sc2c1sc(-c1ccc(-c3c4C(=O)N(C)C(=O)c4cs3)s1)c2C', 'COC(=O)c1cc2c(-c3cccs3)sc(-c3ccc(-c4cc5c(s4)c(-c4ccc(C)s4)c4ccsc4c5-c4ccc(C)s4)s3)c2s1', 'N#CC(=Cc1ccc(-c2ccc(-c3ccc(C=C(C#N)c4ccc([N+](=O)[O-])cc4)s3)c3c2C(=O)OC3=O)s1)c1ccc([N+](=O)[O-])cc1', 'CC1(C)CC(C=Cc2ccc(N(c3ccc(-n4c5ccccc5c5ccccc54)cc3)c3ccc(-n4c5ccccc5c5ccccc54)cc3)cc2)=CC(=C(C#N)C(=O)O)C1', 'CN1C(=O)C2=C(c3ccc(-c4cc5c(s4)-c4c(ccs4)C5(C)C)s3)N(C)C(=O)C2=C1c1cccs1', 'COc1ccc(N(c2ccc(OC)cc2)c2ccc(-c3ccc(-c4ccc(C=C(C#N)C(=O)O)s4)s3)cc2)cc1', 'Cc1cc(C)c(-c2c3cc(-c4ccc(-c5ccc(-c6cccs6)c6nsnc56)s4)sc3c(-c3c(C)cc(C)s3)c3c2scc3)s1', 'C#Cc1cc2c(s1)c1c(cc(C#Cc3c4nc(-c5ccccc5)c(-c5ccccc5)nc4cs3)s1)n2C', 'C[Si]1(C)c2c(scc2)-c2c1cc(-c1nc3c(nc(-c4cc5c(s4)-c4c(cc(-c6cc7c(ccs7)s6)s4)[Si]5(C)C)s3)s1)s2', 'Cc1ccc(-c2c3cc(-c4c(C)c5c(cc(-c6ccc(-c7csc8c7scc8C)c7nsnc67)s5)s4)sc3c(-c3ccc(C)s3)c3c2scc3)s1']
    # return debug_smiles_list
    # end debug    
    smiles_list = []
    with open(filename) as f:
        lines = f.readlines()
        # make sure no dups
        uniq_lines = []
        for l in lines:
            smiles = l.rstrip("\n")
            mol = Chem.MolFromSmiles(smiles)
            Chem.Kekulize(mol)
            smiles = Chem.MolToSmiles(mol)
            if smiles in uniq_lines:
                continue
            uniq_lines.append(smiles)
        print(f"{len(uniq_lines)}/{len(lines)} unique smiles")
        lines = uniq_lines
        if num_samples == -1:
            samples = range(len(lines))
        else:
            samples = np.random.choice(range(len(lines)), (num_samples), replace=False)
        print("samples", samples)
        for l in samples:
            line = lines[l]
            smiles_list.append(line.split(",")[0])
    
    return smiles_list


def write_file(samples, filename):
    with open(filename, "w+") as f:
        for s in samples:
            f.write(f"{s}\n")


def convert_and_write(samples, path):
    for i, sample in enumerate(samples):
        new_path = path.replace(".txt", f"_{i}.txt")
        # relabel types according to subckt
        label_relabel = {
            "yellow": 2,
            "lawngreen": 3,
            "cyan": 6,
            "lightblue": 7,
            "deepskyblue": 8,
            "dodgerblue": 9,
            "orchid": 0,
            "pink": 1,
        }
        ntype_lookup = {
            "yellow": 0,
            "lawngreen": 1,
            "cyan": 2,
            "lightblue": 3,
            "deepskyblue": 4,
            "dodgerblue": 5,
            "orchid": 8,
            "pink": 9,
        }
        label_relabel_inv = {v: k for (k, v) in label_relabel.items()}
        for n in sample:
            sample.nodes[n]["label"] = label_relabel[sample.nodes[n]["label"]]
        input = next(n for n in sample if sample.nodes[n]["label"] == 0)
        output = next(n for n in sample if sample.nodes[n]["label"] == 1)
        relabel_map = {}
        relabel_map[input] = 0
        relabel_map[output] = 1
        index = 2
        stage = 0
        for n in sample:
            if sample.nodes[n]["label"] in [6, 7, 8, 9]:
                relabel_map[n] = index
                index += 1
                stage += 1
        if stage not in [2, 3]:
            continue
        f = open(new_path, "w+")
        for n in sample:
            if n not in relabel_map:
                relabel_map[n] = index
                index += 1
        sample = nx.relabel_nodes(sample, relabel_map)
        f.write(f"{len(sample)} {len(sample)} {stage}\n")
        sample = copy_graph(sample, range(len(sample)))
        for n in sample:
            type_ = sample.nodes[n]["label"]
            preds = list(sample.predecessors(n))
            pred_str = " ".join(map(str, preds))
            num_preds = len(preds)
            if n == 0:
                f.write(f"0 0 0 0 0 1 8 0 1\n")
            elif n == 1:
                f.write(f"{type_} {n} {n} {num_preds} {pred_str} 1 9 0 1\n")
            else:
                labels = ["-1"] + ["11.11" for _ in preds] + ["-1"]
                label_str = " ".join(labels)
                f.write(f"{type_} {n} {n} {num_preds} {pred_str} {label_str}\n")
        f.write("\n")
        f.close()


def debug():
    g = nx.Graph()
    g.add_node(0, label="#e1d8e1ff")
    g.add_node(1, label="#97b6c7ff")
    g.add_node(2, label="#97b6c7ff")
    g.add_node(3, label="#e1d8e1ff")
    g.add_edge(0, 1)
    g.add_edge(2, 1)
    g.add_edge(2, 3)
    return g
