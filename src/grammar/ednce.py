from typing import List, Dict, Any
from src.draw.graph import draw_graph, draw_circuit
from src.config import *
import os
from networkx.algorithms.isomorphism import DiGraphMatcher
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


class EDNCEGrammar(NLCGrammar):    
    def __sample__(self):
        # find the initial rule
        rule_indices = list(range(len(self.rules)))        
        rules = self.search_rules('black')
        assert len(rules) == 1
        cur = rules[0].subgraph
        num_nts = len(self.search_nts(cur, NONTERMS))
        iters = 0
        while num_nts > 0:   
            if iters >= 100:
                return None       
            gray_nodes = self.search_nts(cur, ['gray'])
            node = random.choice(gray_nodes)
            ind = random.choice(rule_indices)
            rule = self.rules[ind]
            cur = rule(cur, node)
            num_nts = len(self.search_nts(cur, NONTERMS))
            iters += 1
        return cur
    

    def search_rules(self, nt):
        rules = []        
        for i, rule in enumerate(self.rules):
            if rule.nt == nt:
                rules.append(rule)    
        return rules
    

    @staticmethod
    def search_nts(cur, nts):
        res = list(filter(lambda x: cur.nodes[x]['label'] in nts, cur))
        return res


    def generate(self, num_samples=10):
        gen_dir = os.path.join(IMG_DIR, "generate/")
        os.makedirs(gen_dir, exist_ok=True)             
        res = []
        while len(res) < num_samples:
            print(len(res))
            sample = self.__sample__()
            if sample is None:
                continue
            bad = False
            for e in sample.edges:
                if sample.edges[e]['label'] in NONFINAL:
                    bad = True
            if bad:
                continue
            exist = False
            if not nx.is_connected(nx.Graph(sample)):
                continue
            for r in res:
                if nx.is_isomorphic(sample, r):
                    exist = True
                    break
            if exist:
                continue                    
            draw_graph(sample, os.path.join(gen_dir, f'graph_{len(res)}.png'), node_size=2000)
            res.append(sample)                    
        return res


class EDNCERule:
    def __init__(self, nt, subgraph, embedding):
        self.nt = nt
        self.subgraph = subgraph
        self.embedding = embedding


    def __call__(self, cur, node): 
        rhs = nx.DiGraph(self.subgraph)            
        start = find_next(cur)
        node_map = {}
        for n in rhs:
            node_map[n] = start                
            start = next(start)     
        inv_node_map = {v: k for k, v in node_map.items()}
        rhs = nx.relabel_nodes(rhs, node_map)
        cur = nx.union(cur, rhs)            
        cur_neis = neis(cur, [node], direction=['in','out'])
        for cur_nei in cur_neis:
            mu = cur.nodes[cur_nei]['label']
            for i, cur_node in enumerate(rhs):
                if cur.has_edge(node, cur_nei):
                    d = 'out'
                    p = cur[node][cur_nei]['label']
                else:
                    d = 'in'
                    p = cur[cur_nei][node]['label']
                if self.embedding is not None:
                    for emb in self.embedding:
                        mu_e, p_e, q_e, i_e, d_e, d__e = emb
                        if mu_e == mu and p_e == p and i_e == i and d_e == d:
                            if d__e == 'in':
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
        g = nx.DiGraph(self.subgraph)
        # g.graph['layout'] = 'spring_layout'
        if self.embedding is not None:
            for e in self.embedding:
                mu, p, q, x, d, d_ = e
                n = list(self.subgraph)[x]                
                if 'INVERSE_LOOKUP' in globals() and mu in INVERSE_LOOKUP:
                    type_ = INVERSE_LOOKUP[mu]
                    name = f"{x},{type_},{p}/{q},{d}/{d_}"
                    g.add_node(name, label=mu, type=type_, alpha=0.5)              
                else:
                    name = f"{x},{mu},{p}/{q},{d}/{d_}"
                    g.add_node(name, label=mu, alpha=0.5)              
                g.add_edge(n, name, style='dashed', 
                           reverse1=d_=='in', 
                           reverse2=d_=='in', 
                           label1=q, loc1=1/4, 
                           label2=p, loc2=3/4)
        draw_graph(g, rhs_path, scale=RULE_SCALE, node_size=RULE_NODE_SIZE, font_size=RULE_FONT_SIZE)
        self.draw_fig(lhs_path, rhs_path, path)

    
    def draw_fig(self, lhs_path, rhs_path, path):
        # Load PNG images
        lhs_image = Image.open(lhs_path)  # Replace 'path_to_lhs_image.png' with your file path
        rhs_image = Image.open(rhs_path)  # Replace 'path_to_rhs_image.png' with your file path

        # Convert images to arrays if necessary
        lhs_array = np.array(lhs_image)
        rhs_array = np.array(rhs_image)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Display images on the left and right axes
        ax1.imshow(lhs_array)
        ax1.set_title("LHS")
        ax1.axis('off')  # Turn off axis for a clean look

        ax3.imshow(rhs_array)
        ax3.set_title("RHS")
        ax3.axis('off')

        # Middle axis for arrow, set axis limits and turn off axis
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

        # Create an arrow that spans horizontally across the middle axis
        arrow = FancyArrowPatch((0, 0.5), (1, 0.5), mutation_scale=20,
                                arrowstyle='-|>', color='black',
                                lw=3,  # Set line width to make the arrow thicker
                                transform=ax2.transAxes)  # Ensure the arrow uses the axis coordinates
        ax2.add_patch(arrow)
        fig.savefig(path, bbox_inches='tight', dpi=300)


class EDNCENode:
    def __init__(self, id: int, attrs: Dict[str, Any]=None, children: List['NCENode']=None):
        self.id = id
        self.attrs = attrs if attrs is not None else {}
        self.children = children if children is not None else []
    

    def add_child(self, node):
        print("add edge", self.id, node.id)
        self.children.append(node)



class EDNCEModel(NLCModel):
    def generate(self, grammar):
        breakpoint()



def equiv_class(graph, nodes, out_ns):
    def label_edge(a, b):
        if b not in graph[a]:
            return '-'
        return graph[a][b]['label']
    lookup = {}
    for n in out_ns:
        key = []
        for node in nodes:
            e = label_edge(n, node) + '_' + label_edge(node, n)
            key.append(e)
        key = '__'.join(key)
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


def insets_and_outsets(graph, nodes, inset=True):
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
    out_ns = neis(graph, nodes, direction=['in', 'out'])       
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
    equiv, lookup = equiv_class(graph, nodes, out_ns)    
    poss_dirs = list(product(*[['in','out'] for _ in equiv]))
    poss_ps = list(product(*[NONFINAL for _ in equiv]))
    # naively, we need to enumerate, for every neighbor, the direction and edge label
    # however, we know inset & ouset whenever there is ambiguity, i.e. for two identical neighbors n1 and n2
    # with same (node label, edge label, direction) then they need to connect with the same subset of nodes
    # for each (y1, y2) with same node labels that do not connect with the same subset of nodes, they can never
    # look identical, so we connect y1-y2 in our incompatibility graph    
    poss = {}
    for a, dirs in enumerate(poss_dirs):
        for b, ps in enumerate(poss_ps):
            res_inset = set()
            res_outset = set()
            for i, x in enumerate(nodes):
                for j, y in enumerate(out_ns):
                    e = lookup[y]
                    d = dirs[e]
                    p = ps[e]
                    label_y = graph.nodes[y]['label']
                    mu = label_y
                    if graph.has_edge(x, y):
                        q = graph[x][y]['label']                                          
                        d_ = 'out'
                        res_inset.add((mu, p, q, i, d, d_))
                    if graph.has_edge(y, x):
                        q = graph[y][x]['label']
                        d_ = 'in'
                        res_inset.add((mu, p, q, i, d, d_))
                    if not graph.has_edge(x, y):
                        d_ = 'out'
                        for q in FINAL+NONFINAL:
                            res_outset.add((mu, p, q, i, d, d_))
                    if not graph.has_edge(y, x):
                        d_ = 'in'
                        for q in FINAL+NONFINAL:
                            res_outset.add((mu, p, q, i, d, d_))
            # (a, b) stores the index of realizations
            res_dirs = {y: dirs[lookup[y]] for y in out_ns}
            res_ps = {y: ps[lookup[y]] for y in out_ns}
            poss[(a, b)] = (res_inset, res_outset, res_dirs, res_ps)
    return poss



def touching(graph, ismA, ismB):
    nodesA = set(ismA)
    neisA = set(neis(graph, ismA))
    nodesB = set(ismB)
    neisB = set(neis(graph, ismB))
    touch = bool((nodesA | neisA) & nodesB) | bool(nodesA & (nodesB | neisB))    
    return touch



def find_iso(subgraph, graph):    
    gm = DiGraphMatcher(graph, subgraph, 
                      node_match=lambda d1, d2: d1.get('label','#')==d2.get('label','#'),
                      edge_match=lambda d1, d2: d1.get('label','#')==d2.get('label','#'))
    isms = list(gm.subgraph_isomorphisms_iter())  
    ism_graph = nx.Graph()    
    for i, ismA in enumerate(isms):
        poss = insets_and_outsets(graph, ismA)        
        for (a, b) in poss:
            inset, outset, dirs, ps = poss[(a, b)]
            if inset & outset:
                continue
            ism_graph.add_node(f"{i}_{a}_{b}", ins=inset, out=outset, ism=list(ismA), dirs=dirs, ps=ps)
    for i in ism_graph:
        for j in ism_graph:
            ismA = isms[int(i.split('_')[0])]
            ismB = isms[int(j.split('_')[0])]
            touch = touching(graph, ismA, ismB)
            inset_i = ism_graph.nodes[i]['ins']
            outset_i = ism_graph.nodes[i]['out']
            inset_j = ism_graph.nodes[j]['ins']
            outset_j = ism_graph.nodes[j]['out']
            overlap = (inset_i | inset_j) & (outset_i | outset_j)
            if not touch and not overlap:
                ism_graph.add_edge(i, j)
    return ism_graph
