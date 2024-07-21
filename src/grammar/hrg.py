import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import VOCAB # C
import networkx as nx
import matplotlib.pyplot as plt

class Hyperedge:
    def __init__(self, nodes, label):
        self.nodes = nodes
        self.label = label

    
    @property
    def type(self):
        return VOCAB[self.label]


class HRG_rule:
    def __init__(self, symbol, rhs):
        assert symbol in VOCAB
        assert VOCAB[symbol] == rhs.type
        self.symbol = symbol
        self.rhs : HG = rhs


    def __call__(self, hg, index):
        if isinstance(index, str):
            assert index in VOCAB
            cands = [i for i in range(len(hg.E)) if hg.E[i].label == index]
            assert len(cands) == 1
            index = cands[0]
        assert hg.E[index].type == VOCAB[self.symbol]
        # Step 1: Remove from hg.E
        he = hg.E.pop(index)
        # Step 2: Add rhs
        # Step 3: Fuse rhs.ext with he.nodes
        node_map = {}
        for n in self.rhs.V:
            hg_n = hg.add_node()
            node_map[n.id] = hg_n       
        for i, e in enumerate(self.rhs.ext):
            node_map[e.id] = he.nodes[i]
        for e in self.rhs.E:
            mapped_n = [node_map[n] for n in e.nodes]
            hg.add_hyperedge(mapped_n, e.label)
        return hg


class HRG():
    """
        A hyperedge replacement grammar is a tuple (N, T, P, S) where:
            - N := nonterminals and subset of C
            - T := terminals and subset of C
            - P := set of productions over N
            - S in N is start symbol
    """
    def __init__(self, nonterms, terms, start):
        self.rules : list[HRG_rule] = []
        self.N = nonterms
        self.T = terms
        self.S = start
        for n in nonterms+terms+[start]:
            assert n in VOCAB
    
    def add_rule(self, rule : HRG_rule):
        """
            Each rule is an ordered pair (A, R) with:
                - A in N
                - R a hypergraph over C
                - type(A) = type(R)
                - A = lhs(p)
                - R = rhs(p)

        """
        self.rules.append(rule)


class Node:
    def __init__(self, id):
        self.id = id





class HG():
    """
        There is a global vocabulary of labels C, each with a type in natural numbers N.
        A hypergraph H over a vocabulary of labels C is defined as a tuple (V, E, att, lab, ext) where:
        - V := finite set of nodes
        - E: finite set of hyperedges
        - att: E -> V^* := mapping assigning a sequence of pairwise distinct attachment nodes to each e in E
        - lab: E -> C := mapping that labels each hyperedge s.t. type(lab(e)) = |att(e)|
        - ext in V^* := sequence of pairwise distinct external nodes
    """
    def __init__(self, num_nodes=0, num_exts=0):
        self.V : list[Node] = [] # finite set of nodes
        self.E : list[Hyperedge] = [] # finite set of hyperedges, contains att, lab
        self.ext : list[Node] = [] # pairwise distinct external nodes
        for _ in range(num_nodes):
            self.add_node()
        for _ in range(num_exts):
            self.add_ext()

    
    @staticmethod
    def edge_type(e):
        return len(e)
    
    
    @property
    def type(self):
        return len(self.ext)
    

    def add_node(self):
        n = f"n{len(self.V)}"
        self.V.append(Node(n)) 
        return n       

    
    def add_ext(self):
        # always number from 1 to len(E)
        n = f"e{len(self.ext)}"
        self.ext.append(Node(n))

    
    def add_hyperedge(self, nodes, type):
        self.E.append(Hyperedge(nodes, type))


    def visualize(self, path):
        """
        The way we visualize hg is by making hyperedges of type > 2 in a special color
        all attached nodes, with one color for each type
        """
        g = nx.Graph()
        v_nodes = []
        ext_nodes = []
        h_nodes = []
        for n in self.V:
            g.add_node(n.id)
            v_nodes.append(n.id)
        for e in self.ext:
            g.add_node(e.id)        
            ext_nodes.append(e.id)
        for ind in range(len(self.E)):
            g.add_node(f"h{ind}", label=self.E[ind].label)
            h_nodes.append(f"h{ind}")
            nodes = self.E[ind].nodes
            for n in nodes:
                g.add_edge(f"h{ind}", n)
        fig, ax = plt.subplots()
        pos = nx.spring_layout(g)
        nx.draw_networkx_nodes(g, pos, v_nodes, node_shape='o', 
                               edgecolors='black', 
                               node_color='white', ax=ax)
        nx.draw_networkx_nodes(g, pos, ext_nodes, node_shape='o', 
                               node_color='black',
                               ax=ax)

        nx.draw_networkx_nodes(g, pos, h_nodes, node_shape='s', 
                               edgecolors='black',
                               node_color='white', ax=ax)
        nx.draw_networkx_edges(g, pos, g.edges, ax=ax)
        nx.draw_networkx_labels(g, pos, {n:n for n in v_nodes}, ax=ax)
        nx.draw_networkx_labels(g, pos, {n:n for n in ext_nodes}, font_color='white', ax=ax)
        nx.draw_networkx_labels(g, pos, {n:g.nodes[n]['label'] for n in h_nodes}, ax=ax)
        fig.savefig(path)



if __name__ == "__main__":
    hrg = HRG(['S', 'A'], ['a', 'b', 'c'], 'S')
    rhs_1 = HG(4, 2)  
    rhs_1.add_hyperedge(["e0", "n0"], 'a')
    rhs_1.add_hyperedge(["n1", "n2"], 'b')
    rhs_1.add_hyperedge(["n2", "n3"], 'c')
    rhs_1.add_hyperedge(["n0", "n1", "n3", "e1"], 'A')
    rule_1 = HRG_rule('S', rhs_1)    
    rhs_2 = HG(2, 2)    
    rhs_2.add_hyperedge(["e0", "n0"], 'a')
    rhs_2.add_hyperedge(["n0", "n1"], 'b')
    rhs_2.add_hyperedge(["n1", "e1"], 'c')    
    rule_2 = HRG_rule('S', rhs_2)
    rhs_3 = HG(3, 4)
    rhs_3.add_hyperedge(["e0", "n0"], 'a')
    rhs_3.add_hyperedge(["n1", "e1"], 'b')
    rhs_3.add_hyperedge(["e2", "n2"], 'c')
    rhs_3.add_hyperedge(["n0", "n1", "n2", "e3"], 'A')
    rule_3 = HRG_rule('A', rhs_3)
    rhs_4 = HG(1, 4)
    rhs_4.add_hyperedge(["e0", "n0"], 'a')
    rhs_4.add_hyperedge(["n0", "e1"], 'b')
    rhs_4.add_hyperedge(["e2", "e3"], 'c')
    rule_4 = HRG_rule('A', rhs_4)    
    hrg.add_rule(rule_1)
    hrg.add_rule(rule_2)
    hrg.add_rule(rule_3)
    hrg.add_rule(rule_4)
    hg = HG(0, 2)    
    hg.add_hyperedge(["e0", "e1"], 'S')
    hg.visualize('/Users/msun415/Documents/GitHub/induction/data/api_ckt_ednce/test_0.png')
    hg = rule_1(hg, 0)
    hg.visualize('/Users/msun415/Documents/GitHub/induction/data/api_ckt_ednce/test_1.png')
    N = 2
    for i in range(N-1):
        hg = rule_3(hg, 'A')
        hg.visualize(f'/Users/msun415/Documents/GitHub/induction/data/api_ckt_ednce/test_{i+2}.png')
    hg = rule_4(hg, 'A')
    hg.visualize('/Users/msun415/Documents/GitHub/induction/data/api_ckt_ednce/test_final.png')
    
