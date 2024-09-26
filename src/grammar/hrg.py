import os
import sys

wd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(wd)
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from collections import defaultdict
import numpy as np


class Hyperedge:
    def __init__(self, nodes, label, **kwargs):
        self.nodes = nodes
        self.label = label
        self.kwargs = kwargs

    def get_type(self, vocab):
        return vocab[self.label]


class HRG_rule:
    def __init__(self, symbol, rhs, vocab):
        self.vocab = vocab
        assert symbol in self.vocab
        assert self.vocab[symbol] == rhs.type
        self.symbol = symbol
        self.rhs: HG = rhs

    def __call__(self, hg, edge, match_info=None):
        if isinstance(edge, tuple):
            index, cand_index = edge
            assert index in self.vocab
            cands = hg.get_cands(index)
            assert cand_index < len(cands)
            index = cands[cand_index]
        else:
            assert isinstance(edge, int)
            assert edge < len(hg.E)
            index = edge
        remove_edges = []
        assert hg.E[index].get_type(self.vocab) == self.vocab[self.symbol]
        # Step 1: Remove from hg.E
        old_e = {}
        for ei in range(len(hg.E)):
            old_e[f"h{ei}"] = hg.E[ei].nodes
        he = hg.E[index]
        remove_edges.append(index)
        # # also remove any TERMINAL edges with the same nodes as he
        for i in range(len(hg.E) - 1, -1, -1):
            if set(hg.E[i].nodes) == set(he.nodes) and hg.E[i].label[0] != "(":
                remove_edges.append(i)
        # print(f"removing {he.nodes}")
        # Step 2: Add rhs
        # Step 3: Fuse rhs.ext with he.nodes
        node_map = {}
        if match_info is not None:
            inv_match_info = dict(zip(match_info.values(), match_info.keys()))
        for n in self.rhs.V:
            if match_info is None:
                if n.id[0] == "e":
                    continue
            if match_info is not None:
                if n.id in inv_match_info:
                    hg_n = inv_match_info[n.id]
                    node_map[n.id] = hg_n
                    continue
            hg_n = hg.add_node(n.label, **n.kwargs)
            node_map[n.id] = hg_n
        for i, e in enumerate(self.rhs.ext):
            if match_info is not None:
                node_map[e.id] = inv_match_info[e.id]
            else:
                node_map[e.id] = he.nodes[i]
        # for each node in rhs.ext, sample its adj atoms from he
        # for e, n in zip(he.nodes, self.rhs.ext):
        #     atom_cts = self.rhs.adj_atoms(n.id)
        #     e_atoms = hg.adj_atoms(e, count=False)
        #     for a, c in atom_cts.items():
        #         to_del += list(np.random.choice(e_atoms[a], c, replace=False))
        # breakpoint()
        # if match_info is not None:
        #     remove_edges = sorted([int(n[1:]) for n in match_info if n[0]=='h'], reverse=True)
        #     for i in remove_edges:
        #         if i > index:
        #             i = i-1
        #         hg.E[i].nodes # add this back!
        #         hg.remove_hyperedge(i)
        for ei, e in enumerate(self.rhs.E):
            mapped_n = [node_map[n] for n in e.nodes]
            if match_info is not None:
                if f"h{ei}" in inv_match_info:
                    old_e_nodes = old_e[inv_match_info[f"h{ei}"]]
                    if set(mapped_n).issubset(old_e_nodes):  # don't dup
                        continue
                    elif set(old_e_nodes).issubset(mapped_n):  # remove old edge
                        remove_edges.append(int(inv_match_info[f"h{ei}"][1:]))
                    else:
                        # update the edge
                        new_e_nodes = set(mapped_n)|set(old_e_nodes)
                        remove_edges.append(int(inv_match_info[f"h{ei}"][1:]))
                        hg.add_hyperedge(list(new_e_nodes), e.label, **e.kwargs)
                        continue
            # if set(mapped_n) & set(he.nodes) == set(mapped_n):
            #     continue
            # old_edges = [i for i in range(len(hg.E)) if set(hg.E[i].nodes) == set(mapped_n)]
            # if len(old_edges) > 1:
            #     breakpoint()
            # for i in sorted(old_edges, reverse=True):
            #     hg.remove_hyperedge(i)
            # any edges connecting only the anchors should be ignored
            hg.add_hyperedge(mapped_n, e.label, **e.kwargs)
            # find any counterpart to remove
        # remove edges consisting of only anchors
        for ei in sorted(list(set(remove_edges)), reverse=True):
            hg.remove_hyperedge(ei)
        return hg


class HRG:
    """
    A hyperedge replacement grammar is a tuple (N, T, P, S) where:
        - N := nonterminals and subset of C
        - T := terminals and subset of C
        - P := set of productions over N
        - S in N is start symbol
    """

    def __init__(self, nonterms, terms, start, vocab):
        self.rules: list[HRG_rule] = []
        self.N = nonterms
        self.T = terms
        self.S = start
        self.vocab = vocab
        for n in nonterms + terms + [start]:
            assert n in vocab
        self.counts = []

    def add_rule(self, rule: HRG_rule):
        """
        Each rule is an ordered pair (A, R) with:
            - A in N
            - R a hypergraph over C
            - type(A) = type(R)
            - A = lhs(p)
            - R = rhs(p)

        """
        self.counts.append(0)
        self.rules.append(rule)

    def set_counts(self, counts):
        assert len(counts) == len(self.rules)
        self.counts = counts

    def combine_counts(self, other, remap_idx):
        for i in range(len(other.counts)):
            self.counts[remap_idx[i]] = other.counts[i]


class Node:
    def __init__(self, id, label, **kwargs):
        assert isinstance(id, str)
        self.id: str = id
        self.label = label
        self.kwargs: Dict = kwargs


class HG:
    """
    There is a global vocabulary of labels C, each with a type in natural numbers N.
    A hypergraph H over a vocabulary of labels C is defined as a tuple (V, E, att, lab, ext) where:
    - V := finite set of nodes
    - E: finite set of hyperedges
    - att: E -> V^* := mapping assigning a sequence of pairwise distinct attachment nodes to each e in E
    - lab: E -> C := mapping that labels each hyperedge s.t. type(lab(e)) = |att(e)|
    - ext in V^* := sequence of pairwise distinct external nodes
    """

    def __init__(self, num_nodes=0, exts=[], node_labels=None):
        """
        num_nodes : number of nodes, numbered 0...num_nodes-1
        exts : list of external nodes amongst 0...num_nodes-1
        """
        self.V: list[Node] = []  # finite set of nodes
        self.E: list[Hyperedge] = []  # finite set of hyperedges, contains att, lab
        self.ext: list[Node] = []  # pairwise distinct external nodes
        self.node_index_lookup = {}
        self.adj_edges = np.zeros((0, 0))
        self.mapping = {}  # mapping to preserve node ids before being marked as ext
        for i in range(num_nodes):
            if node_labels is None:
                nl = {}
                self.add_node("", **nl)
            else:
                nl = node_labels[i]
                assert "label" in nl
                self.add_node(**nl)

        for i in range(len(exts)):
            self.add_ext(exts[i])

    def get_cands(self, index):
        cands = [i for i in range(len(self.E)) if self.E[i].label == index]
        return cands

    @staticmethod
    def edge_type(e):
        return len(e)

    @property
    def type(self):
        return len(self.ext)

    def add_node(self, label, **kwargs):
        # print(f"add hyperedge, adj_edges shape {self.adj_edges.shape}, len(self.E) {len(self.E)}")
        id = max([int(v.id[1:]) for v in self.V]) + 1 if self.V else 0
        n = f"n{id}"
        self.V.append(Node(n, label, **kwargs))
        self.node_index_lookup[n] = len(self.V) - 1
        new_adj_edges = np.pad(self.adj_edges, [(0, 1), (0, 0)])
        assert new_adj_edges.shape[0] == len(self.V)
        assert self.adj_edges.shape[1] == new_adj_edges.shape[1]
        self.adj_edges = new_adj_edges
        # print(f"add hyperedge, adj_edges shape {self.adj_edges.shape}, len(self.E) {len(self.E)}")
        return n

    def adj_atoms(self, node, count=True):
        adj = self.adj_edges[self.node_index_lookup[node]]
        adj = np.argwhere(adj).flatten()
        if count:
            labels = [self.E[i].label for i in adj]
            ctr = {}
            for l in labels:
                ctr[l] = ctr.get(l, 0) + 1
        else:
            ctr = {}
            for i in adj:
                l = self.E[i].label
                ctr[l] = ctr.get(l, []) + [i]
        return ctr

    def add_ext(self, n):
        # has to be one of self.V
        # always number from e0, e1, ...
        node = self.V[n]
        new_name = f"e{len(self.ext)}"
        self.node_index_lookup[new_name] = self.node_index_lookup[node.id]
        self.node_index_lookup.pop(node.id)
        for e in self.E:
            e.nodes = [new_name if n == node.id else n for n in e.nodes]
        self.mapping[node.id] = new_name
        node.id = new_name
        self.ext.append(node)

    def add_hyperedge(self, nodes, label, **kwargs):
        # print(f"add hyperedge, adj_edges shape {self.adj_edges.shape}, len(self.E) {len(self.E)}")
        nodes = [self.mapping[n] if n in self.mapping else n for n in nodes]
        edge = Hyperedge(nodes, label, **kwargs)
        # print(self.adj_edges)
        new_adj_edges = np.pad(self.adj_edges, [(0, 0), (0, 1)])
        for node in nodes:
            new_adj_edges[self.node_index_lookup[node], -1] = 1
        self.E.append(edge)
        assert new_adj_edges.shape[1] == len(self.E)
        self.adj_edges = new_adj_edges
        # import glob
        # import re
        # d = max([int(re.match('/home/msun415/test_(\d+).png', a).groups()[0]) if re.match('/home/msun415/test_(\d+).png', a) is not None else -1 for a in glob.glob(f'/home/msun415/test_*.png')])
        # self.visualize(f'/home/msun415/test_{d+1}.png')
        # print(f"add hyperedge, adj_edges shape {self.adj_edges.shape}, len(self.E) {len(self.E)}")

    def remove_hyperedge(self, index):
        self.adj_edges = np.delete(self.adj_edges, index, -1)
        res = self.E.pop(index)
        return res

    def visualize(self, path, return_g=False):
        """
        The way we visualize hg is by making hyperedges of type > 2 in a special color
        all attached nodes, with one color for each type
        """
        g = nx.Graph()
        v_nodes = []
        ext_nodes = []
        h_nodes = []
        for n in self.V:
            g.add_node(n.id, label=n.label)
            v_nodes.append(n.id)
        for e in self.ext:
            g.add_node(e.id, label=e.label)
            ext_nodes.append(e.id)
        for ind in range(len(self.E)):
            g.add_node(f"h{ind}", label=self.E[ind].label)
            h_nodes.append(f"h{ind}")
            nodes = self.E[ind].nodes
            for n in nodes:
                g.add_edge(f"h{ind}", n)
        if return_g:
            return g
        fig, ax = plt.subplots()
        pos = nx.kamada_kawai_layout(g)
        nx.draw_networkx_nodes(
            g,
            pos,
            v_nodes,
            node_shape="o",
            edgecolors="black",
            node_color="white",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            g, pos, ext_nodes, node_shape="o", node_color="black", ax=ax
        )

        nx.draw_networkx_nodes(
            g,
            pos,
            h_nodes,
            node_shape="s",
            edgecolors="black",
            node_color="white",
            ax=ax,
        )
        nx.draw_networkx_edges(g, pos, g.edges, ax=ax)
        nx.draw_networkx_labels(
            g, pos, {n: n + " " + str(g.nodes[n]["label"]) for n in v_nodes}, ax=ax
        )
        nx.draw_networkx_labels(
            g, pos, {n: n for n in ext_nodes}, font_color="white", ax=ax
        )
        nx.draw_networkx_labels(
            g, pos, {n: n + " " + g.nodes[n]["label"] for n in h_nodes}, ax=ax
        )
        fig.savefig(path)
        plt.close(fig)
        # print(os.path.abspath(path))


if __name__ == "__main__":
    vocab = {"S": 2, "A": 4, "a": None, "b": None, "c": None}
    hrg = HRG(["S", "A"], ["a", "b", "c"], "S", vocab)
    rhs_1 = HG(4 + 2, range(4, 4 + 2))
    rhs_1.add_hyperedge(["e0", "n0"], "a")
    rhs_1.add_hyperedge(["n1", "n2"], "b")
    rhs_1.add_hyperedge(["n2", "n3"], "c")
    rhs_1.add_hyperedge(["n0", "n1", "n3", "e1"], "A")
    rule_1 = HRG_rule("S", rhs_1, vocab)
    rhs_1.visualize(f"{wd}/data/api_mol_hg/rhs_rule_1.png")
    rhs_2 = HG(2 + 2, range(2, 2 + 2))
    rhs_2.add_hyperedge(["e0", "n0"], "a")
    rhs_2.add_hyperedge(["n0", "n1"], "b")
    rhs_2.add_hyperedge(["n1", "e1"], "c")
    rule_2 = HRG_rule("S", rhs_2, vocab)
    rhs_2.visualize(f"{wd}/data/api_mol_hg/rhs_rule_2.png")
    rhs_3 = HG(3 + 4, range(3, 3 + 4))
    rhs_3.add_hyperedge(["e0", "n0"], "a")
    rhs_3.add_hyperedge(["n1", "e1"], "b")
    rhs_3.add_hyperedge(["e2", "n2"], "c")
    rhs_3.add_hyperedge(["n0", "n1", "n2", "e3"], "A")
    rule_3 = HRG_rule("A", rhs_3, vocab)
    rhs_3.visualize(f"{wd}/data/api_mol_hg/rhs_rule_3.png")
    rhs_4 = HG(1 + 4, range(1, 1 + 4))
    rhs_4.add_hyperedge(["e0", "n0"], "a")
    rhs_4.add_hyperedge(["n0", "e1"], "b")
    rhs_4.add_hyperedge(["e2", "e3"], "c")
    rule_4 = HRG_rule("A", rhs_4, vocab)
    rhs_4.visualize(f"{wd}/data/api_mol_hg/rhs_rule_4.png")
    hrg.add_rule(rule_1)
    hrg.add_rule(rule_2)
    hrg.add_rule(rule_3)
    hrg.add_rule(rule_4)
    hg = HG(0 + 2, range(0, 0 + 2))
    hg.add_hyperedge(["e0", "e1"], "S")
    hg.visualize(f"{wd}/data/api_mol_hg/test_0.png")
    hg = rule_1(hg, ("S", 0))
    hg.visualize(f"{wd}/data/api_mol_hg/test_1.png")
    N = 2
    for i in range(N - 1):
        hg = rule_3(hg, ("A", 0))
        hg.visualize(f"{wd}/data/api_mol_hg/test_{i+2}.png")
    hg = rule_4(hg, ("A", 0))
    hg.visualize(f"{wd}/data/api_mol_hg/test_final.png")
