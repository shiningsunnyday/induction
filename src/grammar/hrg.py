import os
import sys

wd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(wd)
import networkx as nx
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
from typing import Dict
from collections import defaultdict
import numpy as np
import pickle
from copy import deepcopy


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
            hg.add_hyperedge(mapped_n, e.label, **e.kwargs)
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
            self.counts[remap_idx[i]] += other.counts[i]


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


def derive(A, H, hrg):
    """
    Algorithm 2.7.4 (Derive(A,H)) for growing HRGs, extended to enumerate
    *all* terminal embeddings of a rule A => R into H.
    Returns True iff H ∈ L_A.
    """
    T = set(hrg.T)
    N = set(hrg.N)

    # H must be over terminals only (no nonterminals left)
    for he in H.E:
        if he.label in N:
            return False

    # ---------- helpers ----------
    def incident_edges(hg, node_id):
        r = hg.node_index_lookup[node_id]
        return np.argwhere(hg.adj_edges[r]).flatten().tolist()

    def build_edge_index(hg):
        idx = defaultdict(list)
        for i, e in enumerate(hg.E):
            idx[(e.label, frozenset(e.nodes))].append(i)
        return idx

    def rule_split_edges(R):
        term, nonterm = [], []
        for i, e in enumerate(R.E):
            (term if e.label in T else nonterm).append((i, e))
        return term, nonterm

    def nodes_R(R):
        ext_ids = [n.id for n in R.ext]
        int_ids = [n.id for n in R.V if n.id not in ext_ids]
        return ext_ids, int_ids

    def find_terminal_embeddings(R, H):
        """
        Enumerate all embeddings (m, used_terminal_edge_indices) such that:
          - m maps R's nodes to H's nodes with externals fixed positionally,
          - for every terminal edge of R, H has a matching edge (label+attachments),
          - used_terminal_edge_indices picks *distinct* concrete H edges for multiplicity.
        Yields tuples (m, used_indices_list) for every valid embedding.
        """
        ext_R, int_R = nodes_R(R)
        if len(ext_R) != len(H.ext):
            return  # no embeddings

        # Fix externals to externals (ids like e0,e1,... are aligned in this codebase)
        m = {eid: eid for eid in ext_R}

        term_R, _ = rule_split_edges(R)
        H_index = build_edge_index(H)
        H_internal_nodes = [n.id for n in H.V if not n.id.startswith("e")]
        int_list = list(int_R)

        def constraint_ok(partial):
            # any fully-instantiated terminal edge must exist in H
            for _, e in term_R:
                if all(v in partial for v in e.nodes):
                    key = (e.label, frozenset(partial[v] for v in e.nodes))
                    if key not in H_index or not H_index[key]:
                        return False
            return True

        def choose_terminals(k, taken, mapping):
            """
            Backtrack to choose a *distinct* H edge index for each terminal edge
            (handles multiplicities / parallel edges).
            """
            if k == len(term_R):
                yield list(taken)
                return
            _, e = term_R[k]
            key = (e.label, frozenset(mapping[v] for v in e.nodes))
            options = H_index.get(key, [])
            for ed_idx in options:
                if ed_idx in taken:
                    continue
                taken.add(ed_idx)
                yield from choose_terminals(k + 1, taken, mapping)
                taken.remove(ed_idx)

        def backtrack_nodes(i):
            if i == len(int_list):
                # All internal nodes assigned; now enumerate distinct terminal-edge choices
                yield from ((dict(m), used) for used in choose_terminals(0, set(), m))
                return
            var = int_list[i]
            for cand in H_internal_nodes:
                if cand in m.values():  # injective mapping
                    continue
                m[var] = cand
                if constraint_ok(m):
                    yield from backtrack_nodes(i + 1)
                del m[var]

        # Start enumeration (handles also the case of zero internal nodes)
        yield from backtrack_nodes(0)

    def bfs_component_from_attachments(hg, seed_nodes, allowed_edges):
        allowed = set(allowed_edges)
        edge_comp = set()
        q = list(seed_nodes)
        seen_nodes = set(seed_nodes)
        while q:
            u = q.pop()
            for ei in incident_edges(hg, u):
                if ei not in allowed or ei in edge_comp:
                    continue
                edge_comp.add(ei)
                for v in hg.E[ei].nodes:
                    if v not in seen_nodes:
                        seen_nodes.add(v)
                        q.append(v)
        return edge_comp

    def induce_subhypergraph(hg, edge_indices, attachment_nodes):
        k = len(attachment_nodes)
        Hsub = HG(k, list(range(k)))  # creates k nodes and marks them all as externals
        old2new = {attachment_nodes[j]: f"e{j}" for j in range(k)}

        def add_or_get(old_id):
            if old_id in old2new:
                return old2new[old_id]
            lbl = ""
            try:
                lbl = hg.V[hg.node_index_lookup[old_id]].label
            except Exception:
                pass
            new_id = Hsub.add_node(lbl)
            old2new[old_id] = new_id
            return new_id

        for ei in edge_indices:
            e = hg.E[ei]
            mapped_nodes = [add_or_get(x) for x in e.nodes]
            Hsub.add_hyperedge(mapped_nodes, e.label, **e.kwargs)
        return Hsub

    # ---------- try every rule/embedding ----------
    for r in hrg.rules:
        if r.symbol != A:
            continue
        R = r.rhs
        if R.type != H.type:
            continue

        for node_map, used_terminals in find_terminal_embeddings(R, H):
            term_R, nonterm_R = rule_split_edges(R)

            remaining_edges = set(range(len(H.E))) - set(used_terminals)

            Hi_list = []
            consumed = set()
            ok_partition = True

            for (_, e_nt) in nonterm_R:
                att_nodes = [node_map[u] for u in e_nt.nodes]
                comp_edges = bfs_component_from_attachments(
                    H, att_nodes, remaining_edges - consumed
                )
                Hi = induce_subhypergraph(H, comp_edges, att_nodes)
                Hi_list.append((e_nt.label, Hi))
                consumed |= comp_edges

            # All remaining edges must be accounted for by the union of Hi
            if consumed != remaining_edges:
                continue

            # Base case: no nonterminals on RHS
            if not nonterm_R:
                return True

            # Recurse
            all_ok = True
            for nt_label, Hi in Hi_list:
                if not derive(nt_label, Hi, hrg):
                    all_ok = False
                    break
            if all_ok:
                return True

    return False

# if __name__ == "__main__":
#     folder = "api_test_hg"
#     vocab = {"S": 2, "A": 4, "a": None, "b": None, "c": None}
#     hrg = HRG(["S", "A"], ["a", "b", "c"], "S", vocab)
#     rhs_1 = HG(4 + 2, range(4, 4 + 2))
#     rhs_1.add_hyperedge(["e0", "n0"], "a")
#     rhs_1.add_hyperedge(["n1", "n2"], "b")
#     rhs_1.add_hyperedge(["n2", "n3"], "c")
#     rhs_1.add_hyperedge(["n0", "n1", "n3", "e1"], "A")
#     rule_1 = HRG_rule("S", rhs_1, vocab)
#     rhs_1.visualize(f"{wd}/data/{folder}/rhs_rule_1.png")
#     rhs_2 = HG(2 + 2, range(2, 2 + 2))
#     rhs_2.add_hyperedge(["e0", "n0"], "a")
#     rhs_2.add_hyperedge(["n0", "n1"], "b")
#     rhs_2.add_hyperedge(["n1", "e1"], "c")
#     rule_2 = HRG_rule("S", rhs_2, vocab)
#     rhs_2.visualize(f"{wd}/data/{folder}/rhs_rule_2.png")
#     rhs_3 = HG(3 + 4, range(3, 3 + 4))
#     rhs_3.add_hyperedge(["e0", "n0"], "a")
#     rhs_3.add_hyperedge(["n1", "e1"], "b")
#     rhs_3.add_hyperedge(["e2", "n2"], "c")
#     rhs_3.add_hyperedge(["n0", "n1", "n2", "e3"], "A")
#     rule_3 = HRG_rule("A", rhs_3, vocab)
#     rhs_3.visualize(f"{wd}/data/{folder}/rhs_rule_3.png")
#     rhs_4 = HG(1 + 4, range(1, 1 + 4))
#     rhs_4.add_hyperedge(["e0", "n0"], "a")
#     rhs_4.add_hyperedge(["n0", "e1"], "b")
#     rhs_4.add_hyperedge(["e2", "e3"], "c")
#     rule_4 = HRG_rule("A", rhs_4, vocab)
#     rhs_4.visualize(f"{wd}/data/{folder}/rhs_rule_4.png")
#     hrg.add_rule(rule_1)
#     hrg.add_rule(rule_2)
#     hrg.add_rule(rule_3)
#     hrg.add_rule(rule_4)
#     hg = HG(0 + 2, range(0, 0 + 2))
#     hg.add_hyperedge(["e0", "e1"], "S")
#     hg.visualize(f"{wd}/data/{folder}/test_0.png")
#     hg = rule_1(hg, ("S", 0))
#     hg.visualize(f"{wd}/data/{folder}/test_1.png")

#     N = 40 # n+1
#     for i in range(N - 1):
#         hg = rule_3(hg, ("A", 0))
#         hg.visualize(f"{wd}/data/{folder}/test_{i+2}.png")
#     hg = rule_4(hg, ("A", 0))

#     print(derive("S", hg, hrg))

#     hg.visualize(f"{wd}/data/{folder}/test_final.png")    
#     n = 1
#     test_hg = HG(n + 2, range(n, n + 2))
#     test_hg.add_hyperedge(["e0", "n0"], "a")
#     test_hg.add_hyperedge(["n0", "e1"], "b")
#     test_hg.visualize(f"{wd}/data/{folder}/test_bad.png")

#     print(derive("S", test_hg, hrg))

#     n = 2
#     test_hg = HG(n + 2, range(n, n + 2))
#     test_hg.add_hyperedge(["e0", "n0"], "a")
#     test_hg.add_hyperedge(["n0", "n1"], "b")
#     test_hg.add_hyperedge(["n1", "e1"], "c")
#     test_hg.visualize(f"{wd}/data/{folder}/test_good.png")    

#     print(derive("S", test_hg, hrg))



if __name__ == "__main__":
    folder = "api_test_hg"
    vocab = {"S": 0, # []
             "A": 1, # []
             "a": None, "b": None, "c": None}
    # node_vocab = {"-", "=", "@", "#"}
    hrg = HRG(["S", "A"], ["a", "b", "c"], "S", vocab)
    rhs_1 = HG(2 + 0, range(2, 2 + 0))
    rhs_1.add_hyperedge(["n0", "n1"], "a")
    rhs_1.add_hyperedge(["n1"], "A")
    rule_1 = HRG_rule("S", rhs_1, vocab)
    rhs_1.visualize(f"{wd}/data/{folder}/rhs_rule_1.png")
    rhs_2 = HG(3 + 1, range(3, 3 + 1))
    rhs_2.add_hyperedge(["e0", "n0"], "b")
    rhs_2.add_hyperedge(["n0", "n1"], "c")
    rhs_2.add_hyperedge(["n1", "n2"], "a")
    rule_2 = HRG_rule("A", rhs_2, vocab)
    rhs_2.visualize(f"{wd}/data/{folder}/rhs_rule_2.png")
    hrg.add_rule(rule_1)
    hrg.add_rule(rule_2)

    test_hg = HG(5 + 0, range(5, 5 + 0))
    test_hg.add_hyperedge(["n0", "n1"], "a")
    test_hg.add_hyperedge(["n1", "n2"], "c")
    test_hg.add_hyperedge(["n2", "n3"], "b")
    test_hg.add_hyperedge(["n4", "n3"], "a")
    test_hg.visualize(f"{wd}/data/{folder}/test_good.png")

    print(derive("S", test_hg, hrg))
