from .common import *
from typing import List, Dict, Any
from src.draw.graph import draw_graph
from src.config import IMG_DIR, NONTERMS
import os
from networkx.algorithms.isomorphism import GraphMatcher
from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import uuid
from copy import deepcopy
from itertools import product
import random
from .utils import *


class NLCGrammar:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def __sample__(self):
        # find the initial rule
        rule_indices = list(range(len(self.rules)))
        for i, rule in enumerate(self.rules):
            if rule.nt == "black":
                cur = nx.Graph(rule.subgraph)
                num_nts = sum([cur.nodes[n]["label"] == "gray" for n in cur])
                rule_indices.remove(i)
                break
        while num_nts > 0:
            gray_nodes = list(filter(lambda x: cur.nodes[x]["label"] == "gray", cur))
            node = random.choice(gray_nodes)
            ind = random.choice(rule_indices)
            rule = self.rules[ind]
            rhs = nx.Graph(rule.subgraph)
            start = find_next(cur)
            node_map = {}
            for n in rhs:
                node_map[n] = start
                start = next(start)
                num_nts += rhs.nodes[n]["label"] == "gray"
            num_nts -= 1
            rhs = nx.relabel_nodes(rhs, node_map)
            cur = nx.union(cur, rhs)
            cur_neis = neis(cur, [node])
            for cur_nei in cur_neis:
                out_label = cur.nodes[cur_nei]["label"]
                for cur_node in rhs:
                    in_label = cur.nodes[cur_node]["label"]
                    if (in_label, out_label) in rule.embedding:
                        cur.add_edge(cur_nei, cur_node)
            cur.remove_node(node)
        return cur

    def generate(self, num_samples=3):
        res = []
        while len(res) < num_samples:
            print(len(res))
            sample = self.__sample__()
            if not nx.is_connected(sample):
                continue
            exist = False
            for r in res:
                if nx.is_isomorphic(sample, r):
                    exist = True
                    break
            if exist:
                continue
            res.append(sample)
        return res


class NLCRule:
    def __init__(self, nt, subgraph, embedding):
        self.nt = nt
        self.subgraph = subgraph
        self.embedding = embedding

    def __call__(self, g, n):
        old_nos = list(g)
        index = old_nos.index(n)
        neis = [old_nos.index(m) for m in g[n]]
        no = len(g)
        g = nx.disjoint_union(g, self.subgraph)
        if self.embedding is not None:
            A = [list(g)[nei] for nei in neis]
            B = list(g)[no:]
            for a, b in product(A, B):
                c = (g.nodes[b]["label"], g.nodes[a]["label"])
                if c in self.embedding:
                    g.add_edge(a, b)
        g.remove_node(list(g)[index])
        g = nx.relabel_nodes(g, mapping={n: i for i, n in enumerate(list(g))})
        return g

    def visualize(self, path):
        g = nx.Graph()
        g.add_node(0, label=self.nt)
        lhs_path = os.path.join(IMG_DIR, f"{uuid.uuid4()}.png")
        draw_graph(g, lhs_path)
        rhs_path = os.path.join(IMG_DIR, f"{uuid.uuid4()}.png")
        draw_graph(self.subgraph, rhs_path)
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
        fig.savefig(path, bbox_inches="tight")


class NLCNode:
    def __init__(
        self, id: int, attrs: Dict[str, Any] = None, children: List["NLCNode"] = None
    ):
        self.id = id
        self.attrs = attrs if attrs is not None else {}
        self.children = children if children is not None else []

    def add_child(self, node):
        print("add edge", self.id, node.id)
        self.children.append(node)


class NLCModel:
    def __init__(self, graph):
        self.graph = graph
        # references to node-level attrs
        self.feat_lookup = {}
        for n in graph:
            for node, feat in zip(graph[n].attrs['nodes'], graph[n].attrs['feats']):
                self.feat_lookup[node] = feat

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
        res = [r for r in res if nx.is_connected(r)]
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
        g = nx.Graph()
        g.add_node(0, label="black")
        res = [g]
        res = self.__generate__(self.graph, grammar, res)
        # prune unique
        new_res = []
        for r_new in res:
            if not nx.is_connected(r_new):
                continue
            exist = False
            for r_old in new_res:
                if nx.is_isomorphic(
                    r_new, r_old, node_match=lambda d1, d2: d1["label"] == d2["label"]
                ):
                    exist = True
                    break
            if not exist:
                new_res.append(r_new)

        gen_dir = os.path.join(IMG_DIR, "generate/")
        os.makedirs(gen_dir, exist_ok=True)
        for i, g in enumerate(new_res):
            draw_graph(g, os.path.join(gen_dir, f"graph_{i}.png"))


def inoutset(graph, nodes, inset=True):
    out_ns = neis(graph, nodes)
    res = set()
    for x in nodes:
        for y in out_ns:
            if inset and graph.has_edge(x, y) or not inset and not graph.has_edge(x, y):
                label_x = graph.nodes[x]["label"]
                label_y = graph.nodes[y]["label"]
                res.add((label_x, label_y))
    return res


def find_iso(subgraph, graph):
    gm = GraphMatcher(
        graph, subgraph, node_match=lambda d1, d2: d1["label"] == d2["label"]
    )
    isms = list(gm.subgraph_isomorphisms_iter())
    insets = []
    outsets = []
    for i, ismA in enumerate(isms):
        insets.append(inoutset(graph, ismA))
        outsets.append(inoutset(graph, ismA, False))
    ism_graph = nx.Graph()
    for i, ism in enumerate(isms):
        if not (insets[i] & outsets[i]):
            ism_graph.add_node(i, ism=list(ism), ins=insets[i], out=outsets[i])
    for i in ism_graph:
        for j in ism_graph:
            ismA = isms[i]
            ismB = isms[j]
            nodesA = set(ismA)
            neisA = set(neis(graph, ismA))
            nodesB = set(ismB)
            neisB = set(neis(graph, ismB))
            touch = bool((nodesA | neisA) & nodesB) | bool(nodesA & (nodesB | neisB))
            overlap = (insets[i] | insets[j]) & (outsets[i] | outsets[j])
            if not touch and not overlap:
                ism_graph.add_edge(i, j)
    return ism_graph
