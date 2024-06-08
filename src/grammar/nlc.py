from src.draw.graph import draw_graph
from src.config import IMG_DIR
import os
from networkx.algorithms.isomorphism import GraphMatcher
from PIL import Image
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import uuid

class NLCGrammar:
    def __init__(self):
        self.rules = []
    
    def add_rule(self, rule):
        self.rules.append(rule)

class NLCRule:
    def __init__(self, nt, subgraph, embedding):
        self.nt = nt
        self.subgraph = subgraph
        self.embedding = embedding

    
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
        fig.savefig(path, bbox_inches='tight')


def get_groups(content):
    groups = []
    for l in content.split():
        l_str = l.replace(' ','').split(',')
        try:
            l_arr = list(map(int, l_str))
        except:
            continue
        groups.append(l_arr)
    return groups


def neis(graph, nodes):
    ns = sum([list(graph[n]) for n in nodes], [])
    out_neis = list(set([n for n in ns if n not in nodes]))
    return out_neis


def inoutset(graph, nodes, inset=True):
    out_ns = neis(graph, nodes)    
    res = set()
    for x in nodes:
        for y in out_ns:
            if inset and graph.has_edge(x, y) or not inset and not graph.has_edge(x, y):
                label_x = graph.nodes[x]['label']
                label_y = graph.nodes[y]['label']
                res.add((label_x, label_y))
    return res


def find_iso(subgraph, graph):
    gm = GraphMatcher(graph, subgraph, node_match=lambda d1, d2: d1['label']==d2['label'])
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



def find_embedding(subgraphs, graph):
    best_ism = None
    best_clique = None
    max_len = 0
    for subgraph in subgraphs:
        ism_subgraph = find_iso(subgraph, graph)
        if len(ism_subgraph) == 0:
            continue
        print(subgraph.nodes, ism_subgraph.nodes)
        max_cliques = list(nx.find_cliques(ism_subgraph))
        better = False
        if len(subgraph) > 1:
            better = len(max_cliques[0]) > max_len       
        if better:
            max_len = len(max_cliques[0])
            best_ism = ism_subgraph    
            best_clique = max_cliques[0]
    # ism_subgraph: compatibility graph
    # best_ism: best subgraph
    # best cliques: best clique in ism_subgraph for best_ism
    # return best_ism, best_clique
    return best_ism, best_clique
