import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
from matplotlib.colors import to_rgba
from src.config import *
from src.draw.utils import hierarchy_pos
from src.grammar.common import *
import matplotlib.colors as mcolors
import igraph

# import pygraphviz as pgv


def draw_custom_arrows(
    ax,
    pos,
    edge,
    color_1="blue",
    color_2="green",
    arrow_1_pos=0.25,
    arrow_2_pos=0.75,
    rev1=False,
    rev2=False,
    arrowstyle="-|>",
    arrowsize=10,
):
    start, end = pos[edge[0]], pos[edge[1]]
    line = np.array([start, end])
    mid_point = (start + end) / 2
    first_half = (start + mid_point) / 2
    second_half = (mid_point + end) / 2
    # Draw first half in color_1
    ax.plot(
        [start[0], mid_point[0]], [start[1], mid_point[1]], color=color_1, linewidth=1.0
    )
    # Draw second half in color_2
    ax.plot(
        [mid_point[0], end[0]], [mid_point[1], end[1]], color=color_2, linewidth=1.0
    )

    # Draw the first arrow at arrow_1_pos
    arrow_pos_1 = (1 - arrow_1_pos) * start + arrow_1_pos * end
    if rev1:
        ax.annotate(
            "",
            xy=arrow_pos_1,
            xycoords="data",
            xytext=mid_point,
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color=color_1,
                lw=2,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=arrowsize,
            ),
        )
    else:
        ax.annotate(
            "",
            xy=arrow_pos_1,
            xycoords="data",
            xytext=start,
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color=color_1,
                lw=2,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=arrowsize,
            ),
        )

    # Draw the second arrow at arrow_2_pos
    arrow_pos_2 = (1 - arrow_2_pos) * start + arrow_2_pos * end
    if rev2:
        ax.annotate(
            "",
            xy=arrow_pos_2,
            xycoords="data",
            xytext=end,
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color=color_2,
                lw=2,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=arrowsize,
            ),
        )
    else:
        ax.annotate(
            "",
            xy=arrow_pos_2,
            xycoords="data",
            xytext=mid_point,
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color=color_2,
                lw=2,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=arrowsize,
            ),
        )


def draw_graph(
    g,
    path=None,
    scale=SCALE,
    node_size=NODE_SIZE,
    font_size=FONT_SIZE,
    layout=LAYOUT,
    ax=None,
):
    conns = list(nx.connected_components(nx.Graph(g)))
    if len(conns) > 1:
        assert np.all(":" in n for n in g) or np.all(":" not in n for n in g)
        num_rows = int(np.sqrt(len(conns) - 1)) + 1
        num_cols = num_rows
        fig = plt.Figure(figsize=(num_rows * DIM_SCALE, num_cols * DIM_SCALE))
        for j, conn in enumerate(conns):
            ax_ = fig.add_subplot(num_rows, num_cols, j + 1)
            g_conn = copy_graph(g, conn)
            g_conn = g_conn.__class__(g_conn)  # deepcopy
            # if these conns are different graphs
            for key in g.graph:
                if ":" in key:
                    index, key_ = key.split(":")
                    index = int(index)
                    if index == j:
                        g_conn.graph[key_] = g.graph[key]
                else:
                    g_conn.graph[key] = g.graph[key]
            draw_graph(
                g_conn,
                path=path,
                scale=scale,
                node_size=node_size,
                font_size=font_size,
                layout=layout,
                ax=ax_,
            )
    else:
        if "layout" in g.graph:
            layout = g.graph["layout"]
        if layout == "spring_layout":
            pos = getattr(nx, layout)(nx.Graph(g), seed=SEED)
        else:
            pos = getattr(nx, layout)(nx.Graph(g))
        pos_np = np.array([v for v in pos.values()])
        w, l = pos_np.max(axis=0) - pos_np.min(axis=0)
        w = max(w, 1)
        l = max(l, 1)
        w, l = w + POS_EPSILON, l + POS_EPSILON
        if ax is None:
            fig = plt.Figure(figsize=(scale * w, scale * l))
            ax_ = fig.add_subplot(1, 1, 1)
        else:
            ax_ = ax
        if "scale" in g.graph:
            scale = g.graph["scale"]
        if "font_size" in g.graph:
            font_size = g.graph["font_size"]
        colors = []
        node_sizes = []
        for n in g:
            if "label" in g.nodes[n]:
                if g.nodes[n]['label'] == 'light_grey':
                    breakpoint()
                c = to_rgba(g.nodes[n]["label"])
            else:
                c = to_rgba("r")
            if "alpha" in g.nodes[n]:
                c = c[:-1] + (g.nodes[n]["alpha"],)
            if "node_size" in g.nodes[n]:
                n_size = g.nodes[n]["node_size"]
                node_sizes.append(n_size)
            else:
                node_sizes.append(node_size)
            colors.append(c)
        labels = {}
        for n in g:
            if (
                isinstance(n, str) and ":" in n
            ):  # used to denote which graph, not relevant
                n_ = n.split(":")[-1]
            else:
                n_ = n
            if "type" in g.nodes[n]:
                type_ = g.nodes[n]["type"]
                labels[n] = f"({n_}) {type_}"
            else:
                labels[n] = f"({n_})"
        nx.draw_networkx_nodes(
            g, ax=ax_, pos=pos, node_color=colors, node_size=node_sizes
        )
        nx.draw_networkx_labels(g, ax=ax_, pos=pos, labels=labels, font_size=font_size)
        for u, v, dic in g.edges(data=True):
            style = dic.get("style", "solid")
            alpha = dic.get("alpha", 1.0)
            if "label" in dic:
                label = dic["label"]
                loc = 0.5
                # arrowprops = dict(arrowstyle="->", color=label, lw=EDGE_THICKNESS)
                # mid = loc*pos[u]+(1-loc)*pos[v]
                # ax_.annotate('', xy=mid, xytext=pos[u], arrowprops=arrowprops)
                # ax_.text(mid[0], mid[1], '', fontsize=12, color=label)
                nx.draw_networkx_edges(
                    g,
                    ax=ax_,
                    pos=pos,
                    edgelist=[(u, v)],
                    style=style,
                    alpha=alpha,
                    edge_color=label,
                    width=EDGE_THICKNESS,
                    node_size=node_size,
                    arrowsize=ARROW_SIZE,
                    arrows=True,
                )
            else:
                label1 = dic["label1"]
                label2 = dic["label2"]
                loc1 = dic["loc1"]
                loc2 = dic["loc2"]
                rev1 = dic["reverse1"]
                rev2 = dic["reverse2"]
                print(
                    f"u={u} v={v}, passing label1={label1}, label2={label2}, loc1={loc1}, loc2={loc2}, reverse1={rev1}, reverse2={rev2}"
                )
                draw_custom_arrows(
                    ax_,
                    pos,
                    (u, v),
                    color_1=label1,
                    color_2=label2,
                    arrow_1_pos=loc1,
                    arrow_2_pos=loc2,
                    rev1=rev1,
                    rev2=rev2,
                )
        if "title" in g.graph:
            ax_.set_title(g.graph["title"], fontsize=TITLE_FONT_SIZE)
    if ax is None:
        if path is None:
            return fig
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
            print(os.path.abspath(path))


def add_node(graph, node_id, label, shape="box", style="filled"):
    color = LOOKUP[label]
    label = f"({node_id}) {label}"
    # label = f"{label}"
    graph.add_node(
        node_id,
        label=label,
        color="black",
        fillcolor=color,
        shape=shape,
        style=style,
        fontsize=24,
    )


def draw_circuit(nx_g, path, title=""):
    g = igraph.Graph(directed=True)
    nx_g = nx.relabel_nodes(nx_g, {n: i for (i, n) in enumerate(nx_g)})
    g.add_vertices(nx_g.nodes())
    g.add_edges(nx_g.edges())
    graph = pgv.AGraph(
        directed=True, strict=True, fontname="Helvetica", arrowtype="open"
    )
    if title:
        graph.graph_attr["label"] = title
    graph.graph_attr["fontsize"] = 50
    if g is None:
        add_node(graph, 0, 0)
        graph.layout(prog="dot")
        graph.draw(path)
        return
    for idx in range(g.vcount()):
        add_node(graph, idx, g.vs[idx]["name"]["type"])
    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            graph.add_edge(node, idx, weight=0)
    graph.layout(prog="dot")
    graph.draw(path)


def draw_tree(node, path):
    g = nx.Graph()
    g.add_node(node.id, label=node.attrs["rule"])
    bfs = [node]
    while bfs:
        cur = bfs.pop(0)
        for nei in cur.children:
            g.add_node(nei.id, label=nei.attrs["rule"])
            g.add_edge(cur.id, nei.id)
            bfs.append(nei)
    pos = hierarchy_pos(g, node.id)
    fig = plt.Figure()
    ax = fig.add_subplot(1, 1, 1)
    labels = {n: g.nodes[n]["label"] for n in g}
    nx.draw(g, ax=ax, pos=pos, labels=labels, with_labels=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(os.path.abspath(path))
