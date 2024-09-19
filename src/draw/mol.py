from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import rdkit.Chem as Chem
from rdkit.Geometry.rdGeometry import Point2D
from PIL import Image
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from src.grammar.utils import flatten
import io
import os


def GetBondPosition(mol, bond, return_atom_pos=False):
    if isinstance(bond, int):
        bond = mol.GetBondWithIdx(bond)
    conf = mol.GetConformer()
    idx1 = bond.GetBeginAtomIdx()
    idx2 = bond.GetEndAtomIdx()
    pos1 = conf.GetAtomPosition(idx1)
    pos2 = conf.GetAtomPosition(idx2)
    mid_x = (pos1.x + pos2.x) / 2
    mid_y = (pos1.y + pos2.y) / 2
    if return_atom_pos:
        return mid_x, mid_y, pos1.x, pos1.y, pos2.x, pos2.y
    else:
        return mid_x, mid_y


def draw_smiles(smiles, ax=None, order=[], path=""):
    mol = Chem.MolFromSmiles(smiles)
    for j, a in enumerate(mol.GetAtoms()):
        a.SetProp("atomLabel", f"{a.GetSymbol()}{j}")
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
    options = drawer.drawOptions()
    options.maxFontSize = 12
    options.atomLabelFontSize = 12
    drawer.DrawMolecule(mol)
    drawer.SetFontSize(15)
    for bond in mol.GetBonds():
        mid_x, mid_y = GetBondPosition(mol, bond)
        bond_type = str(bond.GetBondType())
        if bond.GetIdx() in order:
            index = order.index(bond.GetIdx())
            bond_label = f"({index+1}) bond_{bond.GetIdx()}={bond_type}"
        else:
            bond_label = f"bond_{bond.GetIdx()}={bond_type}"
        drawer.DrawString(bond_label, Point2D(mid_x, mid_y))
    drawer.FinishDrawing()
    # drawer.WriteDrawingText(os.path.join(dir_name, f'{i}.png'))
    img_data = drawer.GetDrawingText()
    if path:
        image = io.BytesIO(img_data)
        img = plt.imread(image)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        fig.savefig(path)
        # print(os.path.abspath(path))
    elif ax:
        image = io.BytesIO(img_data)
        img = plt.imread(image)
        ax.imshow(img)
    else:
        from IPython.display import Image

        return Image(data=img_data)


def draw_mol(mol, ax=None, bonds=None, return_drawer=False):
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
    if bonds is None:
        drawer.DrawMolecule(mol)
    else:
        highlight_bond_map = {}
        for b in bonds:
            highlight_bond_map[b] = [(0, 0, 1)]
        drawer.DrawMoleculeWithHighlights(
            mol,
            "",
            highlight_atom_map={},
            highlight_bond_map=highlight_bond_map,
            highlight_radii={},
            highlight_linewidth_multipliers={},
        )
    if return_drawer:
        return drawer
    drawer.FinishDrawing()
    img_data = drawer.GetDrawingText()
    if ax:
        image = io.BytesIO(img_data)
        img = plt.imread(image)
        ax.imshow(img)
    else:
        return Image(data=img_data)


def draw_cliques(cg, mol, ax=None, cq=None, label=True):
    """
    This function draws the cliques in cq, highlighting them in mol.
    Parameters:
        cg: clique graph, where nodes are bonds of mol, edges are atoms
        mol: the mol object to draw
        ax: if given, draw on ax
        cq: if given, draws predefined cliques, can be one of following:
            tuple (id, nodes, *color)
            list of tuples
        label: whether to annotate the id as text
    Output:
        Image drawn
    """
    for j, a in enumerate(mol.GetAtoms()):
        a.SetProp("atomLabel", f"{a.GetSymbol()}{j}")
    global_color = (1, 0, 0)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
    options = drawer.drawOptions()
    options.noAtomLabels = True
    drawer.SetFillPolys(True)
    drawer.SetColour(global_color)
    options.maxFontSize = 20
    if cq:
        if isinstance(cq, tuple):
            cqs = [cq]
        else:
            assert isinstance(cq, list)
            cqs = cq
    else:
        cqs = list(enumerate(nx.find_cliques(cg)))
    for i, cq_arg in enumerate(cqs):
        if len(cq_arg) == 2:
            e, cq = cq_arg
            color = global_color
        else:
            e, cq, color = cq_arg
        x, y = 0, 0
        highlight_bond_map = {}
        for b in cq:
            bx, by = GetBondPosition(mol, b)
            x += bx
            y += by
            highlight_bond_map[b] = [color]
        drawer.DrawMoleculeWithHighlights(
            mol,
            "",
            highlight_atom_map={},
            highlight_bond_map=highlight_bond_map,
            highlight_radii={},
            highlight_linewidth_multipliers={},
        )
        if label:  # TODO: Support len(cqs) > 1
            x /= len(cq)
            y /= len(cq)
            label = f"e{e}"
            drawer.DrawString(label, Point2D(x, y))
    drawer.FinishDrawing()
    # drawer.WriteDrawingText(os.path.join(dir_name, f'{i}.png'))
    img_data = drawer.GetDrawingText()
    if ax:
        image = io.BytesIO(img_data)
        img = plt.imread(image)
        ax.imshow(img)
    else:
        return Image(data=img_data)


def clique_drawing(cg, mol, path):
    # draw_cliques(cg, mol)
    cliques = list(nx.find_cliques(cg))
    n = len(cliques)
    d = int(np.sqrt(n))
    fig, axes = plt.subplots(d, n // d, figsize=(20, 20))
    axes = flatten(axes)
    for i, (cq, ax) in enumerate(zip(cliques, axes)):
        ax.axis("off")
        ax.set_title(f"{i}")
        draw_cliques(cg, mol, ax=ax, cq=(i, cq), label=False)
    fig.set_facecolor("white")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    print(os.path.abspath(path))
    return cliques


def draw_cycle(cyc, tree, mol, path):
    n = len(cyc)
    d = int(np.sqrt(n))
    fig, axes = plt.subplots(d, n // d, figsize=(20, 20))
    axes = flatten(axes)
    for i, ((cq_1_id, cq_2_id), ax) in enumerate(zip(cyc, axes)):
        ax.axis("off")
        ax.set_title(f"{i}")
        cq_1 = tree.nodes[cq_1_id]["nodes"]
        cq_2 = tree.nodes[cq_2_id]["nodes"]
        draw_cliques(
            None,
            mol,
            ax,
            [(None, cq_1, (1, 0, 0)), (None, cq_2, (0, 1, 0))],
            label=False,
        )
    fig.set_facecolor("white")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    print(os.path.abspath(path))
    return path
