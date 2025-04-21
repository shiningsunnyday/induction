from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import rdkit.Chem as Chem
from rdkit.Geometry.rdGeometry import Point2D
from PIL import Image
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
from src.grammar.utils import flatten
from src.config import VISUALIZE
import io
import os
from pathlib import Path
from copy import deepcopy


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
    


def GetAtomsFromBonds(mol, bonds):
    return list(
        set(
            flatten(
                [
                    [
                        mol.GetBondWithIdx(bond).GetBeginAtomIdx(),
                        mol.GetBondWithIdx(bond).GetEndAtomIdx(),
                    ]
                    for bond in bonds
                ]
            )
        )
    )
    


def _extract_subgraph(mol, selected_atoms, selected_bonds=None):
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [
            y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms
        ]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)
    Chem.Kekulize(new_mol)
    for atom in new_mol.GetAtoms():
        atom.SetIntProp("org_idx", atom.GetIdx())
    for bond in new_mol.GetBonds():
        bond.SetIntProp("org_idx", bond.GetIdx())
    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [
            bond
            for bond in atom.GetBonds()
            if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
        ]
        aroma_bonds = [
            bond
            for bond in aroma_bonds
            if bond.GetBeginAtom().GetIdx() in selected_atoms
            and bond.GetEndAtom().GetIdx() in selected_atoms
        ]
        if len(aroma_bonds) == 0:
            # if atom.GetIsAromatic():
            #     breakpoint()
            atom.SetIsAromatic(False)

    remove_atoms = [
        atom.GetIdx()
        for atom in new_mol.GetAtoms()
        if atom.GetIdx() not in selected_atoms
    ]
    remove_atoms = sorted(remove_atoms, reverse=True)
    if selected_bonds is not None:
        remove_bonds = [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in new_mol.GetBonds()
            if bond.GetIdx() not in selected_bonds
        ]
    if selected_bonds is not None:
        dic = {}
        for b in new_mol.GetBonds():
            dic[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = b.GetBondType()
        for bond in remove_bonds:
            new_mol.RemoveBond(*bond)
        for b in new_mol.GetBonds():
            if dic[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] != b.GetBondType():
                breakpoint()
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)
    new_mol = new_mol.GetMol()
    return new_mol




def draw_smiles(smiles, ax=None, order=[], path="", label_atoms=True, label_atom_idx=False, label_bonds=True, maxFontSize=12, atomLabelFontSize=12, fontSize=15, dpi=500, width=-1, height=-1, label_bond_type=False):
    mol = Chem.MolFromSmiles(smiles)
    for j, a in enumerate(mol.GetAtoms()):
        atom_label = f"{a.GetSymbol()}"
        if label_atom_idx:
            atom_label += str(j)
        a.SetProp("atomLabel", atom_label)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.maxFontSize = maxFontSize
    options.atomLabelFontSize = atomLabelFontSize
    drawer.DrawMolecule(mol)
    drawer.SetFontSize(fontSize)
    for bond in mol.GetBonds():
        mid_x, mid_y = GetBondPosition(mol, bond)
        bond_type = str(bond.GetBondType())
        if bond.GetIdx() in order:
            index = order.index(bond.GetIdx())
            bond_label = f"({index+1}) bond_{bond.GetIdx()}"
        else:
            bond_label = f"bond_{bond.GetIdx()}"
        if label_bond_type:
            bond_label += f"={bond_type}"        
        if label_bonds:
            drawer.DrawString(bond_label, Point2D(mid_x, mid_y))
    drawer.FinishDrawing()
    # drawer.WriteDrawingText(os.path.join(dir_name, f'{i}.png'))
    img_data = drawer.GetDrawingText()
    if path:
        image = io.BytesIO(img_data)
        img = plt.imread(image)
        fig, ax = plt.subplots(1, 1)
        ax.axis("off")
        ax.imshow(img)
        fig.set_facecolor("white")        
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        # print(os.path.abspath(path))
        plt.close(fig)
    else:
        image = io.BytesIO(img_data)
        img = plt.imread(image)        
        if ax:
            ax.imshow(img)    
        return img


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
    image = io.BytesIO(img_data)
    img = plt.imread(image)
    if ax:
        ax.imshow(img)
    return img    


def draw_cliques(cg, mol, ax=None, cq=None, label=True, label_atoms=True, label_atom_idx=False, text_only=False):
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
        label_atoms: whether to label atomic symbols
        text_only: whether to return text only
    Output:
        Image drawn or Textual Description (text_only)
    """
    for j, a in enumerate(mol.GetAtoms()):
        atom_label = f"{a.GetSymbol()}"
        if label_atom_idx:
            atom_label += str(j)
        a.SetProp("atomLabel", atom_label)
    global_color = (0, 1, 0) # for acrylates
    # global_color = (1, 0, 0)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(-1, -1)
    options = drawer.drawOptions()
    options.noAtomLabels = not label_atoms
    drawer.SetFillPolys(True)
    # drawer.SetColour(global_color)
    options.maxFontSize = 20
    if cq:
        if isinstance(cq, tuple):
            cqs = [cq]
        else:
            assert isinstance(cq, list)
            cqs = cq
    else:
        cqs = list(enumerate(nx.find_cliques(cg)))
    highlight_bond_map = {}
    for i, cq_arg in enumerate(cqs):
        if len(cq_arg) == 2:
            e, cq = cq_arg
            color = global_color
        else:
            e, cq, color = cq_arg
        x, y = 0, 0        
        for b in cq:
            bx, by = GetBondPosition(mol, b)
            x += bx
            y += by
            highlight_bond_map[b] = highlight_bond_map.get(b, []) + [color]
        if label: # TODO: Support len(cqs) > 1
            x /= len(cq)
            y /= len(cq)              
            label = f"e{e}"
            drawer.DrawString(label, Point2D(x, y))    
    if text_only:
        atoms = GetAtomsFromBonds(mol, list(highlight_bond_map))
        mol = deepcopy(mol)
        for atom in mol.GetAtoms():
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
        numbered_smi = Chem.MolToSmiles(mol)
        motif_atoms = ','.join(map(str, sorted(atoms)))
        ans = (numbered_smi, motif_atoms)
        return ans
    else:
        drawer.DrawMoleculeWithHighlights(mol, '', 
                                        highlight_atom_map={}, 
                                        highlight_bond_map=highlight_bond_map, 
                                        highlight_radii={}, 
                                        highlight_linewidth_multipliers={})       
        drawer.FinishDrawing()
        # drawer.WriteDrawingText(os.path.join(dir_name, f'{i}.png'))
        img_data = drawer.GetDrawingText()
        image = io.BytesIO(img_data)
        img = plt.imread(image)
        if ax:        
            ax.imshow(img)    
        return img


def clique_drawing(cg, mol, path, isolate=False, scheme='zero', text_only=False):    
    if text_only:
        assert path is None
        assert not VISUALIZE
    # draw_cliques(cg, mol)
    cliques = list(nx.find_cliques(cg))
    n = len(cliques)
    d = int(np.sqrt(n))
    h = (n+d-1) // d
    res = []
    if not text_only:
        fig, axes = plt.subplots(d, h)
        axes = flatten(axes)
        ppath = Path(path)
        for ax in axes:
            ax.axis("off")
        white_shape = None
        pack_args = enumerate(zip(cliques, axes))
    else:
        pack_args = enumerate(cliques)
    for parg in pack_args:
        if text_only:
            i, cq = parg
            ax = None
        else:
            i, (cq, ax) = parg
        if VISUALIZE:
            fig_i, fig_ax = plt.subplots()
            fig_ax.axis("off")
        if not text_only: 
            ax.set_title(str(i if scheme =='zero' else i+1))
        if isolate: 
            cq_atoms = GetAtomsFromBonds(mol, cq)
            submol = _extract_subgraph(mol, selected_atoms=cq_atoms, selected_bonds=cq)
            if text_only:
                smi = Chem.MolToSmiles(submol)
                res.append(smi)
            else:
                img = draw_mol(submol, ax=ax)
                if i == 0 and len(axes) > len(cliques):
                    white_shape = img.shape
            if VISUALIZE:
                draw_mol(submol, ax=fig_ax)
        else:
            img = draw_cliques(cg, mol, ax=ax, cq=(i, cq), label=False, text_only=text_only)
            if text_only:
                orig_smiles, motif_atoms = img
                if i == 0:
                    res.append(f"Molecule: {orig_smiles}\n\nMotif {i}: {motif_atoms}")
                else:
                    res.append(f"Motif {i}: {motif_atoms}")
            elif i == 0 and len(axes) > len(cliques):
                white_shape = img.shape
            if VISUALIZE:
                draw_cliques(cg, mol, ax=fig_ax, cq=(i, cq), label=False)        
        if VISUALIZE:
            fig_i.set_facecolor("white")
            path_i = path.replace(ppath.suffix, f"_{i}{ppath.suffix}")
            fig_i.savefig(path_i, bbox_inches="tight", dpi=500)
            plt.close(fig_i)
    if not text_only:
        for i in range(len(cliques), len(axes)):
            axes[i].imshow(np.ones(white_shape))
        fig.set_facecolor("white")
        fig.savefig(path, bbox_inches="tight", dpi=500)
        print(os.path.abspath(path))
        plt.close(fig)
        return cliques
    else:
        return cliques, res


def draw_cycle(cyc, tree, mol, path, scheme='zero'):
    n = len(cyc)
    d = int(np.sqrt(n))
    fig, axes = plt.subplots(d, n // d)
    axes = flatten(axes)
    white_shape = None
    for i, ((cq_1_id, cq_2_id), ax) in enumerate(zip(cyc, axes)):
        ax.axis("off")
        ax.set_title(str(i if scheme =='zero' else i+1))
        cq_1 = tree.nodes[cq_1_id]["nodes"]
        cq_2 = tree.nodes[cq_2_id]["nodes"]      
        img = draw_cliques(
            None,
            mol,
            ax,
            [(None, cq_1, (1, 0, 0)), (None, cq_2, (0, 1, 0))],
            label=False,
        )
        if i == 0 and len(axes) > len(cyc):
            white_shape = img.shape        
    for i in range(len(cyc), len(axes)):
        axes[i].imshow(np.ones(white_shape))        
    fig.set_facecolor("white")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    print(os.path.abspath(path))
    plt.close(fig)
    return path
