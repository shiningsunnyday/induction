import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "my_data_efficient_grammar"))
import argparse
from rdkit import Chem
from rdkit.Chem import rdchem
from multiprocessing import Pool
from tqdm import tqdm
from itertools import permutations
from functools import reduce
from rdkit.Chem import Draw
from matplotlib.patches import FancyArrowPatch
from networkx.algorithms.isomorphism import GraphMatcher
from collections import defaultdict, Counter
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import json
from rdkit.Chem.rdmolops import FastFindRings
from itertools import accumulate, product
from copy import deepcopy
from private.molecule_graph import MolGraph
import networkx.algorithms.chordal as chordal
import pandas as pd
import pickle
from src.config import *
from src.draw.mol import *
from src.grammar.common import *
from src.grammar.hrg import *
import networkx as nx
import random
import concurrent.futures
import time
import multiprocessing as mp
import re
from rdkit.Chem.rdchem import BondType as btype

bond_lookup = {
    0: btype.UNSPECIFIED,
    1: btype.SINGLE,
    2: btype.DOUBLE,
    3: btype.TRIPLE,
    4: btype.QUADRUPLE,
    5: btype.QUINTUPLE,
    6: btype.HEXTUPLE,
    7: btype.ONEANDAHALF,
    8: btype.TWOANDAHALF,
    9: btype.THREEANDAHALF,
    10: btype.FOURANDAHALF,
    11: btype.FIVEANDAHALF,
    12: btype.AROMATIC,
    13: btype.IONIC,
    14: btype.HYDROGEN,
    15: btype.THREECENTER,
    16: btype.DATIVEONE,
    17: btype.DATIVE,
    18: btype.DATIVEL,
    19: btype.DATIVER,
    20: btype.OTHER,
    21: btype.ZERO,
}
inv_bond_lookup = dict(zip(bond_lookup.values(), bond_lookup.keys()))


def set_stereo(mol):
    """
    0: rdkit.Chem.rdchem.BondDir.NONE,
    1: rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
    2: rdkit.Chem.rdchem.BondDir.BEGINDASH,
    3: rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
    4: rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,
    """
    _mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    Chem.Kekulize(_mol, True)
    substruct_match = mol.GetSubstructMatch(_mol)
    if not substruct_match:
        """mol and _mol are kekulized.
        sometimes, the order of '=' and '-' changes, which causes mol and _mol not matched.
        """
        Chem.SetAromaticity(mol)
        Chem.SetAromaticity(_mol)
        substruct_match = mol.GetSubstructMatch(_mol)
    try:
        atom_match = {
            substruct_match[_mol_atom_idx]: _mol_atom_idx
            for _mol_atom_idx in range(_mol.GetNumAtoms())
        }  # mol to _mol
    except:
        raise ValueError("two molecules obtained from the same data do not match.")

    for each_bond in mol.GetBonds():
        begin_atom_idx = each_bond.GetBeginAtomIdx()
        end_atom_idx = each_bond.GetEndAtomIdx()
        _bond = _mol.GetBondBetweenAtoms(
            atom_match[begin_atom_idx], atom_match[end_atom_idx]
        )
        _bond.SetStereo(each_bond.GetStereo())

    mol = _mol
    for each_bond in mol.GetBonds():
        if int(each_bond.GetStereo()) in [2, 3]:  # 2=Z (same side), 3=E
            begin_stereo_atom_idx = each_bond.GetBeginAtomIdx()
            end_stereo_atom_idx = each_bond.GetEndAtomIdx()
            begin_atom_idx_set = set(
                [
                    each_neighbor.GetIdx()
                    for each_neighbor in mol.GetAtomWithIdx(
                        begin_stereo_atom_idx
                    ).GetNeighbors()
                    if each_neighbor.GetIdx() != end_stereo_atom_idx
                ]
            )
            end_atom_idx_set = set(
                [
                    each_neighbor.GetIdx()
                    for each_neighbor in mol.GetAtomWithIdx(
                        end_stereo_atom_idx
                    ).GetNeighbors()
                    if each_neighbor.GetIdx() != begin_stereo_atom_idx
                ]
            )
            if not begin_atom_idx_set:
                each_bond.SetStereo(Chem.rdchem.BondStereo(0))
                continue
            if not end_atom_idx_set:
                each_bond.SetStereo(Chem.rdchem.BondStereo(0))
                continue
            if len(begin_atom_idx_set) == 1:
                begin_atom_idx = begin_atom_idx_set.pop()
                begin_another_atom_idx = None
            if len(end_atom_idx_set) == 1:
                end_atom_idx = end_atom_idx_set.pop()
                end_another_atom_idx = None
            if len(begin_atom_idx_set) == 2:
                atom_idx_1 = begin_atom_idx_set.pop()
                atom_idx_2 = begin_atom_idx_set.pop()
                if int(mol.GetAtomWithIdx(atom_idx_1).GetProp("_CIPRank")) < int(
                    mol.GetAtomWithIdx(atom_idx_2).GetProp("_CIPRank")
                ):
                    begin_atom_idx = atom_idx_1
                    begin_another_atom_idx = atom_idx_2
                else:
                    begin_atom_idx = atom_idx_2
                    begin_another_atom_idx = atom_idx_1
            if len(end_atom_idx_set) == 2:
                atom_idx_1 = end_atom_idx_set.pop()
                atom_idx_2 = end_atom_idx_set.pop()
                if int(mol.GetAtomWithIdx(atom_idx_1).GetProp("_CIPRank")) < int(
                    mol.GetAtomWithIdx(atom_idx_2).GetProp("_CIPRank")
                ):
                    end_atom_idx = atom_idx_1
                    end_another_atom_idx = atom_idx_2
                else:
                    end_atom_idx = atom_idx_2
                    end_another_atom_idx = atom_idx_1

            if each_bond.GetStereo() == 2:  # same side
                mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 4)
                each_bond.SetStereoAtoms(begin_atom_idx, end_atom_idx)
            elif each_bond.GetStereo() == 3:  # opposite side
                mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 3)
                each_bond.SetStereoAtoms(begin_atom_idx, end_atom_idx)
            else:
                raise ValueError
    return mol


# SEED = 0
# random.seed(SEED)
# np.random.seed(SEED)
# import pygsp as gsp
# from pygsp import graphs

from src.api.get_motifs import prepare_images
import openai

openai.api_key = open("notebooks/api_key.txt").readline().rstrip("\n")

# IMAGE_PATHS = [
#     "/home/msun415/SynTreeNet/induction/CCOC(C(N=C=O)CCCCN=C=O)=O.png",
#     "/home/msun415/SynTreeNet/induction/O=C=NC1CCC(CC2CCC(CC2)N=C=O)CC1.png",
#     "/home/msun415/SynTreeNet/induction/CC1=C(C=C(C=C1)CN=C=O)N=C=O.png",
#     "/home/msun415/SynTreeNet/induction/CC1(CC(CC(CN=C=O)(C1)C)N=C=O)C.png",
#     "/home/msun415/SynTreeNet/induction/O=C=NCCCCCCCCCCCCCCCCCCCCCCCCN=C=O.png"
#     ]


from rdkit.Chem.Draw import IPythonConsole
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Geometry.rdGeometry import Point2D

IPythonConsole.ipython_useSVG = (
    True  # < set this to False if you want PNGs instead of SVGs
)
from fuseprop import (
    find_clusters,
    extract_subgraph,
    get_mol,
    get_smiles,
    find_fragments,
    find_fragments_with_scaffold,
    __extract_subgraph,
)
from private import *
from src.draw.utils import hierarchy_pos
import HRG.create_production_rules as cpr
import importlib
from collections import defaultdict

importlib.reload(cpr)


# subgraphs = []
# subgraphs_idx_i = []
# clusters, atom_cls = find_clusters(mol)
# for i,cls in enumerate(clusters):
#     clusters[i] = set(list(cls))
# for i, cluster in enumerate(clusters):
#     _, subgraph_i_mapped, _ = extract_subgraph(smiles, cluster)
#     subgraphs.append(SubGraph(subgraph_i_mapped, mapping_to_input_mol=subgraph_i_mapped, subfrags=list(cluster)))
#     subgraphs_idx_i.append(list(cluster))


def mol_to_graph(mol):
    g = nx.Graph()
    for b in mol.GetBonds():
        g.add_node(b.GetIdx())
    for a in mol.GetAtoms():
        bs = [b.GetIdx() for b in a.GetBonds()]
        g.add_edges_from(product(bs, bs))
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


class MolHG:
    def __init__(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        self.mol = mol
        self.chordal_graph = self.llm_chordalize(mol)

    @staticmethod
    def llm_chordalize(mol):
        pass


def isRingAromatic(mol, bondRing):
    for id in bondRing:
        if not mol.GetBondWithIdx(id).GetIsAromatic():
            return False
    return True


def GetBondsAmongAtoms(mol, ring):
    return [
        mol.GetBondBetweenAtoms(i, j).GetIdx()
        for i in ring
        for j in ring
        if mol.GetBondBetweenAtoms(i, j)
    ]


def GetRingsWithBond(mol, b_idx):
    res = []
    for b_ring in mol.GetRingInfo().BondRings():
        add = False
        for idx in b_ring:
            if idx == b_idx:
                add = True
                break
        if add:
            res.append(b_ring)
    return res


# def get_clique_graph(mol):
#     g = mol_to_graph(mol)
#     for ring in mol.GetRingInfo().BondRings():
#         if isRingAromatic(mol, ring):
#             g.add_edges_from(product(ring, ring))
#     g.remove_edges_from(nx.selfloop_edges(g))
#     return g


def get_clique_graph(mol):
    g = mol_to_graph(mol)
    for ring in Chem.GetSymmSSSR(mol):  # TODO
        ring = list(ring)
        ring = GetBondsAmongAtoms(mol, ring)
        g.add_edges_from(product(ring, ring))
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def my_complete_to_chordal(cg, mol):
    while True:
        try:
            res, order = chordal._find_chordality_breaker(cg)
        except:
            break
        u, _, w = res
        u_rings = GetRingsWithBond(mol, u)
        w_rings = GetRingsWithBond(mol, w)
        for u_r, w_r in product(u_rings, w_rings):
            cg.add_edges_from(product(u_r, w_r))
        cg.remove_edges_from(nx.selfloop_edges(cg))
    return cg


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


class Grammar(HRG):
    def __init__(self, mol, rules):
        mol = deepcopy(mol)
        Chem.Kekulize(mol)
        self.mol = mol
        self.rules = rules
        self.init_symbol = "(S)"
        self.atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        self.vocab = {
            key: key.count(",") + 1 for key in rules if self.init_symbol != key
        }
        self.vocab.update({self.init_symbol: 0})
        self.vocab.update({atom: None for atom in self.atoms})
        rules = self.ProcessAllRules()
        self.hrg = HRG(list(self.rules), self.atoms, self.init_symbol, self.vocab)
        self.hrg_rule_index_map = deepcopy(self.rules)
        for key, rules_for_key in rules.items():
            for rule, rule_item in rules_for_key.items():
                index = Grammar.check_exist(self.hrg.rules, rule_item)
                if index == -1:
                    self.hrg_rule_index_map[key][rule] = len(self.hrg.rules)
                    self.hrg.add_rule(rule_item)
                else:
                    self.hrg_rule_index_map[key][rule] = index

    @staticmethod
    def check_exist(rules, rule):
        for idx, r in enumerate(rules):
            r_g = r.rhs.visualize("", return_g=True)
            rule_g = rule.rhs.visualize("", return_g=True)
            if len(r_g) != len(rule_g):
                continue
            gm = GraphMatcher(
                r_g, rule_g, node_match=lambda x, y: x["label"] == y["label"]
            )
            subs = list(gm.isomorphisms_iter())
            for sub in subs:
                ext_ids = set([sub[a.id] for a in r.rhs.ext])
                mapped_ids = set([a.id for a in rule.rhs.ext])
                if ext_ids == mapped_ids:
                    return idx
        return -1

    def combine(self, other):
        def _combine_atoms(self, other):
            for a in other.atoms:
                if a not in self.atoms:
                    self.atoms.append(a)

        def _combine_vocabs(self, other):
            for k in other.vocab:
                if k in self.vocab:
                    assert self.vocab[k] == other.vocab[k]
                else:
                    self.vocab[k] = other.vocab[k]

        def _init_mol_lookup(self):
            smiles = Chem.MolToSmiles(self.mol)
            self.mol_lookup = {smiles: deepcopy(self.rules)}
            self.rule_idx_lookup = {smiles: deepcopy(self.hrg_rule_index_map)}

        def _merge_hrgs(self, other):
            merge_keys = set(self.hrg_rule_index_map)
            for other_smi in other.mol_lookup:
                merge_keys |= set(other.mol_lookup[other_smi])
            merge_keys = list(merge_keys)
            hrg = HRG(merge_keys, self.atoms, self.init_symbol, self.vocab)
            for rule in self.hrg.rules:
                hrg.add_rule(rule)
            hrg.set_counts(self.hrg.counts)
            remap_rule_idx = _combine_rules(hrg, other.hrg.rules)
            hrg.combine_counts(other.hrg, remap_rule_idx)
            self.hrg = hrg
            for other_smi in other.mol_lookup:
                if other_smi not in self.mol_lookup:
                    self.mol_lookup[other_smi] = deepcopy(other.mol_lookup[other_smi])
                    self.rule_idx_lookup[other_smi] = deepcopy(other.hrg_rule_index_map)
                else:
                    for key in other.mol_lookup[other_smi]:
                        self.mol_lookup[other_smi][key] = self.mol_lookup[
                            other_smi
                        ].get(key, {})
                        self.rule_idx_lookup[other_smi][key] = self.rule_idx_lookup[
                            other_smi
                        ].get(key, {})
                        for rule, rule_dict in other.mol_lookup[other_smi][key].items():
                            if rule in self.mol_lookup[other_smi][key]:
                                assert (
                                    self.mol_lookup[other_smi][key][rule] == rule_dict
                                )
                            self.mol_lookup[other_smi][key][rule] = self.mol_lookup[
                                other_smi
                            ][key].get(rule, {})
                            self.rule_idx_lookup[other_smi][key][
                                rule
                            ] = self.rule_idx_lookup[other_smi][key].get(rule, {})
                            self.mol_lookup[other_smi][key][rule] = rule_dict
                            org_idx = other.rule_idx_lookup[other_smi][key][rule]
                            self.rule_idx_lookup[other_smi][key][rule] = remap_rule_idx[
                                org_idx
                            ]

        def _combine_rules(hrg, other_rules):
            remap_rule_idx = {}
            for i, rule in enumerate(other_rules):
                index = Grammar.check_exist(hrg.rules, rule)
                if index == -1:
                    index = len(hrg.rules)
                    hrg.add_rule(rule)
                remap_rule_idx[i] = index
            return remap_rule_idx

        def _append_hrg(self, other):
            self.mol_lookup[Chem.MolToSmiles(other.mol)] = deepcopy(other.rules)
            self.rule_idx_lookup[Chem.MolToSmiles(other.mol)] = deepcopy(
                other.hrg_rule_index_map
            )
            merge_keys = list(set(self.hrg_rule_index_map) | set(other.rules))
            hrg = HRG(merge_keys, self.atoms, self.init_symbol, self.vocab)
            for rule in self.hrg.rules:
                hrg.add_rule(rule)
            remap_rule_idx = _combine_rules(hrg, other.hrg.rules)
            self.combine_counts(other, remap_rule_idx)
            self.hrg = hrg
            other_smiles = Chem.MolToSmiles(other.mol)
            for key in self.rule_idx_lookup[other_smiles]:
                for rule in self.rule_idx_lookup[other_smiles][key]:
                    self.rule_idx_lookup[other_smiles][key][rule] = remap_rule_idx[
                        self.rule_idx_lookup[other_smiles][key][rule]
                    ]

        def _combine_hrgs(self, other):
            if hasattr(other, "mol_lookup"):
                _merge_hrgs(self, other)
            else:
                _append_hrg(self, other)

        # merge the vocabs
        # list the init_symbols
        # make hrg with all rules, terms, starts, new vocab
        # add the rules
        assert not (hasattr(self, "mol_lookup") ^ hasattr(self, "rule_idx_lookup"))
        if not hasattr(self, "mol_lookup"):
            _init_mol_lookup(self)
        _combine_atoms(self, other)
        _combine_vocabs(self, other)
        _combine_hrgs(self, other)
        return self

    @staticmethod
    def __extract_subgraph(mol, selected_atoms, selected_bonds=None):
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

    def NumNTs(self):
        return len(self.rules)

    def GetNTs(self):
        return list(self.rules)

    def NumRulesForNT(self, nonterm):
        assert nonterm in self.rules
        return len(self.rules[nonterm])

    def GetRule(self, nonterm, idx):
        assert idx < self.NumRulesForNT(nonterm)
        return list(self.rules[nonterm])[idx]

    def EdgesForRule(self, nonterm, idx, nt_only=False):
        rule = self.GetRule(nonterm, idx)
        nt_or_t = "N" if nt_only else "N|T"
        matches = re.findall(f"(\((?:\w+,)*\w+:(?:{nt_or_t})\))", rule[0])
        is_nts = [":N" in match for match in matches]
        return matches, is_nts

    def GetEdgeForRule(self, nonterm, idx, i, nt_only=False):
        matches, is_nts = self.EdgesForRule(nonterm, idx, nt_only=nt_only)
        return matches[i], is_nts[i]

    def RHSNodes(self, nonterm, idx):
        rule = self.GetRule(nonterm, idx)
        _, nodes, d = self.rules[nonterm][rule]
        return nodes, d

    def RHSMol(self, nonterm, idx):
        nodes, _ = self.RHSNodes(nonterm, idx)
        return Grammar.__extract_subgraph(
            self.get_mol(nonterm, idx),
            GetAtomsFromBonds(self.get_mol(nonterm, idx), nodes),
            nodes,
        )

    def RHSEdgeMol(self, nonterm, idx, i, nt_only=False):
        nodes, d = self.RHSNodes(nonterm, idx)
        inv_d = dict(zip(d.values(), d.keys()))
        # logger.info(d, "map org idx to rhs idx")
        # logger.info(inv_d, "map rhs idx to org idx")
        # logger.info(nodes, "nodes of rhs")
        match, is_nt = self.GetEdgeForRule(nonterm, idx, i, nt_only=nt_only)
        # if ":N" in match:
        #     logger.info(match, "match")
        nt_or_t = "N" if nt_only else "N|T"
        grps = re.match(f"\(((?:\w+,)*\w+):({nt_or_t})\)", match)
        nodes_idx_str, _ = grps.groups()
        nodes_idx = list(
            map(
                lambda ind: int(ind) if ind.isdigit() else ind, nodes_idx_str.split(",")
            )
        )
        # logger.info(nodes_idx, "rhs edge nodes idx")
        bonds = [inv_d[ind] for ind in nodes_idx]  # these refer to bonds of hyperedge
        # logger.info(bonds, "bonds")
        # logger.info([nodes[ind] for ind in nodes_idx], "old bonds")
        rhs_edge_mol = Grammar.__extract_subgraph(
            self.get_mol(nonterm, idx),
            GetAtomsFromBonds(self.get_mol(nonterm, idx), bonds),
            bonds,
        )
        rhs_edge_bond_lookup = Grammar.bond_lookup(rhs_edge_mol)
        # logger.info(rhs_edge_bond_lookup, "rhs edge bond lookup")
        rhs_edge_bonds = [
            rhs_edge_bond_lookup[b] for b in bonds
        ]  # TODO: map bonds of hyperedge to bonds in rhs_mol
        # return rhs_edge_bonds, bonds, rhs_edge_mol
        nodes_rhs_edge_idx = [inv_d[b] for b in nodes_idx]  # org idx in mol
        return nodes_rhs_edge_idx, rhs_edge_bonds, rhs_edge_mol, is_nt

    def VisRule(self, nonterm, idx, nt_only=False):
        edges, _ = self.EdgesForRule(nonterm, idx, nt_only)
        fig, axes = plt.subplots(1, len(edges) + 1, figsize=(10, 10))
        axes = flatten(axes)
        nodes, _ = self.RHSNodes(nonterm, idx)
        rhs_mol = Grammar.__extract_subgraph(
            self.get_mol(nonterm, idx),
            GetAtomsFromBonds(self.get_mol(nonterm, idx), nodes),
            nodes,
        )
        draw_mol(rhs_mol, axes[0])
        for i in range(len(edges)):
            _, bonds, rhs_edge_mol, is_nt = self.RHSEdgeMol(nonterm, idx, i, nt_only)
            draw_mol(rhs_edge_mol, axes[i + 1], bonds=bonds)
        return fig

    @staticmethod
    def bond_lookup(rhs_mol):
        return {bd.GetIntProp("org_idx"): bd.GetIdx() for bd in rhs_mol.GetBonds()}

    @staticmethod
    def atom_lookup(rhs_mol):
        return {at.GetIntProp("org_idx"): at.GetIdx() for at in rhs_mol.GetAtoms()}

    def VisRuleAlt(self, nonterm, idx, nt_only=False, ax=None):
        edges, _ = self.EdgesForRule(nonterm, idx, nt_only=nt_only)
        nodes, d = self.RHSNodes(nonterm, idx)
        inv_d = dict(zip(d.values(), d.keys()))
        rhs_mol = Grammar.__extract_subgraph(
            self.get_mol(nonterm, idx),
            GetAtomsFromBonds(self.get_mol(nonterm, idx), nodes),
            nodes,
        )
        nts = re.match("\(((?:[a-z],)*[a-z])\)", nonterm)
        bond_lookup = Grammar.bond_lookup(rhs_mol)
        if nts is None:
            anchors = None
        else:
            nts = nts.groups()[0].split(",")
            anchors = [inv_d[nt] for nt in nts]
            anchors = [bond_lookup[b] for b in anchors]
        drawer = draw_mol(rhs_mol, bonds=anchors, return_drawer=True)
        dim = 0.05
        for i in range(len(edges)):
            random_color = (random.random(), random.random(), random.random())
            bonds, _, _, is_nt = self.RHSEdgeMol(nonterm, idx, i, nt_only=nt_only)
            bonds = [bond_lookup[b] for b in bonds]
            # logger.info(bonds, "bonds", is_nt, "is_nt")
            bonds_pos = [GetBondPosition(rhs_mol, bond) for bond in bonds]
            bonds_pos_mean = np.array(bonds_pos).mean(axis=0)
            drawer.SetFillPolys(not is_nt)
            if not is_nt:
                continue
            drawer.DrawRect(
                Point2D(*(bonds_pos_mean - dim)), Point2D(*(bonds_pos_mean + dim))
            )
            for bond in bonds:
                x, y = GetBondPosition(rhs_mol, bond)
                drawer.DrawArrow(
                    Point2D(x, y),
                    Point2D(*bonds_pos_mean),
                    asPolygon=True,
                    color=random_color,
                    frac=0.3,
                )
        drawer.FinishDrawing()
        img_data = drawer.GetDrawingText()
        if ax is None:
            from IPython.display import Image

            return Image(data=img_data)
        else:
            image = io.BytesIO(img_data)
            img = plt.imread(image)
            ax.imshow(img)

    def get_mol(self, nonterm, idx):
        if hasattr(self, "mol"):
            return self.mol
        else:
            return self.mol_lookup[nonterm][list(self.rules[nonterm])[idx]]

    def ProcessRule(self, nonterm, idx, nt_only=False):
        edges, _ = self.EdgesForRule(nonterm, idx, nt_only=nt_only)
        nodes, d = self.RHSNodes(nonterm, idx)
        inv_d = dict(zip(d.values(), d.keys()))
        rhs_mol = Grammar.__extract_subgraph(
            self.get_mol(nonterm, idx),
            GetAtomsFromBonds(self.get_mol(nonterm, idx), nodes),
            nodes,
        )
        nts = re.match("\(((?:[a-z],)*[a-z])\)", nonterm)
        bond_lookup = Grammar.bond_lookup(rhs_mol)
        if nts is None:
            anchors = []
        else:
            nts = nts.groups()[0].split(",")
            anchors = [inv_d[nt] for nt in nts]
            anchors = [bond_lookup[b] for b in anchors]
        node_labels = [
            {
                "label": inv_bond_lookup[rhs_mol.GetBondWithIdx(i).GetBondType()],
                "stereo": rhs_mol.GetBondWithIdx(i).GetStereo(),
            }
            for i in range(rhs_mol.GetNumBonds())
        ]
        rhs = HG(len(nodes), anchors, node_labels)
        for i in range(rhs_mol.GetNumAtoms()):
            at = rhs_mol.GetAtomWithIdx(i)
            sub_nodes = [f"n{b.GetIdx()}" for b in at.GetBonds()]
            kwargs = {
                "label": at.GetSymbol(),
                "is_atom": True,
                "num_explicit_Hs": at.GetNumExplicitHs(),
                "formal_charge": at.GetFormalCharge(),
                "chirality": at.GetChiralTag().real,
            }
            rhs.add_hyperedge(sub_nodes, **kwargs)
        # drawer = draw_mol(rhs_mol, bonds=anchors, return_drawer=True)
        for i in range(len(edges)):
            bonds, _, _, is_nt = self.RHSEdgeMol(nonterm, idx, i, nt_only=nt_only)
            bonds = [bond_lookup[b] for b in bonds]
            if not is_nt:
                continue
            label = r",".join([chr(o) for o in range(ord("a"), ord("a") + len(bonds))])
            rhs.add_hyperedge([f"n{b}" for b in bonds], label=f"({label})")
        return HRG_rule(nonterm, rhs, self.vocab)

    def VisAllRules(self):
        max_title_length = 20
        nonterms = self.GetNTs()
        counts = [self.NumRulesForNT(nonterm) for nonterm in nonterms]
        fig, axes = plt.subplots(len(counts), max(counts), figsize=(50, 50))
        if max(counts) == 1:
            if len(counts) == 1:
                axes = [[axes]]
            else:
                axes = axes[:, None]
        for i in range(len(counts)):
            axes[i][0].set_ylabel(nonterms[i], fontsize=36)
        for i in range(len(counts)):
            for j in range(counts[i]):
                nt_edges = self.EdgesForRule(nonterms[i], j, False)[0]
                nt_edges = "".join(nt_edges)
                # logger.info(nt_edges)
                # logger.info(len(nt_edges))
                if len(nt_edges) > max_title_length:
                    title = "..." + nt_edges[-max_title_length:]
                else:
                    title = nt_edges
                axes[i][j].set_title(title, fontsize=36)
                self.VisRuleAlt(nonterms[i], j, ax=axes[i][j])
        return fig

    def ProcessAllRules(self):
        nonterms = self.GetNTs()
        counts = [self.NumRulesForNT(nonterm) for nonterm in nonterms]
        proc_rules = {}
        for i in range(len(counts)):
            proc_rules[nonterms[i]] = {}
            for j in range(counts[i]):
                key = list(self.rules[nonterms[i]])[j]
                proc_rules[nonterms[i]][key] = self.ProcessRule(nonterms[i], j)
        return proc_rules

    @staticmethod
    def atom_neis(g, vs):
        neis = []
        for x in vs:
            neis += [n for n in g[x] if g.nodes[n]["label"][0] != "("]
        return neis

    @staticmethod
    def _generate(hrg, vocab, shared_dict=None):
        logger = globals()["logger"]
        rule_lookup = defaultdict(list)
        for i, r in enumerate(hrg.rules):
            rhs_g = r.rhs.visualize("", return_g=True)
            ext_ids = [a.id for a in r.rhs.ext]
            inc_nodes = Grammar.atom_neis(rhs_g, ext_ids) + ext_ids
            inc_subgraph = copy_graph(rhs_g, inc_nodes)
            r_wl_hash = nx.weisfeiler_lehman_graph_hash(inc_subgraph, node_attr="label")
            rule_lookup[(r.rhs.type, r_wl_hash)].append(i)
        folder = f"data/api_mol_hg/{time.time()}/"
        os.makedirs(folder, exist_ok=True)
        print(f"begin generate {os.path.abspath(folder)}")
        smi = None
        j = 0
        hg = HG(0 + 0, [])
        hg.add_hyperedge([], label="(S)")
        while True:
            choices = []
            choice_probs = []
            for i, e in enumerate(hg.E):
                if hrg.vocab[e.label] is None:  # terminal
                    continue
                hg_g = hg.visualize("", return_g=True)
                inc_nodes = Grammar.atom_neis(hg_g, e.nodes) + e.nodes
                hg_g_sub = copy_graph(hg_g, inc_nodes)
                # use its WL hash to quickly find possible rules
                wl_hash = nx.weisfeiler_lehman_graph_hash(hg_g_sub, node_attr="label")
                for ri in tqdm(
                    rule_lookup[(e.get_type(vocab), wl_hash)], "checking rules"
                ):
                    r = hrg.rules[ri]
                    # r.rhs.visualize(os.path.join(folder, 'test_rhs.png'))
                    # ensure e's node sequence <-> perm(r.rhs.ext)
                    # where:
                    # each node's label matches
                    # each node's adj atoms matches
                    if len(r.rhs.ext) == 0:
                        choices.append((i, r))
                        choice_probs.append(hrg.counts[ri])
                    else:
                        rhs_g = r.rhs.visualize("", return_g=True)
                        ext_ids = [a.id for a in r.rhs.ext]
                        inc_nodes = Grammar.atom_neis(rhs_g, ext_ids) + ext_ids
                        inc_subgraph = copy_graph(rhs_g, inc_nodes)
                        # match_info: hg to rule
                        gm = GraphMatcher(
                            hg_g_sub,
                            inc_subgraph,
                            node_match=lambda x, y: x["label"] == y["label"],
                        )
                        subs = list(gm.subgraph_isomorphisms_iter())
                        for sub in subs:
                            choices.append((i, r, sub))
                            choice_probs.append(hrg.counts[ri] / len(subs))
            if len(choices) == 0:
                break
            good = False
            while len(choices):
                rand_ind = np.random.RandomState().choice(
                    len(choices), p=np.array(choice_probs) / sum(choice_probs)
                )
                # for rand_ind
                edge, rule, *pargs = choices[rand_ind]
                choices.pop(rand_ind)
                choice_probs.pop(rand_ind)
                # rule.rhs.visualize(f"/home/msun415/rule_rhs_{j}.png")
                hg_cand = rule(deepcopy(hg), edge, *pargs)
                if VISUALIZE:
                    rule.rhs.visualize(f"{folder}/rule_{j}_rhs.png")
                    hg_cand.visualize(f"{folder}/pretest_{j}.png")
                try:
                    mol = hg_to_mol(hg_cand)
                    assert len(find_fragments(mol)) == 1
                except:
                    continue
                good = True
                j += 1
                if VISUALIZE:
                    hg_cand.visualize(f"{folder}/test_{j}.png")
                hg = hg_cand
                break
            if not good:
                break
            smi = Chem.MolToSmiles(mol)
            if VISUALIZE:
                draw_smiles(smi, path=f"{folder}/mol_{j}.png")
            with open(os.path.join(folder, "smiles.txt"), "a+") as f:
                f.write(smi)
            # early terminate?
            if shared_dict is not None and smi not in shared_dict:
                shared_dict[smi] = None
                logger.info(f"generated {smi}")
                break
        # if smi is not None:
        #     logger.info(f"generated {smi}")
        return smi

    @staticmethod
    def _single_mp_execute(func, arg, num_samples):
        smis = set()
        while True:
            smi = func(*arg)
            smis.add(smi)
            if len(smis) > num_samples:
                break
        return list(smis)

    @staticmethod
    def _multi_mp_execute(func, arg, num_samples, batch_size=5 * NUM_PROCS):
        with mp.Manager() as manager:
            shared_dict = manager.dict()
            smis = []
            while len(shared_dict) < num_samples:
                args = [arg + (shared_dict,) for _ in range(batch_size)]
                with mp.Pool(NUM_PROCS) as p:
                    p.starmap(func, tqdm(args, desc="submitting generation tasks"))
            smis = list(shared_dict)[:num_samples]
        return smis

    def generate(self, args):
        logger = create_logger(
            "global_logger",
            f"{wd}/data/{METHOD}_{DATASET}_{GRAMMAR}-{args.dataset}-{args.seed}.log",
        )
        globals()["logger"] = logger
        logger.info("===BEGIN GENERATION===")
        if NUM_PROCS == 1:
            smis = Grammar._single_mp_execute(
                Grammar._generate, (self.hrg, self.vocab), args.num_samples
            )
        else:
            smis = Grammar._multi_mp_execute(
                Grammar._generate, (self.hrg, self.vocab), args.num_samples
            )
        return smis


def chordal_mol_graph(smiles):
    logger = logging.getLogger("global_logger")
    mol = Chem.MolFromSmiles(smiles)
    cg = get_clique_graph(mol)
    # res, order = chordal._find_chordality_breaker(cg)
    # cg, _, chords = chordal.complete_to_chordal_graph(cg)
    try:
        cg = my_complete_to_chordal(cg, mol)
    except KeyError:
        logger.error(smiles)
        sys.exit(1)
    assert chordal.is_chordal(cg)
    return mol, cg
    # chordal.chordal_graph_cliques(cg)


def hg_to_mol(hg, verbose=False):
    """convert a hypergraph into Mol object

    Parameters
    ----------
    hg : HG

    Returns
    -------
    mol : Chem.RWMol
    """
    hg = deepcopy(hg)
    for i in range(len(hg.E) - 1, -1, -1):
        if hg.E[i].label[0] == "(":
            hg.E.pop(i)
    mol = Chem.RWMol()
    atom_dict = {}
    bond_set = set([])
    for index, each_edge in enumerate(hg.E):
        atom = Chem.Atom(each_edge.label)
        atom.SetNumExplicitHs(each_edge.kwargs["num_explicit_Hs"])
        atom.SetFormalCharge(each_edge.kwargs["formal_charge"])
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[each_edge.kwargs["chirality"]])
        atom_idx = mol.AddAtom(atom)
        atom_dict[index] = atom_idx

    adj_edges = {}
    for i in range(len(hg.E)):
        for n in hg.E[i].nodes:
            if n not in adj_edges:
                adj_edges[n] = []
            adj_edges[n].append(i)

    for each_node in hg.V:
        edge_1, edge_2 = adj_edges[each_node.id]
        if (edge_1, edge_2) not in bond_set:
            if each_node.label <= 3:
                num_bond = each_node.label
            elif each_node.label == 12:
                num_bond = 1
                # num_bond = each_node.label
            else:
                raise ValueError(f"too many bonds; {each_node.label}")
            _ = mol.AddBond(
                atom_dict[edge_1],
                atom_dict[edge_2],
                order=Chem.rdchem.BondType.values[num_bond],
            )
            bond_idx = mol.GetBondBetweenAtoms(
                atom_dict[edge_1], atom_dict[edge_2]
            ).GetIdx()

            # stereo
            mol.GetBondWithIdx(bond_idx).SetStereo(
                Chem.rdchem.BondStereo.values[each_node.kwargs["stereo"]]
            )
            bond_set.add((edge_1, edge_2))
            bond_set.add((edge_2, edge_1))
    mol.UpdatePropertyCache()
    mol = mol.GetMol()
    not_stereo_mol = deepcopy(mol)
    if Chem.MolFromSmiles(Chem.MolToSmiles(not_stereo_mol)) is None:
        raise RuntimeError("no valid molecule was obtained.")
    try:
        mol = set_stereo(mol)
        is_stereo = True
    except:
        import traceback

        traceback.logger.info_exc()
        is_stereo = False
    mol_tmp = deepcopy(mol)
    Chem.SetAromaticity(mol_tmp)
    if Chem.MolFromSmiles(Chem.MolToSmiles(mol_tmp)) is not None:
        mol = mol_tmp
    else:
        if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
            mol = not_stereo_mol
    mol.UpdatePropertyCache()
    if verbose:
        return mol, is_stereo
    else:
        return mol
