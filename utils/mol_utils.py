"""
utils/mol_utils.py
TopU-LBVS Setting 1 - Molecular graph builder for GNN models.

Converts SMILES strings to PyTorch Geometric Data objects.

Node features (9-dim) per atom:
    0: atomic number         (int, capped at 118)
    1: degree                (number of bonds)
    2: formal charge         (int, can be negative)
    3: total num hydrogens   (int)
    4: num radical electrons (int)
    5: hybridisation         (0=other,1=SP,2=SP2,3=SP3,4=SP3D,5=SP3D2)
    6: aromaticity           (0/1)
    7: ring membership       (0/1)
    8: chirality             (0=none,1=CW,2=CCW)

Edge features (3-dim) per bond (duplicated for both directions):
    0: bond type             (0=other,1=single,2=double,3=triple,4=aromatic)
    1: conjugation           (0/1)
    2: ring membership       (0/1)

Graph is undirected -each bond appears as two directed edges (u -> v, v -> u)
with identical edge features.

Functions
---------
smiles_to_graph(smiles)
    Convert a single SMILES string to a PyG Data object.
    Returns None if SMILES is invalid.

smiles_list_to_graphs(smiles_list)
    Convert a list of SMILES to a list of PyG Data objects.
    Invalid SMILES produce None entries (caller must handle).
"""

from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
from torch_geometric.data import Data

# Uncomment when running full benchmark across all targets to suppress RDKit terminal noise
# RDLogger.DisableLog("rdApp.*")

# Uncomment if needed — restricts * imports to public functions only
# __all__ = [
#   "smiles_to_graph",
#   "smiles_list_to_graphs",
#   "NODE_FEAT_DIM",
#    "EDGE_FEAT_DIM",
#]

# ---------------------------------------------------------------------------
# Feature extraction constants
# ---------------------------------------------------------------------------

_HYBRIDISATION_MAP = {
    rdchem.HybridizationType.SP:    1,
    rdchem.HybridizationType.SP2:   2,
    rdchem.HybridizationType.SP3:   3,
    rdchem.HybridizationType.SP3D:  4,
    rdchem.HybridizationType.SP3D2: 5,
}

_CHIRALITY_MAP = {
    rdchem.ChiralType.CHI_UNSPECIFIED:     0,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW:  1,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}

_BOND_TYPE_MAP = {
    rdchem.BondType.SINGLE:   1,
    rdchem.BondType.DOUBLE:   2,
    rdchem.BondType.TRIPLE:   3,
    rdchem.BondType.AROMATIC: 4,
}

# Dimensions — used by GNN models to set input sizes
NODE_FEAT_DIM = 9
EDGE_FEAT_DIM = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atom_features(atom: rdchem.Atom) -> List[int]:
    """
    Extract 9-dim feature vector for a single atom.

    Returns
    -------
    List of 9 ints - one per feature as documented above.
    """
    return [
        min(atom.GetAtomicNum(), 118),                          # 0: atomic num  (118 = Oganesson, max periodic table element)
        atom.GetDegree(),                                       # 1: degree
        atom.GetFormalCharge(),                                 # 2: formal charge
        atom.GetTotalNumHs(),                                   # 3: num H
        atom.GetNumRadicalElectrons(),                          # 4: radical e-
        _HYBRIDISATION_MAP.get(atom.GetHybridization(), 0),    # 5: hybridisation
        int(atom.GetIsAromatic()),                              # 6: aromaticity
        int(atom.IsInRing()),                                   # 7: ring
        _CHIRALITY_MAP.get(atom.GetChiralTag(), 0),            # 8: chirality
    ]


def _bond_features(bond: rdchem.Bond) -> List[int]:
    """
    Extract 3-dim feature vector for a single bond.

    Returns
    -------
    List of 3 ints - one per feature as documented above.
    """
    return [
        _BOND_TYPE_MAP.get(bond.GetBondType(), 0),  # 0: bond type
        int(bond.GetIsConjugated()),                 # 1: conjugation
        int(bond.IsInRing()),                        # 2: ring membership
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert a SMILES string to a PyG Data object.

    Parameters
    ----------
    smiles : str - SMILES string for one molecule

    Returns
    -------
    torch_geometric.data.Data with fields:
        x          : torch.float32, shape (n_atoms, 9)
        edge_index : torch.long,    shape (2, 2 * n_bonds)
        edge_attr  : torch.float32, shape (2 * n_bonds, 3)

    Returns None if:
        - SMILES is invalid (RDKit cannot parse)
        - Molecule has zero atoms
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # -- Node features --------------------------------------------------------
    node_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float32)   # (n_atoms, 9)

    # -- Edge index and edge features -----------------------------------------
    # Each bond ? two directed edges (u?v) and (v?u) with same features
    src_list  = []
    dst_list  = []
    edge_list = []

    for bond in mol.GetBonds():
        u  = bond.GetBeginAtomIdx()
        v  = bond.GetEndAtomIdx()
        ef = _bond_features(bond)

        src_list  += [u, v]
        dst_list  += [v, u]
        edge_list += [ef, ef]   # same features for both directions

    if len(src_list) > 0:
        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )                                                    # (2, 2*n_bonds)
        edge_attr = torch.tensor(
            edge_list, dtype=torch.float32
        )                                                    # (2*n_bonds, 3)
    else:
        # Isolated atom — no bonds (e.g. [Na+], [He])
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_list_to_graphs(smiles_list: np.ndarray) -> List[Optional[Data]]:
    """
    Convert an array of SMILES strings to a list of PyG Data objects.

    Parameters
    ----------
    smiles_list : np.ndarray of str, shape (n,)

    Returns
    -------
    List of length n.
    Each entry is either a PyG Data object or None (invalid SMILES).

    Note: GNN models must handle None entries - typically by skipping
    or substituting a zero-feature placeholder graph.
    """
    return [smiles_to_graph(smi) for smi in smiles_list]