"""
Graph data preparation for PyTorch Geometric models.

Converts SMILES strings to torch_geometric.data.Data objects with rich
node (atom) and edge (bond) features for molecular property prediction.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors

from src.featurization import smiles_to_mol


def get_atom_features(atom: Chem.Atom, mol: Chem.Mol) -> np.ndarray:
    """
    Extract atom (node) features from an RDKit atom object.
    
    Features include:
    - Atomic number (one-hot encoded for common elements)
    - Formal charge
    - Hybridization (sp, sp2, sp3, etc.)
    - Chirality
    - Ring membership
    - Aromaticity
    - Number of heavy atom neighbors
    - Number of hydrogen neighbors
    - Valence minus attached hydrogens
    
    Args:
        atom: RDKit atom object
        mol: RDKit molecule object (for context)
    
    Returns:
        Feature vector as numpy array of shape (n_features,)
    """
    features = []
    
    # Atomic number (one-hot for common elements: C, N, O, F, P, S, Cl, Br, I)
    # Common elements in drug-like molecules
    atom_num = atom.GetAtomicNum()
    common_elements = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
    atom_onehot = [1 if atom_num == elem else 0 for elem in common_elements]
    features.extend(atom_onehot)
    # Add a generic "other" element feature
    features.append(1 if atom_num not in common_elements else 0)
    
    # Formal charge (normalized)
    features.append(float(atom.GetFormalCharge()))
    
    # Hybridization (one-hot)
    hybrid = atom.GetHybridization()
    hybrid_types = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2
    ]
    hybrid_onehot = [1 if hybrid == h else 0 for h in hybrid_types]
    features.extend(hybrid_onehot)
    
    # Chirality (one-hot)
    chiral = atom.GetChiralTag()
    chiral_types = [
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    ]
    chiral_onehot = [1 if chiral == c else 0 for c in chiral_types]
    features.extend(chiral_onehot)
    
    # Ring membership
    features.append(float(atom.IsInRing()))
    
    # Aromaticity
    features.append(float(atom.GetIsAromatic()))
    
    # Number of heavy atom neighbors (excluding hydrogens)
    features.append(float(len([n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1])))
    
    # Number of hydrogen neighbors
    features.append(float(atom.GetTotalNumHs(includeNeighbors=True)))
    
    # Valence minus attached hydrogens
    features.append(float(atom.GetTotalValence() - atom.GetTotalNumHs()))
    
    # Degree (number of bonds)
    features.append(float(atom.GetDegree()))
    
    return np.array(features, dtype=np.float32)


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """
    Extract bond (edge) features from an RDKit bond object.
    
    Features include:
    - Bond type (single, double, triple, aromatic)
    - Bond direction (forward, backward, both, none)
    - Ring membership (is bond in a ring)
    - Conjugation (is bond conjugated)
    - Stereochemistry (one-hot)
    
    Args:
        bond: RDKit bond object
    
    Returns:
        Feature vector as numpy array of shape (n_features,)
    """
    features = []
    
    # Bond type (one-hot)
    bond_type = bond.GetBondType()
    bond_types = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC
    ]
    bond_type_onehot = [1 if bond_type == bt else 0 for bt in bond_types]
    features.extend(bond_type_onehot)
    
    # Bond direction (one-hot)
    bond_dir = bond.GetBondDir()
    bond_dirs = [
        Chem.BondDir.NONE,
        Chem.BondDir.BEGINWEDGE,
        Chem.BondDir.BEGINDASH,
        Chem.BondDir.ENDDOWNRIGHT,
        Chem.BondDir.ENDUPRIGHT
    ]
    bond_dir_onehot = [1 if bond_dir == bd else 0 for bd in bond_dirs]
    features.extend(bond_dir_onehot)
    
    # Ring membership
    features.append(float(bond.IsInRing()))
    
    # Conjugation
    features.append(float(bond.GetIsConjugated()))
    
    # Stereochemistry (one-hot)
    stereo = bond.GetStereo()
    stereo_types = [
        Chem.BondStereo.STEREONONE,
        Chem.BondStereo.STEREOANY,
        Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOCIS,
        Chem.BondStereo.STEREOTRANS
    ]
    stereo_onehot = [1 if stereo == st else 0 for st in stereo_types]
    features.extend(stereo_onehot)
    
    return np.array(features, dtype=np.float32)


def smiles_to_pyg_data(
    smiles: str,
    label: Optional[Union[float, List[float], np.ndarray]] = None,
) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Supports both single-task (scalar label) and multi-task (array label)
    targets, enabling use with ClinTox and Tox21 without code changes.

    Args:
        smiles: SMILES string representation of the molecule.
        label:  Optional label for supervised learning.
                - float / int  → single-task, stored as y of shape (1,)
                - list / array → multi-task, stored as y of shape (num_tasks,)
                  NaN values are preserved for masked multi-task loss.

    Returns:
        torch_geometric.data.Data object with:
        - x:         Node (atom) features, shape (num_nodes, 25)
        - edge_index: Edge connectivity COO format, shape (2, num_edges)
        - edge_attr:  Edge (bond) features, shape (num_edges, 17)
        - y:          Graph-level label(s), shape (1,) or (num_tasks,)
        - smiles:     Original SMILES string

    Example:
        >>> data = smiles_to_pyg_data("CCO", label=0.0)          # single-task
        >>> data = smiles_to_pyg_data("CCO", label=[1,0,np.nan]) # multi-task
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    # RDKit 2025: ensure implicit valences are computed before accessing H counts.
    # Required when MolFromSmiles falls back to sanitize=False, or in edge cases
    # where calcImplicitValence was not called during sanitization.
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    # Get atom features
    atom_features = [get_atom_features(atom, mol) for atom in mol.GetAtoms()]
    if len(atom_features) == 0:
        return None
    
    x = torch.tensor(np.array(atom_features), dtype=torch.float32)
    
    # Build edge index and edge attributes
    edge_indices = []
    edge_attrs = []
    
    # Add both directions for undirected graphs (common in molecular graphs)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_features = get_bond_features(bond)
        
        # Add edge in both directions (undirected graph)
        edge_indices.append([i, j])
        edge_attrs.append(bond_features)
        edge_indices.append([j, i])
        edge_attrs.append(bond_features)  # Same features for reverse direction
    
    if len(edge_indices) == 0:
        # Handle isolated atoms (no bonds)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # Use expected edge feature dimension (17) for empty edge_attr
        edge_attr = torch.empty((0, 17), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Add label if provided
    if label is not None:
        if isinstance(label, (list, np.ndarray)):
            # Multi-task: shape (1, num_tasks) so PyG batching yields (batch_size, num_tasks).
            # NaN values are preserved for MaskedMultiTaskLoss.
            data.y = torch.tensor(
                np.array(label, dtype=np.float32), dtype=torch.float32
            ).unsqueeze(0)   # (1, num_tasks)
        else:
            # Single-task: scalar → shape (1,)
            data.y = torch.tensor([float(label)], dtype=torch.float32)
    
    # Store SMILES string as attribute (useful for debugging/visualization)
    data.smiles = smiles
    
    return data


def smiles_list_to_pyg_dataset(
    smiles_list: List[str],
    labels: Optional[Union[List[float], List[List[float]], np.ndarray]] = None,
) -> List[Data]:
    """
    Convert a list of SMILES strings to a list of PyG Data objects.

    Handles both single-task (list of floats) and multi-task labels
    (2-D array of shape (n_compounds, n_tasks)), transparently passing
    each row to smiles_to_pyg_data.

    Args:
        smiles_list: List of SMILES strings.
        labels:      Optional labels.
                     - List[float] for single-task (ClinTox).
                     - np.ndarray of shape (N, T) for multi-task (Tox21).

    Returns:
        List of torch_geometric.data.Data objects (failed SMILES skipped).

    Example:
        >>> dataset = smiles_list_to_pyg_dataset(["CCO", "CCN"], [0.0, 1.0])
        >>> # Multi-task:
        >>> dataset = smiles_list_to_pyg_dataset(smiles, labels_array)
    """
    if labels is None:
        labels = [None] * len(smiles_list)

    dataset = []
    for smiles, label in zip(smiles_list, labels):
        data = smiles_to_pyg_data(smiles, label=label)
        if data is not None:
            dataset.append(data)

    return dataset


def get_feature_dims() -> Tuple[int, int]:
    """
    Get the dimensions of node and edge features.
    
    Returns:
        Tuple of (num_node_features, num_edge_features)
    """
    # Node features:
    # - Atomic number one-hot: 10 (9 common + 1 other)
    # - Formal charge: 1
    # - Hybridization one-hot: 5
    # - Chirality one-hot: 3
    # - Ring membership: 1
    # - Aromaticity: 1
    # - Heavy atom neighbors: 1
    # - Hydrogen neighbors: 1
    # - Valence minus H: 1
    # - Degree: 1
    # Total: 25
    
    # Edge features:
    # - Bond type one-hot: 4
    # - Bond direction one-hot: 5
    # - Ring membership: 1
    # - Conjugation: 1
    # - Stereochemistry one-hot: 6
    # Total: 17
    
    return 25, 17

