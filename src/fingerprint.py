"""
ECFP fingerprint featurization and SHAP atom-mapping utilities.

Provides tools to:
- Convert SMILES to Morgan (ECFP) fingerprints
- Map per-bit SHAP values back to the atoms that generated each bit
- Visualize atom-level SHAP contributions on 2-D molecule drawings
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem import AllChem


# ── Featurization ─────────────────────────────────────────────────────────────

def smiles_to_ecfp(
    smiles_list: List[str],
    radius: int = 2,
    nbits: int = 2048,
) -> Tuple[np.ndarray, List[int]]:
    """
    Convert SMILES strings to ECFP bit-vector fingerprints.

    Args:
        smiles_list: List of SMILES strings.
        radius:      Morgan radius (2 → ECFP4, 3 → ECFP6).
        nbits:       Fingerprint length in bits.

    Returns:
        X:         Float32 array of shape (n_valid, nbits).
        valid_idx: Indices into smiles_list that parsed successfully.
    """
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(np.array(fp, dtype=np.float32))
            valid_idx.append(i)
    return np.array(fps), valid_idx


# ── SHAP ↔ atom mapping ───────────────────────────────────────────────────────

def get_bit_info(
    mol: Chem.Mol,
    radius: int = 2,
    nbits: int = 2048,
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Return the bit → atom-environment mapping for a molecule's ECFP fingerprint.

    Args:
        mol:    RDKit molecule.
        radius: Morgan radius used for fingerprinting.
        nbits:  Fingerprint bit length.

    Returns:
        bit_info: {bit_id: [(center_atom_idx, radius), ...]}
    """
    bit_info: Dict = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=bit_info)
    return bit_info


def shap_bits_to_atom_weights(
    mol: Chem.Mol,
    shap_values: np.ndarray,
    radius: int = 2,
    nbits: int = 2048,
) -> np.ndarray:
    """
    Map per-bit SHAP values to per-atom signed weights.

    For each fingerprint bit set on the molecule, the SHAP value for that
    bit is accumulated on the center atom that generated the environment.
    Atoms participating in multiple bits receive the sum of contributions.

    Args:
        mol:         RDKit molecule.
        shap_values: SHAP values per bit, shape (nbits,).
        radius:      Morgan radius.
        nbits:       Fingerprint bit length.

    Returns:
        atom_weights: Array of shape (num_atoms,).
                      Positive → toward toxic prediction.
                      Negative → toward non-toxic prediction.
    """
    bit_info = get_bit_info(mol, radius=radius, nbits=nbits)
    atom_weights = np.zeros(mol.GetNumAtoms())
    for bit_id, env_list in bit_info.items():
        if bit_id < nbits and shap_values[bit_id] != 0:
            for center_atom, _ in env_list:
                atom_weights[center_atom] += shap_values[bit_id]
    return atom_weights


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_molecule_shap(
    smiles: str,
    shap_values: np.ndarray,
    task_name: str = "",
    radius: int = 2,
    nbits: int = 2048,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (4, 3),
) -> Optional[plt.Figure]:
    """
    Draw a molecule with atoms colored by their SHAP contribution.

    Red  → positive SHAP → pushes prediction toward *toxic*.
    Blue → negative SHAP → pushes prediction toward *non-toxic*.

    Uses RDKit SimilarityMaps under the hood; the coloring is proportional
    to the atom's accumulated SHAP weight.

    Args:
        smiles:      SMILES string of the molecule.
        shap_values: Per-bit SHAP values, shape (nbits,).
        task_name:   String label shown as the plot title.
        radius:      Morgan radius used during training.
        nbits:       Fingerprint bit length used during training.
        ax:          Existing matplotlib Axes to draw into (optional).
        figsize:     Figure size when ax is None.

    Returns:
        matplotlib Figure (only when ax is None), else None.
    """
    from rdkit.Chem.Draw import SimilarityMaps

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_weights = shap_bits_to_atom_weights(mol, shap_values, radius=radius, nbits=nbits)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        list(atom_weights),
        colorMap=cm.RdBu_r,
        alpha=0.5,
        ax=ax,
    )
    ax.set_title(task_name, fontsize=9, pad=3)
    ax.axis("off")

    return fig if return_fig else None


def draw_molecule_shap_grid(
    smiles_list: List[str],
    shap_matrix: np.ndarray,
    task_names: List[str],
    title: str = "",
    radius: int = 2,
    nbits: int = 2048,
    mol_size: Tuple[int, int] = (4, 3),
) -> plt.Figure:
    """
    Draw a grid of molecule SHAP maps — one column per task.

    Args:
        smiles_list:  List of N SMILES strings (rows).
        shap_matrix:  Shape (N, nbits) — SHAP values per molecule.
        task_names:   List of T task labels (columns), length T ≤ N.
        title:        Overall figure title.
        radius, nbits: Fingerprint parameters.
        mol_size:     (width, height) per cell in inches.

    Returns:
        matplotlib Figure.
    """
    n_mols = len(smiles_list)
    n_tasks = len(task_names)
    fig, axes = plt.subplots(
        n_mols, n_tasks,
        figsize=(mol_size[0] * n_tasks, mol_size[1] * n_mols),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    for row, smi in enumerate(smiles_list):
        for col, task in enumerate(task_names):
            draw_molecule_shap(
                smi, shap_matrix[row],
                task_name=task if row == 0 else "",
                radius=radius, nbits=nbits,
                ax=axes[row, col],
            )
    plt.tight_layout()
    return fig
