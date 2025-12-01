"""
Visualization utilities for molecular explanations.

Provides functions to draw molecules with atom-level importance coloring
using RDKit.
"""

import numpy as np
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_explained_molecule(
    smiles: str,
    atom_importances: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    colormap: str = "RdYlGn_r"
) -> None:
    """
    Plot a molecule with atoms colored by importance scores.
    
    Args:
        smiles: SMILES string
        atom_importances: Array of importance scores for each atom
                         (length should match number of atoms)
        title: Optional title for the plot
        figsize: Figure size (width, height)
        colormap: Matplotlib colormap name
    
    Example:
        >>> import numpy as np
        >>> mol = smiles_to_mol("CCO")
        >>> importances = np.array([0.1, 0.5, 0.9])
        >>> plot_explained_molecule("CCO", importances, title="Ethanol Explanation")
    """
    from src.featurization import smiles_to_mol
    
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    num_atoms = mol.GetNumAtoms()
    
    # Normalize importances to [0, 1]
    if len(atom_importances) != num_atoms:
        raise ValueError(
            f"Number of atom importances ({len(atom_importances)}) "
            f"does not match number of atoms ({num_atoms})"
        )
    
    normalized_importances = (atom_importances - atom_importances.min()) / (
        atom_importances.max() - atom_importances.min() + 1e-8
    )
    
    # Get colormap
    cmap = plt.cm.get_cmap(colormap)
    
    # Create atom colors
    atom_colors = {}
    for i in range(num_atoms):
        rgba = cmap(normalized_importances[i])
        atom_colors[i] = tuple(int(255 * x) for x in rgba[:3])
    
    # Draw molecule
    drawer = rdMolDraw2D.MolDraw2DCairo(figsize[0] * 100, figsize[1] * 100)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(num_atoms)),
        highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    
    # Convert to numpy array for matplotlib
    img_data = drawer.GetDrawingText()
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(img_data))
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_explained_grid(
    smiles_list: List[str],
    atom_importances_list: List[np.ndarray],
    titles: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize_per_mol: Tuple[int, int] = (4, 4),
    colormap: str = "RdYlGn_r"
) -> None:
    """
    Plot multiple molecules with explanations in a grid layout.
    
    Args:
        smiles_list: List of SMILES strings
        atom_importances_list: List of atom importance arrays
        titles: Optional list of titles for each molecule
        n_cols: Number of columns in the grid
        figsize_per_mol: Size of each molecule subplot
        colormap: Matplotlib colormap name
    """
    from src.featurization import smiles_to_mol
    
    n_mols = len(smiles_list)
    n_rows = (n_mols + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_mol[0] * n_cols, figsize_per_mol[1] * n_rows)
    )
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (smiles, atom_importances) in enumerate(
        zip(smiles_list, atom_importances_list)
    ):
        mol = smiles_to_mol(smiles)
        if mol is None:
            continue
        
        num_atoms = mol.GetNumAtoms()
        if len(atom_importances) != num_atoms:
            continue
        
        # Normalize importances
        normalized_importances = (atom_importances - atom_importances.min()) / (
            atom_importances.max() - atom_importances.min() + 1e-8
        )
        
        # Get colormap
        cmap = plt.cm.get_cmap(colormap)
        
        # Create atom colors
        atom_colors = {}
        for i in range(num_atoms):
            rgba = cmap(normalized_importances[i])
            atom_colors[i] = tuple(int(255 * x) for x in rgba[:3])
        
        # Draw molecule
        drawer = rdMolDraw2D.MolDraw2DCairo(
            figsize_per_mol[0] * 100, figsize_per_mol[1] * 100
        )
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(range(num_atoms)),
            highlightAtomColors=atom_colors
        )
        drawer.FinishDrawing()
        
        # Convert to numpy array
        img_data = drawer.GetDrawingText()
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(img_data))
        
        # Plot
        axes[idx].imshow(img)
        axes[idx].axis("off")
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx], fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_mols, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    plt.show()


def map_fingerprint_to_atoms(
    smiles: str,
    fingerprint_attributions: np.ndarray,
    radius: int = 2,
    n_bits: int = 2048
) -> np.ndarray:
    """
    Map fingerprint-level attributions to atom-level importances.
    
    This is an approximation since fingerprints are circular substructures,
    not directly atom-based. Uses Morgan fingerprint mapping.
    
    Args:
        smiles: SMILES string
        fingerprint_attributions: Attribution scores for each fingerprint bit
        radius: Radius used for Morgan fingerprint
        n_bits: Number of bits in fingerprint
    
    Returns:
        Atom-level importance scores
    """
    from src.featurization import smiles_to_mol
    from rdkit.Chem import AllChem
    
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.array([])
    
    num_atoms = mol.GetNumAtoms()
    atom_importances = np.zeros(num_atoms)
    
    # Get Morgan fingerprint info
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, bitInfo=info
    )
    
    # Map each fingerprint bit to atoms
    # bitInfo format: {bit_index: ((radius, atom_index), ...)}
    for bit_idx, atom_envs in info.items():
        if bit_idx < len(fingerprint_attributions):
            importance = fingerprint_attributions[bit_idx]
            # Distribute importance among all atoms in this fingerprint bit
            # Each element in atom_envs is (radius, atom_index)
            for radius, atom_idx in atom_envs:
                if atom_idx < num_atoms:
                    atom_importances[atom_idx] += importance
    
    # Normalize
    if atom_importances.max() > 0:
        atom_importances = atom_importances / atom_importances.max()
    
    return atom_importances

