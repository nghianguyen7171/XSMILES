"""
Molecular featurization utilities.

Provides functions to convert SMILES strings to molecular representations:
1. Fingerprint mode: Morgan fingerprints (ECFP-like) for baseline models
2. Graph mode: Graph representations compatible with torch-molecule
"""

from typing import List, Union, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_mol(smiles: str, sanitize: bool = True) -> Optional[Chem.Mol]:
    """
    Convert SMILES string to RDKit molecule object with robust error handling.
    
    Args:
        smiles: SMILES string
        sanitize: Whether to sanitize the molecule
    
    Returns:
        RDKit molecule object or None if parsing fails
    
    Example:
        >>> mol = smiles_to_mol("CCO")
        >>> print(mol.GetNumAtoms())
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None and sanitize:
            # Try without sanitization as fallback
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        return mol
    except Exception:
        return None


def featurize_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048
) -> np.ndarray:
    """
    Generate Morgan fingerprint (ECFP-like) from SMILES.
    
    Args:
        smiles: SMILES string
        radius: Radius for Morgan fingerprint (typically 2 for ECFP4)
        n_bits: Number of bits in fingerprint vector
    
    Returns:
        Binary fingerprint vector as numpy array
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def featurize_batch(
    smiles_list: List[str],
    mode: str = "fingerprint",
    **kwargs
) -> Union[np.ndarray, List]:
    """
    Featurize a batch of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        mode: Featurization mode ('fingerprint' or 'graph')
        **kwargs: Additional arguments passed to featurization functions
            - For fingerprint: radius, n_bits
            - For graph: passed to graph featurization
    
    Returns:
        - For 'fingerprint': numpy array of shape (batch_size, n_bits)
        - For 'graph': List of graph representations or SMILES strings
                      (format depends on torch-molecule API requirements)
    
    Example:
        >>> smiles = ["CCO", "CCN"]
        >>> fps = featurize_batch(smiles, mode="fingerprint")
        >>> print(fps.shape)
    """
    if mode == "fingerprint":
        radius = kwargs.get("radius", 2)
        n_bits = kwargs.get("n_bits", 2048)
        fingerprints = [
            featurize_fingerprint(s, radius=radius, n_bits=n_bits)
            for s in smiles_list
        ]
        return np.stack(fingerprints)
    
    elif mode == "graph":
        # For torch-molecule, may return SMILES strings directly
        # or convert to graph tensors based on library API
        # Returning SMILES list as placeholder - will be adapted based on API
        return smiles_list
    
    else:
        raise ValueError(f"Unknown featurization mode: {mode}")


def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string is valid.
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        True if valid, False otherwise
    """
    return smiles_to_mol(smiles) is not None

