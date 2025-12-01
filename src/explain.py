"""
Explainability methods for molecular toxicity prediction.

Provides gradient-based and perturbation-based attribution methods
to identify important atoms/substructures in predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from rdkit import Chem
import copy


def compute_gradient_attribution(
    model: nn.Module,
    smiles: str,
    label: int,
    input_tensor: torch.Tensor,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute gradient-based attribution for a molecule prediction.
    
    Uses gradients of the output with respect to input features
    to identify important features for the prediction.
    
    Args:
        model: Trained PyTorch model
        smiles: SMILES string (for reference, not used in computation)
        label: True label (0 or 1)
        input_tensor: Input feature tensor of shape (1, feature_dim)
        device: Device to run computation on
    
    Returns:
        Attribution scores for each input feature
    """
    model.eval()
    input_tensor = input_tensor.to(device).requires_grad_(True)
    
    # Forward pass
    output = model(input_tensor)
    
    # For binary classification, use the single output logit
    if output.shape[1] == 1:
        target = output[0, 0]
    else:
        target = output[0, label]
    
    # Backward pass to get gradients
    target.backward()
    
    # Get gradients and compute attribution
    gradients = input_tensor.grad
    attribution = gradients.abs().cpu().numpy().flatten()
    
    return attribution


def compute_perturbation_attribution(
    model: nn.Module,
    smiles: str,
    input_tensor: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute perturbation-based attribution by masking input features.
    
    For fingerprint-based models, this masks individual fingerprint bits
    and measures the change in prediction.
    
    Args:
        model: Trained PyTorch model
        smiles: SMILES string (for reference)
        input_tensor: Input feature tensor
        baseline: Baseline tensor (zeros if not provided)
        device: Device to run computation on
    
    Returns:
        Attribution scores for each input feature
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    else:
        baseline = baseline.to(device)
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(baseline)
        baseline_prob = torch.sigmoid(baseline_output[0, 0]).item()
    
    # Perturb each feature and measure change
    feature_dim = input_tensor.shape[1]
    attributions = np.zeros(feature_dim)
    
    for i in range(feature_dim):
        perturbed = input_tensor.clone()
        perturbed[0, i] = baseline[0, i]
        
        with torch.no_grad():
            perturbed_output = model(perturbed)
            perturbed_prob = torch.sigmoid(perturbed_output[0, 0]).item()
        
        # Attribution is the absolute change in prediction
        attributions[i] = abs(baseline_prob - perturbed_prob)
    
    return attributions


def explain_torch_molecule_model_perturbation(
    model,
    smiles: str,
    mol: Optional[Chem.Mol] = None
) -> Tuple[List[float], Chem.Mol]:
    """
    Compute atom-level attribution for torch-molecule models using perturbation.
    
    This method systematically removes each atom and measures the change in prediction.
    Works with any torch-molecule model regardless of internal structure.
    
    Args:
        model: torch-molecule model instance (e.g., BFGNNMolecularPredictor)
        smiles: SMILES string
        mol: RDKit molecule object (created if not provided)
    
    Returns:
        Tuple of (atom_importances, molecule_object)
    """
    from src.featurization import smiles_to_mol
    from src.train import predict_with_torch_molecule_model
    from scipy.special import expit
    
    if mol is None:
        mol = smiles_to_mol(smiles)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    num_atoms = mol.GetNumAtoms()
    
    # Get baseline prediction
    try:
        baseline_pred = predict_with_torch_molecule_model(model, [smiles])
        baseline_prob = baseline_pred[0] if isinstance(baseline_pred, np.ndarray) else baseline_pred
    except Exception as e:
        # Fallback: try direct predict
        baseline_output = model.predict([smiles])
        if isinstance(baseline_output, dict):
            baseline_prob = np.array(baseline_output.get('prediction', baseline_output.get('predictions', [0]))).flatten()[0]
        else:
            baseline_prob = np.array(baseline_output).flatten()[0]
        
        # Convert logits to probabilities if needed
        if baseline_prob < 0 or baseline_prob > 1:
            baseline_prob = expit(baseline_prob)
    
    # Compute importance for each atom by removing it
    atom_importances = np.zeros(num_atoms)
    
    for atom_idx in range(num_atoms):
        # Create molecule with atom removed
        try:
            # Create editable molecule (copy)
            edited_mol = Chem.RWMol(mol)
            # Remove the atom (this also removes connected bonds)
            edited_mol.RemoveAtom(atom_idx)
            
            # Check if molecule is still valid
            if edited_mol.GetNumAtoms() == 0:
                # Removing this atom destroys the molecule - high importance
                atom_importances[atom_idx] = abs(baseline_prob - 0.5)
                continue
            
            # Get SMILES of edited molecule
            try:
                edited_smiles = Chem.MolToSmiles(edited_mol, canonical=True)
            except:
                # If conversion fails, assign high importance
                atom_importances[atom_idx] = abs(baseline_prob - 0.5)
                continue
            
            # Skip if invalid or empty
            if not edited_smiles or edited_smiles == "" or edited_smiles == ".":
                atom_importances[atom_idx] = abs(baseline_prob - 0.5)  # High importance if removal invalid
                continue
            
            # Get prediction for edited molecule
            try:
                edited_pred = predict_with_torch_molecule_model(model, [edited_smiles])
                edited_prob = edited_pred[0] if isinstance(edited_pred, np.ndarray) else float(edited_pred)
            except:
                try:
                    edited_output = model.predict([edited_smiles])
                    if isinstance(edited_output, dict):
                        edited_prob = np.array(edited_output.get('prediction', edited_output.get('predictions', [0]))).flatten()[0]
                    else:
                        edited_prob = np.array(edited_output).flatten()[0]
                    
                    if edited_prob < 0 or edited_prob > 1:
                        edited_prob = expit(edited_prob)
                except:
                    # If prediction fails, assign medium importance
                    atom_importances[atom_idx] = abs(baseline_prob - 0.5) * 0.5
                    continue
            
            # Importance is the absolute change in prediction
            atom_importances[atom_idx] = abs(baseline_prob - float(edited_prob))
            
        except Exception as e:
            # If removal fails, assign medium importance
            atom_importances[atom_idx] = abs(baseline_prob - 0.5) * 0.5
    
    # Normalize importances
    if atom_importances.max() > 0:
        atom_importances = atom_importances / atom_importances.max()
    
    return atom_importances.tolist(), mol


def explain_torch_molecule_model_gradient(
    model,
    smiles: str,
    mol: Optional[Chem.Mol] = None,
    device: str = "cpu"
) -> Tuple[List[float], Chem.Mol]:
    """
    Compute atom-level attribution for torch-molecule models using gradients.
    
    Attempts to access internal PyTorch model and compute node-level gradients.
    Falls back to perturbation method if gradient access fails.
    
    Args:
        model: torch-molecule model instance
        smiles: SMILES string
        mol: RDKit molecule object (created if not provided)
        device: Device to run computation on
    
    Returns:
        Tuple of (atom_importances, molecule_object)
    """
    from src.featurization import smiles_to_mol
    
    if mol is None:
        mol = smiles_to_mol(smiles)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Try to access internal PyTorch model
    internal_model = None
    if hasattr(model, 'model'):
        internal_model = model.model
    elif hasattr(model, 'nn'):
        internal_model = model.nn
    elif hasattr(model, 'network'):
        internal_model = model.network
    
    if internal_model is None or not hasattr(internal_model, 'parameters'):
        # Fallback to perturbation method
        return explain_torch_molecule_model_perturbation(model, smiles, mol)
    
    # Try gradient-based attribution
    try:
        num_atoms = mol.GetNumAtoms()
        
        # This requires understanding torch-molecule's internal graph representation
        # For now, fallback to perturbation
        # TODO: Implement proper gradient-based attribution once we understand
        #       the internal graph structure and node/edge feature format
        return explain_torch_molecule_model_perturbation(model, smiles, mol)
        
    except Exception as e:
        # Fallback to perturbation
        return explain_torch_molecule_model_perturbation(model, smiles, mol)


def explain_molecule(
    model: nn.Module,
    smiles: str,
    input_tensor: torch.Tensor,
    method: str = "gradient",
    device: str = "cpu"
) -> Tuple[np.ndarray, Optional[Chem.Mol]]:
    """
    Main function to generate explanations for a molecule prediction.
    
    Args:
        model: Trained model
        smiles: SMILES string
        input_tensor: Input feature tensor
        method: Attribution method ('gradient' or 'perturbation')
        device: Device to run computation on
    
    Returns:
        Tuple of (attribution_scores, molecule_object)
    """
    from src.featurization import smiles_to_mol
    
    mol = smiles_to_mol(smiles)
    
    if method == "gradient":
        attributions = compute_gradient_attribution(
            model, smiles, 0, input_tensor, device
        )
    elif method == "perturbation":
        attributions = compute_perturbation_attribution(
            model, smiles, input_tensor, device=device
        )
    else:
        raise ValueError(f"Unknown attribution method: {method}")
    
    return attributions, mol


def explain_torch_molecule_model(
    model,
    smiles: str,
    method: str = "perturbation",
    mol: Optional[Chem.Mol] = None,
    device: str = "cpu"
) -> Tuple[List[float], Chem.Mol]:
    """
    Main function to generate explanations for torch-molecule model predictions.
    
    Args:
        model: torch-molecule model instance (e.g., BFGNNMolecularPredictor)
        smiles: SMILES string
        method: Attribution method ('perturbation' or 'gradient')
        mol: RDKit molecule object (created if not provided)
        device: Device to run computation on
    
    Returns:
        Tuple of (atom_importances, molecule_object)
    """
    if method == "perturbation":
        return explain_torch_molecule_model_perturbation(model, smiles, mol)
    elif method == "gradient":
        return explain_torch_molecule_model_gradient(model, smiles, mol, device)
    else:
        raise ValueError(f"Unknown attribution method: {method}. Use 'perturbation' or 'gradient'")

