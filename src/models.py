"""
Model definitions for toxicity prediction.

This module defines two distinct types of models:

1. **Self-designed MLP Model** (FingerprintMLP):
   - Framework: Pure PyTorch
   - Input: Fixed-length fingerprint vectors (Morgan fingerprints)
   - Architecture: Custom multi-layer perceptron
   - Training: Custom training loop (see src.train.train_baseline_model)
   - Usage: Fingerprint-based molecular property prediction
   
2. **Imported torch-molecule Models**:
   - Framework: torch-molecule library (sklearn-style API)
   - Input: SMILES strings (automatic graph conversion)
   - Architecture: Pre-defined GNN architectures (BFGNN, GRIN, etc.)
   - Training: sklearn-style fit() method with hyperparameter optimization
   - Usage: Graph neural network-based molecular property prediction
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class FingerprintMLP(nn.Module):
    """
    Baseline MLP model for molecular property prediction from fingerprints.
    
    Simple feedforward network suitable for fixed-length fingerprint vectors.
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: list = [512, 256, 128],
        num_tasks: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Input fingerprint dimension
            hidden_dims: List of hidden layer dimensions
            num_tasks: Number of output tasks (1 for binary classification)
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, num_tasks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Logits of shape (batch_size, num_tasks)
        """
        x = self.features(x)
        return self.output(x)


def create_baseline_model(
    input_dim: int = 2048,
    hidden_dims: list = [512, 256, 128],
    num_tasks: int = 1,
    dropout: float = 0.2
) -> FingerprintMLP:
    """
    Factory function to create baseline MLP model.
    
    Args:
        input_dim: Input fingerprint dimension
        hidden_dims: List of hidden layer dimensions
        num_tasks: Number of output tasks
        dropout: Dropout probability
    
    Returns:
        FingerprintMLP model instance
    """
    return FingerprintMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_tasks=num_tasks,
        dropout=dropout
    )


def create_torch_molecule_model(
    model_type: str = "BFGNN",
    num_tasks: int = 1,
    **kwargs
):
    """
    Create a torch-molecule model wrapper.
    
    Args:
        model_type: Type of model ('BFGNN', 'GRIN', 'HFPretrained')
        num_tasks: Number of output tasks (may not be needed for some model types)
        **kwargs: Additional model-specific hyperparameters
    
    Returns:
        torch-molecule model instance
    
    Example:
        >>> model = create_torch_molecule_model("BFGNN")
        >>> # Model can be used with sklearn-style fit/predict interface
    """
    try:
        from torch_molecule import BFGNNMolecularPredictor, GRINMolecularPredictor
        from transformers import AutoTokenizer, AutoModel
        
        if model_type == "BFGNN":
            # BFGNNMolecularPredictor - try different initialization approaches
            # The API may not require num_tasks (it may infer from training data)
            try:
                # Try with num_tasks first (if supported)
                model = BFGNNMolecularPredictor(
                    num_tasks=num_tasks,
                    **kwargs
                )
            except TypeError:
                # If num_tasks is not accepted, try without it
                try:
                    model = BFGNNMolecularPredictor(**kwargs)
                except TypeError as e:
                    # Try with no arguments at all
                    try:
                        model = BFGNNMolecularPredictor()
                    except Exception as e2:
                        raise ValueError(
                            f"Could not initialize BFGNNMolecularPredictor. "
                            f"Tried with num_tasks={num_tasks}, with kwargs only, and with no arguments. "
                            f"Last error: {e2}"
                        )
        elif model_type == "GRIN":
            # Similar approach for GRIN
            try:
                model = GRINMolecularPredictor(
                    num_tasks=num_tasks,
                    **kwargs
                )
            except TypeError:
                try:
                    model = GRINMolecularPredictor(**kwargs)
                except TypeError:
                    model = GRINMolecularPredictor()
        elif model_type == "HFPretrained":
            # HFPretrainedMolecularEncoder/Predictor requires transformers
            # Try to import HFPretrainedMolecularPredictor first
            try:
                from torch_molecule import HFPretrainedMolecularPredictor
                # Try with num_tasks first
                try:
                    model = HFPretrainedMolecularPredictor(
                        num_tasks=num_tasks,
                        **kwargs
                    )
                except TypeError:
                    # Try without num_tasks
                    try:
                        model = HFPretrainedMolecularPredictor(**kwargs)
                    except TypeError:
                        # Try with minimal parameters
                        model = HFPretrainedMolecularPredictor()
            except ImportError:
                # Fallback: Try using encoder directly and create a wrapper
                try:
                    from torch_molecule.encoder.pretrained import HFPretrainedMolecularEncoder
                    # For encoder, we might need to create a custom predictor wrapper
                    # Or use it with a different API
                    # Get model_name from kwargs or use default
                    model_name = kwargs.get('model_name', 'ibm-research/MoLFormer-XL-both-10pct')
                    encoder = HFPretrainedMolecularEncoder(model_name=model_name)
                    # Note: Encoder might need additional layers for prediction
                    # For now, raise informative error
                    raise NotImplementedError(
                        f"HFPretrainedMolecularEncoder found, but predictor wrapper not available. "
                        f"Encoder requires additional prediction layers. "
                        f"Please check torch-molecule documentation for HFPretrainedMolecularPredictor."
                    )
                except ImportError:
                    raise ImportError(
                        "HFPretrained model not found. "
                        "Install with: pip install torch-molecule transformers. "
                        "Or check if HFPretrainedMolecularPredictor is available in your torch-molecule version."
                    )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    except ImportError as e:
        raise ImportError(
            f"torch-molecule not installed. Install with: pip install torch-molecule\n"
            f"Original error: {e}"
        )

