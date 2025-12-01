"""
Training and evaluation pipelines for toxicity prediction models.

This module provides high-level functions for:
1. Self-designed MLP models (fingerprint-based, PyTorch)
2. Imported torch-molecule GNN models (graph-based, sklearn-style API)

Model Types:
-----------
- **MLP Model** (Self-designed): FingerprintMLP from src.models
  - Input: Morgan fingerprints (fixed-length vectors)
  - Framework: PyTorch (custom training loop)
  - Featurization: src.featurization.featurize_batch(mode="fingerprint")
  
- **GNN Model** (Imported): torch-molecule BFGNNMolecularPredictor/GRINMolecularPredictor
  - Input: SMILES strings (direct)
  - Framework: torch-molecule (sklearn-style fit/predict)
  - Featurization: Internal (automatic graph conversion)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import pickle

from src.data import load_clintox
from src.featurization import featurize_batch
from src.models import create_baseline_model
from src.train import (
    train_baseline_model, 
    evaluate_model, 
    predict_with_torch_molecule_model
)
from src.utils import set_seed, get_default_config, save_metrics


# ============================================================================
# MLP Model Pipeline (Self-designed)
# ============================================================================

def prepare_mlp_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for MLP model training (fingerprint-based).
    
    Args:
        train_df: Training dataframe with 'smiles' and label columns
        val_df: Validation dataframe
        test_df: Test dataframe
        config: Configuration dictionary with featurization settings
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_labels, val_labels, test_labels)
    """
    # Featurize SMILES to fingerprints
    train_fps = featurize_batch(
        train_df['smiles'].tolist(),
        mode="fingerprint",
        radius=config['featurization']['fingerprint']['radius'],
        n_bits=config['featurization']['fingerprint']['n_bits']
    )
    val_fps = featurize_batch(
        val_df['smiles'].tolist(),
        mode="fingerprint",
        radius=config['featurization']['fingerprint']['radius'],
        n_bits=config['featurization']['fingerprint']['n_bits']
    )
    test_fps = featurize_batch(
        test_df['smiles'].tolist(),
        mode="fingerprint",
        radius=config['featurization']['fingerprint']['radius'],
        n_bits=config['featurization']['fingerprint']['n_bits']
    )
    
    # Get labels
    train_labels = train_df.iloc[:, -1].values.reshape(-1, 1)  # Last column is label
    val_labels = val_df.iloc[:, -1].values.reshape(-1, 1)
    test_labels = test_df.iloc[:, -1].values.reshape(-1, 1)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_fps),
        torch.FloatTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_fps),
        torch.FloatTensor(val_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_fps),
        torch.FloatTensor(test_labels)
    )
    
    # Create DataLoaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_labels, val_labels, test_labels


def train_mlp_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    model: Optional[torch.nn.Module] = None,
    verbose: bool = True
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    """
    Complete training pipeline for MLP model (self-designed).
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        config: Configuration dictionary (uses default if None)
        model: Optional pre-created model (creates new if None)
        verbose: Whether to print training progress
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    if config is None:
        config = get_default_config()
    
    # Prepare data
    train_loader, val_loader, _, _, _, _ = prepare_mlp_data(
        train_df, val_df, val_df, config  # Use val_df for test in this case
    )
    
    # Create model if not provided
    if model is None:
        model = create_baseline_model(
            input_dim=config['featurization']['fingerprint']['n_bits'],
            hidden_dims=config['baseline_model']['hidden_dims'],
            num_tasks=config['baseline_model']['num_tasks'],
            dropout=config['baseline_model']['dropout']
        )
    
    # Train model
    history = train_baseline_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        device=config['training']['device'],
        verbose=verbose
    )
    
    return model, history


def evaluate_mlp_pipeline(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate MLP model on test set.
    
    Args:
        model: Trained MLP model
        test_df: Test dataframe
        config: Configuration dictionary
        device: Device to use for evaluation
    
    Returns:
        Dictionary of metrics
    """
    # Prepare test data
    _, _, test_loader, _, _, _ = prepare_mlp_data(
        test_df, test_df, test_df, config  # Only test_df needed
    )
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device=device)
    return metrics


def save_mlp_model(
    model: torch.nn.Module,
    metrics: Dict[str, float],
    model_dir: Path,
    model_name: str = "baseline_mlp_model"
) -> Tuple[Path, Path]:
    """
    Save MLP model and metrics.
    
    Args:
        model: Trained MLP model
        metrics: Evaluation metrics dictionary
        model_dir: Directory to save model
        model_name: Base name for model file
    
    Returns:
        Tuple of (model_path, metrics_path)
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    
    metrics_path = model_dir / f"{model_name}_metrics.txt"
    save_metrics(metrics, str(metrics_path))
    
    return model_path, metrics_path


def load_mlp_model(
    model_path: Path,
    config: Dict[str, Any]
) -> torch.nn.Module:
    """
    Load a saved MLP model.
    
    Args:
        model_path: Path to saved model state dict
        config: Configuration dictionary
    
    Returns:
        Loaded MLP model
    """
    model = create_baseline_model(
        input_dim=config['featurization']['fingerprint']['n_bits'],
        hidden_dims=config['baseline_model']['hidden_dims'],
        num_tasks=config['baseline_model']['num_tasks'],
        dropout=config['baseline_model']['dropout']
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


# ============================================================================
# GNN Model Pipeline (Imported torch-molecule)
# ============================================================================

def prepare_gnn_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    """
    Prepare data for GNN model training (SMILES strings).
    
    Args:
        train_df: Training dataframe with 'smiles' and label columns
        val_df: Validation dataframe
        test_df: Test dataframe
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        where X are SMILES string lists and y are label arrays
    """
    # Extract SMILES and labels
    X_train = train_df['smiles'].tolist()
    y_train = train_df.iloc[:, -1].values  # Last column is label
    
    X_val = val_df['smiles'].tolist()
    y_val = val_df.iloc[:, -1].values
    
    X_test = test_df['smiles'].tolist()
    y_test = test_df.iloc[:, -1].values
    
    # Convert labels to list of lists format (required by torch-molecule)
    y_train_list = [[int(y)] for y in y_train]
    y_val_list = [[int(y)] for y in y_val]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_gnn_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    search_parameters: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Complete training pipeline for GNN model (imported torch-molecule).
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        config: Configuration dictionary (uses default if None)
        model: Optional pre-created model (creates new if None)
        search_parameters: Hyperparameter search space for autofit (uses default if None)
        verbose: Whether to print training progress
    
    Returns:
        Tuple of (trained_model, training_info)
    """
    if config is None:
        config = get_default_config()
    
    # Prepare data
    X_train, y_train, X_val, y_val, _, _ = prepare_gnn_data(train_df, val_df, val_df)
    y_train_list = [[int(y)] for y in y_train]
    y_val_list = [[int(y)] for y in y_val]
    
    # Create model if not provided
    if model is None:
        try:
            from torch_molecule import BFGNNMolecularPredictor
            from torch_molecule.utils.search import ParameterType, ParameterSpec
            
            model = BFGNNMolecularPredictor(
                num_task=config['torch_molecule']['num_tasks'],
                task_type="classification",
                batch_size=config['training']['batch_size'],
                epochs=config['training']['num_epochs'],
                evaluate_criterion='roc_auc',
                evaluate_higher_better=True,
                verbose='progress_bar' if verbose else False
            )
        except ImportError as e:
            raise ImportError(f"torch-molecule not installed: {e}")
    
    # Define search parameters if not provided
    if search_parameters is None:
        try:
            from torch_molecule.utils.search import ParameterType, ParameterSpec
            search_parameters = {
                "gnn_type": ParameterSpec(ParameterType.CATEGORICAL, ["gin-virtual", "gcn-virtual", "gin", "gcn"]),
                "norm_layer": ParameterSpec(ParameterType.CATEGORICAL, ["batch_norm", "layer_norm"]),
                "graph_pooling": ParameterSpec(ParameterType.CATEGORICAL, ["mean", "sum", "max"]),
                "augmented_feature": ParameterSpec(ParameterType.CATEGORICAL, ["maccs,morgan", "maccs", "morgan", None]),
                "num_layer": ParameterSpec(ParameterType.INTEGER, (2, 5)),
                "hidden_size": ParameterSpec(ParameterType.INTEGER, (64, 256)),
                "drop_ratio": ParameterSpec(ParameterType.FLOAT, (0.0, 0.5)),
                "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-5, 1e-2)),
                "weight_decay": ParameterSpec(ParameterType.LOG_FLOAT, (1e-10, 1e-3)),
            }
        except ImportError:
            search_parameters = {}
    
    # Train with autofit (hyperparameter optimization)
    n_trials = config.get('torch_molecule', {}).get('n_trials', 20)
    
    try:
        if hasattr(model, 'autofit') and search_parameters:
            model.autofit(
                X_train=X_train,
                y_train=y_train_list,
                X_val=X_val,
                y_val=y_val_list,
                n_trials=n_trials,
                search_parameters=search_parameters
            )
            training_info = {"method": "autofit", "n_trials": n_trials}
        else:
            # Fallback to regular fit
            model.fit(X_train, y_train_list, X_val=X_val, y_val=y_val_list)
            training_info = {"method": "fit"}
    except Exception as e:
        if verbose:
            print(f"Warning: autofit failed, trying fit: {e}")
        model.fit(X_train, y_train_list, X_val=X_val, y_val=y_val_list)
        training_info = {"method": "fit", "error": str(e)}
    
    return model, training_info


def evaluate_gnn_pipeline(
    model: Any,
    test_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate GNN model on test set.
    
    Args:
        model: Trained torch-molecule model
        test_df: Test dataframe
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
    
    # Prepare test data
    _, _, _, _, X_test, y_test = prepare_gnn_data(test_df, test_df, test_df)
    
    # Get predictions
    y_pred_proba = predict_with_torch_molecule_model(model, X_test)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'f1': f1_score(y_test, y_pred_binary, zero_division=0),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'auprc': average_precision_score(y_test, y_pred_proba)  # Same as PR-AUC
    }
    
    return metrics


def save_gnn_model(
    model: Any,
    metrics: Dict[str, float],
    model_dir: Path,
    model_name: str = "torch_molecule_model"
) -> Tuple[Path, Path]:
    """
    Save GNN model and metrics.
    
    Args:
        model: Trained torch-molecule model
        metrics: Evaluation metrics dictionary
        model_dir: Directory to save model
        model_name: Base name for model file
    
    Returns:
        Tuple of (model_path, metrics_path)
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Try torch-molecule's save method first
    model_path = model_dir / f"{model_name}.pt"
    metrics_path = model_dir / f"{model_name}_metrics.txt"
    
    try:
        if hasattr(model, 'save_model'):
            model.save_model(str(model_path))
        else:
            # Fallback to pickle
            model_path = model_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    except Exception as e:
        # Fallback to pickle
        model_path = model_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    save_metrics(metrics, str(metrics_path))
    
    return model_path, metrics_path


def load_gnn_model(
    model_path: Path
) -> Any:
    """
    Load a saved GNN model.
    
    Args:
        model_path: Path to saved model file (.pt or .pkl)
    
    Returns:
        Loaded torch-molecule model
    """
    if model_path.suffix == '.pt':
        try:
            from torch_molecule import BFGNNMolecularPredictor
            model = BFGNNMolecularPredictor()
            model.load_model(str(model_path))
            return model
        except Exception:
            # Fallback to pickle
            pass
    
    # Load from pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# ============================================================================
# Convenience Functions
# ============================================================================

def load_clintox_dataset(
    cache_dir: str = "./data",
    split_type: str = "scaffold",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split ClinTox dataset.
    
    Args:
        cache_dir: Directory to cache data
        split_type: Type of split ('scaffold' or 'random')
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    return load_clintox(cache_dir=cache_dir, split_type=split_type, seed=seed)


def get_model_type(model: Any) -> str:
    """
    Determine model type (MLP or GNN).
    
    Args:
        model: Model instance
    
    Returns:
        'mlp' or 'gnn'
    """
    if isinstance(model, torch.nn.Module) and hasattr(model, 'features'):
        return 'mlp'  # Self-designed MLP
    elif hasattr(model, 'predict') or hasattr(model, 'fit'):
        # Check if it's torch-molecule model
        model_type_str = str(type(model))
        if 'torch_molecule' in model_type_str or 'BFGNN' in model_type_str or 'GRIN' in model_type_str:
            return 'gnn'  # Imported torch-molecule GNN
    return 'unknown'

