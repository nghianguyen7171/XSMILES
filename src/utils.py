"""
Utility functions for reproducibility, configuration, and logging.
"""

import random
import numpy as np
import torch
import os
from typing import Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
    
    Example:
        >>> set_seed(42)
        >>> # All random operations will be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration dictionary for the project.
    
    Returns:
        Dictionary with default hyperparameters and settings
    """
    return {
        "seed": 42,
        "data": {
            "cache_dir": "./data",
            "split_type": "scaffold",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1
        },
        "featurization": {
            "fingerprint": {
                "radius": 2,
                "n_bits": 2048
            },
            "graph": {
                "mode": "smiles"  # or "graph_tensors"
            }
        },
        "baseline_model": {
            "input_dim": 2048,
            "hidden_dims": [512, 256, 128],
            "num_tasks": 1,
            "dropout": 0.2
        },
        "training": {
            "num_epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 128,
            "device": "cpu"
        },
        "torch_molecule": {
            "model_type": "BFGNN",
            "num_tasks": 1,
            "n_trials": 20  # Number of hyperparameter search trials for autofit()
        }
    }


def save_metrics(metrics: Dict[str, float], filepath: str):
    """
    Save metrics dictionary to a text file.
    
    Args:
        metrics: Dictionary of metric names and values
        filepath: Path to save metrics file
    """
    with open(filepath, "w") as f:
        f.write("Evaluation Metrics\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")


def load_metrics(filepath: str) -> Dict[str, float]:
    """
    Load metrics from a text file.
    
    Args:
        filepath: Path to metrics file
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    with open(filepath, "r") as f:
        for line in f:
            if ":" in line and not line.startswith("="):
                key, value = line.strip().split(":", 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    metrics[key.strip()] = value.strip()
    return metrics


def ensure_dir(directory: str):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)

