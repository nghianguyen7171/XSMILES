"""
Training utilities for toxicity prediction models.

Provides training loops for baseline PyTorch models and wrappers
for torch-molecule sklearn-style estimators.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Callable, Any
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
import numpy as np

# For sigmoid function (logit to probability conversion)
try:
    from scipy.special import expit
except ImportError:
    # Fallback to numpy-based sigmoid if scipy not available
    def expit(x):
        """Sigmoid function: 1 / (1 + exp(-x))"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow


def train_baseline_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu",
    verbose: bool = True
) -> Dict[str, list]:
    """
    Train a baseline PyTorch model (e.g., FingerprintMLP).
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress
    
    Returns:
        Dictionary with training history containing 'train_loss', 'val_loss', etc.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).float()
            
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_losses = []
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device).float()
                    
                    logits = model(batch_features)
                    loss = criterion(logits, batch_labels)
                    val_losses.append(loss.item())
                    
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs)
                    all_labels.extend(batch_labels.cpu().numpy())
            
            avg_val_loss = np.mean(val_losses)
            val_auc = roc_auc_score(all_labels, all_preds)
            
            history["val_loss"].append(avg_val_loss)
            history["val_auc"].append(val_auc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Val AUC: {val_auc:.4f}"
                )
        elif verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
    
    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate a PyTorch model and compute metrics.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on
    
    Returns:
        Dictionary of metrics: AUC-ROC, accuracy, F1, PR-AUC
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).float()
            
            logits = model(batch_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds_binary = (all_preds > 0.5).astype(int)
    
    metrics = {
        "auc_roc": roc_auc_score(all_labels, all_preds),
        "accuracy": accuracy_score(all_labels, all_preds_binary),
        "f1": f1_score(all_labels, all_preds_binary),
        "pr_auc": average_precision_score(all_labels, all_preds)
    }
    
    return metrics


def predict_with_torch_molecule_model(model, X):
    """
    Helper function to get predictions from torch-molecule model.
    Handles different API variations.
    
    Args:
        model: torch-molecule model instance
        X: Input data (SMILES strings or features)
    
    Returns:
        Predictions as 1D numpy array (probabilities for binary classification)
    """
    try:
        # Try predict_proba first (sklearn-style)
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X)
            # Handle dict output
            if isinstance(y_pred, dict):
                # Try common keys (including 'prediction' singular which torch-molecule uses)
                if 'prediction' in y_pred:
                    y_pred = y_pred['prediction']
                elif 'predictions' in y_pred:
                    y_pred = y_pred['predictions']
                elif 'proba' in y_pred:
                    y_pred = y_pred['proba']
                elif 'y_pred' in y_pred:
                    y_pred = y_pred['y_pred']
                else:
                    # Take first value if dict
                    y_pred = list(y_pred.values())[0]
            
            # Handle 2D output: take positive class probabilities
            if isinstance(y_pred, np.ndarray):
                if len(y_pred.shape) > 1:
                    if y_pred.shape[1] > 1:
                        y_pred = y_pred[:, 1]  # Take positive class
                    else:
                        y_pred = y_pred.flatten()
                y_pred = y_pred.flatten()
            else:
                y_pred = np.array(y_pred).flatten()
            
            # Check if predictions are logits (negative values or outside [0,1])
            # Apply sigmoid to convert logits to probabilities
            if y_pred.min() < 0 or y_pred.max() > 1:
                y_pred = expit(y_pred)
            
            return y_pred
        
        # Try predict (may return probabilities, classes, or dict)
        elif hasattr(model, 'predict'):
            y_pred = model.predict(X)
            
            # Handle dict output first (before any array operations)
            if isinstance(y_pred, dict):
                # Try common keys (including 'prediction' singular which torch-molecule uses)
                if 'prediction' in y_pred:
                    y_pred = y_pred['prediction']
                elif 'predictions' in y_pred:
                    y_pred = y_pred['predictions']
                elif 'y_pred' in y_pred:
                    y_pred = y_pred['y_pred']
                elif 'proba' in y_pred:
                    y_pred = y_pred['proba']
                elif 'probabilities' in y_pred:
                    y_pred = y_pred['probabilities']
                else:
                    # Take first value if dict - try to find numeric array
                    for key, val in y_pred.items():
                        if isinstance(val, (np.ndarray, list)):
                            y_pred = val
                            break
                    else:
                        # If no array found, take first value
                        y_pred = list(y_pred.values())[0]
            
            # Convert to numpy array (handle both list and array inputs)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
            
            # Handle 2D arrays (e.g., shape (n_samples, 1))
            if len(y_pred.shape) > 1:
                if y_pred.shape[1] == 1:
                    y_pred = y_pred[:, 0]  # Take first column
                elif y_pred.shape[1] > 1:
                    y_pred = y_pred[:, 1]  # Take positive class column
                else:
                    y_pred = y_pred.flatten()
            else:
                y_pred = y_pred.flatten()
            
            # Check if predictions are logits (negative values or outside [0,1])
            # Apply sigmoid to convert logits to probabilities
            # Logits can be any real number, probabilities are always in [0,1]
            if y_pred.min() < 0 or y_pred.max() > 1:
                # Apply sigmoid to convert logits to probabilities
                y_pred = expit(y_pred)
            
            # Check if predictions are probabilities (0-1 range) or classes (0/1)
            # Only check after we know it's a numpy array
            unique_vals = np.unique(y_pred)
            if y_pred.dtype == int or (len(unique_vals) == 2 and np.all(np.isin(unique_vals, [0, 1]))):
                # Might be binary classes - convert to float for probability calculations
                y_pred = y_pred.astype(float)
            
            return y_pred
        
        else:
            raise AttributeError("Model has neither predict_proba nor predict method")
            
    except Exception as e:
        raise RuntimeError(f"Error getting predictions from model: {e}")


def train_torch_molecule_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a torch-molecule model using its sklearn-style interface.
    
    Args:
        model: torch-molecule model instance (with fit/predict methods)
        X_train: Training features (SMILES strings or graph representations)
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        **kwargs: Additional arguments for model.fit()
    
    Returns:
        Dictionary with training results and history
    """
    # torch-molecule models typically have sklearn-style fit interface
    print("Starting model training...")
    
    # Check model state before training
    print(f"Model type: {type(model)}")
    if hasattr(model, 'trained_'):
        print(f"Model trained flag (before): {model.trained_}")
    if hasattr(model, 'is_fitted'):
        print(f"Model is_fitted (before): {model.is_fitted}")
    
    # Inspect model weights before training (if possible)
    weight_stats_before = None
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            # Try to get first layer weights
            params = list(model.model.parameters())
            if len(params) > 0:
                first_param = params[0]
                weight_stats_before = {
                    'mean': first_param.data.mean().item(),
                    'std': first_param.data.std().item(),
                    'min': first_param.data.min().item(),
                    'max': first_param.data.max().item()
                }
                print(f"First layer weights (before): mean={weight_stats_before['mean']:.6f}, std={weight_stats_before['std']:.6f}")
    except Exception as e:
        print(f"Could not inspect weights before training: {e}")
    
    # Train the model
    try:
        # Try with verbose if not already in kwargs
        fit_kwargs = kwargs.copy()
        if 'verbose' not in fit_kwargs:
            fit_kwargs['verbose'] = True
        
        # Try with epochs if available
        if 'epochs' not in fit_kwargs and 'n_epochs' not in fit_kwargs:
            fit_kwargs['epochs'] = 50  # Default to 50 epochs
        
        model.fit(X_train, y_train, **fit_kwargs)
        print("✓ Model.fit() completed")
    except TypeError as e:
        # If verbose or epochs cause issues, try without them
        if 'verbose' in str(e) or 'epochs' in str(e) or 'n_epochs' in str(e):
            print(f"Note: {e}")
            print("Retrying without verbose/epochs parameters...")
            fit_kwargs = {k: v for k, v in kwargs.items() if k not in ['verbose', 'epochs', 'n_epochs']}
            model.fit(X_train, y_train, **fit_kwargs)
            print("✓ Model.fit() completed (without verbose/epochs)")
        else:
            raise
    except Exception as e:
        print(f"⚠ Error during model.fit(): {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Check if weights changed after training
    if weight_stats_before is not None:
        try:
            params = list(model.model.parameters())
            if len(params) > 0:
                first_param = params[0]
                weight_stats_after = {
                    'mean': first_param.data.mean().item(),
                    'std': first_param.data.std().item(),
                    'min': first_param.data.min().item(),
                    'max': first_param.data.max().item()
                }
                print(f"First layer weights (after): mean={weight_stats_after['mean']:.6f}, std={weight_stats_after['std']:.6f}")
                
                # Check if weights changed
                mean_diff = abs(weight_stats_after['mean'] - weight_stats_before['mean'])
                if mean_diff < 1e-6:
                    print(f"⚠ WARNING: Weights may not have changed significantly (mean diff: {mean_diff:.9f})")
                    print(f"   This suggests training may not be updating model weights!")
                else:
                    print(f"✓ Weights changed (mean diff: {mean_diff:.6f})")
        except Exception as e:
            print(f"Could not inspect weights after training: {e}")
    
    # Verify training completed
    if hasattr(model, 'trained_'):
        print(f"Model trained flag (after): {model.trained_}")
    if hasattr(model, 'is_fitted'):
        print(f"Model is_fitted (after): {model.is_fitted}")
    
    # Test prediction on a sample to verify model works
    try:
        sample_pred = model.predict(X_train[:1] if len(X_train) > 0 else X_train)
        if isinstance(sample_pred, dict):
            pred_value = list(sample_pred.values())[0] if sample_pred else None
            print(f"Sample training prediction: {pred_value}")
        else:
            print(f"Sample training prediction: {sample_pred}")
    except Exception as e:
        print(f"⚠ Warning: Could not get sample prediction: {e}")
    
    results = {"model": model}
    
    if X_val is not None and y_val is not None:
        try:
            y_pred = predict_with_torch_molecule_model(model, X_val)
            y_val_array = np.array(y_val).flatten()
            
            # Calculate metrics
            results["val_auc"] = roc_auc_score(y_val_array, y_pred)
            results["val_accuracy"] = accuracy_score(y_val_array, (y_pred > 0.5).astype(int))
            
        except Exception as e:
            results["val_error"] = str(e)
            print(f"⚠ Warning: Could not evaluate on validation set: {e}")
    
    return results

