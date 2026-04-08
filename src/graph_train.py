"""
Training utilities for graph neural network models.

Provides training loops, loss functions (focal loss, weighted BCE),
and evaluation functions for molecular property prediction with class imbalance handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Batch
from typing import TYPE_CHECKING, Dict, Optional, List, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

if TYPE_CHECKING:
    from src.datasets.base import TaskConfig


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples,
    which is particularly useful for imbalanced datasets.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size,)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t (probability of true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t (weighting factor)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for class imbalance.
    """
    
    def __init__(self, pos_weight: Optional[float] = None, reduction: str = 'mean'):
        """
        Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class (typically num_neg / num_pos)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size,)
            targets: Ground truth labels of shape (batch_size,)
        
        Returns:
            Weighted BCE loss value
        """
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor(self.pos_weight, device=inputs.device)
            return F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=pos_weight_tensor, reduction=self.reduction
            )
        else:
            return F.binary_cross_entropy_with_logits(
                inputs, targets, reduction=self.reduction
            )


class MaskedMultiTaskLoss(nn.Module):
    """
    Focal loss for multi-task learning with missing (NaN) labels.

    Tox21 has sparse annotations — many (compound, assay) pairs are
    untested. This loss ignores NaN entries and averages over valid
    (compound, task) pairs only, preventing the model from being penalised
    for predictions on untested combinations.

    Reference: Lin et al., Focal Loss for Dense Object Detection, ICCV 2017.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for the rare (toxic) class.
            gamma: Focusing parameter — higher values down-weight easy examples.
        """
        super().__init__()
        self._focal = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Predicted logits, shape (batch_size, num_tasks).
            targets: Ground-truth labels, shape (batch_size, num_tasks).
                     NaN entries are ignored.

        Returns:
            Scalar loss averaged over all valid (non-NaN) entries.
        """
        valid_mask = ~torch.isnan(targets)
        if valid_mask.sum() == 0:
            # Edge case: entire batch has no valid labels (should not happen
            # in practice but guards against empty-batch errors)
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        # Replace NaN with 0 before focal computation (masked out afterward)
        per_entry_loss = self._focal(logits, targets.nan_to_num(0.0))
        return per_entry_loss[valid_mask].mean()


def create_multitask_sampler(
    labels_array: np.ndarray,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Create a balanced sampler for multi-task datasets (e.g. Tox21).

    A compound is treated as "positive" if it is active (label = 1) in
    at least one task. This prevents chronically under-sampling compounds
    that are active in rare tasks.

    Args:
        labels_array: Shape (n_compounds, n_tasks), NaN for missing labels.
        replacement:  Whether to sample with replacement.

    Returns:
        WeightedRandomSampler instance.
    """
    # "any positive" binary label per compound (ignoring NaN)
    any_positive = (np.nanmax(labels_array, axis=1) > 0).astype(int)
    return create_balanced_sampler(any_positive, replacement=replacement)


def create_balanced_sampler(
    labels: List[float],
    replacement: bool = True
) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for balanced sampling.
    
    Args:
        labels: List of labels (0 or 1)
        replacement: Whether to sample with replacement
    
    Returns:
        WeightedRandomSampler instance
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels.astype(int))
    
    # Compute weights for each sample
    # Weight = 1 / (num_samples_in_class * num_classes)
    weights = np.zeros(len(labels))
    for i in range(len(class_counts)):
        if class_counts[i] > 0:
            weights[labels == i] = 1.0 / (class_counts[i] * len(class_counts))
    
    # Normalize weights
    weights = weights / weights.sum() * len(weights)
    
    return WeightedRandomSampler(
        weights=weights.astype(float),
        num_samples=len(labels),
        replacement=replacement
    )


def train_gatv2_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    loss_type: str = "focal",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Optional[float] = None,
    early_stopping_patience: int = 20,
    early_stopping_metric: str = "f1",
    verbose: bool = True,
    task_config: Optional["TaskConfig"] = None,
) -> Dict[str, List[float]]:
    """
    Train a GATv2 model for molecular property prediction.
    
    Args:
        model: GATv2MolecularPredictor model
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        device: Device to train on ('cpu' or 'cuda')
        loss_type: Loss function type ('focal', 'weighted_bce', or 'bce')
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        pos_weight: Positive class weight for weighted BCE loss
        early_stopping_patience: Number of epochs to wait before early stopping
        early_stopping_metric: Metric to use for early stopping ('f1', 'auc_roc', 'loss')
        verbose: Whether to print training progress
    
    Returns:
        Dictionary with training history containing:
        - 'train_loss': Training loss per epoch
        - 'val_loss': Validation loss per epoch
        - 'val_auc_roc': Validation AUC-ROC per epoch
        - 'val_f1': Validation F1 score per epoch
        - 'val_pr_auc': Validation PR-AUC per epoch
    """
    model = model.to(device)
    
    # Resolve early_stopping_metric from task_config when not overridden
    is_multitask = task_config is not None and task_config.is_multitask
    if is_multitask and early_stopping_metric == "f1":
        early_stopping_metric = task_config.primary_metric  # "mean_auc_roc"

    # Initialize loss function
    if loss_type == "masked_focal":
        criterion = MaskedMultiTaskLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == "weighted_bce":
        criterion = WeightedBCELoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max' if early_stopping_metric != 'loss' else 'min',
        factor=0.5, patience=10, verbose=verbose
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc_roc': [],
        'val_mean_auc_roc': [],   # multi-task primary metric
        'val_accuracy': [],
        'val_f1': [],
        'val_pr_auc': [],
    }
    
    # Early stopping
    best_metric = float('-inf') if early_stopping_metric != 'loss' else float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y.squeeze()
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch).squeeze()
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Skip NaN/Inf losses — prevents corrupted gradients from
            # poisoning model weights (observed on CPU, early epochs)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            val_metrics = evaluate_model(
                model, val_loader, device=device, task_config=task_config
            )

            history['val_loss'].append(val_metrics['loss'])
            history['val_auc_roc'].append(val_metrics.get('auc_roc', 0.0))
            history['val_mean_auc_roc'].append(val_metrics.get('mean_auc_roc', 0.0))
            history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
            history['val_f1'].append(val_metrics.get('f1', 0.0))
            history['val_pr_auc'].append(
                val_metrics.get('pr_auc', val_metrics.get('mean_pr_auc', 0.0))
            )

            # Early stopping logic
            if early_stopping_metric == 'mean_auc_roc':
                current_metric = val_metrics.get('mean_auc_roc', 0.0)
                is_better = current_metric > best_metric
            elif early_stopping_metric == 'f1':
                current_metric = val_metrics.get('f1', 0.0)
                is_better = current_metric > best_metric
            elif early_stopping_metric == 'auc_roc':
                current_metric = val_metrics.get('auc_roc', 0.0)
                is_better = current_metric > best_metric
            else:  # loss
                current_metric = val_metrics['loss']
                is_better = current_metric < best_metric

            if is_better:
                best_metric = current_metric
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Update learning rate
            scheduler.step(current_metric)

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if is_multitask:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Mean AUC-ROC: {val_metrics.get('mean_auc_roc', 0.0):.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val AUC-ROC: {val_metrics.get('auc_roc', 0.0):.4f}, "
                        f"Val F1: {val_metrics.get('f1', 0.0):.4f}"
                    )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        elif verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    return_predictions: bool = False,
    task_config: Optional["TaskConfig"] = None,
) -> Dict:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on
        return_predictions: Whether to return predictions and labels
    
    Returns:
        Dictionary containing:
        - 'loss': Average loss
        - 'auc_roc': AUC-ROC score
        - 'accuracy': Accuracy
        - 'f1': F1 score
        - 'pr_auc': PR-AUC score
        - 'confusion_matrix': Confusion matrix
        - 'predictions': Predictions (if return_predictions=True)
        - 'labels': Labels (if return_predictions=True)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    losses = []

    is_multitask = task_config is not None and task_config.is_multitask
    if is_multitask:
        criterion = MaskedMultiTaskLoss()   # handles NaN labels
    else:
        criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            labels = batch.y.squeeze()

            # Forward pass
            logits = model(batch).squeeze()

            # Compute loss
            loss = criterion(logits, labels)
            losses.append(loss.item())
            
            # Get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    all_preds = np.array(all_preds)    # (N,) or (N, T)
    all_labels = np.array(all_labels)  # (N,) or (N, T)
    all_logits = np.array(all_logits)

    # Guard against NaN/Inf from early-epoch numerical instability
    if not np.all(np.isfinite(all_preds)):
        all_preds = np.nan_to_num(all_preds, nan=0.5, posinf=1.0, neginf=0.0)

    # ── Multi-task path (Tox21) ────────────────────────────────────────────
    if task_config is not None and task_config.is_multitask:
        per_task_auc: Dict[str, float] = {}
        per_task_pr_auc: Dict[str, float] = {}

        for t, task_name in enumerate(task_config.task_names):
            valid = ~np.isnan(all_labels[:, t])
            if valid.sum() < 2:
                continue  # not enough samples
            y_true = all_labels[valid, t]
            y_score = all_preds[valid, t]
            if len(np.unique(y_true)) < 2:
                continue  # degenerate task (all same label in this split)
            per_task_auc[task_name] = roc_auc_score(y_true, y_score)
            per_task_pr_auc[task_name] = average_precision_score(y_true, y_score)

        mean_auc = float(np.mean(list(per_task_auc.values()))) if per_task_auc else 0.0
        mean_pr_auc = float(np.mean(list(per_task_pr_auc.values()))) if per_task_pr_auc else 0.0

        metrics = {
            'loss':             float(np.mean(losses)),
            'mean_auc_roc':     mean_auc,
            'mean_pr_auc':      mean_pr_auc,
            'per_task_auc_roc': per_task_auc,
            'per_task_pr_auc':  per_task_pr_auc,
            'num_valid_tasks':  len(per_task_auc),
        }
        if return_predictions:
            metrics['predictions'] = all_preds
            metrics['labels']      = all_labels
            metrics['logits']      = all_logits
        return metrics

    # ── Single-task path (ClinTox) ─────────────────────────────────────────
    binary_preds = (all_preds > 0.5).astype(int)
    metrics = {
        'loss':             float(np.mean(losses)),
        'auc_roc':          roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
        'accuracy':         accuracy_score(all_labels, binary_preds),
        'f1':               f1_score(all_labels, binary_preds, zero_division=0.0),
        'pr_auc':           average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(all_labels, binary_preds).tolist(),
    }
    if return_predictions:
        metrics['predictions'] = all_preds
        metrics['labels']      = all_labels
        metrics['logits']      = all_logits
    return metrics

