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
from typing import Dict, Optional, List, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


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
    verbose: bool = True
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
    
    # Initialize loss function
    if loss_type == "focal":
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
        'val_accuracy': [],
        'val_f1': [],
        'val_pr_auc': []
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
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device=device)
            
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc_roc'].append(val_metrics['auc_roc'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_pr_auc'].append(val_metrics['pr_auc'])
            
            # Early stopping logic
            if early_stopping_metric == 'f1':
                current_metric = val_metrics['f1']
                is_better = current_metric > best_metric
            elif early_stopping_metric == 'auc_roc':
                current_metric = val_metrics['auc_roc']
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
                print(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val AUC-ROC: {val_metrics['auc_roc']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}"
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
    return_predictions: bool = False
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
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Binary predictions (threshold = 0.5)
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Compute metrics
    metrics = {
        'loss': np.mean(losses),
        'auc_roc': roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
        'accuracy': accuracy_score(all_labels, binary_preds),
        'f1': f1_score(all_labels, binary_preds, zero_division=0.0),
        'pr_auc': average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(all_labels, binary_preds).tolist()
    }
    
    if return_predictions:
        metrics['predictions'] = all_preds
        metrics['labels'] = all_labels
        metrics['logits'] = all_logits
    
    return metrics

