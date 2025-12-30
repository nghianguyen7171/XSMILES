#!/usr/bin/env python3
"""
Main training script for GATv2 molecular property prediction model.

Usage:
    python scripts/train_gatv2.py [--config config/gatv2_config.yaml]
"""

import sys
from pathlib import Path
import argparse
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.data import load_clintox
from src.graph_data import smiles_list_to_pyg_dataset, get_feature_dims
from src.graph_models import create_gatv2_model
from src.graph_train import train_gatv2_model, evaluate_model, create_balanced_sampler
from src.utils import set_seed, save_metrics


def collate_fn(batch):
    """Collate function for PyG DataLoader."""
    return Batch.from_data_list(batch)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train GATv2 model for molecular property prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='config/gatv2_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu or cuda)'
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    output_config = config.get('output', {})
    
    # Set seed for reproducibility
    set_seed(data_config.get('seed', 42))
    
    # Device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print("=" * 70)
    print("GATv2 Molecular Property Prediction - Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Configuration: {config_path}")
    print()
    
    # Load dataset
    print("Loading ClinTox dataset...")
    train_df, val_df, test_df = load_clintox(
        cache_dir=str(project_root / data_config['cache_dir']),
        split_type=data_config['split_type'],
        seed=data_config['seed']
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Class distribution - Train: Toxic={train_df['CT_TOX'].sum()}, Non-toxic={len(train_df) - train_df['CT_TOX'].sum()}")
    
    # Check for data leakage (SMILES overlap)
    train_smiles = set(train_df['smiles'].values)
    val_smiles = set(val_df['smiles'].values)
    test_smiles = set(test_df['smiles'].values)
    
    train_val_overlap = train_smiles.intersection(val_smiles)
    train_test_overlap = train_smiles.intersection(test_smiles)
    val_test_overlap = val_smiles.intersection(test_smiles)
    
    if train_val_overlap:
        print(f"⚠ WARNING: {len(train_val_overlap)} SMILES found in both train and val sets!")
    if train_test_overlap:
        print(f"⚠ WARNING: {len(train_test_overlap)} SMILES found in both train and test sets!")
    if val_test_overlap:
        print(f"⚠ WARNING: {len(val_test_overlap)} SMILES found in both val and test sets!")
    
    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("✓ Data split validation passed: No SMILES overlap between train/val/test")
    print()
    
    # Convert SMILES to PyG Data objects
    print("Converting SMILES to graph representations...")
    train_dataset = smiles_list_to_pyg_dataset(
        train_df['smiles'].tolist(),
        labels=train_df['CT_TOX'].tolist()
    )
    val_dataset = smiles_list_to_pyg_dataset(
        val_df['smiles'].tolist(),
        labels=val_df['CT_TOX'].tolist()
    )
    test_dataset = smiles_list_to_pyg_dataset(
        test_df['smiles'].tolist(),
        labels=test_df['CT_TOX'].tolist()
    )
    
    print(f"Train graphs: {len(train_dataset)}, Val graphs: {len(val_dataset)}, Test graphs: {len(test_dataset)}")
    
    # Get feature dimensions
    num_node_features, num_edge_features = get_feature_dims()
    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    
    # Weighted sampler for training (if enabled)
    train_sampler = None
    if training_config.get('use_weighted_sampler', False):
        train_labels = [data.y.item() for data in train_dataset]
        train_sampler = create_balanced_sampler(train_labels)
        print("Using weighted sampler for balanced training")
    
    batch_size = int(training_config['batch_size'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    print()
    
    # Compute positive class weight for weighted BCE loss
    pos_weight = None
    if training_config.get('loss_type') == 'weighted_bce':
        train_labels = train_df['CT_TOX'].values
        num_pos = train_labels.sum()
        num_neg = len(train_labels) - num_pos
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        print(f"Positive class weight: {pos_weight:.4f}")
        print()
    
    # Create model
    print("Creating GATv2 model...")
    model = create_gatv2_model(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=int(model_config['hidden_dim']),
        num_layers=int(model_config['num_layers']),
        num_heads=int(model_config['num_heads']),
        dropout=float(model_config['dropout']),
        use_residual=model_config['use_residual'],
        use_jk=model_config['use_jk'],
        jk_mode=model_config['jk_mode'],
        pooling=model_config['pooling']
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Train model
    print("=" * 70)
    print("Training Model")
    print("=" * 70)
    
    history = train_gatv2_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(training_config['num_epochs']),
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay']),
        device=device,
        loss_type=str(training_config['loss_type']),
        focal_alpha=float(training_config.get('focal_alpha', 0.25)),
        focal_gamma=float(training_config.get('focal_gamma', 2.0)),
        pos_weight=pos_weight,
        early_stopping_patience=int(training_config['early_stopping_patience']),
        early_stopping_metric=str(training_config['early_stopping_metric']),
        verbose=True
    )
    print()
    
    # Evaluate on test set
    print("=" * 70)
    print("Evaluating on Test Set")
    print("=" * 70)
    
    test_metrics = evaluate_model(model, test_loader, device=device, return_predictions=False)
    
    print("Test Metrics:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={test_metrics['confusion_matrix'][0][0]}, FP={test_metrics['confusion_matrix'][0][1]}")
    print(f"    FN={test_metrics['confusion_matrix'][1][0]}, TP={test_metrics['confusion_matrix'][1][1]}")
    print()
    
    # Save model and results
    model_dir = project_root / output_config.get('model_dir', 'models/gatv2_model')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / 'best_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'num_node_features': num_node_features,
        'num_edge_features': num_edge_features,
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save metrics
    metrics_path = model_dir / 'gatv2_model_metrics.txt'
    metrics_to_save = {
        'test_auc_roc': test_metrics['auc_roc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
        'test_pr_auc': test_metrics['pr_auc'],
        'test_loss': test_metrics['loss']
    }
    
    # Add validation metrics if available
    if history['val_auc_roc']:
        best_val_epoch = np.argmax(history['val_f1'])
        metrics_to_save['val_auc_roc'] = history['val_auc_roc'][best_val_epoch]
        metrics_to_save['val_f1'] = history['val_f1'][best_val_epoch]
        metrics_to_save['val_pr_auc'] = history['val_pr_auc'][best_val_epoch]
    
    save_metrics(metrics_to_save, str(metrics_path))
    print(f"Metrics saved to: {metrics_path}")
    
    # Plot training curves
    if len(history['train_loss']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        if history['val_loss']:
            axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC-ROC
        if history['val_auc_roc']:
            axes[0, 1].plot(epochs, history['val_auc_roc'], label='Val AUC-ROC', marker='s', color='green')
            axes[0, 1].axhline(y=test_metrics['auc_roc'], label=f'Test AUC-ROC ({test_metrics["auc_roc"]:.4f})', 
                              color='red', linestyle='--')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC-ROC')
            axes[0, 1].set_title('Validation AUC-ROC')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # F1 Score
        if history['val_f1']:
            axes[1, 0].plot(epochs, history['val_f1'], label='Val F1', marker='s', color='orange')
            axes[1, 0].axhline(y=test_metrics['f1'], label=f'Test F1 ({test_metrics["f1"]:.4f})', 
                              color='red', linestyle='--')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('Validation F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # PR-AUC
        if history['val_pr_auc']:
            axes[1, 1].plot(epochs, history['val_pr_auc'], label='Val PR-AUC', marker='s', color='purple')
            axes[1, 1].axhline(y=test_metrics['pr_auc'], label=f'Test PR-AUC ({test_metrics["pr_auc"]:.4f})', 
                              color='red', linestyle='--')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('PR-AUC')
            axes[1, 1].set_title('Validation PR-AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        curves_path = model_dir / 'training_curves.png'
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {curves_path}")
        plt.close()
    
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

