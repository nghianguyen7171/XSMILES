#!/usr/bin/env python3
"""
Main training script for SMILESGNN molecular property prediction model.

Combines sequence-based (SMILES Transformer) and graph-based (GNN) representations.

Usage:
    python scripts/train_hybrid.py [--config config/hybrid_config.yaml]
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
from src.smiles_tokenizer import create_tokenizer_from_smiles
from src.graph_models_hybrid import create_hybrid_model
from src.graph_train import train_gatv2_model as train_graph_model, evaluate_model, create_balanced_sampler
from src.utils import set_seed, save_metrics, ensure_dir


def collate_fn(batch):
    """Collate function for PyG DataLoader."""
    return Batch.from_data_list(batch)


class HybridDataset:
    """
    Dataset wrapper that provides both graph and SMILES inputs.
    """
    
    def __init__(self, pyg_dataset, smiles_list, tokenizer):
        self.pyg_dataset = pyg_dataset
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.pyg_dataset)
    
    def __getitem__(self, idx):
        data = self.pyg_dataset[idx]
        smiles = self.smiles_list[idx]
        token_ids, attention_mask = self.tokenizer.encode(smiles)
        data.smiles_token_ids = torch.tensor(token_ids, dtype=torch.long)
        data.smiles_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return data


def collate_fn_hybrid(batch):
    """Collate function for hybrid dataset that handles SMILES tokens."""
    # First, use standard PyG collate
    batch_data = Batch.from_data_list(batch)
    
    # Extract SMILES token IDs and attention masks
    if hasattr(batch[0], 'smiles_token_ids'):
        smiles_token_ids = torch.stack([item.smiles_token_ids for item in batch])
        smiles_attention_masks = torch.stack([item.smiles_attention_mask for item in batch])
        batch_data.smiles_token_ids = smiles_token_ids
        batch_data.smiles_attention_masks = smiles_attention_masks
    
    return batch_data


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train SMILESGNN model for molecular property prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='config/smilesgnn_config.yaml',
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
    print("SMILESGNN Molecular Property Prediction - Training")
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
    print(f"Class distribution - Val: Toxic={val_df['CT_TOX'].sum()}, Non-toxic={len(val_df) - val_df['CT_TOX'].sum()}")
    print(f"Class distribution - Test: Toxic={test_df['CT_TOX'].sum()}, Non-toxic={len(test_df) - test_df['CT_TOX'].sum()}")
    
    # Check for data leakage
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
    train_pyg_dataset = smiles_list_to_pyg_dataset(
        train_df['smiles'].tolist(),
        labels=train_df['CT_TOX'].tolist()
    )
    val_pyg_dataset = smiles_list_to_pyg_dataset(
        val_df['smiles'].tolist(),
        labels=val_df['CT_TOX'].tolist()
    )
    test_pyg_dataset = smiles_list_to_pyg_dataset(
        test_df['smiles'].tolist(),
        labels=test_df['CT_TOX'].tolist()
    )
    
    print(f"Train graphs: {len(train_pyg_dataset)}, Val graphs: {len(val_pyg_dataset)}, Test graphs: {len(test_pyg_dataset)}")
    
    # Build SMILES tokenizer from training data
    print("\nBuilding SMILES tokenizer...")
    tokenizer = create_tokenizer_from_smiles(
        smiles_list=train_df['smiles'].tolist(),
        vocab_size=int(model_config.get('smiles_vocab_size', 100)),
        max_length=int(model_config.get('smiles_max_length', 128)),
        min_freq=1
    )
    print(f"Vocabulary size: {len(tokenizer.token_to_id)}")
    print()
    
    # Wrap datasets with SMILES tokenizer
    train_dataset = HybridDataset(train_pyg_dataset, train_df['smiles'].tolist(), tokenizer)
    val_dataset = HybridDataset(val_pyg_dataset, val_df['smiles'].tolist(), tokenizer)
    test_dataset = HybridDataset(test_pyg_dataset, test_df['smiles'].tolist(), tokenizer)
    
    # Get feature dimensions
    num_node_features, num_edge_features = get_feature_dims()
    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    
    # Weighted sampler for training (if enabled)
    train_sampler = None
    if training_config.get('use_weighted_sampler', False):
        train_labels = [data.y.item() for data in train_pyg_dataset]
        train_sampler = create_balanced_sampler(train_labels)
        print("Using weighted sampler for balanced training")
    
    batch_size = int(training_config['batch_size'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn_hybrid,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_hybrid,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_hybrid,
        num_workers=0
    )
    
    print()
    
    # Compute positive class weight for weighted BCE loss
    pos_weight = None
    if training_config.get('loss_type') == 'weighted_bce':
        train_labels = train_df['CT_TOX'].values
        num_pos = train_labels.sum()
        num_neg = len(train_labels) - num_pos
        pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device) if num_pos > 0 else torch.tensor(1.0, dtype=torch.float32).to(device)
    
    # Create model
    print("Creating SMILESGNN model...")
    model = create_hybrid_model(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=int(model_config['hidden_dim']),
        num_graph_layers=int(model_config['num_graph_layers']),
        graph_model=str(model_config.get('graph_model', 'gatv2')),
        num_heads=int(model_config.get('num_heads', 4)),
        dropout=float(model_config['dropout']),
        use_residual=bool(model_config.get('use_residual', True)),
        use_jk=bool(model_config.get('use_jk', True)),
        jk_mode=str(model_config.get('jk_mode', 'cat')),
        graph_pooling=str(model_config.get('graph_pooling', 'meanmax')),
        smiles_vocab_size=len(tokenizer.token_to_id),
        smiles_d_model=int(model_config.get('smiles_d_model', 128)),
        smiles_num_layers=int(model_config.get('smiles_num_layers', 3)),
        fusion_method=str(model_config.get('fusion_method', 'attention'))
    )
    
    print(f"Model architecture:")
    print(f"  Graph encoder: {model_config.get('graph_model', 'gatv2')}")
    print(f"  SMILES encoder: Transformer ({model_config.get('smiles_num_layers', 3)} layers)")
    print(f"  Fusion method: {model_config.get('fusion_method', 'attention')}")
    print()
    
    # Wrap model to handle hybrid forward pass
    class HybridModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, batch):
            return self.model(
                batch,
                smiles_token_ids=batch.smiles_token_ids if hasattr(batch, 'smiles_token_ids') else None,
                smiles_attention_mask=batch.smiles_attention_masks if hasattr(batch, 'smiles_attention_masks') else None
            )
    
    wrapped_model = HybridModelWrapper(model)
    
    # Train model
    print("Starting training...")
    history = train_graph_model(
        model=wrapped_model,
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
    
    # Evaluate on validation set first (to check for overfitting)
    print("\nEvaluating on validation set...")
    val_metrics_final = evaluate_model(wrapped_model, val_loader, device=device, return_predictions=False)
    print("\nValidation Set Results (Final):")
    print("=" * 70)
    for metric_name, metric_value in val_metrics_final.items():
        if metric_name != 'predictions' and metric_name != 'probabilities':
            if isinstance(metric_value, (int, float)):
                print(f"{metric_name.upper()}: {metric_value:.4f}")
            else:
                print(f"{metric_name.upper()}: {metric_value}")
    print("=" * 70)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(wrapped_model, test_loader, device=device, return_predictions=False)
    
    print("\nTest Set Results:")
    print("=" * 70)
    for metric_name, metric_value in test_metrics.items():
        if metric_name != 'predictions' and metric_name != 'probabilities':
            if isinstance(metric_value, (int, float)):
                print(f"{metric_name.upper()}: {metric_value:.4f}")
            else:
                print(f"{metric_name.upper()}: {metric_value}")
    print("=" * 70)
    
    # Save model and results
    model_dir = project_root / output_config.get('model_dir', 'models/smilesgnn_model')
    ensure_dir(str(model_dir))
    
    # Save model state dict
    model_path = model_dir / "best_model.pt"
    torch.save(wrapped_model.model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save tokenizer
    import pickle
    tokenizer_path = model_dir / "tokenizer.pkl"
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to: {tokenizer_path}")
    
    # Save metrics
    metrics = {
        'test_auc_roc': test_metrics['auc_roc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
        'test_pr_auc': test_metrics['pr_auc'],
        'test_loss': test_metrics['loss']
    }
    
    metrics_path = model_dir / "smilesgnn_model_metrics.txt"
    save_metrics(metrics, str(metrics_path))
    print(f"Metrics saved to: {metrics_path}")
    
    # Save training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['val_auc_roc'], label='Val AUC-ROC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC-ROC')
    axes[0, 1].set_title('Validation AUC-ROC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['val_f1'], label='Val F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    training_curves_path = model_dir / "training_curves.png"
    plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {training_curves_path}")
    plt.close()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

