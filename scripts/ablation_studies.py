#!/usr/bin/env python3
"""
Ablation studies for GATv2 model.

Tests different architectural variants, loss functions, and class imbalance
handling strategies to understand their impact on model performance.
"""

import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.data import load_clintox
from src.graph_data import smiles_list_to_pyg_dataset, get_feature_dims
from src.graph_models import create_gatv2_model
from src.graph_train import train_gatv2_model, evaluate_model, create_balanced_sampler
from src.utils import set_seed


def collate_fn(batch):
    """Collate function for PyG DataLoader."""
    return Batch.from_data_list(batch)


def run_ablation_study(
    train_dataset,
    val_dataset,
    test_dataset,
    device='cpu',
    num_epochs=50,
    batch_size=32,
    study_type='architecture'
):
    """
    Run ablation study for different model configurations.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        device: Device to use
        num_epochs: Number of epochs for each configuration
        batch_size: Batch size
        study_type: Type of ablation study ('architecture', 'loss', 'sampling')
    
    Returns:
        Dictionary with results for each configuration
    """
    num_node_features, num_edge_features = get_feature_dims()
    
    # Create data loaders
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
    
    results = {}
    
    if study_type == 'architecture':
        # Test different architectural variants
        configs = [
            {
                'name': 'Baseline (GATv2, no JK, mean pooling)',
                'use_jk': False,
                'pooling': 'global_mean',
                'use_residual': True,
            },
            {
                'name': 'With Jumping Knowledge',
                'use_jk': True,
                'jk_mode': 'cat',
                'pooling': 'global_mean',
                'use_residual': True,
            },
            {
                'name': 'Set2Set Pooling',
                'use_jk': True,
                'jk_mode': 'cat',
                'pooling': 'set2set',
                'use_residual': True,
            },
            {
                'name': 'Attentive Pooling',
                'use_jk': True,
                'jk_mode': 'cat',
                'pooling': 'attentive',
                'use_residual': True,
            },
            {
                'name': 'No Residual Connections',
                'use_jk': True,
                'jk_mode': 'cat',
                'pooling': 'set2set',
                'use_residual': False,
            },
        ]
        
        for config in configs:
            print(f"\n{'='*70}")
            print(f"Testing: {config['name']}")
            print(f"{'='*70}")
            
            model = create_gatv2_model(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=128,
                num_layers=4,
                num_heads=4,
                dropout=0.2,
                use_residual=config.get('use_residual', True),
                use_jk=config.get('use_jk', True),
                jk_mode=config.get('jk_mode', 'cat'),
                pooling=config.get('pooling', 'set2set')
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0
            )
            
            # Train with focal loss
            history = train_gatv2_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=0.001,
                weight_decay=1e-5,
                device=device,
                loss_type='focal',
                early_stopping_patience=15,
                early_stopping_metric='f1',
                verbose=False
            )
            
            # Evaluate
            test_metrics = evaluate_model(model, test_loader, device=device)
            
            results[config['name']] = {
                'test_auc_roc': test_metrics['auc_roc'],
                'test_f1': test_metrics['f1'],
                'test_pr_auc': test_metrics['pr_auc'],
                'test_accuracy': test_metrics['accuracy']
            }
            
            print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")
            print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    
    elif study_type == 'loss':
        # Test different loss functions
        loss_configs = [
            {'name': 'Standard BCE', 'loss_type': 'bce', 'use_weighted_sampler': False},
            {'name': 'Weighted BCE', 'loss_type': 'weighted_bce', 'use_weighted_sampler': False},
            {'name': 'Focal Loss (alpha=0.25, gamma=2.0)', 'loss_type': 'focal', 
             'focal_alpha': 0.25, 'focal_gamma': 2.0, 'use_weighted_sampler': False},
            {'name': 'Focal Loss + Weighted Sampler', 'loss_type': 'focal',
             'focal_alpha': 0.25, 'focal_gamma': 2.0, 'use_weighted_sampler': True},
        ]
        
        # Compute positive class weight
        train_labels = [data.y.item() for data in train_dataset]
        num_pos = sum(train_labels)
        num_neg = len(train_labels) - num_pos
        pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        for config in loss_configs:
            print(f"\n{'='*70}")
            print(f"Testing: {config['name']}")
            print(f"{'='*70}")
            
            model = create_gatv2_model(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=128,
                num_layers=4,
                num_heads=4,
                dropout=0.2,
                use_residual=True,
                use_jk=True,
                jk_mode='cat',
                pooling='set2set'
            )
            
            # Create data loader with/without weighted sampler
            train_sampler = None
            if config.get('use_weighted_sampler', False):
                train_sampler = create_balanced_sampler(train_labels)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                collate_fn=collate_fn,
                num_workers=0
            )
            
            # Train
            history = train_gatv2_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=0.001,
                weight_decay=1e-5,
                device=device,
                loss_type=config['loss_type'],
                focal_alpha=config.get('focal_alpha', 0.25),
                focal_gamma=config.get('focal_gamma', 2.0),
                pos_weight=pos_weight if config['loss_type'] == 'weighted_bce' else None,
                early_stopping_patience=15,
                early_stopping_metric='f1',
                verbose=False
            )
            
            # Evaluate
            test_metrics = evaluate_model(model, test_loader, device=device)
            
            results[config['name']] = {
                'test_auc_roc': test_metrics['auc_roc'],
                'test_f1': test_metrics['f1'],
                'test_pr_auc': test_metrics['pr_auc'],
                'test_accuracy': test_metrics['accuracy']
            }
            
            print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")
            print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    
    elif study_type == 'sampling':
        # Test different sampling strategies
        sampling_configs = [
            {'name': 'No Weighted Sampler', 'use_weighted_sampler': False},
            {'name': 'Weighted Sampler', 'use_weighted_sampler': True},
        ]
        
        for config in sampling_configs:
            print(f"\n{'='*70}")
            print(f"Testing: {config['name']}")
            print(f"{'='*70}")
            
            model = create_gatv2_model(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=128,
                num_layers=4,
                num_heads=4,
                dropout=0.2,
                use_residual=True,
                use_jk=True,
                jk_mode='cat',
                pooling='set2set'
            )
            
            # Create data loader
            train_sampler = None
            if config.get('use_weighted_sampler', False):
                train_labels = [data.y.item() for data in train_dataset]
                train_sampler = create_balanced_sampler(train_labels)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                collate_fn=collate_fn,
                num_workers=0
            )
            
            # Train with focal loss
            history = train_gatv2_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=0.001,
                weight_decay=1e-5,
                device=device,
                loss_type='focal',
                early_stopping_patience=15,
                early_stopping_metric='f1',
                verbose=False
            )
            
            # Evaluate
            test_metrics = evaluate_model(model, test_loader, device=device)
            
            results[config['name']] = {
                'test_auc_roc': test_metrics['auc_roc'],
                'test_f1': test_metrics['f1'],
                'test_pr_auc': test_metrics['pr_auc'],
                'test_accuracy': test_metrics['accuracy']
            }
            
            print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")
            print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies for GATv2 model')
    parser.add_argument(
        '--study_type',
        type=str,
        choices=['architecture', 'loss', 'sampling'],
        default='architecture',
        help='Type of ablation study to run'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=50,
        help='Number of epochs for each configuration'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu or cuda)'
    )
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print("=" * 70)
    print(f"GATv2 Ablation Study: {args.study_type}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs per configuration: {args.num_epochs}")
    print()
    
    # Load dataset
    print("Loading ClinTox dataset...")
    train_df, val_df, test_df = load_clintox(
        cache_dir=str(project_root / "data"),
        split_type="scaffold",
        seed=42
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print()
    
    # Convert to graph format
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
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print()
    
    # Run ablation study
    results = run_ablation_study(
        train_dataset,
        val_dataset,
        test_dataset,
        device=device,
        num_epochs=args.num_epochs,
        study_type=args.study_type
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary of Results")
    print("=" * 70)
    
    df = pd.DataFrame(results).T
    df = df.sort_values('test_f1', ascending=False)
    print(df.to_string())
    
    # Save results
    output_path = project_root / f"models/gatv2_model/ablation_{args.study_type}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

