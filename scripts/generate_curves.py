#!/usr/bin/env python3
"""
Generate ROC and PR curves for all models on the test set.

This script loads predictions from all trained models and generates
comparison plots of ROC curves and Precision-Recall curves.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.data import load_clintox
from src.train import evaluate_model
from src.pipelines import load_mlp_model, load_gnn_model
from src.utils import set_seed

# Try to load PyG models
try:
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from src.graph_data import smiles_list_to_pyg_dataset
    from src.graph_train import evaluate_model as evaluate_graph_model
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# Try to load DeepChem models
try:
    import deepchem as dc
    from deepchem.feat import DMPNNFeaturizer
    from deepchem.data import NumpyDataset
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False


def collate_fn(batch):
    """Collate function for PyG DataLoader."""
    return Batch.from_data_list(batch)


def load_predictions(model_name, test_df, device='cpu'):
    """Load predictions from a trained model."""
    test_smiles = test_df['smiles'].tolist()
    test_labels = test_df['CT_TOX'].values
    
    if model_name == 'Baseline MLP':
        # Load baseline model
        try:
            models_dir = project_root / "models"
            model_path = models_dir / "baseline_mlp_model.pt"
            if not model_path.exists():
                return None, None
            from src.utils import get_default_config
            config = get_default_config()
            model = load_mlp_model(model_path, config)
            from src.featurization import featurize_batch
            test_fps = featurize_batch(test_smiles, mode='fingerprint')
            test_labels_2d = test_labels.reshape(-1, 1)
            
            with torch.no_grad():
                test_tensors = torch.FloatTensor(test_fps)
                logits = model(test_tensors)
                probs = torch.sigmoid(logits).numpy().flatten()
            if isinstance(probs, np.ndarray):
                if len(probs.shape) == 2:
                    probs = probs[:, 0]
            return test_labels, probs
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None, None
    
    elif 'torch-molecule' in model_name.lower():
        # Load torch-molecule models
        try:
            models_dir = project_root / "models"
            model = None
            
            if 'BFGNN' in model_name:
                model_path = models_dir / "torch_molecule_model.pt"
                if model_path.exists():
                    model = load_gnn_model(model_path)
                else:
                    pickle_path = models_dir / "torch_molecule_model.pkl"
                    if pickle_path.exists():
                        import pickle
                        with open(pickle_path, 'rb') as f:
                            model = pickle.load(f)
            elif 'GRIN' in model_name:
                model_path = models_dir / "grin_model.pt"
                if model_path.exists():
                    model = load_gnn_model(model_path)
                else:
                    pickle_path = models_dir / "grin_model.pkl"
                    if pickle_path.exists():
                        import pickle
                        with open(pickle_path, 'rb') as f:
                            model = pickle.load(f)
            elif 'SMILESTransformer' in model_name or 'SMILESTRANSFORMER' in model_name:
                model_path = models_dir / "smilestransformer_model.pt"
                if model_path.exists():
                    model = load_gnn_model(model_path)
                else:
                    pickle_path = models_dir / "smilestransformer_model.pkl"
                    if pickle_path.exists():
                        import pickle
                        with open(pickle_path, 'rb') as f:
                            model = pickle.load(f)
            
            if model is None:
                return None, None
            
            from src.train import predict_with_torch_molecule_model
            probs = predict_with_torch_molecule_model(model, test_smiles)
            if isinstance(probs, np.ndarray):
                if len(probs.shape) == 2:
                    probs = probs[:, 0]
            return test_labels, probs
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    elif 'DMPNN' in model_name and DEEPCHEM_AVAILABLE:
        # Load DeepChem DMPNN
        try:
            models_dir = project_root / "models" / "dmpnn_model"
            import pickle
            
            with open(models_dir / "model_config.pkl", 'rb') as f:
                model_config = pickle.load(f)
            
            from deepchem.models import DMPNNModel
            model = DMPNNModel(
                n_tasks=1, mode='classification',
                atom_fdim=133, bond_fdim=14,
                enc_hidden=300, depth=4, ffn_hidden=300,
                enc_dropout_p=0.25, ffn_dropout_p=0.25,
                batch_size=50, learning_rate=0.001, device='cpu',
                model_dir=str(models_dir)
            )
            model.restore()
            
            featurizer = DMPNNFeaturizer()
            test_features = featurizer.featurize(test_smiles)
            test_labels_dc = test_labels.reshape(-1, 1)
            test_dataset = NumpyDataset(X=test_features, y=test_labels_dc)
            
            outputs = model.predict(test_dataset)
            # Handle different output shapes
            if len(outputs.shape) == 3:
                probs = outputs[:, 0, 1]
            elif len(outputs.shape) == 2:
                if outputs.shape[1] == 1:
                    probs = outputs[:, 0]
                else:
                    probs = outputs[:, 1]
            else:
                probs = outputs
            probs = np.clip(probs, 0.0, 1.0)
            return test_labels, probs
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None, None
    
    elif PYG_AVAILABLE:
        # Load PyG models (GATv2, GIN, SMILESGNN)
        try:
            if 'GATv2' in model_name:
                model_path = project_root / "models" / "gatv2_model" / "best_model.pt"
                from src.graph_models import GATv2MolecularPredictor, create_gatv2_model
                from src.graph_data import get_feature_dims
                num_node_features, num_edge_features = get_feature_dims()
                
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model_config = checkpoint.get('model_config', {})
                    model = create_gatv2_model(
                        num_node_features=num_node_features,
                        num_edge_features=num_edge_features,
                        hidden_dim=model_config.get('hidden_dim', 128),
                        num_layers=model_config.get('num_layers', 4),
                        num_heads=model_config.get('num_heads', 4),
                        dropout=model_config.get('dropout', 0.2),
                        use_residual=model_config.get('use_residual', True),
                        use_jk=model_config.get('use_jk', True),
                        jk_mode=model_config.get('jk_mode', 'cat'),
                        pooling=model_config.get('pooling', 'set2set')
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model = create_gatv2_model(
                        num_node_features=num_node_features,
                        num_edge_features=num_edge_features,
                        hidden_dim=128, num_layers=4, num_heads=4, dropout=0.2,
                        use_residual=True, use_jk=True, jk_mode='cat', pooling='set2set'
                    )
                    if isinstance(checkpoint, dict):
                        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                    else:
                        model.load_state_dict(checkpoint)
                model.eval()
                
            elif 'GIN' in model_name:
                model_path = project_root / "models" / "gin_model" / "best_model.pt"
                from src.graph_models_gin import GINMolecularPredictor, create_gin_model
                from src.graph_data import get_feature_dims
                num_node_features, num_edge_features = get_feature_dims()
                
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model_config = checkpoint.get('model_config', {})
                    model = create_gin_model(
                        num_node_features=num_node_features,
                        num_edge_features=num_edge_features,
                        hidden_dim=model_config.get('hidden_dim', 128),
                        num_layers=model_config.get('num_layers', 4),
                        dropout=model_config.get('dropout', 0.2),
                        use_residual=model_config.get('use_residual', True),
                        use_jk=model_config.get('use_jk', True),
                        jk_mode=model_config.get('jk_mode', 'cat'),
                        pooling=model_config.get('pooling', 'meanmax')
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model = create_gin_model(
                        num_node_features=num_node_features,
                        num_edge_features=num_edge_features,
                        hidden_dim=128, num_layers=4, dropout=0.2,
                        use_residual=True, use_jk=True, jk_mode='cat', pooling='meanmax'
                    )
                    model.load_state_dict(checkpoint)
                model.eval()
                
            elif 'SMILESGNN' in model_name:
                model_path = project_root / "models" / "smilesgnn_model" / "best_model.pt"
                from src.graph_models_hybrid import create_hybrid_model
                from src.graph_data import get_feature_dims
                from src.smiles_tokenizer import create_tokenizer_from_smiles
                import pickle
                
                num_node_features, num_edge_features = get_feature_dims()
                
                # Load tokenizer
                tokenizer_path = project_root / "models" / "smilesgnn_model" / "tokenizer.pkl"
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model_config = checkpoint.get('model_config', {})
                    model = create_hybrid_model(
                        num_node_features=num_node_features,
                        num_edge_features=num_edge_features,
                        hidden_dim=model_config.get('hidden_dim', 96),
                        num_graph_layers=model_config.get('num_graph_layers', 3),
                        graph_model=model_config.get('graph_model', 'gatv2'),
                        num_heads=model_config.get('num_heads', 4),
                        dropout=model_config.get('dropout', 0.4),
                        use_residual=model_config.get('use_residual', True),
                        use_jk=model_config.get('use_jk', True),
                        jk_mode=model_config.get('jk_mode', 'cat'),
                        graph_pooling=model_config.get('graph_pooling', 'meanmax'),
                        smiles_vocab_size=len(tokenizer.token_to_id),
                        smiles_d_model=model_config.get('smiles_d_model', 96),
                        smiles_num_layers=model_config.get('smiles_num_layers', 2),
                        fusion_method=model_config.get('fusion_method', 'attention')
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model = create_hybrid_model(
                        num_node_features=num_node_features,
                        num_edge_features=num_edge_features,
                        hidden_dim=96, num_graph_layers=3, graph_model='gatv2', num_heads=4,
                        dropout=0.4, use_residual=True, use_jk=True, jk_mode='cat',
                        graph_pooling='meanmax',
                        smiles_vocab_size=len(tokenizer.token_to_id),
                        smiles_d_model=96, smiles_num_layers=2, fusion_method='attention'
                    )
                    if isinstance(checkpoint, dict):
                        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                    else:
                        model.load_state_dict(checkpoint)
                model.eval()
                
                # Create hybrid dataset
                test_dataset = smiles_list_to_pyg_dataset(test_smiles, labels=test_labels.tolist())
                
                class HybridDataset:
                    def __init__(self, pyg_dataset, smiles_list, tokenizer):
                        self.pyg_dataset = pyg_dataset
                        self.smiles_list = smiles_list
                        self.tokenizer = tokenizer
                    def __len__(self): return len(self.pyg_dataset)
                    def __getitem__(self, idx):
                        data = self.pyg_dataset[idx]
                        smiles = self.smiles_list[idx]
                        token_ids, attention_mask = self.tokenizer.encode(smiles)
                        data.smiles_token_ids = torch.tensor(token_ids, dtype=torch.long)
                        data.smiles_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                        return data
                
                def collate_fn_hybrid(batch):
                    batch_data = Batch.from_data_list(batch)
                    if hasattr(batch[0], 'smiles_token_ids'):
                        smiles_token_ids = torch.stack([item.smiles_token_ids for item in batch])
                        smiles_attention_masks = torch.stack([item.smiles_attention_mask for item in batch])
                        batch_data.smiles_token_ids = smiles_token_ids
                        batch_data.smiles_attention_masks = smiles_attention_masks
                    return batch_data
                
                hybrid_dataset = HybridDataset(test_dataset, test_smiles, tokenizer)
                test_loader = DataLoader(hybrid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_hybrid, num_workers=0)
                
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
                metrics = evaluate_graph_model(wrapped_model, test_loader, device=device, return_predictions=True)
                return metrics['labels'], metrics['predictions']
            
            if 'SMILESGNN' not in model_name:
                # For GATv2 and GIN
                test_dataset = smiles_list_to_pyg_dataset(test_smiles, labels=test_labels.tolist())
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)
                metrics = evaluate_graph_model(model, test_loader, device=device, return_predictions=True)
                return metrics['labels'], metrics['predictions']
                
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    return None, None


def main():
    """Generate ROC and PR curves for all models."""
    set_seed(42)
    
    # Load test set
    print("Loading test set...")
    _, _, test_df = load_clintox(
        cache_dir=str(project_root / "data"),
        split_type="scaffold",
        seed=42
    )
    
    # List of models to evaluate
    models = [
        'Baseline MLP',
        'BFGNN (torch-molecule)',
        'GRIN (torch-molecule)',
        'SMILESTransformer (torch-molecule)',
        'DMPNN (DeepChem)',
        'GATv2 (PyTorch Geometric)',
        'GIN (PyTorch Geometric)',
        'SMILESGNN (PyTorch Geometric)'
    ]
    
    # Load predictions for each model
    print("\nLoading predictions from all models...")
    all_labels = None
    all_predictions = {}
    
    for model_name in models:
        print(f"  Loading {model_name}...")
        labels, probs = load_predictions(model_name, test_df)
        if labels is not None and probs is not None:
            all_labels = labels  # Should be same for all models
            all_predictions[model_name] = probs
            print(f"    ✓ Loaded {len(probs)} predictions")
        else:
            print(f"    ✗ Failed to load {model_name}")
    
    if all_labels is None or len(all_predictions) == 0:
        print("Error: No predictions loaded. Exiting.")
        return
    
    # Create output directory
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Generate ROC curves
    print("\nGenerating ROC curves...")
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_predictions)))
    for (model_name, probs), color in zip(all_predictions.items(), colors):
        fpr, tpr, _ = roc_curve(all_labels, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', 
                linewidth=2, color=color)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = results_dir / "roc_curves_all_models.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"  Saved ROC curves to: {roc_path}")
    plt.close()
    
    # Generate PR curves
    print("\nGenerating PR curves...")
    plt.figure(figsize=(10, 8))
    
    for (model_name, probs), color in zip(all_predictions.items(), colors):
        precision, recall, _ = precision_recall_curve(all_labels, probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.4f})',
                linewidth=2, color=color)
    
    # Baseline (random classifier)
    baseline = np.sum(all_labels) / len(all_labels)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
               label=f'Random (AUC = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - All Models Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pr_path = results_dir / "pr_curves_all_models.png"
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    print(f"  Saved PR curves to: {pr_path}")
    plt.close()
    
    print("\n✓ All curves generated successfully!")


if __name__ == "__main__":
    main()

