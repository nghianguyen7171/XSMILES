#!/usr/bin/env python3
"""
Generate sample visualization comparing the two best models: SMILESTransformer vs SMILESGNN.

Shows predictions from SMILESTransformer and SMILESGNN on the same molecules,
highlighting where they agree, disagree, and their relative strengths.
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
from matplotlib.patches import Rectangle
from rdkit import Chem
from rdkit.Chem import Draw

from src.data import load_clintox
from src.utils import set_seed
from src.pipelines import load_mlp_model, load_gnn_model

# Focus on the two best models
TOP2_MODELS = [
    'SMILESTransformer (torch-molecule)',
    'SMILESGNN (PyTorch Geometric)'
]

# Import dependencies for model loading
try:
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from src.graph_data import smiles_list_to_pyg_dataset
    from src.graph_train import evaluate_model as evaluate_graph_model
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    import deepchem as dc
    from deepchem.feat import DMPNNFeaturizer
    from deepchem.data import NumpyDataset
    DEEPCHEM_AVAILABLE = True
except ImportError:
    DEEPCHEM_AVAILABLE = False


def load_predictions(model_name, test_df, device='cpu'):
    """Load predictions from a trained model."""
    test_smiles = test_df['smiles'].tolist()
    test_labels = test_df['CT_TOX'].values
    
    # Copy the implementation from generate_sample_visualizations.py
    # (simplified to only handle the two models we need)
    if model_name == 'SMILESTransformer (torch-molecule)':
        try:
            models_dir = project_root / "models"
            model = None
            from src.pipelines import load_gnn_model
            
            model_path = models_dir / "smilestransformer_model.pt"
            if model_path.exists():
                model = load_gnn_model(model_path)
            else:
                import pickle
                pickle_path = models_dir / "smilestransformer_model.pkl"
                if pickle_path.exists():
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
            return None, None
    
    elif model_name == 'SMILESGNN (PyTorch Geometric)' and PYG_AVAILABLE:
        try:
            model_path = project_root / "models" / "smilesgnn_model" / "best_model.pt"
            from src.graph_models_hybrid import create_hybrid_model
            from src.graph_data import get_feature_dims
            from src.smiles_tokenizer import SMILESTokenizer
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
                # Model saved as state_dict directly
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
            
            # Create dataset
            test_dataset = smiles_list_to_pyg_dataset(test_smiles, labels=test_labels.tolist())
            
            class SMILESGNNDataset:
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
            
            def collate_fn_smilesgnn(batch):
                batch_data = Batch.from_data_list(batch)
                if hasattr(batch[0], 'smiles_token_ids'):
                    smiles_token_ids = torch.stack([item.smiles_token_ids for item in batch])
                    smiles_attention_masks = torch.stack([item.smiles_attention_mask for item in batch])
                    batch_data.smiles_token_ids = smiles_token_ids
                    batch_data.smiles_attention_masks = smiles_attention_masks
                return batch_data
            
            smilesgnn_dataset = SMILESGNNDataset(test_dataset, test_smiles, tokenizer)
            test_loader = DataLoader(smilesgnn_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_smilesgnn, num_workers=0)
            
            class SMILESGNNModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, batch):
                    return self.model(
                        batch,
                        smiles_token_ids=batch.smiles_token_ids if hasattr(batch, 'smiles_token_ids') else None,
                        smiles_attention_mask=batch.smiles_attention_masks if hasattr(batch, 'smiles_attention_masks') else None
                    )
            
            wrapped_model = SMILESGNNModelWrapper(model)
            metrics = evaluate_graph_model(wrapped_model, test_loader, device=device, return_predictions=True)
            return metrics['labels'], metrics['predictions']
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    return None, None


def visualize_top2_comparison(smiles_list, labels, predictions_dict, n_cols=4, title_prefix="", save_path=None):
    """
    Visualize molecules comparing SMILESTransformer vs SMILESGNN predictions.
    
    Args:
        smiles_list: List of SMILES strings
        labels: True labels (0 or 1)
        predictions_dict: Dict of {model_name: [probabilities]}
        n_cols: Number of columns
        title_prefix: Title prefix
        save_path: Path to save figure
    """
    n_mols = len(smiles_list)
    n_rows = (n_mols + n_cols - 1) // n_cols
    
    # Larger figure size for clarity
    fig = plt.figure(figsize=(10 * n_cols, 11 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3, 
                         left=0.05, right=0.95, top=0.96, bottom=0.04)
    
    model_display_names = {
        'SMILESTransformer (torch-molecule)': 'SMILESTrans',
        'SMILESGNN (PyTorch Geometric)': 'SMILESGNN'
    }
    
    toxic_color = '#d32f2f'  # Deep red
    nontoxic_color = '#388e3c'  # Deep green
    
    for idx, (smiles, true_label) in enumerate(zip(smiles_list, labels)):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('white')
        
        # Draw molecule
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(700, 700))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Invalid\nSMILES', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='gray')
        except Exception as e:
            ax.text(0.5, 0.5, f'Render\nError', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, color='red')
        
        # Create prediction box
        true_str = "Toxic" if true_label == 1 else "Non-toxic"
        true_color = toxic_color if true_label == 1 else nontoxic_color
        
        pred_lines = [f"True Label: {true_str}"]
        pred_lines.append("")
        
        for model_name in TOP2_MODELS:
            if model_name in predictions_dict:
                probs = predictions_dict[model_name]
                prob = probs[idx] if idx < len(probs) else 0.0
                pred_label = "Toxic" if prob > 0.5 else "Non-toxic"
                correct = "✓" if (true_label == 1 and prob > 0.5) or (true_label == 0 and prob <= 0.5) else "✗"
                display_name = model_display_names.get(model_name, model_name)
                
                pred_color = toxic_color if prob > 0.5 else nontoxic_color
                if correct == "✗":
                    pred_color = '#ff9800'  # Orange for incorrect
                
                pred_lines.append(f"{display_name}: {prob:.3f} ({pred_label}) {correct}")
        
        pred_text = "\n".join(pred_lines)
        
        # Add styled text box
        bbox_props = dict(boxstyle='round,pad=1.0', 
                         facecolor='white', 
                         edgecolor=true_color, 
                         linewidth=2.5,
                         alpha=0.95)
        
        ax.text(0.5, 0.98, pred_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=13, fontweight='bold',
               bbox=bbox_props, family='monospace',
               color='black')
        
        ax.axis('off')
    
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=20, fontweight='bold', 
                    y=0.99, color='#212121', family='sans-serif')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved to: {save_path}")
    
    plt.close()


def main():
    """Generate comparison visualizations for SMILESTransformer vs SMILESGNN."""
    set_seed(42)
    
    # Load test set
    print("Loading test set...")
    _, _, test_df = load_clintox(
        cache_dir=str(project_root / "data"),
        split_type="scaffold",
        seed=42
    )
    
    print(f"\nLoaded {len(test_df)} test samples")
    print(f"Class distribution: Toxic={test_df['CT_TOX'].sum()}, Non-toxic={len(test_df) - test_df['CT_TOX'].sum()}")
    
    # Load predictions for top 2 models
    print("\nLoading predictions from top 2 models...")
    all_predictions = {}
    
    for model_name in TOP2_MODELS:
        labels, probs = load_predictions(model_name, test_df)
        if labels is not None and probs is not None:
            all_predictions[model_name] = probs
            print(f"  ✓ {model_name}: {len(probs)} predictions")
        else:
            print(f"  ✗ {model_name}: Failed to load predictions")
    
    if len(all_predictions) < 2:
        print("\n⚠ Error: Could not load predictions from both models")
        return
    
    # Create DataFrame with predictions
    pred_df = test_df[['smiles', 'CT_TOX']].copy()
    for model_name, probs in all_predictions.items():
        pred_df[model_name] = probs
        pred_df[f'{model_name}_pred'] = (probs > 0.5).astype(int)
        pred_df[f'{model_name}_correct'] = (pred_df['CT_TOX'] == pred_df[f'{model_name}_pred'])
    
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 1. Both models correct
    print("\n1. Generating visualizations where both models are correct...")
    both_correct = pred_df.copy()
    for model_name in TOP2_MODELS:
        both_correct = both_correct[both_correct[f'{model_name}_correct'] == True]
    
    print(f"  Found {len(both_correct)} samples where both models are correct")
    
    if len(both_correct) > 0:
        # Sample correct toxic
        correct_toxic = both_correct[both_correct['CT_TOX'] == 1].head(6)
        if len(correct_toxic) > 0:
            predictions_dict = {name: all_predictions[name] for name in TOP2_MODELS}
            toxic_indices = correct_toxic.index.tolist()
            toxic_smiles = correct_toxic['smiles'].tolist()
            toxic_labels = correct_toxic['CT_TOX'].values
            
            toxic_predictions = {}
            for model_name, probs in predictions_dict.items():
                toxic_predictions[model_name] = [probs[i] for i in toxic_indices]
            
            visualize_top2_comparison(
                toxic_smiles,
                toxic_labels,
                toxic_predictions,
                n_cols=3,
                title_prefix="SMILESTransformer vs SMILESGNN: Both Correct - Toxic Predictions",
                save_path=results_dir / "top2_comparison_both_correct_toxic.png"
            )
        
        # Sample correct non-toxic
        correct_nontoxic = both_correct[both_correct['CT_TOX'] == 0].head(6)
        if len(correct_nontoxic) > 0:
            nontoxic_indices = correct_nontoxic.index.tolist()
            nontoxic_smiles = correct_nontoxic['smiles'].tolist()
            nontoxic_labels = correct_nontoxic['CT_TOX'].values
            
            nontoxic_predictions = {}
            for model_name, probs in all_predictions.items():
                nontoxic_predictions[model_name] = [probs[i] for i in nontoxic_indices]
            
            visualize_top2_comparison(
                nontoxic_smiles,
                nontoxic_labels,
                nontoxic_predictions,
                n_cols=3,
                title_prefix="SMILESTransformer vs SMILESGNN: Both Correct - Non-toxic Predictions",
                save_path=results_dir / "top2_comparison_both_correct_nontoxic.png"
            )
    
    # 2. SMILESGNN correct but SMILESTransformer wrong
    print("\n2. Generating visualizations where SMILESGNN is correct but SMILESTransformer is wrong...")
    smilesgnn_correct_st_wrong = pred_df[
        (pred_df['SMILESGNN (PyTorch Geometric)_correct'] == True) &
        (pred_df['SMILESTransformer (torch-molecule)_correct'] == False)
    ]
    
    print(f"  Found {len(smilesgnn_correct_st_wrong)} samples where SMILESGNN is correct but SMILESTransformer is wrong")
    
    if len(smilesgnn_correct_st_wrong) > 0:
        sample = smilesgnn_correct_st_wrong.head(6)
        sample_indices = sample.index.tolist()
        sample_smiles = sample['smiles'].tolist()
        sample_labels = sample['CT_TOX'].values
        
        sample_predictions = {}
        for model_name, probs in all_predictions.items():
            sample_predictions[model_name] = [probs[i] for i in sample_indices]
        
        visualize_top2_comparison(
            sample_smiles,
            sample_labels,
            sample_predictions,
            n_cols=3,
            title_prefix="SMILESGNN vs SMILESTransformer: SMILESGNN Correct, SMILESTransformer Wrong",
            save_path=results_dir / "top2_comparison_smilesgnn_wins.png"
        )
    
    # 3. SMILESTransformer correct but SMILESGNN wrong
    print("\n3. Generating visualizations where SMILESTransformer is correct but SMILESGNN is wrong...")
    st_correct_smilesgnn_wrong = pred_df[
        (pred_df['SMILESTransformer (torch-molecule)_correct'] == True) &
        (pred_df['SMILESGNN (PyTorch Geometric)_correct'] == False)
    ]
    
    print(f"  Found {len(st_correct_smilesgnn_wrong)} samples where SMILESTransformer is correct but SMILESGNN is wrong")
    
    if len(st_correct_smilesgnn_wrong) > 0:
        sample = st_correct_smilesgnn_wrong.head(6)
        sample_indices = sample.index.tolist()
        sample_smiles = sample['smiles'].tolist()
        sample_labels = sample['CT_TOX'].values
        
        sample_predictions = {}
        for model_name, probs in all_predictions.items():
            sample_predictions[model_name] = [probs[i] for i in sample_indices]
        
        visualize_top2_comparison(
            sample_smiles,
            sample_labels,
            sample_predictions,
            n_cols=3,
            title_prefix="SMILESTransformer vs SMILESGNN: SMILESTransformer Correct, SMILESGNN Wrong",
            save_path=results_dir / "top2_comparison_smilestransformer_wins.png"
        )
    
    # 4. Diverse samples showing both models
    print("\n4. Generating diverse sample comparisons...")
    diverse_indices = []
    seen_mol_weights = set()
    
    for idx, row in pred_df.iterrows():
        try:
            from rdkit.Chem import rdMolDescriptors
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
                weight_bin = int(mol_weight / 50) * 50
                if weight_bin not in seen_mol_weights or len(seen_mol_weights) < 12:
                    diverse_indices.append(idx)
                    seen_mol_weights.add(weight_bin)
                    if len(diverse_indices) >= 12:
                        break
        except:
            continue
    
    if len(diverse_indices) < 12:
        diverse_indices = list(range(min(12, len(pred_df))))
    
    diverse_df = pred_df.loc[diverse_indices[:12]]
    diverse_smiles = diverse_df['smiles'].tolist()
    diverse_labels = diverse_df['CT_TOX'].values
    
    diverse_predictions = {}
    for model_name, probs in all_predictions.items():
        diverse_predictions[model_name] = [probs[i] for i in diverse_indices[:12]]
    
    visualize_top2_comparison(
        diverse_smiles,
        diverse_labels,
        diverse_predictions,
        n_cols=4,
        title_prefix="SMILESTransformer vs SMILESGNN: Comparison on Diverse Molecular Structures",
        save_path=results_dir / "top2_comparison_diverse.png"
    )
    
    print("\n✓ Top 2 model comparison visualizations generated successfully!")


if __name__ == "__main__":
    main()

