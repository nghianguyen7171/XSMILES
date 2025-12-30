#!/usr/bin/env python3
"""
Generate sample visualization results for all models.

Shows predictions from all models on the same molecules, highlighting
correct and incorrect predictions.
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
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from src.data import load_clintox
from src.utils import set_seed
from src.pipelines import load_mlp_model, load_gnn_model

# Select representative models (best performing and diverse architectures)
# This ensures larger, clearer images with fewer models per visualization
REPRESENTATIVE_MODELS = [
    'SMILESGNN (PyTorch Geometric)',      # Best overall - multimodal
    'SMILESTransformer (torch-molecule)', # Best sequence-based  
    'GIN (PyTorch Geometric)',            # Good graph-based
    'Baseline MLP'                         # Baseline reference
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

def collate_fn(batch):
    """Collate function for PyG DataLoader."""
    return Batch.from_data_list(batch)


def load_predictions(model_name, test_df, device='cpu'):
    """Load predictions from a trained model."""
    test_smiles = test_df['smiles'].tolist()
    test_labels = test_df['CT_TOX'].values
    
    if model_name == 'Baseline MLP':
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
            with torch.no_grad():
                test_tensors = torch.FloatTensor(test_fps)
                logits = model(test_tensors)
                probs = torch.sigmoid(logits).numpy().flatten()
            return test_labels, probs
        except Exception as e:
            return None, None
    
    elif 'torch-molecule' in model_name.lower():
        try:
            models_dir = project_root / "models"
            model = None
            if 'BFGNN' in model_name:
                model_path = models_dir / "torch_molecule_model.pt"
                if model_path.exists():
                    model = load_gnn_model(model_path)
                else:
                    import pickle
                    pickle_path = models_dir / "torch_molecule_model.pkl"
                    if pickle_path.exists():
                        with open(pickle_path, 'rb') as f:
                            model = pickle.load(f)
            elif 'GRIN' in model_name:
                model_path = models_dir / "grin_model.pt"
                if model_path.exists():
                    model = load_gnn_model(model_path)
                else:
                    import pickle
                    pickle_path = models_dir / "grin_model.pkl"
                    if pickle_path.exists():
                        with open(pickle_path, 'rb') as f:
                            model = pickle.load(f)
            elif 'SMILESTransformer' in model_name:
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
        except:
            return None, None
    
    elif 'DMPNN' in model_name and DEEPCHEM_AVAILABLE:
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
            if len(outputs.shape) == 3:
                probs = outputs[:, 0, 1]
            elif len(outputs.shape) == 2:
                probs = outputs[:, 1] if outputs.shape[1] > 1 else outputs[:, 0]
            else:
                probs = outputs
            probs = np.clip(probs, 0.0, 1.0)
            return test_labels, probs
        except:
            return None, None
    
    elif PYG_AVAILABLE:
        try:
            if 'GIN' in model_name:
                model_path = project_root / "models" / "gin_model" / "best_model.pt"
                from src.graph_models_gin import create_gin_model
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
                test_dataset = smiles_list_to_pyg_dataset(test_smiles, labels=test_labels.tolist())
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)
                metrics = evaluate_graph_model(model, test_loader, device=device, return_predictions=True)
                return metrics['labels'], metrics['predictions']
            elif 'SMILESGNN' in model_name:
                model_path = project_root / "models" / "smilesgnn_model" / "best_model.pt"
                from src.graph_models_hybrid import create_hybrid_model
                from src.graph_data import get_feature_dims
                import pickle
                num_node_features, num_edge_features = get_feature_dims()
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
        except:
            return None, None
    
    return None, None

# Import visualization function from notebooks pattern
def visualize_molecule_grid(smiles_list, labels_list, n_cols=3, title_prefix="", save_path=None):
    """
    Visualize molecules in a grid with prediction labels above each.
    Uses larger figure size for clearer images with representative models.
    
    Args:
        smiles_list: List of SMILES strings
        labels_list: List of label strings (e.g., "True: Toxic\nMLP: 0.5 | BFGNN: 0.3")
        n_cols: Number of columns (default 3 for larger images)
        title_prefix: Prefix for overall title
        save_path: Path to save figure
    """
    n_mols = len(smiles_list)
    n_rows = (n_mols + n_cols - 1) // n_cols
    
    # Larger figure size for clarity
    fig = plt.figure(figsize=(8 * n_cols, 9 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.25, 
                         left=0.05, right=0.95, top=0.96, bottom=0.04)
    
    for idx, (smiles, label_str) in enumerate(zip(smiles_list, labels_list)):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        # Parse label string to determine colors
        is_toxic = "True: Toxic" in label_str
        box_color = '#ffebee' if is_toxic else '#e8f5e9'
        
        # Draw prediction box
        box = Rectangle((0, 0.9), 1, 0.1, transform=ax.transAxes,
                       facecolor=box_color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        
        # Add label text with larger font for clarity
        ax.text(0.5, 0.95, label_str, transform=ax.transAxes,
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color, 
                        alpha=0.95, edgecolor='black', linewidth=1.5))
        
        # Draw molecule with larger size for clarity
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Larger molecule image for better visibility
                img = Draw.MolToImage(mol, size=(600, 600))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"Invalid SMILES:\n{smiles}", 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12)
        except:
            ax.text(0.5, 0.5, f"Error rendering:\n{smiles[:30]}...", 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12)
        
        ax.axis('off')
    
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=18, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")
    
    plt.close()


def format_prediction_string(true_label, predictions_dict, model_display_names=None):
    """
    Format prediction string for display with representative models only.
    
    Args:
        true_label: True label (0 or 1)
        predictions_dict: Dict of {model_name: probability}
        model_display_names: Optional mapping for shorter names
    
    Returns:
        Formatted string
    """
    true_str = "Toxic" if true_label == 1 else "Non-toxic"
    pred_strs = []
    
    # Default display names for representative models
    if model_display_names is None:
        model_display_names = {
            'SMILESGNN (PyTorch Geometric)': 'SMILESGNN',
            'SMILESTransformer (torch-molecule)': 'SMILESTrans',
            'GIN (PyTorch Geometric)': 'GIN',
            'Baseline MLP': 'MLP'
        }
    
    # Only include representative models
    for model_name in REPRESENTATIVE_MODELS:
        if model_name in predictions_dict:
            prob = predictions_dict[model_name]
            display_name = model_display_names.get(model_name, model_name)
            pred_label = "Toxic" if prob > 0.5 else "Non-toxic"
            correct = "✓" if (true_label == 1 and prob > 0.5) or (true_label == 0 and prob <= 0.5) else "✗"
            pred_strs.append(f"{display_name}: {prob:.3f} ({pred_label}) {correct}")
    
    return f"True: {true_str}\n" + "\n".join(pred_strs)


def main():
    """Generate sample visualizations for all models."""
    set_seed(42)
    
    # Load test set
    print("Loading test set...")
    _, _, test_df = load_clintox(
        cache_dir=str(project_root / "data"),
        split_type="scaffold",
        seed=42
    )
    
    # Load predictions for representative models only
    print(f"\nLoading predictions from representative models:")
    for model in REPRESENTATIVE_MODELS:
        print(f"  • {model}")
    
    all_predictions = {}
    
    for model_name in REPRESENTATIVE_MODELS:
        labels, probs = load_predictions(model_name, test_df)
        if labels is not None and probs is not None:
            all_predictions[model_name] = probs
            print(f"  ✓ {model_name}: {len(probs)} predictions")
        else:
            print(f"  ✗ Failed to load {model_name}")
    
    if len(all_predictions) == 0:
        print("Error: No predictions loaded. Exiting.")
        return
    
    # Create DataFrame with predictions from representative models only
    pred_df = test_df[['smiles', 'CT_TOX']].copy()
    for model_name, probs in all_predictions.items():
        pred_df[model_name] = probs
        pred_df[f'{model_name}_pred'] = (probs > 0.5).astype(int)
        pred_df[f'{model_name}_correct'] = (pred_df['CT_TOX'] == pred_df[f'{model_name}_pred'])
    
    # Store indices for easier indexing
    pred_df = pred_df.reset_index(drop=True)
    
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 1. Correct predictions (all representative models agree and are correct)
    print("\n1. Generating visualizations for correct predictions...")
    all_correct = pred_df.copy()
    for model_name in all_predictions.keys():
        all_correct = all_correct[all_correct[f'{model_name}_correct'] == True]
    
    print(f"  Found {len(all_correct)} samples where all representative models are correct")
    
    if len(all_correct) > 0:
        # Sample correct toxic
        correct_toxic = all_correct[all_correct['CT_TOX'] == 1].head(6)
        print(f"  Correct toxic samples: {len(correct_toxic)}")
        if len(correct_toxic) > 0:
            labels_list = []
            for _, row in correct_toxic.iterrows():
                pred_dict = {name: row[name] for name in all_predictions.keys()}
                labels_list.append(format_prediction_string(row['CT_TOX'], pred_dict))
            
            visualize_molecule_grid(
                correct_toxic['smiles'].tolist(),
                labels_list,
                n_cols=3,  # 3 columns for larger images
                title_prefix="Representative Models: All Correct - Toxic Predictions",
                save_path=results_dir / "sample_representative_models_correct_toxic.png"
            )
        
        # Sample correct non-toxic
        correct_nontoxic = all_correct[all_correct['CT_TOX'] == 0].head(6)
        print(f"  Correct non-toxic samples: {len(correct_nontoxic)}")
        if len(correct_nontoxic) > 0:
            labels_list = []
            for _, row in correct_nontoxic.iterrows():
                pred_dict = {name: row[name] for name in all_predictions.keys()}
                labels_list.append(format_prediction_string(row['CT_TOX'], pred_dict))
            
            visualize_molecule_grid(
                correct_nontoxic['smiles'].tolist(),
                labels_list,
                n_cols=3,  # 3 columns for larger images
                title_prefix="Representative Models: All Correct - Non-toxic Predictions",
                save_path=results_dir / "sample_representative_models_correct_nontoxic.png"
            )
    
    # 2. Misclassifications (where all representative models are wrong)
    print("\n2. Generating visualizations for misclassifications...")
    all_wrong = pred_df.copy()
    for model_name in all_predictions.keys():
        all_wrong = all_wrong[all_wrong[f'{model_name}_correct'] == False]
    
    print(f"  Found {len(all_wrong)} samples where all representative models are wrong")
    
    if len(all_wrong) > 0:
        # Sample wrong toxic (all predict non-toxic)
        wrong_toxic = all_wrong[all_wrong['CT_TOX'] == 1].head(6)
        print(f"  Wrong toxic (FN) samples: {len(wrong_toxic)}")
        if len(wrong_toxic) > 0:
            labels_list = []
            for _, row in wrong_toxic.iterrows():
                pred_dict = {name: row[name] for name in all_predictions.keys()}
                labels_list.append(format_prediction_string(row['CT_TOX'], pred_dict))
            
            visualize_molecule_grid(
                wrong_toxic['smiles'].tolist(),
                labels_list,
                n_cols=3,
                title_prefix="Representative Models: All Wrong - True Toxic (False Negatives)",
                save_path=results_dir / "sample_representative_models_wrong_toxic.png"
            )
        
        # Sample wrong non-toxic (all predict toxic)
        wrong_nontoxic = all_wrong[all_wrong['CT_TOX'] == 0].head(6)
        print(f"  Wrong non-toxic (FP) samples: {len(wrong_nontoxic)}")
        if len(wrong_nontoxic) > 0:
            labels_list = []
            for _, row in wrong_nontoxic.iterrows():
                pred_dict = {name: row[name] for name in all_predictions.keys()}
                labels_list.append(format_prediction_string(row['CT_TOX'], pred_dict))
            
            visualize_molecule_grid(
                wrong_nontoxic['smiles'].tolist(),
                labels_list,
                n_cols=3,
                title_prefix="Representative Models: All Wrong - True Non-toxic (False Positives)",
                save_path=results_dir / "sample_representative_models_wrong_nontoxic.png"
            )
    
    # 3. Model disagreements (expanded to find more samples)
    print("\n3. Generating visualizations for model disagreements...")
    
    # Collect different types of disagreements
    disagreement_samples = []
    
    # Type 1: SMILESGNN is correct but at least one other model is wrong (relaxed)
    if 'SMILESGNN (PyTorch Geometric)' in all_predictions:
        for idx, row in pred_df.iterrows():
            if row['SMILESGNN (PyTorch Geometric)_correct'] == True:
                # Count how many other models are wrong
                wrong_count = 0
                for model_name in all_predictions.keys():
                    if model_name != 'SMILESGNN (PyTorch Geometric)':
                        if row[f'{model_name}_correct'] == False:
                            wrong_count += 1
                if wrong_count >= 1:  # At least one other model is wrong
                    disagreement_samples.append(idx)
        
        print(f"  Found {len(disagreement_samples)} samples where SMILESGNN is correct but at least one other model is wrong")
    
    # Type 2: Find general disagreements where models have different predictions
    general_disagreements = []
    for idx, row in pred_df.iterrows():
        predictions = [row[name] > 0.5 for name in all_predictions.keys()]
        # If not all predictions are the same
        if len(set(predictions)) > 1:
            general_disagreements.append(idx)
    
    print(f"  Found {len(general_disagreements)} samples where models have different predictions")
    
    # Combine both types, prioritizing SMILESGNN correct cases
    combined_disagreements = list(set(disagreement_samples + general_disagreements))
    
    if len(combined_disagreements) > 0:
        # Sort to prioritize SMILESGNN correct cases first
        prioritized = []
        other_disagreements = []
        
        for idx in combined_disagreements:
            if idx in disagreement_samples:
                prioritized.append(idx)
            else:
                other_disagreements.append(idx)
        
        # Combine prioritized first, then others
        final_samples = prioritized[:6] + other_disagreements[:6]
        final_samples = final_samples[:12]  # Max 12 samples
        
        disagreement_df = pred_df.loc[final_samples].head(12)
        
        labels_list = []
        for _, row in disagreement_df.iterrows():
            pred_dict = {name: row[name] for name in all_predictions.keys()}
            labels_list.append(format_prediction_string(row['CT_TOX'], pred_dict))
        
        visualize_molecule_grid(
            disagreement_df['smiles'].tolist(),
            labels_list,
            n_cols=4,  # 4 columns to fit more samples
            title_prefix="Representative Models: Model Disagreements\nSMILESGNN vs Other Models",
            save_path=results_dir / "sample_representative_models_disagreement.png"
        )
    else:
        print("  No disagreement samples found")
    
    # 4. Generate diverse samples showing representative models on diverse molecules
    print("\n4. Generating diverse molecular structure visualizations...")
    
    # Select diverse molecules (different sizes, complexities)
    diverse_indices = []
    seen_mol_weights = set()
    
    for idx, row in pred_df.iterrows():
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
                # Sample diverse molecular weights
                weight_bin = int(mol_weight / 50) * 50
                if weight_bin not in seen_mol_weights or len(seen_mol_weights) < 12:
                    diverse_indices.append(idx)
                    seen_mol_weights.add(weight_bin)
                    if len(diverse_indices) >= 12:
                        break
        except:
            continue
    
    if len(diverse_indices) < 12:
        # Fallback: just sample diverse indices
        diverse_indices = list(range(min(12, len(pred_df))))
    
    diverse_df = pred_df.loc[diverse_indices[:12]]
    
    labels_list = []
    for _, row in diverse_df.iterrows():
        pred_dict = {name: row[name] for name in all_predictions.keys()}
        labels_list.append(format_prediction_string(row['CT_TOX'], pred_dict))
    
    visualize_molecule_grid(
        diverse_df['smiles'].tolist(),
        labels_list,
        n_cols=4,  # 4 columns for diverse samples
        title_prefix="Representative Models: Molecular Structure Understanding\nComparison on Diverse Compounds",
        save_path=results_dir / "sample_representative_models_diverse.png"
    )
    
    print("\n✓ Sample visualizations generated successfully!")


if __name__ == "__main__":
    main()

