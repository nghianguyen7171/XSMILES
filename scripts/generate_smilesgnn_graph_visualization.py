#!/usr/bin/env python3
"""
Generate molecular graph visualization for SMILESGNN model.

Shows how SMILESGNN processes molecules as graphs, similar to the GRIN visualization.
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
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.data import load_clintox
from src.utils import set_seed

# Copy load_predictions for SMILESGNN only
try:
    import torch
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from src.graph_data import smiles_list_to_pyg_dataset
    from src.graph_train import evaluate_model as evaluate_graph_model
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

def collate_fn(batch):
    return Batch.from_data_list(batch)

def load_smilesgnn_predictions(test_df, device='cpu'):
    """Load SMILESGNN predictions."""
    test_smiles = test_df['smiles'].tolist()
    test_labels = test_df['CT_TOX'].values
    
    if not PYG_AVAILABLE:
        return None, None
    
    try:
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
    except Exception as e:
        print(f"Error loading SMILESGNN: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def smiles_to_graph(smiles):
    """Convert SMILES to NetworkX graph for visualization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    G = nx.Graph()
    
    # Add nodes (atoms)
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), 
                  symbol=atom.GetSymbol(),
                  atomic_num=atom.GetAtomicNum(),
                  charge=atom.GetFormalCharge(),
                  is_aromatic=atom.GetIsAromatic())
    
    # Add edges (bonds)
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), 
                  bond.GetEndAtomIdx(),
                  bond_type=bond.GetBondType(),
                  is_aromatic=bond.GetIsAromatic())
    
    return G, mol


def get_node_color(atomic_num):
    """Get color for atom type."""
    color_map = {
        1: '#FFFFFF',   # H - White
        6: '#909090',   # C - Grey
        7: '#3050F8',   # N - Blue
        8: '#FF0D0D',   # O - Red
        9: '#90E050',   # F - Light Green
        15: '#FF8000',  # P - Orange
        16: '#FFFF30',  # S - Yellow
        17: '#1FF01F',  # Cl - Green
        35: '#A62929',  # Br - Dark Red
        53: '#940094',  # I - Purple
    }
    return color_map.get(atomic_num, '#FF1493')  # Default: Deep Pink


def visualize_molecular_graph(G, mol, ax, title="", node_size=500, font_size=8):
    """Visualize a molecular graph using NetworkX."""
    if G is None or mol is None:
        ax.text(0.5, 0.5, "Invalid molecule", ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Use spring layout for graph positioning
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Get node colors
    node_colors = [get_node_color(G.nodes[node]['atomic_num']) for node in G.nodes()]
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_size, alpha=0.9, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, alpha=0.6, edge_color='gray')
    
    # Add atom labels
    labels = {node: G.nodes[node]['symbol'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=font_size, 
                           font_weight='bold', font_color='black')
    
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
    ax.axis('off')


def visualize_graph_structures_smilesgnn(smiles_list, labels, predictions, n_samples=12):
    """
    Visualize molecular graph structures with SMILESGNN predictions.
    
    Args:
        smiles_list: List of SMILES strings
        labels: True labels (0 or 1)
        predictions: SMILESGNN predictions (probabilities)
        n_samples: Number of samples to visualize
    """
    # Sample molecules
    n_samples = min(n_samples, len(smiles_list))
    
    # Sample diverse molecules: mix of toxic/non-toxic and high/low predictions
    toxic_indices = [i for i, l in enumerate(labels) if l == 1]
    nontoxic_indices = [i for i, l in enumerate(labels) if l == 0]
    
    sample_indices = []
    
    # Sample toxic molecules with varying predictions
    if len(toxic_indices) > 0:
        toxic_with_pred = [(i, predictions[i]) for i in toxic_indices]
        toxic_with_pred.sort(key=lambda x: x[1])
        # Sample from low, medium, high predictions
        n_toxic = min(6, len(toxic_with_pred))
        step = max(1, len(toxic_with_pred) // n_toxic)
        sample_indices.extend([toxic_with_pred[i*step][0] for i in range(n_toxic)])
    
    # Sample non-toxic molecules
    if len(nontoxic_indices) > 0:
        nontoxic_with_pred = [(i, predictions[i]) for i in nontoxic_indices]
        nontoxic_with_pred.sort(key=lambda x: x[1])
        # Sample from low, medium predictions (non-toxic should have low prob)
        n_nontoxic = min(6, len(nontoxic_with_pred))
        step = max(1, len(nontoxic_with_pred) // n_nontoxic)
        sample_indices.extend([nontoxic_with_pred[i*step][0] for i in range(n_nontoxic)])
    
    # Limit to n_samples
    sample_indices = sample_indices[:n_samples]
    
    # Create figure
    n_cols = 4
    n_rows = (len(sample_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for plot_idx, data_idx in enumerate(sample_indices):
        smiles = smiles_list[data_idx]
        true_label = labels[data_idx]
        pred_prob = predictions[data_idx]
        
        # Create graph
        G, mol = smiles_to_graph(smiles)
        
        # Create title
        label_str = "Toxic" if true_label == 1 else "Non-toxic"
        pred_label = "Toxic" if pred_prob > 0.5 else "Non-toxic"
        correct = "✓" if (true_label == 1 and pred_prob > 0.5) or (true_label == 0 and pred_prob <= 0.5) else "✗"
        
        title = f"{label_str} | Pred: {pred_label} {correct}\nSMILESGNN: {pred_prob:.3f}"
        
        # Visualize
        visualize_molecular_graph(G, mol, axes[plot_idx], title=title)
    
    # Hide unused subplots
    for idx in range(len(sample_indices), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('SMILESGNN Molecular Graph Structure Visualization\n(Nodes=Atoms, Edges=Bonds - How GNN Sees Molecules)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    return fig


def main():
    """Generate SMILESGNN molecular graph visualizations."""
    set_seed(42)
    
    # Load test set
    print("Loading test set...")
    _, _, test_df = load_clintox(
        cache_dir=str(project_root / "data"),
        split_type="scaffold",
        seed=42
    )
    
    # Load SMILESGNN predictions
    print("\nLoading SMILESGNN predictions...")
    labels, predictions = load_smilesgnn_predictions(test_df)
    
    if labels is None or predictions is None:
        print("Error: Could not load SMILESGNN predictions. Make sure the model is trained.")
        return
    
    print(f"  Loaded {len(predictions)} predictions")
    
    # Create output directory
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Generate visualization
    print("\nGenerating molecular graph visualization...")
    fig = visualize_graph_structures_smilesgnn(
        test_df['smiles'].tolist(),
        labels,
        predictions,
        n_samples=12
    )
    
    save_path = results_dir / "smilesgnn_molecular_graphs.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()
    
    print("\n✓ SMILESGNN molecular graph visualization generated successfully!")
    print("\n" + "=" * 80)
    print("Graph Structure Explanation:")
    print("=" * 80)
    print("SMILESGNN processes molecules as graphs where:")
    print("  • Nodes represent atoms (colored by element type)")
    print("  • Edges represent chemical bonds")
    print("  • The graph encoder uses GATv2 layers to learn atom representations")
    print("  • The SMILES encoder processes the sequence representation")
    print("  • Both representations are fused via attention mechanism")
    print("  • This multimodal approach captures both structural and sequential patterns")
    print("=" * 80)


if __name__ == "__main__":
    main()

