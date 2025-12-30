#!/usr/bin/env python3
"""
Generate paired SMILES sequence and graph visualizations for SMILESGNN.

Shows 4 sample pairs demonstrating how SMILESGNN processes both sequence
(SMILES string) and graph (molecular structure) representations.
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
import matplotlib.patches as mpatches
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors

from src.data import load_clintox
from src.utils import set_seed

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


def visualize_smiles_sequence(smiles, ax, title="SMILES Sequence"):
    """Visualize SMILES sequence as text with molecular structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.text(0.5, 0.5, f"Invalid SMILES:\n{smiles}", 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.axis('off')
        return
    
    # Draw molecule using RDKit
    try:
        img = Draw.MolToImage(mol, size=(500, 500))
        ax.imshow(img)
    except:
        # Fallback to text if rendering fails
        ax.text(0.5, 0.8, "SMILES:", ha='center', va='top', transform=ax.transAxes,
               fontsize=14, fontweight='bold')
        ax.text(0.5, 0.6, smiles, ha='center', va='center', transform=ax.transAxes,
               fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', pad=10))
        ax.axis('off')
        return
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')


def visualize_molecular_graph(G, mol, ax, title="Graph Representation"):
    """Visualize a molecular graph using NetworkX."""
    if G is None or mol is None:
        ax.text(0.5, 0.5, "Invalid molecule", ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        return
    
    # Use spring layout for graph positioning
    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    
    # Get node colors
    node_colors = [get_node_color(G.nodes[node]['atomic_num']) for node in G.nodes()]
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=800, alpha=0.9, edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.5, alpha=0.7, edge_color='gray')
    
    # Add atom labels
    labels = {node: G.nodes[node]['symbol'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, 
                           font_weight='bold', font_color='black')
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')
    
    # Add legend for node colors
    legend_elements = [
        mpatches.Patch(facecolor='#909090', edgecolor='black', label='C'),
        mpatches.Patch(facecolor='#3050F8', edgecolor='black', label='N'),
        mpatches.Patch(facecolor='#FF0D0D', edgecolor='black', label='O'),
        mpatches.Patch(facecolor='#FFFF30', edgecolor='black', label='S'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
             framealpha=0.9, edgecolor='black')


def visualize_smiles_graph_pairs(smiles_list, labels, predictions, n_samples=4):
    """
    Visualize paired SMILES sequence and graph representations.
    
    Args:
        smiles_list: List of SMILES strings
        labels: True labels (0 or 1)
        predictions: SMILESGNN predictions (probabilities)
        n_samples: Number of sample pairs to visualize (default: 4)
    """
    # Select diverse samples
    n_samples = min(n_samples, len(smiles_list))
    
    # Sample diverse molecules: mix of toxic/non-toxic, different sizes
    toxic_indices = [i for i, l in enumerate(labels) if l == 1]
    nontoxic_indices = [i for i, l in enumerate(labels) if l == 0]
    
    sample_indices = []
    
    # Select 2 toxic and 2 non-toxic samples with varying molecular complexity
    selected_smiles = []
    seen_sizes = set()
    
    # Try to get diverse molecular sizes
    for idx in toxic_indices[:10] + nontoxic_indices[:30]:  # Check more candidates
        mol = Chem.MolFromSmiles(smiles_list[idx])
        if mol:
            num_atoms = mol.GetNumAtoms()
            size_bin = (num_atoms // 5) * 5  # Bin by 5 atoms
            if len(sample_indices) < n_samples and (size_bin not in seen_sizes or len(sample_indices) < 2):
                sample_indices.append(idx)
                selected_smiles.append(smiles_list[idx])
                seen_sizes.add(size_bin)
                if len(sample_indices) >= n_samples:
                    break
    
    # Fill remaining slots if needed
    while len(sample_indices) < n_samples:
        for idx in range(len(smiles_list)):
            if idx not in sample_indices:
                sample_indices.append(idx)
                selected_smiles.append(smiles_list[idx])
                if len(sample_indices) >= n_samples:
                    break
    
    # Create figure: 4 rows (samples) x 2 columns (SMILES + Graph)
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(n_samples, 2, hspace=0.3, wspace=0.2,
                         left=0.05, right=0.95, top=0.96, bottom=0.04)
    
    for row, data_idx in enumerate(sample_indices):
        smiles = smiles_list[data_idx]
        true_label = labels[data_idx]
        pred_prob = predictions[data_idx]
        
        # Left: SMILES sequence
        ax_smiles = fig.add_subplot(gs[row, 0])
        label_str = "Toxic" if true_label == 1 else "Non-toxic"
        pred_label = "Toxic" if pred_prob > 0.5 else "Non-toxic"
        correct = "✓" if (true_label == 1 and pred_prob > 0.5) or (true_label == 0 and pred_prob <= 0.5) else "✗"
        
        title_smiles = f"Sample {row+1}: {label_str} | Pred: {pred_label} {correct} | Prob: {pred_prob:.3f}"
        visualize_smiles_sequence(smiles, ax_smiles, title=title_smiles)
        
        # Right: Graph representation
        ax_graph = fig.add_subplot(gs[row, 1])
        G, mol = smiles_to_graph(smiles)
        title_graph = f"Graph Representation\n(Nodes=Atoms, Edges=Bonds)"
        visualize_molecular_graph(G, mol, ax_graph, title=title_graph)
    
    fig.suptitle('SMILESGNN: Understanding Molecules Through Sequence and Graph\n'
                 'Left: SMILES Sequence (How Transformer Encoder Sees It) | '
                 'Right: Graph Structure (How GNN Encoder Sees It)',
                fontsize=14, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    return fig


def main():
    """Generate paired SMILES sequence and graph visualizations."""
    set_seed(42)
    
    # Load test set
    print("Loading test set...")
    _, _, test_df = load_clintox(
        cache_dir=str(project_root / "data"),
        split_type="scaffold",
        seed=42
    )
    
    print(f"Loaded {len(test_df)} test samples")
    print(f"Class distribution: Toxic={test_df['CT_TOX'].sum()}, Non-toxic={len(test_df) - test_df['CT_TOX'].sum()}")
    
    # Load SMILESGNN predictions
    print("\nLoading SMILESGNN predictions...")
    
    if not PYG_AVAILABLE:
        print("Error: PyTorch Geometric not available")
        return
    
    try:
        import torch
        from torch.utils.data import DataLoader
        from torch_geometric.data import Batch
        from src.graph_data import smiles_list_to_pyg_dataset
        from src.graph_train import evaluate_model as evaluate_graph_model
        
        model_path = project_root / "models" / "smilesgnn_model" / "best_model.pt"
        from src.graph_models_hybrid import create_hybrid_model
        from src.graph_data import get_feature_dims
        import pickle
        
        num_node_features, num_edge_features = get_feature_dims()
        tokenizer_path = project_root / "models" / "smilesgnn_model" / "tokenizer.pkl"
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        checkpoint = torch.load(model_path, map_location='cpu')
        
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
        test_dataset = smiles_list_to_pyg_dataset(test_df['smiles'].tolist(), labels=test_df['CT_TOX'].tolist())
        
        class HybridDataset:
            def __init__(self, pyg_dataset, smiles_list, tokenizer):
                self.pyg_dataset = pyg_dataset
                self.smiles_list = smiles_list
                self.tokenizer = tokenizer
            def __len__(self): return len(self.pyg_dataset)
            def __getitem__(self, idx):
                data = self.pyg_dataset[idx]
                smiles = self.smiles_list[idx]
                token_ids, attention_mask = tokenizer.encode(smiles)
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
        
        hybrid_dataset = HybridDataset(test_dataset, test_df['smiles'].tolist(), tokenizer)
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
        metrics = evaluate_graph_model(wrapped_model, test_loader, device='cpu', return_predictions=True)
        labels = metrics['labels']
        predictions = metrics['predictions']
        
        print(f"  ✓ Loaded {len(predictions)} predictions")
        
    except Exception as e:
        print(f"Error loading SMILESGNN: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create output directory
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Generate paired visualizations (4 samples)
    print("\nGenerating paired SMILES sequence and graph visualizations...")
    fig = visualize_smiles_graph_pairs(
        test_df['smiles'].tolist(),
        labels,
        predictions,
        n_samples=4
    )
    
    save_path = results_dir / "smilesgnn_smiles_graph_pairs.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()
    
    print("\n✓ Paired SMILES sequence and graph visualizations generated successfully!")
    print("\n" + "=" * 80)
    print("Visualization Explanation:")
    print("=" * 80)
    print("Each row shows a molecule in two representations:")
    print("  • Left: SMILES Sequence - The sequence representation processed by the Transformer encoder")
    print("  • Right: Graph Structure - The graph representation processed by the GNN encoder")
    print("  • SMILESGNN combines both representations via attention fusion for superior performance")
    print("=" * 80)


if __name__ == "__main__":
    # Define PYG_AVAILABLE at module level
    try:
        import torch
        from torch.utils.data import DataLoader
        from torch_geometric.data import Batch
        from src.graph_data import smiles_list_to_pyg_dataset
        from src.graph_train import evaluate_model as evaluate_graph_model
        PYG_AVAILABLE = True
    except ImportError:
        PYG_AVAILABLE = False
    
    main()

