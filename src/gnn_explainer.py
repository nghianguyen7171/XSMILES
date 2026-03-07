"""
GNNExplainer integration for SMILESGNN toxicity prediction.

Wraps the trained SMILESGraphHybridPredictor so that GNNExplainer can
perturb graph topology and node/edge masks while keeping the SMILES
(Transformer) embedding frozen. This isolates the graph pathway's
contribution to the final prediction, giving atom- and bond-level
importance scores that are faithful to the fused model's decision.

Key design choices:
  - SMILES embedding is pre-computed and frozen per molecule. GNNExplainer
    only optimises masks over the graph pathway, which is where structural
    (atom/bond) information lives.
  - Node indices in PyG match RDKit atom indices (both iterate mol.GetAtoms()).
  - Edge indices are bidirectional pairs: bond k → edges 2k (i→j) and
    2k+1 (j→i). Bond importance is the mean of both directions.
"""

import pickle
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.explain import Explainer, GNNExplainer


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SMILESGNNExplainerWrapper(nn.Module):
    """
    Adapter that exposes the GATv2 graph pathway of a trained
    SMILESGraphHybridPredictor to GNNExplainer.

    On construction the SMILES (Transformer) embedding is computed once
    and frozen as a buffer.  Subsequent calls with masked x / edge_index
    only re-run the graph encoder and the fusion/predictor head, so the
    explanation reflects structural changes to the graph.

    Args:
        model:  Loaded SMILESGraphHybridPredictor (eval mode, no gradients).
        smiles_token_ids:     (1, seq_len) long tensor.
        smiles_attention_mask:(1, seq_len) long tensor, or None.
    """

    def __init__(
        self,
        model: nn.Module,
        smiles_token_ids: torch.Tensor,
        smiles_attention_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = model

        with torch.no_grad():
            smiles_repr = model.smiles_encoder(
                smiles_token_ids, smiles_attention_mask
            )  # (1, d_model)
        self.register_buffer("smiles_repr", smiles_repr)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, F) node feature matrix (possibly masked by GNNExplainer)
            edge_index: (2, E) edge connectivity    (possibly masked)
            edge_attr:  (E, F_e) edge features
            batch:      (N,) batch vector — zeros for a single molecule

        Returns:
            logits: (1, 1) raw toxicity logit
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        data = SimpleNamespace(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        graph_repr = self.model.encode_graph(data)  # (1, graph_repr_dim)

        # Cross-attention fusion: SMILES as query, projected graph as key/value
        graph_proj = self.model.graph_proj(graph_repr).unsqueeze(1)       # (1,1,d)
        smiles_q   = self.smiles_repr.expand(graph_repr.size(0), -1).unsqueeze(1)  # (1,1,d)

        attended, _ = self.model.fusion(smiles_q, graph_proj, graph_proj)
        attended = attended.squeeze(1)  # (1, d)

        fused = torch.cat(
            [self.smiles_repr.expand(graph_repr.size(0), -1), attended], dim=1
        )  # (1, 2d)
        return self.model.predictor(fused)  # (1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_explainer(
    wrapper: SMILESGNNExplainerWrapper,
    epochs: int = 200,
    lr: float = 0.01,
) -> Explainer:
    """
    Construct a PyG Explainer around the wrapper.

    explanation_type='phenomenon' means we explain why the model predicts
    a *specific* class (toxic=1) for *this* molecule, which is more
    biochemically meaningful than a model-level explanation.

    node_mask_type / edge_mask_type = 'object' → one importance scalar
    per node/edge, suitable for atom/bond-level heatmaps.
    """
    return Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="phenomenon",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-molecule explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_molecule(
    smiles: str,
    model: nn.Module,
    tokenizer,
    pyg_data,
    device: str = "cpu",
    epochs: int = 200,
    target_class: int = 1,
) -> Dict:
    """
    Run GNNExplainer on a single molecule.

    Args:
        smiles:       SMILES string.
        model:        Loaded SMILESGraphHybridPredictor (eval mode).
        tokenizer:    Fitted SMILESTokenizer from the training run.
        pyg_data:     torch_geometric.data.Data for this molecule.
        device:       'cpu' or 'cuda'.
        epochs:       GNNExplainer optimisation steps (200 is usually enough).
        target_class: 1 = explain why model says "toxic"; 0 = "non-toxic".

    Returns:
        dict with keys:
            smiles          – input SMILES
            mol             – RDKit Mol object
            atom_importance – np.ndarray (N,), normalised [0,1]
            bond_importance – np.ndarray (num_bonds,), normalised [0,1]
            prediction_prob – float, P(toxic)
            predicted_class – int
            true_label      – int or None
    """
    model.eval()
    model.to(device)
    pyg_data = pyg_data.to(device)

    # Tokenise SMILES
    token_ids, attn_mask = tokenizer.encode(smiles)
    token_ids  = torch.tensor([token_ids],  dtype=torch.long,  device=device)
    attn_mask  = torch.tensor([attn_mask],  dtype=torch.long,  device=device)

    # Build wrapper with frozen SMILES embedding
    wrapper = SMILESGNNExplainerWrapper(model, token_ids, attn_mask).to(device)
    wrapper.eval()

    explainer = build_explainer(wrapper, epochs=epochs)

    batch = torch.zeros(pyg_data.num_nodes, dtype=torch.long, device=device)
    target = torch.tensor([target_class], dtype=torch.long, device=device)

    explanation = explainer(
        x=pyg_data.x,
        edge_index=pyg_data.edge_index,
        edge_attr=pyg_data.edge_attr,
        batch=batch,
        target=target,
    )

    # ── Node (atom) importance ──────────────────────────────────────────────
    node_mask = explanation.node_mask          # (N, 1)
    atom_imp  = node_mask.squeeze(-1).detach().cpu().numpy()
    if atom_imp.max() > 0:
        atom_imp = atom_imp / atom_imp.max()

    # ── Edge (bond) importance ──────────────────────────────────────────────
    # Edges are bidirectional pairs: bond k → edges 2k (i→j) and 2k+1 (j→i)
    edge_mask = explanation.edge_mask.detach().cpu().numpy()  # (E,)
    num_bonds = edge_mask.shape[0] // 2
    bond_imp  = np.array([
        (edge_mask[2 * k] + edge_mask[2 * k + 1]) / 2.0
        for k in range(num_bonds)
    ])
    if bond_imp.max() > 0:
        bond_imp = bond_imp / bond_imp.max()

    # ── Model prediction ────────────────────────────────────────────────────
    with torch.no_grad():
        logit = wrapper(
            pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr, batch
        )
    prob = torch.sigmoid(logit).item()

    mol = Chem.MolFromSmiles(smiles)
    true_label = int(pyg_data.y.item()) if hasattr(pyg_data, "y") and pyg_data.y is not None else None

    return dict(
        smiles=smiles,
        mol=mol,
        atom_importance=atom_imp,
        bond_importance=bond_imp,
        prediction_prob=prob,
        predicted_class=int(prob > 0.5),
        true_label=true_label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _importance_to_rgb(importance: float, cmap_name: str = "RdYlGn_r") -> Tuple[float, float, float]:
    """Map a [0,1] importance score to an RGB colour tuple."""
    cmap  = plt.cm.get_cmap(cmap_name)
    rgba  = cmap(float(importance))
    return rgba[0], rgba[1], rgba[2]


def visualize_explanation(
    result: Dict,
    figsize: Tuple[int, int] = (10, 5),
    atom_threshold: float = 0.3,
    bond_threshold: float = 0.3,
    save_path: Optional[str] = None,
) -> None:
    """
    Render a 2D molecule with GNNExplainer importance as atom/bond colours.

    Left panel  – atom importance heatmap (red = high, green = low).
    Right panel – bond importance overlay (red bonds = high importance).

    Args:
        result:          Output dict from explain_molecule().
        figsize:         Matplotlib figure size.
        atom_threshold:  Atoms with importance >= this value are highlighted.
        bond_threshold:  Bonds with importance >= this value are highlighted.
        save_path:       If given, save PNG to this path instead of plt.show().
    """
    mol          = result["mol"]
    atom_imp     = result["atom_importance"]
    bond_imp     = result["bond_importance"]
    prob         = result["prediction_prob"]
    pred_cls     = result["predicted_class"]
    true_label   = result["true_label"]
    smiles       = result["smiles"]

    from rdkit.Chem import Draw
    from PIL import Image
    import io

    def _draw(highlight_atoms, highlight_bonds, atom_colors, bond_colors, width=500, height=400):
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.drawOptions().addAtomIndices = False
        rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))

    # ── Panel A: atom importance ────────────────────────────────────────────
    atom_colors_A  = {}
    highlight_atoms_A = []
    for i, imp in enumerate(atom_imp):
        r, g, b = _importance_to_rgb(imp)
        atom_colors_A[i] = (r, g, b)
        if imp >= atom_threshold:
            highlight_atoms_A.append(i)

    img_A = _draw(
        highlight_atoms=list(range(mol.GetNumAtoms())),
        highlight_bonds=[],
        atom_colors=atom_colors_A,
        bond_colors={},
    )

    # ── Panel B: bond importance ────────────────────────────────────────────
    bond_colors_B  = {}
    highlight_bonds_B = []
    for k, imp in enumerate(bond_imp):
        r, g, b = _importance_to_rgb(imp)
        bond_colors_B[k] = (r, g, b)
        if imp >= bond_threshold:
            highlight_bonds_B.append(k)

    # Atom colours in panel B: colour by max adjacent bond importance
    atom_colors_B = {}
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        adj_bond_imps = [
            bond_imp[b.GetIdx()]
            for b in atom.GetBonds()
        ]
        max_imp = max(adj_bond_imps) if adj_bond_imps else 0.0
        r, g, b = _importance_to_rgb(max_imp)
        atom_colors_B[i] = (r, g, b)

    img_B = _draw(
        highlight_atoms=list(range(mol.GetNumAtoms())),
        highlight_bonds=list(range(mol.GetNumBonds())),
        atom_colors=atom_colors_B,
        bond_colors=bond_colors_B,
    )

    # ── Layout ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(img_A); axes[0].axis("off")
    axes[0].set_title("Atom importance\n(red = drives prediction)", fontsize=11)

    axes[1].imshow(img_B); axes[1].axis("off")
    axes[1].set_title("Bond importance\n(red = drives prediction)", fontsize=11)

    true_str = f"True: {'Toxic' if true_label == 1 else 'Non-toxic'}" if true_label is not None else ""
    pred_str = f"Predicted: {'Toxic' if pred_cls == 1 else 'Non-toxic'} (P={prob:.3f})"
    fig.suptitle(f"{smiles}\n{pred_str}   {true_str}", fontsize=10, y=1.01)

    # Colourbar
    sm  = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=mcolors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Importance", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Batch analysis
# ─────────────────────────────────────────────────────────────────────────────

def batch_explain(
    smiles_list: List[str],
    labels: List[int],
    model: nn.Module,
    tokenizer,
    pyg_dataset: List,
    device: str = "cpu",
    epochs: int = 200,
) -> List[Dict]:
    """
    Run GNNExplainer on a list of molecules and return all results.

    Args:
        smiles_list:  List of SMILES strings.
        labels:       True labels (0/1) aligned with smiles_list.
        model:        Loaded SMILESGraphHybridPredictor.
        tokenizer:    Fitted tokenizer.
        pyg_dataset:  Corresponding list of PyG Data objects.
        device:       Device string.
        epochs:       GNNExplainer optimisation epochs.

    Returns:
        List of result dicts (same schema as explain_molecule).
    """
    results = []
    for i, (smi, lbl, data) in enumerate(zip(smiles_list, labels, pyg_dataset)):
        print(f"[{i+1}/{len(smiles_list)}] {smi[:50]}…", end="\r")
        target = lbl  # explain from perspective of true label
        res = explain_molecule(smi, model, tokenizer, data, device=device,
                               epochs=epochs, target_class=target)
        results.append(res)
    print()
    return results


def aggregate_atom_importance(
    results: List[Dict],
    label_filter: Optional[int] = 1,
) -> Dict:
    """
    Aggregate atom-level importance across molecules to find which
    atom types are consistently flagged as important.

    Args:
        results:      List of result dicts from batch_explain.
        label_filter: Only include molecules with this true label
                      (1 = toxic, 0 = non-toxic, None = all).

    Returns:
        dict mapping element symbol → mean importance score.
    """
    from collections import defaultdict

    element_scores: Dict[str, List[float]] = defaultdict(list)

    for res in results:
        if label_filter is not None and res["true_label"] != label_filter:
            continue
        mol      = res["mol"]
        atom_imp = res["atom_importance"]
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            idx = atom.GetIdx()
            if idx < len(atom_imp):
                element_scores[sym].append(float(atom_imp[idx]))

    return {sym: float(np.mean(scores)) for sym, scores in element_scores.items()
            if scores}


def plot_element_importance(
    element_scores: Dict[str, float],
    title: str = "Mean atom importance by element (toxic compounds)",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of per-element mean importance scores."""
    sorted_items = sorted(element_scores.items(), key=lambda x: x[1], reverse=True)
    elements     = [k for k, _ in sorted_items]
    scores       = [v for _, v in sorted_items]

    colours = [_importance_to_rgb(s) for s in scores]

    fig, ax = plt.subplots(figsize=(max(6, len(elements)), 4))
    bars = ax.bar(elements, scores, color=colours, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean GNNExplainer importance", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="threshold=0.5")
    ax.legend(fontsize=9)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{score:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
