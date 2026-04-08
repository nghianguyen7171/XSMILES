"""
Integrated Gradients attribution for the SMILES Transformer encoder in SMILESGNN.

The SMILES Transformer encoder alone achieves AUC-ROC = 0.980, making it the
dominant pathway of the fused model.  GNNExplainer faithfulness check confirmed
(comprehensiveness ≈ 0) that graph-pathway masks do not meaningfully change the
prediction, so attribution must target the Transformer.

Design:
  - Baseline: zero embedding (standard for text/embedding IG)
  - Graph pathway: precomputed and frozen per molecule (gradients flow only through
    the SMILES encoder, isolating its contribution)
  - Aggregation: sum|attr| over embedding dim → scalar per token position
  - Token→atom mapping: character-level walk of _tokenize_smiles output with
    bracket tracking; 'l'/'r' (from Cl/Br) are correctly skipped
  - Output: atom_importance np.ndarray (N,) normalised [0,1], parallel to
    GNNExplainer's atom_importance for drop-in dual-heatmap compatibility
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


# ─────────────────────────────────────────────────────────────────────────────
# Token → atom mapping
# ─────────────────────────────────────────────────────────────────────────────

# Atoms that, when outside a bracket group, start a new heavy atom in SMILES.
# 'l' (from Cl) and 'r' (from Br) are intentionally excluded — the tokenizer
# splits Cl→['C','l'] and Br→['B','r'], so the leading char is the representative.
_ORGANIC_SUBSET = frozenset("BCNOPSFIbcnopsi")


def build_token_to_atom_map(raw_tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Map raw token indices (from _tokenize_smiles) to RDKit atom indices.

    The sequence before SOS/EOS prepending is walked character-by-character.
    Bracket atoms like [NH2+] are tracked as a unit: the '[' token position
    represents the atom; all tokens until ']' are consumed.

    Args:
        raw_tokens: Token list from tokenizer._tokenize_smiles(smiles),
                    i.e. BEFORE encode() prepends <SOS>.

    Returns:
        List of (raw_token_idx, rdkit_atom_idx) pairs.
        Encoded position = raw_token_idx + 1  (SOS occupies position 0).
    """
    mapping: List[Tuple[int, int]] = []
    atom_idx = 0
    in_bracket = False
    bracket_start = 0

    for i, tok in enumerate(raw_tokens):
        if tok == "[":
            in_bracket = True
            bracket_start = i
        elif tok == "]" and in_bracket:
            mapping.append((bracket_start, atom_idx))
            atom_idx += 1
            in_bracket = False
        elif not in_bracket and tok in _ORGANIC_SUBSET:
            mapping.append((i, atom_idx))
            atom_idx += 1
        # else: digit, bond char, paren, 'l', 'r', etc. → skip

    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Forward from embedding (bypasses token_embedding lookup for IG)
# ─────────────────────────────────────────────────────────────────────────────

def _run_from_embedding(
    emb: torch.Tensor,
    mask_t: torch.Tensor,
    model: nn.Module,
    graph_proj_frozen: torch.Tensor,
) -> torch.Tensor:
    """
    Full forward pass starting from the pre-embedded token tensor.

    Replicates SimpleSMILESEncoder.forward() from the point after
    token_embedding lookup, then runs cross-attention fusion and the
    predictor MLP.  graph_proj_frozen is a precomputed, no-grad tensor
    so gradients flow only through the SMILES encoder.

    Args:
        emb:               (1, seq_len, d_model) — differentiable embedding
        mask_t:            (1, seq_len) long — 1=real, 0=pad
        model:             SMILESGraphHybridPredictor in eval mode
        graph_proj_frozen: (1, 1, d_model) — precomputed graph projection,
                           no requires_grad

    Returns:
        logit: (1, 1) raw toxicity logit
    """
    encoder = model.smiles_encoder
    seq_len = emb.size(1)

    # 1. Add positional encoding
    x = emb + encoder.pos_encoder[:seq_len, :].unsqueeze(0)

    # 2. Transformer (eval mode → dropout disabled, attention weights stable)
    src_key_padding_mask = (mask_t == 0)  # True = ignore position
    x = encoder.transformer(x, src_key_padding_mask=src_key_padding_mask)

    # 3. Mean pooling over non-PAD positions
    mask_f = mask_t.unsqueeze(-1).float()    # (1, seq_len, 1)
    smiles_repr = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # (1, d)

    # 4. Cross-attention fusion: SMILES=query, graph=key/value
    smiles_q = smiles_repr.unsqueeze(1)      # (1, 1, d)
    attended, _ = model.fusion(smiles_q, graph_proj_frozen, graph_proj_frozen)
    attended = attended.squeeze(1)           # (1, d)

    # 5. Concatenate and predict
    fused = torch.cat([smiles_repr, attended], dim=1)  # (1, 2d)
    return model.predictor(fused)            # (1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main attribution function
# ─────────────────────────────────────────────────────────────────────────────

def attribute_smiles(
    smiles: str,
    model: nn.Module,
    tokenizer,
    pyg_data,
    device: str = "cpu",
    n_steps: int = 50,
    target_class: int = 1,
) -> Dict:
    """
    Compute Integrated Gradients attribution on the SMILES Transformer encoder.

    The graph pathway is frozen (precomputed graph_proj_frozen) so that IG
    gradients flow only through the Transformer token embeddings.

    Args:
        smiles:       SMILES string.
        model:        Loaded SMILESGraphHybridPredictor (eval mode).
        tokenizer:    Fitted SMILESTokenizer from training run.
        pyg_data:     torch_geometric.data.Data for this molecule.
        device:       'cpu' or 'cuda'.
        n_steps:      IG Riemann steps (50 is usually sufficient; use 100 for
                      publication-quality results and smaller convergence delta).
        target_class: 1 = attribute P(toxic); 0 = negate attributions.

    Returns:
        dict with keys:
            smiles              – input SMILES
            mol                 – RDKit Mol object
            atom_importance     – np.ndarray (N,), normalised [0,1]
            atom_importance_signed – np.ndarray (N,), signed (+ = pro-toxic)
            token_importance    – np.ndarray (128,), raw per-token |attr|
            token_importance_signed – np.ndarray (128,), signed per-token attr
            raw_tokens          – List[str] from _tokenize_smiles
            token_to_atom_map   – List[(raw_tok_idx, atom_idx)]
            prediction_prob     – float, P(toxic)
            predicted_class     – int
            true_label          – int or None
            convergence_delta   – float, IG completeness error (< 0.01 is good)
    """
    model.eval()
    model.to(device)
    pyg_data = pyg_data.to(device)

    # ── Tokenise ───────────────────────────────────────────────────────────
    ids, mask = tokenizer.encode(smiles)
    ids_t  = torch.tensor([ids],  dtype=torch.long,  device=device)  # (1,128)
    mask_t = torch.tensor([mask], dtype=torch.long,  device=device)  # (1,128)

    # ── Freeze graph pathway ───────────────────────────────────────────────
    batch_vec = torch.zeros(pyg_data.num_nodes, dtype=torch.long, device=device)
    data_ns   = SimpleNamespace(
        x=pyg_data.x, edge_index=pyg_data.edge_index,
        edge_attr=pyg_data.edge_attr, batch=batch_vec,
    )
    with torch.no_grad():
        graph_repr        = model.encode_graph(data_ns)               # (1, G)
        graph_proj_frozen = model.graph_proj(graph_repr).unsqueeze(1) # (1,1,d)

    # ── Input embedding (float, no grad — IG handles the interpolation) ────
    with torch.no_grad():
        emb = model.smiles_encoder.token_embedding(ids_t).float()     # (1,128,d)

    baseline = torch.zeros_like(emb)  # zero-embedding baseline (standard for text IG)

    # ── Captum IG ─────────────────────────────────────────────────────────
    # Captum's gausslegendre method batches all n_steps interpolated inputs
    # into a single forward call, so emb_in.size(0) == n_steps (not 1).
    # Expand mask_t and graph_proj_frozen to match that batch dimension.
    def _captum_forward(emb_in):
        bs = emb_in.size(0)
        mask_expanded = mask_t.expand(bs, -1)                    # (bs, 128)
        gpf_expanded  = graph_proj_frozen.expand(bs, -1, -1)     # (bs, 1, d)
        logit = _run_from_embedding(emb_in, mask_expanded, model, gpf_expanded)
        return torch.sigmoid(logit).view(-1)  # (bs,)

    ig = IntegratedGradients(_captum_forward)
    attributions, convergence_delta = ig.attribute(
        inputs=emb,
        baselines=baseline,
        n_steps=n_steps,
        method="gausslegendre",
        return_convergence_delta=True,
        # target=None: forward returns a scalar per batch item (P(toxic)),
        # no class indexing needed.
    )
    # attributions: (1, 128, d_model), convergence_delta: (1,)
    convergence_delta = float(convergence_delta.abs().item())

    if target_class == 0:
        attributions = -attributions  # flip sign for non-toxic attribution

    # ── Aggregate to token scalar ──────────────────────────────────────────
    token_attr_abs    = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()   # (128,)
    token_attr_signed = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()         # (128,)

    # ── Map tokens to atoms ────────────────────────────────────────────────
    raw_tokens      = tokenizer._tokenize_smiles(smiles)
    tok_to_atom_map = build_token_to_atom_map(raw_tokens)

    mol       = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    atom_imp        = np.zeros(num_atoms)
    atom_imp_signed = np.zeros(num_atoms)

    for raw_tok_idx, atom_idx in tok_to_atom_map:
        enc_pos = raw_tok_idx + 1       # +1 offset: SOS occupies encoded position 0
        if atom_idx < num_atoms and enc_pos < len(token_attr_abs):
            atom_imp[atom_idx]        = token_attr_abs[enc_pos]
            atom_imp_signed[atom_idx] = token_attr_signed[enc_pos]

    # Normalise to [0, 1]
    if atom_imp.max() > 0:
        atom_imp_norm = atom_imp / atom_imp.max()
    else:
        atom_imp_norm = atom_imp.copy()

    # ── Prediction ─────────────────────────────────────────────────────────
    with torch.no_grad():
        logit = _run_from_embedding(emb, mask_t, model, graph_proj_frozen)
    prob       = torch.sigmoid(logit).item()
    true_label = int(pyg_data.y.item()) if hasattr(pyg_data, "y") and pyg_data.y is not None else None

    return dict(
        smiles                 = smiles,
        mol                    = mol,
        atom_importance        = atom_imp_norm,
        atom_importance_signed = atom_imp_signed,
        token_importance       = token_attr_abs,
        token_importance_signed= token_attr_signed,
        raw_tokens             = raw_tokens,
        token_to_atom_map      = tok_to_atom_map,
        prediction_prob        = prob,
        predicted_class        = int(prob > 0.5),
        true_label             = true_label,
        convergence_delta      = convergence_delta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _importance_to_rgb(importance: float, cmap_name: str = "RdYlGn_r") -> Tuple[float, float, float]:
    """Map a [0,1] importance score to an RGB tuple using the shared colormap."""
    cmap = plt.cm.get_cmap(cmap_name)
    rgba = cmap(float(importance))
    return rgba[0], rgba[1], rgba[2]


def _draw_mol_heatmap(
    mol,
    atom_imp: np.ndarray,
    atom_threshold: float = 0.3,
    width: int = 500,
    height: int = 400,
):
    """Render a molecule with per-atom colour from importance scores."""
    from PIL import Image
    import io

    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    highlight_atoms = list(range(mol.GetNumAtoms()))
    for i, imp in enumerate(atom_imp):
        atom_colors[i] = _importance_to_rgb(imp)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().addAtomIndices = False
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=[],
        highlightAtomColors=atom_colors,
        highlightBondColors={},
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def visualize_transformer_attribution(
    result: Dict,
    figsize: Tuple[int, int] = (8, 5),
    atom_threshold: float = 0.3,
    save_path: Optional[str] = None,
) -> None:
    """
    Render a single-panel atom heatmap for Transformer IG attribution.

    Colour scheme is identical to GNNExplainer visualizations (RdYlGn_r):
    red = high importance (drives toxic prediction), green = low.

    Args:
        result:         Output dict from attribute_smiles().
        figsize:        Matplotlib figure size.
        atom_threshold: Atoms with importance >= threshold are listed as
                        'highlighted' in the drawer (cosmetic only — all atoms
                        are coloured by importance regardless).
        save_path:      If given, save PNG instead of plt.show().
    """
    mol        = result["mol"]
    atom_imp   = result["atom_importance"]
    prob       = result["prediction_prob"]
    pred_cls   = result["predicted_class"]
    true_label = result["true_label"]
    smiles     = result["smiles"]
    delta      = result["convergence_delta"]

    img = _draw_mol_heatmap(mol, atom_imp, atom_threshold)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Transformer IG atom importance\n(red = drives toxic prediction)", fontsize=11)

    true_str = f"True: {'Toxic' if true_label == 1 else 'Non-toxic'}" if true_label is not None else ""
    pred_str = f"Predicted: {'Toxic' if pred_cls == 1 else 'Non-toxic'} (P={prob:.3f})"
    fig.suptitle(
        f"{smiles[:70]}{'…' if len(smiles) > 70 else ''}\n"
        f"{pred_str}   {true_str}   conv_delta={delta:.5f}",
        fontsize=9, y=1.01,
    )

    sm   = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=mcolors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("IG importance", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_dual_heatmap(
    ig_result: Dict,
    gnn_result: Optional[Dict] = None,
    figsize: Tuple[int, int] = (14, 5),
    atom_threshold: float = 0.3,
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side dual-pathway heatmap.

    Left panel:  Transformer IG attribution (dominant encoder, AUC 0.980).
    Right panel: GNNExplainer graph attribution (weaker encoder, AUC 0.885).
                 If gnn_result is None, the right panel shows a caveats note.

    Both panels use the RdYlGn_r colormap with a shared colorbar.

    Args:
        ig_result:  Output dict from attribute_smiles().
        gnn_result: Output dict from explain_molecule(), or None.
        figsize:    Matplotlib figure size.
        atom_threshold: Passed to _draw_mol_heatmap (cosmetic).
        save_path:  If given, save PNG instead of plt.show().
    """
    mol      = ig_result["mol"]
    prob     = ig_result["prediction_prob"]
    pred_cls = ig_result["predicted_class"]
    true_lbl = ig_result["true_label"]
    smiles   = ig_result["smiles"]
    delta    = ig_result["convergence_delta"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Transformer IG
    img_ig = _draw_mol_heatmap(mol, ig_result["atom_importance"], atom_threshold)
    axes[0].imshow(img_ig)
    axes[0].axis("off")
    axes[0].set_title(
        "Transformer IG\n(dominant encoder — AUC 0.980)",
        fontsize=11,
    )

    # Right: GNNExplainer
    if gnn_result is not None:
        img_gnn = _draw_mol_heatmap(mol, gnn_result["atom_importance"], atom_threshold)
        axes[1].imshow(img_gnn)
        axes[1].set_title(
            "GNNExplainer (graph pathway)\n(faithfulness ≈ 0 — use with strong caveats)",
            fontsize=11,
        )
    else:
        axes[1].text(
            0.5, 0.5,
            "GNNExplainer not available\n(faithfulness check failed —\ngraph attribution is noise)",
            ha="center", va="center", transform=axes[1].transAxes,
            fontsize=11, wrap=True,
        )
        axes[1].set_title("GNNExplainer (graph pathway)", fontsize=11)
    axes[1].axis("off")

    true_str = f"True: {'Toxic' if true_lbl == 1 else 'Non-toxic'}" if true_lbl is not None else ""
    pred_str = f"Predicted: {'Toxic' if pred_cls == 1 else 'Non-toxic'} (P={prob:.3f})"
    fig.suptitle(
        f"{smiles[:80]}{'…' if len(smiles) > 80 else ''}\n"
        f"{pred_str}   {true_str}   IG conv_delta={delta:.5f}",
        fontsize=9, y=1.01,
    )

    sm   = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=mcolors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Importance", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Batch attribution
# ─────────────────────────────────────────────────────────────────────────────

def batch_attribute(
    smiles_list: List[str],
    labels: List[int],
    model: nn.Module,
    tokenizer,
    pyg_dataset: List,
    device: str = "cpu",
    n_steps: int = 50,
) -> List[Dict]:
    """
    Run attribute_smiles() on a list of molecules.

    Args:
        smiles_list:  List of SMILES strings.
        labels:       True labels (0/1) aligned with smiles_list.
        model:        Loaded SMILESGraphHybridPredictor.
        tokenizer:    Fitted SMILESTokenizer.
        pyg_dataset:  Corresponding list of PyG Data objects.
        device:       Device string.
        n_steps:      IG steps per molecule.

    Returns:
        List of result dicts (same schema as attribute_smiles).
        Molecules where attribution fails are skipped (warning printed).
    """
    results = []
    for i, (smi, lbl, pyg) in enumerate(zip(smiles_list, labels, pyg_dataset)):
        print(f"[{i+1}/{len(smiles_list)}] {smi[:55]}…", end="\r")
        try:
            res = attribute_smiles(smi, model, tokenizer, pyg,
                                   device=device, n_steps=n_steps,
                                   target_class=lbl)
            results.append(res)
        except Exception as e:
            print(f"\n  [WARN] skipped {smi[:40]}: {e}")
    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate element importance
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_token_importance(
    results: List[Dict],
    label_filter: Optional[int] = 1,
) -> Dict[str, float]:
    """
    Aggregate per-atom IG importance across molecules by element symbol.

    Parallel to aggregate_atom_importance() in gnn_explainer.py.

    Args:
        results:      List of result dicts from batch_attribute().
        label_filter: Include only molecules with this true_label
                      (1 = toxic, 0 = non-toxic, None = all).

    Returns:
        dict mapping element_symbol → mean_importance (float).
    """
    from collections import defaultdict

    element_scores: Dict[str, List[float]] = defaultdict(list)

    for res in results:
        if label_filter is not None and res.get("true_label") != label_filter:
            continue
        mol      = res["mol"]
        atom_imp = res["atom_importance"]
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            idx = atom.GetIdx()
            if idx < len(atom_imp):
                element_scores[sym].append(float(atom_imp[idx]))

    return {
        sym: float(np.mean(scores))
        for sym, scores in element_scores.items()
        if scores
    }
