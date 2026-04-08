"""
Gradient × Input and Joint KernelSHAP attribution for the full SMILESGNN model.

Unlike IG (SMILES pathway only, graph frozen) and GNNExplainer (graph pathway only,
SMILES frozen), both methods here use the COMPLETE model in a single computation —
gradients and perturbations flow through both encoders and the cross-attention fusion
layer simultaneously.

This eliminates the cross-pathway compensation problem:
  - GNNExplainer: masks graph atoms → SMILES encoder compensates → comp ≈ 0
  - IG:           zeros SMILES tokens → graph + pos_encoder compensates → ΔP ≈ 0
  - Grad×Input:   single backward through full model → both pathway gradients coupled
  - Joint SHAP:   mask atom in BOTH x[i] AND SMILES token → no single-pathway bypass

Context: Journal extension of the CITA 2026 paper.
Architecture details: src/graph_models_hybrid.py
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem

from src.transformer_attribution import build_token_to_atom_map, _draw_mol_heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Internal: Full forward pass from grad-enabled raw inputs
# ─────────────────────────────────────────────────────────────────────────────

def _full_forward(
    model: nn.Module,
    x_grad: torch.Tensor,
    emb: torch.Tensor,
    mask_t: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
    batch_vec: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Complete SMILESGNN forward from grad-enabled node features and SMILES embedding.

    Replicates model.forward() but:
      - Accepts x_grad and emb directly (instead of going through lookup layers)
        so both can carry requires_grad=True for joint backpropagation.
      - Extracts cross-attention weights from the fusion layer (Tier B signal).

    Args:
        model:      SMILESGraphHybridPredictor in eval mode.
        x_grad:     (N, F_node) float, node features, requires_grad may be True.
        emb:        (1, 128, d_model) float, token embeddings, requires_grad may be True.
        mask_t:     (1, 128) long, 1=real token, 0=PAD.
        edge_index: (2, E) long.
        edge_attr:  (E, F_edge) float or None.
        batch_vec:  (N,) long, all zeros for single molecule.

    Returns:
        prob:         scalar sigmoid probability P(toxic).
        attn_weights: (1, 1, 1) cross-attention weights from fusion or None.
    """
    # ── Graph pathway ──────────────────────────────────────────────────────
    data_ns = SimpleNamespace(
        x=x_grad, edge_index=edge_index, edge_attr=edge_attr, batch=batch_vec
    )
    graph_repr = model.encode_graph(data_ns)                      # (1, G)

    # ── SMILES pathway from embedding (replicate SimpleSMILESEncoder.forward) ──
    encoder = model.smiles_encoder
    seq_len = emb.size(1)
    x = emb + encoder.pos_encoder[:seq_len, :].unsqueeze(0)      # (1, 128, d)
    src_key_padding_mask = (mask_t == 0)                          # True = ignore
    x = encoder.transformer(x, src_key_padding_mask=src_key_padding_mask)
    mask_f = mask_t.unsqueeze(-1).float()                         # (1, 128, 1)
    smiles_repr = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # (1, d)

    # ── Cross-attention fusion: SMILES = query, graph = key/value ──────────
    graph_proj   = model.graph_proj(graph_repr).unsqueeze(1)      # (1, 1, d)
    smiles_q     = smiles_repr.unsqueeze(1)                       # (1, 1, d)
    attended, attn_weights = model.fusion(
        smiles_q, graph_proj, graph_proj, need_weights=True
    )
    attended = attended.squeeze(1)                                 # (1, d)

    # ── Predictor ──────────────────────────────────────────────────────────
    fused = torch.cat([smiles_repr, attended], dim=1)             # (1, 2d)
    logit = model.predictor(fused)                                # (1, 1)
    prob  = torch.sigmoid(logit).squeeze()                        # scalar
    return prob, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# A1. Gradient × Input through the full model
# ─────────────────────────────────────────────────────────────────────────────

def grad_input_attribution(
    smiles: str,
    model: nn.Module,
    tokenizer,
    pyg_data,
    device: str = "cpu",
) -> Dict:
    """
    Compute Gradient × Input attribution through the complete SMILESGNN model.

    A single backward pass couples both the graph encoder and the SMILES Transformer
    through the cross-attention fusion layer.  This eliminates the compensation
    artefact that invalidates single-pathway methods (GNNExplainer, IG).

    Args:
        smiles:    SMILES string.
        model:     Loaded SMILESGraphHybridPredictor in eval mode.
        tokenizer: Fitted SMILESTokenizer.
        pyg_data:  torch_geometric.data.Data for this molecule.
        device:    'cpu' or 'cuda'.

    Returns:
        dict with keys:
            smiles
            mol
            graph_atom_importance       – (N,) normalised [0,1], graph pathway
            graph_atom_raw              – (N,) raw Grad×Input scores
            smiles_atom_importance      – (N,) normalised [0,1], SMILES pathway
            smiles_atom_importance_signed – (N,) signed (+ = pro-toxic)
            smiles_token_importance     – (128,) per-token Grad×Input magnitude
            smiles_token_importance_signed – (128,) signed per-token
            token_to_atom_map           – List[(raw_tok_idx, atom_idx)]
            attn_weights                – (1,1,1) cross-attention numpy array
            prediction_prob             – float P(toxic)
            predicted_class             – int
            true_label                  – int or None
    """
    model.eval()
    model.to(device)
    pyg_data = pyg_data.to(device)

    # ── Tokenise ───────────────────────────────────────────────────────────
    ids, mask = tokenizer.encode(smiles)
    ids_t  = torch.tensor([ids],  dtype=torch.long,  device=device)
    mask_t = torch.tensor([mask], dtype=torch.long,  device=device)

    # ── Grad-enabled node features ─────────────────────────────────────────
    x_grad = pyg_data.x.float().clone().to(device)
    x_grad.requires_grad_(True)

    # ── Grad-enabled SMILES embedding ─────────────────────────────────────
    with torch.no_grad():
        emb_init = model.smiles_encoder.token_embedding(ids_t).float()  # (1, 128, d)
    emb = emb_init.clone().requires_grad_(True)

    batch_vec = torch.zeros(pyg_data.num_nodes, dtype=torch.long, device=device)

    # ── Forward + backward ────────────────────────────────────────────────
    prob, attn_weights = _full_forward(
        model, x_grad, emb, mask_t,
        pyg_data.edge_index,
        pyg_data.edge_attr if hasattr(pyg_data, "edge_attr") else None,
        batch_vec,
    )
    prob.backward()

    # ── Grad × Input — graph pathway ───────────────────────────────────────
    # x_grad.grad: (N, F_node) → sum |x * grad| over feature dim
    graph_raw  = (x_grad * x_grad.grad).abs().sum(dim=-1).detach().cpu().numpy()  # (N,)
    graph_norm = graph_raw / graph_raw.max() if graph_raw.max() > 0 else graph_raw.copy()

    # ── Grad × Input — SMILES pathway ─────────────────────────────────────
    # emb.grad: (1, 128, d_model) → sum |emb * grad| over embedding dim
    token_gi_abs    = (emb * emb.grad).abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    token_gi_signed = (emb * emb.grad).sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # ── Map tokens → atoms ─────────────────────────────────────────────────
    raw_tokens      = tokenizer._tokenize_smiles(smiles)
    tok_to_atom_map = build_token_to_atom_map(raw_tokens)
    mol             = Chem.MolFromSmiles(smiles)
    num_atoms       = mol.GetNumAtoms()

    smiles_atom_imp        = np.zeros(num_atoms)
    smiles_atom_imp_signed = np.zeros(num_atoms)
    for raw_tok_idx, atom_idx in tok_to_atom_map:
        enc_pos = raw_tok_idx + 1  # +1: SOS occupies position 0
        if atom_idx < num_atoms and enc_pos < len(token_gi_abs):
            smiles_atom_imp[atom_idx]        = token_gi_abs[enc_pos]
            smiles_atom_imp_signed[atom_idx] = token_gi_signed[enc_pos]

    smiles_norm = (
        smiles_atom_imp / smiles_atom_imp.max()
        if smiles_atom_imp.max() > 0
        else smiles_atom_imp.copy()
    )

    true_label = (
        int(pyg_data.y.item())
        if hasattr(pyg_data, "y") and pyg_data.y is not None
        else None
    )

    attn_np = attn_weights.detach().cpu().numpy() if attn_weights is not None else None

    return dict(
        smiles                         = smiles,
        mol                            = mol,
        graph_atom_importance          = graph_norm,
        graph_atom_raw                 = graph_raw,
        smiles_atom_importance         = smiles_norm,
        smiles_atom_importance_signed  = smiles_atom_imp_signed,
        smiles_token_importance        = token_gi_abs,
        smiles_token_importance_signed = token_gi_signed,
        token_to_atom_map              = tok_to_atom_map,
        attn_weights                   = attn_np,
        prediction_prob                = float(prob.detach().cpu()),
        predicted_class                = int(float(prob.detach().cpu()) > 0.5),
        true_label                     = true_label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1b. Joint comprehensiveness check (validates Grad×Input faithfulness)
# ─────────────────────────────────────────────────────────────────────────────

def faithfulness_check_joint(
    result: Dict,
    model: nn.Module,
    tokenizer,
    pyg_data,
    device: str = "cpu",
    top_k: int = 5,
    use_graph_importance: bool = True,
) -> Dict:
    """
    Comprehensiveness check with JOINT masking of both graph and SMILES pathways.

    Unlike the original faithfulness_check() (graph-only masking, allowing SMILES
    to compensate), this zeroes atom i in BOTH:
      - pyg_data.x[i]               (graph node features)
      - SMILES token embedding at the corresponding encoded position

    Args:
        result:               Output dict from grad_input_attribution().
        model:                Loaded SMILESGraphHybridPredictor.
        tokenizer:            Fitted SMILESTokenizer.
        pyg_data:             PyG Data for this molecule.
        device:               Device string.
        top_k:                Number of top-importance atoms to mask.
        use_graph_importance: If True, rank by graph_atom_importance; else by SMILES.

    Returns:
        dict: smiles, p_original, p_joint_masked, joint_comprehensiveness,
              top_k_atoms, top_k
    """
    smiles   = result["smiles"]
    atom_imp = result["graph_atom_importance"] if use_graph_importance \
               else result["smiles_atom_importance"]

    model.eval()
    model.to(device)
    pyg_data = pyg_data.to(device)

    ids, mask = tokenizer.encode(smiles)
    ids_t  = torch.tensor([ids],  dtype=torch.long,  device=device)
    mask_t = torch.tensor([mask], dtype=torch.long,  device=device)

    # Build atom → encoded SMILES position map
    raw_tokens      = tokenizer._tokenize_smiles(smiles)
    tok_to_atom_map = build_token_to_atom_map(raw_tokens)
    atom_to_enc_pos = {atom_idx: raw_tok_idx + 1 for raw_tok_idx, atom_idx in tok_to_atom_map}

    actual_k  = min(top_k, len(atom_imp))
    top_k_idx = np.argsort(atom_imp)[::-1][:actual_k].copy()  # .copy() avoids neg-stride error

    batch_vec = torch.zeros(pyg_data.num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        # ── Baseline: original molecule ─────────────────────────────────────
        emb_full = model.smiles_encoder.token_embedding(ids_t).float()
        x_full   = pyg_data.x.float().clone().to(device)

        prob_orig, _ = _full_forward(
            model, x_full, emb_full, mask_t,
            pyg_data.edge_index,
            pyg_data.edge_attr if hasattr(pyg_data, "edge_attr") else None,
            batch_vec,
        )
        p_original = float(prob_orig.cpu())

        # ── Joint masking: zero top-k in BOTH pathways ─────────────────────
        x_masked   = x_full.clone()
        emb_masked = emb_full.clone()

        for atom_i in top_k_idx:
            x_masked[int(atom_i)] = 0.0
            if int(atom_i) in atom_to_enc_pos:
                enc_pos = atom_to_enc_pos[int(atom_i)]
                emb_masked[0, enc_pos, :] = 0.0

        prob_masked, _ = _full_forward(
            model, x_masked, emb_masked, mask_t,
            pyg_data.edge_index,
            pyg_data.edge_attr if hasattr(pyg_data, "edge_attr") else None,
            batch_vec,
        )
        p_joint_masked = float(prob_masked.cpu())

    joint_comprehensiveness = p_original - p_joint_masked

    return dict(
        smiles                  = smiles,
        p_original              = p_original,
        p_joint_masked          = p_joint_masked,
        joint_comprehensiveness = joint_comprehensiveness,
        top_k_atoms             = top_k_idx.tolist(),
        top_k                   = actual_k,
        use_graph_importance    = use_graph_importance,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A2. Joint KernelSHAP (both pathways masked simultaneously)
# ─────────────────────────────────────────────────────────────────────────────

def joint_shap_attribution(
    smiles: str,
    model: nn.Module,
    tokenizer,
    pyg_data,
    device: str = "cpu",
    n_samples: int = 200,
) -> Dict:
    """
    KernelSHAP attribution with joint graph + SMILES masking.

    Feature space: N atoms (binary coalition).
    For absent atom i: zero x[i] in graph AND zero SMILES token embedding at
    the position corresponding to atom i.  This eliminates cross-pathway
    compensation by design.

    Background: all atoms absent (zero-input baseline), consistent with IG.

    Args:
        smiles:    SMILES string.
        model:     Loaded SMILESGraphHybridPredictor in eval mode.
        tokenizer: Fitted SMILESTokenizer.
        pyg_data:  torch_geometric.data.Data for this molecule.
        device:    Device string.
        n_samples: Number of KernelSHAP coalitions (default 200; use 500+ for publication).

    Returns:
        dict with:
            smiles, mol
            shap_values          – (N,) Shapley values per atom
            atom_importance      – (N,) normalised |shap_values| [0,1]
            atom_importance_signed – (N,) signed shap_values
            prediction_prob      – float P(toxic) at all-present coalition
            base_value           – float P(toxic) at all-absent coalition
            true_label           – int or None
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is required: pip install shap") from e

    model.eval()
    model.to(device)
    pyg_data = pyg_data.to(device)

    ids, mask = tokenizer.encode(smiles)
    ids_t  = torch.tensor([ids],  dtype=torch.long, device=device)
    mask_t = torch.tensor([mask], dtype=torch.long, device=device)

    # Build atom → encoded SMILES position map
    raw_tokens      = tokenizer._tokenize_smiles(smiles)
    tok_to_atom_map = build_token_to_atom_map(raw_tokens)
    atom_to_enc_pos = {atom_idx: raw_tok_idx + 1 for raw_tok_idx, atom_idx in tok_to_atom_map}

    n_atoms   = pyg_data.num_nodes
    batch_vec = torch.zeros(n_atoms, dtype=torch.long, device=device)

    # Pre-compute full embeddings once
    with torch.no_grad():
        emb_full = model.smiles_encoder.token_embedding(ids_t).float()  # (1, 128, d)
    x_full = pyg_data.x.float().clone().to(device)                      # (N, F)

    def model_fn(coalition_matrix: np.ndarray) -> np.ndarray:
        """
        coalition_matrix: (n_samples, n_atoms) binary numpy array.
        Returns:          (n_samples,) P(toxic) probabilities.
        """
        probs = []
        with torch.no_grad():
            for row in coalition_matrix:
                x_m   = x_full.clone()
                emb_m = emb_full.clone()
                for atom_i, present in enumerate(row):
                    if not present:
                        x_m[atom_i] = 0.0
                        if atom_i in atom_to_enc_pos:
                            emb_m[0, atom_to_enc_pos[atom_i], :] = 0.0
                p, _ = _full_forward(
                    model, x_m, emb_m, mask_t,
                    pyg_data.edge_index,
                    pyg_data.edge_attr if hasattr(pyg_data, "edge_attr") else None,
                    batch_vec,
                )
                probs.append(float(p.cpu()))
        return np.array(probs)

    # KernelSHAP: background = all atoms absent, foreground = all atoms present
    background = np.zeros((1, n_atoms))
    foreground = np.ones((1, n_atoms))

    explainer   = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(foreground, nsamples=n_samples, silent=True)
    # shap_values: (1, n_atoms) for single-output model
    sv = shap_values[0] if shap_values.ndim == 2 else shap_values  # (n_atoms,)

    sv_abs  = np.abs(sv)
    sv_norm = sv_abs / sv_abs.max() if sv_abs.max() > 0 else sv_abs.copy()

    # Base value and prediction at foreground
    base_prob = model_fn(background)[0]
    pred_prob = model_fn(foreground)[0]

    mol        = Chem.MolFromSmiles(smiles)
    true_label = (
        int(pyg_data.y.item())
        if hasattr(pyg_data, "y") and pyg_data.y is not None
        else None
    )

    return dict(
        smiles                 = smiles,
        mol                    = mol,
        shap_values            = sv,
        atom_importance        = sv_norm,
        atom_importance_signed = sv,
        prediction_prob        = pred_prob,
        base_value             = base_prob,
        true_label             = true_label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch wrapper
# ─────────────────────────────────────────────────────────────────────────────

def batch_grad_input(
    smiles_list: List[str],
    labels: List[int],
    model: nn.Module,
    tokenizer,
    pyg_dataset: List,
    device: str = "cpu",
) -> List[Dict]:
    """
    Run grad_input_attribution() on a list of molecules.

    Args:
        smiles_list:  List of SMILES strings.
        labels:       True labels (0/1) aligned with smiles_list.
        model:        Loaded SMILESGraphHybridPredictor.
        tokenizer:    Fitted SMILESTokenizer.
        pyg_dataset:  Corresponding list of PyG Data objects.
        device:       Device string.

    Returns:
        List of result dicts (same schema as grad_input_attribution).
        Molecules that fail are skipped (warning printed).
    """
    results = []
    for i, (smi, lbl, pyg) in enumerate(zip(smiles_list, labels, pyg_dataset)):
        print(f"[{i+1}/{len(smiles_list)}] {smi[:55]}…", end="\r")
        try:
            res = grad_input_attribution(smi, model, tokenizer, pyg, device=device)
            results.append(res)
        except Exception as e:
            print(f"\n  [WARN] skipped {smi[:40]}: {e}")
    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_grad_input(
    result: Dict,
    mode: str = "both",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualise Grad×Input atom importance.

    Args:
        result:    Output dict from grad_input_attribution().
        mode:      'both'  → side-by-side graph + SMILES panels (default)
                   'graph' → graph pathway only
                   'smiles'→ SMILES pathway only
                   'triple'→ graph | SMILES | average panels
        figsize:   Matplotlib figure size.
        save_path: If given, save PNG instead of plt.show().
    """
    mol      = result["mol"]
    prob     = result["prediction_prob"]
    pred_cls = result["predicted_class"]
    true_lbl = result["true_label"]
    smiles   = result["smiles"]

    true_str = f"  True: {'Toxic' if true_lbl == 1 else 'Non-toxic'}" if true_lbl is not None else ""
    pred_str = f"Predicted: {'Toxic' if pred_cls == 1 else 'Non-toxic'} (P={prob:.3f})"

    if mode == "triple":
        combined = (result["graph_atom_importance"] + result["smiles_atom_importance"]) / 2
        combined /= combined.max() if combined.max() > 0 else 1.0
        panels = [
            (result["graph_atom_importance"], "Graph Grad×Input\n(GATv2 pathway)"),
            (result["smiles_atom_importance"], "SMILES Grad×Input\n(Transformer pathway)"),
            (combined, "Combined (average)\nGraph + SMILES"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 3 // 2, figsize[1]))
    elif mode == "both":
        panels = [
            (result["graph_atom_importance"], "Graph Grad×Input\n(GATv2 pathway)"),
            (result["smiles_atom_importance"], "SMILES Grad×Input\n(Transformer pathway)"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    elif mode == "graph":
        panels = [(result["graph_atom_importance"], "Graph Grad×Input\n(GATv2 pathway)")]
        fig, axes = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        axes = [axes]
    else:  # smiles
        panels = [(result["smiles_atom_importance"], "SMILES Grad×Input\n(Transformer pathway)")]
        fig, axes = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        axes = [axes]

    for ax, (imp, title) in zip(axes, panels):
        img = _draw_mol_heatmap(mol, imp)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=11)

    fig.suptitle(
        f"{smiles[:80]}{'…' if len(smiles) > 80 else ''}\n"
        f"{pred_str}{true_str}",
        fontsize=9, y=1.01,
    )

    sm   = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=mcolors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Grad×Input importance", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
