"""
Inference utilities for SMILESGNN toxicity predictor.

Provides:
  - load_model()     — load trained checkpoint + tokenizer
  - predict_batch()  — fast Stage A batch scoring

HybridModelWrapper routes SMILES token IDs from the PyG batch
object to model.forward(). This must match the wrapper used during
training, otherwise the SMILES encoder is bypassed (zero vectors).
"""

import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.graph_data import get_feature_dims, smiles_to_pyg_data
from src.graph_models_hybrid import create_hybrid_model


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper (must match training wrapper in scripts/train_hybrid.py)
# ─────────────────────────────────────────────────────────────────────────────

class HybridModelWrapper(nn.Module):
    """Routes SMILES token IDs from the batch object into model.forward()."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.model = m

    def forward(self, batch):
        return self.model(
            batch,
            smiles_token_ids=batch.smiles_token_ids
                if hasattr(batch, "smiles_token_ids") else None,
            smiles_attention_mask=batch.smiles_attention_masks
                if hasattr(batch, "smiles_attention_masks") else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

class _HybridDataset:
    """Attaches SMILES token IDs to each PyG Data object for batched inference."""

    def __init__(self, pyg_dataset: list, smiles_list: List[str], tokenizer):
        self.pyg_dataset = pyg_dataset
        self.smiles_list = smiles_list
        self.tok = tokenizer

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        data = self.pyg_dataset[idx]
        ids, mask = self.tok.encode(self.smiles_list[idx])
        data.smiles_token_ids      = torch.tensor(ids,  dtype=torch.long)
        data.smiles_attention_mask = torch.tensor(mask, dtype=torch.long)
        return data


def _collate(batch):
    b = Batch.from_data_list(batch)
    if hasattr(batch[0], "smiles_token_ids"):
        b.smiles_token_ids       = torch.stack([x.smiles_token_ids      for x in batch])
        b.smiles_attention_masks = torch.stack([x.smiles_attention_mask for x in batch])
    return b


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    model_dir:   Path,
    config_path: Path,
    device:      str = "cpu",
):
    """
    Load a trained SMILESGNN checkpoint.

    IMPORTANT: tokenizer is loaded first so we can read the actual vocab size
    (the config stores 100 as an upper bound; the checkpoint was saved with
    the real vocab size of ~69 tokens — using 100 causes a shape mismatch).

    Returns
    -------
    (model, tokenizer, wrapped_model)  — all in eval mode on `device`.
    """
    model_dir   = Path(model_dir)
    config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config["model"]

    # Load tokenizer first — its vocab size governs the embedding layer shape
    tok_path = model_dir / "tokenizer.pkl"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tok_path}\n"
            "Run 'python scripts/train_hybrid.py' first."
        )
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    actual_vocab_size = len(tokenizer.token_to_id)

    num_node_features, num_edge_features = get_feature_dims()
    model = create_hybrid_model(
        num_node_features = num_node_features,
        num_edge_features = num_edge_features,
        hidden_dim        = int(mc["hidden_dim"]),
        num_graph_layers  = int(mc["num_graph_layers"]),
        graph_model       = mc["graph_model"],
        num_heads         = int(mc["num_heads"]),
        dropout           = float(mc["dropout"]),
        use_residual      = bool(mc.get("use_residual", True)),
        use_jk            = bool(mc.get("use_jk", True)),
        jk_mode           = mc.get("jk_mode", "cat"),
        graph_pooling     = mc.get("graph_pooling", "meanmax"),
        smiles_vocab_size = actual_vocab_size,   # real vocab, NOT config default
        smiles_d_model    = int(mc["smiles_d_model"]),
        smiles_num_layers = int(mc["smiles_num_layers"]),
        fusion_method     = mc.get("fusion_method", "attention"),
    )

    ckpt = model_dir / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Run 'python scripts/train_hybrid.py' first."
        )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    wrapped = HybridModelWrapper(model)
    wrapped.eval()

    return model, tokenizer, wrapped


# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(
    smiles_list:   List[str],
    tokenizer,
    wrapped_model: nn.Module,
    device:        str,
    names:         Optional[List[str]] = None,
    true_labels:   Optional[List[int]] = None,
    threshold:     float = 0.5,
    batch_size:    int   = 32,
) -> pd.DataFrame:
    """
    Stage A — fast batch toxicity prediction.

    Compounds that RDKit cannot featurise (e.g. organometallics, exotic
    coordination chemistry) are reported as 'Parse error' and excluded
    from model scoring.

    Parameters
    ----------
    smiles_list   : list of SMILES strings
    tokenizer     : fitted SMILESTokenizer from training run
    wrapped_model : HybridModelWrapper (eval mode)
    device        : 'cpu' or 'cuda'
    names         : optional compound names (auto-generated if None)
    true_labels   : optional ground-truth labels (0/1)
    threshold     : decision boundary (default 0.5)
    batch_size    : GPU mini-batch size

    Returns
    -------
    pd.DataFrame sorted by P(toxic) descending.
    Parse errors appear at the bottom with P(toxic) = None.
    """
    valid, invalid = [], []

    for i, smi in enumerate(smiles_list):
        name = names[i] if names else f"Mol-{i:03d}"
        lbl  = true_labels[i] if true_labels is not None else None
        try:
            d = smiles_to_pyg_data(smi, label=lbl if lbl is not None else 0)
        except Exception:
            d = None
        if d is None:
            invalid.append({
                "Name":       name,
                "SMILES":     smi,
                "P(toxic)":   None,
                "Predicted":  "Parse error",
                "True label": ("Toxic" if lbl == 1 else "Non-toxic") if lbl is not None else "—",
                "Correct":    "—",
            })
        else:
            valid.append((i, name, smi, lbl, d))

    if not valid:
        return pd.DataFrame(invalid)

    _, vnames, vsmiles, vlabels, pyg_list = zip(*valid)
    dataset = _HybridDataset(list(pyg_list), list(vsmiles), tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    all_probs: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = wrapped_model(batch).squeeze(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])

    rows = []
    for name, smi, lbl, prob in zip(vnames, vsmiles, vlabels, all_probs):
        pred    = 1 if prob >= threshold else 0
        correct = (pred == lbl) if lbl is not None else None
        rows.append({
            "Name":       name,
            "SMILES":     smi,
            "P(toxic)":   round(prob, 4),
            "Predicted":  "Toxic" if pred == 1 else "Non-toxic",
            "True label": ("Toxic" if lbl == 1 else "Non-toxic") if lbl is not None else "—",
            "Correct":    ("✓" if correct else "✗") if correct is not None else "—",
        })

    rows.extend(invalid)
    df = pd.DataFrame(rows).sort_values("P(toxic)", ascending=False, na_position="last")
    return df.reset_index(drop=True)
