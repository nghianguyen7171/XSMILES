#!/usr/bin/env python3
"""
Ensemble inference for Tox21 multi-task toxicity prediction.

Averages sigmoid probabilities from three heterogeneous models:
  1. ChemBERTa-2      (SMILES Transformer, 77M params)
  2. MoLFormer-XL     (SMILES Transformer, 44M params)
  3. AttentiveFP      (Graph model, gated attention)

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule
    python scripts/ensemble_tox21.py --device cuda
"""

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score

from src.datasets import load_dataset, get_task_config, TaskConfig
from src.pretrained_mol_model import create_pretrained_mol_model, get_checkpoint_defaults
from src.attentivefp_model import create_attentivefp_model
from src.graph_data import smiles_to_pyg_data, get_feature_dims
from src.utils import set_seed, save_metrics, ensure_dir


# ── Ensemble config ───────────────────────────────────────────────────────────

ENSEMBLE_MEMBERS = [
    {
        "name": "ChemBERTa-2",
        "type": "smiles",
        "checkpoint": "DeepChem/ChemBERTa-77M-MTR",
        "model_dir": "models/tox21_chemberta_model",
        "max_length": 128,
    },
    {
        "name": "MoLFormer-XL",
        "type": "smiles",
        "checkpoint": "ibm/MoLFormer-XL-both-10pct",
        "model_dir": "models/tox21_molformer_model",
        "max_length": 202,
    },
    {
        "name": "AttentiveFP",
        "type": "graph",
        "model_dir": "models/tox21_attentivefp_model",
        "hidden_channels": 200,
        "num_layers": 2,
        "num_timesteps": 2,
        "dropout": 0.0,   # no dropout at inference
    },
]


# ── SMILES dataset ────────────────────────────────────────────────────────────

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            smiles_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Per-model inference ───────────────────────────────────────────────────────

@torch.no_grad()
def infer_smiles_model(member: dict, smiles_list, labels, task_config, device, batch_size=32):
    """Run inference with a SMILES Transformer model. Returns probs (N, T)."""
    ckpt = member["checkpoint"]
    model_dir = project_root / member["model_dir"]
    max_length = member["max_length"]

    print(f"  Loading tokenizer from {model_dir / 'tokenizer'} ...")
    defaults = get_checkpoint_defaults(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir / "tokenizer"),
        trust_remote_code=defaults["trust_remote_code"],
    )

    print(f"  Building model ({ckpt}) ...")
    model = create_pretrained_mol_model(
        pretrained_model=ckpt,
        num_tasks=task_config.num_tasks,
        dropout=0.0,
    ).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = SMILESDataset(smiles_list, labels, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits = []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)
        all_logits.append(logits.cpu())

    logits = torch.cat(all_logits).numpy()          # (N, T)
    return 1.0 / (1.0 + np.exp(-logits))            # sigmoid → probs


@torch.no_grad()
def infer_graph_model(member: dict, smiles_list, labels, task_config, device, batch_size=64):
    """Run inference with AttentiveFP graph model. Returns probs (N, T)."""
    model_dir = project_root / member["model_dir"]
    node_feat_dim, edge_feat_dim = get_feature_dims()

    print(f"  Building AttentiveFP ...")
    model = create_attentivefp_model(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_channels=member["hidden_channels"],
        num_layers=member["num_layers"],
        num_timesteps=member["num_timesteps"],
        dropout=member["dropout"],
        num_tasks=task_config.num_tasks,
    ).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = []
    for i, smi in enumerate(smiles_list):
        data = smiles_to_pyg_data(smi, label=labels[i])
        if data is not None:
            data._smiles_idx = i      # track original index
            dataset.append(data)

    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = np.full((len(smiles_list), task_config.num_tasks), np.nan)
    ptr = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        bs = logits.shape[0]
        all_logits[ptr:ptr + bs] = logits.cpu().numpy()
        ptr += bs

    return 1.0 / (1.0 + np.exp(-all_logits))        # (N, T)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(probs: np.ndarray, labels: np.ndarray, task_config: TaskConfig) -> dict:
    per_task_auc, per_task_pr = {}, {}
    for t, name in enumerate(task_config.task_names):
        valid = ~np.isnan(labels[:, t])
        if valid.sum() < 2 or len(np.unique(labels[valid, t])) < 2:
            continue
        per_task_auc[name] = roc_auc_score(labels[valid, t], probs[valid, t])
        per_task_pr[name]  = average_precision_score(labels[valid, t], probs[valid, t])

    return {
        "mean_auc_roc":     float(np.mean(list(per_task_auc.values()))),
        "mean_pr_auc":      float(np.mean(list(per_task_pr.values()))),
        "per_task_auc_roc": per_task_auc,
        "per_task_pr_auc":  per_task_pr,
        "num_valid_tasks":  len(per_task_auc),
    }


def print_metrics(metrics: dict, task_config: TaskConfig, label: str = "Ensemble"):
    print(f"\n{label} — Test Set Results:")
    print("=" * 70)
    print(f"MEAN_AUC_ROC : {metrics['mean_auc_roc']:.4f}  "
          f"({metrics['num_valid_tasks']}/{task_config.num_tasks} tasks)")
    print(f"MEAN_PR_AUC  : {metrics['mean_pr_auc']:.4f}")
    print("\nPer-task AUC-ROC:")
    for task, auc in sorted(metrics["per_task_auc_roc"].items()):
        bar = "█" * int(auc * 20)
        print(f"  {task:<20} {auc:.4f}  {bar}")
    print("=" * 70)


# ── Figure ────────────────────────────────────────────────────────────────────

def save_comparison_figure(all_metrics: dict, task_config: TaskConfig, save_path: Path):
    """Bar chart: per-model and ensemble mean AUC-ROC."""
    names = list(all_metrics.keys())
    aucs  = [all_metrics[n]["mean_auc_roc"] for n in names]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, aucs, color=colors[:len(names)], width=0.55, edgecolor="white")
    ax.set_ylim(0.68, 0.78)
    ax.set_ylabel("Mean AUC-ROC (Tox21 test set)", fontsize=11)
    ax.set_title("Tox21 Model Comparison — Including Ensemble", fontsize=12)
    ax.axhline(0.74, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{auc:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {save_path}")


def save_pertask_heatmap(all_metrics: dict, task_config: TaskConfig, save_path: Path):
    """Heatmap of per-task AUC-ROC for all models."""
    model_names = list(all_metrics.keys())
    tasks = task_config.task_names
    data = np.array([
        [all_metrics[m]["per_task_auc_roc"].get(t, np.nan) for t in tasks]
        for m in model_names
    ])

    fig, ax = plt.subplots(figsize=(13, len(model_names) * 0.9 + 1.5))
    im = ax.imshow(data, vmin=0.62, vmax=0.86, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_names))); ax.set_yticklabels(model_names, fontsize=10)
    for i in range(len(model_names)):
        for j in range(len(tasks)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.5,
                        color="black" if 0.68 < val < 0.82 else "white")
    plt.colorbar(im, ax=ax, label="AUC-ROC")
    ax.set_title("Per-Task AUC-ROC — All Models + Ensemble", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ensemble inference for Tox21")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    set_seed(args.seed)
    task_config = get_task_config("tox21")

    print("=" * 70)
    print("Tox21 Ensemble: ChemBERTa-2 + MoLFormer-XL + AttentiveFP")
    print("=" * 70)

    # ── Load test set ─────────────────────────────────────────────────────
    print("\nLoading Tox21 test set...")
    _, _, test_df = load_dataset(
        "tox21", cache_dir=str(project_root / "data"),
        split_type="scaffold", seed=args.seed,
    )
    smiles_list  = test_df["smiles"].tolist()
    labels_array = test_df[task_config.task_names].values.astype(np.float32)
    print(f"Test set: {len(smiles_list)} compounds")

    # ── Per-member inference ──────────────────────────────────────────────
    member_probs   = []
    member_metrics = {}

    for member in ENSEMBLE_MEMBERS:
        print(f"\n[{member['name']}]")
        if member["type"] == "smiles":
            probs = infer_smiles_model(
                member, smiles_list, labels_array, task_config, device, args.batch_size
            )
        else:
            probs = infer_graph_model(
                member, smiles_list, labels_array, task_config, device, args.batch_size
            )
        member_probs.append(probs)
        m = compute_metrics(probs, labels_array, task_config)
        member_metrics[member["name"]] = m
        print(f"  Mean AUC-ROC: {m['mean_auc_roc']:.4f}  |  Mean PR-AUC: {m['mean_pr_auc']:.4f}")

    # ── Ensemble: simple average ──────────────────────────────────────────
    print("\n[Ensemble — simple average]")
    ensemble_probs   = np.mean(member_probs, axis=0)
    ensemble_metrics = compute_metrics(ensemble_probs, labels_array, task_config)
    print_metrics(ensemble_metrics, task_config, label="Ensemble (avg)")

    all_metrics = {**member_metrics, "Ensemble": ensemble_metrics}

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = project_root / "models" / "tox21_ensemble"
    ensure_dir(str(out_dir))

    flat = {
        "ensemble_mean_auc_roc": ensemble_metrics["mean_auc_roc"],
        "ensemble_mean_pr_auc":  ensemble_metrics["mean_pr_auc"],
    }
    flat.update({f"ensemble_auc_{k}": v for k, v in ensemble_metrics["per_task_auc_roc"].items()})
    save_metrics(flat, str(out_dir / "tox21_ensemble_metrics.txt"))

    save_comparison_figure(all_metrics, task_config, out_dir / "ensemble_comparison.png")
    save_pertask_heatmap(all_metrics, task_config, out_dir / "ensemble_pertask_heatmap.png")

    # Also save to assets/ for README
    save_comparison_figure(all_metrics, task_config, project_root / "assets" / "tox21_model_comparison.png")

    print(f"\nResults saved to: {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
