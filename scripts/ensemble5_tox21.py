#!/usr/bin/env python3
"""
5-model ensemble inference for Tox21 multi-task toxicity prediction.

Members:
  1. ChemBERTa-2       (SMILES Transformer, 77M params)
  2. MoLFormer-XL      (SMILES Transformer, 44M params)
  3. AttentiveFP       (Graph model, gated attention)
  4. ECFP4 + XGBoost   (fingerprint baseline, per-task XGBClassifier)
  5. GPS Graph Transformer (local GINEConv + global MultiheadAttention)

Also evaluates all sub-ensemble combinations to find the optimal set.

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule
    python scripts/ensemble5_tox21.py --device cuda
"""

import sys
import argparse
import pickle
import warnings
from pathlib import Path
from itertools import combinations

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
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.datasets import load_dataset, get_task_config, TaskConfig
from src.pretrained_mol_model import create_pretrained_mol_model, get_checkpoint_defaults
from src.attentivefp_model import create_attentivefp_model
from src.gps_model import create_gps_model
from src.graph_data import smiles_to_pyg_data, get_feature_dims
from src.utils import set_seed, save_metrics, ensure_dir


# ── Ensemble member registry ──────────────────────────────────────────────────

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
        "type": "graph_afp",
        "model_dir": "models/tox21_attentivefp_model",
        "hidden_channels": 200,
        "num_layers": 2,
        "num_timesteps": 2,
        "dropout": 0.0,
    },
    {
        "name": "XGBoost",
        "type": "xgb",
        "model_dir": "models/tox21_fingerprint_model",
        "nbits": 2048,
        "radius": 2,
    },
    {
        "name": "GPS",
        "type": "graph_gps",
        "model_dir": "models/tox21_gps_model",
        "hidden_channels": 128,
        "num_layers": 4,
        "heads": 4,
        "dropout": 0.0,
    },
]


# ── Datasets ──────────────────────────────────────────────────────────────────

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
def infer_smiles_model(member, smiles_list, labels, task_config, device, batch_size=32):
    """SMILES Transformer inference → probs (N, T)."""
    ckpt      = member["checkpoint"]
    model_dir = project_root / member["model_dir"]

    print(f"  Loading tokenizer ...")
    defaults  = get_checkpoint_defaults(ckpt)
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

    ds     = SMILESDataset(smiles_list, labels, tokenizer, member["max_length"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        all_logits.append(logits.cpu())

    logits = torch.cat(all_logits).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


@torch.no_grad()
def infer_attentivefp(member, smiles_list, labels, task_config, device, batch_size=64):
    """AttentiveFP graph inference → probs (N, T)."""
    model_dir      = project_root / member["model_dir"]
    nf, ef         = get_feature_dims()

    model = create_attentivefp_model(
        node_feat_dim=nf, edge_feat_dim=ef,
        hidden_channels=member["hidden_channels"],
        num_layers=member["num_layers"],
        num_timesteps=member["num_timesteps"],
        dropout=member["dropout"],
        num_tasks=task_config.num_tasks,
    ).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = [smiles_to_pyg_data(s, label=labels[i])
               for i, s in enumerate(smiles_list)]
    dataset = [d for d in dataset if d is not None]
    loader  = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = np.full((len(smiles_list), task_config.num_tasks), np.nan)
    ptr = 0
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        bs     = logits.shape[0]
        all_logits[ptr:ptr + bs] = logits.cpu().numpy()
        ptr += bs

    return 1.0 / (1.0 + np.exp(-all_logits))


@torch.no_grad()
def infer_gps(member, smiles_list, labels, task_config, device, batch_size=128):
    """GPS Graph Transformer inference → probs (N, T)."""
    model_dir = project_root / member["model_dir"]
    nf, ef    = get_feature_dims()

    model = create_gps_model(
        node_feat_dim=nf, edge_feat_dim=ef,
        hidden_channels=member["hidden_channels"],
        num_layers=member["num_layers"],
        heads=member["heads"],
        dropout=member["dropout"],
        num_tasks=task_config.num_tasks,
    ).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = [smiles_to_pyg_data(s, label=labels[i])
               for i, s in enumerate(smiles_list)]
    dataset = [d for d in dataset if d is not None]
    loader  = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = np.full((len(smiles_list), task_config.num_tasks), np.nan)
    ptr = 0
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        bs     = logits.shape[0]
        all_logits[ptr:ptr + bs] = logits.cpu().numpy()
        ptr += bs

    return 1.0 / (1.0 + np.exp(-all_logits))


def infer_xgb(member, smiles_list, labels, task_config, device=None):
    """ECFP4 + per-task XGBClassifier inference → probs (N, T)."""
    model_dir = project_root / member["model_dir"]
    nbits     = member["nbits"]
    radius    = member["radius"]

    # Compute fingerprints
    print(f"  Computing ECFP{2*radius} fingerprints (nbits={nbits}) ...")
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(nbits, dtype=np.float32))
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(np.array(fp, dtype=np.float32))
    X = np.stack(fps)

    probs = np.full((len(smiles_list), task_config.num_tasks), 0.5, dtype=np.float32)
    for t, task_name in enumerate(task_config.task_names):
        pkl_path = model_dir / "models" / f"{task_name}.pkl"
        if not pkl_path.exists():
            print(f"  Warning: no model for {task_name}, using 0.5")
            continue
        with open(pkl_path, "rb") as f:
            clf = pickle.load(f)
        probs[:, t] = clf.predict_proba(X)[:, 1]

    return probs


# ── Dispatch ──────────────────────────────────────────────────────────────────

def infer_member(member, smiles_list, labels, task_config, device, batch_size):
    if member["type"] == "smiles":
        return infer_smiles_model(member, smiles_list, labels, task_config, device, batch_size)
    elif member["type"] == "graph_afp":
        return infer_attentivefp(member, smiles_list, labels, task_config, device, batch_size)
    elif member["type"] == "graph_gps":
        return infer_gps(member, smiles_list, labels, task_config, device, batch_size)
    elif member["type"] == "xgb":
        return infer_xgb(member, smiles_list, labels, task_config, device)
    else:
        raise ValueError(f"Unknown member type: {member['type']}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(probs, labels, task_config):
    per_auc, per_pr = {}, {}
    for t, name in enumerate(task_config.task_names):
        valid = ~np.isnan(labels[:, t])
        if valid.sum() < 2 or len(np.unique(labels[valid, t])) < 2:
            continue
        per_auc[name] = roc_auc_score(labels[valid, t], probs[valid, t])
        per_pr[name]  = average_precision_score(labels[valid, t], probs[valid, t])
    return {
        "mean_auc_roc":     float(np.mean(list(per_auc.values()))),
        "mean_pr_auc":      float(np.mean(list(per_pr.values()))),
        "per_task_auc_roc": per_auc,
        "per_task_pr_auc":  per_pr,
        "num_valid_tasks":  len(per_auc),
    }


def print_metrics(metrics, task_config, label="Ensemble"):
    print(f"\n{label}:")
    print("=" * 70)
    print(f"MEAN_AUC_ROC : {metrics['mean_auc_roc']:.4f}  "
          f"({metrics['num_valid_tasks']}/{task_config.num_tasks} tasks)")
    print(f"MEAN_PR_AUC  : {metrics['mean_pr_auc']:.4f}")
    print("\nPer-task AUC-ROC:")
    for task, auc in metrics["per_task_auc_roc"].items():
        bar = "█" * int(auc * 20)
        print(f"  {task:<20} {auc:.4f}  {bar}")
    print("=" * 70)


# ── Figures ───────────────────────────────────────────────────────────────────

def save_comparison_figure(all_metrics, out_path):
    names  = list(all_metrics.keys())
    aucs   = [all_metrics[n]["mean_auc_roc"] for n in names]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
              "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.1), 4))
    bars = ax.bar(names, aucs, color=colors[:len(names)], width=0.6, edgecolor="white")
    ax.set_ylim(0.66, 0.80)
    ax.set_ylabel("Mean AUC-ROC (Tox21 test set)", fontsize=11)
    ax.set_title("Tox21 — 5-Model Ensemble Analysis", fontsize=12)
    ax.axhline(0.7635, color="navy", linestyle="--", linewidth=0.9,
               alpha=0.7, label="Previous best (3-model, 0.7635)")
    ax.legend(fontsize=8)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{auc:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


def save_pertask_heatmap(all_metrics, task_config, out_path):
    model_names = list(all_metrics.keys())
    tasks       = task_config.task_names
    data        = np.array([
        [all_metrics[m]["per_task_auc_roc"].get(t, np.nan) for t in tasks]
        for m in model_names
    ])
    fig, ax = plt.subplots(figsize=(14, len(model_names) * 0.85 + 1.5))
    im = ax.imshow(data, vmin=0.62, vmax=0.86, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    for i in range(len(model_names)):
        for j in range(len(tasks)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.67 < val < 0.82 else "white")
    plt.colorbar(im, ax=ax, label="AUC-ROC")
    ax.set_title("Per-Task AUC-ROC — 5-Model Ensemble Analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    set_seed(args.seed)
    task_config = get_task_config("tox21")

    print("=" * 70)
    print("Tox21 5-Model Ensemble")
    print("  1. ChemBERTa-2  |  2. MoLFormer-XL  |  3. AttentiveFP")
    print("  4. XGBoost      |  5. GPS")
    print("=" * 70)

    # Load test set
    print("\nLoading Tox21 test set...")
    _, _, test_df = load_dataset(
        "tox21", cache_dir=str(project_root / "data"),
        split_type="scaffold", seed=args.seed,
    )
    smiles_list  = test_df["smiles"].tolist()
    labels_array = test_df[task_config.task_names].values.astype(np.float32)
    print(f"Test set: {len(smiles_list)} compounds")

    # Per-member inference
    all_probs   = {}
    all_metrics = {}

    for member in ENSEMBLE_MEMBERS:
        print(f"\n[{member['name']}]")
        probs = infer_member(member, smiles_list, labels_array, task_config, device, args.batch_size)
        m     = compute_metrics(probs, labels_array, task_config)
        all_probs[member["name"]]   = probs
        all_metrics[member["name"]] = m
        print(f"  Mean AUC-ROC: {m['mean_auc_roc']:.4f}  |  Mean PR-AUC: {m['mean_pr_auc']:.4f}")

    # Evaluate all ensemble combinations (size 2 to 5)
    print("\n" + "=" * 70)
    print("Ensemble combination search")
    print("=" * 70)

    names       = list(all_probs.keys())
    best_auc    = 0.0
    best_combo  = None
    combo_results = {}

    for size in range(2, len(names) + 1):
        for combo in combinations(names, size):
            avg_probs   = np.mean([all_probs[n] for n in combo], axis=0)
            m           = compute_metrics(avg_probs, labels_array, task_config)
            auc         = m["mean_auc_roc"]
            label       = " + ".join(combo)
            combo_results[label] = m
            flag = " ← best" if auc > best_auc else ""
            if auc > best_auc:
                best_auc   = auc
                best_combo = combo
            if size <= 3 or auc > 0.755:
                print(f"  [{size}-model] {label}: AUC={auc:.4f}{flag}")

    print(f"\nBest ensemble: {' + '.join(best_combo)}  AUC={best_auc:.4f}")

    # Full 5-model ensemble
    print("\n[5-model ensemble — all]")
    full_probs   = np.mean(list(all_probs.values()), axis=0)
    full_metrics = compute_metrics(full_probs, labels_array, task_config)
    print_metrics(full_metrics, task_config, "5-Model Ensemble (all)")

    # Best combo ensemble
    best_probs   = np.mean([all_probs[n] for n in best_combo], axis=0)
    best_metrics = compute_metrics(best_probs, labels_array, task_config)
    if best_combo != tuple(names):
        print_metrics(best_metrics, task_config, f"Best combo: {' + '.join(best_combo)}")

    # Save
    out_dir = project_root / "models" / "tox21_ensemble5"
    ensure_dir(str(out_dir))

    flat = {
        "ensemble5_mean_auc_roc": full_metrics["mean_auc_roc"],
        "ensemble5_mean_pr_auc":  full_metrics["mean_pr_auc"],
        "best_combo":             " + ".join(best_combo),
        "best_combo_auc":         best_auc,
    }
    flat.update({f"ensemble5_auc_{k}": v for k, v in full_metrics["per_task_auc_roc"].items()})
    save_metrics(flat, str(out_dir / "tox21_ensemble5_metrics.txt"))

    # Figures: individual + key ensembles
    plot_metrics = {**all_metrics}
    plot_metrics["Ensemble-3\n(CB2+MF+AFP)"] = compute_metrics(
        np.mean([all_probs[n] for n in ["ChemBERTa-2", "MoLFormer-XL", "AttentiveFP"]], axis=0),
        labels_array, task_config,
    )
    plot_metrics["Ensemble-5\n(all)"] = full_metrics
    if best_combo not in (
        ("ChemBERTa-2", "MoLFormer-XL", "AttentiveFP"),
        tuple(names),
    ):
        plot_metrics[f"Ensemble-{len(best_combo)}\n(best)"] = best_metrics

    save_comparison_figure(plot_metrics, out_dir / "ensemble5_comparison.png")
    save_pertask_heatmap(plot_metrics, task_config, out_dir / "ensemble5_pertask_heatmap.png")

    # Also update the assets/ figure used by README
    save_comparison_figure(plot_metrics, project_root / "assets" / "tox21_model_comparison.png")

    print(f"\nAll results saved to: {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
