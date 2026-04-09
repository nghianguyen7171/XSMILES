#!/usr/bin/env python3
"""
Task-specific weighted ensemble for Tox21 multi-task toxicity prediction.

For each of the 12 tasks, optimizes ensemble weights (w_ChemBERTa2, w_MoLFormer,
w_AttentiveFP) on the validation set by maximising per-task AUC-ROC, then
applies those weights to the test set.

Weights are parameterised via softmax so the optimisation is unconstrained:
    w_i = softmax(logit_i)   → w_i > 0, sum(w_i) = 1

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule
    python scripts/weighted_ensemble_tox21.py --device cuda
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
from scipy.optimize import minimize
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
        "dropout": 0.0,
    },
]

N_MODELS = len(ENSEMBLE_MEMBERS)


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


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def infer_smiles_model(member, smiles_list, labels, task_config, device, batch_size=32):
    ckpt      = member["checkpoint"]
    model_dir = project_root / member["model_dir"]
    defaults  = get_checkpoint_defaults(ckpt)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir / "tokenizer"),
        trust_remote_code=defaults["trust_remote_code"],
    )
    model = create_pretrained_mol_model(
        pretrained_model=ckpt,
        num_tasks=task_config.num_tasks,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    ds     = SMILESDataset(smiles_list, labels, tokenizer, member["max_length"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logits = []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        logits.append(out.cpu())

    logits = torch.cat(logits).numpy()
    return 1.0 / (1.0 + np.exp(-logits))   # (N, T)


@torch.no_grad()
def infer_graph_model(member, smiles_list, labels, task_config, device, batch_size=64):
    model_dir          = project_root / member["model_dir"]
    node_dim, edge_dim = get_feature_dims()

    model = create_attentivefp_model(
        node_feat_dim=node_dim, edge_feat_dim=edge_dim,
        hidden_channels=member["hidden_channels"],
        num_layers=member["num_layers"],
        num_timesteps=member["num_timesteps"],
        dropout=member["dropout"],
        num_tasks=task_config.num_tasks,
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    dataset = []
    for i, smi in enumerate(smiles_list):
        d = smiles_to_pyg_data(smi, label=labels[i])
        if d is not None:
            dataset.append(d)

    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits = np.full((len(smiles_list), task_config.num_tasks), np.nan)
    ptr = 0
    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        bs     = logits.shape[0]
        all_logits[ptr:ptr + bs] = logits.cpu().numpy()
        ptr += bs

    return 1.0 / (1.0 + np.exp(-all_logits))   # (N, T)


def collect_probs(smiles_list, labels, task_config, device, batch_size):
    """Run all ensemble members and return list of (N, T) prob arrays."""
    all_probs = []
    for m in ENSEMBLE_MEMBERS:
        print(f"  [{m['name']}] running inference ...")
        if m["type"] == "smiles":
            p = infer_smiles_model(m, smiles_list, labels, task_config, device, batch_size)
        else:
            p = infer_graph_model(m, smiles_list, labels, task_config, device, batch_size)
        all_probs.append(p)
    return all_probs   # list of (N, T)


# ── Weight optimisation ───────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def optimise_weights_for_task(
    val_probs_list, val_labels, task_idx
) -> np.ndarray:
    """
    Find per-task weights on the validation set that maximise AUC-ROC.

    Returns weights array of shape (N_MODELS,) summing to 1.
    """
    valid = ~np.isnan(val_labels[:, task_idx])
    y     = val_labels[valid, task_idx]

    if valid.sum() < 10 or len(np.unique(y)) < 2:
        return np.ones(N_MODELS) / N_MODELS   # fall back to uniform

    preds = np.stack([p[valid, task_idx] for p in val_probs_list], axis=1)  # (n, M)

    def neg_auc(logits):
        w = softmax(np.array(logits))
        ens = (preds * w).sum(axis=1)
        try:
            return -roc_auc_score(y, ens)
        except Exception:
            return 0.0

    # Multiple random restarts to avoid local optima
    best_val, best_w = np.inf, np.ones(N_MODELS) / N_MODELS
    for _ in range(20):
        x0  = np.random.randn(N_MODELS)
        res = minimize(neg_auc, x0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5})
        if res.fun < best_val:
            best_val = res.fun
            best_w   = softmax(res.x)

    return best_w


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


def blend(probs_list, weights):
    """weights: (T, M) → returns (N, T) blended probabilities."""
    out = np.zeros_like(probs_list[0])
    for m, p in enumerate(probs_list):
        out += p * weights[:, m]
    return out


# ── Figures ───────────────────────────────────────────────────────────────────

def save_weight_heatmap(task_weights, task_names, member_names, save_path):
    """Heatmap: rows=tasks, cols=models, value=optimal weight."""
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(task_weights, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(member_names)))
    ax.set_xticklabels(member_names, fontsize=10)
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names, fontsize=9)
    for i in range(len(task_names)):
        for j in range(len(member_names)):
            ax.text(j, i, f"{task_weights[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if task_weights[i, j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Weight")
    ax.set_title("Task-Specific Ensemble Weights\n(optimised on validation AUC-ROC)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Weight heatmap saved: {save_path}")


def save_comparison_bar(results, save_path):
    """Grouped bar: simple ensemble vs weighted ensemble vs best single."""
    tasks  = list(results["Simple Ensemble"]["per_task_auc_roc"].keys())
    models = list(results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    x   = np.arange(len(tasks))
    w   = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, m) in enumerate(results.items()):
        vals = [m["per_task_auc_roc"].get(t, np.nan) for t in tasks]
        ax.bar(x + i * w - 0.4 + w / 2, vals, width=w,
               label=name, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.60, 0.90)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Per-Task AUC-ROC: Simple vs Weighted Ensemble", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison bar chart saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    set_seed(args.seed)
    task_config  = get_task_config("tox21")
    member_names = [m["name"] for m in ENSEMBLE_MEMBERS]

    print("=" * 70)
    print("Tox21 Task-Specific Weighted Ensemble")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading Tox21 (scaffold split, seed=42)...")
    _, val_df, test_df = load_dataset(
        "tox21", cache_dir=str(project_root / "data"),
        split_type="scaffold", seed=args.seed,
    )
    val_smiles  = val_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()
    val_labels  = val_df[task_config.task_names].values.astype(np.float32)
    test_labels = test_df[task_config.task_names].values.astype(np.float32)
    print(f"Val: {len(val_smiles)}  |  Test: {len(test_smiles)}")

    # ── Val inference ─────────────────────────────────────────────────────
    print("\nRunning val inference ...")
    val_probs = collect_probs(val_smiles, val_labels, task_config, device, args.batch_size)

    # ── Test inference ────────────────────────────────────────────────────
    print("\nRunning test inference ...")
    test_probs = collect_probs(test_smiles, test_labels, task_config, device, args.batch_size)

    # ── Optimise per-task weights on val ──────────────────────────────────
    print("\nOptimising per-task weights on validation set ...")
    task_weights = np.zeros((task_config.num_tasks, N_MODELS))   # (T, M)

    for t, task_name in enumerate(task_config.task_names):
        w = optimise_weights_for_task(val_probs, val_labels, t)
        task_weights[t] = w
        val_ens   = sum(val_probs[m][:, t]  * w[m] for m in range(N_MODELS))
        test_ens  = sum(test_probs[m][:, t] * w[m] for m in range(N_MODELS))
        valid_val = ~np.isnan(val_labels[:, t])
        val_auc   = roc_auc_score(val_labels[valid_val, t], val_ens[valid_val]) \
                    if valid_val.sum() >= 2 else float("nan")
        print(f"  {task_name:<20} weights: [{' '.join(f'{wi:.2f}' for wi in w)}]"
              f"  val_AUC={val_auc:.4f}")

    # ── Apply weights to test ─────────────────────────────────────────────
    weighted_test_probs = blend(test_probs, task_weights)
    simple_test_probs   = np.mean(test_probs, axis=0)

    # ── Metrics ───────────────────────────────────────────────────────────
    simple_metrics   = compute_metrics(simple_test_probs,   test_labels, task_config)
    weighted_metrics = compute_metrics(weighted_test_probs, test_labels, task_config)

    member_metrics = {}
    for i, m in enumerate(ENSEMBLE_MEMBERS):
        member_metrics[m["name"]] = compute_metrics(test_probs[i], test_labels, task_config)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Test Set)")
    print("=" * 70)
    header = f"{'Model':<35} {'Mean AUC':>9} {'PR-AUC':>9}"
    print(header)
    print("-" * 55)
    for name, m in member_metrics.items():
        print(f"  {name:<33} {m['mean_auc_roc']:>9.4f} {m['mean_pr_auc']:>9.4f}")
    print(f"  {'Simple Ensemble':<33} {simple_metrics['mean_auc_roc']:>9.4f}"
          f" {simple_metrics['mean_pr_auc']:>9.4f}")
    print(f"  {'Weighted Ensemble':<33} {weighted_metrics['mean_auc_roc']:>9.4f}"
          f" {weighted_metrics['mean_pr_auc']:>9.4f}")
    print("=" * 70)

    print("\nPer-task AUC-ROC comparison:")
    print(f"  {'Task':<22} {'Simple':>8} {'Weighted':>9} {'Δ':>7}")
    print("  " + "-" * 48)
    for task in task_config.task_names:
        s = simple_metrics["per_task_auc_roc"].get(task, float("nan"))
        w = weighted_metrics["per_task_auc_roc"].get(task, float("nan"))
        delta = w - s
        flag  = "▲" if delta > 0.002 else ("▼" if delta < -0.002 else " ")
        print(f"  {task:<22} {s:>8.4f} {w:>9.4f} {flag}{delta:>+6.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = project_root / "models" / "tox21_weighted_ensemble"
    ensure_dir(str(out_dir))

    np.save(out_dir / "task_weights.npy", task_weights)
    print(f"\nTask weights saved: {out_dir / 'task_weights.npy'}")

    flat = {
        "weighted_ensemble_mean_auc_roc": weighted_metrics["mean_auc_roc"],
        "weighted_ensemble_mean_pr_auc":  weighted_metrics["mean_pr_auc"],
        "simple_ensemble_mean_auc_roc":   simple_metrics["mean_auc_roc"],
        "simple_ensemble_mean_pr_auc":    simple_metrics["mean_pr_auc"],
    }
    flat.update({f"weighted_auc_{k}": v
                 for k, v in weighted_metrics["per_task_auc_roc"].items()})
    save_metrics(flat, str(out_dir / "tox21_weighted_ensemble_metrics.txt"))

    save_weight_heatmap(task_weights, task_config.task_names, member_names,
                        out_dir / "task_weights_heatmap.png")

    comparison_results = {
        **{m["name"]: member_metrics[m["name"]] for m in ENSEMBLE_MEMBERS},
        "Simple Ensemble":   simple_metrics,
        "Weighted Ensemble": weighted_metrics,
    }
    save_comparison_bar(comparison_results, out_dir / "weighted_vs_simple_comparison.png")

    # Also copy weight heatmap to assets/
    import shutil
    shutil.copy(out_dir / "task_weights_heatmap.png",
                project_root / "assets" / "ensemble_weights_heatmap.png")

    print(f"\nAll outputs saved to: {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
