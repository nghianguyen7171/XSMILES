#!/usr/bin/env python3
"""
Uni-Mol + MLP for Tox21 multi-task toxicity prediction.

Uni-Mol (Zhou et al., NeurIPS 2023) is a universal 3D molecular pre-trained
model. It generates 512-dim molecular representations from 3D conformations
produced by RDKit ETKDG. These frozen representations are fed into a
learnable multi-task MLP head trained with MaskedMultiTaskLoss.

Pipeline:
    SMILES → RDKit ETKDG 3D conformer → Uni-Mol encoder → 512-dim CLS repr
    → MLP head (512 → 256 → 12 tasks)

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule
    python scripts/train_unimol_tox21.py \
        --config config/tox21_unimol_config.yaml --device cuda
"""

import sys
import argparse
import warnings
import shutil
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.getLogger("unimol_tools").setLevel(logging.WARNING)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score

from src.datasets import load_dataset, get_task_config, TaskConfig
from src.graph_train import MaskedMultiTaskLoss, create_multitask_sampler
from src.utils import set_seed, save_metrics, ensure_dir


# ── Uni-Mol representation extractor ─────────────────────────────────────────

def extract_unimol_reprs(smiles_list, batch_size=256, cache_path=None):
    """
    Extract 512-dim Uni-Mol CLS representations for a list of SMILES.

    Uses RDKit ETKDG to generate 3D conformations and runs inference
    through the pre-trained Uni-Mol backbone.  Results are cached to
    disk so the expensive extraction only runs once.

    Returns:
        np.ndarray of shape (N, 512). Rows for failed molecules are zero.
    """
    if cache_path is not None and Path(cache_path).exists():
        reprs = np.load(cache_path)
        print(f"  Loaded cached representations: {reprs.shape}")
        return reprs

    from unimol_tools import UniMolRepr
    clf = UniMolRepr(data_type="molecule", remove_hs=False)

    all_reprs = []
    n_failed  = 0
    for start in range(0, len(smiles_list), batch_size):
        batch_smi = smiles_list[start:start + batch_size]
        try:
            batch_reprs = clf.get_repr(batch_smi, return_atomic_reprs=False)
            # Returns list of (512,) arrays — stack to (B, 512)
            all_reprs.append(np.stack(batch_reprs, axis=0))
        except Exception as e:
            print(f"  Warning: batch {start//batch_size} failed ({e}), using zeros")
            all_reprs.append(np.zeros((len(batch_smi), 512), dtype=np.float32))
            n_failed += len(batch_smi)
        if (start // batch_size + 1) % 5 == 0:
            print(f"  Processed {min(start+batch_size, len(smiles_list))}/{len(smiles_list)} molecules...")

    reprs = np.concatenate(all_reprs, axis=0).astype(np.float32)
    if n_failed:
        print(f"  Note: {n_failed} molecules used zero representations")

    if cache_path is not None:
        np.save(cache_path, reprs)
        print(f"  Representations cached: {cache_path}")

    print(f"  Extracted representations: {reprs.shape}")
    return reprs


# ── Dataset ───────────────────────────────────────────────────────────────────

class UniMolDataset(Dataset):
    def __init__(self, reprs: np.ndarray, labels: np.ndarray):
        self.reprs  = torch.tensor(reprs,  dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.reprs[idx], self.labels[idx]


# ── MLP head ──────────────────────────────────────────────────────────────────

class UniMolPredictor(nn.Module):
    """
    Two-layer MLP on top of frozen Uni-Mol 512-dim representations.

    Architecture:
        Linear(512→256) → GELU → Dropout → LayerNorm
        → Linear(256→num_tasks)
    """
    def __init__(self, repr_dim: int = 512, hidden: int = 256,
                 dropout: float = 0.3, num_tasks: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_tasks),
        )
        self.num_tasks = num_tasks

    def forward(self, x):
        return self.net(x)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, criterion, task_config):
    model.eval()
    all_logits, all_labels, losses = [], [], []
    for reprs, lbls in loader:
        reprs, lbls = reprs.to(device), lbls.to(device)
        logits = model(reprs)
        losses.append(criterion(logits, lbls).item())
        all_logits.append(logits.cpu())
        all_labels.append(lbls.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = 1 / (1 + np.exp(-all_logits))

    per_auc, per_pr = {}, {}
    for t, name in enumerate(task_config.task_names):
        valid = ~np.isnan(all_labels[:, t])
        if valid.sum() < 2 or len(np.unique(all_labels[valid, t])) < 2:
            continue
        per_auc[name] = roc_auc_score(all_labels[valid, t], all_probs[valid, t])
        per_pr[name]  = average_precision_score(all_labels[valid, t], all_probs[valid, t])

    return {
        "loss":             float(np.mean(losses)),
        "mean_auc_roc":     float(np.mean(list(per_auc.values()))) if per_auc else 0.0,
        "mean_pr_auc":      float(np.mean(list(per_pr.values()))) if per_pr else 0.0,
        "per_task_auc_roc": per_auc,
        "per_task_pr_auc":  per_pr,
        "num_valid_tasks":  len(per_auc),
    }


def print_metrics(metrics, task_config, split="Test"):
    print(f"\n{split} Set Results:")
    print("=" * 70)
    print(f"LOSS         : {metrics['loss']:.4f}")
    print(f"MEAN_AUC_ROC : {metrics['mean_auc_roc']:.4f}  "
          f"({metrics['num_valid_tasks']}/{task_config.num_tasks} tasks)")
    print(f"MEAN_PR_AUC  : {metrics['mean_pr_auc']:.4f}")
    print("\nPer-task AUC-ROC:")
    for task, auc in sorted(metrics["per_task_auc_roc"].items()):
        bar = "█" * int(auc * 20)
        print(f"  {task:<20} {auc:.4f}  {bar}")
    print("=" * 70)


def save_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Uni-Mol + MLP — Tox21 Training Curves", fontsize=13)
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["val_mean_auc_roc"], color="steelblue")
    axes[1].set_title("Val Mean AUC-ROC"); axes[1].grid(True)
    axes[2].plot(history["val_mean_pr_auc"], color="coral")
    axes[2].set_title("Val Mean PR-AUC"); axes[2].grid(True)
    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Uni-Mol + MLP on Tox21")
    parser.add_argument("--config", type=str, default="config/tox21_unimol_config.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config_path = project_root / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    dc = config["data"]
    oc = config.get("output", {})

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    set_seed(dc.get("seed", 42))
    task_config = get_task_config("tox21")

    print("=" * 70)
    print("Uni-Mol + MLP — Tox21 Multi-Task Toxicity Prediction")
    print("=" * 70)
    print(f"Device  : {device}")
    print(f"Config  : {config_path}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading Tox21 dataset...")
    train_df, val_df, test_df = load_dataset(
        "tox21",
        cache_dir=str(project_root / dc["cache_dir"]),
        split_type=dc["split_type"],
        seed=dc["seed"],
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_labels = train_df[task_config.task_names].values.astype(np.float32)
    val_labels   = val_df[task_config.task_names].values.astype(np.float32)
    test_labels  = test_df[task_config.task_names].values.astype(np.float32)

    # ── Extract Uni-Mol representations ───────────────────────────────────
    cache_dir = project_root / dc["cache_dir"] / "unimol_reprs"
    ensure_dir(str(cache_dir))

    print("\nExtracting Uni-Mol 3D representations (cached after first run)...")
    train_reprs = extract_unimol_reprs(
        train_df["smiles"].tolist(),
        batch_size=int(mc.get("repr_batch_size", 256)),
        cache_path=cache_dir / "train_reprs.npy",
    )
    val_reprs = extract_unimol_reprs(
        val_df["smiles"].tolist(),
        batch_size=int(mc.get("repr_batch_size", 256)),
        cache_path=cache_dir / "val_reprs.npy",
    )
    test_reprs = extract_unimol_reprs(
        test_df["smiles"].tolist(),
        batch_size=int(mc.get("repr_batch_size", 256)),
        cache_path=cache_dir / "test_reprs.npy",
    )

    # ── Datasets + Loaders ────────────────────────────────────────────────
    train_ds = UniMolDataset(train_reprs, train_labels)
    val_ds   = UniMolDataset(val_reprs,   val_labels)
    test_ds  = UniMolDataset(test_reprs,  test_labels)

    train_sampler = None
    if tc.get("use_weighted_sampler", False):
        train_sampler = create_multitask_sampler(train_labels)
        print("Using weighted sampler for balanced training")

    batch_size   = int(tc["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None))
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    repr_dim = train_reprs.shape[1]   # 512
    model = UniMolPredictor(
        repr_dim=repr_dim,
        hidden=int(mc.get("hidden", 256)),
        dropout=float(mc.get("dropout", 0.3)),
        num_tasks=task_config.num_tasks,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nMLP parameters: {n_params:,}  (Uni-Mol backbone: frozen)")

    # ── Optimizer + Loss ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc["learning_rate"]),
        weight_decay=float(tc["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8,
    )
    criterion = MaskedMultiTaskLoss(
        alpha=float(tc.get("focal_alpha", 0.25)),
        gamma=float(tc.get("focal_gamma", 2.0)),
    )

    # ── Training loop ─────────────────────────────────────────────────────
    num_epochs   = int(tc["num_epochs"])
    patience     = int(tc["early_stopping_patience"])
    best_metric  = float("-inf")
    best_state   = None
    patience_cnt = 0

    history = {
        "train_loss": [], "val_loss": [],
        "val_mean_auc_roc": [], "val_mean_pr_auc": [],
    }

    print(f"\nStarting training ({num_epochs} epochs max, patience={patience})...\n")
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for reprs, lbls in train_loader:
            reprs, lbls = reprs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(reprs)
            loss   = criterion(logits, lbls)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history["train_loss"].append(avg_train_loss)

        val_metrics = evaluate(model, val_loader, device, criterion, task_config)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mean_auc_roc"].append(val_metrics["mean_auc_roc"])
        history["val_mean_pr_auc"].append(val_metrics["mean_pr_auc"])

        scheduler.step(val_metrics["mean_auc_roc"])

        current = val_metrics["mean_auc_roc"]
        if current > best_metric:
            best_metric  = current
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{num_epochs} — "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Mean AUC: {current:.4f}  "
                  f"[best={best_metric:.4f}, patience={patience_cnt}/{patience}]")

        if patience_cnt >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ── Evaluate best model ───────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device, criterion, task_config)
    print_metrics(val_metrics, task_config, split="Validation")

    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, criterion, task_config)
    print_metrics(test_metrics, task_config, split="Test")

    # ── Save ──────────────────────────────────────────────────────────────
    model_dir = project_root / oc.get("model_dir", "models/tox21_unimol_model")
    ensure_dir(str(model_dir))

    torch.save(model.state_dict(), model_dir / "best_model.pt")
    shutil.copy(config_path, model_dir / "config.yaml")
    print(f"\nModel saved: {model_dir / 'best_model.pt'}")

    flat = {
        "test_mean_auc_roc": test_metrics["mean_auc_roc"],
        "test_mean_pr_auc":  test_metrics["mean_pr_auc"],
    }
    flat.update({f"test_auc_{k}": v for k, v in test_metrics["per_task_auc_roc"].items()})
    save_metrics(flat, str(model_dir / "tox21_unimol_metrics.txt"))

    save_training_curves(history, model_dir / "training_curves.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
