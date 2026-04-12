#!/usr/bin/env python3
"""
Pre-trained GIN fine-tuning for Tox21 multi-task toxicity prediction.

Uses Hu et al. (NeurIPS 2020) pretrained GIN backbone
(gin_supervised_masking.pth, pretrained on 2M molecules).

Two-group optimizer:
  backbone  → lr=1e-4  (preserve pretrained features)
  head      → lr=1e-3  (train classification head)

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule
    python scripts/train_pretrained_gin_tox21.py \\
        --config config/tox21_pretrained_gin_config.yaml --device cuda
"""

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.datasets import load_dataset, get_task_config
from src.pretrained_gnn import (
    create_pretrained_gin_model,
    smiles_list_to_hu_dataset,
)
from src.graph_train import create_multitask_sampler, evaluate_model
from src.utils import set_seed, save_metrics, ensure_dir


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        prob = torch.sigmoid(logits)
        pt   = torch.where(targets == 1, prob, 1 - prob)
        at   = torch.where(targets == 1,
                           torch.tensor(self.alpha, device=logits.device),
                           torch.tensor(1 - self.alpha, device=logits.device))
        loss = at * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss


class MaskedFocalLoss(nn.Module):
    """Masked multi-task focal loss. Ignores NaN labels."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self._focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="none")

    def forward(self, logits, targets):
        total, n = torch.tensor(0.0, device=logits.device), 0
        for t in range(logits.shape[1]):
            valid = ~torch.isnan(targets[:, t])
            if valid.sum() < 1:
                continue
            loss = self._focal(logits[valid, t], targets[valid, t].nan_to_num(0))
            total = total + loss.mean()
            n += 1
        return total / max(n, 1)


# ── GraphModelWrapper ─────────────────────────────────────────────────────────

class GraphModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    lr_backbone,
    lr_head,
    weight_decay,
    device,
    focal_alpha,
    focal_gamma,
    patience,
    grad_clip,
    task_config,
):
    criterion = MaskedFocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    # Two optimizer param groups
    backbone_params = list(model.model.backbone.parameters())
    head_params     = list(model.model.head.parameters())
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params,     "lr": lr_head,     "weight_decay": weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_auc = 0.0
    best_state   = None
    no_improve   = 0
    history      = {"train_loss": [], "val_auc": [], "val_pr_auc": []}

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            batch  = batch.to(device)
            labels = batch.y.squeeze(1) if batch.y.dim() == 3 else batch.y  # (B, T)
            optimizer.zero_grad()
            logits = model(batch)       # (B, T)
            loss   = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validate
        val_metrics = evaluate_model(model, val_loader, device, task_config=task_config)
        val_auc     = val_metrics.get("mean_auc_roc", 0.0)
        val_pr      = val_metrics.get("mean_pr_auc",  0.0)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr)

        scheduler.step(val_auc)

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            flag = " ← best" if improved else ""
            print(f"Epoch {epoch:3d}/{num_epochs} — "
                  f"Train: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, PR: {val_pr:.4f}"
                  f"  [best={best_val_auc:.4f}, p={no_improve}/{patience}]{flag}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ── Figures ───────────────────────────────────────────────────────────────────

def save_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].legend()
    axes[1].plot(history["val_auc"],    label="val AUC-ROC")
    axes[1].plot(history["val_pr_auc"], label="val PR-AUC", linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def print_task_results(metrics, task_config, label="Test"):
    print(f"\n{label} Results:")
    print("=" * 70)
    print(f"MEAN_AUC_ROC : {metrics['mean_auc_roc']:.4f}  "
          f"({metrics['num_valid_tasks']}/{task_config.num_tasks} tasks)")
    print(f"MEAN_PR_AUC  : {metrics['mean_pr_auc']:.4f}")
    print("\nPer-task AUC-ROC:")
    for task, auc in metrics["per_task_auc_roc"].items():
        bar = "█" * int(auc * 20)
        print(f"  {task:<20} {auc:.4f}  {bar}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc   = cfg["model"]
    tc   = cfg["training"]
    dc   = cfg["data"]
    oc   = cfg["output"]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    set_seed(dc.get("seed", 42))
    task_config  = get_task_config("tox21")

    print("=" * 70)
    print("Pre-trained GIN (Hu et al. 2020) — Tox21 Multi-Task")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Strategy      : {mc['strategy']}")
    print(f"Backbone LR   : {tc['lr_backbone']}")
    print(f"Head LR       : {tc['lr_head']}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading Tox21 dataset...")
    train_df, val_df, test_df = load_dataset(
        "tox21",
        cache_dir=str(project_root / dc["cache_dir"]),
        split_type=dc["split_type"],
        seed=dc.get("seed", 42),
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    def df_to_labels(df):
        return df[task_config.task_names].values.astype(np.float32)

    print("Building graph datasets (Hu et al. featurization) ...")
    train_labels = df_to_labels(train_df)
    val_labels   = df_to_labels(val_df)
    test_labels  = df_to_labels(test_df)

    train_ds = smiles_list_to_hu_dataset(train_df["smiles"].tolist(), train_labels)
    val_ds   = smiles_list_to_hu_dataset(val_df["smiles"].tolist(),   val_labels)
    test_ds  = smiles_list_to_hu_dataset(test_df["smiles"].tolist(),  test_labels)
    print(f"Graphs built: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    bs = int(tc["batch_size"])
    if tc.get("use_weighted_sampler", True):
        # Use labels from the actual dataset (some SMILES may have been filtered out)
        valid_train_labels = np.stack([d.y.numpy().squeeze(0) for d in train_ds])
        sampler      = create_multitask_sampler(valid_train_labels)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)

    val_loader  = DataLoader(val_ds,  batch_size=bs * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nLoading pretrained GIN backbone ...")
    backbone = create_pretrained_gin_model(
        num_tasks    = task_config.num_tasks,
        strategy     = mc["strategy"],
        cache_dir    = str(project_root / dc.get("pretrained_cache", "data/pretrained_gnns")),
        emb_dim      = int(mc["emb_dim"]),
        num_layers   = int(mc["num_layers"]),
        drop_ratio   = float(mc["drop_ratio"]),
        jk           = mc.get("jk", "last"),
        head_dropout = float(mc.get("head_dropout", 0.1)),
    )
    model = GraphModelWrapper(backbone).to(device)

    n_backbone = sum(p.numel() for p in backbone.backbone.parameters())
    n_head     = sum(p.numel() for p in backbone.head.parameters())
    print(f"Backbone params : {n_backbone:,}")
    print(f"Head params     : {n_head:,}")

    # ── Smoke test ────────────────────────────────────────────────────────────
    sample = next(iter(train_loader)).to(device)
    with torch.no_grad():
        out = model(sample)
    assert out.shape == (sample.num_graphs, task_config.num_tasks), \
        f"Unexpected shape: {out.shape}"
    print(f"Smoke test OK: output shape = {out.shape}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training ({tc['num_epochs']} epochs max, patience={tc['early_stopping_patience']}) ...\n")
    history = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_epochs   = int(tc["num_epochs"]),
        lr_backbone  = float(tc["lr_backbone"]),
        lr_head      = float(tc["lr_head"]),
        weight_decay = float(tc.get("weight_decay", 1e-5)),
        device       = device,
        focal_alpha  = float(tc.get("focal_alpha", 0.25)),
        focal_gamma  = float(tc.get("focal_gamma", 2.0)),
        patience     = int(tc["early_stopping_patience"]),
        grad_clip    = float(tc.get("grad_clip", 1.0)),
        task_config  = task_config,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on validation set ...")
    val_metrics  = evaluate_model(model, val_loader,  device, task_config=task_config)
    print_task_results(val_metrics, task_config, "Validation")

    print("\nEvaluating on test set ...")
    test_metrics = evaluate_model(model, test_loader, device, task_config=task_config)
    print_task_results(test_metrics, task_config, "Test")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = project_root / oc["model_dir"]
    ensure_dir(str(out_dir))

    torch.save(model.model.state_dict(), out_dir / "best_model.pt")

    import shutil
    shutil.copy(args.config, out_dir / "config.yaml")

    flat = {
        "val_mean_auc_roc":  val_metrics["mean_auc_roc"],
        "val_mean_pr_auc":   val_metrics["mean_pr_auc"],
        "test_mean_auc_roc": test_metrics["mean_auc_roc"],
        "test_mean_pr_auc":  test_metrics["mean_pr_auc"],
    }
    flat.update({f"val_auc_{k}":  v for k, v in val_metrics["per_task_auc_roc"].items()})
    flat.update({f"test_auc_{k}": v for k, v in test_metrics["per_task_auc_roc"].items()})
    save_metrics(flat, str(out_dir / "tox21_pretrained_gin_metrics.txt"))

    save_training_curves(history, out_dir / "training_curves.png")

    print(f"\nAll outputs saved to: {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
