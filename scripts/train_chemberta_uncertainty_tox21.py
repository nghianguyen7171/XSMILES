#!/usr/bin/env python3
"""
ChemBERTa-2 with homoscedastic uncertainty weighting on Tox21.

Replaces the fixed focal loss with task-specific learnable precision weights
based on Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
Losses for Scene Geometry and Semantics", CVPR 2018.

For each task t, a log-variance parameter log_var_t is learned jointly
with the model. The weighted loss is:

    L = (1 / num_valid_tasks) * sum_t [ exp(-log_var_t) * L_t + log_var_t ]

where L_t is the per-task focal loss. Tasks with high uncertainty
(large sigma_t = exp(log_var_t/2)) are automatically down-weighted,
letting the model focus on tasks it can learn confidently.

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule
    python scripts/train_chemberta_uncertainty_tox21.py \\
        --config config/tox21_chemberta_uncertainty_config.yaml --device cuda
"""

import sys
import argparse
import warnings
import shutil
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score

from src.datasets import load_dataset, get_task_config, TaskConfig
from src.pretrained_mol_model import create_pretrained_mol_model, get_checkpoint_defaults
from src.graph_train import FocalLoss, create_multitask_sampler
from src.utils import set_seed, save_metrics, ensure_dir


# ── Uncertainty-weighted multi-task loss ──────────────────────────────────────

class UncertaintyWeightedLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting for multi-task focal loss.

    Each task t has a learnable log-variance parameter log_var_t.
    The precision (inverse variance) automatically scales the task loss:

        L = mean_over_valid_tasks[ exp(-log_var_t) * L_focal_t + log_var_t ]

    - Tasks the model finds hard → high sigma → low weight → less penalised
    - Tasks the model finds easy → low sigma → weight → more influential

    The regularisation term log_var_t prevents the model from setting all
    weights to zero to trivially minimise the loss.

    Args:
        num_tasks: Number of output tasks.
        alpha:     Focal loss alpha (rare-class weight).
        gamma:     Focal loss gamma (focusing parameter).
    """

    def __init__(self, num_tasks: int, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        # log_var_t = log(sigma_t^2); initialised to 0 → sigma=1 (uniform weights)
        self.log_var  = nn.Parameter(torch.zeros(num_tasks))
        self._focal   = FocalLoss(alpha=alpha, gamma=gamma, reduction="none")
        self.num_tasks = num_tasks

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, T) predicted logits
            targets: (B, T) labels, NaN = missing

        Returns:
            Scalar uncertainty-weighted loss.
        """
        precision  = torch.exp(-self.log_var)   # (T,) — exp(-log_var) = 1/sigma^2
        total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        n_valid    = 0

        for t in range(self.num_tasks):
            valid = ~torch.isnan(targets[:, t])
            if valid.sum() < 1:
                continue
            per_sample = self._focal(
                logits[valid, t], targets[valid, t].nan_to_num(0.0)
            ).mean()
            total_loss = total_loss + precision[t] * per_sample + self.log_var[t]
            n_valid   += 1

        return total_loss / n_valid if n_valid > 0 else total_loss

    @property
    def sigma(self) -> np.ndarray:
        """Current per-task sigma = exp(log_var / 2)."""
        return torch.exp(self.log_var / 2).detach().cpu().numpy()

    @property
    def weight(self) -> np.ndarray:
        """Current per-task precision weight = exp(-log_var)."""
        return torch.exp(-self.log_var).detach().cpu().numpy()


# ── Dataset ───────────────────────────────────────────────────────────────────

class Tox21SMILESDataset(Dataset):
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


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, criterion, task_config):
    model.eval()
    all_logits, all_labels, losses = [], [], []
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)
        logits = model(ids, mask)
        losses.append(criterion(logits, lbls).item())
        all_logits.append(logits.cpu())
        all_labels.append(lbls.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = 1.0 / (1.0 + np.exp(-all_logits))

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
    print(f"\n{split} Results:")
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


# ── Figures ───────────────────────────────────────────────────────────────────

def save_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("ChemBERTa-2 + Uncertainty Weighting — Tox21", fontsize=13)
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


def save_uncertainty_plot(sigma_history, task_names, save_path):
    """Line plot of per-task sigma (uncertainty) across training epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(sigma_history) + 1)
    sigma_arr = np.array(sigma_history)   # (epochs, T)
    colors = plt.cm.tab20(np.linspace(0, 1, len(task_names)))
    for t, (name, c) in enumerate(zip(task_names, colors)):
        ax.plot(epochs, sigma_arr[:, t], label=name, color=c, linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("σ_t (task uncertainty)")
    ax.set_title("Per-Task Uncertainty Evolution During Training")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Uncertainty plot saved: {save_path}")


def save_final_weights_bar(sigma, weights, task_names, save_path):
    """Bar chart of final learned sigma and precision weights per task."""
    x = np.arange(len(task_names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(x, sigma, color="steelblue", edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(task_names, rotation=40, ha="right", fontsize=8)
    axes[0].set_ylabel("σ_t  (higher = more uncertain)"); axes[0].set_title("Learned Task Uncertainty σ")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, weights, color="coral", edgecolor="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels(task_names, rotation=40, ha="right", fontsize=8)
    axes[1].set_ylabel("1/σ²  (higher = more confident)"); axes[1].set_title("Learned Task Precision 1/σ²")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("ChemBERTa-2 Uncertainty Weighting — Final Learned Weights", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Final weights bar chart saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ChemBERTa-2 with uncertainty weighting on Tox21"
    )
    parser.add_argument("--config", type=str,
                        default="config/tox21_chemberta_uncertainty_config.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    dc = config["data"]
    oc = config.get("output", {})

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU"); device = "cpu"

    set_seed(dc.get("seed", 42))
    task_config   = get_task_config("tox21")
    ckpt          = mc["pretrained_model"]
    max_length    = int(tc.get("max_length", 128))
    defaults      = get_checkpoint_defaults(ckpt)
    model_name    = ckpt.split("/")[-1]

    print("=" * 70)
    print("ChemBERTa-2 + Homoscedastic Uncertainty Weighting — Tox21")
    print("=" * 70)
    print(f"Device         : {device}")
    print(f"Checkpoint     : {ckpt}")
    print(f"Backbone LR    : {tc['learning_rate']}")
    print(f"Uncertainty LR : {tc['uncertainty_lr']}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading Tox21 dataset...")
    train_df, val_df, test_df = load_dataset(
        "tox21",
        cache_dir=str(project_root / dc["cache_dir"]),
        split_type=dc["split_type"],
        seed=dc["seed"],
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    def get_labels(df):
        return df[task_config.task_names].values.astype(np.float32)

    train_labels = get_labels(train_df)
    val_labels   = get_labels(val_df)
    test_labels  = get_labels(test_df)

    # ── Tokenizer + Datasets ──────────────────────────────────────────────
    print(f"\nLoading tokenizer: {ckpt} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt, trust_remote_code=defaults["trust_remote_code"]
    )
    print("Tokenizing SMILES ...")
    train_ds = Tox21SMILESDataset(train_df["smiles"].tolist(), train_labels, tokenizer, max_length)
    val_ds   = Tox21SMILESDataset(val_df["smiles"].tolist(),   val_labels,   tokenizer, max_length)
    test_ds  = Tox21SMILESDataset(test_df["smiles"].tolist(),  test_labels,  tokenizer, max_length)

    train_sampler = None
    if tc.get("use_weighted_sampler", False):
        train_sampler = create_multitask_sampler(train_labels)
        print("Using weighted sampler for balanced training")

    batch_size   = int(tc["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model + uncertainty loss ──────────────────────────────────────────
    print(f"\nLoading {model_name} ...")
    model = create_pretrained_mol_model(
        pretrained_model=ckpt,
        num_tasks=task_config.num_tasks,
        dropout=float(mc["dropout"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone parameters: {n_params:,}")

    criterion = UncertaintyWeightedLoss(
        num_tasks=task_config.num_tasks,
        alpha=float(tc.get("focal_alpha", 0.25)),
        gamma=float(tc.get("focal_gamma", 2.0)),
    ).to(device)
    print(f"Uncertainty parameters: {criterion.num_tasks} (log_var_t, one per task)")

    # ── Optimiser: two param groups — backbone + uncertainty ──────────────
    no_decay     = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    backbone_wd  = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": float(tc["weight_decay"]),
         "lr": float(tc["learning_rate"])},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": float(tc["learning_rate"])},
    ]
    # log_var parameters get a higher LR so they adapt faster than backbone
    uncertainty_group = {
        "params": list(criterion.parameters()),
        "weight_decay": 0.0,
        "lr": float(tc.get("uncertainty_lr", 1e-3)),
    }
    optimizer    = torch.optim.AdamW(backbone_wd + [uncertainty_group])
    num_epochs   = int(tc["num_epochs"])
    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * float(tc.get("warmup_ratio", 0.1)))
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    grad_clip    = float(tc.get("grad_clip", 1.0))

    # ── Training loop ──────────────────────────────────────────────────────
    patience     = int(tc["early_stopping_patience"])
    best_metric  = float("-inf")
    best_state   = None
    best_crit    = None
    patience_cnt = 0

    history = {
        "train_loss": [], "val_loss": [],
        "val_mean_auc_roc": [], "val_mean_pr_auc": [],
    }
    sigma_history = []   # track per-task sigma per epoch

    print(f"\nStarting training ({num_epochs} epochs max, patience={patience})...\n")
    for epoch in range(num_epochs):
        model.train(); criterion.train()
        train_losses = []

        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = criterion(logits, lbls)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()), grad_clip
            )
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history["train_loss"].append(avg_train_loss)

        val_metrics = evaluate(model, val_loader, device, criterion, task_config)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mean_auc_roc"].append(val_metrics["mean_auc_roc"])
        history["val_mean_pr_auc"].append(val_metrics["mean_pr_auc"])
        sigma_history.append(criterion.sigma.copy())

        current = val_metrics["mean_auc_roc"]
        if current > best_metric:
            best_metric  = current
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_crit    = {k: v.cpu().clone() for k, v in criterion.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            sigma_str = "  ".join(f"{s:.2f}" for s in criterion.sigma)
            print(f"Epoch {epoch+1:>3}/{num_epochs} — "
                  f"Train: {avg_train_loss:.4f}, "
                  f"Val AUC: {current:.4f}  "
                  f"[best={best_metric:.4f}, p={patience_cnt}/{patience}]")
            print(f"  σ: [{sigma_str}]")

        if patience_cnt >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ── Evaluate best model ───────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        criterion.load_state_dict(best_crit)

    print("\nEvaluating on validation set ...")
    val_metrics = evaluate(model, val_loader, device, criterion, task_config)
    print_metrics(val_metrics, task_config, split="Validation")

    print("\nEvaluating on test set ...")
    test_metrics = evaluate(model, test_loader, device, criterion, task_config)
    print_metrics(test_metrics, task_config, split="Test")

    # ── Print learned weights ─────────────────────────────────────────────
    print("\nFinal learned task uncertainty (σ_t) and precision (1/σ_t²):")
    print(f"  {'Task':<22} {'σ_t':>8} {'1/σ²':>8}")
    print("  " + "-" * 40)
    for name, s, w in zip(task_config.task_names, criterion.sigma, criterion.weight):
        print(f"  {name:<22} {s:>8.4f} {w:>8.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    model_dir = project_root / oc.get("model_dir", "models/tox21_chemberta_uncertainty_model")
    ensure_dir(str(model_dir))

    torch.save(model.state_dict(),     model_dir / "best_model.pt")
    torch.save(criterion.state_dict(), model_dir / "uncertainty_weights.pt")
    tokenizer.save_pretrained(str(model_dir / "tokenizer"))
    shutil.copy(config_path, model_dir / "config.yaml")

    flat = {
        "test_mean_auc_roc": test_metrics["mean_auc_roc"],
        "test_mean_pr_auc":  test_metrics["mean_pr_auc"],
        "test_loss":         test_metrics["loss"],
    }
    flat.update({f"test_auc_{k}": v for k, v in test_metrics["per_task_auc_roc"].items()})
    flat.update({f"sigma_{n}": float(s)
                 for n, s in zip(task_config.task_names, criterion.sigma)})
    save_metrics(flat, str(model_dir / "tox21_chemberta_uncertainty_metrics.txt"))

    save_training_curves(history, model_dir / "training_curves.png")
    save_uncertainty_plot(sigma_history, task_config.task_names,
                          model_dir / "uncertainty_evolution.png")
    save_final_weights_bar(criterion.sigma, criterion.weight,
                           task_config.task_names, model_dir / "final_weights.png")

    # Copy figures to assets/
    import shutil as _sh
    _sh.copy(model_dir / "final_weights.png",
             project_root / "assets" / "uncertainty_weights.png")

    print(f"\nAll outputs saved to: {model_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
