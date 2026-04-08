#!/usr/bin/env python3
"""
Fine-tune ChemBERTa-2 (DeepChem/ChemBERTa-77M-MTR) on Tox21.

ChemBERTa-2 is pre-trained on 77M PubChem SMILES with a multi-task
regression objective, giving richer chemical representations than MLM-only
pre-training.  Fine-tuning uses masked focal loss so NaN labels are ignored.

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule
    python scripts/train_chemberta_tox21.py \\
        --config config/tox21_chemberta_config.yaml \\
        --device cuda
"""

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score

from src.datasets import load_dataset, get_task_config, TaskConfig
from src.chemberta_model import create_chemberta_model
from src.graph_train import MaskedMultiTaskLoss, create_multitask_sampler
from src.utils import set_seed, save_metrics, ensure_dir


# ── Dataset ───────────────────────────────────────────────────────────────────

class Tox21SMILESDataset(Dataset):
    """
    Wraps tokenized SMILES + Tox21 labels for a standard PyTorch DataLoader.

    Labels are stored as float32 with NaN preserved for masked loss.
    """

    def __init__(
        self,
        smiles_list: list,
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, criterion, task_config: TaskConfig) -> dict:
    """Evaluate model on a DataLoader; returns loss + per-task AUC metrics."""
    model.eval()
    all_logits, all_labels, losses = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            losses.append(loss.item())

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()   # (N, T)
    all_labels = torch.cat(all_labels).numpy()   # (N, T)
    all_probs  = 1 / (1 + np.exp(-all_logits))  # sigmoid

    per_task_auc    = {}
    per_task_pr_auc = {}
    for t, task_name in enumerate(task_config.task_names):
        valid = ~np.isnan(all_labels[:, t])
        if valid.sum() < 2 or len(np.unique(all_labels[valid, t])) < 2:
            continue
        y_true  = all_labels[valid, t]
        y_score = all_probs[valid, t]
        per_task_auc[task_name]    = roc_auc_score(y_true, y_score)
        per_task_pr_auc[task_name] = average_precision_score(y_true, y_score)

    mean_auc    = float(np.mean(list(per_task_auc.values()))) if per_task_auc else 0.0
    mean_pr_auc = float(np.mean(list(per_task_pr_auc.values()))) if per_task_pr_auc else 0.0

    return {
        "loss":             float(np.mean(losses)),
        "mean_auc_roc":     mean_auc,
        "mean_pr_auc":      mean_pr_auc,
        "per_task_auc_roc": per_task_auc,
        "per_task_pr_auc":  per_task_pr_auc,
        "num_valid_tasks":  len(per_task_auc),
    }


def print_metrics(metrics: dict, task_config: TaskConfig, split: str = "Test"):
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


def save_training_curves(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("ChemBERTa-2 — Tox21 Training Curves", fontsize=13)

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["val_mean_auc_roc"], color="steelblue")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Mean AUC-ROC")
    axes[1].set_title("Val Mean AUC-ROC"); axes[1].grid(True)

    axes[2].plot(history["val_mean_pr_auc"], color="coral")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Mean PR-AUC")
    axes[2].set_title("Val Mean PR-AUC"); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ChemBERTa-2 on Tox21 multi-task toxicity prediction"
    )
    parser.add_argument("--config", type=str,
                        default="config/tox21_chemberta_config.yaml")
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
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    set_seed(dc.get("seed", 42))
    task_config = get_task_config("tox21")

    print("=" * 70)
    print("ChemBERTa-2 — Tox21 Multi-Task Toxicity Prediction")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Checkpoint : {mc['pretrained_model']}")
    print(f"Tasks      : {task_config.num_tasks}")
    print(f"Loss       : {tc['loss_type']}")
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

    # Labels as float arrays with NaN
    def get_labels(df):
        return df[task_config.task_names].values.astype(np.float32)

    train_labels = get_labels(train_df)
    val_labels   = get_labels(val_df)
    test_labels  = get_labels(test_df)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {mc['pretrained_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(mc["pretrained_model"])
    max_length = int(tc.get("max_length", 128))

    print("Tokenizing SMILES (train)...")
    train_ds = Tox21SMILESDataset(
        train_df["smiles"].tolist(), train_labels, tokenizer, max_length)
    print("Tokenizing SMILES (val)...")
    val_ds   = Tox21SMILESDataset(
        val_df["smiles"].tolist(),   val_labels,   tokenizer, max_length)
    print("Tokenizing SMILES (test)...")
    test_ds  = Tox21SMILESDataset(
        test_df["smiles"].tolist(),  test_labels,  tokenizer, max_length)

    # ── Sampler ───────────────────────────────────────────────────────────
    train_sampler = None
    if tc.get("use_weighted_sampler", False):
        train_sampler = create_multitask_sampler(train_labels)
        print("Using weighted sampler for balanced training")

    batch_size = int(tc["batch_size"])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader  = DataLoader(val_ds,  batch_size=batch_size,
                             shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nLoading ChemBERTa-2 ({mc['pretrained_model']})...")
    model = create_chemberta_model(
        pretrained_model = mc["pretrained_model"],
        num_tasks        = task_config.num_tasks,
        dropout          = float(mc["dropout"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_train:,} trainable")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    # Separate weight decay: apply to weights only, not bias/LayerNorm
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": float(tc["weight_decay"])},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=float(tc["learning_rate"]))

    num_epochs    = int(tc["num_epochs"])
    total_steps   = len(train_loader) * num_epochs
    warmup_steps  = int(total_steps * float(tc.get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = MaskedMultiTaskLoss(
        alpha=float(tc.get("focal_alpha", 0.25)),
        gamma=float(tc.get("focal_gamma", 2.0)),
    )
    grad_clip = float(tc.get("grad_clip", 1.0))

    # ── Training loop ─────────────────────────────────────────────────────
    patience      = int(tc["early_stopping_patience"])
    best_metric   = float("-inf")
    best_state    = None
    patience_cnt  = 0

    history = {
        "train_loss": [], "val_loss": [],
        "val_mean_auc_roc": [], "val_mean_pr_auc": [],
    }

    print(f"\nStarting fine-tuning ({num_epochs} epochs max)...")
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history["train_loss"].append(avg_train_loss)

        # Validation
        val_metrics = evaluate(model, val_loader, device, criterion, task_config)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mean_auc_roc"].append(val_metrics["mean_auc_roc"])
        history["val_mean_pr_auc"].append(val_metrics["mean_pr_auc"])

        current = val_metrics["mean_auc_roc"]
        if current > best_metric:
            best_metric = current
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{num_epochs} — "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Mean AUC: {current:.4f}  "
                  f"[best={best_metric:.4f}, patience={patience_cnt}/{patience}]")

        if patience_cnt >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ── Load best + evaluate ───────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating on validation set...")
    val_metrics  = evaluate(model, val_loader,  device, criterion, task_config)
    print_metrics(val_metrics, task_config, split="Validation")

    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, criterion, task_config)
    print_metrics(test_metrics, task_config, split="Test")

    # ── Save ──────────────────────────────────────────────────────────────
    model_dir = project_root / oc.get("model_dir", "models/tox21_chemberta_model")
    ensure_dir(str(model_dir))

    torch.save(model.state_dict(), model_dir / "best_model.pt")
    tokenizer.save_pretrained(str(model_dir / "tokenizer"))
    print(f"\nModel saved to: {model_dir / 'best_model.pt'}")

    import shutil
    shutil.copy(config_path, model_dir / "config.yaml")

    flat_metrics = {
        "test_mean_auc_roc": test_metrics["mean_auc_roc"],
        "test_mean_pr_auc":  test_metrics["mean_pr_auc"],
        "test_loss":         test_metrics["loss"],
    }
    flat_metrics.update({
        f"test_auc_{k}": v for k, v in test_metrics["per_task_auc_roc"].items()
    })
    metrics_path = model_dir / "tox21_chemberta_metrics.txt"
    save_metrics(flat_metrics, str(metrics_path))
    print(f"Metrics saved to: {metrics_path}")

    save_training_curves(history, model_dir / "training_curves.png")
    print("\nFine-tuning completed successfully!")


if __name__ == "__main__":
    main()
