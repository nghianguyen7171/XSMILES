#!/usr/bin/env python3
"""
Unified fine-tuning script for pre-trained molecular foundation models on Tox21.

Supports any HuggingFace SMILES transformer via --config:
    DeepChem/ChemBERTa-77M-MTR          → config/tox21_chemberta_config.yaml
    ibm/MoLFormer-XL-both-10pct         → config/tox21_molformer_config.yaml
    seyonec/PubChem10M_SMILES_BPE_450k  → config/tox21_chemberta_pubchem_config.yaml

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule

    # MoLFormer-XL
    python scripts/train_pretrained_tox21.py \\
        --config config/tox21_molformer_config.yaml --device cuda

    # ChemBERTa-PubChem
    python scripts/train_pretrained_tox21.py \\
        --config config/tox21_chemberta_pubchem_config.yaml --device cuda
"""

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score

from src.datasets import load_dataset, get_task_config, TaskConfig
from src.pretrained_mol_model import (
    create_pretrained_mol_model,
    get_checkpoint_defaults,
)
from src.graph_train import MaskedMultiTaskLoss, create_multitask_sampler
from src.utils import set_seed, save_metrics, ensure_dir


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

def evaluate(model, loader, device, criterion, task_config: TaskConfig) -> dict:
    model.eval()
    all_logits, all_labels, losses = [], [], []

    with torch.no_grad():
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
    all_probs  = 1 / (1 + np.exp(-all_logits))

    per_task_auc, per_task_pr = {}, {}
    for t, name in enumerate(task_config.task_names):
        valid = ~np.isnan(all_labels[:, t])
        if valid.sum() < 2 or len(np.unique(all_labels[valid, t])) < 2:
            continue
        per_task_auc[name] = roc_auc_score(all_labels[valid, t], all_probs[valid, t])
        per_task_pr[name]  = average_precision_score(all_labels[valid, t], all_probs[valid, t])

    mean_auc    = float(np.mean(list(per_task_auc.values()))) if per_task_auc else 0.0
    mean_pr_auc = float(np.mean(list(per_task_pr.values()))) if per_task_pr else 0.0

    return {
        "loss":             float(np.mean(losses)),
        "mean_auc_roc":     mean_auc,
        "mean_pr_auc":      mean_pr_auc,
        "per_task_auc_roc": per_task_auc,
        "per_task_pr_auc":  per_task_pr,
        "num_valid_tasks":  len(per_task_auc),
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


def save_training_curves(history, model_name, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} — Tox21 Training Curves", fontsize=13)
    axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["val_mean_auc_roc"], color="steelblue")
    axes[1].set_title("Val Mean AUC-ROC"); axes[1].grid(True)
    axes[2].plot(history["val_mean_pr_auc"], color="coral")
    axes[2].set_title("Val Mean PR-AUC"); axes[2].grid(True)
    for ax in axes: ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained molecular model on Tox21"
    )
    parser.add_argument("--config", type=str, required=True)
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
    task_config  = get_task_config("tox21")
    ckpt         = mc["pretrained_model"]
    ckpt_defaults = get_checkpoint_defaults(ckpt)
    max_length   = int(tc.get("max_length", ckpt_defaults["max_length"]))
    model_name   = ckpt.split("/")[-1]

    print("=" * 70)
    print(f"Pre-trained Molecular Model — Tox21 Multi-Task Toxicity Prediction")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt}")
    print(f"Max length : {max_length}")
    print(f"Tasks      : {task_config.num_tasks}")
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

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt, trust_remote_code=ckpt_defaults["trust_remote_code"]
    )

    print("Tokenizing SMILES...")
    train_ds = Tox21SMILESDataset(train_df["smiles"].tolist(), train_labels, tokenizer, max_length)
    val_ds   = Tox21SMILESDataset(val_df["smiles"].tolist(),   val_labels,   tokenizer, max_length)
    test_ds  = Tox21SMILESDataset(test_df["smiles"].tolist(),  test_labels,  tokenizer, max_length)

    # ── Sampler + DataLoaders ─────────────────────────────────────────────
    train_sampler = None
    if tc.get("use_weighted_sampler", False):
        train_sampler = create_multitask_sampler(train_labels)
        print("Using weighted sampler for balanced training")

    batch_size = int(tc["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nLoading {model_name}...")
    model = create_pretrained_mol_model(
        pretrained_model = ckpt,
        num_tasks        = task_config.num_tasks,
        dropout          = float(mc["dropout"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} total")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
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

    num_epochs   = int(tc["num_epochs"])
    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * float(tc.get("warmup_ratio", 0.1)))
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
    patience     = int(tc["early_stopping_patience"])
    best_metric  = float("-inf")
    best_state   = None
    patience_cnt = 0

    history = {
        "train_loss": [], "val_loss": [],
        "val_mean_auc_roc": [], "val_mean_pr_auc": [],
    }

    print(f"\nStarting fine-tuning ({num_epochs} epochs max, patience={patience})...")
    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = criterion(logits, lbls)

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

        val_metrics = evaluate(model, val_loader, device, criterion, task_config)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mean_auc_roc"].append(val_metrics["mean_auc_roc"])
        history["val_mean_pr_auc"].append(val_metrics["mean_pr_auc"])

        current = val_metrics["mean_auc_roc"]
        if current > best_metric:
            best_metric  = current
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    val_metrics = evaluate(model, val_loader, device, criterion, task_config)
    print_metrics(val_metrics, task_config, split="Validation")

    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, criterion, task_config)
    print_metrics(test_metrics, task_config, split="Test")

    # ── Save ──────────────────────────────────────────────────────────────
    model_dir = project_root / oc.get("model_dir", f"models/tox21_{model_name}_model")
    ensure_dir(str(model_dir))

    torch.save(model.state_dict(), model_dir / "best_model.pt")
    tokenizer.save_pretrained(str(model_dir / "tokenizer"))
    shutil.copy(config_path, model_dir / "config.yaml")
    print(f"\nModel saved to: {model_dir / 'best_model.pt'}")

    flat_metrics = {
        "test_mean_auc_roc": test_metrics["mean_auc_roc"],
        "test_mean_pr_auc":  test_metrics["mean_pr_auc"],
        "test_loss":         test_metrics["loss"],
    }
    flat_metrics.update({f"test_auc_{k}": v for k, v in test_metrics["per_task_auc_roc"].items()})
    metrics_path = model_dir / f"tox21_{model_name}_metrics.txt"
    save_metrics(flat_metrics, str(metrics_path))
    print(f"Metrics saved to: {metrics_path}")

    save_training_curves(history, model_name, model_dir / "training_curves.png")
    print("\nFine-tuning completed successfully!")


if __name__ == "__main__":
    main()
