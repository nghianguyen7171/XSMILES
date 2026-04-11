#!/usr/bin/env python3
"""
ChemBERTa-2 fine-tuning with SMILES Enumeration Augmentation on Tox21.

SMILES enumeration generates K random-but-valid SMILES strings for the
same molecule using RDKit's canonical randomisation. Training sees diverse
token sequences for each compound; test-time averaging (TTA) over K
variants reduces variance in predictions.

Training augmentation:
    Each __getitem__ randomly generates a new valid SMILES → different
    token sequence → model cannot overfit to a single canonical form.

Test-time augmentation (TTA):
    For each test molecule, K random SMILES are generated, each is
    tokenised and forwarded through the model, and the K logit vectors
    are averaged before computing probabilities.

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule
    python scripts/train_chemberta_aug_tox21.py \
        --config config/tox21_chemberta_aug_config.yaml --device cuda
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from rdkit import Chem
from sklearn.metrics import roc_auc_score, average_precision_score

from src.datasets import load_dataset, get_task_config, TaskConfig
from src.pretrained_mol_model import create_pretrained_mol_model, get_checkpoint_defaults
from src.graph_train import MaskedMultiTaskLoss, create_multitask_sampler
from src.utils import set_seed, save_metrics, ensure_dir


# ── Augmented SMILES dataset ──────────────────────────────────────────────────

class AugmentedSMILESDataset(Dataset):
    """
    Tox21 dataset with on-the-fly SMILES augmentation.

    In augment mode (training), each __getitem__ call generates a new
    random valid SMILES for the molecule. This means every epoch presents
    a different tokenisation for each compound, preventing the model from
    memorising canonical SMILES token patterns.

    Falls back to canonical SMILES for invalid molecules or when
    augment=False (val/test sets).
    """

    def __init__(
        self,
        smiles_list,
        labels,
        tokenizer,
        max_length: int = 128,
        augment: bool = True,
    ):
        self.smiles_list = smiles_list
        self.labels      = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.augment     = augment

        # Pre-parse mol objects once; None = invalid SMILES
        self.mols = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                self.mols.append(mol)
            except Exception:
                self.mols.append(None)

    def _sample_smiles(self, idx: int) -> str:
        """Return a random SMILES for molecule idx (canonical fallback)."""
        mol = self.mols[idx]
        if mol is None or not self.augment:
            return self.smiles_list[idx]
        try:
            return Chem.MolToSmiles(mol, doRandom=True)
        except Exception:
            return self.smiles_list[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        smi = self._sample_smiles(idx)
        enc = self.tokenizer(
            smi,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         self.labels[idx],
        }


# ── Test-time augmentation inference ─────────────────────────────────────────

@torch.no_grad()
def tta_evaluate(model, smiles_list, labels, tokenizer, max_length,
                 device, criterion, task_config, tta_k=10, batch_size=32):
    """
    Evaluate with test-time augmentation.

    For each molecule, generate tta_k random SMILES and average logits.
    Falls back to canonical SMILES for invalid molecules.
    """
    model.eval()
    n = len(smiles_list)
    T = task_config.num_tasks

    # Pre-parse mols
    mols = []
    for smi in smiles_list:
        try:
            mols.append(Chem.MolFromSmiles(smi))
        except Exception:
            mols.append(None)

    # Build (n * tta_k) augmented SMILES list, tracking molecule index
    aug_smiles = []
    mol_idx    = []
    for i, (smi, mol) in enumerate(zip(smiles_list, mols)):
        for _ in range(tta_k):
            if mol is not None:
                try:
                    aug_smiles.append(Chem.MolToSmiles(mol, doRandom=True))
                except Exception:
                    aug_smiles.append(smi)
            else:
                aug_smiles.append(smi)
            mol_idx.append(i)

    # Batch inference over all augmented SMILES
    all_logits = []
    for start in range(0, len(aug_smiles), batch_size):
        batch_smi = aug_smiles[start:start + batch_size]
        enc = tokenizer(
            batch_smi,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        logits = model(ids, mask)
        all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits).numpy()   # (n * tta_k, T)

    # Average logits per molecule
    avg_logits = np.zeros((n, T), dtype=np.float32)
    counts     = np.zeros(n, dtype=np.int32)
    for k, i in enumerate(mol_idx):
        avg_logits[i] += all_logits[k]
        counts[i]     += 1
    avg_logits /= counts[:, None]

    # Compute loss on averaged logits
    labels_t = torch.tensor(labels, dtype=torch.float32)
    logits_t = torch.tensor(avg_logits)
    loss_val = criterion(logits_t, labels_t).item()

    # Compute metrics
    all_probs = 1.0 / (1.0 + np.exp(-avg_logits))
    per_auc, per_pr = {}, {}
    for t, name in enumerate(task_config.task_names):
        valid = ~np.isnan(labels[:, t])
        if valid.sum() < 2 or len(np.unique(labels[valid, t])) < 2:
            continue
        per_auc[name] = roc_auc_score(labels[valid, t], all_probs[valid, t])
        per_pr[name]  = average_precision_score(labels[valid, t], all_probs[valid, t])

    return {
        "loss":             loss_val,
        "mean_auc_roc":     float(np.mean(list(per_auc.values()))) if per_auc else 0.0,
        "mean_pr_auc":      float(np.mean(list(per_pr.values()))) if per_pr else 0.0,
        "per_task_auc_roc": per_auc,
        "per_task_pr_auc":  per_pr,
        "num_valid_tasks":  len(per_auc),
    }


# ── Standard (non-TTA) evaluation ────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, criterion, task_config):
    model.eval()
    all_logits, all_labels, losses = [], [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
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


def save_training_curves(history, model_name, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} + SMILES Augmentation — Tox21 Training", fontsize=13)
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
    parser = argparse.ArgumentParser(
        description="ChemBERTa-2 + SMILES augmentation on Tox21"
    )
    parser.add_argument("--config", type=str,
                        default="config/tox21_chemberta_aug_config.yaml")
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
    ckpt        = mc["pretrained_model"]
    max_length  = int(tc.get("max_length", 128))
    num_aug_k   = int(tc.get("num_aug_k", 5))
    tta_k       = int(tc.get("tta_k", 10))
    model_name  = ckpt.split("/")[-1]

    print("=" * 70)
    print("ChemBERTa-2 + SMILES Enumeration Augmentation — Tox21")
    print("=" * 70)
    print(f"Device      : {device}")
    print(f"Checkpoint  : {ckpt}")
    print(f"Train aug K : {num_aug_k} random SMILES/molecule/epoch")
    print(f"TTA K       : {tta_k} random SMILES/molecule at test time")
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
    defaults  = get_checkpoint_defaults(ckpt)
    print(f"\nLoading tokenizer: {ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt, trust_remote_code=defaults["trust_remote_code"]
    )

    # Training: augment=True; Val/Test: augment=False (canonical SMILES)
    print("Building datasets (augmented train, canonical val/test)...")
    train_ds = AugmentedSMILESDataset(
        train_df["smiles"].tolist(), train_labels, tokenizer, max_length, augment=True
    )
    val_ds = AugmentedSMILESDataset(
        val_df["smiles"].tolist(), val_labels, tokenizer, max_length, augment=False
    )

    # ── Samplers + DataLoaders ────────────────────────────────────────────
    train_sampler = None
    if tc.get("use_weighted_sampler", False):
        train_sampler = create_multitask_sampler(train_labels)
        print("Using weighted sampler for balanced training")

    batch_size   = int(tc["batch_size"])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nLoading {model_name}...")
    model = create_pretrained_mol_model(
        pretrained_model=ckpt,
        num_tasks=task_config.num_tasks,
        dropout=float(mc["dropout"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    no_decay     = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": float(tc["weight_decay"])},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer    = torch.optim.AdamW(param_groups, lr=float(tc["learning_rate"]))
    num_epochs   = int(tc["num_epochs"])
    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * float(tc.get("warmup_ratio", 0.1)))
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion  = MaskedMultiTaskLoss(
        alpha=float(tc.get("focal_alpha", 0.25)),
        gamma=float(tc.get("focal_gamma", 2.0)),
    )
    grad_clip  = float(tc.get("grad_clip", 1.0))

    # ── Training loop ─────────────────────────────────────────────────────
    patience     = int(tc["early_stopping_patience"])
    best_metric  = float("-inf")
    best_state   = None
    patience_cnt = 0

    history = {
        "train_loss": [], "val_loss": [],
        "val_mean_auc_roc": [], "val_mean_pr_auc": [],
    }

    print(f"\nStarting training ({num_epochs} epochs, patience={patience})...\n")
    for epoch in range(num_epochs):
        model.train()
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
                  f"Train: {avg_train_loss:.4f}, "
                  f"Val: {val_metrics['loss']:.4f}, "
                  f"Val AUC: {current:.4f}  "
                  f"[best={best_metric:.4f}, p={patience_cnt}/{patience}]")

        if patience_cnt >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ── Load best + evaluate ───────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n--- Standard evaluation (canonical SMILES) ---")
    val_metrics = evaluate(model, val_loader, device, criterion, task_config)
    print_metrics(val_metrics, task_config, split="Validation (canonical)")

    print(f"\n--- Test-time augmentation (TTA, K={tta_k}) ---")
    test_tta = tta_evaluate(
        model,
        test_df["smiles"].tolist(), test_labels,
        tokenizer, max_length, device, criterion, task_config,
        tta_k=tta_k, batch_size=batch_size,
    )
    print_metrics(test_tta, task_config, split=f"Test (TTA K={tta_k})")

    print("\n--- Standard test evaluation (canonical, no TTA) ---")
    test_canonical_ds = AugmentedSMILESDataset(
        test_df["smiles"].tolist(), test_labels, tokenizer, max_length, augment=False
    )
    test_loader = DataLoader(test_canonical_ds, batch_size=batch_size, shuffle=False)
    test_canonical = evaluate(model, test_loader, device, criterion, task_config)
    print_metrics(test_canonical, task_config, split="Test (canonical, no TTA)")

    # ── Save ──────────────────────────────────────────────────────────────
    model_dir = project_root / oc.get("model_dir", "models/tox21_chemberta_aug_model")
    ensure_dir(str(model_dir))

    torch.save(model.state_dict(), model_dir / "best_model.pt")
    tokenizer.save_pretrained(str(model_dir / "tokenizer"))
    shutil.copy(config_path, model_dir / "config.yaml")
    print(f"\nModel saved: {model_dir / 'best_model.pt'}")

    flat = {
        "test_tta_mean_auc_roc":       test_tta["mean_auc_roc"],
        "test_tta_mean_pr_auc":        test_tta["mean_pr_auc"],
        "test_canonical_mean_auc_roc": test_canonical["mean_auc_roc"],
        "test_canonical_mean_pr_auc":  test_canonical["mean_pr_auc"],
    }
    flat.update({f"test_tta_auc_{k}": v
                 for k, v in test_tta["per_task_auc_roc"].items()})
    save_metrics(flat, str(model_dir / "tox21_chemberta_aug_metrics.txt"))

    save_training_curves(history, model_name, model_dir / "training_curves.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
