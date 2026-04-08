#!/usr/bin/env python3
"""
Unified training script for SMILESGNN molecular property prediction.

Supports both single-task (ClinTox) and multi-task (Tox21) experiments
via a dataset-agnostic configuration interface.  The dataset is selected
by the ``data.dataset`` field in the YAML config.

Usage:
    # Single-task: binary clinical toxicity (ClinTox)
    python scripts/train.py --config config/smilesgnn_config.yaml --device cuda

    # Multi-task: 12 mechanistic toxicity endpoints (Tox21)
    python scripts/train.py --config config/tox21_smilesgnn_config.yaml --device cuda
"""

import sys
import pickle
import argparse
import warnings
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
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from src.datasets import load_dataset, get_task_config, TaskConfig
from src.graph_data import smiles_to_pyg_data, get_feature_dims
from src.smiles_tokenizer import create_tokenizer_from_smiles
from src.graph_models_hybrid import create_hybrid_model
from src.graph_train import (
    train_gatv2_model as _train,
    evaluate_model,
    create_balanced_sampler,
    create_multitask_sampler,
)
from src.utils import set_seed, save_metrics, ensure_dir


# ── Dataset helpers ──────────────────────────────────────────────────────────

def prepare_dataset(df, task_config: TaskConfig):
    """
    Convert a DataFrame to aligned (pyg_dataset, smiles_list) pair.

    Filters out compounds whose SMILES cannot be parsed by RDKit.
    Both returned lists are index-aligned (same order, same length).

    Args:
        df:          DataFrame with "smiles" column + task columns.
        task_config: TaskConfig describing task columns and types.

    Returns:
        pyg_dataset: List of torch_geometric.data.Data objects.
        smiles_list: List of valid SMILES strings (same order).
    """
    if task_config.is_multitask:
        labels_array = df[task_config.task_names].values  # (N, T)
    else:
        labels_array = df[task_config.task_names[0]].values  # (N,)

    pyg_dataset, smiles_list = [], []
    for i, smi in enumerate(df["smiles"]):
        label = labels_array[i]
        data = smiles_to_pyg_data(smi, label=label)
        if data is not None:
            pyg_dataset.append(data)
            smiles_list.append(smi)

    return pyg_dataset, smiles_list


def print_data_stats(train_df, val_df, test_df, task_config: TaskConfig):
    """Print dataset statistics to stdout."""
    print(f"Dataset      : {task_config.name.upper()}")
    print(f"Train / Val / Test : {len(train_df)} / {len(val_df)} / {len(test_df)}")

    if task_config.is_multitask:
        print(f"\nPer-task positive rates (train):")
        for task in task_config.task_names:
            col = train_df[task]
            valid = col.dropna()
            rate = valid.mean() * 100 if len(valid) > 0 else 0.0
            print(f"  {task:<20} {rate:5.1f}%  ({int(valid.sum())}/{len(valid)} labeled)")
    else:
        task = task_config.task_names[0]
        for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            n_pos = int(df[task].sum())
            n_neg = len(df) - n_pos
            print(f"  {split_name}: Toxic={n_pos}, Non-toxic={n_neg}")


def check_data_leakage(train_df, val_df, test_df):
    """Warn if any SMILES appear in more than one split."""
    train_s, val_s, test_s = (
        set(train_df["smiles"]),
        set(val_df["smiles"]),
        set(test_df["smiles"]),
    )
    tv = train_s & val_s
    tt = train_s & test_s
    vt = val_s & test_s
    if tv:
        print(f"  ⚠ WARNING: {len(tv)} SMILES overlap train/val")
    if tt:
        print(f"  ⚠ WARNING: {len(tt)} SMILES overlap train/test")
    if vt:
        print(f"  ⚠ WARNING: {len(vt)} SMILES overlap val/test")
    if not (tv or tt or vt):
        print("  ✓ Data split validation passed: no SMILES overlap")


# ── Dataset / DataLoader wrappers ────────────────────────────────────────────

class HybridDataset:
    """Wraps a PyG dataset to also return SMILES token tensors."""

    def __init__(self, pyg_dataset, smiles_list, tokenizer):
        assert len(pyg_dataset) == len(smiles_list), (
            f"PyG dataset ({len(pyg_dataset)}) and SMILES list ({len(smiles_list)}) "
            "must have the same length."
        )
        self.pyg_dataset = pyg_dataset
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        data = self.pyg_dataset[idx]
        token_ids, attn_mask = self.tokenizer.encode(self.smiles_list[idx])
        data.smiles_token_ids = torch.tensor(token_ids, dtype=torch.long)
        data.smiles_attention_mask = torch.tensor(attn_mask, dtype=torch.long)
        return data


def collate_fn_hybrid(batch):
    """Collate PyG graphs + SMILES token tensors into a single batch."""
    batch_data = Batch.from_data_list(batch)
    if hasattr(batch[0], "smiles_token_ids"):
        batch_data.smiles_token_ids = torch.stack(
            [item.smiles_token_ids for item in batch]
        )
        batch_data.smiles_attention_masks = torch.stack(
            [item.smiles_attention_mask for item in batch]
        )
    return batch_data


class HybridModelWrapper(torch.nn.Module):
    """Thin wrapper that unpacks SMILES token fields from a PyG batch."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(
            batch,
            smiles_token_ids=getattr(batch, "smiles_token_ids", None),
            smiles_attention_mask=getattr(batch, "smiles_attention_masks", None),
        )


# ── Metric reporting ─────────────────────────────────────────────────────────

def print_metrics(metrics: dict, task_config: TaskConfig, split: str = "Test"):
    """Print evaluation results in a consistent format."""
    print(f"\n{split} Set Results:")
    print("=" * 70)
    if task_config.is_multitask:
        print(f"LOSS         : {metrics['loss']:.4f}")
        print(f"MEAN_AUC_ROC : {metrics['mean_auc_roc']:.4f}  "
              f"({metrics['num_valid_tasks']}/{task_config.num_tasks} tasks)")
        print(f"MEAN_PR_AUC  : {metrics['mean_pr_auc']:.4f}")
        print("\nPer-task AUC-ROC:")
        for task, auc in sorted(metrics['per_task_auc_roc'].items()):
            bar = "█" * int(auc * 20)
            print(f"  {task:<20} {auc:.4f}  {bar}")
    else:
        for k, v in metrics.items():
            if k in ("predictions", "labels", "logits"):
                continue
            print(f"{k.upper():<20}: {v:.4f}" if isinstance(v, float) else
                  f"{k.upper():<20}: {v}")
    print("=" * 70)


# ── Training curve plots ──────────────────────────────────────────────────────

def save_training_curves(history: dict, task_config: TaskConfig, save_path: Path):
    """Save training curves to a PNG file."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Training Curves — {task_config.name.upper()} "
        f"({'multi-task' if task_config.is_multitask else 'single-task'})",
        fontsize=13,
    )

    # Panel 1: Loss
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"],   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss"); ax.legend(); ax.grid(True)

    # Panel 2: Primary metric
    ax = axes[0, 1]
    if task_config.is_multitask:
        ax.plot(history["val_mean_auc_roc"], label="Val Mean AUC-ROC", color="steelblue")
        ax.set_ylabel("Mean AUC-ROC"); ax.set_title("Val Mean AUC-ROC")
    else:
        ax.plot(history["val_auc_roc"], label="Val AUC-ROC", color="steelblue")
        ax.set_ylabel("AUC-ROC"); ax.set_title("Val AUC-ROC")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    # Panel 3: F1 / PR-AUC
    ax = axes[1, 0]
    if not task_config.is_multitask:
        ax.plot(history["val_f1"], label="Val F1", color="coral")
        ax.set_ylabel("F1 Score"); ax.set_title("Val F1")
    else:
        ax.plot(history["val_pr_auc"], label="Val Mean PR-AUC", color="coral")
        ax.set_ylabel("Mean PR-AUC"); ax.set_title("Val Mean PR-AUC")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    # Panel 4: Accuracy (single-task) / loss zoom (multi-task)
    ax = axes[1, 1]
    if not task_config.is_multitask:
        ax.plot(history["val_accuracy"], label="Val Accuracy", color="green")
        ax.set_ylabel("Accuracy"); ax.set_title("Val Accuracy")
    else:
        ax.plot(history["val_loss"], label="Val Loss", color="dimgray")
        ax.set_ylabel("Loss"); ax.set_title("Val Loss")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train SMILESGNN for single-task or multi-task toxicity prediction"
    )
    parser.add_argument(
        "--config", type=str, default="config/smilesgnn_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use: 'cpu' or 'cuda'"
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    config_path = project_root / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc  = config["model"]
    tc  = config["training"]
    dc  = config["data"]
    oc  = config.get("output", {})

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    set_seed(dc.get("seed", 42))

    # ── TaskConfig ────────────────────────────────────────────────────────
    dataset_name = dc.get("dataset", "clintox")   # default → backward compat
    task_config = get_task_config(
        dataset_name,
        loss_type=tc.get("loss_type"),
    )

    print("=" * 70)
    print("SMILESGNN — Molecular Toxicity Prediction (Unified Trainer)")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Config     : {config_path}")
    print(f"Dataset    : {task_config.name}  ({task_config.num_tasks} task(s))")
    print(f"Loss       : {task_config.loss_type}")
    print(f"Metric     : {task_config.primary_metric}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading {task_config.name} dataset...")
    train_df, val_df, test_df = load_dataset(
        dataset_name,
        cache_dir=str(project_root / dc["cache_dir"]),
        split_type=dc["split_type"],
        seed=dc["seed"],
    )
    print()
    print_data_stats(train_df, val_df, test_df, task_config)
    print()
    check_data_leakage(train_df, val_df, test_df)
    print()

    # ── Build PyG datasets (with aligned SMILES filtering) ────────────────
    print("Converting SMILES to graph representations...")
    train_pyg, train_smiles = prepare_dataset(train_df, task_config)
    val_pyg,   val_smiles   = prepare_dataset(val_df,   task_config)
    test_pyg,  test_smiles  = prepare_dataset(test_df,  task_config)
    print(f"Train: {len(train_pyg)}, Val: {len(val_pyg)}, Test: {len(test_pyg)}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print("\nBuilding SMILES tokenizer...")
    tokenizer = create_tokenizer_from_smiles(
        smiles_list=train_smiles,
        vocab_size=int(mc.get("smiles_vocab_size", 100)),
        max_length=int(mc.get("smiles_max_length", 128)),
        min_freq=1,
    )
    actual_vocab_size = len(tokenizer.token_to_id)
    print(f"Vocabulary size: {actual_vocab_size}")

    # ── Datasets & DataLoaders ────────────────────────────────────────────
    train_ds = HybridDataset(train_pyg, train_smiles, tokenizer)
    val_ds   = HybridDataset(val_pyg,   val_smiles,   tokenizer)
    test_ds  = HybridDataset(test_pyg,  test_smiles,  tokenizer)

    num_node_features, num_edge_features = get_feature_dims()
    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}\n")

    batch_size = int(tc["batch_size"])

    # Weighted sampler for balanced training
    train_sampler = None
    if tc.get("use_weighted_sampler", False):
        if task_config.is_multitask:
            # d.y has shape (1, num_tasks) after batching fix → squeeze to (num_tasks,)
            labels_arr = np.array([d.y.numpy().squeeze(0) for d in train_pyg])
            train_sampler = create_multitask_sampler(labels_arr)
        else:
            labels_list = [d.y.item() for d in train_pyg]
            train_sampler = create_balanced_sampler(labels_list)
        print("Using weighted sampler for balanced training")

    print("Creating data loaders...")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=collate_fn_hybrid, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_hybrid, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_hybrid, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nCreating SMILESGNN model...")
    model = create_hybrid_model(
        num_node_features = num_node_features,
        num_edge_features = num_edge_features,
        hidden_dim        = int(mc["hidden_dim"]),
        num_graph_layers  = int(mc["num_graph_layers"]),
        graph_model       = str(mc.get("graph_model", "gatv2")),
        num_heads         = int(mc.get("num_heads", 4)),
        dropout           = float(mc["dropout"]),
        use_residual      = bool(mc.get("use_residual", True)),
        use_jk            = bool(mc.get("use_jk", True)),
        jk_mode           = str(mc.get("jk_mode", "cat")),
        graph_pooling     = str(mc.get("graph_pooling", "meanmax")),
        smiles_vocab_size = actual_vocab_size,
        smiles_d_model    = int(mc.get("smiles_d_model", 96)),
        smiles_num_layers = int(mc.get("smiles_num_layers", 2)),
        fusion_method     = str(mc.get("fusion_method", "attention")),
        smiles_pos_encoder_type = str(mc.get("smiles_pos_encoder_type", "learned")),
        output_dim        = task_config.num_tasks,   # 1 (ClinTox) or 12 (Tox21)
    )
    wrapped_model = HybridModelWrapper(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Graph encoder  : {mc.get('graph_model', 'gatv2')}")
    print(f"  SMILES encoder : Transformer ({mc.get('smiles_num_layers', 2)} layers)")
    print(f"  Fusion         : {mc.get('fusion_method', 'attention')}")
    print(f"  Output dim     : {task_config.num_tasks}  ({task_config.name})")
    print(f"  Parameters     : {n_params:,}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    print("Starting training...")
    pos_weight = None
    if tc.get("loss_type") == "weighted_bce" and not task_config.is_multitask:
        y = train_df[task_config.task_names[0]].values
        n_pos = y.sum(); n_neg = len(y) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0).to(device)

    history = _train(
        model                   = wrapped_model,
        train_loader            = train_loader,
        val_loader              = val_loader,
        num_epochs              = int(tc["num_epochs"]),
        learning_rate           = float(tc["learning_rate"]),
        weight_decay            = float(tc["weight_decay"]),
        device                  = device,
        loss_type               = task_config.loss_type,
        focal_alpha             = float(tc.get("focal_alpha", 0.25)),
        focal_gamma             = float(tc.get("focal_gamma", 2.0)),
        pos_weight              = pos_weight,
        early_stopping_patience = int(tc["early_stopping_patience"]),
        early_stopping_metric   = str(tc.get("early_stopping_metric",
                                             task_config.primary_metric)),
        verbose                 = True,
        task_config             = task_config,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(
        wrapped_model, val_loader, device=device, task_config=task_config
    )
    print_metrics(val_metrics, task_config, split="Validation")

    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        wrapped_model, test_loader, device=device, task_config=task_config
    )
    print_metrics(test_metrics, task_config, split="Test")

    # ── Save ──────────────────────────────────────────────────────────────
    model_dir = project_root / oc.get("model_dir", f"models/{task_config.name}_smilesgnn_model")
    ensure_dir(str(model_dir))

    torch.save(wrapped_model.model.state_dict(), model_dir / "best_model.pt")
    print(f"\nModel saved to: {model_dir / 'best_model.pt'}")

    with open(model_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to: {model_dir / 'tokenizer.pkl'}")

    # Metrics file
    if task_config.is_multitask:
        flat_metrics = {
            "test_mean_auc_roc": test_metrics["mean_auc_roc"],
            "test_mean_pr_auc":  test_metrics["mean_pr_auc"],
            "test_loss":         test_metrics["loss"],
        }
        flat_metrics.update({
            f"test_auc_{k}": v
            for k, v in test_metrics["per_task_auc_roc"].items()
        })
    else:
        flat_metrics = {
            "test_auc_roc": test_metrics["auc_roc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1":       test_metrics["f1"],
            "test_pr_auc":   test_metrics["pr_auc"],
            "test_loss":     test_metrics["loss"],
        }
    metrics_path = model_dir / f"{task_config.name}_smilesgnn_metrics.txt"
    save_metrics(flat_metrics, str(metrics_path))
    print(f"Metrics saved to: {metrics_path}")

    save_training_curves(history, task_config, model_dir / "training_curves.png")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
