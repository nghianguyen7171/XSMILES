#!/usr/bin/env python3
"""
Train GPS Graph Transformer on the Tox21 multi-task toxicity benchmark.

GPS (Rampášek et al., NeurIPS 2022) combines local GINEConv message-passing
with global multi-head self-attention in each layer — capturing both local
chemical environments and long-range molecular interactions.

Usage:
    conda activate drug-tox-env
    cd /media/nghia/HDD/PROJECT/Drug/molecule
    python scripts/train_gps_tox21.py \
        --config config/tox21_gps_config.yaml --device cuda
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
from torch_geometric.loader import DataLoader

from src.datasets import load_dataset, get_task_config, TaskConfig
from src.graph_data import smiles_to_pyg_data, get_feature_dims
from src.gps_model import create_gps_model
from src.graph_train import (
    train_gatv2_model as _train,
    evaluate_model,
    create_multitask_sampler,
)
from src.utils import set_seed, save_metrics, ensure_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def prepare_dataset(df, task_config: TaskConfig):
    labels_array = df[task_config.task_names].values
    dataset = []
    for i, smi in enumerate(df["smiles"]):
        data = smiles_to_pyg_data(smi, label=labels_array[i])
        if data is not None:
            dataset.append(data)
    return dataset


class GraphModelWrapper(torch.nn.Module):
    """Unpacks a PyG Batch and routes fields to GPSPredictor.forward()."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


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
    fig.suptitle("GPS Graph Transformer — Tox21 Training Curves", fontsize=13)
    axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["val_mean_auc_roc"], color="steelblue")
    axes[1].set_title("Val Mean AUC-ROC"); axes[1].grid(True)
    axes[2].plot(history["val_pr_auc"], color="coral")
    axes[2].set_title("Val Mean PR-AUC"); axes[2].grid(True)
    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GPS Graph Transformer on Tox21")
    parser.add_argument("--config", type=str, default="config/tox21_gps_config.yaml")
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
    task_config = get_task_config("tox21", loss_type=tc.get("loss_type"))

    print("=" * 70)
    print("GPS Graph Transformer — Tox21 Multi-Task Toxicity Prediction")
    print("=" * 70)
    print(f"Device         : {device}")
    print(f"Hidden channels: {mc['hidden_channels']}")
    print(f"GPS layers     : {mc['num_layers']}")
    print(f"Attention heads: {mc['heads']}")
    print(f"Tasks          : {task_config.num_tasks}")
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

    print("Converting SMILES → PyG graphs...")
    train_pyg = prepare_dataset(train_df, task_config)
    val_pyg   = prepare_dataset(val_df,   task_config)
    test_pyg  = prepare_dataset(test_df,  task_config)
    print(f"Graphs — Train: {len(train_pyg)}, Val: {len(val_pyg)}, Test: {len(test_pyg)}")

    # ── Sampler ───────────────────────────────────────────────────────────
    train_sampler = None
    if tc.get("use_weighted_sampler", False):
        labels_arr    = np.array([d.y.numpy().squeeze(0) for d in train_pyg])
        train_sampler = create_multitask_sampler(labels_arr)
        print("Using weighted sampler for balanced training")

    batch_size   = int(tc["batch_size"])
    train_loader = DataLoader(
        train_pyg, batch_size=batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
    )
    val_loader  = DataLoader(val_pyg,  batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_pyg, batch_size=batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    node_feat_dim, edge_feat_dim = get_feature_dims()
    print(f"\nNode features: {node_feat_dim}, Edge features: {edge_feat_dim}")

    gps = create_gps_model(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_channels=int(mc["hidden_channels"]),
        num_layers=int(mc["num_layers"]),
        heads=int(mc["heads"]),
        dropout=float(mc["dropout"]),
        num_tasks=task_config.num_tasks,
    )
    model = GraphModelWrapper(gps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Smoke test ────────────────────────────────────────────────────────
    sample_batch = next(iter(train_loader)).to(device)
    model.to(device)
    with torch.no_grad():
        out = model(sample_batch)
    assert out.shape == (sample_batch.num_graphs, task_config.num_tasks), \
        f"Unexpected output shape: {out.shape}"
    print(f"Smoke test passed — output shape: {out.shape}")

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nStarting training ({tc['num_epochs']} epochs max, "
          f"patience={tc['early_stopping_patience']})...\n")

    history = _train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=int(tc["num_epochs"]),
        learning_rate=float(tc["learning_rate"]),
        weight_decay=float(tc["weight_decay"]),
        device=device,
        loss_type=tc.get("loss_type", "masked_focal"),
        focal_alpha=float(tc.get("focal_alpha", 0.25)),
        focal_gamma=float(tc.get("focal_gamma", 2.0)),
        early_stopping_patience=int(tc["early_stopping_patience"]),
        early_stopping_metric=tc.get("early_stopping_metric", "mean_auc_roc"),
        task_config=task_config,
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device=device, task_config=task_config)
    print_metrics(val_metrics, task_config, split="Validation")

    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device=device, task_config=task_config)
    print_metrics(test_metrics, task_config, split="Test")

    # ── Save ──────────────────────────────────────────────────────────────
    model_dir = project_root / oc.get("model_dir", "models/tox21_gps_model")
    ensure_dir(str(model_dir))

    torch.save(model.model.state_dict(), model_dir / "best_model.pt")
    shutil.copy(config_path, model_dir / "config.yaml")
    print(f"\nModel saved: {model_dir / 'best_model.pt'}")

    flat = {
        "test_mean_auc_roc": test_metrics["mean_auc_roc"],
        "test_mean_pr_auc":  test_metrics["mean_pr_auc"],
        "test_loss":         test_metrics["loss"],
    }
    flat.update({f"test_auc_{k}": v for k, v in test_metrics["per_task_auc_roc"].items()})
    save_metrics(flat, str(model_dir / "tox21_gps_metrics.txt"))
    print(f"Metrics saved: {model_dir / 'tox21_gps_metrics.txt'}")

    save_training_curves(history, model_dir / "training_curves.png")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
