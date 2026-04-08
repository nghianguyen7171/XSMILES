#!/usr/bin/env python3
"""
Train ECFP + XGBoost models for Tox21 12-task toxicity prediction.

One XGBoost classifier is trained per Tox21 task using only the labeled
(non-NaN) compounds for that task.  Class imbalance is handled via
scale_pos_weight.  SHAP TreeExplainers are built and saved alongside
the models so that the analysis notebook can load them without re-fitting.

Usage:
    conda activate drug-tox-env
    python scripts/train_fingerprint_tox21.py [--device cuda] [--radius 2] [--nbits 2048]

Outputs (models/tox21_fingerprint_model/):
    models/<task>.pkl          — XGBoost classifier per task (12 files)
    shap/<task>.pkl            — SHAP TreeExplainer per task (12 files)
    X_test.npy                 — ECFP fingerprint matrix of the test set
    test_smiles.pkl            — SMILES list aligned to X_test
    test_labels.npy            — Label matrix (n_test, 12), NaN for missing
    tox21_fingerprint_metrics.txt
"""

import sys
import pickle
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import xgboost as xgb
import shap
from typing import Tuple
from sklearn.metrics import roc_auc_score, average_precision_score

from src.datasets import load_dataset, get_task_config
from src.fingerprint import smiles_to_ecfp
from src.utils import set_seed, save_metrics, ensure_dir


# ── Per-task helpers ──────────────────────────────────────────────────────────

def train_one_task(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_v: np.ndarray,
    y_v: np.ndarray,
    device: str = "cpu",
) -> xgb.XGBClassifier:
    """
    Train a single XGBoost binary classifier for one Tox21 assay.

    Uses scale_pos_weight to compensate for the heavy class imbalance
    (typically 15–30:1 negative:positive in Tox21).
    Early stopping is applied on the validation AUC to avoid overfitting.
    """
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators        = 500,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        min_child_weight    = 5,
        scale_pos_weight    = scale_pos_weight,
        eval_metric         = "auc",
        early_stopping_rounds = 30,
        tree_method         = "hist",
        device              = "cuda" if device == "cuda" else "cpu",
        verbosity           = 0,
        random_state        = 42,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        verbose=False,
    )
    return model


def evaluate_one_task(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
) -> Tuple:
    """Return (auc_roc, pr_auc) or (None, None) if task has only one class."""
    if len(np.unique(y_true)) < 2:
        return None, None
    probs = model.predict_proba(X)[:, 1]
    return roc_auc_score(y_true, probs), average_precision_score(y_true, probs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train ECFP+XGBoost for Tox21 multi-task toxicity prediction"
    )
    parser.add_argument("--device", type=str, default="cpu",
                        help="'cuda' or 'cpu' (XGBoost GPU acceleration)")
    parser.add_argument("--radius", type=int, default=2,
                        help="Morgan radius: 2→ECFP4 (default), 3→ECFP6")
    parser.add_argument("--nbits",  type=int, default=2048,
                        help="Fingerprint bit length (default: 2048)")
    args = parser.parse_args()

    set_seed(42)

    print("=" * 70)
    print("ECFP + XGBoost  —  Tox21 Multi-Task Toxicity Prediction")
    print("=" * 70)
    print(f"Fingerprint : ECFP{args.radius * 2}  ({args.nbits} bits)")
    print(f"Device      : {args.device}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading Tox21 (scaffold split)...")
    train_df, val_df, test_df = load_dataset(
        "tox21",
        cache_dir  = str(project_root / "data"),
        split_type = "scaffold",
        seed       = 42,
    )
    task_config = get_task_config("tox21")
    print(f"Train / Val / Test : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print()

    # ── ECFP featurization ────────────────────────────────────────────────
    print("Computing ECFP fingerprints...")
    X_train, train_ok = smiles_to_ecfp(list(train_df["smiles"]), args.radius, args.nbits)
    X_val,   val_ok   = smiles_to_ecfp(list(val_df["smiles"]),   args.radius, args.nbits)
    X_test,  test_ok  = smiles_to_ecfp(list(test_df["smiles"]),  args.radius, args.nbits)

    y_train = train_df[task_config.task_names].values[train_ok]  # (N, 12)
    y_val   = val_df[task_config.task_names].values[val_ok]
    y_test  = test_df[task_config.task_names].values[test_ok]

    test_smiles = [list(test_df["smiles"])[i] for i in test_ok]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()

    # ── Output directories ────────────────────────────────────────────────
    out_dir = project_root / "models" / "tox21_fingerprint_model"
    for sub in ("", "models", "shap"):
        ensure_dir(str(out_dir / sub))

    # ── Per-task training ─────────────────────────────────────────────────
    header = (f"{'Task':<20} {'Pos':>5} {'Neg':>5} | "
              f"{'Val AUC':>8} {'Test AUC':>9} {'PR-AUC':>8}")
    print(header)
    print("-" * 65)

    task_metrics = {}

    for t, task in enumerate(task_config.task_names):
        # Mask out NaN rows for this task
        tr_mask   = ~np.isnan(y_train[:, t])
        v_mask    = ~np.isnan(y_val[:, t])
        te_mask   = ~np.isnan(y_test[:, t])

        X_tr_t = X_train[tr_mask];  y_tr_t = y_train[tr_mask, t]
        X_v_t  = X_val[v_mask];     y_v_t  = y_val[v_mask, t]
        X_te_t = X_test[te_mask];   y_te_t = y_test[te_mask, t]

        if len(np.unique(y_tr_t)) < 2 or len(X_tr_t) < 20:
            print(f"  {task:<20} — skipped (insufficient data)")
            continue

        n_pos = int(y_tr_t.sum())
        n_neg = len(y_tr_t) - n_pos

        model = train_one_task(X_tr_t, y_tr_t, X_v_t, y_v_t, device=args.device)

        val_auc,  _        = evaluate_one_task(model, X_v_t,  y_v_t)
        test_auc, test_pr  = evaluate_one_task(model, X_te_t, y_te_t)

        print(f"  {task:<20} {n_pos:>5} {n_neg:>5} | "
              f"{val_auc:>8.4f} {test_auc:>9.4f} {test_pr:>8.4f}")

        # Save model
        with open(out_dir / "models" / f"{task}.pkl", "wb") as f:
            pickle.dump(model, f)

        # Build and save SHAP explainer
        explainer = shap.TreeExplainer(model)
        with open(out_dir / "shap" / f"{task}.pkl", "wb") as f:
            pickle.dump(explainer, f)

        task_metrics[task] = {
            "val_auc":  val_auc,
            "test_auc": test_auc,
            "test_pr":  test_pr,
        }

    # ── Summary ───────────────────────────────────────────────────────────
    valid_aucs = [v["test_auc"] for v in task_metrics.values() if v["test_auc"]]
    mean_auc   = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    mean_pr    = float(np.mean([v["test_pr"] for v in task_metrics.values() if v["test_pr"]]))

    print("-" * 65)
    print(f"  {'MEAN':<20}       | {'':>8} {mean_auc:>9.4f} {mean_pr:>8.4f}")
    print()

    # ── Save test set artefacts (needed by notebook) ──────────────────────
    np.save(out_dir / "X_test.npy",     X_test)
    np.save(out_dir / "test_labels.npy", y_test)
    with open(out_dir / "test_smiles.pkl", "wb") as f:
        pickle.dump(test_smiles, f)
    print("Test-set artefacts saved (X_test, test_labels, test_smiles).")

    # ── Save metrics ──────────────────────────────────────────────────────
    flat = {"test_mean_auc_roc": mean_auc, "test_mean_pr_auc": mean_pr}
    for task, m in task_metrics.items():
        flat[f"test_auc_{task}"] = m["test_auc"]
        flat[f"test_pr_{task}"]  = m["test_pr"]
    save_metrics(flat, str(out_dir / "tox21_fingerprint_metrics.txt"))
    print(f"Metrics  → {out_dir / 'tox21_fingerprint_metrics.txt'}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
