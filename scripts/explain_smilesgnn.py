#!/usr/bin/env python3
"""
GNNExplainer-based atom and bond importance attribution for SMILESGNN.

Loads a trained SMILESGNN model and runs GNNExplainer to produce per-atom
and per-bond importance scores that explain why the model predicts a molecule
as toxic or non-toxic.

Usage examples
--------------
# Explain a single molecule from its SMILES string:
    python scripts/explain_smilesgnn.py --smiles "O=C(O)c1ccc(Cl)cc1"

# Explain all toxic molecules in the test split:
    python scripts/explain_smilesgnn.py --split test --label-filter 1

# Use CUDA, save figures to a custom directory:
    python scripts/explain_smilesgnn.py --smiles "CCN(CC)CC" \\
        --device cuda --save-dir output/explanations

# Adjust GNNExplainer optimisation steps (default 200):
    python scripts/explain_smilesgnn.py --split test --label-filter 1 --epochs 500
"""

import argparse
import pickle
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml

from src.data import load_clintox
from src.graph_data import smiles_list_to_pyg_dataset, get_feature_dims, smiles_to_pyg_data
from src.graph_models_hybrid import create_hybrid_model
from src.gnn_explainer import (
    explain_molecule,
    visualize_explanation,
    batch_explain,
    aggregate_atom_importance,
    plot_element_importance,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path, device: str):
    """Load trained SMILESGNN model and tokenizer."""
    config_path = project_root / "config" / "smilesgnn_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config["model"]

    # Load tokenizer first — its actual vocab size governs the embedding shape
    tok_path = model_dir / "tokenizer.pkl"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    actual_vocab_size = len(tokenizer.token_to_id)

    num_node_features, num_edge_features = get_feature_dims()
    model = create_hybrid_model(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=int(mc["hidden_dim"]),
        num_graph_layers=int(mc["num_graph_layers"]),
        graph_model=mc["graph_model"],
        num_heads=int(mc["num_heads"]),
        dropout=float(mc["dropout"]),
        use_residual=bool(mc.get("use_residual", True)),
        use_jk=bool(mc.get("use_jk", True)),
        jk_mode=mc.get("jk_mode", "cat"),
        graph_pooling=mc.get("graph_pooling", "meanmax"),
        smiles_vocab_size=actual_vocab_size,   # use real vocab, not config default
        smiles_d_model=int(mc["smiles_d_model"]),
        smiles_num_layers=int(mc["smiles_num_layers"]),
        fusion_method=mc.get("fusion_method", "attention"),
    )

    ckpt = model_dir / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {ckpt}\n"
            "Run 'python scripts/train_hybrid.py' first."
        )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GNNExplainer attribution for SMILESGNN toxicity model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--smiles", type=str,
        help="Single SMILES string to explain.",
    )
    src.add_argument(
        "--split", choices=["train", "val", "test"],
        help="Explain molecules from a dataset split.",
    )

    # Filters (only relevant with --split)
    parser.add_argument(
        "--label-filter", type=int, choices=[0, 1], default=None,
        help="Only explain molecules with this true label (1=toxic, 0=non-toxic). "
             "Default: explain all molecules in the split.",
    )
    parser.add_argument(
        "--max-molecules", type=int, default=None,
        help="Cap number of molecules to explain (useful for quick tests).",
    )

    # Target class
    parser.add_argument(
        "--target-class", type=int, choices=[0, 1], default=1,
        help="Class to explain (1=toxic, 0=non-toxic). Default: 1.",
    )

    # GNNExplainer settings
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="GNNExplainer optimisation steps per molecule. Default: 200.",
    )

    # Infrastructure
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or 'cuda'. Default: cpu.",
    )
    parser.add_argument(
        "--model-dir", type=str, default="models/smilesgnn_model",
        help="Path to trained model directory. Default: models/smilesgnn_model.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save visualisation PNGs. If not set, figures are shown interactively.",
    )
    parser.add_argument(
        "--element-chart", action="store_true",
        help="After batch explanation, also save an element-level importance bar chart.",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    model_dir = project_root / args.model_dir
    save_dir  = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model …")
    model, tokenizer = load_model(model_dir, device)
    print(f"  Model:     {model_dir / 'best_model.pt'}")
    print(f"  Tokenizer: {model_dir / 'tokenizer.pkl'}")
    print(f"  Device:    {device}")
    print()

    # ── Single-molecule mode ──────────────────────────────────────────────────
    if args.smiles:
        smi = args.smiles
        print(f"Explaining: {smi}")
        pyg_data = smiles_to_pyg_data(smi, label=args.target_class)
        if pyg_data is None:
            print("ERROR: RDKit could not parse the SMILES string.")
            sys.exit(1)

        result = explain_molecule(
            smiles=smi,
            model=model,
            tokenizer=tokenizer,
            pyg_data=pyg_data,
            device=device,
            epochs=args.epochs,
            target_class=args.target_class,
        )

        print(f"\nPrediction: {'Toxic' if result['predicted_class'] == 1 else 'Non-toxic'} "
              f"(P(toxic) = {result['prediction_prob']:.4f})")
        print("\nTop-5 atoms by importance:")
        atom_imp = result["atom_importance"]
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        ranked = sorted(enumerate(atom_imp), key=lambda x: x[1], reverse=True)[:5]
        for rank, (idx, imp) in enumerate(ranked, 1):
            sym = mol.GetAtomWithIdx(idx).GetSymbol()
            print(f"  {rank}. Atom {idx} ({sym}): importance = {imp:.4f}")

        save_path = str(save_dir / "explanation.png") if save_dir else None
        visualize_explanation(result, save_path=save_path)
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    print(f"Loading ClinTox {args.split} split …")
    config_path = project_root / "config" / "smilesgnn_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    dc = config["data"]

    train_df, val_df, test_df = load_clintox(
        cache_dir=str(project_root / dc["cache_dir"]),
        split_type=dc["split_type"],
        seed=dc["seed"],
    )
    split_map = {"train": train_df, "val": val_df, "test": test_df}
    df = split_map[args.split]

    smiles_list = df["smiles"].tolist()
    labels      = df["CT_TOX"].tolist()

    # Apply label filter
    if args.label_filter is not None:
        pairs = [(s, l) for s, l in zip(smiles_list, labels) if l == args.label_filter]
        smiles_list, labels = zip(*pairs) if pairs else ([], [])
        smiles_list = list(smiles_list)
        labels      = list(labels)
        print(f"  Label filter = {args.label_filter}: {len(smiles_list)} molecules")

    # Cap
    if args.max_molecules is not None:
        smiles_list = smiles_list[: args.max_molecules]
        labels      = labels[: args.max_molecules]

    print(f"  Explaining {len(smiles_list)} molecules …\n")

    pyg_dataset = smiles_list_to_pyg_dataset(smiles_list, labels=labels)

    results = batch_explain(
        smiles_list=smiles_list,
        labels=labels,
        model=model,
        tokenizer=tokenizer,
        pyg_dataset=pyg_dataset,
        device=device,
        epochs=args.epochs,
    )

    # Save / show visualisations
    for i, res in enumerate(results):
        lbl_str  = "toxic" if res["true_label"] == 1 else "nontoxic"
        filename = f"{i:03d}_{lbl_str}.png"
        save_path = str(save_dir / filename) if save_dir else None
        visualize_explanation(res, save_path=save_path)

    # Element-level summary
    if args.element_chart:
        element_scores = aggregate_atom_importance(results, label_filter=args.label_filter)
        print("\nElement-level mean importance (filtered by label):")
        for elem, score in sorted(element_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {elem}: {score:.4f}")
        chart_path = str(save_dir / "element_importance.png") if save_dir else None
        plot_element_importance(element_scores, save_path=chart_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
