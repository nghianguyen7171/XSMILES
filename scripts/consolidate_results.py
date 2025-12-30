#!/usr/bin/env python3
"""
Consolidate all model results into a single comparison table.
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_metrics

models_dir = project_root / "models"
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

# Load all model metrics
results = []

# Baseline MLP
baseline_path = models_dir / "baseline_metrics.txt"
if baseline_path.exists():
    metrics = load_metrics(str(baseline_path))
    results.append({
        'Model': 'Baseline MLP',
        'AUC-ROC': metrics.get('auc_roc', 'N/A'),
        'Accuracy': metrics.get('accuracy', 'N/A'),
        'F1 Score': metrics.get('f1', 'N/A'),
        'AUPRC': metrics.get('auprc', metrics.get('pr_auc', 'N/A'))
    })

# BFGNN (torch-molecule)
bfgnn_path = models_dir / "torch_molecule_metrics.txt"
if bfgnn_path.exists():
    metrics = load_metrics(str(bfgnn_path))
    results.append({
        'Model': 'BFGNN (torch-molecule)',
        'AUC-ROC': metrics.get('auc_roc', 'N/A'),
        'Accuracy': metrics.get('accuracy', 'N/A'),
        'F1 Score': metrics.get('f1', 'N/A'),
        'PR-AUC': metrics.get('pr_auc', 'N/A'),
        'AUPRC': metrics.get('auprc', metrics.get('pr_auc', 'N/A'))
    })

# GRIN
grin_path = models_dir / "grin_model_metrics.txt"
if grin_path.exists():
    metrics = load_metrics(str(grin_path))
    results.append({
        'Model': 'GRIN (torch-molecule)',
        'AUC-ROC': metrics.get('auc_roc', 'N/A'),
        'Accuracy': metrics.get('accuracy', 'N/A'),
        'F1 Score': metrics.get('f1', 'N/A'),
        'PR-AUC': metrics.get('pr_auc', 'N/A'),
        'AUPRC': metrics.get('auprc', metrics.get('pr_auc', 'N/A'))
    })

# SMILESTransformer
smiles_path = models_dir / "smilestransformer_model_metrics.txt"
if smiles_path.exists():
    metrics = load_metrics(str(smiles_path))
    results.append({
        'Model': 'SMILESTransformer (torch-molecule)',
        'AUC-ROC': metrics.get('auc_roc', 'N/A'),
        'Accuracy': metrics.get('accuracy', 'N/A'),
        'F1 Score': metrics.get('f1', 'N/A'),
        'PR-AUC': metrics.get('pr_auc', 'N/A'),
        'AUPRC': metrics.get('auprc', metrics.get('pr_auc', 'N/A'))
    })

# DMPNN (DeepChem)
dmpnn_path = models_dir / "dmpnn_model" / "dmpnn_model_metrics.txt"
if dmpnn_path.exists():
    metrics = load_metrics(str(dmpnn_path))
    results.append({
        'Model': 'DMPNN (DeepChem)',
        'AUC-ROC': metrics.get('test_auc_roc', metrics.get('auc_roc', 'N/A')),
        'Accuracy': metrics.get('test_accuracy', metrics.get('accuracy', 'N/A')),
        'F1 Score': metrics.get('test_f1', metrics.get('f1', 'N/A')),
        'PR-AUC': metrics.get('test_pr_auc', metrics.get('pr_auc', 'N/A')),
        'AUPRC': metrics.get('test_pr_auc', metrics.get('pr_auc', 'N/A'))
    })

# GATv2 (PyTorch Geometric)
gatv2_path = models_dir / "gatv2_model" / "gatv2_model_metrics.txt"
if gatv2_path.exists():
    metrics = load_metrics(str(gatv2_path))
    results.append({
        'Model': 'GATv2 (PyTorch Geometric)',
        'AUC-ROC': metrics.get('test_auc_roc', metrics.get('auc_roc', 'N/A')),
        'Accuracy': metrics.get('test_accuracy', metrics.get('accuracy', 'N/A')),
        'F1 Score': metrics.get('test_f1', metrics.get('f1', 'N/A')),
        'PR-AUC': metrics.get('test_pr_auc', metrics.get('pr_auc', 'N/A')),
        'AUPRC': metrics.get('test_pr_auc', metrics.get('pr_auc', 'N/A'))
    })

# GIN (PyTorch Geometric)
gin_path = models_dir / "gin_model" / "gin_model_metrics.txt"
if gin_path.exists():
    metrics = load_metrics(str(gin_path))
    results.append({
        'Model': 'GIN (PyTorch Geometric)',
        'AUC-ROC': metrics.get('test_auc_roc', metrics.get('auc_roc', 'N/A')),
        'Accuracy': metrics.get('test_accuracy', metrics.get('accuracy', 'N/A')),
        'F1 Score': metrics.get('test_f1', metrics.get('f1', 'N/A')),
        'PR-AUC': metrics.get('test_pr_auc', metrics.get('pr_auc', 'N/A')),
        'AUPRC': metrics.get('test_pr_auc', metrics.get('pr_auc', 'N/A'))
    })

# SMILESGNN
smilesgnn_path = models_dir / "smilesgnn_model" / "smilesgnn_model_metrics.txt"
if smilesgnn_path.exists():
    metrics = load_metrics(str(smilesgnn_path))
    results.append({
        'Model': 'SMILESGNN (PyTorch Geometric)',
        'AUC-ROC': metrics.get('test_auc_roc', metrics.get('auc_roc', 'N/A')),
        'Accuracy': metrics.get('test_accuracy', metrics.get('accuracy', 'N/A')),
        'F1 Score': metrics.get('test_f1', metrics.get('f1', 'N/A')),
        'AUPRC': metrics.get('test_pr_auc', metrics.get('pr_auc', 'N/A'))
    })

# Create DataFrame
df = pd.DataFrame(results)

# Remove PR-AUC column if it exists
if 'PR-AUC' in df.columns:
    df = df.drop(columns=['PR-AUC'])

# Sort by AUC-ROC (ascending, so highest at bottom)
# Handle 'N/A' values by converting to float if possible
def safe_float(x):
    try:
        return float(x) if x != 'N/A' else -1
    except (ValueError, TypeError):
        return -1

df['_sort_key'] = df['AUC-ROC'].apply(safe_float)
df = df.sort_values('_sort_key', ascending=True)
df = df.drop(columns=['_sort_key'])

# Save to CSV
csv_path = results_dir / "overall_results.csv"
df.to_csv(csv_path, index=False)
print(f"Results saved to: {csv_path}")
print("\nOverall Results:")
print("=" * 80)
print(df.to_string(index=False))

# Also save as markdown table (simple format)
md_path = results_dir / "overall_results.md"
with open(md_path, 'w') as f:
    f.write("# Overall Model Performance Comparison\n\n")
    f.write("Results on ClinTox test set. Models sorted by AUC-ROC (lowest to highest).\n\n")
    f.write("| Model | AUC-ROC | Accuracy | F1 Score | AUPRC |\n")
    f.write("|-------|---------|----------|----------|-------|\n")
    for _, row in df.iterrows():
        f.write(f"| {row['Model']} | {row['AUC-ROC']} | {row['Accuracy']} | "
                f"{row['F1 Score']} | {row['AUPRC']} |\n")
    f.write("\n")
print(f"\nMarkdown table saved to: {md_path}")

