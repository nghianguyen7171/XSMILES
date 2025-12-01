"""
Analysis and visualization functions for model comparison and error analysis.

This module provides reusable functions for:
1. Model performance comparison (MLP vs GNN)
2. ROC and Precision-Recall curve visualization
3. Confusion matrix analysis
4. Error overlap analysis
5. Prediction distribution visualization
6. Model agreement analysis
7. Sample molecule visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_curve, 
    precision_recall_curve, 
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Optional dependencies
try:
    from matplotlib_venn import venn2
    VENN_AVAILABLE = True
except ImportError:
    VENN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# ============================================================================
# Performance Comparison
# ============================================================================

def compare_model_performance(
    mlp_metrics: Optional[Dict[str, float]] = None,
    gnn_metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> pd.DataFrame:
    """
    Compare performance metrics between MLP and GNN models.
    
    Args:
        mlp_metrics: Metrics dictionary for MLP model
        gnn_metrics: Metrics dictionary for GNN model
        save_path: Optional path to save comparison figure
        figsize: Figure size for visualization
    
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    if mlp_metrics:
        auprc = mlp_metrics.get('auprc', mlp_metrics.get('pr_auc', 'N/A'))
        comparison_data.append({
            'Model': 'Baseline MLP (Fingerprint)',
            'AUC-ROC': mlp_metrics.get('auc_roc', 'N/A'),
            'Accuracy': mlp_metrics.get('accuracy', 'N/A'),
            'F1': mlp_metrics.get('f1', 'N/A'),
            'PR-AUC': mlp_metrics.get('pr_auc', 'N/A'),
            'AUPRC': auprc
        })
    
    if gnn_metrics:
        auprc = gnn_metrics.get('auprc', gnn_metrics.get('pr_auc', 'N/A'))
        comparison_data.append({
            'Model': 'torch-molecule GNN',
            'AUC-ROC': gnn_metrics.get('auc_roc', 'N/A'),
            'Accuracy': gnn_metrics.get('accuracy', 'N/A'),
            'F1': gnn_metrics.get('f1', 'N/A'),
            'PR-AUC': gnn_metrics.get('pr_auc', 'N/A'),
            'AUPRC': auprc
        })
    
    if not comparison_data:
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Visualize if both models available
    if len(comparison_data) > 1 and save_path:
        metrics_to_plot = ['AUC-ROC', 'Accuracy', 'F1', 'PR-AUC', 'AUPRC']
        available_metrics = [
            m for m in metrics_to_plot 
            if any(row.get(m, 'N/A') != 'N/A' for row in comparison_data)
        ]
        
        n_metrics = len(available_metrics)
        if n_metrics <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        else:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(available_metrics):
            values = []
            labels = []
            for row in comparison_data:
                val = row.get(metric, 'N/A')
                if val != 'N/A' and isinstance(val, (int, float)):
                    values.append(float(val))
                    labels.append(row['Model'])
            
            if values:
                axes[idx].bar(labels, values, alpha=0.7, color=['skyblue', 'salmon'][:len(values)])
                axes[idx].set_ylabel(metric)
                axes[idx].set_title(f'{metric} Comparison')
                axes[idx].grid(axis='y', alpha=0.3)
                axes[idx].set_ylim([0, 1])
                axes[idx].tick_params(axis='x', rotation=45)
        
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return comparison_df


# ============================================================================
# ROC and Precision-Recall Curves
# ============================================================================

def plot_roc_pr_curves(
    mlp_probs: Optional[np.ndarray],
    gnn_probs: Optional[np.ndarray],
    y_true: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 6)
) -> None:
    """
    Plot ROC and Precision-Recall curves for both models.
    
    Args:
        mlp_probs: MLP prediction probabilities
        gnn_probs: GNN prediction probabilities
        y_true: True labels
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROC Curves
    if mlp_probs is not None:
        fpr_mlp, tpr_mlp, _ = roc_curve(y_true, mlp_probs)
        roc_auc_mlp = roc_auc_score(y_true, mlp_probs)
        axes[0].plot(fpr_mlp, tpr_mlp, 
                    label=f'Baseline MLP (AUC = {roc_auc_mlp:.3f})', 
                    linewidth=2, color='#2E86AB')
    
    if gnn_probs is not None:
        fpr_gnn, tpr_gnn, _ = roc_curve(y_true, gnn_probs)
        roc_auc_gnn = roc_auc_score(y_true, gnn_probs)
        axes[0].plot(fpr_gnn, tpr_gnn, 
                    label=f'torch-molecule GNN (AUC = {roc_auc_gnn:.3f})', 
                    linewidth=2, color='#A23B72')
    
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    
    # Precision-Recall Curves
    if mlp_probs is not None:
        precision_mlp, recall_mlp, _ = precision_recall_curve(y_true, mlp_probs)
        pr_auc_mlp = average_precision_score(y_true, mlp_probs)
        axes[1].plot(recall_mlp, precision_mlp, 
                    label=f'Baseline MLP (AP = {pr_auc_mlp:.3f})', 
                    linewidth=2, color='#2E86AB')
    
    if gnn_probs is not None:
        precision_gnn, recall_gnn, _ = precision_recall_curve(y_true, gnn_probs)
        pr_auc_gnn = average_precision_score(y_true, gnn_probs)
        axes[1].plot(recall_gnn, precision_gnn, 
                    label=f'torch-molecule GNN (AP = {pr_auc_gnn:.3f})', 
                    linewidth=2, color='#A23B72')
    
    baseline_pr = y_true.mean()
    axes[1].axhline(y=baseline_pr, color='k', linestyle='--', 
                   label=f'Baseline (AP = {baseline_pr:.3f})', alpha=0.5)
    axes[1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1].set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# Confusion Matrices
# ============================================================================

def plot_confusion_matrices(
    mlp_probs: Optional[np.ndarray],
    gnn_probs: Optional[np.ndarray],
    y_true: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> Dict[str, Dict[str, int]]:
    """
    Plot confusion matrices for both models.
    
    Args:
        mlp_probs: MLP prediction probabilities
        gnn_probs: GNN prediction probabilities
        y_true: True labels
        threshold: Classification threshold
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Dictionary with confusion matrix statistics
    """
    results = {}
    
    if mlp_probs is not None or gnn_probs is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes_used = []
        
        if mlp_probs is not None:
            mlp_pred = (mlp_probs > threshold).astype(int)
            cm_mlp = confusion_matrix(y_true, mlp_pred)
            sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Non-toxic', 'Toxic'],
                        yticklabels=['Non-toxic', 'Toxic'],
                        ax=axes[0], cbar_kws={'label': 'Count'})
            axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
            axes[0].set_title('Baseline MLP Confusion Matrix', fontsize=14, fontweight='bold')
            axes_used.append(0)
            results['mlp'] = {
                'TN': int(cm_mlp[0,0]),
                'FP': int(cm_mlp[0,1]),
                'FN': int(cm_mlp[1,0]),
                'TP': int(cm_mlp[1,1])
            }
        else:
            axes[0].axis('off')
        
        if gnn_probs is not None:
            gnn_pred = (gnn_probs > threshold).astype(int)
            cm_gnn = confusion_matrix(y_true, gnn_pred)
            ax_idx = 1 if mlp_probs is not None else 0
            sns.heatmap(cm_gnn, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=['Non-toxic', 'Toxic'],
                        yticklabels=['Non-toxic', 'Toxic'],
                        ax=axes[ax_idx], cbar_kws={'label': 'Count'})
            axes[ax_idx].set_xlabel('Predicted', fontsize=12, fontweight='bold')
            axes[ax_idx].set_ylabel('Actual', fontsize=12, fontweight='bold')
            axes[ax_idx].set_title('torch-molecule GNN Confusion Matrix', fontsize=14, fontweight='bold')
            axes_used.append(ax_idx)
            results['gnn'] = {
                'TN': int(cm_gnn[0,0]),
                'FP': int(cm_gnn[0,1]),
                'FN': int(cm_gnn[1,0]),
                'TP': int(cm_gnn[1,1])
            }
        else:
            if mlp_probs is None:
                axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    return results


# ============================================================================
# Error Overlap Analysis
# ============================================================================

def analyze_error_overlap(
    mlp_probs: Optional[np.ndarray],
    gnn_probs: Optional[np.ndarray],
    y_true: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Analyze overlap in misclassified samples between models.
    
    Args:
        mlp_probs: MLP prediction probabilities
        gnn_probs: GNN prediction probabilities
        y_true: True labels
        threshold: Classification threshold
        save_path: Optional path to save Venn diagram
    
    Returns:
        Dictionary with error overlap statistics
    """
    mlp_errors = set()
    gnn_errors = set()
    
    if mlp_probs is not None:
        mlp_pred = (mlp_probs > threshold).astype(int)
        mlp_errors = set(np.where(mlp_pred != y_true)[0])
    
    if gnn_probs is not None:
        gnn_pred = (gnn_probs > threshold).astype(int)
        gnn_errors = set(np.where(gnn_pred != y_true)[0])
    
    common_errors = mlp_errors & gnn_errors
    mlp_only = mlp_errors - gnn_errors
    gnn_only = gnn_errors - mlp_errors
    total_unique = mlp_errors | gnn_errors
    
    results = {
        'mlp_only': len(mlp_only),
        'gnn_only': len(gnn_only),
        'common': len(common_errors),
        'total_unique': len(total_unique),
        'mlp_errors': mlp_errors,
        'gnn_errors': gnn_errors
    }
    
    # Plot Venn diagram if both models available
    if mlp_probs is not None and gnn_probs is not None and VENN_AVAILABLE and save_path:
        if len(mlp_errors) > 0 or len(gnn_errors) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            venn = venn2([mlp_errors, gnn_errors], 
                        set_labels=('Baseline MLP\nErrors', 'torch-molecule GNN\nErrors'),
                        ax=ax,
                        set_colors=('#2E86AB', '#A23B72'),
                        alpha=0.7)
            
            if venn.get_label_by_id('10'):
                venn.get_label_by_id('10').set_text(f'{len(mlp_only)}')
            if venn.get_label_by_id('01'):
                venn.get_label_by_id('01').set_text(f'{len(gnn_only)}')
            if venn.get_label_by_id('11'):
                venn.get_label_by_id('11').set_text(f'{len(common_errors)}')
            
            ax.set_title('Error Overlap: Misclassified Molecules', 
                        fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    return results


# ============================================================================
# Model Agreement Analysis
# ============================================================================

def analyze_model_agreement(
    mlp_probs: np.ndarray,
    gnn_probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 6)
) -> Dict[str, float]:
    """
    Analyze agreement between MLP and GNN predictions.
    
    Args:
        mlp_probs: MLP prediction probabilities
        gnn_probs: GNN prediction probabilities
        y_true: True labels
        threshold: Classification threshold
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Dictionary with agreement statistics
    """
    mlp_pred = (mlp_probs > threshold).astype(int)
    gnn_pred = (gnn_probs > threshold).astype(int)
    
    agreement = (mlp_pred == gnn_pred).mean()
    correlation = np.corrcoef(mlp_probs, gnn_probs)[0, 1]
    mean_abs_diff = np.mean(np.abs(mlp_probs - gnn_probs))
    
    results = {
        'agreement': agreement,
        'correlation': correlation,
        'mean_abs_diff': mean_abs_diff
    }
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    colors = ['#D00000' if label == 1 else '#06A77D' for label in y_true]
    axes[0].scatter(mlp_probs, gnn_probs, c=colors, alpha=0.6, s=50, 
                   edgecolors='black', linewidth=0.5)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    axes[0].set_xlabel('Baseline MLP Probability', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('torch-molecule GNN Probability', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Agreement: Prediction Probabilities', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D00000', alpha=0.6, label='Actually Toxic'),
        Patch(facecolor='#06A77D', alpha=0.6, label='Actually Non-toxic'),
        plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label='Perfect Agreement')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left')
    axes[0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=axes[0].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8))
    
    # Density heatmap
    H, xedges, yedges = np.histogram2d(mlp_probs, gnn_probs, bins=20)
    H = H.T
    im = axes[1].imshow(H, origin='lower', aspect='auto', cmap='YlOrRd', 
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    axes[1].set_xlabel('Baseline MLP Probability', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('torch-molecule GNN Probability', fontsize=12, fontweight='bold')
    axes[1].set_title('Model Agreement: Density Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1], label='Number of Samples')
    axes[1].plot([0, 1], [0, 1], 'w--', alpha=0.7, linewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return results


# ============================================================================
# Prediction Distribution Visualization
# ============================================================================

def plot_prediction_distributions(
    mlp_probs: Optional[np.ndarray],
    gnn_probs: Optional[np.ndarray],
    y_true: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Plot prediction probability distributions for both models.
    
    Args:
        mlp_probs: MLP prediction probabilities
        gnn_probs: GNN prediction probabilities
        y_true: True labels
        save_path: Optional path to save figure
        figsize: Figure size
    """
    n_models = sum([mlp_probs is not None, gnn_probs is not None])
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    row = 0
    
    # MLP distributions
    if mlp_probs is not None:
        mlp_toxic = mlp_probs[y_true == 1]
        mlp_nontoxic = mlp_probs[y_true == 0]
        
        axes[row, 0].hist(mlp_nontoxic, bins=30, alpha=0.6, label='Non-toxic', 
                         color='#06A77D', edgecolor='black')
        axes[row, 0].hist(mlp_toxic, bins=30, alpha=0.6, label='Toxic', 
                         color='#D00000', edgecolor='black')
        axes[row, 0].axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Threshold (0.5)')
        axes[row, 0].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[row, 0].set_title('Baseline MLP: Prediction Distribution', fontsize=13, fontweight='bold')
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)
        
        # KDE plot
        axes[row, 1].hist(mlp_nontoxic, bins=30, alpha=0.4, density=True, 
                         label='Non-toxic', color='#06A77D', edgecolor='black')
        axes[row, 1].hist(mlp_toxic, bins=30, alpha=0.4, density=True, 
                         label='Toxic', color='#D00000', edgecolor='black')
        try:
            from scipy import stats
            if len(mlp_nontoxic) > 1 and mlp_nontoxic.std() > 0:
                x = np.linspace(mlp_probs.min(), mlp_probs.max(), 100)
                axes[row, 1].plot(x, stats.gaussian_kde(mlp_nontoxic)(x), 
                                '--', linewidth=2, color='#06A77D', alpha=0.8)
            if len(mlp_toxic) > 1 and mlp_toxic.std() > 0:
                x = np.linspace(mlp_probs.min(), mlp_probs.max(), 100)
                axes[row, 1].plot(x, stats.gaussian_kde(mlp_toxic)(x), 
                                '--', linewidth=2, color='#D00000', alpha=0.8)
        except:
            pass
        axes[row, 1].axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Threshold (0.5)')
        axes[row, 1].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        axes[row, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[row, 1].set_title('Baseline MLP: Probability Density', fontsize=13, fontweight='bold')
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)
        row += 1
    
    # GNN distributions
    if gnn_probs is not None:
        gnn_toxic = gnn_probs[y_true == 1]
        gnn_nontoxic = gnn_probs[y_true == 0]
        
        axes[row, 0].hist(gnn_nontoxic, bins=30, alpha=0.6, label='Non-toxic', 
                         color='#06A77D', edgecolor='black')
        axes[row, 0].hist(gnn_toxic, bins=30, alpha=0.6, label='Toxic', 
                         color='#D00000', edgecolor='black')
        axes[row, 0].axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Threshold (0.5)')
        axes[row, 0].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[row, 0].set_title('torch-molecule GNN: Prediction Distribution', fontsize=13, fontweight='bold')
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)
        
        # KDE plot
        axes[row, 1].hist(gnn_nontoxic, bins=30, alpha=0.4, density=True, 
                         label='Non-toxic', color='#06A77D', edgecolor='black')
        axes[row, 1].hist(gnn_toxic, bins=30, alpha=0.4, density=True, 
                         label='Toxic', color='#D00000', edgecolor='black')
        try:
            from scipy import stats
            if len(gnn_nontoxic) > 1 and gnn_nontoxic.std() > 0:
                x = np.linspace(gnn_probs.min(), gnn_probs.max(), 100)
                axes[row, 1].plot(x, stats.gaussian_kde(gnn_nontoxic)(x), 
                                '--', linewidth=2, color='#06A77D', alpha=0.8)
            if len(gnn_toxic) > 1 and gnn_toxic.std() > 0:
                x = np.linspace(gnn_probs.min(), gnn_probs.max(), 100)
                axes[row, 1].plot(x, stats.gaussian_kde(gnn_toxic)(x), 
                                '--', linewidth=2, color='#D00000', alpha=0.8)
        except:
            pass
        axes[row, 1].axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Threshold (0.5)')
        axes[row, 1].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        axes[row, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[row, 1].set_title('torch-molecule GNN: Probability Density', fontsize=13, fontweight='bold')
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

