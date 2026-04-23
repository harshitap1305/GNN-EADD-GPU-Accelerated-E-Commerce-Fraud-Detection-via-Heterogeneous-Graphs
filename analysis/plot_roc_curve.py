"""
Plot 9: ROC Curve with AUC (Stage 2 — GAT)
==========================================
Standard Receiver Operating Characteristic curve.
Generates overall + per-entity (User/Product) ROC curves.

Data Source: anomaly_scores_stage2.npy, labels.npy, node_counts.json
"""

import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import apply_style, save_figure, data_path, check_file, COLORS


def _find_file(name):
    path = data_path(name)
    if os.path.exists(path):
        return path
    alt = data_path(f'data/{name}')
    if os.path.exists(alt):
        return alt
    return path


def plot_roc_curve():
    """Generate ROC curves for Stage 2 predictions."""
    apply_style()

    scores_file = _find_file('anomaly_scores_stage2.npy')
    labels_file = _find_file('labels.npy')
    counts_file = _find_file('node_counts.json')

    if not check_file(scores_file):
        return None
    if not check_file(labels_file):
        return None
    if not check_file(counts_file):
        return None

    all_scores = np.load(scores_file)
    labels_data = np.load(labels_file)
    with open(counts_file, 'r') as f:
        counts = json.load(f)

    N_u = counts['users']
    N_p = counts['products']

    indices = labels_data[:, 0].astype(int)
    y_true  = labels_data[:, 1].astype(int)
    y_scores = all_scores[indices]

    user_mask = indices < N_u
    prod_mask = (indices >= N_u) & (indices < N_u + N_p)

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # Random baseline
    ax.plot([0, 1], [0, 1], color=COLORS['grid'], linestyle=':', linewidth=1.2,
            label='Random (AUC = 0.500)', alpha=0.7)

    # Overall ROC
    fpr_all, tpr_all, _ = roc_curve(y_true, y_scores)
    auc_all = roc_auc_score(y_true, y_scores)
    ax.plot(fpr_all, tpr_all, color=COLORS['text'], linewidth=2.5,
            label=f'Overall  (AUC = {auc_all:.4f})', zorder=3)

    # User ROC
    if user_mask.sum() > 0 and len(np.unique(y_true[user_mask])) > 1:
        fpr_u, tpr_u, _ = roc_curve(y_true[user_mask], y_scores[user_mask])
        auc_u = roc_auc_score(y_true[user_mask], y_scores[user_mask])
        ax.plot(fpr_u, tpr_u, color=COLORS['user'], linewidth=1.8,
                linestyle='--', label=f'Users    (AUC = {auc_u:.4f})')

    # Product ROC
    if prod_mask.sum() > 0 and len(np.unique(y_true[prod_mask])) > 1:
        fpr_p, tpr_p, _ = roc_curve(y_true[prod_mask], y_scores[prod_mask])
        auc_p = roc_auc_score(y_true[prod_mask], y_scores[prod_mask])
        ax.plot(fpr_p, tpr_p, color=COLORS['product'], linewidth=1.8,
                linestyle='-.', label=f'Products (AUC = {auc_p:.4f})')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_title('Stage 2: ROC Curve\n'
                 'Semi-Supervised GAT Anomaly Detection', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')

    save_figure(fig, 'plot09_roc_curve.png')
    print("  ✓ Plot 9: ROC Curve generated.")
    return fig


if __name__ == '__main__':
    plot_roc_curve()
