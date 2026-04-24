"""
Plot 3: Precision-Recall Curve (Stage 2 — GAT)
===============================================
Gold-standard metric for imbalanced fraud detection.
Generates overall + per-entity (User/Product) PR curves.

Data Source: anomaly_scores_stage2.npy, labels.npy, node_counts.json
"""

import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import apply_style, save_figure, data_path, check_file, COLORS


def _find_file(name):
    """Check in project root first, then data/ subdirectory."""
    path = data_path(name)
    if os.path.exists(path):
        return path
    alt = data_path(f'data/{name}')
    if os.path.exists(alt):
        return alt
    return path  # Will fail at check_file


def plot_pr_curve():
    """Generate Precision-Recall curves for Stage 2 predictions."""
    apply_style()

    scores_file = _find_file('anomaly_scores_stage2.npy')
    labels_file = _find_file('labels.npy')
    counts_file = _find_file('node_counts.json')

    if not check_file(scores_file, "Run stage2.py to generate anomaly_scores_stage2.npy"):
        return None
    if not check_file(labels_file, "Run generate_labels.py to generate labels.npy"):
        return None
    if not check_file(counts_file, "Run data_preprocessing.py to generate node_counts.json"):
        return None

    # Load data
    all_scores = np.load(scores_file)
    labels_data = np.load(labels_file)
    with open(counts_file, 'r') as f:
        counts = json.load(f)

    N_u = counts['users']
    N_p = counts['products']

    indices = labels_data[:, 0].astype(int)
    y_true  = labels_data[:, 1].astype(int)
    y_scores = all_scores[indices]

    # Masks for entity types
    user_mask = indices < N_u
    prod_mask = (indices >= N_u) & (indices < N_u + N_p)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Overall PR Curve
    prec_all, rec_all, _ = precision_recall_curve(y_true, y_scores)
    ap_all = average_precision_score(y_true, y_scores)
    ax.plot(rec_all, prec_all, color=COLORS['text'], linewidth=2.5,
            label=f'Overall  (AP = {ap_all:.4f})', zorder=3)

    # User PR Curve
    if user_mask.sum() > 0 and len(np.unique(y_true[user_mask])) > 1:
        prec_u, rec_u, _ = precision_recall_curve(y_true[user_mask], y_scores[user_mask])
        ap_u = average_precision_score(y_true[user_mask], y_scores[user_mask])
        ax.plot(rec_u, prec_u, color=COLORS['user'], linewidth=1.8,
                linestyle='--', label=f'Users    (AP = {ap_u:.4f})')

    # Product PR Curve
    if prod_mask.sum() > 0 and len(np.unique(y_true[prod_mask])) > 1:
        prec_p, rec_p, _ = precision_recall_curve(y_true[prod_mask], y_scores[prod_mask])
        ap_p = average_precision_score(y_true[prod_mask], y_scores[prod_mask])
        ax.plot(rec_p, prec_p, color=COLORS['product'], linewidth=1.8,
                linestyle='-.', label=f'Products (AP = {ap_p:.4f})')

    # Random baseline
    prevalence = y_true.mean()
    ax.axhline(y=prevalence, color=COLORS['grid'], linestyle=':', linewidth=1.2,
               label=f'Random Baseline ({prevalence:.4f})', alpha=0.7)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_title('Stage 2: Precision-Recall Curve\n'
                 'Semi-Supervised GAT Anomaly Detection', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)

    save_figure(fig, 'plot03_precision_recall_curve.png')
    print("  ✓ Plot 3: Precision-Recall Curve generated.")
    return fig


if __name__ == '__main__':
    plot_pr_curve()
