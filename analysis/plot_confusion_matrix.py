"""
Plot 5: Confusion Matrix Heatmap (Stage 2 — GAT)
=================================================
Standard 2×2 confusion matrix with counts and percentages.
Uses the same type-specific thresholding as performance_evaluation.py.

Data Source: anomaly_scores_stage2.npy, labels.npy, node_counts.json
"""

import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix

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


def plot_confusion_matrix():
    """Generate confusion matrix heatmap replicating performance_evaluation.py logic."""
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

    # Apply type-specific thresholding (mirrors performance_evaluation.py exactly)
    y_pred = np.zeros_like(y_scores, dtype=int)

    user_mask = indices < N_u
    prod_mask = (indices >= N_u) & (indices < N_u + N_p)

    if user_mask.sum() > 0:
        user_thresh = np.percentile(y_scores[user_mask], 82)
        y_pred[user_mask] = (y_scores[user_mask] >= user_thresh).astype(int)

    if prod_mask.sum() > 0:
        y_pred[prod_mask] = (y_scores[prod_mask] >= 0.5).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()

    # Percentages
    cm_pct = cm / total * 100

    # Labels
    labels_text = np.array([
        [f'TN\n{cm[0,0]:,}\n({cm_pct[0,0]:.1f}%)',
         f'FP\n{cm[0,1]:,}\n({cm_pct[0,1]:.1f}%)'],
        [f'FN\n{cm[1,0]:,}\n({cm_pct[1,0]:.1f}%)',
         f'TP\n{cm[1,1]:,}\n({cm_pct[1,1]:.1f}%)']
    ])

    # Custom colormap: white → blue for the heatmap intensity
    cmap = mcolors.LinearSegmentedColormap.from_list('cm_cmap', ['#FFFFFF', '#4C72B0', '#2C4270'])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(cm, cmap=cmap, aspect='equal')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > total * 0.3 else COLORS['text']
            ax.text(j, i, labels_text[i, j], ha='center', va='center',
                    fontsize=14, fontweight='bold', color=text_color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nNormal', 'Predicted\nAnomaly'], fontsize=11)
    ax.set_yticklabels(['Actual\nNormal', 'Actual\nAnomaly'], fontsize=11)

    ax.set_title('Stage 2: Confusion Matrix\n'
                 'Type-Specific Thresholding', fontweight='bold', fontsize=13)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label('Count', fontsize=10)

    # Summary stats below the plot
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    summary = f'Accuracy: {accuracy:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}'
    fig.text(0.5, -0.02, summary, ha='center', va='top', fontsize=10,
             fontweight='bold', color=COLORS['text'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor=COLORS['grid']))

    save_figure(fig, 'plot05_confusion_matrix.png')
    print("  ✓ Plot 5: Confusion Matrix Heatmap generated.")
    return fig


if __name__ == '__main__':
    plot_confusion_matrix()
