"""
Plot 5: Confusion Matrix Heatmap (Stage 2 — GAT)
=================================================
Standard 2×2 confusion matrix with counts and percentages.
Uses pre-computed confusion matrix from performance_evaluation.json.

Data Source: {dataset}/performance_evaluation.json
"""

import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import (apply_style, save_figure, load_dataset_json,
                         COLORS, DEFAULT_DATASET, DATASET_DISPLAY_NAMES)


def plot_confusion_matrix(dataset=None):
    """Generate confusion matrix heatmap from pre-computed performance evaluation."""
    apply_style()

    if dataset is None:
        dataset = DEFAULT_DATASET

    perf = load_dataset_json(dataset, 'performance_evaluation.json')
    if perf is None:
        return None

    overall = perf['overall_detection_performance']
    cm_data = overall['confusion_matrix']

    # Build confusion matrix from pre-computed values
    tn = cm_data['true_negatives_0_to_0']
    fp = cm_data['false_positives_0_to_1']
    fn = cm_data['false_negatives_1_to_0']
    tp = cm_data['true_positives_1_to_1']

    cm = np.array([[tn, fp], [fn, tp]])
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

    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    ax.set_title(f'Stage 2: Confusion Matrix — {display_name}\n'
                 'Type-Specific Thresholding', fontweight='bold', fontsize=13)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label('Count', fontsize=10)

    # Summary stats below the plot (from pre-computed metrics)
    metrics = overall['metrics']
    accuracy = (tn + tp) / total
    precision = metrics.get('global_precision', tp / (fp + tp) if (fp + tp) > 0 else 0)
    recall = metrics.get('global_recall', tp / (fn + tp) if (fn + tp) > 0 else 0)
    f1 = metrics.get('global_f1_score', 0)

    summary = f'Accuracy: {accuracy:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}'
    fig.text(0.5, -0.02, summary, ha='center', va='top', fontsize=10,
             fontweight='bold', color=COLORS['text'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor=COLORS['grid']))

    save_figure(fig, f'plot05_confusion_matrix_{dataset}.png')
    print(f"  ✓ Plot 5: Confusion Matrix Heatmap generated for {display_name}.")
    return fig


if __name__ == '__main__':
    plot_confusion_matrix()
