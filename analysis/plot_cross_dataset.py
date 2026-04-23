"""
Plot 6: Cross-Dataset Performance Matrix (Grouped Bar Chart)
============================================================
Compares F1-Score, AUC-ROC, and AUC-PR across the 4 Amazon datasets.
Proves the GNN-EADD pipeline generalizes across sub-economies.

Data Source: analysis/templates/cross_dataset_results.json
"""

import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import (apply_style, save_figure, template_path, check_file,
                         COLORS, DATASET_COLORS, DATASET_NAMES)


def plot_cross_dataset(results_path=None):
    """Generate cross-dataset performance grouped bar chart."""
    apply_style()

    if results_path is None:
        results_path = template_path('cross_dataset_results.json')

    if not check_file(results_path, "Fill in cross_dataset_results.json with metrics from all 4 datasets."):
        return None

    with open(results_path, 'r') as f:
        data = json.load(f)

    datasets = data['datasets']

    # Validate: skip if all zeros
    if all(d['auc_roc'] == 0.0 for d in datasets):
        print("  [SKIP] cross_dataset_results.json contains only zeros — fill with real data first.")
        return None

    # Metrics to plot
    metric_keys  = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall']
    metric_labels = ['AUC-ROC', 'AUC-PR', 'F1-Score', 'Precision', 'Recall']

    dataset_names = [d['name'] for d in datasets]
    n_datasets = len(datasets)
    n_metrics  = len(metric_keys)

    x = np.arange(n_metrics)
    bar_width = 0.18
    offsets = np.arange(n_datasets) - (n_datasets - 1) / 2

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, ds in enumerate(datasets):
        values = [ds[k] for k in metric_keys]
        bars = ax.bar(x + offsets[i] * bar_width, values, bar_width,
                      label=dataset_names[i], color=DATASET_COLORS[i % len(DATASET_COLORS)],
                      edgecolor='white', linewidth=0.5, zorder=3)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7.5,
                        fontweight='bold', color=COLORS['text'])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.set_title('Cross-Dataset Detection Performance\n'
                 'GNN-EADD Across 4 Amazon Sub-Economies', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)

    # Add grid only on y-axis
    ax.yaxis.grid(True, alpha=0.5)
    ax.xaxis.grid(False)

    save_figure(fig, 'plot06_cross_dataset_performance.png')
    print("  ✓ Plot 6: Cross-Dataset Performance Matrix generated.")
    return fig


if __name__ == '__main__':
    plot_cross_dataset()
