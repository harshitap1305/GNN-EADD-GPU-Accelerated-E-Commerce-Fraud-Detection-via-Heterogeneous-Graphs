"""
Plot 6: Cross-Dataset Performance Matrix (Grouped Bar Chart)
============================================================
Compares F1-Score, AUC-ROC, and AUC-PR across the 4 Amazon datasets.
Proves the GNN-EADD pipeline generalizes across sub-economies.

Data Source: Auto-aggregated from {dataset}/performance_evaluation.json
"""

import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import (apply_style, save_figure, load_dataset_json,
                         get_available_datasets, COLORS, DATASET_COLORS,
                         DATASET_DISPLAY_NAMES)


def plot_cross_dataset():
    """Generate cross-dataset performance grouped bar chart from per-dataset JSONs."""
    apply_style()

    available = get_available_datasets()
    if not available:
        print("  [SKIP] No dataset directories found.")
        return None

    datasets = []
    for ds_key in available:
        perf = load_dataset_json(ds_key, 'performance_evaluation.json')
        if perf is None:
            continue

        overall = perf.get('overall_detection_performance', {})
        metrics = overall.get('metrics', {})

        datasets.append({
            'name': DATASET_DISPLAY_NAMES.get(ds_key, ds_key),
            'auc_roc':   metrics.get('auc_roc', 0.0),
            'auc_pr':    metrics.get('auc_pr', 0.0),
            'f1_score':  metrics.get('global_f1_score', 0.0),
            'precision': metrics.get('global_precision', 0.0),
            'recall':    metrics.get('global_recall', 0.0),
        })

    if not datasets:
        print("  [SKIP] No performance_evaluation.json files found in any dataset.")
        return None

    # Metrics to plot
    metric_keys  = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall']
    metric_labels = ['AUC-ROC', 'AUC-PR', 'F1-Score', 'Precision', 'Recall']

    n_datasets = len(datasets)
    n_metrics  = len(metric_keys)

    x = np.arange(n_metrics)
    bar_width = 0.18
    offsets = np.arange(n_datasets) - (n_datasets - 1) / 2

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, ds in enumerate(datasets):
        values = [ds[k] for k in metric_keys]
        bars = ax.bar(x + offsets[i] * bar_width, values, bar_width,
                      label=ds['name'], color=DATASET_COLORS[i % len(DATASET_COLORS)],
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
