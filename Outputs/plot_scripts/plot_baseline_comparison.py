"""
Plot 7: Baseline Comparison (Grouped Bar Chart)
===============================================
Compares GNN-EADD vs. HeteroDOMINANT vs. GraphSAGE AE
on all key metrics using pre-computed results from JSON files.

Data Source: {dataset}/baseline_performance_metrics.json,
            {dataset}/performance_evaluation.json
"""

import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import (apply_style, save_figure, load_dataset_json,
                         COLORS, DEFAULT_DATASET, DATASET_DISPLAY_NAMES)


def plot_baseline_comparison(dataset=None):
    """Generate baseline comparison grouped bar chart from pre-computed JSON metrics."""
    apply_style()

    if dataset is None:
        dataset = DEFAULT_DATASET

    # Load baseline metrics
    baseline = load_dataset_json(dataset, 'baseline_performance_metrics.json')
    if baseline is None:
        return None

    # Load GNN-EADD performance metrics
    perf = load_dataset_json(dataset, 'performance_evaluation.json')
    if perf is None:
        return None

    # Extract GNN-EADD global metrics
    overall_metrics = perf['overall_detection_performance']['metrics']
    metrics_gnn = {
        'AUC-ROC':   overall_metrics['auc_roc'],
        'AUC-PR':    overall_metrics['auc_pr'],
        'Precision': overall_metrics['global_precision'],
        'Recall':    overall_metrics['global_recall'],
        'F1-Score':  overall_metrics['global_f1_score'],
    }

    # Extract DOMINANT Global row
    dom_rows = baseline['dominant_baseline_performance']['rows']
    dom_global = next((r for r in dom_rows if r['entity'] == 'Global'), dom_rows[-1])
    metrics_dom = {
        'AUC-ROC':   dom_global['auc_roc'],
        'AUC-PR':    dom_global['auc_pr'],
        'Precision': dom_global['prec'],
        'Recall':    dom_global['rec'],
        'F1-Score':  dom_global['f1'],
    }

    # Extract GraphSAGE Global row
    sage_rows = baseline['graphsage_baseline_performance']['rows']
    sage_global = next((r for r in sage_rows if r['entity'] == 'Global'), sage_rows[-1])
    metrics_sage = {
        'AUC-ROC':   sage_global['auc_roc'],
        'AUC-PR':    sage_global['auc_pr'],
        'Precision': sage_global['prec'],
        'Recall':    sage_global['rec'],
        'F1-Score':  sage_global['f1'],
    }

    # Build chart
    metric_names = list(metrics_gnn.keys())
    n_metrics = len(metric_names)

    vals_gnn  = [metrics_gnn[m] for m in metric_names]
    vals_sage = [metrics_sage[m] for m in metric_names]
    vals_dom  = [metrics_dom[m] for m in metric_names]

    x = np.arange(n_metrics)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - bar_width, vals_gnn, bar_width, label='GNN-EADD (Ours)',
                   color=COLORS['gnn_eadd'], edgecolor='white', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x, vals_dom, bar_width, label='HeteroDOMINANT',
                   color=COLORS['dominant'], edgecolor='white', linewidth=0.5, zorder=3)
    bars3 = ax.bar(x + bar_width, vals_sage, bar_width, label='GraphSAGE AE',
                   color=COLORS['graphsage'], edgecolor='white', linewidth=0.5, zorder=3)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8,
                        fontweight='bold', color=COLORS['text'])

    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.set_title(f'Baseline Comparison — {display_name}\n'
                 'GNN-EADD vs. HeteroDOMINANT vs. GraphSAGE AE', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.5)
    ax.xaxis.grid(False)

    save_figure(fig, f'plot07_baseline_comparison_{dataset}.png')
    print(f"  ✓ Plot 7: Baseline Comparison generated for {display_name}.")
    return fig


if __name__ == '__main__':
    plot_baseline_comparison()
