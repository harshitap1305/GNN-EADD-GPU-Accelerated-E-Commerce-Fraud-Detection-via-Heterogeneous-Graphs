"""
Plot 7: Baseline Comparison (Grouped Bar Chart)
===============================================
Compares GNN-EADD vs. HeteroDOMINANT vs. GraphSAGE AE
on all key metrics using the same evaluation labels.

Data Source: anomaly_scores_stage2.npy, sage_anomalies.npy, 
            dominant_anomalies.npy, labels.npy, node_counts.json
"""

import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

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


def _compute_metrics(y_true, y_scores, threshold_pct=82):
    """Compute standard metrics using percentile-based thresholding."""
    metrics = {}

    try:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_scores)
    except ValueError:
        metrics['AUC-ROC'] = 0.0

    try:
        metrics['AUC-PR'] = average_precision_score(y_true, y_scores)
    except ValueError:
        metrics['AUC-PR'] = 0.0

    threshold = np.percentile(y_scores, threshold_pct)
    y_pred = (y_scores >= threshold).astype(int)

    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall']    = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1-Score']  = f1_score(y_true, y_pred, zero_division=0)

    return metrics


def plot_baseline_comparison():
    """Generate baseline comparison grouped bar chart."""
    apply_style()

    # Required files
    gnn_file = _find_file('anomaly_scores_stage2.npy')
    sage_file = _find_file('sage_anomalies.npy')
    dom_file = _find_file('dominant_anomalies.npy')
    labels_file = _find_file('labels.npy')

    required = [
        (gnn_file, "Run stage2.py"),
        (sage_file, "Run baselines/sage_anomaly.py"),
        (dom_file, "Run baselines/dominant_anomaly.py"),
        (labels_file, "Run generate_labels.py"),
    ]

    for fpath, msg in required:
        if not check_file(fpath, msg):
            return None

    # Load
    gnn_scores  = np.load(gnn_file)
    sage_scores = np.load(sage_file)
    dom_scores  = np.load(dom_file)
    labels_data = np.load(labels_file)

    indices = labels_data[:, 0].astype(int)
    y_true  = labels_data[:, 1].astype(int)

    # Extract scores at labeled indices
    y_gnn  = gnn_scores[indices]
    y_sage = sage_scores[indices]
    y_dom  = dom_scores[indices]

    # Compute metrics for each method
    metrics_gnn  = _compute_metrics(y_true, y_gnn)
    metrics_sage = _compute_metrics(y_true, y_sage)
    metrics_dom  = _compute_metrics(y_true, y_dom)

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

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.set_title('Baseline Comparison\n'
                 'GNN-EADD vs. HeteroDOMINANT vs. GraphSAGE AE', fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.5)
    ax.xaxis.grid(False)

    save_figure(fig, 'plot07_baseline_comparison.png')
    print("  ✓ Plot 7: Baseline Comparison generated.")
    return fig


if __name__ == '__main__':
    plot_baseline_comparison()
