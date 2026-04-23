"""
Plot 8: Hardware Scalability Curve
==================================
Plots Total Graph Size (Nodes + Edges) vs. Total Training Time
across the 4 datasets to demonstrate efficient scaling.

Data Source: analysis/templates/cross_dataset_results.json
"""

import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import (apply_style, save_figure, template_path, check_file,
                         COLORS, DATASET_COLORS)


def plot_scalability(results_path=None):
    """Generate hardware scalability curve."""
    apply_style()

    if results_path is None:
        results_path = template_path('cross_dataset_results.json')

    if not check_file(results_path, "Fill in cross_dataset_results.json with timing data."):
        return None

    with open(results_path, 'r') as f:
        data = json.load(f)

    datasets = data['datasets']

    # Validate
    if all(d['total_training_time_seconds'] == 0.0 for d in datasets):
        print("  [SKIP] cross_dataset_results.json has no timing data — fill it after running all 4 datasets.")
        return None

    # Extract data
    names     = [d['name'] for d in datasets]
    graph_sizes = [d['total_nodes'] + d['total_edges'] for d in datasets]
    times_s1  = [d['stage1_training_time_seconds'] for d in datasets]
    times_s2  = [d['stage2_training_time_seconds'] for d in datasets]
    times_total = [d['total_training_time_seconds'] for d in datasets]

    # Sort by graph size for a clean curve
    order = np.argsort(graph_sizes)
    graph_sizes = [graph_sizes[i] for i in order]
    names       = [names[i] for i in order]
    times_s1    = [times_s1[i] for i in order]
    times_s2    = [times_s2[i] for i in order]
    times_total = [times_total[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Total time curve
    ax.plot(graph_sizes, times_total, color=COLORS['gnn_eadd'], linewidth=2.5,
            marker='o', markersize=9, label='Total (Stage 1 + Stage 2)', zorder=5)

    # Stage 1 and Stage 2 breakdown
    ax.plot(graph_sizes, times_s1, color=COLORS['user'], linewidth=1.8,
            marker='s', markersize=7, linestyle='--', label='Stage 1 (GAE)', zorder=4)
    ax.plot(graph_sizes, times_s2, color=COLORS['product'], linewidth=1.8,
            marker='^', markersize=7, linestyle='--', label='Stage 2 (GAT)', zorder=4)

    # Linear reference line (shows ideal scaling)
    if graph_sizes[-1] > 0 and times_total[-1] > 0:
        linear_rate = times_total[0] / graph_sizes[0] if graph_sizes[0] > 0 else 0
        linear_ref = [s * linear_rate for s in graph_sizes]
        ax.plot(graph_sizes, linear_ref, color=COLORS['grid'], linewidth=1.2,
                linestyle=':', label='Linear Reference', alpha=0.7)

    # Annotate dataset names next to points
    for i, name in enumerate(names):
        ax.annotate(name, (graph_sizes[i], times_total[i]),
                    textcoords='offset points', xytext=(10, 8),
                    fontsize=9, fontweight='bold', color=COLORS['text'],
                    arrowprops=dict(arrowstyle='-', color=COLORS['grid'], lw=0.8))

    ax.set_xlabel('Total Graph Size (Nodes + Edges)')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Hardware Scalability\n'
                 'Custom CUDA Kernels: SpMM + Warp-GAT', fontweight='bold', fontsize=13)
    ax.legend(loc='upper left', framealpha=0.9)

    # Format x-axis with K/M suffixes
    def format_size(x, _):
        if x >= 1e6:
            return f'{x / 1e6:.1f}M'
        elif x >= 1e3:
            return f'{x / 1e3:.0f}K'
        return str(int(x))

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_size))

    save_figure(fig, 'plot08_scalability_curve.png')
    print("  ✓ Plot 8: Hardware Scalability Curve generated.")
    return fig


if __name__ == '__main__':
    plot_scalability()
