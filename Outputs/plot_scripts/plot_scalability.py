"""
Plot 8: Hardware Scalability Curve
==================================
Plots Total Graph Size (Nodes + Edges) vs. Total Training Time
across the 4 datasets to demonstrate efficient scaling.

Data Source: Auto-aggregated from {dataset}/preprocessing.json + stage1.json + stage2.json
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


def plot_scalability():
    """Generate hardware scalability curve from per-dataset JSONs."""
    apply_style()

    available = get_available_datasets()
    if not available:
        print("  [SKIP] No dataset directories found.")
        return None

    names = []
    graph_sizes = []
    times_s1 = []
    times_s2 = []
    times_total = []

    for ds_key in available:
        preproc = load_dataset_json(ds_key, 'preprocessing.json')
        s1_data = load_dataset_json(ds_key, 'stage1.json')
        s2_data = load_dataset_json(ds_key, 'stage2.json')

        if preproc is None or s1_data is None or s2_data is None:
            continue

        # Get node counts from preprocessing_statistics or top-level
        stats = preproc.get('preprocessing_statistics', preproc)
        node_counts = stats.get('node_counts', {})

        # Try preprocessing_statistics node_counts keys first, fall back to pass_1 keys
        n_users = node_counts.get('V_u_users', 0)
        n_prods = node_counts.get('V_p_products', 0)
        n_sellers = node_counts.get('V_s_sellers', 0)
        if n_users == 0:
            # Fall back to pass_1 node_counts
            p1_counts = preproc.get('pass_1', {}).get('node_counts', {})
            n_users = p1_counts.get('users', 0)
            n_prods = p1_counts.get('products', 0)
            n_sellers = p1_counts.get('sellers', 0)

        total_nodes = n_users + n_prods + n_sellers

        # Get edge counts
        edge_counts = stats.get('edge_counts_including_gcn_self_loops', {})
        total_edges = sum(edge_counts.values())

        # Get training times
        s1_time = s1_data.get('training_summary', {}).get('total_time_seconds', 0.0)
        s2_time = s2_data.get('training_summary', {}).get('total_time_seconds', 0.0)

        display_name = DATASET_DISPLAY_NAMES.get(ds_key, ds_key)
        names.append(display_name)
        graph_sizes.append(total_nodes + total_edges)
        times_s1.append(s1_time)
        times_s2.append(s2_time)
        times_total.append(s1_time + s2_time)

    if not names:
        print("  [SKIP] No complete dataset results found (need preprocessing.json + stage1.json + stage2.json).")
        return None

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
