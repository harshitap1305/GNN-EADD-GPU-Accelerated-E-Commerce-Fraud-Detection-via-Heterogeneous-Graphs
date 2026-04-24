"""
Plot 4: Anomaly Score Density Histogram (Stage 2 — GAT)
=======================================================
Two overlapping, semi-transparent histograms showing the distribution
of predicted probabilities for Normal vs. Anomaly nodes.

Data Source: anomaly_scores_stage2.npy, labels.npy, node_counts.json
"""

import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt

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


def plot_score_distribution():
    """Generate anomaly score density histograms."""
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

    all_scores = np.load(scores_file)
    labels_data = np.load(labels_file)
    with open(counts_file, 'r') as f:
        counts = json.load(f)

    N_u = counts['users']

    indices = labels_data[:, 0].astype(int)
    y_true  = labels_data[:, 1].astype(int)
    y_scores = all_scores[indices]

    scores_normal  = y_scores[y_true == 0]
    scores_anomaly = y_scores[y_true == 1]

    # ──────── Main Histogram ────────
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bins = np.linspace(0, 1, 60)

    ax.hist(scores_normal, bins=bins, alpha=0.55, color=COLORS['normal'],
            label=f'Normal (n={len(scores_normal):,})', density=True,
            edgecolor='white', linewidth=0.3)
    ax.hist(scores_anomaly, bins=bins, alpha=0.65, color=COLORS['anomaly'],
            label=f'Anomaly (n={len(scores_anomaly):,})', density=True,
            edgecolor='white', linewidth=0.3)

    # Add decision threshold lines
    # User threshold: 82nd percentile of user scores
    user_mask = indices < N_u
    if user_mask.sum() > 0:
        user_scores = y_scores[user_mask]
        user_thresh = np.percentile(user_scores, 82)
        ax.axvline(x=user_thresh, color=COLORS['user'], linestyle='--', linewidth=1.8,
                   alpha=0.85, label=f'User Threshold ({user_thresh:.3f})')

    # Product threshold: 0.5
    ax.axvline(x=0.5, color=COLORS['product'], linestyle='--', linewidth=1.8,
               alpha=0.85, label='Product Threshold (0.500)')

    ax.set_xlabel('Predicted Anomaly Probability')
    ax.set_ylabel('Density')
    ax.set_xlim(-0.02, 1.02)
    ax.set_title('Stage 2: Anomaly Score Distribution\n'
                 'Normal vs. Anomaly Score Separation', fontweight='bold')
    ax.legend(loc='upper center', framealpha=0.9)

    # Add text annotation for separation quality
    median_normal  = np.median(scores_normal)
    median_anomaly = np.median(scores_anomaly)
    separation = median_anomaly - median_normal

    ax.annotate(f'Median separation: {separation:.3f}',
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=COLORS['grid'], alpha=0.9))

    save_figure(fig, 'plot04_score_distribution.png')
    print("  ✓ Plot 4: Anomaly Score Density Histogram generated.")
    return fig


if __name__ == '__main__':
    plot_score_distribution()
