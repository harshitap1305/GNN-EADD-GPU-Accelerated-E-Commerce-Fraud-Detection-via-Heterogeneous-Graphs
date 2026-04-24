"""
Plot 1: Contrastive Margin Curve (Stage 1 — GAE)
=================================================
Visualizes P(pos), P(neg), and the contrastive margin across training epochs.
Proves the model defeated mode collapse and the "Cosine Trap."

Data Source: {dataset}/stage1.json → epoch_logs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import (apply_style, save_figure, load_dataset_json,
                         COLORS, DEFAULT_DATASET, DATASET_DISPLAY_NAMES)


def plot_contrastive_margin(dataset=None):
    """Generate the contrastive margin curve from Stage 1 training logs."""
    apply_style()

    if dataset is None:
        dataset = DEFAULT_DATASET

    data = load_dataset_json(dataset, 'stage1.json')
    if data is None:
        return None

    entries = data['epoch_logs']
    epochs  = [e['epoch'] for e in entries]
    p_pos   = [e['p_pos'] for e in entries]
    p_neg   = [e['p_neg'] for e in entries]
    margins = [e['margin'] for e in entries]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Primary Y-axis: P(pos) and P(neg)
    line_pos, = ax1.plot(epochs, p_pos, color=COLORS['normal'], linewidth=2.2,
                         marker='o', markersize=4, label='P(positive)', zorder=3)
    line_neg, = ax1.plot(epochs, p_neg, color=COLORS['anomaly'], linewidth=2.2,
                         marker='s', markersize=4, label='P(negative)', zorder=3)

    # Shaded margin area between P(pos) and P(neg)
    ax1.fill_between(epochs, p_neg, p_pos, alpha=0.18, color=COLORS['margin'],
                     label='Contrastive Margin', zorder=1)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Sigmoid Probability')
    ax1.set_ylim(-0.05, 1.05)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))

    # Secondary Y-axis: Margin value
    ax2 = ax1.twinx()
    line_margin, = ax2.plot(epochs, margins, color=COLORS['margin'], linewidth=1.8,
                            linestyle='--', alpha=0.85, label='Margin Δ')
    ax2.set_ylabel('Margin (P_pos − P_neg)', color=COLORS['margin'])
    ax2.tick_params(axis='y', labelcolor=COLORS['margin'])
    ax2.set_ylim(-0.1, max(margins) * 1.3 if max(margins) > 0 else 1.0)

    # Combined legend
    lines = [line_pos, line_neg, line_margin]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', framealpha=0.9)

    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    ax1.set_title(f'Stage 1: Contrastive Margin Evolution — {display_name}\n'
                  'Temperature-Annealed GAE Training', fontweight='bold', fontsize=13)

    # Annotate final margin
    final_margin = margins[-1]
    ax1.annotate(f'Final Margin: +{final_margin:.3f}',
                 xy=(epochs[-1], p_pos[-1]),
                 xytext=(-100, -30), textcoords='offset points',
                 fontsize=10, fontweight='bold', color=COLORS['margin'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['margin'], lw=1.5))

    save_figure(fig, f'plot01_contrastive_margin_{dataset}.png')
    print(f"  ✓ Plot 1: Contrastive Margin Curve generated for {display_name}.")
    return fig


if __name__ == '__main__':
    plot_contrastive_margin()
