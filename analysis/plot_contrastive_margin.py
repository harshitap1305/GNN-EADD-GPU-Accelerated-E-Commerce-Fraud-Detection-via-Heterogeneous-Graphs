"""
Plot 1: Contrastive Margin Curve (Stage 1 — GAE)
=================================================
Visualizes P(pos), P(neg), and the contrastive margin across training epochs.
Proves the model defeated mode collapse and the "Cosine Trap."

Data Source: analysis/templates/stage1_training_log.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import apply_style, save_figure, template_path, check_file, COLORS


def plot_contrastive_margin(log_path=None):
    """Generate the contrastive margin curve from Stage 1 training logs."""
    apply_style()

    if log_path is None:
        log_path = template_path('stage1_training_log.json')

    if not check_file(log_path, "Fill in stage1_training_log.json with your Stage 1 terminal output."):
        return None

    with open(log_path, 'r') as f:
        data = json.load(f)

    entries = data['epochs']
    epochs  = [e['epoch'] for e in entries]
    p_pos   = [e['p_pos'] for e in entries]
    p_neg   = [e['p_neg'] for e in entries]
    margins = [e['margin'] for e in entries]

    # Validate: skip if all zeros (template not filled)
    if all(v == 0.0 for v in p_pos):
        print("  [SKIP] stage1_training_log.json contains only zeros — fill it with real data first.")
        return None

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

    ax1.set_title('Stage 1: Contrastive Margin Evolution\n'
                  'Temperature-Annealed GAE Training', fontweight='bold', fontsize=13)

    # Annotate final margin
    final_margin = margins[-1]
    ax1.annotate(f'Final Margin: +{final_margin:.3f}',
                 xy=(epochs[-1], p_pos[-1]),
                 xytext=(-100, -30), textcoords='offset points',
                 fontsize=10, fontweight='bold', color=COLORS['margin'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['margin'], lw=1.5))

    save_figure(fig, 'plot01_contrastive_margin.png')
    print("  ✓ Plot 1: Contrastive Margin Curve generated.")
    return fig


if __name__ == '__main__':
    plot_contrastive_margin()
