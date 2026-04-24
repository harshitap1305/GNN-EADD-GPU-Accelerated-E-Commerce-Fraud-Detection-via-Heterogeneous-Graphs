"""
Plot 11: Training Loss Curves (Stage 1 + Stage 2)
=================================================
Standard ML training convergence plots.
- Stage 1: Loss + Edge AUC across 200 epochs
- Stage 2: L_sup, L_unsup, L_total + Val AUC-ROC across 100 epochs

Data Source: {dataset}/stage1.json, {dataset}/stage2.json
"""

import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import (apply_style, save_figure, load_dataset_json,
                         COLORS, DEFAULT_DATASET, DATASET_DISPLAY_NAMES)


def plot_training_curves(dataset=None):
    """Generate training loss curves for Stage 1 and Stage 2."""
    apply_style()

    if dataset is None:
        dataset = DEFAULT_DATASET

    s1_data = load_dataset_json(dataset, 'stage1.json')
    s2_data = load_dataset_json(dataset, 'stage2.json')

    if s1_data is None and s2_data is None:
        print("  [SKIP] No training log files found.")
        return None

    display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ──────── Stage 1 ────────
    ax1 = axes[0]
    if s1_data is not None:
        entries = s1_data['epoch_logs']
        epochs = [e['epoch'] for e in entries]
        losses = [e['loss'] for e in entries]
        aucs   = [e['edge_auc'] for e in entries]

        color_loss = COLORS['anomaly']
        color_auc  = COLORS['gnn_eadd']

        line1, = ax1.plot(epochs, losses, color=color_loss, linewidth=2.0,
                          marker='o', markersize=3, label='BCE Loss')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax1_twin = ax1.twinx()
        line2, = ax1_twin.plot(epochs, aucs, color=color_auc, linewidth=2.0,
                               marker='s', markersize=3, linestyle='--', label='Edge AUC')
        ax1_twin.set_ylabel('Edge AUC', color=color_auc)
        ax1_twin.tick_params(axis='y', labelcolor=color_auc)
        ax1_twin.set_ylim(0.4, 1.02)

        lines = [line1, line2]
        ax1.legend(lines, [l.get_label() for l in lines], loc='center right')
    else:
        ax1.text(0.5, 0.5, 'Stage 1 data not available',
                 ha='center', va='center', transform=ax1.transAxes,
                 fontsize=11, color=COLORS['text'], alpha=0.6)

    ax1.set_xlabel('Epoch')
    ax1.set_title('Stage 1: GAE Training\n(Contrastive Loss + Edge Prediction)', fontweight='bold')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))

    # ──────── Stage 2 ────────
    ax2 = axes[1]
    if s2_data is not None:
        entries = s2_data['epoch_logs']
        epochs = [e['epoch'] for e in entries]
        l_sup   = [e['l_sup'] for e in entries]
        l_unsup = [e['l_unsup'] for e in entries]
        l_total = [e['l_total'] for e in entries]
        val_auc = [e['val_auc_roc'] for e in entries]

        ax2.plot(epochs, l_total, color=COLORS['text'], linewidth=2.2,
                 label='L_total', marker='o', markersize=3, zorder=3)
        ax2.plot(epochs, l_sup, color=COLORS['anomaly'], linewidth=1.5,
                 linestyle='--', label='L_supervised', marker='s', markersize=2.5)
        ax2.plot(epochs, l_unsup, color=COLORS['user'], linewidth=1.5,
                 linestyle='-.', label='L_unsupervised', marker='^', markersize=2.5)

        ax2.set_ylabel('Loss')

        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, val_auc, color=COLORS['seller'], linewidth=2.0,
                      linestyle=':', label='Val AUC-ROC', marker='D', markersize=3)
        ax2_twin.set_ylabel('Val AUC-ROC', color=COLORS['seller'])
        ax2_twin.tick_params(axis='y', labelcolor=COLORS['seller'])
        ax2_twin.set_ylim(0.4, 1.02)

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Stage 2 data not available',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=11, color=COLORS['text'], alpha=0.6)

    ax2.set_xlabel('Epoch')
    ax2.set_title('Stage 2: GAT Fine-Tuning\n(Hybrid Supervised + Unsupervised)', fontweight='bold')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))

    fig.suptitle(f'GNN-EADD Training Convergence — {display_name}',
                 fontweight='bold', fontsize=14, y=1.03)

    save_figure(fig, f'plot11_training_curves_{dataset}.png')
    print(f"  ✓ Plot 11: Training Loss Curves generated for {display_name}.")
    return fig


if __name__ == '__main__':
    plot_training_curves()
