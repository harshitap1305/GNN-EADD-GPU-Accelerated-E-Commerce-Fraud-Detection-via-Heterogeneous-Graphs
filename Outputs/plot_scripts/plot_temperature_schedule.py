"""
Plot 10: Temperature Annealing Schedule (Stage 1 — GAE)
=======================================================
Visualizes the τ(epoch) temperature schedule used for cosine similarity
scaling in contrastive loss. No runtime data needed — purely mathematical.

Derived from: stage1.py → get_temperature()
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import apply_style, save_figure, COLORS


def get_temperature(epoch, warmup_epochs=20, max_tau=10.0, min_tau=2.0):
    """
    Exact replica of the temperature schedule from stage1.py.
    Warmup phase → Peak → Cosine decay to maintain gradient flow.
    """
    if epoch <= warmup_epochs:
        return min_tau + (max_tau - min_tau) * (epoch / warmup_epochs)
    progress = (epoch - warmup_epochs) / (200 - warmup_epochs)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return 3.0 + (max_tau - 3.0) * cosine


def plot_temperature_schedule():
    """Generate temperature annealing schedule visualization."""
    apply_style()

    epochs = np.arange(1, 201)
    temperatures = [get_temperature(e) for e in epochs]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Main temperature curve
    ax.plot(epochs, temperatures, color=COLORS['gnn_eadd'], linewidth=2.5, zorder=3)

    # Shade the three phases
    # Phase 1: Warmup (1–20)
    ax.axvspan(1, 20, alpha=0.08, color=COLORS['anomaly'], zorder=1)
    ax.text(10, max(temperatures) * 0.97, 'Warmup\nPhase',
            ha='center', va='top', fontsize=9, fontweight='bold',
            color=COLORS['anomaly'], alpha=0.7)

    # Phase 2: Peak + Early Cosine Decay (20–100)
    ax.axvspan(20, 100, alpha=0.06, color=COLORS['margin'], zorder=1)
    ax.text(60, max(temperatures) * 0.97, 'Peak →\nCosine Decay',
            ha='center', va='top', fontsize=9, fontweight='bold',
            color=COLORS['margin'], alpha=0.7)

    # Phase 3: Fine-tuning (100–200)
    ax.axvspan(100, 200, alpha=0.06, color=COLORS['seller'], zorder=1)
    ax.text(150, max(temperatures) * 0.97, 'Fine-Tuning\nRegime',
            ha='center', va='top', fontsize=9, fontweight='bold',
            color=COLORS['seller'], alpha=0.7)

    # Annotate key points
    peak_epoch = 20
    peak_tau = get_temperature(peak_epoch)
    ax.annotate(f'Peak τ = {peak_tau:.1f}',
                xy=(peak_epoch, peak_tau),
                xytext=(40, peak_tau + 0.5),
                fontsize=10, fontweight='bold', color=COLORS['gnn_eadd'],
                arrowprops=dict(arrowstyle='->', color=COLORS['gnn_eadd'], lw=1.5))

    final_tau = get_temperature(200)
    ax.annotate(f'Final τ = {final_tau:.1f}',
                xy=(200, final_tau),
                xytext=(175, final_tau + 2),
                fontsize=10, fontweight='bold', color=COLORS['gnn_eadd'],
                arrowprops=dict(arrowstyle='->', color=COLORS['gnn_eadd'], lw=1.5))

    # Horizontal reference lines
    ax.axhline(y=max(temperatures), color=COLORS['grid'], linestyle=':', alpha=0.5)
    ax.axhline(y=3.0, color=COLORS['grid'], linestyle=':', alpha=0.5)
    ax.text(202, 3.0, 'τ_min=3.0', fontsize=8, va='center', color=COLORS['text'], alpha=0.6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Temperature τ')
    ax.set_xlim(0, 205)
    ax.set_ylim(0, max(temperatures) + 1.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.set_title('Stage 1: Temperature Annealing Schedule\n'
                 'Contrastive Loss Sigmoid Scaling — τ(epoch)', fontweight='bold', fontsize=13)

    save_figure(fig, 'plot10_temperature_schedule.png')
    print("  ✓ Plot 10: Temperature Annealing Schedule generated.")
    return fig


if __name__ == '__main__':
    plot_temperature_schedule()
