"""
GNN-EADD: Complete Analysis & Visualization Suite
==================================================
Master script that runs ALL visualization modules in sequence.
Each plot is independent — if a data file is missing, that plot is
skipped with a clear message, and execution continues.

Usage:
    cd Pop_project-main
    python analysis/generate_all_plots.py

All figures are saved to: analysis/figures/
"""

import os
import sys
import time

# Ensure the analysis directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from plot_config import apply_style, FIGURES_DIR


def run_all():
    """Execute all visualization scripts in order."""
    print("=" * 62)
    print("  GNN-EADD: Analysis & Visualization Suite")
    print("=" * 62)
    print(f"  Output directory: {FIGURES_DIR}")
    print("=" * 62)

    t0 = time.time()
    results = {}

    # ─── Plot 10: Temperature Schedule (NO DATA REQUIRED — runs first) ───
    print("\n─── Plot 10: Temperature Annealing Schedule ───")
    try:
        from plot_temperature_schedule import plot_temperature_schedule
        fig = plot_temperature_schedule()
        results['Plot 10'] = '✓' if fig else '○ Skipped'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 10'] = f'✗ Error: {e}'

    # ─── Plot 1: Contrastive Margin Curve ───
    print("\n─── Plot 1: Contrastive Margin Curve ───")
    try:
        from plot_contrastive_margin import plot_contrastive_margin
        fig = plot_contrastive_margin()
        results['Plot 01'] = '✓' if fig else '○ Skipped (need log data)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 01'] = f'✗ Error: {e}'

    # ─── Plot 2: UMAP Embedding Projection ───
    print("\n─── Plot 2: UMAP Embedding Projection ───")
    try:
        from plot_embedding_projection import plot_embedding_projection
        fig = plot_embedding_projection()
        results['Plot 02'] = '✓' if fig else '○ Skipped (need .npy files)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 02'] = f'✗ Error: {e}'

    # ─── Plot 3: Precision-Recall Curve ───
    print("\n─── Plot 3: Precision-Recall Curve ───")
    try:
        from plot_pr_curve import plot_pr_curve
        fig = plot_pr_curve()
        results['Plot 03'] = '✓' if fig else '○ Skipped (need .npy files)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 03'] = f'✗ Error: {e}'

    # ─── Plot 4: Score Density Histogram ───
    print("\n─── Plot 4: Anomaly Score Distribution ───")
    try:
        from plot_score_distribution import plot_score_distribution
        fig = plot_score_distribution()
        results['Plot 04'] = '✓' if fig else '○ Skipped (need .npy files)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 04'] = f'✗ Error: {e}'

    # ─── Plot 5: Confusion Matrix ───
    print("\n─── Plot 5: Confusion Matrix Heatmap ───")
    try:
        from plot_confusion_matrix import plot_confusion_matrix
        fig = plot_confusion_matrix()
        results['Plot 05'] = '✓' if fig else '○ Skipped (need .npy files)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 05'] = f'✗ Error: {e}'

    # ─── Plot 6: Cross-Dataset Performance ───
    print("\n─── Plot 6: Cross-Dataset Performance Matrix ───")
    try:
        from plot_cross_dataset import plot_cross_dataset
        fig = plot_cross_dataset()
        results['Plot 06'] = '✓' if fig else '○ Skipped (need cross_dataset_results.json)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 06'] = f'✗ Error: {e}'

    # ─── Plot 7: Baseline Comparison ───
    print("\n─── Plot 7: Baseline Comparison ───")
    try:
        from plot_baseline_comparison import plot_baseline_comparison
        fig = plot_baseline_comparison()
        results['Plot 07'] = '✓' if fig else '○ Skipped (need baseline .npy files)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 07'] = f'✗ Error: {e}'

    # ─── Plot 8: Scalability Curve ───
    print("\n─── Plot 8: Hardware Scalability Curve ───")
    try:
        from plot_scalability import plot_scalability
        fig = plot_scalability()
        results['Plot 08'] = '✓' if fig else '○ Skipped (need cross_dataset_results.json)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 08'] = f'✗ Error: {e}'

    # ─── Plot 9: ROC Curve ───
    print("\n─── Plot 9: ROC Curve ───")
    try:
        from plot_roc_curve import plot_roc_curve
        fig = plot_roc_curve()
        results['Plot 09'] = '✓' if fig else '○ Skipped (need .npy files)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 09'] = f'✗ Error: {e}'

    # ─── Plot 11: Training Loss Curves ───
    print("\n─── Plot 11: Training Loss Curves ───")
    try:
        from plot_training_curves import plot_training_curves
        fig = plot_training_curves()
        results['Plot 11'] = '✓' if fig else '○ Skipped (need log data)'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['Plot 11'] = f'✗ Error: {e}'

    # ─── Summary ───
    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print("  GENERATION SUMMARY")
    print("=" * 62)
    for plot_name, status in sorted(results.items()):
        print(f"  {plot_name}: {status}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("=" * 62)


if __name__ == '__main__':
    run_all()
