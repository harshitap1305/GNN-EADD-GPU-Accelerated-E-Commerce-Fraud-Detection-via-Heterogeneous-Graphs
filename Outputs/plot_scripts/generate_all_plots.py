"""
GNN-EADD: Complete Analysis & Visualization Suite
==================================================
Master script that runs ALL visualization modules for ALL datasets.

Per-dataset plots (×4 datasets = 16 files):
  - Plot 01: Contrastive Margin Curve
  - Plot 05: Confusion Matrix Heatmap
  - Plot 07: Baseline Comparison
  - Plot 11: Training Loss Curves

Cross-dataset plots (×1 each = 3 files):
  - Plot 06: Cross-Dataset Performance Matrix
  - Plot 08: Hardware Scalability Curve
  - Plot 10: Temperature Annealing Schedule

Usage:
    cd Outputs
    python3 analysis/generate_all_plots.py

All figures are saved to: analysis/figures/
"""

import os
import sys
import time

# Ensure the analysis directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from plot_config import (apply_style, FIGURES_DIR, DATASETS,
                         DATASET_DISPLAY_NAMES, get_available_datasets)


def run_all():
    """Execute all visualization scripts for all datasets."""
    print("=" * 62)
    print("  GNN-EADD: Analysis & Visualization Suite")
    print("  Generating ALL plots for ALL datasets")
    print("=" * 62)
    print(f"  Output directory: {FIGURES_DIR}")
    print("=" * 62)

    t0 = time.time()
    results = {}
    available = get_available_datasets()

    # ─── Plot 10: Temperature Schedule (NO DATA REQUIRED — universal) ───
    print("\n─── Plot 10: Temperature Annealing Schedule ───")
    try:
        from plot_temperature_schedule import plot_temperature_schedule
        fig = plot_temperature_schedule()
        results['plot10_temperature_schedule'] = '✓'
    except Exception as e:
        print(f"  [ERROR] {e}")
        results['plot10_temperature_schedule'] = f'✗ {e}'

    # ─── Per-Dataset Plots ───
    for ds_key in available:
        display_name = DATASET_DISPLAY_NAMES.get(ds_key, ds_key)
        print(f"\n{'─' * 62}")
        print(f"  Dataset: {display_name} ({ds_key})")
        print(f"{'─' * 62}")

        # Plot 1: Contrastive Margin Curve
        print(f"\n  ─── Plot 1: Contrastive Margin Curve ───")
        try:
            from plot_contrastive_margin import plot_contrastive_margin
            fig = plot_contrastive_margin(dataset=ds_key)
            results[f'plot01_contrastive_margin_{ds_key}'] = '✓' if fig else '○ Skipped'
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[f'plot01_contrastive_margin_{ds_key}'] = f'✗ {e}'

        # Plot 5: Confusion Matrix
        print(f"\n  ─── Plot 5: Confusion Matrix Heatmap ───")
        try:
            from plot_confusion_matrix import plot_confusion_matrix
            fig = plot_confusion_matrix(dataset=ds_key)
            results[f'plot05_confusion_matrix_{ds_key}'] = '✓' if fig else '○ Skipped'
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[f'plot05_confusion_matrix_{ds_key}'] = f'✗ {e}'

        # Plot 7: Baseline Comparison
        print(f"\n  ─── Plot 7: Baseline Comparison ───")
        try:
            from plot_baseline_comparison import plot_baseline_comparison
            fig = plot_baseline_comparison(dataset=ds_key)
            results[f'plot07_baseline_comparison_{ds_key}'] = '✓' if fig else '○ Skipped'
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[f'plot07_baseline_comparison_{ds_key}'] = f'✗ {e}'

        # Plot 11: Training Loss Curves
        print(f"\n  ─── Plot 11: Training Loss Curves ───")
        try:
            from plot_training_curves import plot_training_curves
            fig = plot_training_curves(dataset=ds_key)
            results[f'plot11_training_curves_{ds_key}'] = '✓' if fig else '○ Skipped'
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[f'plot11_training_curves_{ds_key}'] = f'✗ {e}'

    # ─── Cross-Dataset Plots (run once, aggregate all datasets) ───
    print(f"\n{'─' * 62}")
    print(f"  Cross-Dataset Plots (all datasets combined)")
    print(f"{'─' * 62}")

    # Plot 6: Cross-Dataset Performance
    print("\n  ─── Plot 6: Cross-Dataset Performance Matrix ───")
    try:
        from plot_cross_dataset import plot_cross_dataset
        fig = plot_cross_dataset()
        results['plot06_cross_dataset_performance'] = '✓' if fig else '○ Skipped'
    except Exception as e:
        print(f"    [ERROR] {e}")
        results['plot06_cross_dataset_performance'] = f'✗ {e}'

    # Plot 8: Scalability Curve
    print("\n  ─── Plot 8: Hardware Scalability Curve ───")
    try:
        from plot_scalability import plot_scalability
        fig = plot_scalability()
        results['plot08_scalability_curve'] = '✓' if fig else '○ Skipped'
    except Exception as e:
        print(f"    [ERROR] {e}")
        results['plot08_scalability_curve'] = f'✗ {e}'

    # ─── Summary ───
    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print("  GENERATION SUMMARY")
    print("=" * 62)

    success = 0
    skipped = 0
    errors = 0
    for plot_name, status in sorted(results.items()):
        print(f"  {plot_name}: {status}")
        if status == '✓':
            success += 1
        elif status.startswith('○'):
            skipped += 1
        else:
            errors += 1

    print(f"\n  ✓ Generated: {success}  |  ○ Skipped: {skipped}  |  ✗ Errors: {errors}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print("=" * 62)


if __name__ == '__main__':
    run_all()
