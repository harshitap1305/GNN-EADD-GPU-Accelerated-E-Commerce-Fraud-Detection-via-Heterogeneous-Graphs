"""
Plot 2: UMAP 2D Embedding Projection (Stage 1 — GAE)
=====================================================
Reduces 128-D node embeddings to 2D using UMAP and creates:
  Plot A: Colored by node type (User / Product / Seller)
  Plot B: Colored by ground truth label (Normal / Anomaly)

Data Source: Z_embeddings_stage1.npy, node_counts.json, labels.npy
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from plot_config import apply_style, save_figure, data_path, check_file, COLORS

# Maximum number of nodes to plot (UMAP is O(n*log(n)) but still slow for millions)
MAX_SAMPLES = 50000


def _subsample_stratified(N_u, N_p, N_s, max_total=MAX_SAMPLES):
    """
    Stratified subsampling that preserves the proportional representation
    of all 3 node types, even for million-node graphs.
    """
    total = N_u + N_p + N_s
    if total <= max_total:
        return np.arange(total)

    ratio = max_total / total
    n_u = max(100, int(N_u * ratio))
    n_p = max(100, int(N_p * ratio))
    n_s = max(50,  int(N_s * ratio))

    # Adjust to not exceed max_total
    actual = n_u + n_p + n_s
    if actual > max_total:
        scale = max_total / actual
        n_u = int(n_u * scale)
        n_p = int(n_p * scale)
        n_s = int(n_s * scale)

    idx_u = np.random.choice(N_u, size=min(n_u, N_u), replace=False)
    idx_p = np.random.choice(N_p, size=min(n_p, N_p), replace=False) + N_u
    idx_s = np.random.choice(N_s, size=min(n_s, N_s), replace=False) + N_u + N_p

    return np.concatenate([idx_u, idx_p, idx_s])


def plot_embedding_projection():
    """Generate UMAP 2D projections of Stage 1 embeddings."""
    apply_style()

    # Check required files
    emb_file = data_path('Z_embeddings_stage1.npy')
    counts_file = data_path('node_counts.json')

    if not check_file(emb_file, "Run stage1.py to generate Z_embeddings_stage1.npy"):
        return None
    if not check_file(counts_file, "Run data_preprocessing.py to generate node_counts.json"):
        return None

    # Also check data/ subdirectory as fallback
    if not os.path.exists(counts_file):
        counts_file = data_path('data/node_counts.json')
        if not check_file(counts_file):
            return None

    # Load data
    Z = np.load(emb_file)
    with open(counts_file, 'r') as f:
        counts = json.load(f)

    N_u = counts['users']
    N_p = counts['products']
    N_s = counts['sellers']
    total = N_u + N_p + N_s

    print(f"  Loaded embeddings: {Z.shape}  |  U={N_u:,}  P={N_p:,}  S={N_s:,}")

    # Subsample for tractable UMAP
    np.random.seed(42)
    sample_idx = _subsample_stratified(N_u, N_p, N_s)
    Z_sample = Z[sample_idx]
    print(f"  Subsampled to {len(sample_idx):,} nodes for UMAP projection...")

    # Run UMAP
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                       metric='cosine', random_state=42, verbose=False)
    except ImportError:
        print("  [FALLBACK] umap-learn not installed. Falling back to t-SNE (slower).")
        print("  Install UMAP for better results: pip install umap-learn")
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)

    embedding_2d = reducer.fit_transform(Z_sample)

    # Assign node types
    node_types = np.empty(len(sample_idx), dtype=object)
    node_types[sample_idx < N_u] = 'User'
    node_types[(sample_idx >= N_u) & (sample_idx < N_u + N_p)] = 'Product'
    node_types[sample_idx >= N_u + N_p] = 'Seller'

    # ──────── PLOT A: Color by Node Type ────────
    import matplotlib.pyplot as plt

    fig_a, ax_a = plt.subplots(figsize=(8, 7))

    type_colors = {'User': COLORS['user'], 'Product': COLORS['product'], 'Seller': COLORS['seller']}
    for ntype in ['User', 'Product', 'Seller']:
        mask = node_types == ntype
        ax_a.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                     c=type_colors[ntype], label=f'{ntype} ({mask.sum():,})',
                     s=3, alpha=0.45, rasterized=True)

    ax_a.set_title('Stage 1 Embedding Space — Node Types\n'
                   '128-D → 2D UMAP Projection', fontweight='bold')
    ax_a.set_xlabel('UMAP-1')
    ax_a.set_ylabel('UMAP-2')
    ax_a.legend(markerscale=5, loc='best', framealpha=0.9)
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    save_figure(fig_a, 'plot02a_embedding_by_type.png')
    print("  ✓ Plot 2A: Embedding by Node Type generated.")

    # ──────── PLOT B: Color by Anomaly Label ────────
    labels_file = data_path('labels.npy')
    if not os.path.exists(labels_file):
        labels_file = data_path('data/labels.npy')

    if not check_file(labels_file, "Run generate_labels.py to generate labels.npy"):
        print("  [SKIP] Plot 2B (anomaly coloring) requires labels.npy")
        return fig_a

    labels_data = np.load(labels_file)
    anomaly_set = set(labels_data[labels_data[:, 1] == 1, 0].astype(int))

    is_anomaly = np.array([idx in anomaly_set for idx in sample_idx])

    fig_b, ax_b = plt.subplots(figsize=(8, 7))

    # Plot normal nodes first (background), then anomalies on top
    mask_n = ~is_anomaly
    mask_a = is_anomaly

    ax_b.scatter(embedding_2d[mask_n, 0], embedding_2d[mask_n, 1],
                 c=COLORS['normal'], label=f'Normal ({mask_n.sum():,})',
                 s=3, alpha=0.3, rasterized=True)
    ax_b.scatter(embedding_2d[mask_a, 0], embedding_2d[mask_a, 1],
                 c=COLORS['anomaly'], label=f'Anomaly ({mask_a.sum():,})',
                 s=8, alpha=0.8, rasterized=True, zorder=5)

    ax_b.set_title('Stage 1 Embedding Space — Anomaly Distribution\n'
                   '128-D → 2D UMAP Projection', fontweight='bold')
    ax_b.set_xlabel('UMAP-1')
    ax_b.set_ylabel('UMAP-2')
    ax_b.legend(markerscale=4, loc='best', framealpha=0.9)
    ax_b.set_xticks([])
    ax_b.set_yticks([])

    save_figure(fig_b, 'plot02b_embedding_by_anomaly.png')
    print("  ✓ Plot 2B: Embedding by Anomaly Label generated.")

    return fig_a, fig_b


if __name__ == '__main__':
    plot_embedding_projection()
