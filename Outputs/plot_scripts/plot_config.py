"""
GNN-EADD Visualization Configuration
=====================================
Shared styling, color palettes, and helper functions used by all plot scripts.
Ensures publication-quality consistency across every figure.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
# All paths are relative to the PROJECT ROOT (parent of analysis/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR     = PROJECT_ROOT              # Where .npy and .json files live after pipeline run
FIGURES_DIR  = os.path.join(os.path.dirname(__file__), 'figures')
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')

os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PER-DATASET DIRECTORIES
# ─────────────────────────────────────────────
# Each dataset has a folder with consistently-named JSON outputs:
#   stage1.json, stage2.json, preprocessing.json,
#   performance_evaluation.json, baseline_performance_metrics.json,
#   generate_labels.json, label_data.json, dominant_anomaly.json, sage_anomaly.json
DATASETS = {
    'Electronics':   os.path.join(PROJECT_ROOT, 'Electronics'),
    'Cell_Phones':   os.path.join(PROJECT_ROOT, 'Cell_Phones'),
    'Arts_Crafts':   os.path.join(PROJECT_ROOT, 'Arts_Crafts'),
    'Luxury_Beauty': os.path.join(PROJECT_ROOT, 'Luxury_Beauty'),
}

# Display names (for plot labels)
DATASET_DISPLAY_NAMES = {
    'Electronics':   'Electronics',
    'Cell_Phones':   'Cell Phones',
    'Arts_Crafts':   'Arts & Crafts',
    'Luxury_Beauty': 'Luxury Beauty',
}

# Default dataset for single-dataset plots
DEFAULT_DATASET = 'Luxury_Beauty'


def dataset_dir(dataset_key):
    """Return the directory path for a dataset key."""
    return DATASETS[dataset_key]


def dataset_json(dataset_key, json_filename):
    """Return the full path to a JSON file within a dataset directory."""
    return os.path.join(DATASETS[dataset_key], json_filename)


def load_dataset_json(dataset_key, json_filename):
    """Load and return a parsed JSON file from a dataset directory."""
    path = dataset_json(dataset_key, json_filename)
    if not os.path.exists(path):
        print(f"  [SKIP] Missing: {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def get_available_datasets():
    """Return list of dataset keys that have output directories."""
    return [k for k, v in DATASETS.items() if os.path.isdir(v)]

# ─────────────────────────────────────────────
# COLOR PALETTE  (Publication-friendly, colorblind-safe)
# ─────────────────────────────────────────────
COLORS = {
    # Node types
    'user':      '#4C72B0',   # Steel Blue
    'product':   '#DD8452',   # Warm Orange
    'seller':    '#55A868',   # Sage Green

    # Anomaly labels
    'normal':    '#4C72B0',   # Steel Blue
    'anomaly':   '#C44E52',   # Muted Red

    # Pipeline comparison
    'gnn_eadd':  '#4C72B0',   # Steel Blue
    'dominant':  '#DD8452',   # Warm Orange
    'graphsage': '#55A868',   # Sage Green

    # Misc
    'margin':    '#8172B3',   # Soft Purple (margin shading)
    'threshold': '#CCB974',   # Gold (threshold lines)
    'grid':      '#E0E0E0',   # Light gray
    'text':      '#333333',   # Near-black
    'bg':        '#FAFAFA',   # Off-white background
}

# Dataset-specific colors (for cross-dataset plots)
DATASET_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
DATASET_NAMES  = [
    'Electronics',
    'Cell Phones',
    'Arts & Crafts',
    'Luxury Beauty',
]

# ─────────────────────────────────────────────
# MATPLOTLIB GLOBAL STYLE
# ─────────────────────────────────────────────
def apply_style():
    """Apply publication-quality matplotlib style globally."""
    plt.rcParams.update({
        # Font
        'font.family':        'sans-serif',
        'font.sans-serif':    ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size':          11,
        'axes.titlesize':     13,
        'axes.labelsize':     12,
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'legend.fontsize':    10,

        # Figure
        'figure.figsize':     (8, 5),
        'figure.dpi':         150,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.15,

        # Axes
        'axes.facecolor':     COLORS['bg'],
        'axes.edgecolor':     '#CCCCCC',
        'axes.linewidth':     0.8,
        'axes.grid':          True,
        'axes.grid.which':    'major',
        'grid.color':         COLORS['grid'],
        'grid.linewidth':     0.5,
        'grid.alpha':         0.7,

        # Lines
        'lines.linewidth':    2.0,
        'lines.markersize':   6,

        # Legend
        'legend.frameon':      True,
        'legend.framealpha':   0.9,
        'legend.edgecolor':    '#CCCCCC',

        # Layout
        'figure.constrained_layout.use': True,
    })


def save_figure(fig, filename, tight=True):
    """Save figure to the figures directory as high-res PNG."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight' if tight else None,
                facecolor='white', edgecolor='none')
    print(f"  [Saved] {path}")
    plt.close(fig)


def data_path(filename):
    """Return the full path to a data file in the project root."""
    return os.path.join(DATA_DIR, filename)


def template_path(filename):
    """Return the full path to a template file."""
    return os.path.join(TEMPLATES_DIR, filename)


def check_file(filepath, description=""):
    """Check if a required data file exists. Returns True/False with a message."""
    if os.path.exists(filepath):
        return True
    print(f"  [SKIP] Missing: {filepath}")
    if description:
        print(f"         {description}")
    return False
