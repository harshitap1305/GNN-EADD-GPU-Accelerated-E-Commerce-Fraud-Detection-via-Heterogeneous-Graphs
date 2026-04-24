# GNN-EADD: Quick Run Guide

> For the full project overview, architecture details, and documentation, see [README.md](README.md).

**Important:** Verify dataset file paths in each script before running. All scripts expect data files in the project root unless noted otherwise.

---

## Step 1: Data Preprocessing

Edit the dataset paths at the top of `data_preprocessing.py` to match your files:

```python
REVIEWS_FILE  = "5core_reviews.json"   # 5-core reviews file (.json or .json.gz)
METADATA_FILE = "meta_products.json"   # product metadata file (.json or .json.gz)
```

```bash
python3 data_preprocessing.py
```

Outputs: CSR binary files, feature memmaps (`X_combined.memmap`, `V_p/u/s_features.memmap`), `node_counts.json`, `node_id_mappings.json`.

---

## Step 2: Generate Anomaly Labels

```bash
python3 label_data.py
```

Outputs: `labelling_asin_meta.txt`, `labelling_asin_5_core.txt`.

---

## Step 3: Build the Stage 2 Label File

```bash
python3 generate_labels.py
```

Outputs: `labels.npy` (balanced, globally-indexed label array for Stage 2).

---

## Step 4: Build the CUDA Extensions

Both CUDA kernels (`custom_spmm` for Stage 1, `warp_gat` for Stage 2) are compiled in a single step:

```bash
cd cuda_spmm
python3 setup.py install
cd ..
```

---

## Step 5: Train Stage 1 — Graph Autoencoder (GAE)

```bash
python3 stage1.py
```

Outputs: `Z_embeddings_stage1.npy`.

---

## Step 6: Train Stage 2 — Graph Attention Network (GAT)

```bash
python3 stage2.py
```

Outputs: `anomaly_scores_stage2.npy`, `Z_stage2.npy`, `gat_stage2_best.pt`.

---

## Step 7: Evaluate Performance

```bash
python3 performance_evaluation.py
```

---

## Step 8: (Optional) Run Baselines

```bash
python3 baselines/dominant_anomaly.py
python3 baselines/sage_anomaly.py
python3 baselines/baseline_performance_metrics.py
```

---

## Step 9: (Optional) Generate Visualizations

```bash
cd analysis
python3 generate_all_plots.py
cd ..
```

Output plots are saved to `analysis/figures/`.
