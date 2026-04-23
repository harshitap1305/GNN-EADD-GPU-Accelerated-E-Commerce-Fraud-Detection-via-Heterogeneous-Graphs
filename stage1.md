# Stage 1: Unsupervised Graph Autoencoder (GAE) — Structural Node Embedding Generation

> **Goal:** Learn 128-dimensional structural node embeddings from a heterogeneous e-commerce graph (Users, Products, Sellers) using unsupervised link prediction. These embeddings capture topological neighborhood structure and serve as input features for Stage 2 (Supervised GAT Fine-tuning).

---

## Table of Contents

1. [Overview](#overview)
2. [Project Files](#project-files)
3. [Prerequisites](#prerequisites)
4. [Installation & Build Commands](#installation--build-commands)
5. [Execution Commands](#execution-commands)
6. [Input Data Format](#input-data-format)
7. [Output Data Format](#output-data-format)
8. [Architecture: Triple-Stream GCN Encoder](#architecture-triple-stream-gcn-encoder)
9. [Custom CUDA Kernel — `spmm_kernel.cu`](#custom-cuda-kernel--spmm_kernelcu)
10. [Link Prediction Decoder & Loss](#link-prediction-decoder--loss)
11. [Training Hyperparameters](#training-hyperparameters)
12. [Monitoring & Metrics](#monitoring--metrics)
13. [Results](#results)

---

## Overview

Stage 1 implements a **Graph Autoencoder (GAE)** that operates on a heterogeneous e-commerce graph containing three node types — **Users** (728,719), **Products** (756,077), and **Sellers** (55,679) — totalling **1,540,475 nodes**.

The model is trained via **unsupervised link prediction**: it learns to predict which edges exist in the graph, forcing each node's 128-D embedding to encode its neighborhood structure. The approach combines:

- A **2-layer heterogeneous GCN encoder** with relation-specific weight matrices
- A **custom warp-aligned CUDA SpMM kernel** for fast sparse message passing
- **Type-consistent negative sampling** for meaningful contrastive learning
- **Temperature annealing** to prevent sigmoid saturation during training
- **FP16 mixed-precision** training with FP32 gradient accumulation

The final output — `Z_embeddings_stage1.npy` — contains L2-normalized embeddings for every node, ready for Stage 2 supervised anomaly fine-tuning.

---

## Project Files

| File | Location | Description |
|:-----|:---------|:------------|
| `stage1.py` | `./stage1.py` | Main training script — model definition, training loop, edge sampling, and embedding export |
| `spmm_kernel.cu` | `./cuda_spmm/spmm_kernel.cu` | Custom CUDA kernel implementing warp-aligned CSR SpMM (encoder) and sparse dot product (decoder) |
| `setup.py` | `./setup.py` | Build script that compiles the CUDA extension into a Python-importable module (`custom_spmm`) using PyTorch's `CUDAExtension` |

### How They Connect

```
setup.py ──compiles──► spmm_kernel.cu ──produces──► custom_spmm (Python module)
                                                          │
stage1.py ──imports──► custom_spmm.forward()              │
          ──imports──► custom_spmm.decoder_dot()  ◄───────┘
```

1. **`setup.py`** uses `torch.utils.cpp_extension.CUDAExtension` to compile `spmm_kernel.cu` into a shared library
2. **`stage1.py`** imports the compiled module as `custom_spmm` and calls its two exported functions:
   - `custom_spmm.forward()` — Warp-aligned CSR SpMM for GCN message passing
   - `custom_spmm.decoder_dot()` — Warp-aligned sparse dot product for edge scoring

---

## Prerequisites

| Requirement | Minimum Version | Purpose |
|:------------|:----------------|:--------|
| Python | 3.8+ | Runtime |
| PyTorch | 2.0+ (with CUDA) | Deep learning framework |
| CUDA Toolkit | 11.7+ | Compile and run CUDA kernels |
| NVIDIA GPU | Compute Capability 7.0+ (e.g. RTX 2080, T4, A100) | GPU acceleration |
| NumPy | 1.21+ | Array operations and memmap loading |
| scikit-learn | 0.24+ | AUC-ROC metric computation |

Ensure that `nvcc` (the CUDA compiler) is on your `PATH` and that PyTorch's CUDA version matches your system CUDA toolkit.

---

## Installation & Build Commands

### Step 1: Set Up Python Environment (Recommended)

```bash
python -m venv gnn_env
source gnn_env/bin/activate
pip install torch numpy scikit-learn
```

### Step 2: Build the CUDA Extension

The `setup.py` script compiles `spmm_kernel.cu` (and `warp_gat_kernel.cu` for Stage 2) into Python-importable shared libraries.

```bash
cd cuda_spmm
pip install -e .
```

**Or equivalently:**

```bash
cd cuda_spmm
python setup.py build_ext --inplace
```

> **Note:** The `setup.py` is located at `./setup.py` (project root). It references `spmm_kernel.cu` which is inside `./cuda_spmm/`. If you run from the project root, use:
> ```bash
> pip install -e .
> ```

**Expected output on successful build:**

```
running build_ext
building 'custom_spmm' extension
...
```

If the build fails, verify:
- `nvcc --version` returns a valid CUDA version
- `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- Your PyTorch CUDA version matches your system CUDA toolkit

### Step 3: Verify the Extension Loads

```bash
python -c "import custom_spmm; print('CUDA SpMM extension loaded successfully')"
```

---

## Execution Commands

### Running Stage 1 Training

From the project root directory (where `stage1.py` and the data files reside):

```bash
python stage1.py
```

**What happens when you run this command:**

1. **Graph Loading** — Reads CSR binary files (`epu_*.bin`, `eps_*.bin`, `euu_*.bin`) and `node_counts.json`
2. **Normalization** — Computes GCN symmetric edge weights: `1 / √(deg(i) · deg(j))`
3. **Feature Loading** — Memory-maps `X_combined.memmap` (128-D FP16 features for all 1.54M nodes)
4. **Training** — Runs 200 epochs of unsupervised link prediction training
5. **Export** — Saves L2-normalized embeddings to `Z_embeddings_stage1.npy`

### Environment Variable (Optional)

The script sets this automatically, but you can also export it manually to help PyTorch manage GPU memory:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python stage1.py
```

### Expected Runtime

| GPU | Approximate Time |
|:----|:-----------------|
| NVIDIA RTX (Desktop) | ~290 seconds (~5 min) |
| NVIDIA T4 (Cloud) | ~400–500 seconds |
| NVIDIA A100 | ~150–200 seconds |

---

## Input Data Format

All input files must be present in the **same directory** as `stage1.py`.

### `node_counts.json`

```json
{"users": 728719, "products": 756077, "sellers": 55679, "total": 1540475}
```

Defines the number of nodes per type. Nodes are stored contiguously: Users first `[0, N_u)`, then Products `[N_u, N_u+N_p)`, then Sellers `[N_u+N_p, N_total)`.

### `X_combined.memmap`

- **Shape:** `(1,540,475 × 128)` — one 128-D feature vector per node
- **Dtype:** `float16`
- **Format:** NumPy memory-mapped file — loaded directly from disk without reading the entire file into RAM

### CSR Topology Binaries

Edge connectivity is stored in **Compressed Sparse Row (CSR)** format as raw binary files:

| File Pair | Edge Relation | Nodes |
|:----------|:--------------|:------|
| `epu_row_ptr.bin` / `epu_col_idx.bin` | User → Product | Purchases/interactions |
| `epu_T_row_ptr.bin` / `epu_T_col_idx.bin` | Product → User | Transpose of above |
| `eps_row_ptr.bin` / `eps_col_idx.bin` | Product → Seller | Sold-by relationship |
| `eps_T_row_ptr.bin` / `eps_T_col_idx.bin` | Seller → Product | Transpose of above |
| `euu_row_ptr.bin` / `euu_col_idx.bin` | User → User | Social/co-purchase links |

Each `_row_ptr.bin` is an `int32` array of length `(N_src + 1)`. Each `_col_idx.bin` is an `int32` array of length `(num_edges)`. Together they define the adjacency in CSR format.

---

## Output Data Format

### `Z_embeddings_stage1.npy`

| Property | Value |
|:---------|:------|
| Shape | `(1,540,475, 128)` |
| Dtype | `float32` |
| Normalization | L2-normalized (each row has unit norm) |
| File size | ~750 MB |

**Loading the output:**

```python
import numpy as np
Z = np.load('Z_embeddings_stage1.npy')
print(Z.shape)   # (1540475, 128)
print(np.linalg.norm(Z[0]))  # ≈ 1.0 (L2-normalized)
```

---

## Architecture: Triple-Stream GCN Encoder

The encoder is a **2-layer heterogeneous GCN** with **type-specific weight matrices** for each edge relation and **parallel CUDA streams** for concurrent message passing across node types.

### Data Flow Diagram

```
   X_combined (1.54M nodes × 128-D, float16)
          │
          ▼
  ┌───────────────────┐
  │    GCN Layer 1     │
  │   (128 → 128)      │
  │  3 parallel streams │
  │      + ReLU         │
  └─────────┬───────────┘
            │
            ▼
  ┌───────────────────┐
  │    GCN Layer 2     │
  │   (128 → 128)      │
  │  3 parallel streams │
  │    (no activation)  │
  └─────────┬───────────┘
            │
            ▼
     L2 Normalize
            │
            ▼
     Z (128-D embeddings)
```

### Per-Layer Message Passing (3 Parallel Streams)

Each GCN layer processes **three node types concurrently** using separate CUDA streams:

#### Stream 1 — User Nodes

1. **SpMM** (CUDA kernel): Aggregate product neighbor features via E_pu edges → `raw_pu`
2. **SpMM** (CUDA kernel): Aggregate user neighbor features via E_uu edges → `raw_uu`
3. **Linear transform + combine**: `H_user = W_pu(raw_pu) + W_uu(raw_uu) + W_self_u(X_user)`
4. **ReLU** activation (Layer 1 only)

#### Stream 2 — Product Nodes

1. **SpMM**: Aggregate user features via E_up edges (transpose of E_pu) → `raw_up`
2. **SpMM**: Aggregate seller features via E_ps edges → `raw_ps`
3. **Combine**: `H_product = W_up(raw_up) + W_ps(raw_ps) + W_self_p(X_product)`
4. **ReLU** activation (Layer 1 only)

#### Stream 3 — Seller Nodes

1. **SpMM**: Aggregate product features via E_sp edges (transpose of E_ps) → `raw_sp`
2. **Combine**: `H_seller = W_sp(raw_sp) + W_self_s(X_seller)`
3. **ReLU** activation (Layer 1 only)

All three streams **synchronize**, and outputs are concatenated: `[H_user | H_product | H_seller]`.

### Key Design Decisions

- **Relation-specific weights**: Each edge type has its own weight matrix (e.g. `W_pu ≠ W_uu`), allowing the model to learn different transformations for different relationship semantics
- **Explicit self-loops**: Separate `W_self_*` matrices preserve a node's own features during neighbor aggregation
- **FP16 input → FP32 accumulation**: The SpMM kernel reads features in half-precision but accumulates in float32 to prevent precision loss
- **No activation in Layer 2**: The raw embedding is used as the latent representation for maximum expressiveness

### GCN Symmetric Normalization

Edge weights are pre-computed using the standard GCN normalization formula:

```
weight(i → j) = 1 / √(deg(i) · deg(j))
```

This prevents high-degree nodes from dominating aggregation. Degrees are computed **per edge type**, and weights are calculated once at load time before training begins.

---

## Custom CUDA Kernel — `spmm_kernel.cu`

**File:** [`spmm_kernel.cu`](cuda_spmm/spmm_kernel.cu)

This file contains two CUDA kernels that form the computational backbone of Stage 1.

### Kernel 1: `csr_spmm_warp_kernel` (Encoder)

Performs **Sparse Matrix × Dense Matrix** multiplication (`out = A · X`) using CSR format. This is the core GCN message-passing operation.

**Thread mapping:** 1 warp (32 threads) processes exactly 1 graph node.

```
Warp Assignment:
  warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)
  lane_id = threadIdx.x % 32

Per-Warp Execution:
  1. Read row_ptr[node] → row_ptr[node+1] to find neighbor range
  2. For each neighbor:
     - Read pre-computed GCN edge weight
     - Each of 32 lanes handles a different feature dimension
       (lane 0 → dim 0, 32, 64, ...; lane 1 → dim 1, 33, 65, ...)
     - Accumulate weight × feature in FP32 registers
  3. Write aggregated result back to global memory
```

**Design choices:**
- **FP16 reads, FP32 accumulation** — `__half2float()` converts on the fly
- **Chunked dimensions** — Supports up to 1024-D features (32 chunks × 32 lanes)
- **Register-only** — No shared memory needed, avoiding bank conflicts
- **Launch config** — 256 threads/block = 8 warps/block

### Kernel 2: `warp_sparse_dot_product_kernel` (Decoder)

Computes batched dot products `Z[src] · Z[dst]` for sampled edge pairs during link prediction.

```
Per-Warp (1 warp = 1 edge pair):
  1. Each lane accumulates partial dot product over its assigned dimensions
  2. Warp reduction via __shfl_down_sync reduces 32 partial sums → 1 scalar
  3. Lane 0 writes the final score to global memory
```

### Python Binding

The kernels are exposed to Python via `pybind11`:

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &spmm_cuda, "Warp-Optimized CSR SpMM Forward (CUDA)");
    m.def("decoder_dot", &decoder_forward, "Warp-Aligned Sparse Dot Product (CUDA)");
}
```

### Autograd Backpropagation (`SpMM_Autograd`)

The CUDA kernel only implements a forward pass. Gradients flow through a custom `torch.autograd.Function` in `stage1.py` that uses **chunk-streaming** (5,000 rows at a time) to bound VRAM usage:

1. For each chunk of 5,000 source nodes:
   - Read row lengths from `row_ptr`
   - Expand `grad_output` rows via `repeat_interleave` for each node's neighbor count
   - Multiply by pre-computed edge weights
   - `scatter_add_` scaled gradients back to destination nodes
2. Cast final gradient from FP32 back to input dtype

This ensures the backward pass never materializes the full adjacency's worth of gradients at once, preventing OOM on large graphs.

---

## Link Prediction Decoder & Loss

The decoder is **parameter-free** — it computes cosine similarity (dot product of L2-normalized embeddings) between node pairs.

### Training Step (Each Epoch)

1. **Forward pass**: Run 2-layer GCN encoder in FP16 mixed precision → get `Z`
2. **L2 normalize**: `Z = normalize(Z)` — dot products become cosine similarities in [-1, 1]
3. **Sample positive edges**: Randomly draw real edges from the graph
4. **Sample negative edges**: Generate fake edges (type-consistent)
5. **Score all edges**: `score(i, j) = Z[i] · Z[j]`
6. **Compute loss**: `BCE(score × τ, label)` where label = 1 for real edges, 0 for fake
7. **Backprop + optimizer step** with gradient clipping (max_norm = 1.0)

### Type-Consistent Negative Sampling

Negatives are sampled **within the correct node type boundary** to force fine-grained learning:

| Positive Edge Type | Negative Destination Sampled From |
|:-------------------|:----------------------------------|
| User → Product | Random **Product** node |
| Product → Seller | Random **Seller** node |
| User → User | Random **User** node |

### Heterogeneous Sampling Budget

| Edge Type | Samples/Epoch | Rationale |
|:----------|:--------------|:----------|
| E_pu (User→Product) | 30,000 | Largest edge set, primary interaction signal |
| E_ps (Product→Seller) | 8,000 | Minority edge set — over-sampled relative to size |
| E_uu (User→User) | 12,000 | Dense cliques — constrained to prevent domination |

### Temperature Annealing

A temperature parameter `τ` scales logits before BCE to prevent sigmoid saturation:

```
loss = BCE(score × τ, label)
```

| Phase | Epochs | τ Range | Behavior |
|:------|:-------|:--------|:---------|
| Warmup | 1 → 20 | 2.0 → 10.0 | Gradually sharpens loss, keeps gradients flowing |
| Cosine Decay | 21 → 200 | 10.0 → 3.0 | Prevents over-confident predictions as model converges |

---

## Training Hyperparameters

| Parameter | Value |
|:----------|:------|
| Epochs | 200 |
| Optimizer | Adam (lr = 0.001) |
| LR Schedule | CosineAnnealingLR (T_max=200, eta_min=1e-5) |
| Gradient Clipping | max_norm = 1.0 |
| Mixed Precision | FP16 autocast (encoder) + FP32 (decoder & loss) |
| Temperature | Warmup 2→10 (epochs 1–20), cosine decay 10→3 (21–200) |
| Positive samples/epoch | 30K (EPU) + 8K (EPS) + 12K (EUU) = 50K total |
| Negative sampling | Type-consistent (within correct node type) |
| Embedding dimension | 128 |
| Output normalization | L2 (unit sphere) |
| VRAM management | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Backward chunk size | 5,000 rows per chunk |

---

## Monitoring & Metrics

Every 10 epochs, the training loop logs the following metrics:

| Metric | Description | Healthy Range |
|:-------|:------------|:--------------|
| **Loss** | BCE loss on sampled edges | Steadily decreasing |
| **P(pos)** | Average sigmoid score for real edges | Increasing → ~0.75+ |
| **P(neg)** | Average sigmoid score for fake edges | Decreasing → ~0.38 |
| **Margin** | P(pos) − P(neg) | Widening → ~0.35+ |
| **Edge AUC** | AUC-ROC on the sampled edges | Increasing → ~0.84+ |

A well-trained model shows clear **separation** between positive and negative edge scores, indicating that the embeddings encode genuine topological structure.

### Final Training Summary

At the end of 200 epochs, the script prints a summary block:

```
══════════════════════════════════════════════════════════════
  STAGE 1: TRAINING SUMMARY
══════════════════════════════════════════════════════════════
  Total Time:        290.91 seconds
  Final Loss:        0.4853
  Final Separation:  P(pos)=0.759 vs P(neg)=0.387
  Link Predict AUC:  0.8431
  Stage 2 Input:     Z_embeddings_stage1.npy
══════════════════════════════════════════════════════════════
```

---

## Results

### Training Output Screenshot

> *Add screenshot of terminal output showing the full training run here.*

<!-- 
To add your screenshot, replace the placeholder below with:
![Stage 1 Training Output](stage1_result.png)
-->

![Stage 1 Training Output](stage1_result.png)

---

### Epoch-wise Training Metrics

> The table below logs key metrics every 10 epochs. Fill in values from your training run.

| Epoch | Loss | P(pos) | P(neg) | Margin | Edge AUC |
|:-----:|:----:|:------:|:------:|:------:|:--------:|
| 10 | 1.0692 | 0.782 | 0.593 | +0.189 | 0.7269 |
| 20 | 1.3734 | 0.859 | 0.396 | +0.463 | 0.7452 |
| 30 | 1.1516 | 0.669 | 0.425 | +0.244 | 0.7295 |
| 40 | 1.1307 | 0.700 | 0.392 | +0.308 | 0.7487 |
| 50 | 0.8597 | 0.650 | 0.340 | +0.310 | 0.7626 |
| 60 | 0.8431 | 0.767 | 0.366 | +0.402 | 0.7847 |
| 70 | 0.7211 | 0.841 | 0.335 | +0.507 | 0.8134 |
| 80 | 0.7139 | 0.864 | 0.343 | +0.521 | 0.8130 |
| 90 | 0.6817 | 0.703 | 0.300 | +0.402 | 0.8142 |
| 100 | 0.6154 | 0.770 | 0.306 | +0.464 | 0.8259 |
| 110 | 0.5822 | 0.789 | 0.310 | +0.479 | 0.8316 |
| 120 | 0.5552 | 0.804 | 0.324 | +0.480 | 0.8353 |
| 130 | 0.5347 | 0.773 | 0.333 | +0.441 | 0.8359 |
| 140 | 0.5079 | 0.790 | 0.343 | +0.446 | 0.8425 |
| 150 | 0.5007 | 0.802 | 0.363 | +0.439 | 0.8420 |
| 160 | 0.4983 | 0.771 | 0.367 | +0.405 | 0.8391 |
| 170 | 0.4929 | 0.775 | 0.375 | +0.400 | 0.8398 |
| 180 | 0.4851 | 0.764 | 0.380 | +0.384 | 0.8437 |
| 190 | 0.4922 | 0.765 | 0.389 | +0.375 | 0.8360 |
| 200 | 0.4853 | 0.759 | 0.387 | +0.372 | 0.8431 |

### Final Summary

| Metric | Value |
|:-------|:------|
| Total Training Time | 290.91 seconds |
| Final Loss | 0.4853 |
| Final P(pos) | 0.759 |
| Final P(neg) | 0.387 |
| Final Margin | +0.372 |
| Final Edge AUC | 0.8431 |
| Nodes Embedded | 1,540,475 |
| Output File | `Z_embeddings_stage1.npy` |

---

### Cross-Dataset Results

> *Fill in this table as you evaluate Stage 1 on additional datasets.*

| Dataset | Nodes | Edges | Edge Types | Final Loss | Edge AUC | P(pos) | P(neg) | Margin | Training Time |
|:--------|------:|------:|-----------:|-----------:|---------:|:------:|:------:|:------:|:--------------|
| E-commerce (Primary) | 1,540,475 | — | 5 | 0.4853 | 0.8431 | 0.759 | 0.387 | +0.372 | 290.91s |
| *Dataset 2* | — | — | — | — | — | — | — | — | — |
| *Dataset 3* | — | — | — | — | — | — | — | — | — |
| *Dataset 4* | — | — | — | — | — | — | — | — | — |

> **Instructions:** Replace `Dataset 2`, `Dataset 3`, etc. with the actual dataset names and fill in the corresponding metrics after running Stage 1 on each dataset.

---

### Additional Screenshots

> *Add screenshots from additional dataset runs below.*

<!-- 
![Dataset 2 Results](screenshots/dataset2_result.png)
![Dataset 3 Results](screenshots/dataset3_result.png)
![Dataset 4 Results](screenshots/dataset4_result.png)
-->

---

*After Stage 1 completes, proceed to **Stage 2: Supervised GAT Fine-tuning** using `Z_embeddings_stage1.npy` as input.*
