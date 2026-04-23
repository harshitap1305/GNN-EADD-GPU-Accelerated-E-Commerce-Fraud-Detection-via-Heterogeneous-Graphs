# Stage 2: Semi-Supervised GAT & Custom CUDA Kernels

**Files:** `stage2.py` · `cuda_spmm/warp_gat_kernel.cu` · `cuda_spmm/spmm_kernel.cu` · `cuda_spmm/setup.py`

Stage 2 takes the **unsupervised embeddings from Stage 1** and fine-tunes them using a **Graph Attention Network (GAT)** with a small set of labeled anomaly nodes. The goal: assign every node in the 1.5M-node graph an **anomaly score between 0 and 1**.

---

## Inputs & Outputs

### Inputs

| File | Description |
|:---|:---|
| `Z_embeddings_stage1.npy` | `(N_total, 128)` float32 — pre-trained node embeddings from Stage 1 GAE |
| `data/labels.npy` | `(L, 2)` int — column 0: global node ID, column 1: binary label (1 = anomaly) |
| `data/node_counts.json` | `{ users, products, sellers, total }` |
| `data/epu_*.bin`, `eps_*.bin`, `euu_*.bin` | CSR topology binaries from preprocessing |

### Outputs

| File | Description |
|:---|:---|
| `anomaly_scores_stage2.npy` | `(N_total,)` float32 — sigmoid anomaly probability for every node |
| `Z_stage2.npy` | `(N_total, 128)` float32 — refined embeddings after GAT fine-tuning |
| `gat_stage2_best.pt` | PyTorch state dict of the best model checkpoint |

---

## Architecture: `TripleStreamGAT`

The model is a **2-layer heterogeneous GAT** followed by a **linear anomaly scoring head**:

```
   Z_stage1 (128-D)
        │
        ▼
┌─────────────────┐
│  GAT Layer 1    │  ← Type-specific attention across 3 edge streams
│  (128 → 128)    │
│  + ELU          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GAT Layer 2    │
│  (128 → 128)    │
│  + ELU          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Linear Head    │  ← nn.Linear(128, 1)
│  → Anomaly      │
│    Logit        │
└─────────────────┘
```

### How Each GAT Layer Works

Each `TypeSpecificGATLayer` processes **three edge streams** independently, each with its own learned weights:

| Stream | Edge Type | Weight Matrix | Attention Vector | What It Does |
|:---|:---|:---|:---|:---|
| **Purchase** | E_pu + E_pu_T | `W_pu` | `a_pu` | Messages between Users ↔ Products |
| **Selling** | E_ps + E_ps_T | `W_ps` | `a_ps` | Messages between Products ↔ Sellers |
| **Co-purchase** | E_uu | `W_uu` | `a_uu` | Messages between Users ↔ Users |

**Step-by-step per stream** (e.g. for Purchase edges):

1. **Linear projection:** Apply `W_pu` to both user and product features → `Wh_u`, `Wh_p` (both 128-D)
2. **Attention computation:** For each edge `(u, p)`, compute attention logit:
   - `e_up = LeakyReLU(a_pu[:128]·Wh_u + a_pu[128:]·Wh_p)`
3. **Softmax normalization:** Normalize attention weights across each node's neighbors: `α_up = softmax(e_up)` (with numerically stable max-subtraction)
4. **Weighted aggregation:** Each destination node's output = Σ (α_up · Wh_source) over all its neighbors
5. **Accumulate:** Add this stream's output into the node's accumulator

After all three streams, apply **ELU activation** (with scaling: user/product outputs are divided by 2 since they receive messages from 2 streams).

The final output is `[H_u_out | H_p_out | H_s_out]` concatenated back into a single `(N_total, 128)` tensor.

---

## Hybrid Training/Inference Strategy

The most interesting engineering decision in Stage 2 — **training and inference use different backends**:

| Mode | Backend | Why |
|:---|:---|:---|
| **Training** (`model.train()`) | Native PyTorch with `torch.autograd.checkpoint` | Needs gradient computation; CUDA kernel has no backward pass |
| **Inference** (`model.eval()`) | Custom `warp_gat` CUDA kernel | No gradients needed; kernel runs 3-5× faster |

**Training VRAM optimizations:**
- **85% DropEdge**: During training, 85% of edges are randomly dropped per chunk — this prevents over-smoothing AND saves massive VRAM
- **Micro-chunking**: Nodes are processed in chunks of 2,500 to avoid VRAM spikes from "super-nodes" with thousands of neighbors
- **Intra-loop garbage collection**: All intermediate tensors are explicitly deleted after each chunk
- **Gradient checkpointing**: `torch.utils.checkpoint.checkpoint` recomputes forward activations during backward instead of storing them, trading compute for VRAM

---

## Loss Function

The total loss is a weighted sum of supervised and unsupervised components:

```
L_total = L_supervised + λ · L_unsupervised     (λ = 0.5)
```

### Supervised Loss (L_sup)
Standard **Binary Cross-Entropy with Logits** on the labeled nodes only:
```python
L_sup = BCE(logits[labeled_idx], labels)
```
Only labeled nodes contribute gradients — unlabeled nodes still participate through message passing in the GAT layers.

### Unsupervised Loss (L_unsup) — Score Smoothness
Encourages **connected nodes to have similar anomaly scores** (Equation 17 from the paper):

1. Convert logits to bounded scores via sigmoid: `s_i = σ(logit_i)`
2. Sample 5,000 random edges from each of the 3 edge types
3. For each sampled edge `(i, j)`: compute `(s_i - s_j)²`
4. Average across all sampled edges

This enforces the intuition that if two nodes are connected (e.g. a user and a product they bought), their anomaly scores should be similar — anomaly signal propagates through the graph.

---

## Data Loading: CSR Sanitization

Stage 2's data loader (`load_and_sanitize_csr`) does something Stage 1 didn't need — it **strips cross-type self-loops** from the CSR files:

Stage 1 preprocessing added self-loops where every node connects to itself via its *global ID*. But in Stage 2's GAT, edges are processed per-relation with strict bipartite boundaries (e.g. E_pu should only connect Users → Products). A user self-loop in the E_pu CSR would point to a User ID, which is out-of-bounds for the Product node type.

The sanitizer:
1. Creates a validity mask: `(col_idx >= valid_min) & (col_idx <= valid_max)`
2. Filters `col_idx` to keep only edges pointing to the correct node type
3. **Reconstructs `row_ptr`** from scratch to reflect the new edge counts per row

---

## Label Splitting & Evaluation

Labels are split into **Train / Validation / Test** sets (60% / 20% / 20%) with a fixed seed for reproducibility.

**Metric: AUC-ROC** (Area Under the ROC Curve) — implemented manually without sklearn:
1. Sort all labeled nodes by their anomaly score (descending)
2. Compute cumulative true positive rate (TPR) and false positive rate (FPR)
3. Integrate using the trapezoidal rule

Validation AUC-ROC is checked every 10 epochs, and the best checkpoint is saved. Final evaluation runs on the held-out test set.

---

## Training Hyperparameters

| Parameter | Value |
|:---|:---|
| Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| λ (unsupervised weight) | 0.5 |
| Mixed Precision | FP16 autocast + GradScaler |
| DropEdge rate (training) | 85% |
| Chunk size | 2,500 nodes |
| Unsupervised edge samples | 5,000 per edge type |
| Train / Val / Test split | 60% / 20% / 20% |

---

## Custom CUDA Kernels (`cuda_spmm/`)

Two separate CUDA extensions are built via `setup.py`:

### 1. `warp_gat` — Warp-Level GAT Aggregation (Stage 2 Inference)

**File:** `warp_gat_kernel.cu` — Used during `model.eval()` for fast inference.

**How it works — 1 warp (32 threads) processes 1 destination node:**

```
For each destination node u:
  ┌─────────────────────────────────────────────────────┐
  │ Step 1: Compute dst_proj = Σ H_dst[u] * a_dst      │
  │         (distributed across 32 lanes, warp-reduced) │
  ├─────────────────────────────────────────────────────┤
  │ Step 2: For each neighbor v of u:                   │
  │         Compute src_proj = Σ H_src[v] * a_src      │
  │         e_uv = LeakyReLU(dst_proj + src_proj)      │
  │         Track max(e_uv) for numerical stability     │
  ├─────────────────────────────────────────────────────┤
  │ Step 3: Second pass over neighbors:                 │
  │         exp_sum += exp(e_uv - max_e)               │
  │         (warp-reduced for softmax denominator)      │
  ├─────────────────────────────────────────────────────┤
  │ Step 4: Feature aggregation:                        │
  │         For each feature dim d (striped by lane):   │
  │         out[u,d] = Σ_v α_uv * H_src[v,d]          │
  │         where α_uv = exp(e_uv-max)/exp_sum         │
  └─────────────────────────────────────────────────────┘
```

**Key CUDA primitives:**
- `__shfl_down_sync(0xffffffff, val, offset)` — warp shuffle for in-register reductions (no shared memory needed)
- `warpReduceMax` — finds maximum attention logit across all 32 lanes for numerically stable softmax
- `warpReduceSum` — sums exp values across all 32 lanes for softmax denominator

**Launch config:** 128 threads/block (4 warps/block), `ceil(N_dst × 32 / 128)` blocks.

### 2. `custom_spmm` — Warp-Level SpMM + Sparse Dot Product (Stage 1)

**File:** `spmm_kernel.cu` — Used in Stage 1 GAE training.

**Kernel 1: `csr_spmm_warp_kernel` (Encoder)**
- Performs Sparse Matrix × Dense Matrix multiplication: `out = A · X`
- 1 warp = 1 row of the sparse adjacency matrix
- Reads FP16 input features, accumulates in FP32 to prevent precision loss
- Applies pre-computed symmetric GCN weights (`D^{-1/2} A D^{-1/2}`) at the neighbor aggregation step
- Supports up to 1024-D features via chunked accumulation (32 chunks × 32 lanes)

**Kernel 2: `warp_sparse_dot_product_kernel` (Decoder)**
- Computes batched inner products `Z_src · Z_dst` for sampled edge pairs
- 1 warp = 1 edge: each warp computes the dot product for one (src, dst) pair
- Uses `__shfl_down_sync` warp shuffle for fast in-warp reduction — no global memory synchronization

**Python-side backprop (`SpMM_Autograd`):**
The custom SpMM has a hand-written PyTorch `autograd.Function` that implements **chunk-streaming backpropagation** — processing 5,000 rows at a time to prevent VRAM spikes during gradient computation on the 1.5M-node graph.

### Building the Extensions

```bash
cd cuda_spmm
python setup.py install
```

This compiles both extensions (`custom_spmm` and `warp_gat`) as importable Python modules via PyTorch's `CUDAExtension` build system. Requires CUDA Toolkit + a CUDA-capable GPU.

---

## End-to-End Flow

```
┌──────────────────────────────────────────────────────────────┐
│ Stage 1 (Unsupervised)                                      │
│ train_stage1_gae.py + custom_spmm CUDA kernel               │
│                                                              │
│ X_combined.memmap ──▶ 2-Layer GCN ──▶ Z_embeddings_stage1   │
│ (128-D features)      (SpMM encoder)    (128-D latent)       │
│                       (dot-product                           │
│                        decoder)                              │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2 (Semi-Supervised)                                    │
│ stage2.py + warp_gat CUDA kernel                            │
│                                                              │
│ Z_stage1 + labels.npy ──▶ 2-Layer GAT ──▶ anomaly_scores   │
│ (128-D embeddings)        (attention      (0 to 1 per node) │
│                            message                           │
│                            passing)                          │
└──────────────────────────────────────────────────────────────┘
```

**Stage 1** learns *what the graph looks like* (structure) without any labels.
**Stage 2** learns *what anomalies look like* by fine-tuning with a small labeled set, while the unsupervised loss ensures anomaly signals propagate through the graph topology.
