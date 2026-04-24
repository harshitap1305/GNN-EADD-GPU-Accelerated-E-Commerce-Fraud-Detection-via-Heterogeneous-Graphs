# GNN-EADD: GPU-Accelerated E-Commerce Fraud Detection via Heterogeneous Graphs

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Paper Reference](#research-paper-reference)
3. [Project Architecture](#project-architecture)
4. [Dataset: Challenges & Engineering Decisions](#dataset-challenges--engineering-decisions)
5. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
6. [Anomaly Labeling Pipeline](#anomaly-labeling-pipeline)
   - 6a. [Heuristic Labeling (`label_data.py`)](#heuristic-labeling-label_datapy)
   - 6b. [Label Generation for Stage 2 (`generate_labels.py`)](#label-generation-for-stage-2-generate_labelspy)
7. [Baseline Models](#baseline-models)
8. [Stage 1: GPU-Accelerated Graph Autoencoder](#stage-1-gpu-accelerated-graph-autoencoder)
9. [Stage 2: Semi-Supervised GAT Fine-Tuning](#stage-2-semi-supervised-gat-fine-tuning)
10. [Custom CUDA Extension](#custom-cuda-extension-cuda_spmm)
11. [Preprocessing Results](#preprocessing-results)
12. [How to Run](#how-to-run)
13. [Dependencies](#dependencies)
14. [Key Takeaways](#key-takeaways-from-the-project)

---

## Project Overview

This project implements a **GPU-accelerated version** of the **GNN-EADD** (Graph Neural Network-based E-Commerce Anomaly Detection via Dual-Stage Learning) framework. The system models an Amazon e-commerce ecosystem as a **heterogeneous graph** and uses Graph Neural Networks to detect:

- **Fraudulent Products** — fake listings with inflated ratings and suppressed prices
- **Fake Reviews / Review Bombs** — coordinated, synchronized review bursts
- **Compromised / Shill Accounts** — buyers operating in dense collusive clusters

The dual-stage learning approach combines:
- **Stage 1 (Unsupervised):** A Graph Autoencoder (GAE) learns rich structural embeddings from the graph topology alone — no labels required.
- **Stage 2 (Semi-supervised):** A Graph Attention Network (GAT) fine-tunes those embeddings using a small set of labeled anomalies.

---

## Research Paper Reference

This project is a GPU-accelerated reimplementation and extension of:

> *GNN-EADD: Graph Neural Network-Based E-Commerce Anomaly Detection via Dual-Stage Learning*

The paper proposes a heterogeneous graph framework with a two-stage learning pipeline for detecting fraud in online marketplaces. Our implementation targets the **Amazon Electronics** dataset and introduces significant hardware optimizations — including custom CUDA C++ kernels — that are **not present in the original paper**.

> **Note:** The original research paper leaves several implementation details underspecified — including the exact dataset version used, the edge construction methodology, and the anomaly labeling approach. These gaps required independent research and engineering decisions by our team, all of which are documented below.

---

## Project Architecture

```
Pop_project-main/
|
|-- data_preprocessing.py       # Data Prep: Full heterogeneous graph construction pipeline
|-- generate_labels.py          # Stage 2 Prep: Maps anomaly labels to global graph IDs
|-- label_data.py               # Anomaly labeling (K-Core + Heuristics)
|-- performance_evaluation.py   # Computes Top-K metrics and thresholding
|
|-- analysis/                   # Comprehensive visualization and plotting suite
|   |-- generate_all_plots.py   # Master script to compile all evaluation figures
|   |-- plot_*.py               # (10+ specific metric plotting scripts)
|   `-- figures/                # Output directory for generated plots
|
|-- baselines/
|   |-- baseline_performance_metrics.py # Evaluates and compares baseline models
|   |-- dominant_anomaly.py     # Baseline 1: HeteroDOMINANT model
|   |-- sage_anomaly.py         # Baseline 2: GraphSAGE Autoencoder
|   `-- readme.md               # Baselines documentation
|
|-- cuda_spmm/
|   |-- spmm_kernel.cu          # Custom CUDA warp-aligned SpMM kernel for Stage 1
|   |-- warp_gat_kernel.cu      # Custom CUDA warp-level GAT aggregation for Stage 2
|   `-- setup.py                # PyTorch C++ Extension build script
|
|-- stage1.py                   # Stage 1: Triple-Stream GAE Training Loop
|-- stage2.py                   # Stage 2: Semi-Supervised GAT Fine-Tuning
|
|-- documentation/              # Detailed documentation for individual pipeline components
|   |-- README_data_processing.md
|   |-- README_generate_labels.md
|   |-- README_label_data.md
|   |-- README_stage1.md
|   `-- README_stage2.md
|
|-- image_preprocessing.png     # Preprocessing output statistics screenshot
`-- README.md                   # This file
```

### Detailed Component Documentation
For deep-dives into individual components, refer to the documents in the `documentation/` folder:
- [Data Preprocessing Pipeline (`documentation/README_data_processing.md`)](documentation/README_data_processing.md)
- [Anomaly Labeling (`documentation/README_label_data.md`)](documentation/README_label_data.md)
- [Label Generation (`documentation/README_generate_labels.md`)](documentation/README_generate_labels.md)
- [Stage 1 Unsupervised GAE (`documentation/README_stage1.md`)](documentation/README_stage1.md)
- [Stage 2 Semi-Supervised GAT (`documentation/README_stage2.md`)](documentation/README_stage2.md)
- [Baseline Models (`baselines/readme.md`)](baselines/readme.md)
- [Quick Run Guide (`INSTRUCTIONS.md`)](INSTRUCTIONS.md)

| Component | Description |
| :--- | :--- |
| **Data Preprocessing** | SSD-streaming pipeline, NLP features, CSR generation |
| **Anomaly Labeling** | K-Core decomposition + multi-heuristic flagging |
| **Label Generation** | Maps anomaly IDs to global graph space, balanced sampling |
| **Baseline Models** | DOMINANT and GraphSAGE baselines for comparison |
| **Custom CUDA Kernels** | Warp-aligned SpMM, sparse dot-product decoder, and warp GAT aggregation |
| **Stage 1 (GAE)** | 2-layer Triple-Stream GCN encoder trained for unsupervised link prediction |
| **Stage 2 (GAT)** | 2-layer Semi-supervised GAT fine-tuning with node anomaly scoring head |

---

## Dataset: Challenges & Engineering Decisions

This section documents the key engineering challenges our team faced during the data preparation phase — challenges that the original paper does not address in detail.

### 1. Finding the Right Dataset

The first challenge was identifying a suitable dataset. We explored sources beyond Amazon — including other e-commerce platforms and academic repositories — but found nothing that provided the structural richness (users, products, seller relationships, and reviews) needed for heterogeneous graph modeling.

We ultimately settled on the **Amazon Electronics** dataset (5-core reviews + full metadata), which required extensive validation and filtering to bring it close to the scale and structure implied by the paper.

### 2. The Seller Node Problem

The most fundamental graph design challenge: **the Amazon dataset does not contain a "Seller" node type.** There is no seller ID, no seller table, nothing explicit.

Our solution was to **engineer seller nodes from the `brand` field** in the product metadata. However, brand names in raw Amazon data are notoriously inconsistent:
- `"Amazon"`, `"amazon"`, `"AMAZON"`, `"Amazon.com"` — all represent the same entity
- Many entries had `None`, `"Unknown"`, `"N/A"`, `"Generic"`, or blank values

We built a brand normalization pipeline (`_clean_brand()`) that:
1. Lowercased and stripped all whitespace
2. Filtered out a comprehensive set of `INVALID_BRANDS` (unknown, generic, n/a, no brand, amazon, etc.)
3. Treated each unique cleaned brand name as one seller node

This gave us a **meaningful, deduplicated seller node set** that could represent real merchants in the graph.

### 3. ASIN String IDs to GPU-Compatible Integer IDs

Amazon's product IDs (ASINs) are **alphanumeric strings** (e.g., `"B00005N5PF"`). GPUs cannot process string indices — all graph operations require integer node IDs.

We built a **unified, non-overlapping global ID namespace** across all three node types:
- Users: IDs `[0, N_u)`
- Products: IDs `[N_u, N_u + N_p)`
- Sellers: IDs `[N_u + N_p, N_u + N_p + N_s)`

Critically, the project uses **two separate data files** — the `5-core` review interactions and the full `metadata`. Both files contain ASINs, but they must be **cross-linked correctly** so that review interactions and product features align to the same integer node ID. Getting this mapping right across both files, at scale (~750K products), required careful multi-pass parsing.

### 4. Dataset Scale & Consistency

The research paper's reported dataset statistics were difficult to match exactly:

- We tested multiple yearly snapshots of the Electronics dataset (2014, 2018, 2023 versions)
- The paper does not specify the exact dataset version or release year used
- Preprocessing details such as minimum interaction thresholds, filtering criteria, and node selection rules were not described

After applying our own filtering (removing unknown brands, enforcing the 5-core constraint, etc.), we arrived at a graph scale in the same order of magnitude as the paper's reported figures. We proceeded with this filtered dataset, focusing on faithfully reproducing the methodology.

### 5. Edge Construction

The paper describes a heterogeneous graph schema at a high level but does not specify the rules for building edges between node types. We independently designed the following edge schema based on the semantics of the data:

| Edge Type | Direction | Semantics |
| :--- | :--- | :--- |
| `E_pu` | User -> Product | User has reviewed/purchased the product |
| `E_up` | Product -> User | Transpose of E_pu (for bidirectional GCN) |
| `E_ps` | Product -> Seller | Product is listed by this brand/seller |
| `E_sp` | Seller -> Product | Transpose of E_ps |
| `E_uu` | User <-> User | Users who co-purchased the same product (co-buy graph) |

The User-User (`E_uu`) edges in particular were a deliberate design choice — they allow the GCN to propagate signals through social collusion networks. To prevent an explosion in edge count for highly popular products, we cap the **co-buyer sampling at 25 users per product** (`MAX_USERS_PER_PROD_EUU = 25`).

### 6. Anomaly Labels

Stage 2 requires **labeled anomaly nodes** for semi-supervised fine-tuning, but the paper does not describe an anomaly labeling methodology, release a label set, or cite any external labeling tool. With no reference implementation available, we independently designed a **multi-criteria heuristic labeling pipeline** (`label_data.py` + `generate_labels.py`) that uses:

1. **K-Core Decomposition** — dense subgraph membership (top 0.6% of core numbers) signals coordinated shill networks
2. **Review Boosting Heuristic** — very high ratings (>= 4.8 stars) + suspiciously low price (< 15% of category median) + brand-title mismatch
3. **Fake Seller Heuristic** — high "also_buy" count (> 80) combined with < 25% verified purchase ratio
4. **Temporal Burst Detection** — users posting > 35 reviews at the exact same Unix timestamp

Any node satisfying **any** of these criteria is flagged as anomalous. The resulting labels are then processed by `generate_labels.py` into a balanced, globally-indexed format ready for Stage 2 training.

---

## Data Preprocessing Pipeline

**File:** `data_preprocessing.py` | **Docs:** [`documentation/README_data_processing.md`](documentation/README_data_processing.md)

The pipeline is designed for **16GB RAM efficiency** using chunked NLP encoding and memory-mapped outputs. It executes in four sequential passes over the raw data files.

### Graph Schema

```
[User V_u]--E_pu-->[Product V_p]--E_ps-->[Seller V_s]
     ^                   ^
     `------E_uu----------                (co-purchase proximity)
     <--E_up--           <--E_sp--
```

### Pass 1 — ID Space Construction
- Scans metadata to enumerate all valid Products (ASINs) and Sellers (brands)
- Scans reviews to enumerate all valid Users
- Allocates a unified, non-overlapping 32-bit integer namespace

### Pass 2 — NLP Fraud Signatures
Streams reviews without storing raw text. Per product and per user, computes:
- **Review count, average rating, rating variance**
- **Temporal span** (max timestamp - min timestamp)
- **Lexical diversity** (unique words / total words)
- **Sentiment-rating mismatch** via VADER (`compound` score vs. normalized star rating)
- **Helpful vote count, word count**

### Pass 3 — Multi-modal Feature Encoding (128-D float16)
Each node is projected into a **128-dimensional float16** feature vector:

| Node Type | Composition |
| :--- | :--- |
| **Product** `V_p` | 96-D text embedding (MiniLM -> PCA) + 24-D category multi-hot (PCA) + 8-D behavioral stats |
| **User** `V_u` | 112-D category preference (PCA) + 16-D behavioral stats |
| **Seller** `V_s` | 112-D mean product embedding (PCA) + 16-D catalog stats |

Text encoding uses `sentence-transformers/all-MiniLM-L6-v2` with chunked batching (10K texts at a time) to avoid RAM overflow. PCA is applied incrementally via `IncrementalPCA`.

### Pass 4 — CSR Topology Generation
Builds **Compressed Sparse Row (CSR)** binary files for all 5 edge types (including transposes and GCN self-loops). CSR reduces memory from $\mathcal{O}(N^2)$ to $\mathcal{O}(|E|)$.

**Output files:**
```
epu_row_ptr.bin / epu_col_idx.bin
epu_T_row_ptr.bin / epu_T_col_idx.bin
eps_row_ptr.bin / eps_col_idx.bin
eps_T_row_ptr.bin / eps_T_col_idx.bin
euu_row_ptr.bin / euu_col_idx.bin
X_combined.memmap       <- (N_u + N_p + N_s) x 128 float16
V_p_features.memmap
V_u_features.memmap
V_s_features.memmap
node_counts.json
node_id_mappings.json
```

---

## Anomaly Labeling Pipeline

### Heuristic Labeling (`label_data.py`)

**File:** `label_data.py` | **Docs:** [`documentation/README_label_data.md`](documentation/README_label_data.md)

This pipeline operates on the **bipartite user-product interaction graph** to generate anomaly labels using a combination of graph-structural analysis and domain heuristics.

### Algorithm

```
Bipartite Graph G = (Users U Products, Reviews)
     |
     v
K-Core Decomposition  ->  core_number per node
     |
     v
Heuristic Flags applied per entity type:

Products:
  is_fake_product   = (avg_rating >= 4.8) AND (price < 0.15 x category_median) AND (brand not in title)
  is_fake_seller    = (also_buy_count > 80) AND (verified_ratio < 0.25)
  is_kcore_anomaly  = core_number >= 99.4th percentile
  is_anomaly        = ANY of the above

Users:
  is_kcore_anomaly  = core_number >= 99.4th percentile
  is_burst_reviewer = posted > 35 reviews at the exact same Unix timestamp
  is_anomaly        = EITHER of the above
```

**Output:**
- `labelling_meta.csv` — anomalous products with all flags
- `labelling_5core.csv` — anomalous users
- `labelling_asin_meta.txt` — flagged ASIN list
- `labelling_asin_5_core.txt` — ASINs targeted by anomalous users

---

### Label Generation for Stage 2 (`generate_labels.py`)

**File:** `generate_labels.py` | **Docs:** [`documentation/README_generate_labels.md`](documentation/README_generate_labels.md)

This script acts as the **bridge between the anomaly labeling pipeline and Stage 2 GAT training**. It takes the raw flagged IDs and converts them into a balanced, globally-indexed label file.

**Workflow:**

1. **Global ID Mapping** — Loads `node_id_mappings.json` from Stage 1 and maps each flagged user/product string ID to its integer global graph ID. Any node filtered out during preprocessing is safely skipped.
2. **Anomaly Registration** — Reads `labelling_asin_5_core.txt` (users) and `labelling_asin_meta.txt` (products), collects their global IDs as the positive (anomaly) set.
3. **Balanced Negative Sampling** — Samples an equal number of non-anomalous nodes from the remaining graph to create a **1:1 anomaly-to-normal ratio**, preventing class imbalance from biasing the BCE loss in Stage 2.
4. **Compilation & Shuffling** — Assigns binary labels (`1 = Anomaly`, `0 = Normal`), stacks into a `[L, 2]` array (column 0: global ID, column 1: label), and shuffles with a fixed seed (`42`) for reproducibility.

**Output:** `labels.npy` — shape `[L, 2]`, consumed directly by the Stage 2 GAT training script.

---

## Baseline Models

**Docs:** [`baselines/readme.md`](baselines/readme.md)

Two baselines are implemented for comparison with our GNN-EADD approach.

### Baseline 1: HeteroDOMINANT (`baselines/dominant_anomaly.py`)

A heterogeneous adaptation of the **DOMINANT** (Deep Anomaly Detection on Attributed Networks) algorithm.

- **Encoder:** `HeteroConv` with `SAGEConv` operators across Buyer->Product and Seller->Product relations
- **Decoder:** Dual objective — Attribute reconstruction (MSE) + Structural link prediction (BCE)
- **Loss:** $\mathcal{L} = \alpha \cdot \text{MSE}(X, \hat{X}) + (1-\alpha) \cdot \text{BCE}(S, \hat{S})$, with $\alpha = 0.7$
- **Anomaly Score:** Euclidean norm of attribute reconstruction error; 3-sigma thresholding

### Baseline 2: GraphSAGE Autoencoder (`baselines/sage_anomaly.py`)

A homogeneous product-to-product graph baseline using `GraphSAGE`:

- **Graph:** Product nodes connected by `also_buy` / `also_viewed` metadata links
- **Features:** Price (normalized) + Category ID (hashed)
- **Objective:** MSE reconstruction of original node features
- **Anomaly Score:** $L_2$ norm of $(X - \hat{X})$; dynamic percentile threshold targeting ~4,500 anomalies

---

## Stage 1: GPU-Accelerated Graph Autoencoder

**File:** `stage1.py` | **Docs:** [`documentation/README_stage1.md`](documentation/README_stage1.md)

### Architecture: `TripleStreamGAE`

A 2-layer heterogeneous GCN encoder with dedicated **type-specific weight matrices** per relation and **three parallel CUDA streams** for concurrent heterogeneous message passing.

**Layer computation per node type (Paper Eq. 7):**

$$H^{(l)}_u = \text{ReLU}\left(W^{(l)}_{pu} \cdot \tilde{A}_{pu} X + W^{(l)}_{uu} \cdot \tilde{A}_{uu} X + W^{(l)}_{\text{self},u} \cdot X_u\right)$$

$$H^{(l)}_p = \text{ReLU}\left(W^{(l)}_{up} \cdot \tilde{A}_{up} X + W^{(l)}_{ps} \cdot \tilde{A}_{ps} X + W^{(l)}_{\text{self},p} \cdot X_p\right)$$

$$H^{(l)}_s = \text{ReLU}\left(W^{(l)}_{sp} \cdot \tilde{A}_{sp} X + W^{(l)}_{\text{self},s} \cdot X_s\right)$$

Where $\tilde{A} = D^{-1/2} A D^{-1/2}$ is the **symmetric GCN normalization** computed globally across all node types.

**Training Details:**

| Hyperparameter | Value |
| :--- | :--- |
| Input / Hidden / Output Dim | 128 / 128 / 128 |
| Epochs | 200 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy with Logits (link prediction) |
| Positive Samples per epoch | E_pu: 30K + E_ps: 8K + E_uu: 12K = 50K total |
| Mixed Precision | FP16 forward pass (autocast), FP32 decoder & loss |

**Key Design Decisions:**
- **Self-loop masking in edge sampler:** Self-loops are included in the CSR for GCN normalization but are **explicitly filtered out** from the link prediction loss (`src != dst`), ensuring the model learns genuine topological connectivity rather than trivially reconstructing self-edges.
- **FP32 decoder:** The dot-product decoder is deliberately pulled *outside* of `autocast` to prevent FP16 overflow in the inner product computation.
- **Mixed precision gradient scaling:** `torch.amp.GradScaler` is used to prevent FP16 gradient underflow during backpropagation.

**Output:** `Z_embeddings_stage1.npy` — a `(N_total x 128)` float32 array of latent node embeddings, consumed by Stage 2.

---

## Stage 2: Semi-Supervised GAT Fine-Tuning

**File:** `stage2.py` | **Docs:** [`documentation/README_stage2.md`](documentation/README_stage2.md)

Using `Z_embeddings_stage1.npy` and `labels.npy`, Stage 2 implements a **semi-supervised Graph Attention Network (GAT)** that fine-tunes the unsupervised embeddings using a small set of labeled anomaly nodes to produce a per-node anomaly probability.

### Architecture: `TripleStreamGAT`

A 2-layer heterogeneous GAT encoder with type-specific attention per edge stream, followed by a linear anomaly scoring head:

| Layer | Details |
| :--- | :--- |
| **GAT Layers 1 & 2** | `TypeSpecificGATLayer` — three independent attention mechanisms: Purchase (E_pu), Selling (E_ps), Co-purchase (E_uu) |
| **Anomaly Head** | `nn.Linear(128, 1)` — maps final node embeddings to a sigmoid anomaly logit |

**Key Design Decisions:**
- **Hybrid backends:** Native PyTorch + `torch.utils.checkpoint` during training (gradients required); custom `warp_gat` CUDA kernel at inference (3–5× faster, no backward pass needed).
- **DropEdge (85%):** Randomly drops 85% of edges per training chunk — prevents over-smoothing and reduces peak VRAM pressure.
- **Micro-chunking (2,500 nodes):** Prevents VRAM spikes caused by high-degree nodes with thousands of neighbors.

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda \cdot \mathcal{L}_{\text{unsup}}, \quad \lambda = 0.5$$

- **Supervised** $\mathcal{L}_{\text{sup}}$: Binary cross-entropy on the labeled anomaly/normal nodes only — unlabeled nodes still participate via message passing.
- **Unsupervised** $\mathcal{L}_{\text{unsup}}$: Mean squared difference of anomaly scores across sampled edges — enforces that connected nodes share similar anomaly probabilities, propagating signal through unlabeled graph regions.

### Training Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Epochs | 100 |
| Optimizer | Adam (lr = 0.001) |
| Mixed Precision | FP16 autocast + GradScaler |
| λ (unsupervised weight) | 0.5 |
| DropEdge rate | 85% |
| Chunk size | 2,500 nodes |
| Unsupervised edge samples | 5,000 per edge type |
| Train / Val / Test split | 60% / 20% / 20% |

**Outputs:**

| File | Description |
| :--- | :--- |
| `anomaly_scores_stage2.npy` | `(N_total,)` float32 — sigmoid anomaly probability per node |
| `Z_stage2.npy` | `(N_total, 128)` float32 — refined node embeddings after GAT fine-tuning |
| `gat_stage2_best.pt` | Best model checkpoint (saved at peak validation AUC-ROC) |

---

## Custom CUDA Extension (`cuda_spmm/`)

**Files:** `cuda_spmm/spmm_kernel.cu` · `cuda_spmm/warp_gat_kernel.cu`

Two custom warp-aligned CUDA C++ extensions are compiled from this directory, each producing a separate Python-importable module:

| Extension | Module Name | Source File | Used In |
| :--- | :--- | :--- | :--- |
| Warp-aligned CSR SpMM + Sparse Dot Product | `custom_spmm` | `spmm_kernel.cu` | Stage 1 encoder & decoder |
| Warp-level GAT Aggregation | `warp_gat` | `warp_gat_kernel.cu` | Stage 2 inference |

### Kernel 1: `csr_spmm_warp_kernel` (Encoder)

Performs **Sparse Matrix x Dense Matrix** multiplication in CSR format:

- **1 warp (32 threads) = 1 node:** Each warp processes one row of the sparse adjacency matrix
- **Dynamic dimension support:** Handles up to 1024-D feature vectors via chunked accumulation (up to 32 chunks of 32 lanes)
- **Pre-computed symmetric edge weights:** Accepts the $D^{-1/2}_{ij}$ normalization weights as a float buffer, applying them at the neighbor aggregation step
- **FP16 input, FP32 accumulation:** Reads `half` precision features, accumulates in `float32` to prevent precision loss

```cuda
// Core loop: Each thread in the warp handles one feature dimension
for (int i = row_start; i < row_end; ++i) {
    int neighbor = col_idx[i];
    float weight = edge_weights[i];
    for (int c = 0; c < num_chunks; ++c) {
        int col = lane_id + c * 32;
        acc[c] += __half2float(in_features[neighbor * dim + col]) * weight;
    }
}
```

### Kernel 2: `warp_sparse_dot_product_kernel` (`spmm_kernel.cu` — Decoder)

Performs **batched sparse inner products** $Z_{src} \cdot Z_{dst}$ for sampled edge pairs:

- **1 warp = 1 edge:** Each warp computes the dot product for one (src, dst) node pair
- **Warp shuffle reduction:** Uses `__shfl_down_sync(0xffffffff, acc, offset)` for fast in-warp summation — eliminates global synchronization barriers
- **Dynamic dimension:** Supports arbitrary embedding dimensions

### Kernel 3: `warp_gat_forward_kernel` (`warp_gat_kernel.cu` — Stage 2 Inference)

Implements **warp-level GAT attention aggregation** for fast Stage 2 inference (`model.eval()`). Each warp (32 threads) processes one destination node across four sequential phases:

1. **Destination projection** — Each thread handles a stripe of feature dimensions (`d = lane_id, lane_id+32, ...`); a warp-parallel dot product computes `dst_proj = H_dst[u] · a_dst` and the result is broadcast to all 32 lanes via `__shfl_sync`.
2. **Max attention logit pass** — Edges are striped across lanes (`i = start + lane_id, start + lane_id + 32, ...`); each lane computes `src_proj = H_src[v] · a_src` and `e_uv = LeakyReLU(dst_proj + src_proj)`. `warpReduceMax` finds the global max logit for numerically stable softmax.
3. **Softmax denominator pass** — Second traversal over neighbors computes `exp_sum = Σ exp(e_uv − max_e)`; reduced via `warpReduceSum`. Both `max_e` and `exp_sum` are broadcast to all lanes.
4. **Feature aggregation** — Feature dimensions are striped across lanes; for each dimension, all neighbors are accumulated: `out[u, d] = Σ_v α_uv · H_src[v, d]` where `α_uv = exp(e_uv − max_e) / (exp_sum + 1e-16)`.

**Launch config:** 128 threads/block (4 warps/block). **Attention vector split:** the parameter `a_vec` (length `2 × dim`) is divided as `a_dst = a_vec[:dim]`, `a_src = a_vec[dim:]` inside the Python wrapper before passing to the kernel.

### Backpropagation (Python-side)

The `SpMM_Autograd` custom `torch.autograd.Function` implements **chunk-streaming backprop** — processing 5,000 rows at a time to eliminate VRAM spikes during gradient computation on large graphs.

**Build:**
```bash
cd cuda_spmm
python setup.py install
cd ..
```

---

## Preprocessing Results

Run on an E-Commerce dataset (node and edge counts scale dynamically to any provided source):

| Metric | Value |
| :--- | :--- |
| **Users** $\|V_u\|$ | 728,719 |
| **Products** $\|V_p\|$ | 756,077 |
| **Sellers** $\|V_s\|$ | 55,679 |
| **Total Nodes** | **1,540,475** |
| **E_pu** (User->Product) | 7,253,058 |
| **E_up** (Product->User) | 7,280,416 |
| **E_ps** (Product->Seller) | 1,492,561 |
| **E_sp** (Seller->Product) | 792,163 |
| **E_uu** (User->User) | 42,645,547 |
| **Total Edges** | **~59.4 Million** (incl. self-loops) |
| **Feature Matrix Size** | V_p: 193.6 MB, V_u: 186.6 MB, V_s: 14.3 MB |
| **GPU VRAM Footprint** | **~0.39 GB** (features only) |
| **Processing Time** | **52.7 Minutes** |

![GNN-EADD Preprocessing Statistics](image_preprocessing.png)

---

## How to Run

> For a concise step-by-step reference, see [INSTRUCTIONS.md](INSTRUCTIONS.md).

### Step 1: Preprocess Data
Edit the file paths at the top of `data_preprocessing.py` to match your dataset:
```python
REVIEWS_FILE  = "5core_reviews.json"      # path to 5-core reviews file (.json or .json.gz)
METADATA_FILE = "meta_products.json"      # path to product metadata file (.json or .json.gz)
```
Then run:
```bash
python3 data_preprocessing.py
```

### Step 2: Generate Anomaly Labels
```bash
python label_data.py
```
This produces `labelling_asin_meta.txt` and `labelling_asin_5_core.txt`.

### Step 3: Build Stage 2 Label File
```bash
python generate_labels.py
```
This reads the Stage 1 ID mappings and the heuristic labels and produces `labels.npy` for Stage 2.

### Step 4: (Optional) Run Baselines
```bash
python baselines/dominant_anomaly.py
python baselines/sage_anomaly.py
python baselines/baseline_performance_metrics.py
```

### Step 5: Build the CUDA Extension
```bash
cd cuda_spmm
python setup.py install
cd ..
```

### Step 6: Train Stage 1 GAE
```bash
python stage1.py
```
Output: `Z_embeddings_stage1.npy` (fed into Stage 2 along with `labels.npy`)

### Step 7: Train Stage 2 GAT
```bash
python stage2.py
```
Output: `anomaly_scores_stage2.npy` and `Z_stage2.npy`

### Step 8: Evaluate Model Performance
```bash
python performance_evaluation.py
```
*Computes Top-K ranking precision/recall and applies type-specific (Users/Products) decision boundaries.*

### Step 9: Generate Visualizations
```bash
cd analysis
python generate_all_plots.py
cd ..
```
*Compiles PR curves, ROC, baseline comparisons, and embedding projections into `analysis/figures/`.*

---

## Dependencies

First, install PyTorch with the CUDA version matching your system toolkit:

```bash
# CUDA 11.8 (RTX 30xx / older):
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1 (recommended for RTX 40xx and newer):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install all remaining dependencies:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

> Requires a CUDA-capable GPU (Compute Capability 7.0+) and a matching CUDA Toolkit for building the custom kernel extensions.

---


## Key Takeaways from the Project

This project implemented and extended the GNN-EADD framework end-to-end — from raw JSON to trained model to evaluation metrics. The paper defines the dual-stage learning architecture; everything below represents original engineering work required to make it function at scale.

---

### 1. Reproducing a Paper's Data Pipeline Is Harder Than Reproducing Its Model

The GNN-EADD paper leaves every data engineering decision underspecified. Each of the following required independent research and iterative design:

- **Seller node construction:** Amazon provides no seller entity. We derived seller nodes from the product `brand` field, building a normalization pipeline (`_clean_brand()`) that deduplicates and filters a comprehensive set of invalid strings (`generic`, `unknown`, `amazon`, `n/a`, etc.) — without which the seller namespace would be polluted with thousands of junk nodes.
- **Co-purchase graph (E_uu):** The paper describes a user-user edge type but gives no construction rule. We built it from co-reviewer sets per product, with an explicit density cap of `MAX_USERS_PER_PROD_EUU = 25` — without this cap, a single product with 1,000 reviewers would produce ~500K edges.
- **Feature engineering:** The paper specifies 128-D node vectors but no extraction methodology. We built a multi-modal blueprint using MiniLM sentence embeddings (384-D → 96-D via PCA), multi-hot product categories (PCA-compressed), and 8–16 behavioral fraud signatures per node type — including VADER sentiment-rating mismatch, lexical diversity, and temporal review span.
- **ASIN-to-integer ID namespace:** Amazon's product identifiers are alphanumeric strings. GPUs require integer indices. We constructed a unified, non-overlapping 32-bit namespace across all three node types across two separate source files (reviews and metadata), ensuring consistent alignment at ~750K products.
- **Anomaly labeling:** The paper requires labeled nodes for Stage 2 but neither describes a labeling approach nor releases any label set. We independently designed a multi-criteria heuristic pipeline combining:
  - **K-Core decomposition** (top 0.6% of core values — 99.4th percentile threshold) for structural collusion detection
  - **Review boosting heuristic:** avg_rating ≥ 4.8 AND price < 15% of category median AND brand absent from product title
  - **Fake seller heuristic:** `also_buy_count > 80` AND verified purchase ratio < 25%
  - **Temporal burst detection:** any user posting > 35 reviews at the exact same Unix timestamp (physically impossible for a human reviewer)
  - **Two user fraud profiles:** *Collusive Shill* (high K-Core + low verified ratio) vs *Spambot* (burst pattern) — the dual-profile design protects high-activity legitimate "power users" from false flagging

---

### 2. Custom CUDA Kernels Delivered Concrete Speedups

Three CUDA kernels were written from scratch — none of these are described anywhere in the original paper:

- **`csr_spmm_warp_kernel`:** 1 warp (32 threads) processes one graph node. FP16 feature reads accumulate into FP32 registers, preventing precision loss across up to 32 chunked dimension blocks (1024-D max). Zero shared memory allocation avoids bank conflicts entirely. Launch config: 256 threads/block.
- **`warp_sparse_dot_product_kernel`:** 1 warp per sampled edge pair. In-warp dot product reduction via `__shfl_down_sync(0xffffffff, acc, offset)` — no global synchronization barriers. Allows the link prediction decoder to remain parameter-free (pure cosine similarity) without VRAM overhead.
- **`warp_gat_forward_kernel`:** 4-phase neighborhood traversal per warp: (1) broadcast destination projection, (2) max logit pass for numerical stability, (3) softmax denominator pass, (4) feature aggregation. Achieves **3–5× faster inference** than native PyTorch at 1.5M nodes. The hybrid strategy — native PyTorch with `torch.utils.checkpoint` for training, custom CUDA for `model.eval()` — exploits the asymmetry that backpropagation requires computational graph flexibility, while inference only needs raw throughput.

---

### 3. Scaling to ~60 Million Edges Required Simultaneous VRAM and RAM Engineering

No single technique was sufficient; all of the following were needed together:

- **Selective FP16/FP32 precision:** Feature matrices stored as `float16` (~0.39 GB vs ~0.78 GB for FP32). However, the Stage 1 link prediction decoder and BCE loss are deliberately pulled *outside* `autocast` — FP16 dot products of 128-D vectors overflow, corrupting gradients silently.
- **Chunk-streaming backpropagation (`SpMM_Autograd`):** Gradients flow through the custom `torch.autograd.Function` in 5,000-row chunks, capping peak VRAM allocation during backward passes regardless of graph size.
- **85% DropEdge (Stage 2 training):** Drops 85% of edges per micro-chunk during training. Primary effect: eliminates VRAM spikes from dense super-nodes. Secondary effect: prevents over-smoothing in 2-layer GAT.
- **Micro-chunking (2,500 nodes per chunk):** Prevents the full neighbor list of high-degree hub nodes from materializing simultaneously in GPU memory.
- **16 GB RAM preprocessing budget:** Chunked NLP encoding (10K texts at a time), `IncrementalPCA` (handles dimensions > sample count via zero-padding), and `float16` memmap outputs kept the entire preprocessing pipeline below 16 GB of system RAM for a graph with 1.54M nodes.

---

### 4. Training Dynamics Matter as Much as Architecture

Several non-obvious design decisions had a measurable impact on whether the model converged at all:

- **Temperature annealing (`τ`) in Stage 1:** Without the warmup schedule (`τ`: 2.0 → 10.0 over 20 epochs), `sigmoid(score × τ)` saturates at training start, collapsing gradients before the encoder learns anything. The subsequent cosine decay (`τ`: 10.0 → 3.0, epochs 21–200) produces sharper decision boundaries as the model matures. Both phases are implemented as a single closed-form schedule in `get_temperature()`.
- **Type-consistent negative sampling:** Negative edges are sampled *within* the correct destination node-type boundary (fake User→Product edges point only to real Product nodes). Without this constraint, the model trivially separates negatives by node type rather than learning genuine structural patterns.
- **Heterogeneous sampling budget:** EPU:EPS:EUU = 30K:8K:12K. Over-sampling the minority Product→Seller edges (8K vs their proportion of total edges) prevents the link prediction loss from being dominated by the majority User→Product relation.
- **CSR self-loop sanitization (Stage 2):** Stage 1 preprocessing inserts GCN self-loops in *global* ID space — each node loops to itself via its global integer ID. In Stage 2's bipartite-constrained GAT, a user's self-loop in the E_pu CSR points to a User-range ID, which is out-of-bounds for the Product node type. `load_and_sanitize_csr()` reconstructs the row pointer array from scratch after filtering, an engineering step with no analogue in the paper.

---

### 5. The Dual-Stage Pipeline Validated the Design Hypothesis

Stage 1 unsupervised training converged to **Edge AUC = 0.8431** with a positive-negative sigmoid margin of **+0.372** (P(pos) = 0.759 vs P(neg) = 0.387) after 200 epochs — in **290 seconds** on a consumer GPU. This confirms the custom CUDA encoder's efficiency benefit at this scale, and demonstrates that the embeddings encode genuine topological structure rather than trivial node-type separation.

Stage 2 then leveraged these structural embeddings as a high-quality prior. The semi-supervised loss (`L_total = L_sup + 0.5 · L_unsup`) ensures anomaly signal from labeled nodes propagates into unlabeled regions through the smoothness constraint — meaning even nodes with no label assignment receive meaningful gradient updates through message passing. The type-specific evaluation thresholds (82nd percentile for users, sigmoid midpoint 0.5 for products) reflect the domain insight that user anomaly score distributions are more skewed than product score distributions.

Collectively, this project demonstrates that applying a GNN-based fraud detection paper to a real-world dataset at million-node scale is primarily an exercise in systems engineering: the novel research contribution is clear, but executing it requires solving a chain of concrete engineering problems that the paper abstracts away entirely.

