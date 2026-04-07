# GNN-EADD: GPU-Accelerated E-Commerce Fraud Detection via Heterogeneous Graphs
### Phase 1 Report — Graph Construction, Data Engineering & Unsupervised Learning

> **Status:** Phase 1 Complete | Phase 2 (Semi-supervised GAT) — Pending

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
9. [Custom CUDA Extension](#custom-cuda-extension-cuda_spmm)
10. [Preprocessing Results](#preprocessing-results)
11. [How to Run](#how-to-run)
12. [Dependencies](#dependencies)
13. [What's Next: Phase 2](#whats-next-phase-2)

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

The paper proposes a heterogeneous graph framework with a two-stage learning pipeline for detecting fraud in online marketplaces. Our implementation targets the **Amazon Electronics** dataset and introduces significant hardware optimizations — including a custom CUDA C++ kernel — that are **not present in the original paper**.

> **Note:** The original research paper leaves several implementation details underspecified — including the exact dataset version used, the edge construction methodology, and the anomaly labeling approach. These gaps required independent research and engineering decisions by our team, all of which are documented below.

---

## Project Architecture

```
Pop_project-main/
|
|-- data_preprocessing.py       # Phase 1A: Full heterogeneous graph construction pipeline
|
|-- generate_labels.py          # Stage 2 Prep: Maps anomaly labels to global graph IDs
|-- README_generate_labels.md   # Documentation for generate_labels.py
|
|-- baselines/
|   |-- label_data.py           # Anomaly labeling (K-Core + Heuristics)
|   |-- dominant_anomaly.py     # Baseline 1: HeteroDOMINANT model
|   |-- sage_anomaly.py         # Baseline 2: GraphSAGE Autoencoder
|   `-- readme.md               # Baselines documentation
|
|-- cuda_spmm/
|   |-- spmm_kernel.cu          # Custom CUDA warp-aligned SpMM kernel
|   `-- setup.py                # PyTorch C++ Extension build script
|
|-- train_stage1_gae.py         # Stage 1: Triple-Stream GAE Training Loop
|-- image_preprocessing.png     # Preprocessing output statistics screenshot
`-- README.md                   # This file
```

| Component | Status | Description |
| :--- | :---: | :--- |
| **Data Preprocessing** | Complete | SSD-streaming pipeline, NLP features, CSR generation |
| **Anomaly Labeling** | Complete | K-Core decomposition + multi-heuristic flagging |
| **Label Generation** | Complete | Maps anomaly IDs to global graph space, balanced sampling |
| **Baseline Models** | Complete | DOMINANT and GraphSAGE baselines for comparison |
| **Custom CUDA Kernel** | Complete | Warp-aligned SpMM + sparse dot-product decoder |
| **Stage 1 (GAE)** | Complete | 2-layer Triple-Stream GCN encoder, trained 200 epochs |
| **Stage 2 (GAT)** | Pending | Semi-supervised fine-tuning with labeled anomaly nodes |

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

Stage 2 requires **labeled anomaly nodes** for semi-supervised fine-tuning, but the paper does not describe an anomaly labeling methodology, release a label set, or cite any external labeling tool. With no reference implementation available, we independently designed a **multi-criteria heuristic labeling pipeline** (`baselines/label_data.py` + `generate_labels.py`) that uses:

1. **K-Core Decomposition** — dense subgraph membership (top 0.6% of core numbers) signals coordinated shill networks
2. **Review Boosting Heuristic** — very high ratings (>= 4.8 stars) + suspiciously low price (< 15% of category median) + brand-title mismatch
3. **Fake Seller Heuristic** — high "also_buy" count (> 80) combined with < 25% verified purchase ratio
4. **Temporal Burst Detection** — users posting > 35 reviews at the exact same Unix timestamp

Any node satisfying **any** of these criteria is flagged as anomalous. The resulting labels are then processed by `generate_labels.py` into a balanced, globally-indexed format ready for Stage 2 training.

---

## Data Preprocessing Pipeline

**File:** `data_preprocessing.py`

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

**File:** `baselines/label_data.py`

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

**File:** `generate_labels.py` | **Docs:** `README_generate_labels.md`

This script acts as the **bridge between the anomaly labeling pipeline and Stage 2 GAT training**. It takes the raw flagged IDs and converts them into a balanced, globally-indexed label file.

**Workflow:**

1. **Global ID Mapping** — Loads `node_id_mappings.json` from Stage 1 and maps each flagged user/product string ID to its integer global graph ID. Any node filtered out during preprocessing is safely skipped.
2. **Anomaly Registration** — Reads `labelling_asin_5_core.txt` (users) and `labelling_asin_meta.txt` (products), collects their global IDs as the positive (anomaly) set.
3. **Balanced Negative Sampling** — Samples an equal number of non-anomalous nodes from the remaining graph to create a **1:1 anomaly-to-normal ratio**, preventing class imbalance from biasing the BCE loss in Stage 2.
4. **Compilation & Shuffling** — Assigns binary labels (`1 = Anomaly`, `0 = Normal`), stacks into a `[L, 2]` array (column 0: global ID, column 1: label), and shuffles with a fixed seed (`42`) for reproducibility.

**Output:** `labels.npy` — shape `[L, 2]`, consumed directly by the Stage 2 GAT training script.

---

## Baseline Models

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

**File:** `train_stage1_gae.py`

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
| Positive Samples per epoch | 30K per edge type (90K total) |
| Mixed Precision | FP16 forward pass (autocast), FP32 decoder & loss |

**Key Design Decisions:**
- **Self-loop masking in edge sampler:** Self-loops are included in the CSR for GCN normalization but are **explicitly filtered out** from the link prediction loss (`src != dst`), ensuring the model learns genuine topological connectivity rather than trivially reconstructing self-edges.
- **FP32 decoder:** The dot-product decoder is deliberately pulled *outside* of `autocast` to prevent FP16 overflow in the inner product computation.
- **Mixed precision gradient scaling:** `torch.amp.GradScaler` is used to prevent FP16 gradient underflow during backpropagation.

**Output:** `Z_embeddings_stage1.npy` — a `(N_total x 128)` float32 array of latent node embeddings, consumed by Stage 2.

---

## Custom CUDA Extension (`cuda_spmm/`)

**File:** `cuda_spmm/spmm_kernel.cu`

To satisfy the project's GPU kernel requirement and eliminate PyTorch OOM spikes on large sparse operations, we implemented a **custom warp-aligned CUDA C++** extension with two kernels:

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

### Kernel 2: `warp_sparse_dot_product_kernel` (Decoder)

Performs **batched sparse inner products** $Z_{src} \cdot Z_{dst}$ for sampled edge pairs:

- **1 warp = 1 edge:** Each warp computes the dot product for one (src, dst) node pair
- **Warp shuffle reduction:** Uses `__shfl_down_sync(0xffffffff, acc, offset)` for fast in-warp summation — eliminates global synchronization barriers
- **Dynamic dimension:** Supports arbitrary embedding dimensions

### Backpropagation (Python-side)

The `SpMM_Autograd` custom `torch.autograd.Function` implements **chunk-streaming backprop** — processing 5,000 rows at a time to eliminate VRAM spikes during gradient computation on 1.5M-node graphs.

**Build:**
```bash
cd cuda_spmm
python setup.py install
```

---

## Preprocessing Results

Run on the **Amazon Electronics 5-core** dataset:

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

### Step 1: Preprocess Data
Edit `data_preprocessing.py` to point to your raw JSON files:
```python
REVIEWS_FILE  = "Electronics_5.json"      # or .json.gz
METADATA_FILE = "meta_Electronics.json"   # or .json.gz
```
Then run:
```bash
python3 data_preprocessing.py
```

### Step 2: Generate Anomaly Labels
```bash
cd baselines
python label_data.py
cd ..
```
This produces `labelling_asin_meta.txt` and `labelling_asin_5_core.txt`.

### Step 3: Build Stage 2 Label File
```bash
python generate_labels.py
```
This reads the Stage 1 ID mappings and the heuristic labels and produces `labels.npy` for Stage 2.

### Step 4: (Optional) Run Baselines
```bash
cd baselines
python dominant_anomaly.py
python sage_anomaly.py
cd ..
```

### Step 5: Build the CUDA Extension
```bash
cd cuda_spmm
python setup.py install
cd ..
```

### Step 6: Train Stage 1 GAE
```bash
python train_stage1_gae.py
```
Output: `Z_embeddings_stage1.npy` (fed into Stage 2 along with `labels.npy`)

---

## Dependencies

```bash
pip install numpy scikit-learn sentence-transformers nltk
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric pandas networkx tqdm
python -c "import nltk; nltk.download('vader_lexicon')"
```

> Requires a CUDA-capable GPU and CUDA Toolkit installed for the custom kernel.

---

## What's Next: Phase 2

Phase 2 will implement **Stage 2: Semi-supervised Graph Attention Network (GAT)** fine-tuning.

Using `Z_embeddings_stage1.npy` and `labels.npy`, Stage 2 will:

1. **Load pre-trained embeddings** from Stage 1 as node initialization
2. **Apply a GAT layer** with multi-head attention over heterogeneous edges
3. **Fine-tune** using a semi-supervised cross-entropy loss — only labeled anomaly nodes are supervised; unlabeled nodes still contribute through message passing
4. **Score all nodes** for anomaly probability using the refined attention-weighted embeddings

**Planned hardware optimization:** A hardware-level GAT attention kernel using `__shfl_sync` warp primitives for all-neighbor reductions, eliminating global synchronization barriers and using numerically-stable softmax to prevent floating-point overflow.

---

## Key Takeaways from Phase 1

> The most significant insight from Phase 1 is that **reproducing a research paper's data pipeline is often harder than reproducing its model.** The GNN-EADD paper provides a compelling model architecture, but leaves the full data engineering problem — seller node construction, edge schema design, anomaly labeling, ASIN-to-integer mapping, and dataset version selection — largely unspecified. Every one of these decisions required independent research, multiple design iterations, and careful engineering to get right at the scale of 1.5 million nodes and ~60 million edges.
