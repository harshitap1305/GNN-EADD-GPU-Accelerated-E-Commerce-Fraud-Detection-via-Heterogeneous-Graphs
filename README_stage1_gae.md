# Stage 1: Unsupervised Graph Autoencoder (GAE)

This directory contains the implementation of **Stage 1** from the *GNN-EADD (Graph Neural Network-Based E-Commerce Anomaly Detection via Dual-Stage Learning)* paper. 

The goal of this stage is to learn robust, low-dimensional structural representations (embeddings, `Z`) of nodes in a heterogeneous e-commerce graph (Users, Products, Sellers) in an entirely unsupervised manner.

## Implementation Features

This implementation is highly optimized for large graphs and limited VRAM:
- **Triple-Stream GCN Encoder**: Aggregates heterogeneous neighbor features using type-specific weight matrices and symmetric GCN normalization. Includes self-loops to preserve node features during message passing.
- **Custom CUDA SpMM Kernel**: Utilizes a warp-aligned, custom C++ / CUDA Sparse Matrix-Matrix Multiplication (SpMM) kernel with chunk streaming backprop to eliminate PyTorch OOM spikes.
- **Mixed Precision**: Uses `torch.amp.autocast` (FP16 forward pass, FP32 dot-product decoder and loss) to accelerate training and reduce memory footprint.
- **Out-of-Core Feature Loading**: Maps raw float16 node features directly from SSD (`.memmap`), loading only what is needed.

## Prerequisites

Before running the training script, ensure you have:
1. Compiled the custom CUDA extension in the `cuda_spmm/` directory:
   ```bash
   cd cuda_spmm
   python setup.py install
   cd ..
   ```
2. Generated the compiled graph dataset binaries:
   - CSR structural files (e.g., `epu_row_ptr.bin`, `epu_col_idx.bin`, etc.)
   - Node count metadata (`node_counts.json`)
   - Combined feature memmap (`X_combined.memmap`)

## Usage

Run the training script directly:

```bash
python train_stage1_gae.py
```

### Training Flow:
1. Loads the heterogeneous graph structure (`epu`, `eps`, `euu`) and compute symmetric structural normalizations.
2. Trains a 2-layer GCN encoder over 200 epochs using a sampled inner-product decoder and Binary Cross-Entropy (BCE) reconstruction loss.
3. Automatically switches to pure inference at the end to compute the final embeddings for all nodes.

### Output:
- Saves the computed 128-D latent embeddings to `Z_embeddings_stage1.npy`.
- These embeddings are consumed by Stage 2 (Semi-supervised GAT specific for anomaly detection).
<img width="1056" height="686" alt="Screenshot from 2026-04-07 17-36-37" src="https://github.com/user-attachments/assets/346f4d06-73cd-4ef4-8572-a82df456c24b" />

