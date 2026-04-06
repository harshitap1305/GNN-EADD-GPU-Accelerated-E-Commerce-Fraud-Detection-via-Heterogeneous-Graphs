import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time

# Import our custom CUDA kernel!
import custom_spmm 

# ---------------------------------------------------------
# 1. Memory-Mapped DataLoader
# ---------------------------------------------------------
def load_graph_data():
    print("Loading Graph Topology from Binaries...")
    with open('node_counts.json', 'r') as f:
        counts = json.load(f)
    
    N_u = counts['users']
    N_p = counts['products']
    N_s = counts['sellers']
    num_nodes = counts['total']

    def load_csr(prefix):
        rp = np.fromfile(f"{prefix}_row_ptr.bin", dtype=np.int32)
        ci = np.fromfile(f"{prefix}_col_idx.bin", dtype=np.int32)
        return torch.from_numpy(rp).cuda(), torch.from_numpy(ci).cuda()

    # Load all forward and backward edges
    epu_rp, epu_ci = load_csr('epu')
    epu_T_rp, epu_T_ci = load_csr('epu_T')
    
    eps_rp, eps_ci = load_csr('eps')
    eps_T_rp, eps_T_ci = load_csr('eps_T')
    
    euu_rp, euu_ci = load_csr('euu')

    print("Mapping 128-D Float16 Features directly from SSD...")
    X_memmap = np.memmap('X_combined.memmap', dtype='float16', mode='r', shape=(num_nodes, 128))
    # Wrapping in np.array() fixes the PyTorch read-only warning before moving to VRAM
    X_tensor = torch.from_numpy(np.array(X_memmap)).cuda() 

    return X_tensor, N_u, N_p, N_s, num_nodes, \
           epu_rp, epu_ci, epu_T_rp, epu_T_ci, \
           eps_rp, eps_ci, eps_T_rp, eps_T_ci, \
           euu_rp, euu_ci

# ---------------------------------------------------------
# 2. Triple-Stream GAE Architecture
# ---------------------------------------------------------
class TripleStreamGAE(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64):
        super().__init__()
        # Compression Layer (Encoder)
        self.compressor = nn.Linear(in_dim, hidden_dim)

    def forward(self, X, epu_rp, epu_ci, epu_T_rp, epu_T_ci, eps_rp, eps_ci, eps_T_rp, eps_T_ci, euu_rp, euu_ci):
        
        # --- STREAM 1: User Node Representation ---
        H_pu = custom_spmm.forward(epu_rp, epu_ci, X)
        H_uu = custom_spmm.forward(euu_rp, euu_ci, X)
        H_user = (H_pu + H_uu) / 2.0

        # --- STREAM 2: Product Node Representation ---
        H_up = custom_spmm.forward(epu_T_rp, epu_T_ci, X)
        H_ps = custom_spmm.forward(eps_rp, eps_ci, X)
        H_product = (H_up + H_ps) / 2.0

        # --- STREAM 3: Seller Node Representation ---
        # Note: eps_T output maps back to the Seller dimension
        H_sp = custom_spmm.forward(eps_T_rp, eps_T_ci, X)
        H_seller = H_sp

        # Re-concatenate the isolated streams back into a unified global matrix
        H_combined = torch.cat([H_user, H_product, H_seller], dim=0)
        
        # Compress down to 64-D Latent Space Z
        Z = F.relu(self.compressor(H_combined))
        return Z

# ---------------------------------------------------------
# 3. Dynamic CSR Edge Sampler
# ---------------------------------------------------------
def sample_positive_edges(rp, ci, src_offset, num_samples):
    """Dynamically reconstructs actual graph edges from CSR arrays for training."""
    row_lengths = rp[1:] - rp[:-1]
    src = torch.repeat_interleave(torch.arange(len(rp)-1, device=rp.device), row_lengths) + src_offset
    dst = ci
    
    if len(src) == 0:
        return torch.empty(0, dtype=torch.long, device=rp.device), torch.empty(0, dtype=torch.long, device=rp.device)
        
    idx = torch.randint(0, len(src), (min(num_samples, len(src)),), device=rp.device)
    return src[idx].long(), dst[idx].long()

# ---------------------------------------------------------
# 4. Fast Edge Reconstruction & Training
# ---------------------------------------------------------
def train_gae():
    X, N_u, N_p, N_s, num_nodes, epu_rp, epu_ci, epu_T_rp, epu_T_ci, eps_rp, eps_ci, eps_T_rp, eps_T_ci, euu_rp, euu_ci = load_graph_data()
    
    model = TripleStreamGAE(in_dim=128, hidden_dim=64).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print("\nStarting Stage 1: Unsupervised Training...")
    t0 = time.time()
    
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        # 1. Encode: Get Latent Graph Representation Z using Custom CUDA
        Z = model(X, epu_rp, epu_ci, epu_T_rp, epu_T_ci, eps_rp, eps_ci, eps_T_rp, eps_T_ci, euu_rp, euu_ci)

        # 2. Sample True Edges from the CSR Graph (Reconstruction Targets)
        # We sample 10,000 edges from the User-Product interactions
        pos_src, pos_dst = sample_positive_edges(epu_rp, epu_ci, src_offset=0, num_samples=10000)
        
        # Decode: Dot Product between true node pairs
        pos_score = (Z[pos_src] * Z[pos_dst]).sum(dim=-1)
        
        # 3. Negative Sampling (Fake edges the model should learn to reject)
        neg_dst = torch.randint(0, num_nodes, (len(pos_src),), device='cuda').long()
        neg_score = (Z[pos_src] * Z[neg_dst]).sum(dim=-1)

        # 4. Loss Calculation (Binary Cross Entropy)
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        preds = torch.cat([pos_score, neg_score])
        loss = F.binary_cross_entropy_with_logits(preds, labels)

        # 5. Backpropagate
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Reconstruction Loss: {loss.item():.4f}")

    print(f"\nTraining Complete in {time.time() - t0:.2f}s!")
    
    # Extract Anomaly Scores (Nodes that fail reconstruction severely)
    model.eval()
    with torch.no_grad():
        Z_final = model(X, epu_rp, epu_ci, epu_T_rp, epu_T_ci, eps_rp, eps_ci, eps_T_rp, eps_T_ci, euu_rp, euu_ci)
        # Calculate how "far" each node is from the global graph norm (Anomaly Flag)
        anomaly_scores = torch.norm(Z_final - Z_final.mean(dim=0), dim=1).cpu().numpy()
        
    np.save('anomaly_scores.npy', anomaly_scores)
    print(f"Saved anomaly scores for {num_nodes} nodes to 'anomaly_scores.npy'")

if __name__ == "__main__":
    train_gae()
