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
    X_tensor = torch.from_numpy(np.array(X_memmap)).cuda() 

    return X_tensor, N_u, N_p, N_s, num_nodes, \
           epu_rp, epu_ci, epu_T_rp, epu_T_ci, \
           eps_rp, eps_ci, eps_T_rp, eps_T_ci, \
           euu_rp, euu_ci

# ---------------------------------------------------------
# 2. Triple-Stream GAE Architecture (2-Layer, 128-D)
# ---------------------------------------------------------
class TripleStreamGAE(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128):
        super().__init__()
        # --- LAYER 1 WEIGHTS ---
        self.W1_pu = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W1_uu = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W1_up = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W1_ps = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W1_sp = nn.Linear(in_dim, hidden_dim, bias=False)

        # --- LAYER 2 WEIGHTS ---
        self.W2_pu = nn.Linear(hidden_dim, out_dim, bias=False)
        self.W2_uu = nn.Linear(hidden_dim, out_dim, bias=False)
        self.W2_up = nn.Linear(hidden_dim, out_dim, bias=False)
        self.W2_ps = nn.Linear(hidden_dim, out_dim, bias=False)
        self.W2_sp = nn.Linear(hidden_dim, out_dim, bias=False)

        # Explicit CUDA Streams for True Parallelism
        self.stream_u = torch.cuda.Stream()
        self.stream_p = torch.cuda.Stream()
        self.stream_s = torch.cuda.Stream()

    def _forward_layer(self, X_input, epu_rp, epu_ci, epu_T_rp, epu_T_ci, eps_rp, eps_ci, eps_T_rp, eps_T_ci, euu_rp, euu_ci, layer=1):
        """Executes one single hop of Triple-Stream Graph Message Passing."""
        torch.cuda.synchronize() # Sync default stream before branching
        
        # FIX: Downcast input back to Float16 (Half) for the aggressively optimized CUDA kernel
        X_input = X_input.half()
        
        # Select appropriate weights for Layer 1 or Layer 2
        W_pu = self.W1_pu if layer == 1 else self.W2_pu
        W_uu = self.W1_uu if layer == 1 else self.W2_uu
        W_up = self.W1_up if layer == 1 else self.W2_up
        W_ps = self.W1_ps if layer == 1 else self.W2_ps
        W_sp = self.W1_sp if layer == 1 else self.W2_sp

        # Stream 1: User Node Representation
        with torch.cuda.stream(self.stream_u):
            raw_pu = custom_spmm.forward(epu_rp, epu_ci, X_input)
            raw_uu = custom_spmm.forward(euu_rp, euu_ci, X_input)
            H_user = F.relu(W_pu(raw_pu) + W_uu(raw_uu))

        # Stream 2: Product Node Representation
        with torch.cuda.stream(self.stream_p):
            raw_up = custom_spmm.forward(epu_T_rp, epu_T_ci, X_input)
            raw_ps = custom_spmm.forward(eps_rp, eps_ci, X_input)
            H_product = F.relu(W_up(raw_up) + W_ps(raw_ps))

        # Stream 3: Seller Node Representation
        with torch.cuda.stream(self.stream_s):
            raw_sp = custom_spmm.forward(eps_T_rp, eps_T_ci, X_input)
            H_seller = F.relu(W_sp(raw_sp))

        # Wait for all 3 streams to finish parallel computations
        torch.cuda.synchronize()

        # Re-concatenate into unified [N_total, Dim] matrix for the next layer
        return torch.cat([H_user, H_product, H_seller], dim=0)

    def forward(self, X, *args):
        # Layer 1: 1st Hop Neighborhood Aggregation
        H1 = self._forward_layer(X, *args, layer=1)
        
        # Layer 2: 2nd Hop Neighborhood Aggregation -> Outputs final Latent Space Z (128-D)
        Z = self._forward_layer(H1, *args, layer=2)
        
        return Z

# ---------------------------------------------------------
# 3. Dynamic CSR Edge Sampler
# ---------------------------------------------------------
def sample_positive_edges(rp, ci, src_offset, num_samples):
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
    
    # Corrected Latent Dimension to 128 as per Proposal Sec II.B
    model = TripleStreamGAE(in_dim=128, hidden_dim=128, out_dim=128).cuda()
    
    # Optimizer specs exactly match Paper Sec V.A.4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nStarting Stage 1: Unsupervised 2-Layer Training...")
    t0 = time.time()
    
    # Epochs exactly match Paper Sec V.A.4
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        # 1. Encode: Get Latent Graph Representation Z using Custom CUDA
        Z = model(X, epu_rp, epu_ci, epu_T_rp, epu_T_ci, eps_rp, eps_ci, eps_T_rp, eps_T_ci, euu_rp, euu_ci)

        # 2. Sample True Edges from the CSR Graph (Reconstruction Targets)
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
    
    # Extract Anomaly Scores
    model.eval()
    with torch.no_grad():
        Z_final = model(X, epu_rp, epu_ci, epu_T_rp, epu_T_ci, eps_rp, eps_ci, eps_T_rp, eps_T_ci, euu_rp, euu_ci)
        anomaly_scores = torch.norm(Z_final - Z_final.mean(dim=0), dim=1).cpu().numpy()
        
    np.save('anomaly_scores.npy', anomaly_scores)
    print(f"Saved anomaly scores for {num_nodes} nodes to 'anomaly_scores.npy'")

if __name__ == "__main__":
    train_gae()