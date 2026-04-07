import os
# Optimizes PyTorch VRAM allocator to prevent fragmentation on RTX 3060/4060
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time

# Import our custom CUDA kernel!
import custom_spmm 

# ---------------------------------------------------------
# AUTOGRAD WRAPPERS: Bridging Custom C++ and PyTorch Gradients
# ---------------------------------------------------------
class SpMM_Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rp, ci, edge_weights, X):
        ctx.save_for_backward(rp, ci, edge_weights)
        ctx.x_shape = X.shape
        ctx.x_dtype = X.dtype
        return custom_spmm.forward(rp, ci, edge_weights, X)

    @staticmethod
    def backward(ctx, grad_output):
        """Extreme VRAM-Safe Chunk Streaming Backprop"""
        rp, ci, edge_weights = ctx.saved_tensors
        N = len(rp) - 1
        
        grad_X_fp32 = torch.zeros(ctx.x_shape, device=grad_output.device, dtype=torch.float32)
        CHUNK_SIZE = 5000 
        
        for start_idx in range(0, N, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, N)
            
            sub_rp = rp[start_idx : end_idx + 1]
            sub_row_lengths = sub_rp[1:] - sub_rp[:-1]
            
            if sub_row_lengths.sum() == 0:
                continue
                
            start_edge = sub_rp[0].item()
            end_edge = sub_rp[-1].item()
            sub_ci = ci[start_edge : end_edge].long()
            sub_weights = edge_weights[start_edge : end_edge].unsqueeze(1)
            
            local_grad = grad_output[start_idx:end_idx]
            expanded_grad = torch.repeat_interleave(local_grad, sub_row_lengths, dim=0)
            
            scaled_grad = (expanded_grad * sub_weights).to(torch.float32)
            grad_X_fp32.scatter_add_(0, sub_ci.unsqueeze(1).expand_as(scaled_grad), scaled_grad)
            
        return None, None, None, grad_X_fp32.to(ctx.x_dtype)

# ---------------------------------------------------------
# 1. Memory-Mapped DataLoader & Normalization
# ---------------------------------------------------------
def compute_symmetric_edge_weights(rp, ci, global_deg, src_offset):
    """Computes exact GCN symmetric normalization mapping directly via Global IDs"""
    row_lengths = rp[1:] - rp[:-1]
    src_local = torch.repeat_interleave(torch.arange(len(rp)-1, device=rp.device), row_lengths)
    src_global = src_local + src_offset
    
    d_s = torch.clamp(global_deg[src_global], min=1.0)
    d_d = torch.clamp(global_deg[ci], min=1.0) 
    
    return (1.0 / torch.sqrt(d_s * d_d)).float()

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

    epu_rp, epu_ci = load_csr('epu')
    epu_T_rp, epu_T_ci = load_csr('epu_T')
    
    eps_rp, eps_ci = load_csr('eps')
    eps_T_rp, eps_T_ci = load_csr('eps_T')
    
    euu_rp, euu_ci = load_csr('euu')

    print("Computing Strict GCN Symmetric Normalization...")
    deg_epu = torch.zeros(num_nodes, device='cuda')
    deg_epu[:N_u] = (epu_rp[1:] - epu_rp[:-1]).float()
    deg_epu[N_u:N_u+N_p] = (epu_T_rp[1:] - epu_T_rp[:-1]).float()

    deg_eps = torch.zeros(num_nodes, device='cuda')
    deg_eps[N_u:N_u+N_p] = (eps_rp[1:] - eps_rp[:-1]).float()
    deg_eps[N_u+N_p:] = (eps_T_rp[1:] - eps_T_rp[:-1]).float()

    deg_euu = torch.zeros(num_nodes, device='cuda')
    deg_euu[:N_u] = (euu_rp[1:] - euu_rp[:-1]).float()

    epu_w = compute_symmetric_edge_weights(epu_rp, epu_ci, deg_epu, src_offset=0)
    epu_T_w = compute_symmetric_edge_weights(epu_T_rp, epu_T_ci, deg_epu, src_offset=N_u)
    
    eps_w = compute_symmetric_edge_weights(eps_rp, eps_ci, deg_eps, src_offset=N_u)
    eps_T_w = compute_symmetric_edge_weights(eps_T_rp, eps_T_ci, deg_eps, src_offset=N_u + N_p)
    
    euu_w = compute_symmetric_edge_weights(euu_rp, euu_ci, deg_euu, src_offset=0)

    print("Mapping 128-D Float16 Features directly from SSD...")
    X_memmap = np.memmap('X_combined.memmap', dtype='float16', mode='r', shape=(num_nodes, 128))
    X_tensor = torch.from_numpy(np.array(X_memmap)).cuda() 

    return X_tensor, N_u, N_p, N_s, num_nodes, \
           epu_rp, epu_ci, epu_w, epu_T_rp, epu_T_ci, epu_T_w, \
           eps_rp, eps_ci, eps_w, eps_T_rp, eps_T_ci, eps_T_w, \
           euu_rp, euu_ci, euu_w

# ---------------------------------------------------------
# 2. Triple-Stream GAE Architecture (2-Layer, 128-D)
# ---------------------------------------------------------
class TripleStreamGAE(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128, N_u=0, N_p=0):
        super().__init__()
        self.N_u = N_u
        self.N_p = N_p

        # Type-specific relation weights (with bias per paper Eq. 6)
        self.W1_pu = nn.Linear(in_dim, hidden_dim, bias=True)
        self.W1_uu = nn.Linear(in_dim, hidden_dim, bias=True)
        self.W1_up = nn.Linear(in_dim, hidden_dim, bias=True)
        self.W1_ps = nn.Linear(in_dim, hidden_dim, bias=True)
        self.W1_sp = nn.Linear(in_dim, hidden_dim, bias=True)

        self.W2_pu = nn.Linear(hidden_dim, out_dim, bias=True)
        self.W2_uu = nn.Linear(hidden_dim, out_dim, bias=True)
        self.W2_up = nn.Linear(hidden_dim, out_dim, bias=True)
        self.W2_ps = nn.Linear(hidden_dim, out_dim, bias=True)
        self.W2_sp = nn.Linear(hidden_dim, out_dim, bias=True)

        # Self-loop weight matrices (paper Section III.A.4: preserve own features)
        self.W1_self_u = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W1_self_p = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W1_self_s = nn.Linear(in_dim, hidden_dim, bias=False)

        self.W2_self_u = nn.Linear(hidden_dim, out_dim, bias=False)
        self.W2_self_p = nn.Linear(hidden_dim, out_dim, bias=False)
        self.W2_self_s = nn.Linear(hidden_dim, out_dim, bias=False)

        self.stream_u = torch.cuda.Stream()
        self.stream_p = torch.cuda.Stream()
        self.stream_s = torch.cuda.Stream()

    def _forward_layer(self, X_input, 
                       epu_rp, epu_ci, epu_w, epu_T_rp, epu_T_ci, epu_T_w, 
                       eps_rp, eps_ci, eps_w, eps_T_rp, eps_T_ci, eps_T_w, 
                       euu_rp, euu_ci, euu_w, layer=1):
        
        W_pu = self.W1_pu if layer == 1 else self.W2_pu
        W_uu = self.W1_uu if layer == 1 else self.W2_uu
        W_up = self.W1_up if layer == 1 else self.W2_up
        W_ps = self.W1_ps if layer == 1 else self.W2_ps
        W_sp = self.W1_sp if layer == 1 else self.W2_sp

        # Self-loop weights per node type (paper Section III.A.4)
        W_self_u = self.W1_self_u if layer == 1 else self.W2_self_u
        W_self_p = self.W1_self_p if layer == 1 else self.W2_self_p
        W_self_s = self.W1_self_s if layer == 1 else self.W2_self_s

        current_stream = torch.cuda.current_stream()

        self.stream_u.wait_stream(current_stream)
        self.stream_p.wait_stream(current_stream)
        self.stream_s.wait_stream(current_stream)

        X_half = X_input.half()

        # Slice node features by type for self-loop connections
        X_user = X_input[:self.N_u]
        X_product = X_input[self.N_u:self.N_u + self.N_p]
        X_seller = X_input[self.N_u + self.N_p:]

        with torch.cuda.stream(self.stream_u):
            raw_pu = SpMM_Autograd.apply(epu_rp, epu_ci, epu_w, X_half)
            raw_uu = SpMM_Autograd.apply(euu_rp, euu_ci, euu_w, X_half)
            # Paper Eq. 7: Sum over relation types + self-loop (no averaging)
            H_user = W_pu(raw_pu) + W_uu(raw_uu) + W_self_u(X_user)
            if layer == 1: H_user = F.relu(H_user)

        with torch.cuda.stream(self.stream_p):
            raw_up = SpMM_Autograd.apply(epu_T_rp, epu_T_ci, epu_T_w, X_half)
            raw_ps = SpMM_Autograd.apply(eps_rp, eps_ci, eps_w, X_half)
            # Paper Eq. 7: Sum over relation types + self-loop (no averaging)
            H_product = W_up(raw_up) + W_ps(raw_ps) + W_self_p(X_product)
            if layer == 1: H_product = F.relu(H_product)

        with torch.cuda.stream(self.stream_s):
            raw_sp = SpMM_Autograd.apply(eps_T_rp, eps_T_ci, eps_T_w, X_half)
            # Self-loop for sellers
            H_seller = W_sp(raw_sp) + W_self_s(X_seller)
            if layer == 1: H_seller = F.relu(H_seller)

        current_stream.wait_stream(self.stream_u)
        current_stream.wait_stream(self.stream_p)
        current_stream.wait_stream(self.stream_s)
        
        return torch.cat([H_user, H_product, H_seller], dim=0)

    def forward(self, X, *args):
        H1 = self._forward_layer(X, *args, layer=1)
        Z = self._forward_layer(H1, *args, layer=2)
        return Z

# ---------------------------------------------------------
# 3. Dynamic CSR Edge Sampler (Fixed: No Self-Loops)
# ---------------------------------------------------------
def sample_positive_edges(rp, ci, src_offset, num_samples):
    row_lengths = rp[1:] - rp[:-1]
    src = torch.repeat_interleave(torch.arange(len(rp)-1, device=rp.device), row_lengths) + src_offset
    dst = ci
    
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    if len(src) == 0:
        return torch.empty(0, dtype=torch.long, device=rp.device), torch.empty(0, dtype=torch.long, device=rp.device)
        
    idx = torch.randint(0, len(src), (min(num_samples, len(src)),), device=rp.device)
    return src[idx].long(), dst[idx].long()

# ---------------------------------------------------------
# 4. Fast Edge Reconstruction & Training
# ---------------------------------------------------------
def train_gae():
    X, N_u, N_p, N_s, num_nodes, \
    epu_rp, epu_ci, epu_w, epu_T_rp, epu_T_ci, epu_T_w, \
    eps_rp, eps_ci, eps_w, eps_T_rp, eps_T_ci, eps_T_w, \
    euu_rp, euu_ci, euu_w = load_graph_data()
    
    model = TripleStreamGAE(in_dim=128, hidden_dim=128, out_dim=128, N_u=N_u, N_p=N_p).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')

    print("\nStarting Stage 1: Unsupervised 2-Layer Training...")
    t0 = time.time()
    
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        # The Forward Pass runs in ultra-fast FP16 Mixed Precision
        with torch.amp.autocast('cuda'):
            Z = model(X, epu_rp, epu_ci, epu_w, epu_T_rp, epu_T_ci, epu_T_w, 
                         eps_rp, eps_ci, eps_w, eps_T_rp, eps_T_ci, eps_T_w, 
                         euu_rp, euu_ci, euu_w)

        # FIX: Pull the Dot Product Decoder OUT of Mixed Precision.
        # Forces the dot product to compute in FP32, making it impossible to overflow!
        Z = Z.float()

        pu_src, pu_dst = sample_positive_edges(epu_rp, epu_ci, src_offset=0, num_samples=30000)
        ps_src, ps_dst = sample_positive_edges(eps_rp, eps_ci, src_offset=N_u, num_samples=30000)
        uu_src, uu_dst = sample_positive_edges(euu_rp, euu_ci, src_offset=0, num_samples=30000)

        pos_src = torch.cat([pu_src, ps_src, uu_src])
        pos_dst = torch.cat([pu_dst, ps_dst, uu_dst])
        
        pos_score = (Z[pos_src] * Z[pos_dst]).sum(dim=-1)
        
        neg_dst = torch.randint(0, num_nodes, (len(pos_src),), device='cuda').long()
        neg_score = (Z[pos_src] * Z[neg_dst]).sum(dim=-1)

        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        preds = torch.cat([pos_score, neg_score])
        
        # Loss calculation is now guaranteed stable FP32
        loss = F.binary_cross_entropy_with_logits(preds, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 10 == 0:
            with torch.no_grad():
                pos_prob = torch.sigmoid(pos_score).mean().item()
                neg_prob = torch.sigmoid(neg_score).mean().item()
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | P(pos): {pos_prob:.3f} | P(neg): {neg_prob:.3f}")

    print(f"\nTraining Complete in {time.time() - t0:.2f}s!")
    
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        Z_final = model(X, epu_rp, epu_ci, epu_w, epu_T_rp, epu_T_ci, epu_T_w, 
                           eps_rp, eps_ci, eps_w, eps_T_rp, eps_T_ci, eps_T_w, 
                           euu_rp, euu_ci, euu_w)
        
    np.save('Z_embeddings_stage1.npy', Z_final.cpu().numpy())
    print(f"Saved latent embeddings (Z) for {num_nodes} nodes to 'Z_embeddings_stage1.npy'")
    print("Ready for Stage 2 (GAT Fine-tuning)!")

if __name__ == "__main__":
    train_gae()