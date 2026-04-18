import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from torch.utils.checkpoint import checkpoint
import gc

# IMPORT YOUR CUSTOM CUDA EXTENSION
import warp_gat 

# ---------------------------------------------------------
# 1. DataLoader
# ---------------------------------------------------------
def load_graph_data_stage2():
    print("Loading Graph Topology from Binaries...")
    with open('data/node_counts.json', 'r') as f:
        counts = json.load(f)

    N_u = counts['users']
    N_p = counts['products']
    N_s = counts['sellers']
    num_nodes = counts['total']

    def load_and_sanitize_csr(prefix, valid_min, valid_max):
        """
        Loads the CSR arrays and strips out toxic self-loops injected by Stage 1 
        that violate the bipartite graph boundaries of Stage 2.
        """
        rp = torch.from_numpy(np.fromfile(f"data/{prefix}_row_ptr.bin", dtype=np.int32))
        ci = torch.from_numpy(np.fromfile(f"data/{prefix}_col_idx.bin", dtype=np.int32))
        
        # Identify edges that correctly belong to the target node type
        mask = (ci >= valid_min) & (ci <= valid_max)
        
        if not mask.all():
            # Filter out invalid edges (like cross-type self loops)
            clean_ci = ci[mask]
            
            # Reconstruct the row_ptr array to account for the deleted edges
            row_lengths = rp[1:] - rp[:-1]
            row_indices = torch.repeat_interleave(torch.arange(len(rp) - 1), row_lengths)
            valid_rows = row_indices[mask]
            
            new_row_lengths = torch.bincount(valid_rows, minlength=len(rp) - 1).to(torch.int32)
            clean_rp = torch.zeros(len(rp), dtype=torch.int32)
            clean_rp[1:] = torch.cumsum(new_row_lengths, dim=0)
            
            return clean_rp, clean_ci
            
        return rp, ci 

    # Strictly enforce boundaries: Users (0 to N_u-1), Products (N_u to N_u+N_p-1), Sellers (N_u+N_p to End)
    epu_rp, epu_ci     = load_and_sanitize_csr('epu',   valid_min=N_u,         valid_max=N_u + N_p - 1)
    epu_T_rp, epu_T_ci = load_and_sanitize_csr('epu_T', valid_min=0,           valid_max=N_u - 1)
    eps_rp, eps_ci     = load_and_sanitize_csr('eps',   valid_min=N_u + N_p,   valid_max=num_nodes - 1)
    eps_T_rp, eps_T_ci = load_and_sanitize_csr('eps_T', valid_min=N_u,         valid_max=N_u + N_p - 1)
    euu_rp, euu_ci     = load_and_sanitize_csr('euu',   valid_min=0,           valid_max=N_u - 1)

    print("Loading Stage 1 embeddings (Z) as node features...")
    Z_np = np.load('Z_embeddings_stage1.npy')                         
    Z_tensor = torch.from_numpy(Z_np).float()                   

    try:
        labels_np = np.load('data/labels.npy')
    except FileNotFoundError:
        labels_np = np.load('labels.npy')
                                          
    labeled_idx  = torch.from_numpy(labels_np[:, 0]).long()
    labeled_y    = torch.from_numpy(labels_np[:, 1]).float()

    return (Z_tensor, N_u, N_p, N_s, num_nodes,
            epu_rp, epu_ci, epu_T_rp, epu_T_ci,
            eps_rp, eps_ci, eps_T_rp, eps_T_ci,
            euu_rp, euu_ci,
            labeled_idx, labeled_y)
# ---------------------------------------------------------
# 2. Hybrid Type-Specific GAT Layer (CUDA + Native)
# ---------------------------------------------------------
class TypeSpecificGATLayer(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, leaky_slope=0.2):
        super().__init__()
        self.leaky_slope = leaky_slope

        self.W_pu = nn.Linear(in_dim, out_dim, bias=False)
        self.W_ps = nn.Linear(in_dim, out_dim, bias=False)
        self.W_uu = nn.Linear(in_dim, out_dim, bias=False)

        self.a_pu = nn.Parameter(torch.empty(2 * out_dim))
        self.a_ps = nn.Parameter(torch.empty(2 * out_dim))
        self.a_uu = nn.Parameter(torch.empty(2 * out_dim))

        nn.init.xavier_uniform_(self.W_pu.weight)
        nn.init.xavier_uniform_(self.W_ps.weight)
        nn.init.xavier_uniform_(self.W_uu.weight)
        nn.init.xavier_normal_(self.a_pu.unsqueeze(0))
        nn.init.xavier_normal_(self.a_ps.unsqueeze(0))
        nn.init.xavier_normal_(self.a_uu.unsqueeze(0))

    @staticmethod
    def _native_sparse_agg(row_ptr, col_idx, H_src_local, H_dst_local, a_vec, leaky_slope, src_offset, is_training, chunk_size=2500):
        """
        Micro-chunked Message-Passing with Aggressive DropEdge and Intra-Loop GC.
        Engineered specifically to survive the 11GB VRAM ceiling.
        """
        with torch.amp.autocast('cuda', enabled=False): 
            a = a_vec.float() 
            N_dst, dim = len(row_ptr) - 1, H_src_local.shape[1]
           
            H_dst_proj = (H_dst_local * a[:dim].half()).sum(dim=-1).float()
            H_src_proj = (H_src_local * a[dim:].half()).sum(dim=-1).float()
           
            chunks = [] 
           
            # FIX 1: Micro-chunking down to 2500 nodes to avoid super-node spikes
            for start in range(0, N_dst, chunk_size):
                end = min(start + chunk_size, N_dst)
                edge_start = row_ptr[start].item()
                edge_end = row_ptr[end].item()

                if edge_start == edge_end: 
                    chunks.append(torch.zeros((end - start, dim), device=H_src_local.device, dtype=H_src_local.dtype))
                    continue

                r_lengths = row_ptr[start+1:end+1] - row_ptr[start:end]
                r_indices = torch.repeat_interleave(torch.arange(end - start, device=row_ptr.device), r_lengths)
                c_indices = col_idx[edge_start:edge_end].long() - src_offset
               
                # FIX 2: Aggressive DropEdge. Dropping 85% prevents over-smoothing and saves massive VRAM.
                if is_training:
                    keep_mask = torch.rand(len(r_indices), device=r_indices.device) > 0.85
                    r_indices = r_indices[keep_mask]
                    c_indices = c_indices[keep_mask]
                    
                    if len(r_indices) == 0:
                        chunks.append(torch.zeros((end - start, dim), device=H_src_local.device, dtype=H_src_local.dtype))
                        continue

                attn_dst = H_dst_proj[start:end][r_indices]
                attn_src = H_src_proj[c_indices]
               
                e_ij = F.leaky_relu(attn_dst + attn_src, negative_slope=leaky_slope)
                e_ij_max = e_ij.max().detach()
                exp_e = torch.exp(e_ij - e_ij_max)
                
                exp_sum = torch.zeros(end - start, device=exp_e.device, dtype=exp_e.dtype)
                exp_sum.scatter_add_(0, r_indices, exp_e)
                
                alpha = (exp_e / (exp_sum[r_indices] + 1e-16)).half()
                
                src_features = H_src_local[c_indices] 
                weighted_messages = src_features * alpha.unsqueeze(1) 
                
                chunk_out = torch.zeros((end - start, dim), device=H_src_local.device, dtype=H_src_local.dtype)
                chunk_out.index_add_(0, r_indices, weighted_messages)
                
                chunks.append(chunk_out)
                
                # FIX 3: Explicit Intra-Loop Garbage Collection
                # Forces Autograd to release the VRAM immediately before moving to the next chunk
                del r_indices, c_indices, attn_dst, attn_src, e_ij, exp_e, exp_sum, alpha, src_features, weighted_messages

            return torch.cat(chunks, dim=0)
            
    def forward(self, H, epu_rp, epu_ci, epu_T_rp, epu_T_ci, eps_rp, eps_ci, eps_T_rp, eps_T_ci, euu_rp, euu_ci, N_u, N_p, N_s):
        H_u = H[:N_u]
        H_p = H[N_u: N_u + N_p]
        H_s = H[N_u + N_p:]

        # 1. Initialize Accumulators
        H_u_out = torch.zeros_like(H_u)
        H_p_out = torch.zeros_like(H_p)
        H_s_out = torch.zeros_like(H_s)

        # --------------------------------------------------
        # STREAM 1: Purchase Edges
        # --------------------------------------------------
        Wh_u_pu = self.W_pu(H_u).contiguous()
        Wh_p_pu = self.W_pu(H_p).contiguous()
        
        if self.training:
            H_u_out += checkpoint(self._native_sparse_agg, epu_rp, epu_ci, Wh_p_pu, Wh_u_pu, self.a_pu, self.leaky_slope, torch.tensor(N_u), self.training, use_reentrant=False)
            H_p_out += checkpoint(self._native_sparse_agg, epu_T_rp, epu_T_ci, Wh_u_pu, Wh_p_pu, self.a_pu, self.leaky_slope, torch.tensor(0), self.training, use_reentrant=False)
        else:
            H_u_out += warp_gat.forward(epu_rp, epu_ci, Wh_p_pu.float(), Wh_u_pu.float(), self.a_pu.float(), self.leaky_slope, N_u).to(H_u_out.dtype)
            H_p_out += warp_gat.forward(epu_T_rp, epu_T_ci, Wh_u_pu.float(), Wh_p_pu.float(), self.a_pu.float(), self.leaky_slope, 0).to(H_p_out.dtype)
            
        del Wh_u_pu, Wh_p_pu 

        # --------------------------------------------------
        # STREAM 2: Selling Edges
        # --------------------------------------------------
        Wh_p_ps = self.W_ps(H_p).contiguous()
        Wh_s_ps = self.W_ps(H_s).contiguous()
        
        if self.training:
            H_p_out += checkpoint(self._native_sparse_agg, eps_rp, eps_ci, Wh_s_ps, Wh_p_ps, self.a_ps, self.leaky_slope, torch.tensor(N_u + N_p), self.training, use_reentrant=False)
            H_s_out += checkpoint(self._native_sparse_agg, eps_T_rp, eps_T_ci, Wh_p_ps, Wh_s_ps, self.a_ps, self.leaky_slope, torch.tensor(N_u), self.training, use_reentrant=False)
        else:
            H_p_out += warp_gat.forward(eps_rp, eps_ci, Wh_s_ps.float(), Wh_p_ps.float(), self.a_ps.float(), self.leaky_slope, N_u + N_p).to(H_p_out.dtype)
            H_s_out += warp_gat.forward(eps_T_rp, eps_T_ci, Wh_p_ps.float(), Wh_s_ps.float(), self.a_ps.float(), self.leaky_slope, N_u).to(H_s_out.dtype)
            
        del Wh_p_ps, Wh_s_ps 

        # --------------------------------------------------
        # STREAM 3: User Edges
        # --------------------------------------------------
        Wh_u_uu = self.W_uu(H_u).contiguous()
        
        if self.training:
            H_u_out += checkpoint(self._native_sparse_agg, euu_rp, euu_ci, Wh_u_uu, Wh_u_uu, self.a_uu, self.leaky_slope, torch.tensor(0), self.training, use_reentrant=False)
        else:
            H_u_out += warp_gat.forward(euu_rp, euu_ci, Wh_u_uu.float(), Wh_u_uu.float(), self.a_uu.float(), self.leaky_slope, 0).to(H_u_out.dtype)
            
        del Wh_u_uu

        # 2. In-Place Activations
        H_u_out = F.elu(H_u_out / 2.0, inplace=True)
        H_p_out = F.elu(H_p_out / 2.0, inplace=True)
        H_s_out = F.elu(H_s_out, inplace=True)

        return torch.cat([H_u_out, H_p_out, H_s_out], dim=0)
        
        
# ---------------------------------------------------------
# 3. Two-Layer GAT + Anomaly Scoring Head
# ---------------------------------------------------------
class TripleStreamGAT(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128):
        super().__init__()
        self.gat1 = TypeSpecificGATLayer(in_dim, hidden_dim)
        self.gat2 = TypeSpecificGATLayer(hidden_dim, out_dim)
        self.score_head = nn.Linear(out_dim, 1, bias=True)

    def forward(self, H, *graph_args):
        H1 = self.gat1(H, *graph_args)
        H2 = self.gat2(H1, *graph_args)
        logits = self.score_head(H2).squeeze(-1) 
        return logits, H2

# ---------------------------------------------------------
# 4. Loss Functions
# ---------------------------------------------------------
def supervised_loss(logits, labeled_idx, labeled_y):
    logits_labeled = logits[labeled_idx]
    return F.binary_cross_entropy_with_logits(logits_labeled, labeled_y)

def unsupervised_loss(logits, epu_rp, epu_ci, eps_rp, eps_ci, euu_rp, euu_ci, N_u, num_samples=5000):
    # BUG FIX: Apply sigmoid to convert raw logits into bounded anomaly scores (s_i).
    # Equation 17 requires the difference between scores (0 to 1), not unbounded logits.
    scores = torch.sigmoid(logits)
    
    def edge_loss(rp, ci, src_offset):
        row_lengths = rp[1:] - rp[:-1]
        src = torch.repeat_interleave(torch.arange(len(rp) - 1, device=rp.device), row_lengths) + src_offset
        dst = ci.long()
        if len(src) == 0: return torch.tensor(0.0, device=scores.device)
        
        idx = torch.randint(0, len(src), (min(num_samples, len(src)),), device=rp.device)
        
        # Calculate the squared difference using the sigmoid scores
        diff = scores[src[idx]] - scores[dst[idx]]
        return (diff ** 2).mean()

    total_loss = (edge_loss(epu_rp, epu_ci, src_offset=0)
               + edge_loss(eps_rp, eps_ci, src_offset=N_u)
               + edge_loss(euu_rp, euu_ci, src_offset=0))
               
    return total_loss / 3.0

def split_labels(labeled_idx, labeled_y, train_ratio=0.6, val_ratio=0.2, seed=42):
    torch.manual_seed(seed)
    n = len(labeled_idx)
    perm = torch.randperm(n)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    tr = perm[:n_train]
    va = perm[n_train: n_train + n_val]
    te = perm[n_train + n_val:]
    return (labeled_idx[tr], labeled_y[tr], labeled_idx[va], labeled_y[va], labeled_idx[te], labeled_y[te])

def compute_auc_roc(scores_np, labels_np):
    order = np.argsort(-scores_np)
    labels_sorted = labels_np[order]
    n_pos = labels_np.sum()
    n_neg = len(labels_np) - n_pos
    if n_pos == 0 or n_neg == 0: return float('nan')
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapezoid(tpr, fpr))

# ---------------------------------------------------------
# 5. Main Training Loop
# ---------------------------------------------------------
def train_gat():
    (Z, N_u, N_p, N_s, num_nodes,
     epu_rp, epu_ci, epu_T_rp, epu_T_ci,
     eps_rp, eps_ci, eps_T_rp, eps_T_ci,
     euu_rp, euu_ci, labeled_idx, labeled_y) = load_graph_data_stage2()

    graph_args = (epu_rp.cuda(), epu_ci.cuda(), epu_T_rp.cuda(), epu_T_ci.cuda(),
                  eps_rp.cuda(), eps_ci.cuda(), eps_T_rp.cuda(), eps_T_ci.cuda(),
                  euu_rp.cuda(), euu_ci.cuda(), N_u, N_p, N_s)
    Z = Z.cuda()

    (tr_idx, tr_y, va_idx, va_y, te_idx, te_y) = split_labels(labeled_idx, labeled_y)
    
    model = TripleStreamGAT(in_dim=128, hidden_dim=128, out_dim=128).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # RE-ENABLE MIXED PRECISION TO SAVE VRAM
    scaler = torch.amp.GradScaler('cuda') 
   
    LAMBDA, EPOCHS, best_val_auc = 0.5, 100, 0.0
    best_state = None

    print("\nStarting Stage 2: Semi-Supervised GAT Fine-Tuning (Hybrid Mode)...")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train() # Engages Native PyTorch Math for Autograd
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            logits, _ = model(Z, *graph_args)
            l_sup = supervised_loss(logits, tr_idx.cuda(), tr_y.cuda())
            l_unsup = unsupervised_loss(logits, epu_rp.cuda(), epu_ci.cuda(), eps_rp.cuda(), eps_ci.cuda(), euu_rp.cuda(), euu_ci.cuda(), N_u)
            loss = l_sup + LAMBDA * l_unsup

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 1. Save the scalar values for logging BEFORE deleting the massive tensors
        log_sup = l_sup.item()
        log_unsup = l_unsup.item()
        log_loss = loss.item()

        # 2. Obliterate the computational graph from VRAM
        del logits, loss, l_sup, l_unsup
        
        if epoch % 10 == 0:
            model.eval() # Automatically engages custom `warp_gat` CUDA kernel
            with torch.no_grad():
                # Temporarily cast validation inputs for the C++ kernel
                val_logits, _ = model(Z, *graph_args)
                
                val_s = torch.sigmoid(val_logits[va_idx.cuda()]).cpu().numpy()
                val_l = va_y.cpu().numpy()
                val_auc = compute_auc_roc(val_s, val_l)
                
                # Free validation memory
                del val_logits

            # 3. Print using the saved scalars!
            print(f"Epoch {epoch:03d} | L_sup: {log_sup:.4f} | L_unsup: {log_unsup:.4f} | L_total: {log_loss:.4f} | Val AUC-ROC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                
        # 4. Sweep the fragmentation out of the GPU
        torch.cuda.empty_cache()

    print(f"\nTraining Complete in {time.time() - t0:.2f}s | Best Val AUC-ROC: {best_val_auc:.4f}")

    print("\nEvaluating best checkpoint on held-out test set with Custom CUDA Kernel...")
    model.load_state_dict(best_state)
    model.eval() 
    with torch.no_grad():
        test_logits, Z_final = model(Z, *graph_args)
        te_s = torch.sigmoid(test_logits[te_idx.cuda()]).cpu().numpy()
        te_l = te_y.cpu().numpy()
        test_auc = compute_auc_roc(te_s, te_l)

    print(f"Test AUC-ROC: {test_auc:.4f}")

    all_scores = torch.sigmoid(test_logits).cpu().numpy()
    np.save('anomaly_scores_stage2.npy', all_scores)
    np.save('Z_stage2.npy', Z_final.cpu().numpy())
    torch.save(best_state, 'gat_stage2_best.pt')

if __name__ == "__main__":
    train_gat()
