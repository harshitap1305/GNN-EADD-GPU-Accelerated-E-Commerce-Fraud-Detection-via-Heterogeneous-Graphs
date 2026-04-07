#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
// This is new code changed
// =========================================================
// 1. Warp-Aligned CSR Sparse Matrix Multiplication (Encoder)
// Fixed: Accepts dynamic dimensions and pre-computed edge_weights
// =========================================================
__global__ void csr_spmm_warp_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ edge_weights, // NEW: Supports symmetric GCN normalization
    const half* __restrict__ in_features,
    float* __restrict__ out_features,
    int num_nodes,
    int dim) 
{
    // 1 Warp (32 threads) processes exactly 1 Node
    int warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane_id = threadIdx.x % 32;

    if (warp_id < num_nodes) {
        int row_start = row_ptr[warp_id];
        int row_end = row_ptr[warp_id + 1];
        
        // FIX: Dynamic dimensions. Supports up to 1024-D features
        const int MAX_CHUNKS = 32;
        float acc[MAX_CHUNKS];
        int num_chunks = (dim + 31) / 32;
        if (num_chunks > MAX_CHUNKS) num_chunks = MAX_CHUNKS;

        for(int c = 0; c < num_chunks; ++c) {
            acc[c] = 0.0f;
        }

        // Loop over neighbors
        for (int i = row_start; i < row_end; ++i) {
            int neighbor = col_idx[i];
            float weight = edge_weights[i]; // Apply pre-computed symmetric norm
            
            #pragma unroll 4
            for(int c = 0; c < num_chunks; ++c) {
                int col = lane_id + c * 32;
                if (col < dim) {
                    acc[c] += __half2float(in_features[neighbor * dim + col]) * weight;
                }
            }
        }

        // Write back to Global Memory
        #pragma unroll 4
        for(int c = 0; c < num_chunks; ++c) {
            int col = lane_id + c * 32;
            if (col < dim) {
                out_features[warp_id * dim + col] = acc[c];
            }
        }
    }
}

torch::Tensor spmm_cuda(torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor edge_weights, torch::Tensor in_features) {
    int num_nodes = row_ptr.size(0) - 1;
    int dim = in_features.size(1);
    
    auto out_features = torch::zeros({num_nodes, dim}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (num_nodes + warps_per_block - 1) / warps_per_block;

    csr_spmm_warp_kernel<<<blocks, threads_per_block>>>(
        row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(), edge_weights.data_ptr<float>(),
        reinterpret_cast<half*>(in_features.data_ptr<at::Half>()),
        out_features.data_ptr<float>(), num_nodes, dim
    );
    return out_features;
}

// =========================================================
// 2. Warp-Aligned Sparse Dot Product (Decoder)
// Fixed: Accepts dynamic dimensions
// =========================================================
__global__ void warp_sparse_dot_product_kernel(
    const float* __restrict__ Z,
    const int* __restrict__ src_idx,
    const int* __restrict__ dst_idx,
    float* __restrict__ out_scores,
    int num_edges,
    int dim) 
{
    int warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane_id = threadIdx.x % 32;

    if (warp_id < num_edges) {
        int u = src_idx[warp_id];
        int v = dst_idx[warp_id];
        
        float acc = 0.0f;
        int num_chunks = (dim + 31) / 32;
        
        // FIX: Dynamic Dimension support instead of hardcoded 4 loops
        #pragma unroll 4
        for(int c = 0; c < num_chunks; ++c) {
            int col = lane_id + c * 32;
            if (col < dim) {
                acc += Z[u * dim + col] * Z[v * dim + col];
            }
        }

        // Standard 32-thread Warp Reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }

        if (lane_id == 0) {
            out_scores[warp_id] = acc;
        }
    }
}

torch::Tensor decoder_forward(torch::Tensor Z, torch::Tensor src_idx, torch::Tensor dst_idx) {
    src_idx = src_idx.to(torch::kInt32);
    dst_idx = dst_idx.to(torch::kInt32);

    int num_edges = src_idx.size(0);
    int dim = Z.size(1); 
    
    auto out_scores = torch::zeros({num_edges}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (num_edges + warps_per_block - 1) / warps_per_block;

    warp_sparse_dot_product_kernel<<<blocks, threads_per_block>>>(
        Z.data_ptr<float>(), src_idx.data_ptr<int>(), dst_idx.data_ptr<int>(),
        out_scores.data_ptr<float>(), num_edges, dim
    );

    return out_scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &spmm_cuda, "Warp-Optimized CSR SpMM Forward (CUDA)");
    m.def("decoder_dot", &decoder_forward, "Warp-Aligned Sparse Dot Product (CUDA)");
}
