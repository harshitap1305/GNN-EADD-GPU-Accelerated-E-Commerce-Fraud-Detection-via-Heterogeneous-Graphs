#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------
// Warp-Aligned CSR Sparse Matrix Multiplication (SpMM)
// Optimized strictly for 128-Dimension Float16 Features
// ---------------------------------------------------------
__global__ void csr_spmm_warp_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
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
        int degree = row_end - row_start;
        
        // Registers for accumulation. Since Dim=128, each thread computes 4 dimensions (32*4 = 128)
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Loop over neighbors
        for (int i = row_start; i < row_end; ++i) {
            int neighbor = col_idx[i];
            
            // Fetch Float16 and accumulate into Float32
            #pragma unroll
            for(int d = 0; d < 4; ++d) {
                int col = lane_id + d * 32;
                acc[d] += __half2float(in_features[neighbor * dim + col]);
            }
        }

        // Write back to Global Memory (Normalized by Degree for GCN mathematically)
        float norm = degree > 0 ? 1.0f / (float)degree : 0.0f;
        #pragma unroll
        for(int d = 0; d < 4; ++d) {
            int col = lane_id + d * 32;
            out_features[warp_id * dim + col] = acc[d] * norm;
        }
    }
}

// ---------------------------------------------------------
// PyTorch C++ Binding
// ---------------------------------------------------------
torch::Tensor spmm_cuda(torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor in_features) {
    int num_nodes = row_ptr.size(0) - 1;
    int dim = in_features.size(1);
    
    // Output is upgraded to Float32 for stable Neural Network Gradients
    auto out_features = torch::zeros({num_nodes, dim}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (num_nodes + warps_per_block - 1) / warps_per_block;

    csr_spmm_warp_kernel<<<blocks, threads_per_block>>>(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        reinterpret_cast<half*>(in_features.data_ptr<at::Half>()),
        out_features.data_ptr<float>(),
        num_nodes,
        dim
    );

    return out_features;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &spmm_cuda, "Warp-Optimized CSR SpMM Forward (CUDA)");
}
