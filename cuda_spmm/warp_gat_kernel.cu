#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define FULL_MASK 0xffffffff

// Warp-level reduction for maximum (numerical stability)
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(FULL_MASK, val, offset));
    return val;
}

// Warp-level reduction for sum (softmax denominator)
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// Warp-level GAT Aggregation Kernel
__global__ void warp_gat_forward_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ H_src,
    const float* __restrict__ H_dst,
    const float* __restrict__ a_src,
    const float* __restrict__ a_dst,
    float* __restrict__ out,
    const int num_dst_nodes,
    const int dim,
    const float leaky_slope,
    const int src_offset) 
{
    // One warp (32 threads) per destination node
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= num_dst_nodes) return;

    int u = warp_id; // Destination node
    int start_idx = row_ptr[u];
    int end_idx = row_ptr[u + 1];

    if (start_idx == end_idx) return; // No neighbors

    // 1. Compute Destination Node Projection (done by lane 0 to save compute)
    float dst_proj = 0.0f;
    for (int d = lane_id; d < dim; d += 32) {
        dst_proj += H_dst[u * dim + d] * a_dst[d];
    }
    dst_proj = warpReduceSum(dst_proj);
    dst_proj = __shfl_sync(FULL_MASK, dst_proj, 0);

    // 2. Compute Attention Logits and find Max (for numerical stability)
    float max_e = -1e20f;
    
    // Each thread processes a subset of neighbors
    for (int i = start_idx + lane_id; i < end_idx; i += 32) {
        int v = col_idx[i] - src_offset; // Local source index
        float src_proj = 0.0f;
        for (int d = 0; d < dim; ++d) { // Simplified loop for dot product
            src_proj += H_src[v * dim + d] * a_src[d];
        }
        
        float e_uv = dst_proj + src_proj;
        e_uv = e_uv > 0 ? e_uv : leaky_slope * e_uv; // LeakyReLU
        max_e = max(max_e, e_uv);
    }
    max_e = warpReduceMax(max_e);
    max_e = __shfl_sync(FULL_MASK, max_e, 0);

    // 3. Compute Softmax Denominator
    float exp_sum = 0.0f;
    for (int i = start_idx + lane_id; i < end_idx; i += 32) {
        int v = col_idx[i] - src_offset;
        float src_proj = 0.0f;
        for (int d = 0; d < dim; ++d) {
            src_proj += H_src[v * dim + d] * a_src[d];
        }
        float e_uv = dst_proj + src_proj;
        e_uv = e_uv > 0 ? e_uv : leaky_slope * e_uv;
        exp_sum += expf(e_uv - max_e);
    }
    exp_sum = warpReduceSum(exp_sum);
    exp_sum = __shfl_sync(FULL_MASK, exp_sum, 0);

    // 4. Feature Aggregation
    for (int d = lane_id; d < dim; d += 32) {
        float agg_feature = 0.0f;
        for (int i = start_idx; i < end_idx; ++i) {
            int v = col_idx[i] - src_offset;
            float src_proj = 0.0f;
            for (int k = 0; k < dim; ++k) {
                src_proj += H_src[v * dim + k] * a_src[k];
            }
            float e_uv = dst_proj + src_proj;
            e_uv = e_uv > 0 ? e_uv : leaky_slope * e_uv;
            float alpha = expf(e_uv - max_e) / (exp_sum + 1e-16f);
            
            agg_feature += alpha * H_src[v * dim + d];
        }
        out[u * dim + d] = agg_feature;
    }
}

torch::Tensor warp_gat_forward(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor H_src,
    torch::Tensor H_dst,
    torch::Tensor a_vec,
    float leaky_slope,
    int src_offset) 
{
    int num_dst_nodes = row_ptr.size(0) - 1;
    int dim = H_src.size(1);
    auto out = torch::zeros_like(H_dst);

    int threads = 128; // 4 warps per block
    int blocks = (num_dst_nodes * 32 + threads - 1) / threads;

    // Split attention vector into src and dst halves
    auto a_dst = a_vec.slice(0, 0, dim);
    auto a_src = a_vec.slice(0, dim, 2 * dim);

    warp_gat_forward_kernel<<<blocks, threads>>>(
        row_ptr.data_ptr<int>(),
        col_idx.data_ptr<int>(),
        H_src.data_ptr<float>(),
        H_dst.data_ptr<float>(),
        a_src.data_ptr<float>(),
        a_dst.data_ptr<float>(),
        out.data_ptr<float>(),
        num_dst_nodes, dim, leaky_slope, src_offset
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &warp_gat_forward, "Warp GAT forward (CUDA)");
}
