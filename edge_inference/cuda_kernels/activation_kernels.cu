#include "activation_kernels.h"
#include <cmath>
#include <iostream>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace arat {
namespace edge {
namespace cuda {

// CUDA Device Kernel for SwiGLU: Swish(x) * y
__global__ void swiglu_kernel(const float* x, const float* y, float* out, size_t N, float beta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x_val = x[idx];
        float y_val = y[idx];
        float swish = x_val / (1.0f + __expf(-beta * x_val));
        out[idx] = swish * y_val;
    }
}

// CUDA Device Kernel for GELU
__global__ void gelu_kernel(const float* x, float* out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x_val = x[idx];
        float inner = 0.79788456f * (x_val + 0.044715f * x_val * x_val * x_val);
        out[idx] = 0.5f * x_val * (1.0f + tanhf(inner));
    }
}

// CUDA Device Kernel for RMSNorm
__global__ void rmsnorm_kernel(const float* input, const float* gamma, float* output, size_t batch_size, size_t hidden_dim, float eps) {
    size_t row = blockIdx.x;
    if (row < batch_size) {
        const float* row_in = input + row * hidden_dim;
        float* row_out = output + row * hidden_dim;

        // Block-wide sum of squares
        float sum_sq = 0.0f;
        for (size_t d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
            float val = row_in[d];
            sum_sq += val * val;
        }

        // Shared memory reduction for sum_sq
        __shared__ float s_sum;
        if (threadIdx.x == 0) s_sum = 0.0f;
        __syncthreads();

        atomicAdd(&s_sum, sum_sq);
        __syncthreads();

        float rms = rsqrtf((s_sum / static_cast<float>(hidden_dim)) + eps);

        // Normalize and scale by gamma
        for (size_t d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
            row_out[d] = row_in[d] * rms * gamma[d];
        }
    }
}

// CUDA Device Kernel for RoPE
__global__ void rope_kernel(float* qk, size_t seq_len, size_t num_heads, size_t head_dim, size_t pos_offset) {
    size_t token_idx = blockIdx.x; // token position in sequence
    size_t head_idx = blockIdx.y;  // attention head
    size_t dim_pair = threadIdx.x * 2; // pair index: (dim_pair, dim_pair + 1)

    if (token_idx < seq_len && head_idx < num_heads && dim_pair < head_dim) {
        size_t pos = token_idx + pos_offset;
        float freq = 1.0f / powf(10000.0f, static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        float theta = static_cast<float>(pos) * freq;
        float cos_th = cosf(theta);
        float sin_th = sinf(theta);

        size_t base_idx = ((token_idx * num_heads + head_idx) * head_dim) + dim_pair;
        float x0 = qk[base_idx];
        float x1 = qk[base_idx + 1];

        // 2D Rotation
        qk[base_idx]     = x0 * cos_th - x1 * sin_th;
        qk[base_idx + 1] = x0 * sin_th + x1 * cos_th;
    }
}

bool launch_swiglu_kernel(const float* input_x, const float* input_y, float* output, size_t N, float beta) {
    if (N == 0) return true;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_x, input_y, output, N, beta);
    return cudaGetLastError() == cudaSuccess;
}

void swiglu_cpu_fallback(const float* input_x, const float* input_y, float* output, size_t N, float beta) {
    for (size_t i = 0; i < N; ++i) {
        float x = input_x[i];
        float swish = x / (1.0f + std::exp(-beta * x));
        output[i] = swish * input_y[i];
    }
}

bool launch_gelu_kernel(const float* input, float* output, size_t N) {
    if (N == 0) return true;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    return cudaGetLastError() == cudaSuccess;
}

void gelu_cpu_fallback(const float* input, float* output, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        float x = input[i];
        float inner = 0.79788456f * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

bool launch_rmsnorm_kernel(const float* input, const float* gamma, float* output, size_t batch_size, size_t hidden_dim, float eps) {
    if (batch_size == 0 || hidden_dim == 0) return true;
    int threads = (hidden_dim > 256) ? 256 : static_cast<int>(hidden_dim);
    rmsnorm_kernel<<<batch_size, threads>>>(input, gamma, output, batch_size, hidden_dim, eps);
    return cudaGetLastError() == cudaSuccess;
}

void rmsnorm_cpu_fallback(const float* input, const float* gamma, float* output, size_t batch_size, size_t hidden_dim, float eps) {
    for (size_t b = 0; b < batch_size; ++b) {
        const float* row_in = input + b * hidden_dim;
        float* row_out = output + b * hidden_dim;

        float sum_sq = 0.0f;
        for (size_t d = 0; d < hidden_dim; ++d) {
            sum_sq += row_in[d] * row_in[d];
        }
        float rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(hidden_dim)) + eps);
        for (size_t d = 0; d < hidden_dim; ++d) {
            row_out[d] = row_in[d] * rms * gamma[d];
        }
    }
}

bool launch_rope_kernel(float* qk_tensor, size_t seq_len, size_t num_heads, size_t head_dim, size_t pos_offset) {
    dim3 grid(seq_len, num_heads);
    int threads = (head_dim / 2 > 256) ? 256 : static_cast<int>(head_dim / 2);
    rope_kernel<<<grid, threads>>>(qk_tensor, seq_len, num_heads, head_dim, pos_offset);
    return cudaGetLastError() == cudaSuccess;
}

void rope_cpu_fallback(float* qk_tensor, size_t seq_len, size_t num_heads, size_t head_dim, size_t pos_offset) {
    for (size_t t = 0; t < seq_len; ++t) {
        size_t pos = t + pos_offset;
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t d = 0; d < head_dim; d += 2) {
                float freq = 1.0f / std::pow(10000.0f, static_cast<float>(d) / static_cast<float>(head_dim));
                float theta = static_cast<float>(pos) * freq;
                float cos_th = std::cos(theta);
                float sin_th = std::sin(theta);

                size_t base_idx = ((t * num_heads + h) * head_dim) + d;
                float x0 = qk_tensor[base_idx];
                float x1 = qk_tensor[base_idx + 1];

                qk_tensor[base_idx]     = x0 * cos_th - x1 * sin_th;
                qk_tensor[base_idx + 1] = x0 * sin_th + x1 * cos_th;
            }
        }
    }
}

} // namespace cuda
} // namespace edge
} // namespace arat

#else
// Non-CUDA C++ Fallbacks

namespace arat {
namespace edge {
namespace cuda {

bool launch_swiglu_kernel(const float* /*x*/, const float* /*y*/, float* /*out*/, size_t /*N*/, float /*beta*/) {
    return false;
}

void swiglu_cpu_fallback(const float* input_x, const float* input_y, float* output, size_t N, float beta) {
    for (size_t i = 0; i < N; ++i) {
        float x = input_x[i];
        float swish = x / (1.0f + std::exp(-beta * x));
        output[i] = swish * input_y[i];
    }
}

bool launch_gelu_kernel(const float* /*input*/, float* /*output*/, size_t /*N*/) {
    return false;
}

void gelu_cpu_fallback(const float* input, float* output, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        float x = input[i];
        float inner = 0.79788456f * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

bool launch_rmsnorm_kernel(const float* /*input*/, const float* /*gamma*/, float* /*output*/, size_t /*batch_size*/, size_t /*hidden_dim*/, float /*eps*/) {
    return false;
}

void rmsnorm_cpu_fallback(const float* input, const float* gamma, float* output, size_t batch_size, size_t hidden_dim, float eps) {
    for (size_t b = 0; b < batch_size; ++b) {
        const float* row_in = input + b * hidden_dim;
        float* row_out = output + b * hidden_dim;

        float sum_sq = 0.0f;
        for (size_t d = 0; d < hidden_dim; ++d) {
            sum_sq += row_in[d] * row_in[d];
        }
        float rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(hidden_dim)) + eps);
        for (size_t d = 0; d < hidden_dim; ++d) {
            row_out[d] = row_in[d] * rms * gamma[d];
        }
    }
}

bool launch_rope_kernel(float* /*qk_tensor*/, size_t /*seq_len*/, size_t /*num_heads*/, size_t /*head_dim*/, size_t /*pos_offset*/) {
    return false;
}

void rope_cpu_fallback(float* qk_tensor, size_t seq_len, size_t num_heads, size_t head_dim, size_t pos_offset) {
    for (size_t t = 0; t < seq_len; ++t) {
        size_t pos = t + pos_offset;
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t d = 0; d < head_dim; d += 2) {
                float freq = 1.0f / std::pow(10000.0f, static_cast<float>(d) / static_cast<float>(head_dim));
                float theta = static_cast<float>(pos) * freq;
                float cos_th = std::cos(theta);
                float sin_th = std::sin(theta);

                size_t base_idx = ((t * num_heads + h) * head_dim) + d;
                float x0 = qk_tensor[base_idx];
                float x1 = qk_tensor[base_idx + 1];

                qk_tensor[base_idx]     = x0 * cos_th - x1 * sin_th;
                qk_tensor[base_idx + 1] = x0 * sin_th + x1 * cos_th;
            }
        }
    }
}

} // namespace cuda
} // namespace edge
} // namespace arat

#endif
