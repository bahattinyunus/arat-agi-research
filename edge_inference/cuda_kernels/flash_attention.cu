#include "flash_attention.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace arat {
namespace edge {
namespace cuda {

// FlashAttention GPU Device Kernel
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Output,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim,
    float scale
) {
    size_t b = blockIdx.x;
    size_t h = blockIdx.y;
    size_t q_idx = blockIdx.z * blockDim.x + threadIdx.x;

    if (b < batch_size && h < num_heads && q_idx < seq_len) {
        size_t head_stride = seq_len * head_dim;
        size_t batch_stride = num_heads * head_stride;
        const float* q_vec = Q + b * batch_stride + h * head_stride + q_idx * head_dim;
        float* out_vec = Output + b * batch_stride + h * head_stride + q_idx * head_dim;

        float max_score = -1e20f;
        float sum_exp = 0.0f;

        // Initialize output vector
        for (size_t d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }

        // Tiled stream through key-value sequence
        for (size_t k_idx = 0; k_idx < seq_len; ++k_idx) {
            const float* k_vec = K + b * batch_stride + h * head_stride + k_idx * head_dim;
            const float* v_vec = V + b * batch_stride + h * head_stride + k_idx * head_dim;

            // Dot product Q_i * K_j
            float score = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                score += q_vec[d] * k_vec[d];
            }
            score *= scale;

            // Online softmax update
            float new_max = fmaxf(max_score, score);
            float exp_old = __expf(max_score - new_max);
            float exp_cur = __expf(score - new_max);

            sum_exp = sum_exp * exp_old + exp_cur;

            for (size_t d = 0; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_old + exp_cur * v_vec[d];
            }
            max_score = new_max;
        }

        // Final normalization
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (size_t d = 0; d < head_dim; ++d) {
            out_vec[d] *= inv_sum;
        }
    }
}

bool launch_flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim,
    float scale
) {
    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    int threads = 64;
    int blocks_z = (seq_len + threads - 1) / threads;
    dim3 grid(batch_size, num_heads, blocks_z);

    flash_attention_kernel<<<grid, threads>>>(Q, K, V, Output, batch_size, num_heads, seq_len, head_dim, scale);
    return cudaGetLastError() == cudaSuccess;
}

void flash_attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim,
    float scale
) {
    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    size_t head_stride = seq_len * head_dim;
    size_t batch_stride = num_heads * head_stride;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                const float* q_vec = Q + b * batch_stride + h * head_stride + i * head_dim;
                float* out_vec = Output + b * batch_stride + h * head_stride + i * head_dim;

                float max_val = -1e20f;
                float sum_exp = 0.0f;
                std::fill(out_vec, out_vec + head_dim, 0.0f);

                for (size_t j = 0; j < seq_len; ++j) {
                    const float* k_vec = K + b * batch_stride + h * head_stride + j * head_dim;
                    const float* v_vec = V + b * batch_stride + h * head_stride + j * head_dim;

                    float score = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        score += q_vec[d] * k_vec[d];
                    }
                    score *= scale;

                    float new_max = std::max(max_val, score);
                    float exp_old = std::exp(max_val - new_max);
                    float exp_cur = std::exp(score - new_max);

                    sum_exp = sum_exp * exp_old + exp_cur;

                    for (size_t d = 0; d < head_dim; ++d) {
                        out_vec[d] = out_vec[d] * exp_old + exp_cur * v_vec[d];
                    }
                    max_val = new_max;
                }

                float inv_sum = 1.0f / (sum_exp + 1e-12f);
                for (size_t d = 0; d < head_dim; ++d) {
                    out_vec[d] *= inv_sum;
                }
            }
        }
    }
}

} // namespace cuda
} // namespace edge
} // namespace arat

#else
// Non-CUDA C++ Implementation

namespace arat {
namespace edge {
namespace cuda {

bool launch_flash_attention(
    const float* /*Q*/, const float* /*K*/, const float* /*V*/, float* /*Output*/,
    size_t /*batch_size*/, size_t /*num_heads*/, size_t /*seq_len*/, size_t /*head_dim*/,
    float /*scale*/
) {
    return false;
}

void flash_attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim,
    float scale
) {
    if (scale <= 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    size_t head_stride = seq_len * head_dim;
    size_t batch_stride = num_heads * head_stride;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                const float* q_vec = Q + b * batch_stride + h * head_stride + i * head_dim;
                float* out_vec = Output + b * batch_stride + h * head_stride + i * head_dim;

                float max_val = -1e20f;
                float sum_exp = 0.0f;
                std::fill(out_vec, out_vec + head_dim, 0.0f);

                for (size_t j = 0; j < seq_len; ++j) {
                    const float* k_vec = K + b * batch_stride + h * head_stride + j * head_dim;
                    const float* v_vec = V + b * batch_stride + h * head_stride + j * head_dim;

                    float score = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        score += q_vec[d] * k_vec[d];
                    }
                    score *= scale;

                    float new_max = std::max(max_val, score);
                    float exp_old = std::exp(max_val - new_max);
                    float exp_cur = std::exp(score - new_max);

                    sum_exp = sum_exp * exp_old + exp_cur;

                    for (size_t d = 0; d < head_dim; ++d) {
                        out_vec[d] = out_vec[d] * exp_old + exp_cur * v_vec[d];
                    }
                    max_val = new_max;
                }

                float inv_sum = 1.0f / (sum_exp + 1e-12f);
                for (size_t d = 0; d < head_dim; ++d) {
                    out_vec[d] *= inv_sum;
                }
            }
        }
    }
}

} // namespace cuda
} // namespace edge
} // namespace arat

#endif
