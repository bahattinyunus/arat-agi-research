#include "gemv_int4.h"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace arat {
namespace edge {
namespace cuda {

// CUDA Device Kernel for INT4 Dequantization & GEMV
__global__ void gemv_int4_kernel(
    const uint32_t* __restrict__ W_packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const float* __restrict__ x,
    float* __restrict__ y,
    size_t M,
    size_t K
) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float scale = scales[row];
        float zero = zeros[row];
        size_t packed_k = K / 8;
        const uint32_t* row_packed = W_packed + row * packed_k;

        float sum = 0.0f;
        for (size_t pk = 0; pk < packed_k; ++pk) {
            uint32_t packed_val = row_packed[pk];
            size_t base_k = pk * 8;

            // Unpack 8 4-bit weights
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                uint32_t w4 = (packed_val >> (b * 4)) & 0x0F;
                float w_real = (static_cast<float>(w4) - zero) * scale;
                sum += w_real * x[base_k + b];
            }
        }
        y[row] = sum;
    }
}

bool launch_gemv_int4(
    const uint32_t* W_packed,
    const float* scales,
    const float* zeros,
    const float* x,
    float* y,
    size_t M,
    size_t K
) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    gemv_int4_kernel<<<blocks, threads>>>(W_packed, scales, zeros, x, y, M, K);
    return cudaGetLastError() == cudaSuccess;
}

void gemv_int4_cpu(
    const uint32_t* W_packed,
    const float* scales,
    const float* zeros,
    const float* x,
    float* y,
    size_t M,
    size_t K
) {
    size_t packed_k = K / 8;
    for (size_t row = 0; row < M; ++row) {
        float scale = scales[row];
        float zero = zeros[row];
        const uint32_t* row_packed = W_packed + row * packed_k;

        float sum = 0.0f;
        for (size_t pk = 0; pk < packed_k; ++pk) {
            uint32_t packed_val = row_packed[pk];
            size_t base_k = pk * 8;

            for (int b = 0; b < 8; ++b) {
                uint32_t w4 = (packed_val >> (b * 4)) & 0x0F;
                float w_real = (static_cast<float>(w4) - zero) * scale;
                sum += w_real * x[base_k + b];
            }
        }
        y[row] = sum;
    }
}

void pack_weights_int4(
    const std::vector<float>& W_float,
    std::vector<uint32_t>& W_packed,
    std::vector<float>& scales,
    std::vector<float>& zeros,
    size_t M,
    size_t K
) {
    size_t packed_k = (K + 7) / 8;
    W_packed.assign(M * packed_k, 0);
    scales.resize(M);
    zeros.resize(M);

    for (size_t row = 0; row < M; ++row) {
        float min_val = 1e20f;
        float max_val = -1e20f;
        for (size_t col = 0; col < K; ++col) {
            float val = W_float[row * K + col];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        float scale = (max_val - min_val) / 15.0f;
        if (scale < 1e-6f) scale = 1.0f;
        float zero = -min_val / scale;

        scales[row] = scale;
        zeros[row] = zero;

        for (size_t pk = 0; pk < packed_k; ++pk) {
            uint32_t packed_val = 0;
            for (int b = 0; b < 8; ++b) {
                size_t col = pk * 8 + b;
                if (col < K) {
                    float val = W_float[row * K + col];
                    int q = static_cast<int>(std::round(val / scale + zero));
                    q = std::max(0, std::min(15, q));
                    packed_val |= (static_cast<uint32_t>(q) & 0x0F) << (b * 4);
                }
            }
            W_packed[row * packed_k + pk] = packed_val;
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

bool launch_gemv_int4(
    const uint32_t* /*W_packed*/, const float* /*scales*/, const float* /*zeros*/,
    const float* /*x*/, float* /*y*/, size_t /*M*/, size_t /*K*/
) {
    return false;
}

void gemv_int4_cpu(
    const uint32_t* W_packed,
    const float* scales,
    const float* zeros,
    const float* x,
    float* y,
    size_t M,
    size_t K
) {
    size_t packed_k = K / 8;
    for (size_t row = 0; row < M; ++row) {
        float scale = scales[row];
        float zero = zeros[row];
        const uint32_t* row_packed = W_packed + row * packed_k;

        float sum = 0.0f;
        for (size_t pk = 0; pk < packed_k; ++pk) {
            uint32_t packed_val = row_packed[pk];
            size_t base_k = pk * 8;

            for (int b = 0; b < 8; ++b) {
                uint32_t w4 = (packed_val >> (b * 4)) & 0x0F;
                float w_real = (static_cast<float>(w4) - zero) * scale;
                sum += w_real * x[base_k + b];
            }
        }
        y[row] = sum;
    }
}

void pack_weights_int4(
    const std::vector<float>& W_float,
    std::vector<uint32_t>& W_packed,
    std::vector<float>& scales,
    std::vector<float>& zeros,
    size_t M,
    size_t K
) {
    size_t packed_k = (K + 7) / 8;
    W_packed.assign(M * packed_k, 0);
    scales.resize(M);
    zeros.resize(M);

    for (size_t row = 0; row < M; ++row) {
        float min_val = 1e20f;
        float max_val = -1e20f;
        for (size_t col = 0; col < K; ++col) {
            float val = W_float[row * K + col];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        float scale = (max_val - min_val) / 15.0f;
        if (scale < 1e-6f) scale = 1.0f;
        float zero = -min_val / scale;

        scales[row] = scale;
        zeros[row] = zero;

        for (size_t pk = 0; pk < packed_k; ++pk) {
            uint32_t packed_val = 0;
            for (int b = 0; b < 8; ++b) {
                size_t col = pk * 8 + b;
                if (col < K) {
                    float val = W_float[row * K + col];
                    int q = static_cast<int>(std::round(val / scale + zero));
                    q = std::max(0, std::min(15, q));
                    packed_val |= (static_cast<uint32_t>(q) & 0x0F) << (b * 4);
                }
            }
            W_packed[row * packed_k + pk] = packed_val;
        }
    }
}

} // namespace cuda
} // namespace edge
} // namespace arat

#endif
