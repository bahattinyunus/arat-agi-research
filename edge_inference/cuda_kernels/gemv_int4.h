#ifndef GEMV_INT4_H
#define GEMV_INT4_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace arat {
namespace edge {
namespace cuda {

// 4-bit Weight-Only Matrix-Vector Multiplication (GEMV INT4)
// y = W * x
// W_packed: packed 4-bit weights (K / 8 elements per uint32_t). Total shape: M x (K / 8)
// scales: per-channel scale factors (size M)
// zeros: per-channel zero points (size M)
// x: input activation vector (size K)
// y: output activation vector (size M)
bool launch_gemv_int4(
    const uint32_t* W_packed,
    const float* scales,
    const float* zeros,
    const float* x,
    float* y,
    size_t M,
    size_t K
);

// CPU reference implementation for testing and non-CUDA environments
void gemv_int4_cpu(
    const uint32_t* W_packed,
    const float* scales,
    const float* zeros,
    const float* x,
    float* y,
    size_t M,
    size_t K
);

// Utility to pack float weights into INT4 packed uint32 representation
void pack_weights_int4(
    const std::vector<float>& W_float,
    std::vector<uint32_t>& W_packed,
    std::vector<float>& scales,
    std::vector<float>& zeros,
    size_t M,
    size_t K
);

} // namespace cuda
} // namespace edge
} // namespace arat

#endif // GEMV_INT4_H
