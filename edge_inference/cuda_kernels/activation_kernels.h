#ifndef ACTIVATION_KERNELS_H
#define ACTIVATION_KERNELS_H

#include <cstddef>

namespace arat {
namespace edge {
namespace cuda {

// Launches the custom SwiGLU activation kernel on GPU.
// SwiGLU(x, y) = Swish(x) * y = (x / (1 + exp(-beta * x))) * y
bool launch_swiglu_kernel(const float* input_x, const float* input_y, float* output, size_t N, float beta = 1.0f);

// CPU fallback implementation for testing/debugging.
void swiglu_cpu_fallback(const float* input_x, const float* input_y, float* output, size_t N, float beta = 1.0f);

// Launches the custom GELU activation kernel on GPU.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
bool launch_gelu_kernel(const float* input, float* output, size_t N);

// CPU fallback implementation for testing/debugging.
void gelu_cpu_fallback(const float* input, float* output, size_t N);

// Launches the custom RMSNorm kernel on GPU.
// RMSNorm(x)_i = (x_i / sqrt(mean(x^2) + eps)) * gamma_i
bool launch_rmsnorm_kernel(const float* input, const float* gamma, float* output, size_t batch_size, size_t hidden_dim, float eps = 1e-6f);

// CPU fallback implementation for RMSNorm.
void rmsnorm_cpu_fallback(const float* input, const float* gamma, float* output, size_t batch_size, size_t hidden_dim, float eps = 1e-6f);

// Launches the custom RoPE (Rotary Position Embedding) kernel on GPU.
bool launch_rope_kernel(float* qk_tensor, size_t seq_len, size_t num_heads, size_t head_dim, size_t pos_offset = 0);

// CPU fallback implementation for RoPE.
void rope_cpu_fallback(float* qk_tensor, size_t seq_len, size_t num_heads, size_t head_dim, size_t pos_offset = 0);

} // namespace cuda
} // namespace edge
} // namespace arat

#endif // ACTIVATION_KERNELS_H
