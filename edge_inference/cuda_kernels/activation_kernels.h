#ifndef ACTIVATION_KERNELS_H
#define ACTIVATION_KERNELS_H

#include <cstddef>

namespace arat {
namespace edge {
namespace cuda {

// Launches the custom SwiGLU activation kernel on GPU.
// SwiGLU(x, y) = Swish(x) * y = (x / (1 + exp(-beta * x))) * y
// input_x and input_y are float pointers to GPU device buffers of size N.
// output is a float pointer to a GPU device buffer of size N.
bool launch_swiglu_kernel(const float* input_x, const float* input_y, float* output, size_t N, float beta = 1.0f);

// CPU fallback implementation for testing/debugging.
void swiglu_cpu_fallback(const float* input_x, const float* input_y, float* output, size_t N, float beta = 1.0f);

// Launches the custom GELU activation kernel on GPU.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
bool launch_gelu_kernel(const float* input, float* output, size_t N);

// CPU fallback implementation for testing/debugging.
void gelu_cpu_fallback(const float* input, float* output, size_t N);

} // namespace cuda
} // namespace edge
} // namespace arat


#endif // ACTIVATION_KERNELS_H
