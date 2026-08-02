#include "activation_kernels.h"
#include <cmath>
#include <iostream>

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
        // Swish(x) = x / (1.0 + exp(-beta * x))
        float swish = x_val / (1.0f + __expf(-beta * x_val));
        out[idx] = swish * y_val;
    }
}

bool launch_swiglu_kernel(const float* input_x, const float* input_y, float* output, size_t N, float beta) {
    if (N == 0) return true;

    // Standard CUDA block and grid configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input_x, input_y, output, N, beta);
    
    // Check for kernel launch or execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

void swiglu_cpu_fallback(const float* input_x, const float* input_y, float* output, size_t N, float beta) {
    for (size_t i = 0; i < N; ++i) {
        float x = input_x[i];
        float swish = x / (1.0f + std::exp(-beta * x));
        output[i] = swish * input_y[i];
    }
}

} // namespace cuda
} // namespace edge
} // namespace arat

#else
// Compilation fallback when CUDA Compiler is not present

namespace arat {
namespace edge {
namespace cuda {

bool launch_swiglu_kernel(const float* /*input_x*/, const float* /*input_y*/, float* /*output*/, size_t /*N*/, float /*beta*/) {
    std::cerr << "[CUDA Warning] CUDA compiler not detected. GPU kernel cannot be executed." << std::endl;
    return false;
}

void swiglu_cpu_fallback(const float* input_x, const float* input_y, float* output, size_t N, float beta) {
    for (size_t i = 0; i < N; ++i) {
        float x = input_x[i];
        float swish = x / (1.0f + std::exp(-beta * x));
        output[i] = swish * input_y[i];
    }
}

} // namespace cuda
} // namespace edge
} // namespace arat

#endif
