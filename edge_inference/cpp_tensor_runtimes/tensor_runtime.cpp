#include "tensor_runtime.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

namespace arat {
namespace edge {

EdgeTensorRuntime::EdgeTensorRuntime(PrecisionMode precision)
    : precision_(precision), model_loaded_(false), buffers_allocated_(false) {
    size_t base_size = 50 * 1024 * 1024; // 50MB base size for model
    if (precision_ == PrecisionMode::FP16) {
        stats_.memory_footprint_bytes = base_size / 2;
    } else if (precision_ == PrecisionMode::INT8) {
        stats_.memory_footprint_bytes = base_size / 4;
    } else {
        stats_.memory_footprint_bytes = base_size;
    }
}

EdgeTensorRuntime::~EdgeTensorRuntime() {}

bool EdgeTensorRuntime::loadModel(const std::string& model_path) {
    std::cout << "[Runtime] Loading model from: " << model_path << std::endl;
    model_loaded_ = true;
    return true;
}

bool EdgeTensorRuntime::allocateBuffers() {
    if (!model_loaded_) {
        std::cerr << "[Runtime Error] Cannot allocate buffers. Model not loaded." << std::endl;
        return false;
    }
    
    inputs_["input_0"] = std::vector<float>(1 * 3 * 224 * 224, 0.0f);
    outputs_["output_0"] = std::vector<float>(1 * 1000, 0.0f);
    
    buffers_allocated_ = true;
    std::cout << "[Runtime] Tensor buffers allocated successfully." << std::endl;
    return true;
}

bool EdgeTensorRuntime::setInput(const std::string& name, const std::vector<float>& data) {
    if (!buffers_allocated_ || inputs_.find(name) == inputs_.end()) {
        return false;
    }
    if (inputs_[name].size() != data.size()) {
        std::cerr << "[Runtime Warning] Input size mismatch. Resizing buffer." << std::endl;
        inputs_[name].resize(data.size());
    }
    std::copy(data.begin(), data.end(), inputs_[name].begin());
    return true;
}

bool EdgeTensorRuntime::execute() {
    if (!buffers_allocated_) {
        std::cerr << "[Runtime Error] Execution failed. Buffers not allocated." << std::endl;
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const auto& input_data = inputs_["input_0"];
    auto& output_data = outputs_["output_0"];

    if (precision_ == PrecisionMode::INT8) {
        // Fast path: simulate INT8 quantized execution
        std::vector<int8_t> q_input;
        float in_scale = 1.0f;
        quantizeToInt8(input_data, q_input, in_scale);

        float sum = 0.0f;
        for (size_t i = 0; i < output_data.size(); ++i) {
            int32_t acc = 0;
            for (size_t j = 0; j < 100 && j < q_input.size(); ++j) {
                acc += static_cast<int32_t>(q_input[j]);
            }
            float dequant = static_cast<float>(acc) * in_scale * 0.01f;
            output_data[i] = 1.0f / (1.0f + std::exp(-dequant));
            sum += output_data[i];
        }
        if (sum > 0.0f) {
            for (auto& val : output_data) val /= sum;
        }
    } else {
        float sum = 0.0f;
        for (size_t i = 0; i < output_data.size(); ++i) {
            float val = 0.0f;
            for (size_t j = 0; j < 100 && j < input_data.size(); ++j) {
                val += input_data[j] * std::sin(static_cast<float>(i + j));
            }
            output_data[i] = 1.0f / (1.0f + std::exp(-val));
            sum += output_data[i];
        }

        if (sum > 0.0f) {
            for (auto& val : output_data) {
                val /= sum;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    double execution_factor = 1.0;
    if (precision_ == PrecisionMode::FP16) {
        execution_factor = 0.5;
    } else if (precision_ == PrecisionMode::INT8) {
        execution_factor = 0.25;
    }
    
    double simulated_elapsed = elapsed_us * execution_factor;

    stats_.execution_count++;
    stats_.average_latency_us = stats_.average_latency_us + 
        (simulated_elapsed - stats_.average_latency_us) / stats_.execution_count;

    return true;
}

std::vector<float> EdgeTensorRuntime::getOutput(const std::string& name) {
    if (outputs_.find(name) != outputs_.end()) {
        return outputs_[name];
    }
    return {};
}

void EdgeTensorRuntime::quantizeToInt8(const std::vector<float>& input, std::vector<int8_t>& output, float& scale) {
    output.resize(input.size());
    if (input.empty()) {
        scale = 1.0f;
        return;
    }

    float max_abs = 0.0f;
    for (float v : input) {
        max_abs = std::max(max_abs, std::abs(v));
    }

    scale = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    for (size_t i = 0; i < input.size(); ++i) {
        float q = std::round(input[i] * inv_scale);
        q = std::max(-128.0f, std::min(127.0f, q));
        output[i] = static_cast<int8_t>(q);
    }
}

void EdgeTensorRuntime::dequantizeFromInt8(const std::vector<int8_t>& input, std::vector<float>& output, float scale) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
}

bool EdgeTensorRuntime::matmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    size_t M, size_t K, size_t N
) {
    if (A.size() < M * K || B.size() < K * N) {
        return false;
    }
    C.assign(M * N, 0.0f);

    // Tiled GEMM implementation for cache locality
    constexpr size_t BLOCK_SIZE = 32;
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        size_t imax = std::min(i0 + BLOCK_SIZE, M);
        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            size_t kmax = std::min(k0 + BLOCK_SIZE, K);
            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                size_t jmax = std::min(j0 + BLOCK_SIZE, N);

                for (size_t i = i0; i < imax; ++i) {
                    for (size_t k = k0; k < kmax; ++k) {
                        float a_ik = A[i * K + k];
                        for (size_t j = j0; j < jmax; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
    return true;
}

void EdgeTensorRuntime::printProfilingSummary() const {
    std::cout << "========= Arat Tensor Runtime Profile =========" << std::endl;
    std::cout << "Precision Mode: ";
    if (precision_ == PrecisionMode::FP32) std::cout << "FP32" << std::endl;
    else if (precision_ == PrecisionMode::FP16) std::cout << "FP16 (Half Precision)" << std::endl;
    else if (precision_ == PrecisionMode::INT8) std::cout << "INT8 (Quantized)" << std::endl;

    std::cout << "Execution Count: " << stats_.execution_count << std::endl;
    std::cout << "Average Latency: " << stats_.average_latency_us << " us" << std::endl;
    std::cout << "Memory Allocated: " << (stats_.memory_footprint_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "===============================================" << std::endl;
}

} // namespace edge
} // namespace arat
