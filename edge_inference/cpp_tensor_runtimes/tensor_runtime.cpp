#include "tensor_runtime.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace arat {
namespace edge {

EdgeTensorRuntime::EdgeTensorRuntime(PrecisionMode precision)
    : precision_(precision), model_loaded_(false), buffers_allocated_(false) {
    // Determine memory footprint mock based on precision
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
    // Mock successful parsing
    model_loaded_ = true;
    return true;
}

bool EdgeTensorRuntime::allocateBuffers() {
    if (!model_loaded_) {
        std::cerr << "[Runtime Error] Cannot allocate buffers. Model not loaded." << std::endl;
        return false;
    }
    
    // Allocate mock input/output bindings
    inputs_["input_0"] = std::vector<float>(1 * 3 * 224 * 224, 0.0f); // Image tensor shape: BCHW
    outputs_["output_0"] = std::vector<float>(1 * 1000, 0.0f);       // Classification logits
    
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

    // Mock Tensor execution (simulating network evaluation and quantization overhead)
    const auto& input_data = inputs_["input_0"];
    auto& output_data = outputs_["output_0"];

    // Compute mock activations
    float sum = 0.0f;
    for (size_t i = 0; i < output_data.size(); ++i) {
        float val = 0.0f;
        // Simple mock fully-connected activation contribution
        for (size_t j = 0; j < 100 && j < input_data.size(); ++j) {
            val += input_data[j] * std::sin(static_cast<float>(i + j));
        }
        output_data[i] = 1.0f / (1.0f + std::exp(-val)); // Sigmoid mock
        sum += output_data[i];
    }

    // Normalize output mock to simulate softmax
    if (sum > 0.0f) {
        for (auto& val : output_data) {
            val /= sum;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    // Scale latency based on precision mode
    double execution_factor = 1.0;
    if (precision_ == PrecisionMode::FP16) {
        execution_factor = 0.5; // FP16 is twice as fast
    } else if (precision_ == PrecisionMode::INT8) {
        execution_factor = 0.25; // INT8 is four times as fast
    }
    
    double simulated_elapsed = elapsed_us * execution_factor;

    // Update cumulative moving average of latency
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
