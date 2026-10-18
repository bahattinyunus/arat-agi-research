#ifndef EDGE_TENSOR_RUNTIME_H
#define EDGE_TENSOR_RUNTIME_H

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <memory>
#include <cstdint>

namespace arat {
namespace edge {

enum class PrecisionMode {
    FP32,
    FP16,
    INT8
};

struct ProfilingStats {
    double average_latency_us = 0.0;
    size_t execution_count = 0;
    size_t memory_footprint_bytes = 0;
};

class EdgeTensorRuntime {
public:
    EdgeTensorRuntime(PrecisionMode precision = PrecisionMode::FP16);
    ~EdgeTensorRuntime();

    bool loadModel(const std::string& model_path);
    bool allocateBuffers();
    bool setInput(const std::string& name, const std::vector<float>& data);
    bool execute();
    std::vector<float> getOutput(const std::string& name);

    // Quantization routines
    static void quantizeToInt8(const std::vector<float>& input, std::vector<int8_t>& output, float& scale);
    static void dequantizeFromInt8(const std::vector<int8_t>& input, std::vector<float>& output, float scale);

    // Matrix Multiplication GEMM: C (M x N) = A (M x K) * B (K x N)
    static bool matmul(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        size_t M, size_t K, size_t N
    );

    ProfilingStats getStats() const { return stats_; }
    void printProfilingSummary() const;

private:
    PrecisionMode precision_;
    bool model_loaded_;
    bool buffers_allocated_;
    
    std::unordered_map<std::string, std::vector<float>> inputs_;
    std::unordered_map<std::string, std::vector<float>> outputs_;
    
    ProfilingStats stats_;
};

} // namespace edge
} // namespace arat

#endif // EDGE_TENSOR_RUNTIME_H
