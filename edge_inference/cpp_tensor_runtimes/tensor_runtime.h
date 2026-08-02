#ifndef EDGE_TENSOR_RUNTIME_H
#define EDGE_TENSOR_RUNTIME_H

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <memory>

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
