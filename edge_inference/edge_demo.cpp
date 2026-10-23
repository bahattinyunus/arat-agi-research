#include <iostream>
#include <vector>
#include "tensor_runtime.h"
#include "arena_allocator.h"
#include "activation_kernels.h"
#include "flash_attention.h"
#include "kv_cache_manager.h"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "    ARAT Edge Inference & HPC Hardware Engine    " << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. Test Arena Allocator with Watermark Rollback
    std::cout << "\n[1] Testing Deterministic SIMD Arena Allocator..." << std::endl;
    arat::edge::ArenaAllocator arena(1024 * 1024, 16); // 1MB arena
    std::cout << "Total Arena Capacity: " << arena.total_size() << " bytes." << std::endl;

    void* p1 = arena.allocate(256);
    std::cout << "Allocated block 1 (256 bytes): " << p1 << " | Used: " << arena.used_bytes() << " bytes." << std::endl;

    size_t checkpoint = arena.get_checkpoint();
    std::cout << "Saved Watermark Checkpoint: " << checkpoint << " bytes." << std::endl;

    void* p2 = arena.allocate(1024);
    std::cout << "Allocated temporary scratch (1024 bytes): " << p2 << " | Used: " << arena.used_bytes() << " bytes." << std::endl;

    arena.rollback_to(checkpoint);
    std::cout << "Rolled back to checkpoint. Current Used: " << arena.used_bytes() << " bytes (Cleaned O(1))." << std::endl;

    // 2. Test Tensor Runtime with FP16 and INT8 Precision
    std::cout << "\n[2] Testing Tensor Runtime and INT8 Symmetric Quantization..." << std::endl;
    arat::edge::EdgeTensorRuntime runtime(arat::edge::PrecisionMode::INT8);
    runtime.loadModel("weights/arat_edge_backbone.onnx");
    runtime.allocateBuffers();

    std::vector<float> sample_input(1 * 3 * 224 * 224, 0.5f);
    runtime.setInput("input_0", sample_input);
    runtime.execute();
    runtime.printProfilingSummary();

    // 3. Test INT8 Quantization and Dequantization
    std::vector<float> raw_weights = {-1.5f, -0.5f, 0.0f, 0.75f, 1.5f};
    std::vector<int8_t> quantized_weights;
    float scale = 0.0f;
    arat::edge::EdgeTensorRuntime::quantizeToInt8(raw_weights, quantized_weights, scale);
    std::vector<float> dequantized_weights;
    arat::edge::EdgeTensorRuntime::dequantizeFromInt8(quantized_weights, dequantized_weights, scale);

    std::cout << "Original weights:  ";
    for (float v : raw_weights) std::cout << v << " ";
    std::cout << "\nQuantized (scale=" << scale << "): ";
    for (int8_t q : quantized_weights) std::cout << static_cast<int>(q) << " ";
    std::cout << "\nDequantized:       ";
    for (float dq : dequantized_weights) std::cout << dq << " ";
    std::cout << std::endl;

    // 4. Test Tiled MatMul GEMM
    std::cout << "\n[3] Testing Tiled GEMM Matrix Multiplication (4x4)..." << std::endl;
    std::vector<float> A = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f
    };
    std::vector<float> B = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    std::vector<float> C(16, 0.0f);
    arat::edge::EdgeTensorRuntime::matmul(A, B, C, 4, 4, 4);
    std::cout << "GEMM Result MatMul(A, Identity):" << std::endl;
    for (size_t r = 0; r < 4; ++r) {
        std::cout << "  [ ";
        for (size_t c = 0; c < 4; ++c) {
            std::cout << C[r * 4 + c] << " ";
        }
        std::cout << "]" << std::endl;
    }

    // 5. Test FlashAttention Tiling
    std::cout << "\n[4] Testing FlashAttention Tiled Online Softmax..." << std::endl;
    std::vector<float> Q = {1.0f, 0.0f, 0.0f, 1.0f}; // 1 batch, 1 head, 2 seq, 2 dim
    std::vector<float> K = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> V = {2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> AttnOut(4, 0.0f);
    arat::edge::cuda::flash_attention_cpu(Q.data(), K.data(), V.data(), AttnOut.data(), 1, 1, 2, 2);
    std::cout << "FlashAttention Output: [ ";
    for (float v : AttnOut) std::cout << v << " ";
    std::cout << "]" << std::endl;

    // 6. Test Paged KV Cache Manager
    std::cout << "\n[5] Testing Paged KV-Cache Manager..." << std::endl;
    arat::edge::PagedKVCacheManager kv_mgr(16, 4, 1, 2); // 16 blocks, 4 tokens/block
    kv_mgr.allocateBlock(101);
    kv_mgr.appendToken(101, {0.1f, 0.2f}, {0.3f, 0.4f});
    std::cout << "Allocated sequence 101. Total tokens cached: " << kv_mgr.getSequenceLength(101) << std::endl;
    std::cout << "Remaining free cache blocks: " << kv_mgr.getFreeBlockCount() << std::endl;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "  ARAT Edge Engine Verification Complete!" << std::endl;
    std::cout << "==================================================" << std::endl;
    return 0;
}
