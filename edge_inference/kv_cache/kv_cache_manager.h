#ifndef KV_CACHE_MANAGER_H
#define KV_CACHE_MANAGER_H

#include <vector>
#include <unordered_map>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace arat {
namespace edge {

struct CacheBlock {
    int block_id;
    std::vector<int8_t> key_data;   // INT8 quantized keys (tokens * head_dim)
    std::vector<int8_t> val_data;   // INT8 quantized values (tokens * head_dim)
    float key_scale = 1.0f;
    float val_scale = 1.0f;
    size_t num_tokens_stored = 0;
};

class PagedKVCacheManager {
public:
    PagedKVCacheManager(size_t total_blocks, size_t block_size_tokens, size_t num_heads, size_t head_dim);

    // Allocates a physical block for a sequence
    int allocateBlock(int seq_id);

    // Appends token K, V projections into the sequence's allocated paged cache
    bool appendToken(int seq_id, const std::vector<float>& key, const std::vector<float>& val);

    // Retrieves total tokens currently cached for a sequence
    size_t getSequenceLength(int seq_id) const;

    // Releases sequence memory back to free pool
    void freeSequence(int seq_id);

    // Memory stats
    size_t getFreeBlockCount() const { return free_block_ids_.size(); }
    size_t getTotalCapacityTokens() const { return total_blocks_ * block_size_tokens_; }

private:
    size_t total_blocks_;
    size_t block_size_tokens_;
    size_t num_heads_;
    size_t head_dim_;

    std::vector<CacheBlock> blocks_;
    std::vector<int> free_block_ids_;

    // Map: seq_id -> list of allocated physical block_ids
    std::unordered_map<int, std::vector<int>> sequence_table_;
};

} // namespace edge
} // namespace arat

#endif // KV_CACHE_MANAGER_H
