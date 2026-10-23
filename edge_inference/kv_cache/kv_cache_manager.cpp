#include "kv_cache_manager.h"
#include <cmath>
#include <algorithm>

namespace arat {
namespace edge {

PagedKVCacheManager::PagedKVCacheManager(
    size_t total_blocks,
    size_t block_size_tokens,
    size_t num_heads,
    size_t head_dim
) : total_blocks_(total_blocks),
    block_size_tokens_(block_size_tokens),
    num_heads_(num_heads),
    head_dim_(head_dim) {

    size_t block_bytes = block_size_tokens * num_heads * head_dim;
    blocks_.resize(total_blocks);
    free_block_ids_.reserve(total_blocks);

    for (size_t i = 0; i < total_blocks; ++i) {
        blocks_[i].block_id = static_cast<int>(i);
        blocks_[i].key_data.resize(block_bytes, 0);
        blocks_[i].val_data.resize(block_bytes, 0);
        blocks_[i].num_tokens_stored = 0;
        free_block_ids_.push_back(static_cast<int>(i));
    }
}

int PagedKVCacheManager::allocateBlock(int seq_id) {
    if (free_block_ids_.empty()) {
        return -1; // Out of memory
    }
    int block_id = free_block_ids_.back();
    free_block_ids_.pop_back();

    blocks_[block_id].num_tokens_stored = 0;
    sequence_table_[seq_id].push_back(block_id);
    return block_id;
}

bool PagedKVCacheManager::appendToken(int seq_id, const std::vector<float>& key, const std::vector<float>& val) {
    size_t token_dim = num_heads_ * head_dim_;
    if (key.size() != token_dim || val.size() != token_dim) {
        return false;
    }

    if (sequence_table_.find(seq_id) == sequence_table_.end() || sequence_table_[seq_id].empty()) {
        if (allocateBlock(seq_id) < 0) return false;
    }

    int active_block_id = sequence_table_[seq_id].back();
    CacheBlock& block = blocks_[active_block_id];

    // If current block is full, allocate next block
    if (block.num_tokens_stored >= block_size_tokens_) {
        active_block_id = allocateBlock(seq_id);
        if (active_block_id < 0) return false;
    }

    CacheBlock& cur_block = blocks_[active_block_id];
    size_t offset = cur_block.num_tokens_stored * token_dim;

    // Simple INT8 quantization
    for (size_t d = 0; d < token_dim; ++d) {
        float q_k = std::max(-128.0f, std::min(127.0f, std::round(key[d] * 100.0f)));
        float q_v = std::max(-128.0f, std::min(127.0f, std::round(val[d] * 100.0f)));
        cur_block.key_data[offset + d] = static_cast<int8_t>(q_k);
        cur_block.val_data[offset + d] = static_cast<int8_t>(q_v);
    }
    cur_block.num_tokens_stored++;
    return true;
}

size_t PagedKVCacheManager::getSequenceLength(int seq_id) const {
    if (sequence_table_.find(seq_id) == sequence_table_.end()) {
        return 0;
    }
    size_t total = 0;
    for (int block_id : sequence_table_.at(seq_id)) {
        total += blocks_[block_id].num_tokens_stored;
    }
    return total;
}

void PagedKVCacheManager::freeSequence(int seq_id) {
    if (sequence_table_.find(seq_id) != sequence_table_.end()) {
        for (int block_id : sequence_table_[seq_id]) {
            blocks_[block_id].num_tokens_stored = 0;
            free_block_ids_.push_back(block_id);
        }
        sequence_table_.erase(seq_id);
    }
}

} // namespace edge
} // namespace arat
