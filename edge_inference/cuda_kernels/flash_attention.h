#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include <cstddef>
#include <vector>

namespace arat {
namespace edge {
namespace cuda {

// FlashAttention-2 style fused scaled dot-product attention
// Q, K, V have shape (batch_size, num_heads, seq_len, head_dim)
// Output has shape (batch_size, num_heads, seq_len, head_dim)
bool launch_flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim,
    float scale = 0.0f // if 0.0, defaults to 1.0 / sqrt(head_dim)
);

// CPU reference implementation using online softmax (tiled evaluation)
void flash_attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* Output,
    size_t batch_size,
    size_t num_heads,
    size_t seq_len,
    size_t head_dim,
    float scale = 0.0f
);

} // namespace cuda
} // namespace edge
} // namespace arat

#endif // FLASH_ATTENTION_H
