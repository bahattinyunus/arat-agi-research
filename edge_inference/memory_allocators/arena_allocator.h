#ifndef ARENA_ALLOCATOR_H
#define ARENA_ALLOCATOR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>

namespace arat {
namespace edge {

class ArenaAllocator {
public:
    // Pre-allocates an arena of size_bytes. Alignments default to 16 bytes for SIMD safety.
    ArenaAllocator(size_t size_bytes, size_t alignment = 16);
    ~ArenaAllocator();

    // Prevent copying to ensure ownership safety
    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Move constructors
    ArenaAllocator(ArenaAllocator&& other) noexcept;
    ArenaAllocator& operator=(ArenaAllocator&& other) noexcept;

    // Allocates memory block from the arena. O(1) time complexity.
    void* allocate(size_t size);

    // Resets the offset back to 0. Deallocates all objects allocated in the arena at once.
    void reset();

    // Checkpoint / watermark marker for nested temporary allocations
    size_t get_checkpoint() const { return offset_; }
    void rollback_to(size_t checkpoint);

    // Stats
    size_t total_size() const { return size_; }
    size_t used_bytes() const { return offset_; }
    size_t free_bytes() const { return size_ - offset_; }

private:
    uint8_t* buffer_;
    size_t size_;
    size_t alignment_;
    size_t offset_;
};

} // namespace edge
} // namespace arat

#endif // ARENA_ALLOCATOR_H
