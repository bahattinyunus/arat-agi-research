#include "arena_allocator.h"
#include <new>
#include <algorithm>
#include <utility>

namespace arat {
namespace edge {

ArenaAllocator::ArenaAllocator(size_t size_bytes, size_t alignment)
    : size_(size_bytes), alignment_(alignment), offset_(0) {
    
    // Allocate raw cache-aligned byte buffer
#if defined(_MSC_VER)
    buffer_ = static_cast<uint8_t*>(_aligned_malloc(size_, alignment_));
    if (!buffer_) {
        throw std::bad_alloc();
    }
#else
    // POSIX or GCC/Clang aligned alloc
    buffer_ = static_cast<uint8_t*>(std::aligned_alloc(alignment_, size_));
    if (!buffer_) {
        throw std::bad_alloc();
    }
#endif
}

ArenaAllocator::~ArenaAllocator() {
    if (buffer_) {
#if defined(_MSC_VER)
        _aligned_free(buffer_);
#else
        free(buffer_);
#endif
    }
}

ArenaAllocator::ArenaAllocator(ArenaAllocator&& other) noexcept
    : buffer_(other.buffer_), size_(other.size_), alignment_(other.alignment_), offset_(other.offset_) {
    other.buffer_ = nullptr;
    other.size_ = 0;
    other.offset_ = 0;
}

ArenaAllocator& ArenaAllocator::operator=(ArenaAllocator&& other) noexcept {
    if (this != &other) {
        if (buffer_) {
#if defined(_MSC_VER)
            _aligned_free(buffer_);
#else
            free(buffer_);
#endif
        }
        buffer_ = other.buffer_;
        size_ = other.size_;
        alignment_ = other.alignment_;
        offset_ = other.offset_;

        other.buffer_ = nullptr;
        other.size_ = 0;
        other.offset_ = 0;
    }
    return *this;
}

void* ArenaAllocator::allocate(size_t size) {
    // Calculate aligned offset
    size_t current_ptr = reinterpret_cast<size_t>(buffer_ + offset_);
    size_t align_mask = alignment_ - 1;
    size_t aligned_ptr = (current_ptr + align_mask) & ~align_mask;
    
    size_t new_offset = (aligned_ptr - reinterpret_cast<size_t>(buffer_)) + size;

    // Out of memory check
    if (new_offset > size_) {
        return nullptr;
    }

    offset_ = new_offset;
    return reinterpret_cast<void*>(aligned_ptr);
}

void ArenaAllocator::reset() {
    offset_ = 0;
}

} // namespace edge
} // namespace arat
