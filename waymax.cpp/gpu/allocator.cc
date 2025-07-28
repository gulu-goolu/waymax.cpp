#include "waymax.cpp/gpu/allocator.h"

#include <cstddef>
#include <memory>
#include <mutex>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "waymax.cpp/utils/macros.h"

namespace waymax_cpp {

class BfcAllocatorImpl : public IAllocator {
 public:
  BfcAllocatorImpl(std::shared_ptr<IAllocator> base, CachingAllocatorOptions options)
      : options_(options), base_(base), large_pool_(base), small_pool_(base) {}

  constexpr size_t round_size(size_t size) {
    if (size < options_.large_alloc) {
      return (size + options_.alloc_alignemnt - 1) & ~(options_.alloc_alignemnt - 1);
    } else {
      return (size + options_.large_round - 1) & ~(options_.large_round - 1);
    }
  }
  struct MemoryPool;

  struct MemoryBlock {
    size_t size = 0;
    void *addr = nullptr;

    MemoryPool *pool = nullptr;

    MemoryBlock *prev = nullptr, *next = nullptr;

    bool allocated = false;
  };

  struct memory_block_greater {
    constexpr bool operator()(const MemoryBlock *l, const MemoryBlock *r) const noexcept {
      if (l->size != r->size) {
        return l->size < r->size;
      }

      return reinterpret_cast<intptr_t>(l->addr) < reinterpret_cast<intptr_t>(r->addr);
    }
  };

  struct MemoryPool {
    std::shared_ptr<IAllocator> base;
    absl::btree_set<MemoryBlock *, memory_block_greater> blocks;
    std::vector<MemoryBlock *> block_pool;
    size_t next_block_size = 1024L * 1024L;

    MemoryPool(std::shared_ptr<IAllocator> _base) : base(_base) {}

    ~MemoryPool() {
      for (auto block : blocks) {
        if (block->prev || block->next) {
          ABSL_LOG(FATAL) << ("memory not released");
        }

        base->deallocate(block->addr);

        block_pool.push_back(block);
      }

      for (auto block : block_pool) {
        delete block;
      }
    }

    MemoryBlock *create_block() {
      if (block_pool.empty()) {
        return new MemoryBlock();
      } else {
        auto block = block_pool.back();
        block_pool.pop_back();
        return block;
      }
    }
  };

  absl::StatusOr<MemoryBlock *> prepare_block(MemoryPool *pool, size_t rounded_size) {
    MemoryBlock key = {};
    key.size = rounded_size;
    key.addr = nullptr;
    auto iter = pool->blocks.lower_bound(&key);
    if (iter != pool->blocks.end()) {
      MemoryBlock *block = *iter;
      pool->blocks.erase(iter);
      return block;
    }

    const size_t block_size = std::max<size_t>(pool->next_block_size, rounded_size);
    pool->next_block_size = std::min<size_t>(pool->next_block_size * 2, options_.max_block_size);

    MemoryBlock *new_block = pool->create_block();

    COMMON_ASSIGN_OR_RETURN(new_block->addr, base_->allocate(block_size));

    new_block->size = block_size;
    new_block->allocated = false;
    new_block->pool = pool;
    new_block->prev = nullptr;
    new_block->next = nullptr;

    return new_block;
  }

  void try_split_block(MemoryBlock *block, size_t size) {
    const size_t remaining = block->size - size;
    if (remaining < options_.alloc_alignemnt) {
      return;
    }

    MemoryPool *pool = block->pool;
    if (block->next && !block->next->allocated) {
      pool->blocks.erase(block->next);

      block->next->size += remaining;
      block->next->addr = reinterpret_cast<uint8_t *>(block->addr) + size;
    } else {
      MemoryBlock *new_block = pool->create_block();

      new_block->size = remaining;
      new_block->addr = reinterpret_cast<uint8_t *>(block->addr) + size;
      new_block->pool = pool;
      new_block->allocated = false;

      // insert into block list
      new_block->prev = block;
      new_block->next = block->next;
      if (block->next) {
        block->next->prev = new_block;
      }
      block->next = new_block;
    }

    pool->blocks.insert(block->next);
    block->size = size;
  }

  void try_merge_block(MemoryBlock *block) {
    MemoryPool *pool = block->pool;

    MemoryBlock *prev = block->prev;
    if (prev && !prev->allocated) {
      block->addr = prev->addr;
      block->size += prev->size;

      block->prev = prev->prev;
      if (prev->prev) {
        prev->prev->next = block;
      }

      pool->blocks.erase(prev);
      pool->block_pool.push_back(prev);
    }

    MemoryBlock *next = block->next;
    if (next && !next->allocated) {
      block->size += next->size;
      block->next = next->next;
      if (next->next) {
        next->next->prev = block;
      }

      pool->blocks.erase(next);
      pool->block_pool.push_back(next);
    }
  }

  absl::StatusOr<void *> allocate(size_t size) {
    if (size == 0) {
      return absl::InvalidArgumentError("size must greater than zero");
    }

    std::lock_guard<std::mutex> ll(mutex_);

    const size_t rounded_size = round_size(size);

    MemoryPool *pool = nullptr;
    if (rounded_size >= options_.large_alloc) {
      pool = &large_pool_;
    } else {
      pool = &small_pool_;
    }

    COMMON_CONSTRUCT_OR_RETURN(block, prepare_block(pool, rounded_size));
    try_split_block(block, rounded_size);
    block->allocated = true;

    allocated_blocks_[block->addr] = block;

    return block->addr;
  }

  void deallocate(void *addr) {
    std::lock_guard<std::mutex> ll(mutex_);

    auto iter = allocated_blocks_.find(addr);
    MemoryBlock *block = iter->second;
    allocated_blocks_.erase(iter);

    block->allocated = false;
    try_merge_block(block);

    MemoryPool *pool = block->pool;
    pool->blocks.insert(block);
  }

 private:
  const CachingAllocatorOptions options_;

  std::shared_ptr<IAllocator> base_;

  std::mutex mutex_;
  MemoryPool large_pool_;
  MemoryPool small_pool_;
  absl::flat_hash_map<void *, MemoryBlock *> allocated_blocks_;
};

BfcAllocator::BfcAllocator(std::shared_ptr<IAllocator> base, CachingAllocatorOptions options)
    : impl_(std::make_unique<BfcAllocatorImpl>(base, options)) {}

BfcAllocator::~BfcAllocator() {}

absl::StatusOr<void *> BfcAllocator::allocate(size_t size) { return impl_->allocate(size); }

void BfcAllocator::deallocate(void *addr) { impl_->deallocate(addr); }

}  // namespace waymax_cpp
