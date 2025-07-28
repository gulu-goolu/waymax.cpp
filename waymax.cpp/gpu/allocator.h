#pragma once

#include <cstddef>
#include <memory>

#include "absl/status/statusor.h"

namespace waymax_cpp {
struct CachingAllocatorOptions {
  // allocate 调用所返回内存的 alignemtn
  size_t alloc_alignemnt = 512;

  // 当大于此参数时，从 large_pool 分配。
  // 当申请的内存小于 large_malloc 时，从 small pool 分配
  size_t large_alloc = 512 * 1024;

  size_t large_round = 32 * 1024;

  // 内存最大的 block size 大小
  size_t max_block_size = 2L * 1024L * 1024L * 1024L;
};

class IAllocator : public std::enable_shared_from_this<IAllocator> {
 public:
  virtual ~IAllocator() = default;

  // allocate memory
  virtual absl::StatusOr<void *> allocate(size_t size) = 0;

  // deallocate memory
  virtual void deallocate(void *addr) = 0;
};

// 基于 best fit 实现的 allocator
class BfcAllocatorImpl;
class BfcAllocator : public IAllocator {
 public:
  BfcAllocator(std::shared_ptr<IAllocator> base,
               CachingAllocatorOptions options = CachingAllocatorOptions());

  ~BfcAllocator() override;

  absl::StatusOr<void *> allocate(size_t size) override;

  void deallocate(void *addr) override;

 private:
  std::unique_ptr<BfcAllocatorImpl> impl_;
};
}  // namespace waymax_cpp
