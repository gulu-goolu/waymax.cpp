#include <cstddef>
#include <cstdint>

#include "waymax.cpp/geometry/box2d.cuh"
#include "waymax.cpp/geometry/box2d.h"

namespace waymax_cpp {
__global__ void box2d_overlap_kernel(uint32_t num_thread, const Box2d *boxes,
                                     const OverlapTestTask *tasks, bool *results) {
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_thread) {
    return;
  }

  const auto &box1 = boxes[tasks[idx].box1_id];
  const auto &box2 = boxes[tasks[idx].box2_id];

  results[idx] = box2d_is_overlapped(box1, box2);
}

absl::Status box2d_overlap_test(cudaStream_t stream, const Box2d *boxes, uint32_t num_task,
                                const OverlapTestTask *tasks, bool *results) {
  constexpr size_t block_size = 256;
  const size_t num_grid = (num_task + block_size - 1) / block_size;

  box2d_overlap_kernel<<<num_grid, block_size, 0, stream>>>(num_task, boxes, tasks, results);

  return absl::OkStatus();
}
}  // namespace waymax_cpp
