#pragma once

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "driver_types.h"
#include "waymax.cpp/geometry/box2d.h"

namespace waymax_cpp {
struct OverlapTestTask {
  uint32_t box1_id, box2_id;
};

// 判断 boxes 数组是否有重叠的部分
absl::Status box2d_overlap_test(cudaStream_t stream, const Box2d* boxes, uint32_t num_task,
                                const OverlapTestTask* tasks, bool* results);

// 判断 boxes 之间是否存在相互重叠
//
// * 会在 cpu 上构建一个 bvh 树，然后用 gpu 判断 box 和 box 之间是否存在重叠
bool box2d_is_overlapped(int32_t device_ordinal, absl::Span<const Box2d> boxes);

}  // namespace waymax_cpp
