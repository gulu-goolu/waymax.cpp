#include "waymax.cpp/geometry/bvh.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "waymax.cpp/geometry/box2d.h"

namespace waymax_cpp {
std::shared_ptr<Bvh2d> Bvh2d::build(const Box2d *boxes, absl::Span<uint32_t> ids) {
  // 计算包围盒
  AABB2d aabb = {std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
                 std::numeric_limits<float>::max(), std::numeric_limits<float>::min()};

  for (uint32_t id : ids) {
    aabb = AABB2d::merge(aabb, boxes[id].aabb());
  }

  // 如果 box 的数目小于 8，创建一个叶子节点
  if (ids.size() < 8) {
    auto result = std::make_shared<Bvh2d>();
    for (size_t i = 0; i < ids.size(); ++i) {
      result->ids_.push_back(ids[i]);
    }
    result->aabb_ = aabb;

    return result;
  } else {
    if (aabb.width() > aabb.height()) {
      std::sort(ids.begin(), ids.end(), [boxes](uint32_t id1, uint32_t id2) {
        return boxes[id1].center.x < boxes[id2].center.x;
      });
    } else {
      std::sort(ids.begin(), ids.end(), [boxes](uint32_t id1, uint32_t id2) {
        return boxes[id1].center.y < boxes[id2].center.y;
      });
    }

    size_t pos = ids.size() / 2;

    auto result = std::make_shared<Bvh2d>();
    result->aabb_ = aabb;
    result->left_ = build(boxes, ids.subspan(0, pos));
    result->right_ = build(boxes, ids.subspan(pos));

    return nullptr;
  }
}
}  // namespace waymax_cpp
