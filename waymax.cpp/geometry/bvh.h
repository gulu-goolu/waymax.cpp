#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "waymax.cpp/geometry/box2d.h"

namespace waymax_cpp {

class Bvh2d {
 public:
  constexpr AABB2d aabb() const { return aabb_; }

  static std::shared_ptr<Bvh2d> build(const Box2d* boxes, absl::Span<uint32_t> ids);

 private:
  AABB2d aabb_;

  std::vector<uint32_t> ids_;

  std::shared_ptr<Bvh2d> left_;
  std::shared_ptr<Bvh2d> right_;
};
}  // namespace waymax_cpp
