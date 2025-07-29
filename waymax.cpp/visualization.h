#pragma once

#include <cstdint>
#include <memory>

#include "absl/types/span.h"
#include "waymax.cpp/geometry/box2d.h"
#include "waymax.cpp/geometry/matrix.h"

namespace waymax_cpp {
struct ClipRect {
  Float2 offset;
  float width;
  float height;
};

class Bitmap {
 public:
 private:
};

struct Color {};

std::shared_ptr<Bitmap> vis_draw(ClipRect clip, absl::Span<const Box2d> boxes,
                                 absl::Span<const Color> boxes_color, uint32_t width,
                                 uint32_t height);
}  // namespace waymax_cpp
