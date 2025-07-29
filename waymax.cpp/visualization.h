#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "waymax.cpp/geometry/box2d.h"
#include "waymax.cpp/geometry/matrix.h"

namespace waymax_cpp {
struct ClipRect {
  Float2 offset;
  float width;
  float height;
};

struct Color {};

class Bitmap {
 public:
  Bitmap(uint32_t width, uint32_t height)
      : width_(width), height_(height), colors_(width * height, Color()) {}

  void draw_line(Float2 p0, Float2 p1);

 private:
  const uint32_t width_;
  const uint32_t height_;
  std::vector<Color> colors_;
};

std::shared_ptr<Bitmap> vis_draw(ClipRect clip, absl::Span<const Box2d> boxes,
                                 absl::Span<const Color> boxes_color, uint32_t width,
                                 uint32_t height);
}  // namespace waymax_cpp
