#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "waymax.cpp/geometry/box2d.h"
#include "waymax.cpp/geometry/matrix.h"

namespace waymax_cpp {
struct ClipRect {
  float left, right;
  float top, boottom;

  constexpr bool contains(Float2 p) const {
    if (p.x > left && p.x < right && p.y > boottom && p.y < top) {
      return true;
    }

    return false;
  }
};

struct Color {
  uint8_t r = 0, g = 0, b = 0, a = 0;

  static Color kColorBlack;
};

class Bitmap {
 public:
  Bitmap(uint32_t width, uint32_t height)
      : width_(width), height_(height), colors_(width * height, Color()) {}

  void clear(Color color);

  void draw_line(Float2 p0, Float2 p1, Color color);

  void draw_rect(absl::Span<const Float2> borders, Color color);

  // "png", "jpeg"
  std::string as_blob(const std::string& format);

 private:
  const uint32_t width_;
  const uint32_t height_;
  std::vector<Color> colors_;
};

std::shared_ptr<Bitmap> vis_draw(ClipRect clip, absl::Span<const Box2d> boxes,
                                 absl::Span<const Color> colors, uint32_t width, uint32_t height);
}  // namespace waymax_cpp
