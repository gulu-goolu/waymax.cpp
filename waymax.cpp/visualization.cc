#include "waymax.cpp/visualization.h"

#include "waymax.cpp/geometry/matrix.h"

namespace waymax_cpp {

// --------------------------------- Bitmap ---------------------------------------

void Bitmap::draw_line(Float2 p0, Float2 p1) {}

// --------------------------------- vis_draw -------------------------------------

std::shared_ptr<Bitmap> vis_draw(ClipRect clip, absl::Span<const Box2d> boxes,
                                 absl::Span<const Color> boxes_color, uint32_t width,
                                 uint32_t height) {
  return nullptr;
}
}  // namespace waymax_cpp
