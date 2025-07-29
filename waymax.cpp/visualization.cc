#include "waymax.cpp/visualization.h"

#include <cstddef>
#include <memory>
#include <string>

#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "waymax.cpp/geometry/matrix.h"

namespace waymax_cpp {

// --------------------------------- Color ----------------------------------------

Color Color::kColorBlack = {.r = 0, .g = 0, .b = 0, .a = 255};

// --------------------------------- Bitmap ---------------------------------------

void Bitmap::clear(Color color) {
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      colors_[y * width_ + x] = color;
    }
  }
}

void Bitmap::draw_line(Float2 p0, Float2 p1, Color color) {}

void Bitmap::draw_rect(absl::Span<const Float2> borders, Color color) {
  draw_line(borders[0], borders[1], color);
  draw_line(borders[1], borders[2], color);
  draw_line(borders[2], borders[3], color);
  draw_line(borders[3], borders[0], color);
}

std::string Bitmap::as_blob(const std::string& format) {
  const auto write_callback = [](void* context, void* data, int size) {
    reinterpret_cast<std::string*>(context)->append(reinterpret_cast<char*>(data), size);
  };

  if (format == "png") {
    std::string blob;

    stbi_write_png_to_func(write_callback, &blob, width_, height_, 4, colors_.data(), 4);

    return blob;
  } else if (format == "jpg") {
    std::string blob;

    stbi_write_jpg_to_func(write_callback, &blob, width_, height_, 4, colors_.data(), 100);

    return blob;
  }

  return "";
}

// --------------------------------- vis_draw -------------------------------------

std::shared_ptr<Bitmap> vis_draw(ClipRect clip, absl::Span<const Box2d> boxes,
                                 absl::Span<const Color> colors, uint32_t width, uint32_t height) {
  auto bitmap = std::make_shared<Bitmap>(width, height);

  const float clip_width = clip.right - clip.left;
  const float clip_height = clip.top - clip.boottom;

  const Float2 scale = {float(width) / clip_width, float(height) / clip_height};

  for (size_t idx = 0; idx < boxes.size(); ++idx) {
    const auto& box = boxes[idx];

    float x = box.width * 0.5f, y = box.height * 0.5f;
    Float2 borders[4] = {
        {x, y},
        {x, -y},
        {-x, y},
        {-x, -y},
    };

    for (auto& border : borders) {
      border = border * box.rotation;
    }

    bool clipped = true;

    for (auto& border : borders) {
      border = border + box.center;

      if (clip.contains(border)) {
        clipped = false;
        break;
      }
    }

    if (clipped) {
      continue;
    }

    for (auto& border : borders) {
      border.x -= clip.left;
      border.y -= clip.boottom;
    }

    for (auto& border : borders) {
      border = border * scale;
    }

    bitmap->draw_rect(borders, colors[idx]);
  }

  return bitmap;
}
}  // namespace waymax_cpp
