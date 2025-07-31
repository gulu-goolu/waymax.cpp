#pragma once

#include <algorithm>

#include "waymax.cpp/geometry/matrix.h"

namespace waymax_cpp {

struct AABB2d {
  float left, right, bottom, top;

  static constexpr AABB2d merge(const AABB2d &aabb1, const AABB2d &aabb2) {
    return AABB2d{
        std::min(aabb1.left, aabb2.left),
        std::max(aabb1.right, aabb2.right),
        std::min(aabb1.bottom, aabb2.bottom),
        std::max(aabb1.top, aabb2.top),
    };
  }

  constexpr Float2 center() const { return Float2{(left + right) * 0.5f, (top + bottom) * 0.5f}; }
};

struct Box2d {
  Float2 center;
  float width, height;

  Matrix2d rotation;

  Box2d(Float2 _center, float _width, float _height)
      : center(_center), width(_width), height(_height), rotation(Matrix2d::identity()) {}

  constexpr AABB2d aabb() const {
    const float x = width * 0.5f, y = height * 0.5f;
    Float2 borders[4] = {
        {x, y},
        {x, -y},
        {-x, y},
        {-x, -y},
    };

    // rotate
    for (auto &border : borders) {
      border = border * rotation;
    }

    // translate
    for (auto &border : borders) {
      border = border + center;
    }

    const float left =
        std::min(std::min(borders[0].x, borders[1].x), std::min(borders[2].x, borders[3].x));
    const float right =
        std::max(std::max(borders[0].x, borders[1].x), std::max(borders[2].x, borders[3].x));

    const float bottom =
        std::min(std::min(borders[0].y, borders[1].y), std::min(borders[2].y, borders[3].y));
    const float top =
        std::max(std::max(borders[0].y, borders[1].y), std::max(borders[2].y, borders[3].y));

    return AABB2d{left, right, bottom, top};
  }
};

constexpr bool is_rect_contains(float width, float height, Float2 p) {
  float max_x = width * 0.5f;
  float max_y = height * 0.5f;
  float min_x = -max_x;
  float min_y = -max_y;

  if (p.x > min_x && p.x < max_x && p.y > min_y && p.y < max_y) {
    return true;
  }

  return false;
}

constexpr bool is_rect_overlapped(float width, float height, float left, float right, float top,
                                  float bottom) {
  float max_x = width * 0.5f;
  float max_y = height * 0.5f;
  float min_x = -max_x;
  float min_y = -max_y;

  bool is_separate_horizontally = left > max_x || right < min_x;

  bool is_separate_vertically = bottom > max_y || top < min_y;

  return !(is_separate_horizontally || is_separate_vertically);
}

constexpr inline bool box2d_is_overlapped(const Box2d &a, const Box2d &b) {
  Float2 borders[4] = {
      {b.width * 0.5f, b.height * 0.5f},
      {b.width * 0.5f, -b.height * 0.5f},
      {-b.width * 0.5f, -b.height * 0.5f},
      {-b.width * 0.5f, b.height * 0.5f},
  };

  // 先旋转
  Matrix2d m = b.rotation.transpose() * a.rotation;
  for (auto &p : borders) {
    p = p * m;
  }

  // 再平移
  Float2 offset = {b.center.x - a.center.x, b.center.y - a.center.y};
  for (auto &p : borders) {
    p.x += offset.x;
    p.y += offset.y;
  }

  float left = std::min(std::min(borders[0].x, borders[1].x), std::min(borders[2].x, borders[3].x));
  float right =
      std::max(std::max(borders[0].x, borders[1].x), std::max(borders[2].x, borders[3].x));

  float bottom =
      std::min(std::min(borders[0].y, borders[1].y), std::min(borders[2].y, borders[3].y));
  float top = std::max(std::max(borders[0].y, borders[1].y), std::max(borders[2].y, borders[3].y));

  return is_rect_overlapped(a.width, a.height, left, right, top, bottom);
}

}  // namespace waymax_cpp
