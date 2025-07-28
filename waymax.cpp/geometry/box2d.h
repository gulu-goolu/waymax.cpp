#pragma once

#include "waymax.cpp/geometry/matrix.h"

namespace waymax_cpp {

struct Box2d {
  Float2 center;
  float width, height;

  Matrix2d rotation;

  Box2d(Float2 _center, float _width, float _height)
      : center(_center), width(_width), height(_height), rotation(Matrix2d::identity()) {}
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

  for (auto p : borders) {
    if (is_rect_contains(a.width, a.height, p)) {
      return true;
    }
  }

  return false;
}

}  // namespace waymax_cpp
