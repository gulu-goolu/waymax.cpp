#pragma once

#include <cmath>

namespace waymax_cpp {
struct Float2 {
  float x = 0, y = 0;
};

constexpr Float2 operator*(const Float2& lhs, const Float2& rhs) {
  return Float2{lhs.x * rhs.x, lhs.y * rhs.y};
}

// row major matrix
struct Matrix2d {
  float m[2][2] = {};

  constexpr Matrix2d operator*(const Matrix2d& other) const {
    Matrix2d result;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        result.m[i][j] = 0;
        for (int k = 0; k < 2; ++k) {
          result.m[i][j] += m[i][k] * other.m[k][j];
        }
      }
    }

    return result;
  }

  constexpr Matrix2d transpose() const {
    Matrix2d result;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        result.m[i][j] = m[j][i];
      }
    }
    return result;
  }

  static Matrix2d rotate(float theta) {
    Matrix2d result;
    result.m[0][0] = std::cos(theta);
    result.m[0][1] = -std::sin(theta);
    result.m[1][0] = std::sin(theta);
    result.m[1][1] = std::cos(theta);
    return result;
  }

  static Matrix2d identity() {
    return Matrix2d{
        .m =
            {
                {1, 0},
                {0, 1},
            },
    };
  }
};

inline constexpr Float2 operator*(const Float2& v, const Matrix2d& matrix) {
  Float2 result;
  result.x = v.x * matrix.m[0][0] + v.y * matrix.m[0][1];
  result.y = v.x * matrix.m[1][0] + v.y * matrix.m[1][1];
  return result;
}

}  // namespace waymax_cpp
