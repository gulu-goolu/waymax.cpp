#pragma once

#include <cstddef>

namespace waymax_cpp {

template <size_t ndim>
class Action {
 public:
  float data[ndim];
  float valid;
};

class TrajectoryUpdate {
 public:
  float x;
  float y;
  float yaw;
  float vel_x;
  float vel_y;
  float valid;
};
}  // namespace waymax_cpp
