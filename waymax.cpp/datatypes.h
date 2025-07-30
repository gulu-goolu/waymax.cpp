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
};
}  // namespace waymax_cpp
