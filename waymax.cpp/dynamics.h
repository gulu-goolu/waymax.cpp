#pragma once

namespace waymax_cpp {
class DynamicsModel {
 public:
  virtual ~DynamicsModel() = default;

  virtual void update() = 0;

  virtual void inverse() = 0;
};

class StateDynamics {
 public:
};

class InvertibleBicycleModel {
 public:
};
}  // namespace waymax_cpp
