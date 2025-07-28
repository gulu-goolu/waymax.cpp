#pragma once

namespace waymax_cpp {
class TransferManager {
 public:
  static TransferManager *singleton();

  void run();

  void run_async();
};
}  // namespace waymax_cpp
