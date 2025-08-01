#include "waymax.cpp/geometry/bvh.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/random/random.h"
#include "gtest/gtest.h"
#include "waymax.cpp/geometry/box2d.h"
#include "waymax.cpp/geometry/matrix.h"

namespace waymax_cpp {
TEST(bvh, test) {
  constexpr size_t count = 100000;

  absl::BitGen bitgen;

  std::vector<Box2d> boxes;
  for (size_t i = 0; i < count; ++i) {
    float center_x = absl::Uniform<float>(bitgen, -10000.f, 10000.f);
    float center_y = absl::Uniform<float>(bitgen, -10000.f, 10000.f);

    float width = absl::Uniform<float>(bitgen, 0.2, 2.2);
    float height = absl::Uniform<float>(bitgen, 0.2, 2.2);

    float theta = absl::Uniform<float>(bitgen, 0, 3.1415f);

    Box2d box(Float2{center_x, center_y}, width, height);
    box.rotation = Matrix2d::rotate(theta);

    boxes.push_back(box);
  }

  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < count; ++i) {
    ids.push_back(i);
  }

  auto tp0 = std::chrono::steady_clock::now();
  auto bvh = Bvh2d::build(boxes.data(), absl::MakeSpan(ids));
  auto tp1 = std::chrono::steady_clock::now();
  std::cout << "build bvh, elapsed_time: " << (tp1 - tp0).count() / 1.0e3 << " us" << std::endl;
}
}  // namespace waymax_cpp

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return ::RUN_ALL_TESTS();
}
