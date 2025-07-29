#include <iostream>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "waymax.cpp/visualization.h"

namespace waymax_cpp {
TEST(vis_test, bitmap) {
  auto bitmap = std::make_shared<Bitmap>(1920, 1024);
  bitmap->clear(Color::kColorBlack);

  std::string png_blob = bitmap->as_blob("png");
  std::cout << "blob.size: " << png_blob.size() << std::endl;
}
}  // namespace waymax_cpp

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
