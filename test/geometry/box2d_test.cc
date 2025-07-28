#include "waymax.cpp/geometry/box2d.h"

#include <iostream>

#include "absl/log/absl_check.h"
#include "cuda_runtime_api.h"
#include "gtest/gtest.h"
#include "waymax.cpp/geometry/box2d.cuh"
#include "waymax.cpp/geometry/matrix.h"

namespace waymax_cpp {
bool is_overlapped(Box2d a, Box2d b) {
  Box2d boxes[] = {a, b};

  // copy to gpu
  void* boxes_gpu_addr = nullptr;
  ABSL_CHECK_EQ(cudaSuccess, cudaMalloc(&boxes_gpu_addr, sizeof(boxes)));
  ABSL_CHECK_EQ(cudaSuccess,
                cudaMemcpy(boxes_gpu_addr, boxes, sizeof(boxes), cudaMemcpyHostToDevice));

  OverlapTestTask tasks[] = {{0, 1}};
  void* task_gpu_addr = nullptr;
  ABSL_CHECK_EQ(cudaSuccess, cudaMalloc(&task_gpu_addr, sizeof(tasks)));
  ABSL_CHECK_EQ(cudaSuccess,
                cudaMemcpy(task_gpu_addr, tasks, sizeof(tasks), cudaMemcpyHostToDevice));

  void* result_gpu_addr = nullptr;
  ABSL_CHECK_EQ(cudaSuccess, cudaMalloc(&result_gpu_addr, 1));

  auto st =
      box2d_overlap_test(0, absl::MakeConstSpan(reinterpret_cast<const Box2d*>(boxes_gpu_addr), 2),
                         absl::MakeSpan(reinterpret_cast<const OverlapTestTask*>(task_gpu_addr), 1),
                         absl::MakeSpan(reinterpret_cast<bool*>(result_gpu_addr), 1));
  ABSL_CHECK(st.ok()) << st.ToString();

  bool results[1] = {};
  ABSL_CHECK_EQ(cudaSuccess, cudaMemcpy(results, result_gpu_addr, 1, cudaMemcpyDeviceToHost))
      << "fail to copy result";

  ABSL_CHECK_EQ(cudaSuccess, cudaFree(boxes_gpu_addr));
  ABSL_CHECK_EQ(cudaSuccess, cudaFree(task_gpu_addr));
  ABSL_CHECK_EQ(cudaSuccess, cudaFree(result_gpu_addr));

  return results[0];
}

TEST(box2d_test, rect_contains) {
  ASSERT_TRUE(is_rect_contains(1, 1, Float2{0.2, 0.2}));
  ASSERT_FALSE(is_rect_contains(1, 1, Float2{0.6, 0.6}));
}

TEST(box2d_test, overlapped) {
  ASSERT_FALSE(box2d_is_overlapped(Box2d(Float2{0, 0}, 1, 1), Box2d(Float2{2, 0}, 1, 1)));
  ASSERT_TRUE(box2d_is_overlapped(Box2d(Float2{0, 0}, 1, 1), Box2d(Float2{0.5, 0}, 1, 1)));
}

TEST(box2d_test, gpu) {
  ASSERT_FALSE(is_overlapped(Box2d(Float2{0, 0}, 1, 1), Box2d(Float2{2, 0}, 1, 1)));
  ASSERT_TRUE(is_overlapped(Box2d(Float2{0, 0}, 1, 1), Box2d(Float2{0.5, 0}, 1, 1)));
}
}  // namespace waymax_cpp

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return ::RUN_ALL_TESTS();
}
