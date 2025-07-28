#include "waymax.cpp/geometry/box2d.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

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

  auto st = box2d_overlap_test(0, reinterpret_cast<const Box2d*>(boxes_gpu_addr), 1,
                               reinterpret_cast<const OverlapTestTask*>(task_gpu_addr),
                               reinterpret_cast<bool*>(result_gpu_addr));
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

TEST(Box2d_test, bench) {
  const int64_t repeats = 1000 * 1000;
  const Box2d a = Box2d(Float2{0, 0}, 1, 1), b = Box2d(Float2{0.5, 0}, 1, 1);
  Box2d boxes[] = {a, b};

  auto cpu_start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; ++i) {
    volatile bool overlapped =
        box2d_is_overlapped(boxes[0], boxes[1]) && box2d_is_overlapped(boxes[1], boxes[0]);
    ASSERT_TRUE(overlapped);
  }
  auto cpu_end = std::chrono::steady_clock::now();
  std::cout << "elapsed time, cpu: " << (cpu_end - cpu_start).count() / 1.0e3 << " us" << std::endl;

  // gpu
  void* boxes_gpu_addr = nullptr;
  ABSL_CHECK_EQ(cudaSuccess, cudaMalloc(&boxes_gpu_addr, sizeof(boxes)));
  ABSL_CHECK_EQ(cudaSuccess,
                cudaMemcpy(boxes_gpu_addr, boxes, sizeof(boxes), cudaMemcpyHostToDevice));

  std::vector<OverlapTestTask> tasks(repeats, OverlapTestTask{0, 1});
  void* task_gpu_addr = nullptr;
  ABSL_CHECK_EQ(cudaSuccess, cudaMalloc(&task_gpu_addr, sizeof(OverlapTestTask) * repeats));
  ABSL_CHECK_EQ(cudaSuccess, cudaMemcpy(task_gpu_addr, tasks.data(),
                                        sizeof(OverlapTestTask) * repeats, cudaMemcpyHostToDevice));

  void* result_gpu_addr = nullptr;
  ABSL_CHECK_EQ(cudaSuccess, cudaMalloc(&result_gpu_addr, repeats));

  ABSL_CHECK_EQ(cudaSuccess, cudaStreamSynchronize(0));

  auto gpu_start = std::chrono::steady_clock::now();

  auto st = box2d_overlap_test(0, reinterpret_cast<const Box2d*>(boxes_gpu_addr), repeats,
                               reinterpret_cast<const OverlapTestTask*>(task_gpu_addr),
                               reinterpret_cast<bool*>(result_gpu_addr));
  ABSL_CHECK(st.ok());
  ABSL_CHECK_EQ(cudaSuccess, cudaStreamSynchronize(0));

  auto gpu_stop = std::chrono::steady_clock::now();

  std::cout << "elapsed time, gpu: " << (gpu_stop - gpu_start).count() / 1.0e3 << " us"
            << std::endl;

  ABSL_CHECK_EQ(cudaSuccess, cudaFree(boxes_gpu_addr));
  ABSL_CHECK_EQ(cudaSuccess, cudaFree(task_gpu_addr));
  ABSL_CHECK_EQ(cudaSuccess, cudaFree(result_gpu_addr));
}

}  // namespace waymax_cpp

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return ::RUN_ALL_TESTS();
}
