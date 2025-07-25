// clang-format off
// Auto-generated file. Do not edit!
//   Template: test/simd/s32-simd.cc.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// This header needs to go first for the arch test macros.
#include "src/xnnpack/common.h"

#if XNN_ARCH_X86 || XNN_ARCH_X86_64

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/simd/s32-sse41.h"
#include "test/replicable_random_device.h"

namespace xnnpack {

class S32SimdSSE41Test : public ::testing::Test {
 protected:
  void SetUp() override {
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_x86_sse4_1);
    inputs_.resize(3 * xnn_simd_size_s32);
    output_.resize(xnn_simd_size_s32);
    std::uniform_int_distribution<int32_t> s32dist(-10000, 10000);
    std::generate(inputs_.begin(), inputs_.end(),
                  [&]() { return s32dist(rng_); });
  }

  xnnpack::ReplicableRandomDevice rng_;
  std::vector<int32_t> inputs_;
  std::vector<int32_t> output_;
};

TEST_F(S32SimdSSE41Test, Mul) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_mul_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] * inputs_[k + xnn_simd_size_s32]);
  }
}

TEST_F(S32SimdSSE41Test, Sub) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_sub_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], inputs_[k] - inputs_[k + xnn_simd_size_s32]);
  }
}

TEST_F(S32SimdSSE41Test, Min) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_min_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], std::min(inputs_[k], inputs_[k + xnn_simd_size_s32]));
  }
}

TEST_F(S32SimdSSE41Test, Max) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  const xnn_simd_s32_t b = xnn_loadu_s32(inputs_.data() + xnn_simd_size_s32);
  const xnn_simd_s32_t res = xnn_max_s32(a, b);
  xnn_storeu_s32(output_.data(), res);
  for (size_t k = 0; k < xnn_simd_size_s32; k++) {
    ASSERT_EQ(output_[k], std::max(inputs_[k], inputs_[k + xnn_simd_size_s32]));
  }
}

TEST_F(S32SimdSSE41Test, StoreTail) {
  const xnn_simd_s32_t a = xnn_loadu_s32(inputs_.data());
  for (size_t num_elements = 1; num_elements < xnn_simd_size_s32;
      num_elements++) {
    xnn_store_tail_s32(output_.data(), a, num_elements);
    for (size_t k = 0; k < num_elements; k++) {
      ASSERT_EQ(output_[k], inputs_[k]);
    }
    for (size_t k = num_elements; k < xnn_simd_size_s32; k++) {
      ASSERT_EQ(output_[k], 0.0f);
    }
  }
}
}  // namespace xnnpack

#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
