// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_IBILINEAR_MICROKERNEL_TESTER_H_
#define XNNPACK_TEST_IBILINEAR_MICROKERNEL_TESTER_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "test/replicable_random_device.h"

class IBilinearMicrokernelTester {
 public:
  IBilinearMicrokernelTester& pixels(uint32_t pixels) {
    assert(pixels >= 1);
    this->pixels_ = pixels;
    return *this;
  }

  uint32_t pixels() const { return this->pixels_; }

  IBilinearMicrokernelTester& channels(uint32_t channels) {
    assert(channels >= 1);
    this->channels_ = channels;
    return *this;
  }

  uint32_t channels() const { return this->channels_; }

  IBilinearMicrokernelTester& input_offset(uint32_t input_offset) {
    this->input_offset_ = input_offset;
    return *this;
  }

  uint32_t input_offset() const { return this->input_offset_; }

  IBilinearMicrokernelTester& output_stride(uint32_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  uint32_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  IBilinearMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  IBilinearMicrokernelTester& input_stride(uint32_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  uint32_t input_stride() const {
    if (this->input_stride_ == 0) {
      return 4 * pixels();
    } else {
      assert(this->input_stride_ >= 4 * pixels());
      return this->input_stride_;
    }
  }

  void Test(xnn_f16_ibilinear_ukernel_fn ibilinear) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    xnnpack::Buffer<const xnn_float16*> indirection(pixels() * 4);
    xnnpack::Buffer<xnn_float16> input(indirection.size() * channels(),
                                       xnnpack::XnnExtraBytes);
    xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> packed_weights(
        pixels() * 2);
    xnnpack::Buffer<xnn_float16> output((pixels() - 1) * output_stride() +
                                        channels());
    xnnpack::Buffer<float> output_ref(pixels() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(packed_weights.begin(), packed_weights.end(),
                    [&]() { return f32dist(rng); });

      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float alpha_h = packed_weights[i * 2 + 0];
          const float alpha_v = packed_weights[i * 2 + 1];
          output_ref[i * channels() + c] =
              indirection[i * 4 + 0][c + input_offset()] * (1.0f - alpha_h) *
                  (1.0f - alpha_v) +
              indirection[i * 4 + 1][c + input_offset()] * alpha_h *
                  (1.0f - alpha_v) +
              indirection[i * 4 + 2][c + input_offset()] * (1.0f - alpha_h) *
                  alpha_v +
              indirection[i * 4 + 3][c + input_offset()] * alpha_h * alpha_v;
        }
      }

      // Call optimized micro-kernel.
      ibilinear(pixels(), channels() * sizeof(xnn_float16),
                reinterpret_cast<const xnn_float16**>(indirection.data()),
                input_offset() * sizeof(xnn_float16), packed_weights.data(),
                output.data(),
                (output_stride() - channels()) * sizeof(xnn_float16));

      // Verify results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              output[i * output_stride() + c], output_ref[i * channels() + c],
              std::abs(output_ref[i * channels() + c]) * 1.0e-2f + 1.0e-4f)
              << "pixel " << i << " / " << pixels() << ", channel " << c
              << " / " << channels();
        }
      }
    }
  }

  void Test(xnn_f32_ibilinear_ukernel_fn ibilinear) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    xnnpack::Buffer<const float*> indirection(pixels() * 4);
    xnnpack::Buffer<float> input(indirection.size() * channels(),
                                 xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(pixels() *
                                                                    2);
    xnnpack::Buffer<float> output((pixels() - 1) * output_stride() +
                                  channels());
    xnnpack::Buffer<float> output_ref(pixels() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(packed_weights.begin(), packed_weights.end(),
                    [&]() { return f32dist(rng); });

      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float alpha_h = packed_weights[i * 2 + 0];
          const float alpha_v = packed_weights[i * 2 + 1];
          output_ref[i * channels() + c] =
              indirection[i * 4 + 0][c + input_offset()] * (1.0f - alpha_h) *
                  (1.0f - alpha_v) +
              indirection[i * 4 + 1][c + input_offset()] * alpha_h *
                  (1.0f - alpha_v) +
              indirection[i * 4 + 2][c + input_offset()] * (1.0f - alpha_h) *
                  alpha_v +
              indirection[i * 4 + 3][c + input_offset()] * alpha_h * alpha_v;
        }
      }

      // Call optimized micro-kernel.
      ibilinear(pixels(), channels() * sizeof(float), indirection.data(),
                input_offset() * sizeof(float), packed_weights.data(),
                output.data(), (output_stride() - channels()) * sizeof(float));

      // Verify results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              output_ref[i * channels() + c], output[i * output_stride() + c],
              std::abs(output_ref[i * channels() + c]) * 1.0e-4 + 1.0e-6f)
              << "pixel " << i << " / " << pixels() << ", channel " << c
              << " / " << channels();
        }
      }
    }
  }

  void Test(xnn_s8_ibilinear_ukernel_fn ibilinear) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    std::uniform_int_distribution<int16_t> w11dist(0, 2047);

    xnnpack::Buffer<const int8_t*> indirection(pixels() * 4);
    xnnpack::Buffer<int8_t> input(indirection.size() * channels(),
                                  xnnpack::XnnExtraBytes);
    xnnpack::Buffer<int16_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(pixels() *
                                                                      2);
    xnnpack::Buffer<int8_t> output((pixels() - 1) * output_stride() +
                                   channels());
    xnnpack::Buffer<int8_t> output_ref(pixels() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::generate(packed_weights.begin(), packed_weights.end(),
                    [&]() { return w11dist(rng); });

      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const int32_t alpha_h = packed_weights[i * 2 + 0];
          const int32_t alpha_v = packed_weights[i * 2 + 1];
          const int32_t acc = math_asr_s32(
              int32_t(indirection[i * 4 + 0][c + input_offset()]) *
                      (2048 - alpha_h) * (2048 - alpha_v) +
                  int32_t(indirection[i * 4 + 1][c + input_offset()]) *
                      alpha_h * (2048 - alpha_v) +
                  int32_t(indirection[i * 4 + 2][c + input_offset()]) *
                      (2048 - alpha_h) * alpha_v +
                  int32_t(indirection[i * 4 + 3][c + input_offset()]) *
                      alpha_h * alpha_v +
                  2097152,
              22);
          ASSERT_GE(acc, std::numeric_limits<int8_t>::min());
          ASSERT_LE(acc, std::numeric_limits<int8_t>::max());
          output_ref[i * channels() + c] = (int8_t)acc;
        }
      }

      // Call optimized micro-kernel.
      ibilinear(pixels(), channels() * sizeof(int8_t), indirection.data(),
                input_offset() * sizeof(int8_t), packed_weights.data(),
                output.data(), (output_stride() - channels()) * sizeof(int8_t));

      // Verify results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(int32_t(output_ref[i * channels() + c]),
                    int32_t(output[i * output_stride() + c]))
              << "pixel " << i << " / " << pixels() << ", channel " << c
              << " / " << channels();
        }
      }
    }
  }

  void Test(xnn_u8_ibilinear_ukernel_fn ibilinear) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max());
    std::uniform_int_distribution<int16_t> w11dist(0, 2047);

    xnnpack::Buffer<const uint8_t*> indirection(pixels() * 4);
    xnnpack::Buffer<uint8_t> input(indirection.size() * channels(),
                                   xnnpack::XnnExtraBytes);
    xnnpack::Buffer<int16_t, XNN_ALLOCATION_ALIGNMENT> packed_weights(pixels() *
                                                                      2);
    xnnpack::Buffer<uint8_t> output((pixels() - 1) * output_stride() +
                                    channels());
    xnnpack::Buffer<uint8_t> output_ref(pixels() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::generate(packed_weights.begin(), packed_weights.end(),
                    [&]() { return w11dist(rng); });

      for (size_t i = 0; i < indirection.size(); i++) {
        indirection[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const uint32_t alpha_h = uint32_t(int32_t(packed_weights[i * 2 + 0]));
          const uint32_t alpha_v = uint32_t(int32_t(packed_weights[i * 2 + 1]));
          const uint32_t acc =
              (2097152 +
               int32_t(indirection[i * 4 + 0][c + input_offset()]) *
                   (2048 - alpha_h) * (2048 - alpha_v) +
               int32_t(indirection[i * 4 + 1][c + input_offset()]) * alpha_h *
                   (2048 - alpha_v) +
               int32_t(indirection[i * 4 + 2][c + input_offset()]) *
                   (2048 - alpha_h) * alpha_v +
               int32_t(indirection[i * 4 + 3][c + input_offset()]) * alpha_h *
                   alpha_v) >>
              22;
          ASSERT_LE(acc, std::numeric_limits<uint8_t>::max());
          output_ref[i * channels() + c] = (uint8_t)acc;
        }
      }

      // Call optimized micro-kernel.
      ibilinear(pixels(), channels() * sizeof(uint8_t), indirection.data(),
                input_offset() * sizeof(uint8_t), packed_weights.data(),
                output.data(),
                (output_stride() - channels()) * sizeof(uint8_t));

      // Verify results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(uint32_t(output_ref[i * channels() + c]),
                    uint32_t(output[i * output_stride() + c]))
              << "pixel " << i << " / " << pixels() << ", channel " << c
              << " / " << channels();
        }
      }
    }
  }

  void TestCHW(xnn_f16_ibilinear_chw_ukernel_fn ibilinear) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    xnnpack::Buffer<const xnn_float16*> indirection(pixels() * 2);
    xnnpack::Buffer<xnn_float16> input(
        (channels() - 1) * input_stride() + 4 * pixels(),
        xnnpack::XnnExtraBytes);
    xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> packed_weights(
        pixels() * 2);
    xnnpack::Buffer<xnn_float16> output(pixels() * channels());
    xnnpack::Buffer<float> output_ref(pixels() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(packed_weights.begin(), packed_weights.end(),
                    [&]() { return f32dist(rng); });

      // Indirection will point to the even ("left") pixels of the input.
      // The kernels will expect "right" pixels to be placed right next to them.
      for (size_t i = 0; i < indirection.size(); i++) {
        const xnn_float16* left_corner = input.data() + 2 * i - input_offset();
        indirection[i] = left_corner;
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float alpha_h = packed_weights[i * 2 + 0];
          const float alpha_v = packed_weights[i * 2 + 1];
          // `c * pixels() + i` because the output is NCHW.
          output_ref[c * pixels() + i] =
              // `c * indirection.size()` because the input is NCHW.
              (indirection[i * 2 + 0] +
               0)[c * input_stride() + input_offset()] *
                  (1.0f - alpha_h) * (1.0f - alpha_v) +
              (indirection[i * 2 + 0] +
               1)[c * input_stride() + input_offset()] *
                  alpha_h * (1.0f - alpha_v) +
              (indirection[i * 2 + 1] +
               0)[c * input_stride() + input_offset()] *
                  (1.0f - alpha_h) * alpha_v +
              (indirection[i * 2 + 1] +
               1)[c * input_stride() + input_offset()] *
                  alpha_h * alpha_v;
        }
      }

      // Call optimized micro-kernel.
      ibilinear(pixels(), channels(),
                reinterpret_cast<const xnn_float16**>(indirection.data()),
                input_offset() * sizeof(xnn_float16), packed_weights.data(),
                output.data(), input_stride() * sizeof(xnn_float16));

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        for (size_t i = 0; i < pixels(); i++) {
          ASSERT_NEAR(
              output[c * pixels() + i], output_ref[c * pixels() + i],
              std::abs(output_ref[c * pixels() + i]) * 1.0e-2f + 1.0e-4f)
              << "i = " << i << ", channel = " << c;
        }
      }
    }
  }

  void TestCHW(xnn_f32_ibilinear_chw_ukernel_fn ibilinear) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    xnnpack::Buffer<const float*> indirection(pixels() * 2);
    xnnpack::Buffer<float> input(
        (channels() - 1) * input_stride() + 4 * pixels(),
        xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(pixels() *
                                                                    2);
    xnnpack::Buffer<float> output(pixels() * channels());
    xnnpack::Buffer<float> output_ref(pixels() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(packed_weights.begin(), packed_weights.end(),
                    [&]() { return f32dist(rng); });

      // Indirection will point to the even ("left") pixels of the input.
      // The kernels will expect "right" pixels to be placed right next to them.
      for (size_t i = 0; i < indirection.size(); i++) {
        const float* left_corner = input.data() + 2 * i - input_offset();
        indirection[i] = left_corner;
      }
      std::shuffle(indirection.begin(), indirection.end(), rng);

      // Compute reference results.
      for (size_t i = 0; i < pixels(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const float alpha_h = packed_weights[i * 2 + 0];
          const float alpha_v = packed_weights[i * 2 + 1];
          // `c * pixels() + i` because the output is NCHW.
          output_ref[c * pixels() + i] =
              // `c * indirection.size()` because the input is NCHW.
              (indirection[i * 2 + 0] +
               0)[c * input_stride() + input_offset()] *
                  (1.0f - alpha_h) * (1.0f - alpha_v) +
              (indirection[i * 2 + 0] +
               1)[c * input_stride() + input_offset()] *
                  alpha_h * (1.0f - alpha_v) +
              (indirection[i * 2 + 1] +
               0)[c * input_stride() + input_offset()] *
                  (1.0f - alpha_h) * alpha_v +
              (indirection[i * 2 + 1] +
               1)[c * input_stride() + input_offset()] *
                  alpha_h * alpha_v;
        }
      }

      // Call optimized micro-kernel.
      ibilinear(pixels(), channels(), indirection.data(),
                input_offset() * sizeof(float), packed_weights.data(),
                output.data(), input_stride() * sizeof(float));

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        for (size_t i = 0; i < pixels(); i++) {
          ASSERT_NEAR(
              output_ref[c * pixels() + i], output[c * pixels() + i],
              std::abs(output_ref[c * pixels() + i]) * 1.0e-3f + 1.0e-6f)
              << "i = " << i << ", channel = " << c;
        }
      }
    }
  }

 private:
  uint32_t channels_{1};
  uint32_t pixels_{1};
  uint32_t output_stride_{0};
  uint32_t input_stride_{0};
  uint32_t input_offset_{0};
  size_t iterations_{3};
};

#endif  // XNNPACK_TEST_IBILINEAR_MICROKERNEL_TESTER_H_
