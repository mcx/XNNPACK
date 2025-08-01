// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_MAXPOOL_MICROKERNEL_TESTER_H_
#define XNNPACK_TEST_MAXPOOL_MICROKERNEL_TESTER_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams.h"
#include "test/next_prime.h"
#include "test/replicable_random_device.h"

class MaxPoolMicrokernelTester {
 public:
  MaxPoolMicrokernelTester() = default;

  MaxPoolMicrokernelTester& output_pixels(size_t output_pixels) {
    assert(output_pixels != 0);
    this->output_pixels_ = output_pixels;
    return *this;
  }

  size_t output_pixels() const { return this->output_pixels_; }

  MaxPoolMicrokernelTester& step(size_t step) {
    assert(step != 0);
    this->step_ = step;
    return *this;
  }

  size_t step() const { return this->step_; }

  MaxPoolMicrokernelTester& input_offset(size_t input_offset) {
    assert(input_offset != 0);
    this->input_offset_ = input_offset;
    return *this;
  }

  size_t input_offset() const { return this->input_offset_; }

  MaxPoolMicrokernelTester& pooling_elements(size_t pooling_elements) {
    assert(pooling_elements != 0);
    this->pooling_elements_ = pooling_elements;
    return *this;
  }

  size_t pooling_elements() const { return this->pooling_elements_; }

  MaxPoolMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const { return this->channels_; }

  MaxPoolMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  MaxPoolMicrokernelTester& qmin(int16_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  int16_t qmin() const { return this->qmin_; }

  MaxPoolMicrokernelTester& qmax(int16_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  int16_t qmax() const { return this->qmax_; }

  MaxPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  void Test(xnn_s8_maxpool_ukernel_fn maxpool,
            xnn_init_s8_minmax_params_fn init_params) const {
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    xnnpack::Buffer<const int8_t*> indirect_input(
        (output_pixels() - 1) * step() + pooling_elements());
    xnnpack::Buffer<int8_t> input(indirect_input.size() * channels(),
                                  xnnpack::XnnExtraBytes);
    xnnpack::Buffer<int8_t> output(
        (output_pixels() - 1) * output_stride() + channels(),
        xnnpack::XnnExtraBytes);
    xnnpack::Buffer<int8_t> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(),
                      [&]() { return i8dist(rng); });
      } while (input.size() > 1 &&
               *std::max_element(input.cbegin(), input.cend()) ==
                   *std::min_element(input.cbegin(), input.cend()));

      for (size_t i = 0;
           i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
                   indirect_input.begin() + (output_pixels() - 1) * step() +
                       pooling_elements(),
                   rng);

      // Prepare parameters.
      xnn_s8_minmax_params params;
      init_params(&params, static_cast<int8_t>(qmin()),
                  static_cast<int8_t>(qmax()));

      // Compute reference results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          int8_t max_value = std::numeric_limits<int8_t>::min();
          for (size_t p = 0; p < pooling_elements(); p++) {
            max_value = std::max(
                max_value, indirect_input[x * step() + p][c + input_offset()]);
          }
          max_value = std::min(max_value, static_cast<int8_t>(qmax()));
          max_value = std::max(max_value, static_cast<int8_t>(qmin()));
          output_ref[x * channels() + c] = max_value;
        }
      }

      // Call optimized micro-kernel.
      maxpool(output_pixels(), pooling_elements(), channels(),
              indirect_input.data(), input_offset() * sizeof(int8_t),
              /*input_pixel_stride=*/0, output.data(), step() * sizeof(void*),
              output_stride() * sizeof(int8_t), &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(int16_t(output[x * output_stride() + c]), qmin())
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
          ASSERT_LE(int16_t(output[x * output_stride() + c]), qmax())
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
          ASSERT_EQ(int32_t(output_ref[x * channels() + c]),
                    int32_t(output[x * output_stride() + c]))
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_u8_maxpool_ukernel_fn maxpool,
            xnn_init_u8_minmax_params_fn init_params) const {
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max());

    xnnpack::Buffer<const uint8_t*> indirect_input(
        (output_pixels() - 1) * step() + pooling_elements());
    xnnpack::Buffer<uint8_t> input(indirect_input.size() * channels(),
                                   xnnpack::XnnExtraBytes);
    xnnpack::Buffer<uint8_t> output(
        (output_pixels() - 1) * output_stride() + channels(),
        xnnpack::XnnExtraBytes);
    xnnpack::Buffer<uint8_t> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(),
                      [&]() { return u8dist(rng); });
      } while (input.size() > 1 &&
               *std::max_element(input.cbegin(), input.cend()) ==
                   *std::min_element(input.cbegin(), input.cend()));

      for (size_t i = 0;
           i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
                   indirect_input.begin() + (output_pixels() - 1) * step() +
                       pooling_elements(),
                   rng);

      // Prepare parameters.
      xnn_u8_minmax_params params;
      init_params(&params, static_cast<uint8_t>(qmin()),
                  static_cast<uint8_t>(qmax()));

      // Compute reference results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          uint8_t max_value = 0;
          for (size_t p = 0; p < pooling_elements(); p++) {
            max_value = std::max(
                max_value, indirect_input[x * step() + p][c + input_offset()]);
          }
          max_value = std::min(max_value, static_cast<uint8_t>(qmax()));
          max_value = std::max(max_value, static_cast<uint8_t>(qmin()));
          output_ref[x * channels() + c] = max_value;
        }
      }

      // Call optimized micro-kernel.
      maxpool(output_pixels(), pooling_elements(), channels(),
              indirect_input.data(), input_offset() * sizeof(uint8_t),
              /*input_pixel_stride=*/0, output.data(), step() * sizeof(void*),
              output_stride() * sizeof(uint8_t), &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(int16_t(output[x * output_stride() + c]), qmin())
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
          ASSERT_LE(int16_t(output[x * output_stride() + c]), qmax())
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
          ASSERT_EQ(int32_t(output_ref[x * channels() + c]),
                    int32_t(output[x * output_stride() + c]))
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f16_maxpool_ukernel_fn maxpool,
            xnn_init_f16_minmax_params_fn init_params) const {
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    xnnpack::Buffer<const xnn_float16*> indirect_input(
        (output_pixels() - 1) * step() + pooling_elements());
    xnnpack::Buffer<xnn_float16> input(
        ((output_pixels() - 1) * step() + pooling_elements()) * channels(),
        xnnpack::XnnExtraBytes);
    xnnpack::Buffer<xnn_float16> output(
        (output_pixels() - 1) * output_stride() + channels(),
        xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      for (size_t i = 0;
           i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
                   indirect_input.begin() + (output_pixels() - 1) * step() +
                       pooling_elements(),
                   rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float max_value = -std::numeric_limits<float>::infinity();
          for (size_t p = 0; p < pooling_elements(); p++) {
            max_value = std::max<float>(
                max_value, indirect_input[x * step() + p][c + input_offset()]);
          }
          output_ref[x * channels() + c] = max_value;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min =
          *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max =
          *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min =
          accumulated_min +
          accumulated_range *
              (static_cast<float>(qmin() -
                                  std::numeric_limits<int16_t>::min()) /
               static_cast<float>(std::numeric_limits<int16_t>::max() -
                                  std::numeric_limits<int16_t>::min()));
      if (qmin() == std::numeric_limits<int16_t>::min()) {
        output_min = -std::numeric_limits<float>::infinity();
      }
      float output_max =
          accumulated_max -
          accumulated_range *
              (static_cast<float>(std::numeric_limits<int16_t>::max() -
                                  qmax()) /
               static_cast<float>(std::numeric_limits<int16_t>::max() -
                                  std::numeric_limits<int16_t>::min()));
      if (qmax() == std::numeric_limits<int16_t>::max()) {
        output_max = +std::numeric_limits<float>::infinity();
      }
      output_min = xnn_float16(output_min);
      output_max = xnn_float16(output_max);

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params, static_cast<xnn_float16>(output_min),
                  static_cast<xnn_float16>(output_max));

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Call optimized micro-kernel.
      maxpool(output_pixels(), pooling_elements(), channels(),
              reinterpret_cast<const xnn_float16**>(indirect_input.data()),
              input_offset() * sizeof(xnn_float16), /*input_pixel_stride=*/0,
              output.data(), step() * sizeof(void*),
              output_stride() * sizeof(xnn_float16), &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(output[x * output_stride() + c], output_min)
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max)
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
          ASSERT_EQ(output[x * output_stride() + c],
                    output_ref[x * channels() + c])
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_maxpool_ukernel_fn maxpool,
            xnn_init_f32_minmax_params_fn init_params) const {
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    xnnpack::Buffer<const float*> indirect_input(
        (output_pixels() - 1) * step() + pooling_elements());
    xnnpack::Buffer<float> input(
        ((output_pixels() - 1) * step() + pooling_elements()) * channels(),
        xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> output(
        (output_pixels() - 1) * output_stride() + channels(),
        xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      for (size_t i = 0;
           i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
                   indirect_input.begin() + (output_pixels() - 1) * step() +
                       pooling_elements(),
                   rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float max_value = -std::numeric_limits<float>::infinity();
          for (size_t p = 0; p < pooling_elements(); p++) {
            max_value = std::max(
                max_value, indirect_input[x * step() + p][c + input_offset()]);
          }
          output_ref[x * channels() + c] = max_value;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min =
          *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max =
          *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min =
          accumulated_min +
          accumulated_range *
              (static_cast<float>(qmin() -
                                  std::numeric_limits<int16_t>::min()) /
               static_cast<float>(std::numeric_limits<int16_t>::max() -
                                  std::numeric_limits<int16_t>::min()));
      if (qmin() == std::numeric_limits<int16_t>::min()) {
        output_min = -std::numeric_limits<float>::infinity();
      }
      float output_max =
          accumulated_max -
          accumulated_range *
              (static_cast<float>(std::numeric_limits<int16_t>::max() -
                                  qmax()) /
               static_cast<float>(std::numeric_limits<int16_t>::max() -
                                  std::numeric_limits<int16_t>::min()));
      if (qmax() == std::numeric_limits<int16_t>::max()) {
        output_max = +std::numeric_limits<float>::infinity();
      }

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, output_min, output_max);

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Call optimized micro-kernel.
      maxpool(output_pixels(), pooling_elements(), channels(),
              indirect_input.data(), input_offset() * sizeof(float),
              /*input_pixel_stride=*/0, output.data(), step() * sizeof(void*),
              output_stride() * sizeof(float), &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(output[x * output_stride() + c], output_min)
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max)
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
          ASSERT_EQ(output_ref[x * channels() + c],
                    output[x * output_stride() + c])
              << "at pixel " << x << " / " << output_pixels() << ", channel "
              << c << " / " << channels()
              << ", pooling elements = " << pooling_elements()
              << ", step = " << step() << ", input offset = " << input_offset();
        }
      }
    }
  }

  struct Kernel {
    int16_t qmin;
    int16_t qmax;

    explicit Kernel(xnn_s8_maxpool_ukernel_fn maxpool,
                    xnn_init_s8_minmax_params_fn init_params)
        : qmin(-64), qmax(64) {
      dispatch = [maxpool, init_params](MaxPoolMicrokernelTester& tester) {
        tester.qmin(std::numeric_limits<int8_t>::min())
            .qmax(std::numeric_limits<int8_t>::max())
            .Test(maxpool, init_params);
      };
    }
    explicit Kernel(xnn_u8_maxpool_ukernel_fn maxpool,
                    xnn_init_u8_minmax_params_fn init_params)
        : qmin(0), qmax(192) {
      dispatch = [maxpool, init_params](MaxPoolMicrokernelTester& tester) {
        tester.qmin(std::numeric_limits<uint8_t>::min())
            .qmax(std::numeric_limits<uint8_t>::max())
            .Test(maxpool, init_params);
      };
    }
    explicit Kernel(xnn_f16_maxpool_ukernel_fn maxpool,
                    xnn_init_f16_minmax_params_fn init_params)
        : qmin(-16384), qmax(16384) {
      dispatch = [maxpool, init_params](MaxPoolMicrokernelTester& tester) {
        tester.Test(maxpool, init_params);
      };
    }
    explicit Kernel(xnn_f32_maxpool_ukernel_fn maxpool,
                    xnn_init_f32_minmax_params_fn init_params)
        : qmin(-16384), qmax(16384) {
      dispatch = [maxpool, init_params](MaxPoolMicrokernelTester& tester) {
        tester.Test(maxpool, init_params);
      };
    }
    std::function<void(MaxPoolMicrokernelTester&)> dispatch;
  };

  void Test(const Kernel& kernel) { kernel.dispatch(*this); }

 private:
  size_t output_pixels_{1};
  size_t pooling_elements_{1};
  size_t channels_{1};
  size_t input_offset_{0};
  size_t step_{1};
  size_t output_stride_{0};
  int16_t qmin_{std::numeric_limits<int16_t>::min()};
  int16_t qmax_{std::numeric_limits<int16_t>::max()};
  size_t iterations_{3};
};

#endif  // XNNPACK_TEST_MAXPOOL_MICROKERNEL_TESTER_H_
