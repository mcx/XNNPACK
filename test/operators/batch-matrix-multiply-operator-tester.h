// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_OPERATORS_BATCH_MATRIX_MULTIPLY_OPERATOR_TESTER_H_
#define XNNPACK_TEST_OPERATORS_BATCH_MATRIX_MULTIPLY_OPERATOR_TESTER_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packq.h"
#include "src/xnnpack/requantization.h"
#include "src/xnnpack/quantization.h"
#include "test/replicable_random_device.h"

constexpr int kIterations = 1;

class BatchMatMulOperatorTester {
 public:
  BatchMatMulOperatorTester& m(size_t m) {
    assert(m >= 1);
    this->m_ = m;
    return *this;
  }

  size_t m() const { return this->m_; }

  BatchMatMulOperatorTester& k(size_t k) {
    assert(k >= 1);
    this->k_ = k;
    return *this;
  }

  size_t k() const { return this->k_; }

  BatchMatMulOperatorTester& n(size_t n) {
    assert(n >= 1);
    this->n_ = n;
    return *this;
  }

  size_t n() const { return this->n_; }

  inline BatchMatMulOperatorTester& batch_dims_a(
      std::vector<size_t> batch_dims_a) {
    this->batch_dims_a_ = std::move(batch_dims_a);
    return *this;
  }

  const std::vector<size_t>& batch_dims_a() const {
    return this->batch_dims_a_;
  }

  inline BatchMatMulOperatorTester& batch_dims_b(
      std::vector<size_t> batch_dims_b) {
    this->batch_dims_b_ = std::move(batch_dims_b);
    return *this;
  }

  inline BatchMatMulOperatorTester& expected_status_reshape(
      enum xnn_status expected_status_reshape) {
    this->expected_status_reshape_ = expected_status_reshape;
    return *this;
  }

  const std::vector<size_t>& batch_dims_b() const {
    return this->batch_dims_b_;
  }

  BatchMatMulOperatorTester& transpose_b(bool transpose_b) {
    this->transpose_b_ = transpose_b;
    return *this;
  }

  bool transpose_b() const { return this->transpose_b_; }

  uint32_t flags() const {
    if (transpose_b()) {
      return XNN_FLAG_TRANSPOSE_B;
    } else {
      return 0;
    }
  }

  enum xnn_status expected_status_reshape() const {
    return expected_status_reshape_;
  }

  template <class T>
  static void ComputeRef(size_t m, size_t k, size_t n, bool transpose_b,
                         const T* input_a, const T* input_b,
                         float* output_ref) {
    std::fill(output_ref, output_ref + m * n, static_cast<T>(0));

    if (transpose_b) {
      // lhs is B*M*K, rhs is B*N*K.
      for (size_t mi = 0; mi < m; mi++) {
        for (size_t ni = 0; ni < n; ni++) {
          for (size_t ki = 0; ki < k; ki++) {
            output_ref[mi * n + ni] +=
                static_cast<float>(input_a[mi * k + ki]) *
                static_cast<float>(input_b[ni * k + ki]);
          }
        }
      }
    } else {
      // lhs is B*M*K, rhs is B*K*N.
      for (size_t mi = 0; mi < m; mi++) {
        for (size_t ni = 0; ni < n; ni++) {
          for (size_t ki = 0; ki < k; ki++) {
            output_ref[mi * n + ni] +=
                static_cast<float>(input_a[mi * k + ki]) *
                static_cast<float>(input_b[ki * n + ni]);
          }
        }
      }
    }
  }

  static void ComputeRefQS8(size_t m, size_t k, size_t n, bool transpose_b,
                            const int8_t* input_a, const int8_t* input_b,
                            int8_t input_zero_point, int32_t* accumulators) {
    std::fill(accumulators, accumulators + m * n, static_cast<int32_t>(0));

    if (transpose_b) {
      // lhs is B*M*K, rhs is B*N*K.
      for (size_t mi = 0; mi < m; mi++) {
        for (size_t ni = 0; ni < n; ni++) {
          for (size_t ki = 0; ki < k; ki++) {
            accumulators[mi * n + ni] +=
                (static_cast<int32_t>(input_a[mi * k + ki]) -
                 static_cast<int32_t>(input_zero_point)) *
                static_cast<int32_t>(input_b[ni * k + ki]);
          }
        }
      }
    } else {
      // lhs is B*M*K, rhs is B*K*N.
      for (size_t mi = 0; mi < m; mi++) {
        for (size_t ni = 0; ni < n; ni++) {
          for (size_t ki = 0; ki < k; ki++) {
            accumulators[mi * n + ni] +=
                (static_cast<int32_t>(input_a[mi * k + ki]) -
                 static_cast<int32_t>(input_zero_point)) *
                static_cast<int32_t>(input_b[ki * n + ni]);
          }
        }
      }
    }
  }

  void ComputeReferenceQS8(const std::vector<size_t>& batch_dims_output,
                           const int8_t* input_a, const int8_t* input_b,
                           int8_t input_zero_point, int32_t* accumulators,
                           void (*ref_fun)(size_t, size_t, size_t, bool,
                                           const int8_t*, const int8_t*,
                                           int8_t, int32_t*)) const {
    const int num_batch_dims = batch_dims_output.size();

    // Compute reference results.
    if (num_batch_dims == 0) {
      ref_fun(m(), k(), n(), transpose_b(), input_a, input_b, input_zero_point,
              accumulators);
    } else if (num_batch_dims == 1) {
      for (size_t b = 0; b < batch_dims_output[0]; b++) {
        const size_t ba = b % batch_dims_a()[0];
        const size_t bb = b % batch_dims_b()[0];
        ref_fun(m(), k(), n(), transpose_b(), &input_a[ba * m() * k()],
                &input_b[bb * n() * k()], input_zero_point,
                &accumulators[b * m() * n()]);
      }
    } else if (num_batch_dims == 2) {
      for (size_t b0 = 0; b0 < batch_dims_output[0]; b0++) {
        const size_t b0a = b0 % batch_dims_a()[0];
        const size_t b0b = b0 % batch_dims_b()[0];
        for (size_t b1 = 0; b1 < batch_dims_output[1]; b1++) {
          const size_t ba = b0a * batch_dims_a()[1] + (b1 % batch_dims_a()[1]);
          const size_t bb = b0b * batch_dims_b()[1] + (b1 % batch_dims_b()[1]);
          const size_t bc = b0 * batch_dims_output[1] + b1;
          ref_fun(m(), k(), n(), transpose_b(), &input_a[ba * m() * k()],
                  &input_b[bb * n() * k()], input_zero_point,
                  &accumulators[bc * m() * n()]);
        }
      }
    } else if (num_batch_dims == 3) {
      for (size_t b0 = 0; b0 < batch_dims_output[0]; b0++) {
        const size_t b0a = b0 % batch_dims_a()[0];
        const size_t b0b = b0 % batch_dims_b()[0];
        for (size_t b1 = 0; b1 < batch_dims_output[1]; b1++) {
          const size_t b1a = b0a * batch_dims_a()[1] + (b1 % batch_dims_a()[1]);
          const size_t b1b = b0b * batch_dims_b()[1] + (b1 % batch_dims_b()[1]);
          const size_t b1c = b0 * batch_dims_output[1] + b1;
          for (size_t b2 = 0; b2 < batch_dims_output[2]; b2++) {
            const size_t ba =
                b1a * batch_dims_a()[2] + (b2 % batch_dims_a()[2]);
            const size_t bb =
                b1b * batch_dims_b()[2] + (b2 % batch_dims_b()[2]);
            const size_t bc = b1c * batch_dims_output[2] + b2;
            ref_fun(m(), k(), n(), transpose_b(), &input_a[ba * m() * k()],
                    &input_b[bb * n() * k()], input_zero_point,
                    &accumulators[bc * m() * n()]);
          }
        }
      }
    } else if (num_batch_dims == 4) {
      for (size_t b0 = 0; b0 < batch_dims_output[0]; b0++) {
        const size_t b0a = b0 % batch_dims_a()[0];
        const size_t b0b = b0 % batch_dims_b()[0];
        for (size_t b1 = 0; b1 < batch_dims_output[1]; b1++) {
          const size_t b1a = b0a * batch_dims_a()[1] + (b1 % batch_dims_a()[1]);
          const size_t b1b = b0b * batch_dims_b()[1] + (b1 % batch_dims_b()[1]);
          const size_t b1c = b0 * batch_dims_output[1] + b1;
          for (size_t b2 = 0; b2 < batch_dims_output[2]; b2++) {
            const size_t b2a =
                b1a * batch_dims_a()[2] + (b2 % batch_dims_a()[2]);
            const size_t b2b =
                b1b * batch_dims_b()[2] + (b2 % batch_dims_b()[2]);
            const size_t b2c = b1c * batch_dims_output[2] + b2;
            for (size_t b3 = 0; b3 < batch_dims_output[3]; b3++) {
              const size_t ba =
                  b2a * batch_dims_a()[3] + (b3 % batch_dims_a()[3]);
              const size_t bb =
                  b2b * batch_dims_b()[3] + (b3 % batch_dims_b()[3]);
              const size_t bc = b2c * batch_dims_output[3] + b3;
              ref_fun(m(), k(), n(), transpose_b(), &input_a[ba * m() * k()],
                      &input_b[bb * n() * k()], input_zero_point,
                      &accumulators[bc * m() * n()]);
            }
          }
        }
      }
    } else {
      FAIL() << "Number of batch dims must be <= 4 (got " << num_batch_dims
             << ")";
    }
  }

  template <typename T>
  void ComputeReference(const std::vector<size_t>& batch_dims_output,
                        const T* input_a, const T* input_b, float* output_ref,
                        void (*ref_fun)(size_t, size_t, size_t, bool, const T*,
                                        const T*, float*)) const {
    const int num_batch_dims = batch_dims_output.size();

    // Compute reference results.
    if (num_batch_dims == 0) {
      ref_fun(m(), k(), n(), transpose_b(), input_a, input_b, output_ref);
    } else if (num_batch_dims == 1) {
      for (size_t b = 0; b < batch_dims_output[0]; b++) {
        const size_t ba = b % batch_dims_a()[0];
        const size_t bb = b % batch_dims_b()[0];
        ref_fun(m(), k(), n(), transpose_b(), &input_a[ba * m() * k()],
                &input_b[bb * n() * k()], &output_ref[b * m() * n()]);
      }
    } else if (num_batch_dims == 2) {
      for (size_t b0 = 0; b0 < batch_dims_output[0]; b0++) {
        const size_t b0a = b0 % batch_dims_a()[0];
        const size_t b0b = b0 % batch_dims_b()[0];
        for (size_t b1 = 0; b1 < batch_dims_output[1]; b1++) {
          const size_t ba = b0a * batch_dims_a()[1] + (b1 % batch_dims_a()[1]);
          const size_t bb = b0b * batch_dims_b()[1] + (b1 % batch_dims_b()[1]);
          const size_t bc = b0 * batch_dims_output[1] + b1;
          ref_fun(m(), k(), n(), transpose_b(), &input_a[ba * m() * k()],
                  &input_b[bb * n() * k()], &output_ref[bc * m() * n()]);
        }
      }
    } else if (num_batch_dims == 3) {
      for (size_t b0 = 0; b0 < batch_dims_output[0]; b0++) {
        const size_t b0a = b0 % batch_dims_a()[0];
        const size_t b0b = b0 % batch_dims_b()[0];
        for (size_t b1 = 0; b1 < batch_dims_output[1]; b1++) {
          const size_t b1a = b0a * batch_dims_a()[1] + (b1 % batch_dims_a()[1]);
          const size_t b1b = b0b * batch_dims_b()[1] + (b1 % batch_dims_b()[1]);
          const size_t b1c = b0 * batch_dims_output[1] + b1;
          for (size_t b2 = 0; b2 < batch_dims_output[2]; b2++) {
            const size_t ba =
                b1a * batch_dims_a()[2] + (b2 % batch_dims_a()[2]);
            const size_t bb =
                b1b * batch_dims_b()[2] + (b2 % batch_dims_b()[2]);
            const size_t bc = b1c * batch_dims_output[2] + b2;
            ref_fun(m(), k(), n(), transpose_b(), &input_a[ba * m() * k()],
                    &input_b[bb * n() * k()], &output_ref[bc * m() * n()]);
          }
        }
      }
    } else if (num_batch_dims == 4) {
      for (size_t b0 = 0; b0 < batch_dims_output[0]; b0++) {
        const size_t b0a = b0 % batch_dims_a()[0];
        const size_t b0b = b0 % batch_dims_b()[0];
        for (size_t b1 = 0; b1 < batch_dims_output[1]; b1++) {
          const size_t b1a = b0a * batch_dims_a()[1] + (b1 % batch_dims_a()[1]);
          const size_t b1b = b0b * batch_dims_b()[1] + (b1 % batch_dims_b()[1]);
          const size_t b1c = b0 * batch_dims_output[1] + b1;
          for (size_t b2 = 0; b2 < batch_dims_output[2]; b2++) {
            const size_t b2a =
                b1a * batch_dims_a()[2] + (b2 % batch_dims_a()[2]);
            const size_t b2b =
                b1b * batch_dims_b()[2] + (b2 % batch_dims_b()[2]);
            const size_t b2c = b1c * batch_dims_output[2] + b2;
            for (size_t b3 = 0; b3 < batch_dims_output[3]; b3++) {
              const size_t ba =
                  b2a * batch_dims_a()[3] + (b3 % batch_dims_a()[3]);
              const size_t bb =
                  b2b * batch_dims_b()[3] + (b3 % batch_dims_b()[3]);
              const size_t bc = b2c * batch_dims_output[3] + b3;
              ref_fun(m(), k(), n(), transpose_b(), &input_a[ba * m() * k()],
                      &input_b[bb * n() * k()], &output_ref[bc * m() * n()]);
            }
          }
        }
      }
    } else {
      FAIL() << "Number of batch dims must be <= 4 (got " << num_batch_dims
             << ")";
    }
  }

  void TestF16() const {
    ASSERT_EQ(batch_dims_a().size(), batch_dims_b().size());
    const size_t num_batch_dims = batch_dims_a().size();

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    size_t batch_size_a = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_a *= batch_dims_a()[k];
    }
    size_t batch_size_b = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_b *= batch_dims_b()[k];
    }
    std::vector<size_t> batch_dims_output(num_batch_dims);
    size_t batch_size_output = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_dims_output[k] = std::max(batch_dims_a()[k], batch_dims_b()[k]);
      batch_size_output *= batch_dims_output[k];
    }

    xnnpack::Buffer<xnn_float16> input_a(batch_size_a * m() * k(),
                                         xnnpack::XnnExtraBytes);
    xnnpack::Buffer<xnn_float16> input_b(batch_size_b * k() * n(),
                                         xnnpack::XnnExtraBytes);
    xnnpack::Buffer<xnn_float16> output(batch_size_output * m() * n());
    xnnpack::Buffer<float> output_ref(batch_size_output * m() * n());

    for (size_t iteration = 0; iteration < kIterations; iteration++) {
      std::generate(input_a.begin(), input_a.end(),
                    [&]() { return f32dist(rng); });
      std::generate(input_b.begin(), input_b.end(),
                    [&]() { return f32dist(rng); });

      // Compute reference results.
      ComputeReference(batch_dims_output, input_a.data(), input_b.data(),
                       output_ref.data(), ComputeRef<xnn_float16>);

      // Create, setup, run, and destroy Batch Matrix Multiply operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t batch_matrix_multiply_op = nullptr;

      const xnn_status status = xnn_create_batch_matrix_multiply_nc_f16(
          flags(), &batch_matrix_multiply_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, batch_matrix_multiply_op);

      // Smart pointer to automatically delete batch_matrix_multiply_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_batch_matrix_multiply_op(batch_matrix_multiply_op,
                                        xnn_delete_operator);

      size_t workspace_size = 0;
      ASSERT_EQ(expected_status_reshape(),
                xnn_reshape_batch_matrix_multiply_nc_f16(
                    batch_matrix_multiply_op, num_batch_dims,
                    batch_dims_a().data(), batch_dims_b().data(), m(), k(), n(),
                    &workspace_size,
                    /*threadpool=*/nullptr));
      if (expected_status_reshape() != xnn_status_success) {
        return;
      }
      ASSERT_NE(workspace_size, 0);
      xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> workspace(workspace_size);
      // TODO(b/372731180): This should probably be initialized by the operator.
      std::fill(workspace.begin(), workspace.end(), 0);

      ASSERT_EQ(xnn_status_success,
                xnn_setup_batch_matrix_multiply_nc_f16(
                    batch_matrix_multiply_op, workspace.data(), input_a.data(),
                    input_b.data(), output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(batch_matrix_multiply_op,
                                                     /*threadpool=*/nullptr));

      VerifyF16(output, output_ref);
    }
  }

  void TestQS8() const {
    ASSERT_EQ(batch_dims_a().size(), batch_dims_b().size());
    const size_t num_batch_dims = batch_dims_a().size();

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> s8dist(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    size_t batch_size_a = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_a *= batch_dims_a()[k];
    }
    size_t batch_size_b = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_b *= batch_dims_b()[k];
    }
    std::vector<size_t> batch_dims_output(num_batch_dims);
    size_t batch_size_output = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_dims_output[k] = std::max(batch_dims_a()[k], batch_dims_b()[k]);
      batch_size_output *= batch_dims_output[k];
    }

    xnnpack::Buffer<int8_t> input_a(batch_size_a * m() * k(),
                                    xnnpack::XnnExtraBytes);
    xnnpack::Buffer<int8_t> input_b(batch_size_b * k() * n(),
                                    xnnpack::XnnExtraBytes);
    xnnpack::Buffer<int8_t> output(batch_size_output * m() * n());
    xnnpack::Buffer<int32_t> accumulators(batch_size_output * m() * n());
    xnnpack::Buffer<double> output_ref(batch_size_output * m() * n());

    for (bool const_weights : {true, false}) {
      for (size_t iteration = 0; iteration < kIterations; iteration++) {
        std::generate_n(input_a.begin(), batch_size_a * m() * k(),
                        [&]() { return s8dist(rng); });
        std::generate_n(input_b.begin(), batch_size_b * k() * n(),
                        [&]() { return s8dist(rng); });

        const int8_t input_zero_point = s8dist(rng);
        const int8_t output_min = std::numeric_limits<int8_t>::min();
        const int8_t output_max = std::numeric_limits<int8_t>::max();

        // Compute reference results without requantization.
        ComputeReferenceQS8(batch_dims_output, input_a.data(), input_b.data(),
                            input_zero_point, accumulators.data(),
                            ComputeRefQS8);

        // Compute renormalization parameters.
        const int32_t accumulated_min =
            *std::min_element(accumulators.cbegin(), accumulators.cend());
        const int32_t accumulated_max =
            *std::max_element(accumulators.cbegin(), accumulators.cend());

        const xnn_qd8_quantization_params output_quantization =
            xnn_qd8_asymmetric_quantization_params(
                accumulated_min, accumulated_max);
        const float scale = 1.0f / output_quantization.inv_scale;

        // Renormalize reference results.
        std::transform(
          accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
          [scale, output_quantization](int32_t x) -> double {
            return static_cast<double>(
                xnn_qs8_quantize(
                    static_cast<float>(x), scale,
                    static_cast<int32_t>(output_quantization.zero_point)));
          });

        // Create, setup, run, and destroy Batch Matrix Multiply operator.
        ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
        xnn_operator_t batch_matrix_multiply_op = nullptr;

        xnn_status status = xnn_status_success;
        if (const_weights) {
          status = xnn_create_batch_matrix_multiply_nc_qs8_const_weights(
              batch_size_b, k(), n(), input_b.data(), input_zero_point,
              output_quantization.zero_point, output_min, output_max,
              output_quantization.inv_scale, flags(),
              &batch_matrix_multiply_op);
        } else {
          status = xnn_create_batch_matrix_multiply_nc_qs8(
              input_zero_point, output_quantization.zero_point,
              output_min, output_max, &output_quantization.inv_scale, flags(),
              &batch_matrix_multiply_op);
        }
        if (status == xnn_status_unsupported_hardware) {
          GTEST_SKIP();
        }
        ASSERT_EQ(xnn_status_success, status);
        ASSERT_NE(nullptr, batch_matrix_multiply_op);

        // Smart pointer to automatically delete batch_matrix_multiply_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_batch_matrix_multiply_op(batch_matrix_multiply_op,
                                          xnn_delete_operator);

        size_t workspace_size = 0;
        if (const_weights) {
          ASSERT_EQ(expected_status_reshape(),
                    xnn_reshape_batch_matrix_multiply_nc_qs8_const_weights(
                        batch_matrix_multiply_op, num_batch_dims,
                        batch_dims_a().data(), batch_dims_b().data(), m(), k(),
                        n(), /*threadpool=*/nullptr));
        } else {
          ASSERT_EQ(expected_status_reshape(),
                    xnn_reshape_batch_matrix_multiply_nc_qs8(
                        batch_matrix_multiply_op, num_batch_dims,
                        batch_dims_a().data(), batch_dims_b().data(), m(), k(),
                        n(), &workspace_size,
                        /*threadpool=*/nullptr));
        }
        if (expected_status_reshape() != xnn_status_success) {
          return;
        }
        if (const_weights) {
          ASSERT_EQ(workspace_size, 0);
        } else {
          ASSERT_NE(workspace_size, 0);
        }
        xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> workspace(
            workspace_size);

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_batch_matrix_multiply_nc_qs8(
                      batch_matrix_multiply_op, workspace.data(),
                      input_a.data(), input_b.data(), output.data()));

        ASSERT_EQ(xnn_status_success, xnn_run_operator(batch_matrix_multiply_op,
                                                      /*threadpool=*/nullptr));

        VerifyQS8(output, output_ref);
      }
    }
  }

  void TestBF16F32() const {
    ASSERT_EQ(batch_dims_a().size(), batch_dims_b().size());
    const size_t num_batch_dims = batch_dims_a().size();

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.f);

    size_t batch_size_a = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_a *= batch_dims_a()[k];
    }
    size_t batch_size_b = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_b *= batch_dims_b()[k];
    }
    std::vector<size_t> batch_dims_output(num_batch_dims);
    size_t batch_size_output = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_dims_output[k] = std::max(batch_dims_a()[k], batch_dims_b()[k]);
      batch_size_output *= batch_dims_output[k];
    }

    xnnpack::Buffer<xnn_bfloat16> input_a(batch_size_a * m() * k(),
                                          xnnpack::XnnExtraBytes);
    xnnpack::Buffer<xnn_bfloat16> input_b(batch_size_b * k() * n(),
                                          xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> output(batch_size_output * m() * n());
    xnnpack::Buffer<float> output_ref(batch_size_output * m() * n());

    for (size_t iteration = 0; iteration < kIterations; iteration++) {
      std::generate(input_a.begin(), input_a.end(),
                    [&]() { return f32dist(rng); });
      std::generate(input_b.begin(), input_b.end(),
                    [&]() { return f32dist(rng); });

      // Compute reference results.
      ComputeReference(batch_dims_output, input_a.data(), input_b.data(),
                       output_ref.data(), ComputeRef<xnn_bfloat16>);

      // Create, setup, run, and destroy Batch Matrix Multiply operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t batch_matrix_multiply_op = nullptr;

      const xnn_status status = xnn_create_batch_matrix_multiply_nc_bf16_f32(
          flags(), &batch_matrix_multiply_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, batch_matrix_multiply_op);

      // Smart pointer to automatically delete batch_matrix_multiply_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_batch_matrix_multiply_op(batch_matrix_multiply_op,
                                        xnn_delete_operator);

      size_t workspace_size = 0;
      ASSERT_EQ(expected_status_reshape(),
                xnn_reshape_batch_matrix_multiply_nc_bf16_f32(
                    batch_matrix_multiply_op, num_batch_dims,
                    batch_dims_a().data(), batch_dims_b().data(), m(), k(), n(),
                    &workspace_size,
                    /*threadpool=*/nullptr));
      if (expected_status_reshape() != xnn_status_success) {
        return;
      }
      ASSERT_NE(workspace_size, 0);
      xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> workspace(workspace_size);
      // TODO(b/372731180): This should probably be initialized by the operator.
      std::fill(workspace.begin(), workspace.end(), 0);

      ASSERT_EQ(xnn_status_success,
                xnn_setup_batch_matrix_multiply_nc_bf16_f32(
                    batch_matrix_multiply_op, workspace.data(), input_a.data(),
                    input_b.data(), output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(batch_matrix_multiply_op,
                                                     /*threadpool=*/nullptr));

      VerifyF32(output, output_ref);
    }
  }

  void TestF32() const {
    ASSERT_EQ(batch_dims_a().size(), batch_dims_b().size());
    const size_t num_batch_dims = batch_dims_a().size();

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    size_t batch_size_a = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_a *= batch_dims_a()[k];
    }
    size_t batch_size_b = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_b *= batch_dims_b()[k];
    }
    std::vector<size_t> batch_dims_output(num_batch_dims);
    size_t batch_size_output = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_dims_output[k] = std::max(batch_dims_a()[k], batch_dims_b()[k]);
      batch_size_output *= batch_dims_output[k];
    }

    xnnpack::Buffer<float> input_a(batch_size_a * m() * k(),
                                   xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> input_b(batch_size_b * k() * n(),
                                   xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> output(batch_size_output * m() * n());
    xnnpack::Buffer<float> output_ref(batch_size_output * m() * n());

    for (bool const_weights : {true, false}) {
      for (size_t iteration = 0; iteration < kIterations; iteration++) {
        std::generate(input_a.begin(), input_a.end(),
                      [&]() { return f32dist(rng); });
        std::generate(input_b.begin(), input_b.end(),
                      [&]() { return f32dist(rng); });

        // Compute reference results.
        ComputeReference(batch_dims_output, input_a.data(), input_b.data(),
                         output_ref.data(), ComputeRef<float>);

        // Create, setup, run, and destroy Batch Matrix Multiply operator.
        ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
        xnn_operator_t batch_matrix_multiply_op = nullptr;

        xnn_status status;
        if (const_weights) {
          status = xnn_create_batch_matrix_multiply_nc_f32_const_weights(
              batch_size_b, k(), n(), input_b.data(), flags(),
              &batch_matrix_multiply_op);
        } else {
          status = xnn_create_batch_matrix_multiply_nc_f32(
              flags(), &batch_matrix_multiply_op);
        }
        if (status == xnn_status_unsupported_hardware) {
          GTEST_SKIP();
        }
        ASSERT_EQ(xnn_status_success, status);
        ASSERT_NE(nullptr, batch_matrix_multiply_op);

        // Smart pointer to automatically delete batch_matrix_multiply_op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
            auto_batch_matrix_multiply_op(batch_matrix_multiply_op,
                                          xnn_delete_operator);

        size_t workspace_size = 0;
        ASSERT_EQ(expected_status_reshape(),
                  xnn_reshape_batch_matrix_multiply_nc_f32(
                      batch_matrix_multiply_op, num_batch_dims,
                      batch_dims_a().data(), batch_dims_b().data(), m(), k(),
                      n(), &workspace_size,
                      /*threadpool=*/nullptr));
        if (expected_status_reshape() != xnn_status_success) {
          return;
        }
        if (const_weights) {
          ASSERT_EQ(workspace_size, 0);
        } else {
          ASSERT_NE(workspace_size, 0);
        }
        xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> workspace(
            workspace_size);
        // TODO(b/372731180): This should probably be initialized by the
        // operator.
        std::fill(workspace.begin(), workspace.end(), 0);

        ASSERT_EQ(xnn_status_success,
                  xnn_setup_batch_matrix_multiply_nc_f32(
                      batch_matrix_multiply_op, workspace.data(),
                      input_a.data(), input_b.data(), output.data()));

        ASSERT_EQ(xnn_status_success, xnn_run_operator(batch_matrix_multiply_op,
                                                       /*threadpool=*/nullptr));

        VerifyF32(output, output_ref);
      }
    }
  }

  void ComputeQC8W(xnnpack::Buffer<float>& input_b, size_t batch_size_b,
                   xnnpack::Buffer<int8_t>& input_b_qc8,
                   xnnpack::Buffer<float>& channelwise_scale_b) const {
    if (transpose_b_) {
      for (size_t b = 0; b < batch_size_b; b++) {
        for (size_t c = 0; c < n(); c++) {
          const size_t offset = b * n() * k() + c * k();
          float max_abs = 0.0f;
          for (size_t i = 0; i < k(); i++) {
            max_abs = std::max(max_abs, std::abs(input_b[offset + i]));
          }
          if (max_abs == 0.0f) {
            max_abs = 1.0f;
          }
          const float scale = max_abs / std::numeric_limits<int8_t>::max();
          const float inv_scale = 1.0f / scale;
          for (size_t i = 0; i < k(); i++) {
            input_b_qc8[offset + i] = static_cast<int8_t>(
                std::round(input_b[offset + i] * inv_scale));
          }
          channelwise_scale_b[b * n() + c] = scale;
        }
      }
    } else {
      for (size_t b = 0; b < batch_size_b; b++) {
        const size_t bnk = b * n() * k();
        for (size_t c = 0; c < n(); c++) {
          float max_abs = 0.0f;
          for (size_t i = 0; i < k(); i++) {
            max_abs = std::max(max_abs, std::abs(input_b[bnk + i * n() + c]));
          }
          if (max_abs == 0.0f) {
            max_abs = 1.0f;
          }
          const float scale = max_abs / std::numeric_limits<int8_t>::max();
          const float inv_scale = 1.0f / scale;
          for (size_t i = 0; i < k(); i++) {
            input_b_qc8[bnk + i * n() + c] = static_cast<int8_t>(
                std::round(input_b[bnk + i * n() + c] * inv_scale));
          }
          channelwise_scale_b[b * n() + c] = scale;
        }
      }
    }
  }

  void TestQD8F32QC8W() const {
    ASSERT_EQ(batch_dims_a().size(), batch_dims_b().size());
    const size_t num_batch_dims = batch_dims_a().size();

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(range_f32_.first,
                                                  range_f32_.second);

    size_t batch_size_a = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_a *= batch_dims_a()[k];
    }
    size_t batch_size_b = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_b *= batch_dims_b()[k];
    }
    std::vector<size_t> batch_dims_output(num_batch_dims);
    size_t batch_size_output = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_dims_output[k] = std::max(batch_dims_a()[k], batch_dims_b()[k]);
      batch_size_output *= batch_dims_output[k];
    }

    xnnpack::Buffer<float> input_a(batch_size_a * m() * k(),
                                   xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> input_b(batch_size_b * k() * n(),
                                   xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> output(batch_size_output * m() * n());
    xnnpack::Buffer<float> output_ref(batch_size_output * m() * n());

    for (size_t iteration = 0; iteration < kIterations; iteration++) {
      std::generate(input_a.begin(), input_a.end(),
                    [&]() { return f32dist(rng); });
      std::generate(input_b.begin(), input_b.end(),
                    [&]() { return f32dist(rng); });

      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

      // Create the dynamically quantized input data with the corresponding
      // `quantization_params`.
      xnnpack::Buffer<xnn_quantization_params> quantization_params(
          batch_size_a * m() + XNN_EXTRA_QUANTIZATION_PARAMS);
      xnnpack::Buffer<int8_t> input_a_qd8(batch_size_a * m() * k(),
                                          xnnpack::XnnExtraBytes);
      xnn_operator_t convert_op = nullptr;
      xnn_status status = xnn_create_convert_nc_f32_qd8(
          /*flags=*/0, &convert_op);
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_convert_op(convert_op, xnn_delete_operator);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convert_op);
      ASSERT_EQ(xnn_status_success, xnn_reshape_convert_nc_f32_qd8(
                                        convert_op, batch_size_a * m(), k(),
                                        k(), k(), /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success,
                xnn_setup_convert_nc_f32_qd8(convert_op, input_a.data(),
                                             input_a_qd8.data(),
                                             quantization_params.data()));
      ASSERT_EQ(xnn_status_success,
                xnn_run_operator(convert_op, /*threadpool=*/nullptr));

      // Compute the channelwise quantized input_b.
      xnnpack::Buffer<int8_t> input_b_qc8(batch_size_b * k() * n(),
                                          xnnpack::XnnExtraBytes);
      xnnpack::Buffer<float> channelwise_scale_b(batch_size_b * n(),
                                                 xnnpack::XnnExtraBytes);
      ComputeQC8W(input_b, batch_size_b, input_b_qc8, channelwise_scale_b);
      // Compute reference results.
      ComputeReference(batch_dims_output, input_a.data(), input_b.data(),
                       output_ref.data(), ComputeRef<float>);

      // Create, setup, run, and destroy Batch Matrix Multiply operator.
      xnn_operator_t batch_matrix_multiply_op = nullptr;

      status = xnn_create_batch_matrix_multiply_nc_qd8_f32_qc8w(
          batch_size_b, k(), n(), input_b_qc8.data(),
          channelwise_scale_b.data(), flags(), &batch_matrix_multiply_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, batch_matrix_multiply_op);

      // Smart pointer to automatically delete batch_matrix_multiply_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_batch_matrix_multiply_op(batch_matrix_multiply_op,
                                        xnn_delete_operator);

      size_t workspace_size = 0;
      ASSERT_EQ(expected_status_reshape(),
                xnn_reshape_batch_matrix_multiply_nc_qd8_f32_qc8w(
                    batch_matrix_multiply_op, num_batch_dims,
                    batch_dims_a().data(), batch_dims_b().data(), m(), k(), n(),
                    &workspace_size, /*threadpool=*/nullptr));
      if (expected_status_reshape() != xnn_status_success) {
        return;
      }
      xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> workspace(
          workspace_size);

      ASSERT_EQ(
          xnn_status_success,
          xnn_setup_batch_matrix_multiply_nc_qd8_f32_qc8w(
              batch_matrix_multiply_op, workspace.data(), input_a_qd8.data(),
              /*input_b=*/nullptr, quantization_params.data(), output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(batch_matrix_multiply_op,
                                                     /*threadpool=*/nullptr));

      VerifyQD8F32QC8W(output, output_ref);
    }
  }

  void TestQP8F32QC8W() const {
    const struct xnn_gemm_config* gemm_config =
        xnn_init_qp8_f32_qc8w_gemm_config();
    if (gemm_config == nullptr) {
      GTEST_SKIP();
    }

    ASSERT_EQ(batch_dims_a().size(), batch_dims_b().size());
    const size_t num_batch_dims = batch_dims_a().size();

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(range_f32_.first,
                                                  range_f32_.second);

    size_t batch_size_a = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_a *= batch_dims_a()[k];
    }
    size_t batch_size_b = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_size_b *= batch_dims_b()[k];
    }
    std::vector<size_t> batch_dims_output(num_batch_dims);
    size_t batch_size_output = 1;
    for (int k = 0; k < num_batch_dims; k++) {
      batch_dims_output[k] = std::max(batch_dims_a()[k], batch_dims_b()[k]);
      batch_size_output *= batch_dims_output[k];
    }

    xnnpack::Buffer<float> input_a(batch_size_a * m() * k(),
                                   xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> input_b(batch_size_b * k() * n(),
                                   xnnpack::XnnExtraBytes);
    xnnpack::Buffer<float> output(batch_size_output * m() * n());
    xnnpack::Buffer<float> output_ref(batch_size_output * m() * n());

    for (size_t iteration = 0; iteration < kIterations; iteration++) {
      std::generate(input_a.begin(), input_a.end(),
                    [&]() { return f32dist(rng); });
      std::generate(input_b.begin(), input_b.end(),
                    [&]() { return f32dist(rng); });

      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

      // Create the dynamically quantized input data.
      xnnpack::Buffer<int8_t> input_a_qp8(
          batch_size_a *
              xnn_x8_packq_f32qp8_gemm_packed_size(gemm_config, m(), k()),
          xnnpack::XnnExtraBytes);
      xnn_operator_t convert_op = nullptr;
      xnn_status status = xnn_create_convert_nc_f32_qp8(
          /*flags=*/0, gemm_config, &convert_op);
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_convert_op(convert_op, xnn_delete_operator);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convert_op);
      ASSERT_EQ(xnn_status_success, xnn_reshape_convert_nc_f32_qp8(
                                        convert_op, batch_size_a, m(), k(), k(),
                                        /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success,
                xnn_setup_convert_nc_f32_qp8(convert_op, input_a.data(),
                                             input_a_qp8.data()));
      ASSERT_EQ(xnn_status_success,
                xnn_run_operator(convert_op, /*threadpool=*/nullptr));

      // Compute the channelwise quantized input_b.
      xnnpack::Buffer<int8_t> input_b_qc8(batch_size_b * k() * n(),
                                          xnnpack::XnnExtraBytes);
      xnnpack::Buffer<float> channelwise_scale_b(batch_size_b * n(),
                                                 xnnpack::XnnExtraBytes);
      ComputeQC8W(input_b, batch_size_b, input_b_qc8, channelwise_scale_b);

      // Compute reference results.
      ComputeReference(batch_dims_output, input_a.data(), input_b.data(),
                       output_ref.data(), ComputeRef<float>);

      // Create, setup, run, and destroy Batch Matrix Multiply operator.
      xnn_operator_t batch_matrix_multiply_op = nullptr;

      status = xnn_create_batch_matrix_multiply_nc_qp8_f32_qc8w(
          batch_size_b, k(), n(), input_b_qc8.data(),
          channelwise_scale_b.data(), flags(), &batch_matrix_multiply_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, batch_matrix_multiply_op);

      // Smart pointer to automatically delete batch_matrix_multiply_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_batch_matrix_multiply_op(batch_matrix_multiply_op,
                                        xnn_delete_operator);

      size_t workspace_size = 0;
      ASSERT_EQ(expected_status_reshape(),
                xnn_reshape_batch_matrix_multiply_nc_qp8_f32_qc8w(
                    batch_matrix_multiply_op, num_batch_dims,
                    batch_dims_a().data(), batch_dims_b().data(), m(), k(), n(),
                    &workspace_size, /*threadpool=*/nullptr));
      if (expected_status_reshape() != xnn_status_success) {
        return;
      }
      xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> workspace(
          workspace_size);

      ASSERT_EQ(xnn_status_success,
                xnn_setup_batch_matrix_multiply_nc_qp8_f32_qc8w(
                    batch_matrix_multiply_op, workspace.data(),
                    input_a_qp8.data(), /*input_b=*/nullptr, output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(batch_matrix_multiply_op,
                                                     /*threadpool=*/nullptr));

      VerifyQD8F32QC8W(output, output_ref);
    }
  }

  void VerifyF16(const xnnpack::Buffer<xnn_float16>& output,
                 const xnnpack::Buffer<float>& output_ref) const {
    const size_t batch_size_output = output.size() / (m() * n());
    for (size_t bi = 0; bi < batch_size_output; bi++) {
      for (size_t mi = 0; mi < m(); mi++) {
        for (size_t ni = 0; ni < n(); ni++) {
          ASSERT_NEAR(
              output_ref[bi * m() * n() + mi * n() + ni],
              output[bi * m() * n() + mi * n() + ni],
              1.0e-2f * std::abs(output_ref[bi * m() * n() + mi * n() + ni]))
              << "batch = " << bi << " / " << batch_size_output
              << ", m = " << mi << " / " << m() << ", n = " << ni << " / "
              << n();
        }
      }
    }
  }

  void VerifyQS8(const xnnpack::Buffer<int8_t>& output,
                 const xnnpack::Buffer<double>& output_ref) const {
    const size_t batch_size_output = output.size() / (m() * n());
    for (size_t bi = 0; bi < batch_size_output; bi++) {
      for (size_t mi = 0; mi < m(); mi++) {
        for (size_t ni = 0; ni < n(); ni++) {
          ASSERT_NEAR(
              output_ref[bi * m() * n() + mi * n() + ni],
              output[bi * m() * n() + mi * n() + ni], 1)
              << "batch = " << bi << " / " << batch_size_output
              << ", m = " << mi << " / " << m() << ", n = " << ni << " / "
              << n();
        }
      }
    }
  }

  void VerifyF32(const xnnpack::Buffer<float>& output,
                 const xnnpack::Buffer<float>& output_ref) const {
    // Verify results.
    const size_t batch_size_output = output.size() / (m() * n());
    for (size_t bi = 0; bi < batch_size_output; bi++) {
      for (size_t mi = 0; mi < m(); mi++) {
        for (size_t ni = 0; ni < n(); ni++) {
          ASSERT_NEAR(
              output_ref[bi * m() * n() + mi * n() + ni],
              output[bi * m() * n() + mi * n() + ni],
              1.0e-4f * std::abs(output_ref[bi * m() * n() + mi * n() + ni]))
              << "batch = " << bi << " / " << batch_size_output
              << ", m = " << mi << " / " << m() << ", n = " << ni << " / "
              << n();
        }
      }
    }
  }

  void VerifyQD8F32QC8W(const xnnpack::Buffer<float>& output,
                        const xnnpack::Buffer<float>& output_ref) const {
    // Compute the expected error bound, which is the error bound of the
    // quantized dot product between the rows of $A$ and the columns of $B$.
    //
    // Assuming that every entry in the output $C$ is computed as
    //
    //   $C_{ij} = \sum_k A_{ik} B_{kj}$
    //
    // Then the output subject to quantization errors $\tilde{C}$ is
    //
    //   $\tilde{C}_{ij} = \sum_k \tilde{A}_{ik} \tilde{B}_{kj}$
    //   $\tilde{C}_{ij} = \sum_k (A_{ik} + \vardelta_a))(B_{kj} + \vardelta_b)$
    //   $\tilde{C}_{ij} = \sum_k (A_{ik}B_{kj} + A_{ik}\vardelta_b +
    //                             B_{kj}\vardelta_a + \vardelta_a\vardelta_b)$
    //   $\tilde{C}_{ij} = C_{ij} + \sum_k (A_{ik}\vardelta_b +
    //                           `          B_{kj}\vardelta_a +
    //                                      \vardelta_a\vardelta_b)$
    //
    // which can be bounded by
    //
    //   $|\tilde{C}_{ij} - C_{ij}| \leq k (\hat{A}\vardelta_b +
    //                                      \hat{B}\vardelta_a +
    //                                      \vardelta_a\vardelta_b)$
    //
    // where $\hat{A}$ and $\hat{B}$ are the maximum absolute values of $A$ and
    // $B$, respectively.
    //
    // Note that we assume that the row/column quantization ranges are all
    // equal. For a more nuanced estimate, we could/should compute and apply the
    // row/column maxima separately.
    const float max_abs_a =
        std::max(std::abs(range_f32_.first), std::abs(range_f32_.second));
    const float max_abs_b =
        std::max(std::abs(range_f32_.first), std::abs(range_f32_.second));
    const float delta_a = 0.5f * max_abs_a / std::numeric_limits<int8_t>::max();
    const float delta_b = 0.5f * max_abs_b / std::numeric_limits<int8_t>::max();
    const float max_abs_err =
        k() * (delta_a * max_abs_b + delta_b * max_abs_a + delta_a * delta_b);

    // Verify results.
    const size_t batch_size_output = output.size() / (m() * n());
    for (size_t bi = 0; bi < batch_size_output; bi++) {
      for (size_t mi = 0; mi < m(); mi++) {
        for (size_t ni = 0; ni < n(); ni++) {
          ASSERT_NEAR(output_ref[bi * m() * n() + mi * n() + ni],
                      output[bi * m() * n() + mi * n() + ni], max_abs_err)
              << "batch = " << bi << " / " << batch_size_output
              << ", m = " << mi << " / " << m() << ", n = " << ni << " / "
              << n();
        }
      }
    }
  }

 private:
  // TODO(zhin): support flags for transpose lhs.
  size_t m_{1};
  size_t k_{1};
  size_t n_{1};
  std::vector<size_t> batch_dims_a_ = {1};
  std::vector<size_t> batch_dims_b_ = {1};
  std::pair<float, float> range_f32_ = {-1.0f, 1.0f};
  bool transpose_b_{false};
  enum xnn_status expected_status_reshape_ = xnn_status_success;
};

#endif  // XNNPACK_TEST_OPERATORS_BATCH_MATRIX_MULTIPLY_OPERATOR_TESTER_H_
