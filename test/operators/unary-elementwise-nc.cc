// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <sys/types.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "test/replicable_random_device.h"
#include "test/unary-ops.h"

enum class RunMode {
  kCreateReshapeRun,
  kEager,
};

struct UnaryOpTestParams {
  UnaryOpTestParams(std::string test_name_, size_t batch_size_,
                    size_t channels_)
      : test_name(test_name_), batch_size(batch_size_), channels(channels_) {}

  static UnaryOpTestParams UnitBatch() {
    return UnaryOpTestParams("unit_batch", 1, 100);
  }
  static UnaryOpTestParams SmallBatch() {
    return UnaryOpTestParams("small_batch", 3, 100);
  }
  static UnaryOpTestParams StridedBatch() {
    return UnaryOpTestParams("strided_batch", 3, 100)
        .InputStride(129)
        .OutputStride(117);
  }
  UnaryOpTestParams& BatchSize(size_t batch_size) {
    this->batch_size = batch_size;
    return *this;
  }
  UnaryOpTestParams& Channels(size_t channels) {
    this->channels = channels;
    return *this;
  }
  UnaryOpTestParams& Iterations(size_t iterations) {
    this->iterations = iterations;
    return *this;
  }
  UnaryOpTestParams& InputStride(size_t input_stride) {
    this->input_stride = input_stride;
    return *this;
  }
  UnaryOpTestParams& OutputStride(size_t output_stride) {
    this->output_stride = output_stride;
    return *this;
  }
  UnaryOpTestParams& InputQuantization(
      const xnn_quantization_params& input_quantization) {
    this->input_quantization = input_quantization;
    return *this;
  }
  UnaryOpTestParams& OutputQuantization(
      const xnn_quantization_params& output_quantization) {
    this->output_quantization = output_quantization;
    return *this;
  }

  std::string test_name;
  size_t batch_size;
  size_t iterations = 3;
  size_t channels = 100;
  size_t input_stride = 0;
  size_t output_stride = 0;
  xnn_quantization_params input_quantization = {0, 1.0f};
  xnn_quantization_params output_quantization = {0, 1.0f};
};

struct Param {
  using UnaryT = std::tuple<xnn_unary_operator, xnn_datatype, RunMode>;
  using ConvertT =
      std::tuple<xnn_unary_operator, xnn_datatype, xnn_datatype, RunMode>;

  explicit Param(UnaryT p)
      : unary_operator(std::get<0>(p)),
        input_datatype(std::get<1>(p)),
        output_datatype(std::get<1>(p)),
        run_mode(std::get<2>(p)) {}
  explicit Param(ConvertT p)
      : unary_operator(std::get<0>(p)),
        input_datatype(std::get<1>(p)),
        output_datatype(std::get<2>(p)),
        run_mode(std::get<3>(p)) {}

  std::string Name() const {
    std::stringstream sstr;
    sstr << xnn_unary_operator_to_string(unary_operator) << "_"
         << xnn_datatype_to_string(input_datatype);
    if (input_datatype != output_datatype) {
      sstr << "_" << xnn_datatype_to_string(output_datatype);
    }
    if (run_mode == RunMode::kCreateReshapeRun) {
      sstr << "_CreateReshapeRun";
    } else if (run_mode == RunMode::kEager) {
      sstr << "_Eager";
    }
    std::string s = sstr.str();
    // Test names must be alphanumeric with no spaces
    std::replace(s.begin(), s.end(), ' ', '_');
    std::replace(s.begin(), s.end(), '(', '_');
    std::replace(s.begin(), s.end(), ')', '_');
    return s;
  }

  xnn_unary_operator unary_operator;
  xnn_datatype input_datatype;
  xnn_datatype output_datatype;
  RunMode run_mode;
};

class UnaryNCTest : public testing::TestWithParam<Param> {
 public:
  xnnpack::ReplicableRandomDevice rng_;

  template <typename In, typename Out>
  void RunUnaryTest(const UnaryOpTestParams& test_params, const Param& param) {
    if (::testing::Test::IsSkipped()) {
      return;
    }
    ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

    const xnn_unary_operator unary_op = param.unary_operator;
    const xnn_datatype input_datatype = param.input_datatype;
    const xnn_datatype output_datatype = param.output_datatype;

    const UnaryOpInfo* op_info = GetUnaryOpInfo(unary_op);
    const xnn_unary_params op_params = op_info->DefaultParams();
    xnn_quantization_params input_quantization =
        xnnpack::is_quantized<In>::value
            ? test_params.input_quantization
            : op_info->InputQuantizationParams(input_datatype);
    xnn_quantization_params output_quantization =
        xnnpack::is_quantized<Out>::value
            ? test_params.output_quantization
            : op_info->OutputQuantizationParams(output_datatype);
    op_info->InputQuantizationParams(input_datatype);

    Interval domain = op_info->Domain(input_datatype);

    const size_t batch_size = test_params.batch_size;
    const size_t iterations = test_params.iterations;
    const size_t channels = test_params.channels;
    const size_t input_stride =
        test_params.input_stride == 0 ? channels : test_params.input_stride;
    const size_t output_stride =
        test_params.output_stride == 0 ? channels : test_params.output_stride;
    xnnpack::Buffer<In> input((batch_size - 1) * input_stride + channels,
                              xnnpack::XnnExtraBytes);
    xnnpack::Buffer<Out> output((batch_size - 1) * output_stride + channels);
    xnnpack::Buffer<Out> output_ref(batch_size * channels);
    xnnpack::DatatypeGenerator<In> input_generator(domain.min, domain.max,
                                                   input_quantization);
    for (size_t iteration = 0; iteration < iterations; iteration++) {
      for (size_t i = 0; i < batch_size; i++) {
        std::generate_n(input.data() + i * input_stride, channels,
                        [&]() { return input_generator(rng_); });
      }

      if (param.run_mode == RunMode::kEager) {
        xnn_status status = xnn_run_unary_elementwise_nc(
            unary_op, input_datatype, output_datatype, &op_params,
            &input_quantization, &output_quantization,
            /*flags=*/0, batch_size, channels, input_stride, output_stride,
            /*threadpool=*/nullptr, input.data(), output.data());
        if (status == xnn_status_unsupported_parameter) {
          GTEST_SKIP();
          return;
        }
        ASSERT_EQ(xnn_status_success, status);
      } else if (param.run_mode == RunMode::kCreateReshapeRun) {
        xnn_operator_t op = nullptr;
        xnn_status status = xnn_create_unary_elementwise_nc(
            unary_op, input_datatype, output_datatype, &op_params,
            /*lut=*/nullptr, &input_quantization, &output_quantization,
            /*flags=*/0, &op);
        if (status == xnn_status_unsupported_parameter) {
          GTEST_SKIP();
          return;
        }
        ASSERT_EQ(xnn_status_success, status);
        ASSERT_NE(nullptr, op);

        // Smart pointer to automatically delete op.
        std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
            op, xnn_delete_operator);

        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_unary_elementwise_nc(op, batch_size, channels,
                                                   input_stride, output_stride,
                                                   /*threadpool=*/nullptr));
        ASSERT_EQ(xnn_status_success, xnn_setup_unary_elementwise_nc(
                                          op, input.data(), output.data()));
        ASSERT_EQ(xnn_status_success,
                  xnn_run_operator(op, /*threadpool=*/nullptr));
      } else {
        XNN_UNREACHABLE;
      }

      // Verify results.
      for (size_t i = 0; i < batch_size; i++) {
        // Compute reference results.
        UnaryReferenceImpl(input.data() + i * input_stride, channels,
                           output_ref.data() + i * channels, *op_info,
                           input_quantization, output_quantization, op_params);

        for (size_t c = 0; c < channels; c++) {
          const float x = input[i * input_stride + c];
          const float y = output[i * output_stride + c];
          const float y_ref = output_ref[i * channels + c];
          if (op_info->IsInSupportedRange(y_ref)) {
            if (std::isnan(static_cast<float>(y_ref))) {
              ASSERT_TRUE(std::isnan(static_cast<float>(y)));
            } else {
              ASSERT_NEAR(y, y_ref, op_info->Tolerance(y_ref, output_datatype))
                  << "x = " << x << ", y = " << y
                  << ", input1 zero point = " << input_quantization.zero_point
                  << ", input1 scale = " << input_quantization.scale
                  << ", output zero point = " << output_quantization.zero_point
                  << ", output scale = " << output_quantization.scale;
            }
          }
        }
      }
    }
  }

  template <typename In>
  void RunUnaryTest(const UnaryOpTestParams& test_params, const Param& param) {
    switch (param.output_datatype) {
      case xnn_datatype_fp16:
        RunUnaryTest<In, xnn_float16>(test_params, param);
        break;
      case xnn_datatype_bf16:
        RunUnaryTest<In, xnn_bfloat16>(test_params, param);
        break;
      case xnn_datatype_fp32:
        RunUnaryTest<In, float>(test_params, param);
        break;
      case xnn_datatype_int32:
        RunUnaryTest<In, int32_t>(test_params, param);
        break;
      case xnn_datatype_quint8:
        RunUnaryTest<In, xnnpack::quantized<uint8_t>>(test_params, param);
        break;
      case xnn_datatype_qint8:
        RunUnaryTest<In, xnnpack::quantized<int8_t>>(test_params, param);
        break;
      default:
        XNN_UNREACHABLE;
    }
  }

  void RunUnaryTest(const UnaryOpTestParams& test_params, const Param& param) {
    switch (param.input_datatype) {
      case xnn_datatype_fp16:
        RunUnaryTest<xnn_float16>(test_params, param);
        break;
      case xnn_datatype_bf16:
        RunUnaryTest<xnn_bfloat16>(test_params, param);
        break;
      case xnn_datatype_fp32:
        RunUnaryTest<float>(test_params, param);
        break;
      case xnn_datatype_int32:
        RunUnaryTest<int32_t>(test_params, param);
        break;
      case xnn_datatype_quint8:
        RunUnaryTest<xnnpack::quantized<uint8_t>>(test_params, param);
        break;
      case xnn_datatype_qint8:
        RunUnaryTest<xnnpack::quantized<int8_t>>(test_params, param);
        break;
      default:
        XNN_UNREACHABLE;
    }
  }
};

using UnaryNCTest_InputQuantized = UnaryNCTest;
using UnaryNCTest_OutputQuantized = UnaryNCTest;

TEST_P(UnaryNCTest, UnitBatch) {
  for (size_t c = 0; c < 100; c += 15) {
    RunUnaryTest(UnaryOpTestParams::UnitBatch().Channels(c), GetParam());
  }
}

TEST_P(UnaryNCTest, SmallBatch) {
  for (size_t c = 0; c < 100; c += 15) {
    RunUnaryTest(UnaryOpTestParams::SmallBatch().Channels(c), GetParam());
  }
}

TEST_P(UnaryNCTest, SmallBatch_InputStride) {
  for (size_t c = 0; c < 100; c += 15) {
    RunUnaryTest(UnaryOpTestParams::UnitBatch().Channels(c).InputStride(129),
                 GetParam());
  }
}

TEST_P(UnaryNCTest, UnitBatch_OutputStride) {
  for (size_t c = 0; c < 100; c += 15) {
    RunUnaryTest(UnaryOpTestParams::UnitBatch().Channels(c).OutputStride(117),
                 GetParam());
  }
}

TEST_P(UnaryNCTest, StridedBatch) {
  for (size_t c = 0; c < 100; c += 15) {
    RunUnaryTest(UnaryOpTestParams::StridedBatch().Channels(c), GetParam());
  }
}

std::vector<float> ZeroPoints(xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_qint8:
      return {-128, -127, -1, 0, 1, 126, 127};
    case xnn_datatype_quint8:
      return {0, 1, 127, 128, 129, 254, 255};
    default:
      XNN_UNREACHABLE;
  }
}

TEST_P(UnaryNCTest_InputQuantized, InputQuantized) {
  for (int zero_point : ZeroPoints(GetParam().input_datatype)) {
    for (float scale : {1.0e-2f, 1.0e2f, 10.0f}) {
      RunUnaryTest(
          UnaryOpTestParams::UnitBatch().InputQuantization({zero_point, scale}),
          GetParam());
    }
  }
}

TEST_P(UnaryNCTest_OutputQuantized, OutputQuantized) {
  for (int zero_point : ZeroPoints(GetParam().output_datatype)) {
    for (float scale : {1.0e-2f, 1.0e2f, 10.0f}) {
      RunUnaryTest(UnaryOpTestParams::UnitBatch().OutputQuantization(
                       {zero_point, scale}),
                   GetParam());
    }
  }
}

xnn_unary_operator all_unary_ops[] = {
    xnn_unary_clamp,
    xnn_unary_abs,
    xnn_unary_bankers_rounding,
    xnn_unary_ceiling,
    xnn_unary_elu,
    xnn_unary_exp,
    xnn_unary_floor,
    xnn_unary_gelu,
    xnn_unary_hardswish,
    xnn_unary_leaky_relu,
    xnn_unary_log,
    xnn_unary_negate,
    xnn_unary_sigmoid,
    xnn_unary_square,
    xnn_unary_square_root,
    xnn_unary_reciprocal_square_root,
    xnn_unary_tanh,
    xnn_unary_cube_root,
    xnn_unary_cosine,
    xnn_unary_sine,
    xnn_unary_count_leading_zeros,
    xnn_unary_bitwise_not,
    xnn_unary_popcount,
    xnn_unary_sign,
};

xnn_datatype all_datatypes[] = {
    xnn_datatype_quint8, xnn_datatype_qint8,
#ifndef XNN_EXCLUDE_F16_TESTS
    xnn_datatype_fp16,
#endif
    xnn_datatype_bf16,   xnn_datatype_fp32,  xnn_datatype_int32,
};

xnn_datatype quantized_datatypes[] = {
    xnn_datatype_quint8,
    xnn_datatype_qint8,
};

xnn_datatype unquantized_datatypes[] = {
#ifndef XNN_EXCLUDE_F16_TESTS
    xnn_datatype_fp16,
#endif
    xnn_datatype_bf16,
    xnn_datatype_fp32,
    xnn_datatype_int32,
};

RunMode run_modes[] = {RunMode::kCreateReshapeRun, RunMode::kEager};

// Run non-quantized tests on all unary ops and all datatypes.
INSTANTIATE_TEST_SUITE_P(UnaryNCTest, UnaryNCTest,
                         testing::ConvertGenerator<Param::UnaryT>(
                             testing::Combine(testing::ValuesIn(all_unary_ops),
                                              testing::ValuesIn(all_datatypes),
                                              testing::ValuesIn(run_modes))),
                         [](const auto& info) { return info.param.Name(); });

// Run quantized input and output tests on all unary ops and all quantized
// datatypes.
INSTANTIATE_TEST_SUITE_P(
    UnaryNCTest_InputQuantized, UnaryNCTest_InputQuantized,
    testing::ConvertGenerator<Param::UnaryT>(testing::Combine(
        testing::ValuesIn(all_unary_ops),
        testing::ValuesIn(quantized_datatypes), testing::ValuesIn(run_modes))),
    [](const auto& info) { return info.param.Name(); });

INSTANTIATE_TEST_SUITE_P(
    UnaryNCTest_OutputQuantized, UnaryNCTest_OutputQuantized,
    testing::ConvertGenerator<Param::UnaryT>(testing::Combine(
        testing::ValuesIn(all_unary_ops),
        testing::ValuesIn(quantized_datatypes), testing::ValuesIn(run_modes))),
    [](const auto& info) { return info.param.Name(); });

// Run non-quantized tests all all possible convert datatype combinations.
INSTANTIATE_TEST_SUITE_P(
    ConvertNCTest, UnaryNCTest,
    testing::ConvertGenerator<Param::ConvertT>(testing::Combine(
        testing::Values(xnn_unary_convert), testing::ValuesIn(all_datatypes),
        testing::ValuesIn(all_datatypes), testing::ValuesIn(run_modes))),
    [](const auto& info) { return info.param.Name(); });

// Run quantized input conversions.
INSTANTIATE_TEST_SUITE_P(
    ConvertNCTest_InputQuantized, UnaryNCTest_InputQuantized,
    testing::ConvertGenerator<Param::ConvertT>(testing::Combine(
        testing::Values(xnn_unary_convert),
        testing::ValuesIn(quantized_datatypes),
        testing::ValuesIn(all_datatypes), testing::ValuesIn(run_modes))),
    [](const auto& info) { return info.param.Name(); });

// Run quantized output conversions.
INSTANTIATE_TEST_SUITE_P(
    ConvertNCTest_OutputQuantized, UnaryNCTest_OutputQuantized,
    testing::ConvertGenerator<Param::ConvertT>(testing::Combine(
        testing::Values(xnn_unary_convert), testing::ValuesIn(all_datatypes),
        testing::ValuesIn(quantized_datatypes), testing::ValuesIn(run_modes))),
    [](const auto& info) { return info.param.Name(); });
