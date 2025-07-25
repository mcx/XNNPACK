// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_OPERATORS_CONVOLUTION_TEST_HELPERS_H_
#define XNNPACK_TEST_OPERATORS_CONVOLUTION_TEST_HELPERS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/xnnpack/buffer.h"
#include "src/xnnpack/quantization.h"

namespace xnnpack {
void compute_convolution_qd8_f32_qc8w_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    const xnnpack::Buffer<int8_t>& input, const xnnpack::Buffer<int8_t>& filter,
    const xnnpack::Buffer<float>& filter_scale,
    const xnnpack::Buffer<xnn_qd8_quantization_params>& quantization_params,
    xnnpack::Buffer<float>& output, bool has_bias,
    const xnnpack::Buffer<float>& bias);

void compute_convolution_qs8_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    int8_t input_zero_point, const xnnpack::Buffer<int8_t>& input,
    const xnnpack::Buffer<int8_t>& filter,
    xnnpack::Buffer<int32_t>& accumulators, bool has_bias,
    const xnnpack::Buffer<int32_t>& bias);

void compute_convolution_qs8_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t groups, size_t group_input_channels,
    size_t group_output_channels, int8_t input_zero_point,
    const xnnpack::Buffer<int8_t>& input, const xnnpack::Buffer<int8_t>& filter,
    xnnpack::Buffer<int32_t>& accumulators, bool has_bias,
    const xnnpack::Buffer<int32_t>& bias);

void compute_convolution_qu8_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t groups, size_t group_input_channels,
    size_t group_output_channels, uint8_t input_zero_point,
    uint8_t kernel_zero_point, const xnnpack::Buffer<uint8_t>& input,
    const xnnpack::Buffer<uint8_t>& filter,
    xnnpack::Buffer<int32_t>& accumulators, bool has_bias,
    const xnnpack::Buffer<int32_t>& bias);

void compute_convolution_qu8_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    uint8_t input_zero_point, uint8_t kernel_zero_point,
    const xnnpack::Buffer<uint8_t>& input,
    const xnnpack::Buffer<uint8_t>& filter,
    xnnpack::Buffer<int32_t>& accumulators, bool has_bias,
    const xnnpack::Buffer<int32_t>& bias);

void compute_depthwise_convolution_qs8_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t input_channels, size_t depth_multiplier,
    size_t input_channel_stride, int8_t input_zero_point,
    const xnnpack::Buffer<int8_t>& input, const xnnpack::Buffer<int8_t>& filter,
    xnnpack::Buffer<int32_t>& accumulators, bool has_bias,
    const xnnpack::Buffer<int32_t>& bias);

void compute_depthwise_convolution_qs8_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t input_channels, size_t depth_multiplier,
    int8_t input_zero_point, const xnnpack::Buffer<int8_t>& input,
    const xnnpack::Buffer<int8_t>& filter,
    xnnpack::Buffer<int32_t>& accumulators, bool has_bias,
    const xnnpack::Buffer<int32_t>& bias);

void compute_depthwise_convolution_qu8_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t input_channels, size_t depth_multiplier,
    size_t input_channel_stride, uint8_t input_zero_point,
    uint8_t kernel_zero_point, const xnnpack::Buffer<uint8_t>& input,
    const xnnpack::Buffer<uint8_t>& filter,
    xnnpack::Buffer<int32_t>& accumulators, bool has_bias,
    const xnnpack::Buffer<int32_t>& bias);

void compute_depthwise_convolution_qu8_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t input_channels, size_t depth_multiplier,
    uint8_t input_zero_point, uint8_t kernel_zero_point,
    const xnnpack::Buffer<uint8_t>& input,
    const xnnpack::Buffer<uint8_t>& filter,
    xnnpack::Buffer<int32_t>& accumulators, bool has_bias,
    const xnnpack::Buffer<int32_t>& bias);
}  // namespace xnnpack

#endif  // XNNPACK_TEST_OPERATORS_CONVOLUTION_TEST_HELPERS_H_
