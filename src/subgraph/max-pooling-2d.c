// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/requantization.h"
#include "src/xnnpack/subgraph-validation.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

static enum xnn_status create_max_pooling_operator(
  const struct xnn_node* node,
  const struct xnn_runtime_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 1);
  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const uint32_t input_id = node->inputs[0];
  assert(input_id < num_values);
  const struct xnn_runtime_value *input_value = &values[input_id];
  const enum xnn_datatype datatype = input_value->datatype;

  enum xnn_status status;
  switch (datatype) {
    case xnn_datatype_fp16:
      status = xnn_create_max_pooling2d_nhwc_f16(
        node->params.pooling_2d.padding_top,
        node->params.pooling_2d.padding_right,
        node->params.pooling_2d.padding_bottom,
        node->params.pooling_2d.padding_left,
        node->params.pooling_2d.pooling_height,
        node->params.pooling_2d.pooling_width,
        node->params.pooling_2d.stride_height,
        node->params.pooling_2d.stride_width,
        node->params.pooling_2d.dilation_height,
        node->params.pooling_2d.dilation_width,
        node->activation.output_min,
        node->activation.output_max,
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case xnn_datatype_fp32:
      status = xnn_create_max_pooling2d_nhwc_f32(
        node->params.pooling_2d.padding_top,
        node->params.pooling_2d.padding_right,
        node->params.pooling_2d.padding_bottom,
        node->params.pooling_2d.padding_left,
        node->params.pooling_2d.pooling_height,
        node->params.pooling_2d.pooling_width,
        node->params.pooling_2d.stride_height,
        node->params.pooling_2d.stride_width,
        node->params.pooling_2d.dilation_height,
        node->params.pooling_2d.dilation_width,
        node->activation.output_min,
        node->activation.output_max,
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case xnn_datatype_qint8:
    {
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const int8_t output_min = xnn_qs8_quantize(node->activation.output_min, output_scale, output_zero_point);
      const int8_t output_max = xnn_qs8_quantize(node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_max_pooling2d_nhwc_s8(
        node->params.pooling_2d.padding_top,
        node->params.pooling_2d.padding_right,
        node->params.pooling_2d.padding_bottom,
        node->params.pooling_2d.padding_left,
        node->params.pooling_2d.pooling_height,
        node->params.pooling_2d.pooling_width,
        node->params.pooling_2d.stride_height,
        node->params.pooling_2d.stride_width,
        node->params.pooling_2d.dilation_height,
        node->params.pooling_2d.dilation_width,
        output_min,
        output_max,
        node->flags,
        &opdata->operator_objects[0]);
      break;
    }
    case xnn_datatype_quint8:
    {
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const uint8_t output_min = xnn_qu8_quantize(node->activation.output_min, output_scale, output_zero_point);
      const uint8_t output_max = xnn_qu8_quantize(node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_max_pooling2d_nhwc_u8(
        node->params.pooling_2d.padding_top,
        node->params.pooling_2d.padding_right,
        node->params.pooling_2d.padding_bottom,
        node->params.pooling_2d.padding_left,
        node->params.pooling_2d.pooling_height,
        node->params.pooling_2d.pooling_width,
        node->params.pooling_2d.stride_height,
        node->params.pooling_2d.stride_width,
        node->params.pooling_2d.dilation_height,
        node->params.pooling_2d.dilation_width,
        output_min,
        output_max,
        node->flags,
        &opdata->operator_objects[0]);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status reshape_max_pooling_operator(
  struct xnn_operator_data* opdata,
  struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);

  struct xnn_runtime_value* output_value = values + output_id;

  const size_t batch_size = values[input_id].shape.dim[0];
  const size_t input_height = values[input_id].shape.dim[1];
  const size_t input_width = values[input_id].shape.dim[2];
  const size_t channels = values[input_id].shape.dim[3];

  enum xnn_status status = xnn_status_invalid_state;
  size_t output_height, output_width;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_max_pooling_nhwc_f16:
      status = xnn_reshape_max_pooling2d_nhwc_f16(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channels,
        /*input_pixel_stride=*/channels,
        /*output_pixel_stride=*/channels,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_max_pooling_nhwc_f32:
      status = xnn_reshape_max_pooling2d_nhwc_f32(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channels,
        /*input_pixel_stride=*/channels,
        /*output_pixel_stride=*/channels,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_max_pooling_nhwc_s8:
      status = xnn_reshape_max_pooling2d_nhwc_s8(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channels,
        /*input_pixel_stride=*/channels,
        /*output_pixel_stride=*/channels,
        &output_height,
        &output_width,
        threadpool);
      break;
    case xnn_operator_type_max_pooling_nhwc_u8:
      status = xnn_reshape_max_pooling2d_nhwc_u8(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channels,
        /*input_pixel_stride=*/channels,
        /*output_pixel_stride=*/channels,
        &output_height,
        &output_width,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (status != xnn_status_success) {
    return status;
  }

  output_value->shape.dim[0] = batch_size;
  output_value->shape.dim[1] = output_height;
  output_value->shape.dim[2] = output_width;
  output_value->shape.dim[3] = channels;

  output_value->shape.num_dims = 4;
  const size_t new_size = xnn_runtime_tensor_get_size(output_value);
  if (new_size > output_value->size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_max_pooling_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_runtime_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  const struct xnn_runtime_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_max_pooling_nhwc_f16:
      return xnn_setup_max_pooling2d_nhwc_f16(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_max_pooling_nhwc_f32:
      return xnn_setup_max_pooling2d_nhwc_f32(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_max_pooling_nhwc_s8:
      return xnn_setup_max_pooling2d_nhwc_s8(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_max_pooling_nhwc_u8:
      return xnn_setup_max_pooling2d_nhwc_u8(
        opdata->operator_objects[0],
        input_data,
        output_data);
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_max_pooling_2d(
  xnn_subgraph_t subgraph,
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_max_pooling_2d)) != xnn_status_success) {
    return status;
  }

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), pooling_width, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (stride_height == 0 || stride_width == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " stride: stride dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), stride_width, stride_height);
    return xnn_status_invalid_parameter;
  }

  if (dilation_height == 0 || dilation_width == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_max_pooling_2d), dilation_width, dilation_height);
    return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_min_max(xnn_node_type_max_pooling_2d, output_min, output_max);
  if (status != xnn_status_success) {
    return status;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to define %s operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        xnn_node_type_to_string(xnn_node_type_max_pooling_2d),
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      return xnn_status_invalid_parameter;
    }
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_max_pooling_2d, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_max_pooling_2d, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_max_pooling_2d), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_max_pooling_2d, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_max_pooling_2d, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (output_value->datatype) {
    case xnn_datatype_fp16:
      break;
    case xnn_datatype_fp32:
      break;
    case xnn_datatype_qint8:
      break;
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_max_pooling_2d), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_datatype_matches(
    xnn_node_type_max_pooling_2d, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_quantization_parameter_matches(
      xnn_node_type_max_pooling_2d, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_max_pooling_2d;
  node->params.pooling_2d.padding_top = input_padding_top;
  node->params.pooling_2d.padding_right = input_padding_right;
  node->params.pooling_2d.padding_bottom = input_padding_bottom;
  node->params.pooling_2d.padding_left = input_padding_left;
  node->params.pooling_2d.pooling_height = pooling_height;
  node->params.pooling_2d.pooling_width = pooling_width;
  node->params.pooling_2d.stride_height = stride_height;
  node->params.pooling_2d.stride_width = stride_width;
  node->params.pooling_2d.dilation_height = dilation_height;
  node->params.pooling_2d.dilation_width = dilation_width;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_max_pooling_operator;
  node->reshape = reshape_max_pooling_operator;
  node->setup = setup_max_pooling_operator;

  return xnn_status_success;
}
