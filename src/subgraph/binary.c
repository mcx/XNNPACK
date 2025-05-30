// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/reshape-helpers.h"
#include "src/xnnpack/subgraph-validation.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

static enum xnn_status create_binary_operator(
  const struct xnn_node* node,
  const struct xnn_runtime_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  xnn_weights_cache_t weights_cache)
{
  const uint32_t input1_id = opdata->inputs[0];
  assert(input1_id < num_values);
  const uint32_t input2_id = opdata->inputs[1];
  assert(input2_id < num_values);
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);

  enum xnn_datatype datatype = values[output_id].datatype;
  struct xnn_quantization_params a_quantization = {
    .scale = values[input1_id].quantization.scale,
    .zero_point = values[input1_id].quantization.zero_point,
  };
  struct xnn_quantization_params b_quantization = {
    .scale = values[input2_id].quantization.scale,
    .zero_point = values[input2_id].quantization.zero_point,
  };
  struct xnn_quantization_params output_quantization = {
    .scale = values[output_id].quantization.scale,
    .zero_point = values[output_id].quantization.zero_point,
  };

  return xnn_create_binary_elementwise_nd(
    node->binary_operator,
    datatype, &a_quantization, &b_quantization, &output_quantization,
    node->flags,
    &opdata->operator_objects[0]);
}

static enum xnn_status reshape_binary_operator(
  struct xnn_operator_data* opdata,
  struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input1_id = opdata->inputs[0];
  assert(input1_id < num_values);
  const uint32_t input2_id = opdata->inputs[1];
  assert(input2_id < num_values);
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);

  struct xnn_shape shape2;
  opdata->shape1.num_dims = values[input1_id].shape.num_dims;
  shape2.num_dims = values[input2_id].shape.num_dims;
  if (values[output_id].flags & XNN_VALUE_FLAG_LAYOUT_NCHW) {
    assert(values[input1_id].flags & XNN_VALUE_FLAG_LAYOUT_NCHW);
    assert(values[input2_id].flags & XNN_VALUE_FLAG_LAYOUT_NCHW);
    opdata->shape1.dim[0] = values[input1_id].shape.dim[0];
    opdata->shape1.dim[1] = values[input1_id].shape.dim[values[input1_id].shape.num_dims - 1];
    if (values[input1_id].shape.num_dims > 2) {
      memcpy(&opdata->shape1.dim[2], &values[input1_id].shape.dim[1], (values[input1_id].shape.num_dims - 2) * sizeof(size_t));
    }
    shape2.dim[0] = values[input2_id].shape.dim[0];
    shape2.dim[1] = values[input2_id].shape.dim[values[input2_id].shape.num_dims - 1];
    if (values[input1_id].shape.num_dims > 2) {
      memcpy(&shape2.dim[2], &values[input2_id].shape.dim[1], (values[input2_id].shape.num_dims - 2) * sizeof(size_t));
    }
  } else {
    assert((values[output_id].flags & XNN_VALUE_FLAG_LAYOUT_NCHW) == 0);
    assert((values[input1_id].flags & XNN_VALUE_FLAG_LAYOUT_NCHW) == 0);
    assert((values[input2_id].flags & XNN_VALUE_FLAG_LAYOUT_NCHW) == 0);
    memcpy(opdata->shape1.dim, values[input1_id].shape.dim, values[input1_id].shape.num_dims * sizeof(size_t));
    memcpy(shape2.dim, values[input2_id].shape.dim, values[input2_id].shape.num_dims * sizeof(size_t));
  }

  // Handle scalars. Although the output shape is dimensionless, the reshape
  // function must be passed a valid shape to prevent skipping the op.
  if (opdata->shape1.num_dims == 0) {
    opdata->shape1.num_dims = 1;
    opdata->shape1.dim[0] = 1;
  }
  if (shape2.num_dims == 0) {
    shape2.num_dims = 1;
    shape2.dim[0] = 1;
  }
  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_reshape_binary_elementwise_nd(
    opdata->operator_objects[0],
    opdata->shape1.num_dims,
    opdata->shape1.dim,
    shape2.num_dims,
    shape2.dim,
    threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  return resize_binary_elementwise_output_tensor(opdata, values, num_values, old_workspace_size, threadpool);
}

static enum xnn_status setup_binary_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input1_id = opdata->inputs[0];
  assert(input1_id != XNN_INVALID_VALUE_ID);
  assert(input1_id < num_values);

  const uint32_t input2_id = opdata->inputs[1];
  assert(input2_id != XNN_INVALID_VALUE_ID);
  assert(input2_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_runtime_value* input1_value = values + input1_id;
  const void* input1_data = input1_value->data;
  assert(input1_data != NULL);

  const struct xnn_runtime_value* input2_value = values + input2_id;
  const void* input2_data = input2_value->data;
  assert(input2_data != NULL);

  const struct xnn_runtime_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  return xnn_setup_binary_elementwise_nd(
    opdata->operator_objects[0],
    input1_data, input2_data, output_data);
}

enum xnn_status xnn_define_binary(
  xnn_subgraph_t subgraph,
  enum xnn_binary_operator type,
  const struct xnn_binary_params* params,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_binary_elementwise)) != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_nth_input_node_id(xnn_node_type_binary_elementwise, input1_id, subgraph->num_values, 1)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input1_value = &subgraph->values[input1_id];
  status = xnn_subgraph_check_nth_input_type_dense(xnn_node_type_binary_elementwise, input1_id, input1_value, 1);
  if (status != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_nth_input_node_id(xnn_node_type_binary_elementwise, input2_id, subgraph->num_values, 2)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input2_value = &subgraph->values[input2_id];
  status = xnn_subgraph_check_nth_input_type_dense(xnn_node_type_binary_elementwise, input2_id, input2_value, 2);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_binary_elementwise, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_binary_elementwise, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_datatype_matches_two_inputs(
      xnn_node_type_binary_elementwise, input1_id, input1_value, input2_id, input2_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_binary_elementwise;
  node->binary_operator = type;
  node->num_inputs = 2;
  node->inputs[0] = input1_id;
  node->inputs[1] = input2_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_binary_operator;
  node->reshape = reshape_binary_operator;
  node->setup = setup_binary_operator;

  if (params) {
    if (params->output_min != -INFINITY || params->output_max != INFINITY) {
      xnn_insert_clamp_node(subgraph, params->output_min, params->output_max, node);
    }
  }

  return xnn_status_success;
}
