// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/packq.h"
#include "src/xnnpack/params.h"
#include "src/xnnpack/subgraph.h"

static void set_allocation_type(struct xnn_value* value)
{
  if (value->data != NULL) {
    value->allocation_type = xnn_allocation_type_static;
  } else if ((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0) {
    value->allocation_type = xnn_allocation_type_external;
  } else {
    value->allocation_type = xnn_allocation_type_workspace;
  }
}

static void set_shape(struct xnn_value* value, size_t num_dims, const size_t* dims)
{
  value->shape.num_dims = num_dims;
  if (num_dims != 0) {
    if (dims != NULL) {
      memcpy(value->shape.dim, dims, num_dims * sizeof(size_t));
    } else {
      memset(value->shape.dim, 0, num_dims * sizeof(size_t));
    }
  }
}

static enum xnn_status check_zero_point(
  enum xnn_datatype datatype,
  int32_t zero_point)
{
  switch (datatype) {
    case xnn_datatype_qcint4:
    case xnn_datatype_qbint4:
      if (zero_point < 0 || zero_point > 15) {
        xnn_log_error(
          "failed to create Quantized Dense Tensor value: invalid zero point %" PRId32" outside the [0, 15] range",
          zero_point);
        return xnn_status_invalid_parameter;
      }
      break;
    case xnn_datatype_qcint8:
    case xnn_datatype_qint8:
      if ((int32_t) (int8_t) zero_point != zero_point) {
        xnn_log_error(
          "failed to create Quantized Dense Tensor value: invalid zero point %" PRId32" outside the [-128, 127] range",
          zero_point);
        return xnn_status_invalid_parameter;
      }
      break;
    case xnn_datatype_quint8:
      if ((int32_t) (uint8_t) zero_point != zero_point) {
        xnn_log_error(
          "failed to create Quantized Dense Tensor value: invalid zero point %" PRId32" outside the [0, 255] range",
          zero_point);
        return xnn_status_invalid_parameter;
      }
      break;
    case xnn_datatype_qcint32:
    case xnn_datatype_qint32:
      if (zero_point != 0) {
        xnn_log_error(
          "failed to create Quantized Dense Tensor value: invalid non-zero zero point %" PRId32,
          zero_point);
        return xnn_status_invalid_parameter;
      }
      break;
    default:
      xnn_log_error("failed to create Quantized Dense Tensor value: unsupported datatype %s (%d)",
        xnn_datatype_to_string(datatype), datatype);
      return xnn_status_unsupported_parameter;
  }

  return xnn_status_success;
}

enum xnn_status xnn_define_tensor_value(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    size_t num_dims,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create Dense Tensor value: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (external_id != XNN_INVALID_VALUE_ID && external_id >= subgraph->external_value_ids) {
    xnn_log_error(
      "failed to create Dense Tensor value: "
      "external ID %" PRIu32 " exceeds the number of reserved external IDs in subgraph (%" PRIu32 ")",
      external_id, subgraph->external_value_ids);
    return xnn_status_invalid_parameter;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error("failed to create Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)",
      XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  switch (datatype) {
    case xnn_datatype_fp32:
    case xnn_datatype_fp16:
    case xnn_datatype_bf16:
    case xnn_datatype_int32:
    case xnn_datatype_pfp16:  // TODO: Does this really belong here?
    case xnn_datatype_pfp32:  // TODO: Does this really belong here?
      break;
    default:
      xnn_log_error("failed to create Dense Tensor value: unsupported datatype %s (%d)",
        xnn_datatype_to_string(datatype), datatype);
      return xnn_status_unsupported_parameter;
  }

  struct xnn_value* value = subgraph->values + external_id;
  if (external_id == XNN_INVALID_VALUE_ID) {
    value = xnn_subgraph_new_internal_value(subgraph);
    if (value == NULL) {
      return xnn_status_out_of_memory;
    }
  }
  value->type = xnn_value_type_dense_tensor;
  value->datatype = datatype;
  set_shape(value, num_dims, dims);
  value->size = xnn_tensor_get_size_by_id(subgraph, value->id);
  value->flags = flags;
  value->data = (void*) (uintptr_t) data;
  set_allocation_type(value);

  *id_out = value->id;
  return xnn_status_success;
}

enum xnn_status xnn_define_quantized_tensor_value(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    int32_t zero_point,
    float scale,
    size_t num_dims,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create Quantized Dense Tensor value: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (external_id != XNN_INVALID_VALUE_ID && external_id >= subgraph->external_value_ids) {
    xnn_log_error(
      "failed to create Quantized Dense Tensor value: "
      "external ID %" PRIu32 " exceeds the number of reserved external IDs in subgraph (%" PRIu32 ")",
      external_id, subgraph->external_value_ids);
    return xnn_status_invalid_parameter;
  }

  enum xnn_status status = xnn_validate_quantized_tensor(
      datatype,
      zero_point,
      scale,
      num_dims,
      dims);
  if (status != xnn_status_success) {
    return status;
  }
  struct xnn_value* value = subgraph->values + external_id;
  if (external_id == XNN_INVALID_VALUE_ID) {
    value = xnn_subgraph_new_internal_value(subgraph);
    if (value == NULL) {
      return xnn_status_out_of_memory;
    }
  }
  value->type = xnn_value_type_dense_tensor;
  value->datatype = datatype;
  value->quantization.zero_point = zero_point;
  value->quantization.scale = scale;
  set_shape(value, num_dims, dims);
  value->size = xnn_tensor_get_size_by_id(subgraph, value->id);
  value->flags = flags;
  value->data = (void*) (uintptr_t) data;
  set_allocation_type(value);

  *id_out = value->id;
  return xnn_status_success;
}

enum xnn_status xnn_define_dynamically_quantized_tensor_value(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    size_t num_dims,
    size_t num_nonbatch_dims,
    const size_t* dims,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create Dynamically Quantized Dense Tensor value: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (external_id != XNN_INVALID_VALUE_ID && external_id >= subgraph->external_value_ids) {
    xnn_log_error(
      "failed to create Dynamically Quantized Dense Tensor value: "
      "external ID %" PRIu32 " exceeds the number of reserved external IDs in subgraph (%" PRIu32 ")",
      external_id, subgraph->external_value_ids);
    return xnn_status_invalid_parameter;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create Dynamically Quantized Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)",
      XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  if (num_nonbatch_dims > num_dims) {
    xnn_log_error(
      "failed to create Dynamically Quantized Dense Tensor value: "
      "non batch dimensions %zu is greater than number of dimensions %zu",
      num_nonbatch_dims, num_dims);
    return xnn_status_invalid_parameter;
  }

  switch (datatype) {
    case xnn_datatype_qdint8:
    case xnn_datatype_qpint8:
      break;
    default:
      xnn_log_error("failed to create Dynamically Quantized Dense Tensor value: unsupported datatype %s (%d)",
        xnn_datatype_to_string(datatype), datatype);
      return xnn_status_unsupported_parameter;
  }

  if ((flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0) {
    xnn_log_error(
        "failed to create Dynamically Quantized Dense Tensor value: "
        "external dynamically quantized tensors are not supported.");
    return xnn_status_unsupported_parameter;
  }

  struct xnn_value* value = subgraph->values + external_id;
  if (external_id == XNN_INVALID_VALUE_ID) {
    value = xnn_subgraph_new_internal_value(subgraph);
    if (value == NULL) {
      return xnn_status_out_of_memory;
    }
  }
  value->type = xnn_value_type_dense_tensor;
  value->datatype = datatype;
  value->quantization.num_nonbatch_dims = num_nonbatch_dims;
  set_shape(value, num_dims, dims);
  value->size = xnn_tensor_get_size_by_id(subgraph, value->id);
  value->quantization.dynamic_params_size =  xnn_tensor_get_dynamic_quant_param_size(value->datatype, &value->shape, value->quantization.num_nonbatch_dims);
  value->flags = flags;
  value->data = NULL;
  set_allocation_type(value);

  *id_out = value->id;
  return xnn_status_success;
}

enum xnn_status xnn_define_channelwise_quantized_tensor_value(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    const float* scale,
    size_t num_dims,
    size_t channel_dim,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  return xnn_define_channelwise_quantized_tensor_value_v2(
    subgraph, datatype,
    /*zero_point=*/0, scale,
    num_dims, channel_dim, dims, data,
    external_id, flags,
    id_out);
}

enum xnn_status xnn_validate_quantized_tensor(
    enum xnn_datatype datatype,
    int32_t zero_point,
    float scale,
    size_t num_dims,
    const size_t* dims)
{
  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create Quantized Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)",
      XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  enum xnn_status status = check_zero_point(datatype, zero_point);
  if (status != xnn_status_success) {
    return status;
  }

  if (scale <= 0.0f || !isnormal(scale)) {
    xnn_log_error(
      "failed to create Quantized Dense Tensor value with %.7g scale: scale must be finite, normalized, and positive",
      scale);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_validate_channelwise_quantized_tensor(
    enum xnn_datatype datatype,
    int32_t zero_point,
    const float* scale,
    size_t num_dims,
    size_t channel_dim,
    const size_t* dims)
{
  if (num_dims == 0) {
    xnn_log_error(
      "failed to create Channelwise Quantized Dense Tensor value: no channel dimension exists");
    return xnn_status_invalid_parameter;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create Channelwise Quantized Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)",
      XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  if (channel_dim >= num_dims) {
    xnn_log_error(
      "failed to create Channelwise Quantized Dense Tensor value: "
      "channel dimension index %zu is out of range for %zu-dimensional tensor",
      channel_dim, num_dims);
    return xnn_status_invalid_parameter;
  }

  enum xnn_status status = check_zero_point(datatype, zero_point);
  if (status != xnn_status_success) {
    return status;
  }

  switch (datatype) {
    case xnn_datatype_qcint4:
    case xnn_datatype_qcint8:
    case xnn_datatype_qcint32:
      break;
    default:
      xnn_log_error("failed to create Channelwise Quantized Dense Tensor value: unsupported datatype %s (%d)",
        xnn_datatype_to_string(datatype), datatype);
      return xnn_status_unsupported_parameter;
  }

  const size_t channels = dims[channel_dim];
  for (size_t channel = 0; channel < channels; channel++) {
    if (scale[channel] <= 0.0f || !isnormal(scale[channel])) {
      xnn_log_error(
        "failed to create Channelwise Quantized Dense Tensor value with %.7g scale in channel #%zu: "
        "scale must be finite, normalized, and positive",
        scale[channel], channel);
      return xnn_status_invalid_parameter;
    }
  }
  return xnn_status_success;
}

enum xnn_status xnn_define_channelwise_quantized_tensor_value_v2(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    int32_t zero_point,
    const float* scale,
    size_t num_dims,
    size_t channel_dim,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create Channelwise Quantized Dense Tensor value: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (external_id != XNN_INVALID_VALUE_ID && external_id >= subgraph->external_value_ids) {
    xnn_log_error(
      "failed to create Channelwise Quantized Dense Tensor value: "
      "external ID %" PRIu32 " exceeds the number of reserved external IDs in subgraph (%" PRIu32 ")",
      external_id, subgraph->external_value_ids);
    return xnn_status_invalid_parameter;
  }

  enum xnn_status status = xnn_validate_channelwise_quantized_tensor(
      datatype,
      zero_point,
      scale,
      num_dims,
      channel_dim,
      dims);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_value* value = subgraph->values + external_id;
  if (external_id == XNN_INVALID_VALUE_ID) {
    value = xnn_subgraph_new_internal_value(subgraph);
    if (value == NULL) {
      return xnn_status_out_of_memory;
    }
  }
  value->type = xnn_value_type_dense_tensor;
  value->datatype = datatype;
  value->quantization.zero_point = zero_point;
  value->quantization.channelwise_scale = scale;
  value->quantization.channel_dimension = channel_dim;
  set_shape(value, num_dims, dims);
  value->size = xnn_tensor_get_size_by_id(subgraph, value->id);
  value->flags = flags;
  value->data = (void*) (uintptr_t) data;
  set_allocation_type(value);

  *id_out = value->id;
  return xnn_status_success;
}

enum xnn_status xnn_define_blockwise_quantized_tensor_value_v2(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    int32_t zero_point,
    const void* scale,
    size_t num_dims,
    size_t channel_dim,
    size_t block_size,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    enum xnn_datatype scale_type,
    uint32_t* id_out)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create Blockwise Quantized Dense Tensor value: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (external_id != XNN_INVALID_VALUE_ID && external_id >= subgraph->external_value_ids) {
    xnn_log_error(
      "failed to create Blockwise Quantized Dense Tensor value: "
      "external ID %" PRIu32 " exceeds the number of reserved external IDs in subgraph (%" PRIu32 ")",
      external_id, subgraph->external_value_ids);
    return xnn_status_invalid_parameter;
  }

  if (num_dims == 0) {
    xnn_log_error(
      "failed to create Blockwise Quantized Dense Tensor value: no channel dimension exists");
    return xnn_status_invalid_parameter;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create Blockwise Quantized Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)",
      XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  if (channel_dim >= num_dims) {
    xnn_log_error(
      "failed to create Blockwise Quantized Dense Tensor value: "
      "channel dimension index %zu is out of range for %zu-dimensional tensor",
      channel_dim, num_dims);
    return xnn_status_invalid_parameter;
  }

  if (block_size <= 0) {
    xnn_log_error(
      "failed to create Blockwise Quantized Dense Tensor value: "
      "block size is invalid. Got %zu\n", block_size);
  }

  enum xnn_status status = check_zero_point(datatype, zero_point);
  if (status != xnn_status_success) {
    return status;
  }

  switch (datatype) {
    case xnn_datatype_qbint4:
      break;
    default:
      xnn_log_error("failed to create Blockwise Quantized Dense Tensor value: unsupported datatype %s (%d)",
        xnn_datatype_to_string(datatype), datatype);
      return xnn_status_unsupported_parameter;
  }

  switch (scale_type) {
    case xnn_datatype_bf16:
    case xnn_datatype_fp16:
      break;
    default:
      xnn_log_error("failed to create Blockwise Quantized Dense Tensor value: unsupported scale datatype %s (%d)",
        xnn_datatype_to_string(scale_type), datatype);
      return xnn_status_unsupported_parameter;
  }
  const size_t block_count = dims[0] * dims[1] / block_size;
  for (size_t block = 0; block < block_count; block++) {
    float float_scale;
    switch (scale_type) {
      case xnn_datatype_bf16:
        float_scale = math_cvt_fp32_bf16(((const uint16_t*) scale)[block]);
        break;
      case xnn_datatype_fp16:
        float_scale = xnn_float16_to_float(((const xnn_float16*) scale)[block]);
        break;
      default:
        XNN_UNREACHABLE;
    }
    if (float_scale <= 0.0f || !isnormal(float_scale)) {
      xnn_log_error(
        "failed to create Blockwise Quantized Dense Tensor value with %.7g scale in block #%zu: "
        "scale must be finite, normalized, and positive",
        float_scale, block);
      return xnn_status_invalid_parameter;
    }
  }

  struct xnn_value* value = subgraph->values + external_id;
  if (external_id == XNN_INVALID_VALUE_ID) {
    value = xnn_subgraph_new_internal_value(subgraph);
    if (value == NULL) {
      return xnn_status_out_of_memory;
    }
  }
  value->type = xnn_value_type_dense_tensor;
  value->datatype = datatype;
  value->quantization.zero_point = zero_point;
  switch (scale_type) {
    case xnn_datatype_bf16:
      value->quantization.blockwise_scale.bf16_scale = (const xnn_bfloat16*) scale;
      value->quantization.scale_type = xnn_datatype_bf16;
      break;
    case xnn_datatype_fp16:
      value->quantization.blockwise_scale.fp16_scale = (const xnn_float16*) scale;
      value->quantization.scale_type = xnn_datatype_fp16;
      break;
    default:
      XNN_UNREACHABLE;
  }
  value->quantization.channel_dimension_blockwise = channel_dim;
  value->quantization.block_size = block_size;
  set_shape(value, num_dims, dims);
  value->size = xnn_tensor_get_size_by_id(subgraph, value->id);
  value->flags = flags;
  value->data = (void*) (uintptr_t) data;
  set_allocation_type(value);

  *id_out = value->id;
  return xnn_status_success;
}

enum xnn_status xnn_define_blockwise_quantized_tensor_value(
    xnn_subgraph_t subgraph,
    enum xnn_datatype datatype,
    int32_t zero_point,
    const uint16_t* scale,
    size_t num_dims,
    size_t channel_dim,
    size_t block_size,
    const size_t* dims,
    const void* data,
    uint32_t external_id,
    uint32_t flags,
    uint32_t* id_out) {
  return xnn_define_blockwise_quantized_tensor_value_v2(subgraph, datatype, zero_point, scale, num_dims, channel_dim,
                                                        block_size, dims, data, external_id, flags, xnn_datatype_bf16, id_out);
}

size_t xnn_shape_multiply_all_dims(
  const struct xnn_shape* shape)
{
  size_t batch_size = 1;
  for (size_t i = 0; i < shape->num_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

size_t xnn_shape_multiply_batch_dims(
  const struct xnn_shape* shape,
  size_t num_nonbatch_dims)
{
  size_t batch_size = 1;
  for (size_t i = 0; i + num_nonbatch_dims < shape->num_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

size_t xnn_shape_multiply_non_channel_dims(
  const struct xnn_shape* shape)
{
  size_t batch_size = 1;
  for (size_t i = 0; i + 1 < shape->num_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

size_t xnn_shape_multiply_leading_dims(
  const struct xnn_shape* shape,
  size_t num_leading_dims)
{
  size_t batch_size = 1;
  for (size_t i = 0; i < num_leading_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

size_t xnn_shape_multiply_trailing_dims(
  const struct xnn_shape* shape,
  size_t start_dim)
{
  size_t product = 1;
  for (size_t i = start_dim; i < shape->num_dims; i++) {
    product *= shape->dim[i];
  }
  return product;
}

size_t get_tensor_size(const struct xnn_gemm_config* gemm_config, enum xnn_value_type type,
                 enum xnn_datatype datatype, const struct xnn_shape *shape, uint32_t flags) {
  assert(type == xnn_value_type_dense_tensor);
  assert(datatype != xnn_datatype_invalid);

  // Special handling for packed quantized types.
  if (datatype == xnn_datatype_qpint8) {
    assert(gemm_config != NULL);
    size_t num_groups = xnn_shape_multiply_batch_dims(shape, 2);
    size_t m = shape->dim[shape->num_dims - 2];
    const size_t k = shape->dim[shape->num_dims - 1];
    if (flags & XNN_FLAG_SQUASH_GROUPS) {
      m *= num_groups;
      num_groups = 1;
    }
    return num_groups *
           xnn_x8_packq_f32qp8_gemm_packed_size(gemm_config, m, k);
  }

  uint64_t size_bits = xnn_datatype_size_bits(datatype);

  size_bits *= xnn_shape_multiply_all_dims(shape);

  // Round size up to the nearest byte.
  // TODO: We should not be using this helper for non-byte-addressable types,
  // perhaps we should just assert here.
  return (size_bits + 7) >> 3;
}

size_t xnn_runtime_tensor_get_size(const struct xnn_runtime_value* value)
{
  return get_tensor_size(value->gemm_config, value->type, value->datatype, &value->shape, value->flags);
}

size_t xnn_tensor_get_size(const struct xnn_value* value)
{
  return get_tensor_size(value->gemm_config, value->type, value->datatype, &value->shape, value->flags);
}

// Return size of the dynamic quantization params in this value
size_t xnn_tensor_get_dynamic_quant_param_size(enum xnn_datatype datatype,
                                               const struct xnn_shape *shape,
                                               size_t num_nonbatch_dims)
{
  switch (datatype) {
    case xnn_datatype_qdint8:
    case xnn_datatype_qduint8: {
      const size_t batch_dims_size = xnn_shape_multiply_batch_dims(
          shape, num_nonbatch_dims);
      return batch_dims_size * sizeof(struct xnn_quantization_params);
    }
    default:
      return 0;
  }
  return 0;
}

size_t xnn_tensor_get_size_by_id(xnn_subgraph_t subgraph, uint32_t value_id)
{
  assert(value_id < subgraph->num_values);

  const struct xnn_value* value = subgraph->values + value_id;
  return xnn_tensor_get_size(value);
}
