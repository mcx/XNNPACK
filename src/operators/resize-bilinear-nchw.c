// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

static const struct xnn_ibilinear_chw_config* get_ibilinear_nchw_config(enum xnn_datatype datatype)
{
  switch (datatype) {
    case xnn_datatype_fp16:
      return xnn_init_f16_ibilinear_chw_config();
    case xnn_datatype_fp32:
      return xnn_init_f32_ibilinear_chw_config();
    default:
      return NULL;
  }
}

enum xnn_status xnn_create_resize_bilinear2d_nchw(
    enum xnn_datatype datatype,
    size_t output_height,
    size_t output_width,
    uint32_t flags,
    xnn_operator_t* resize_op_out)
{
  const struct xnn_ibilinear_chw_config* ibilinear_chw_config =
      get_ibilinear_nchw_config(datatype);
  if (ibilinear_chw_config == NULL || ibilinear_chw_config->ukernel == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw));
    return xnn_status_unsupported_hardware;
  }

  xnn_operator_t resize_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (output_width == 0 || output_height == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu output: output dimensions must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw), output_width, output_height);
    goto error;
  }

  if (max(output_width, output_height) >= (1L << 24)) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu output: output dimensions must be below 2**24",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw), output_width, output_height);
    goto error;
  }

  status = xnn_status_out_of_memory;

  resize_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (resize_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw));
    goto error;
  }
  resize_op->compute = xnn_allocate_zero_memory(sizeof(struct compute_parameters));
  if (resize_op->compute == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct compute_parameters),
                  xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw));
    goto error;
  }
  resize_op->num_compute_invocations = 1;
  resize_op->convolution_op = xnn_allocate_zero_memory(sizeof(struct xnn_convolution_operator));
  if (resize_op->convolution_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct xnn_convolution_operator),
                  xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw));
    return xnn_status_out_of_memory;
  }

  resize_op->convolution_op->output_height = output_height;
  resize_op->convolution_op->output_width = output_width;

  resize_op->type = xnn_operator_type_resize_bilinear_nchw;
  resize_op->flags = flags;
  resize_op->ibilinear_chw_config = ibilinear_chw_config;

  resize_op->state = xnn_run_state_invalid;

  *resize_op_out = resize_op;
  return xnn_status_success;

error:
  xnn_delete_operator(resize_op);
  return status;
}

enum xnn_status xnn_reshape_resize_bilinear2d_nchw(
    xnn_operator_t resize_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    pthreadpool_t threadpool)
{
  if (resize_op->type != xnn_operator_type_resize_bilinear_nchw) {
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw),
        xnn_operator_type_to_string_v2(resize_op));
    return xnn_status_invalid_parameter;
  }
  resize_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw));
    return xnn_status_uninitialized;
  }

  if (input_width <= 1 || input_height <= 1) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be greater than 1",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (max(input_width, input_height) >= (1L << 24)) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be below 2**24",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw), input_width, input_height);
    return xnn_status_unsupported_parameter;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw), input_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_pixel_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw), output_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    resize_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t log2_data_element_size = resize_op->ibilinear_chw_config->log2_data_element_size;
  const size_t log2_weight_element_size = resize_op->ibilinear_chw_config->log2_weight_element_size;
  const size_t output_height = resize_op->convolution_op->output_height;
  const size_t output_width = resize_op->convolution_op->output_width;
  if (output_height * output_width != resize_op->convolution_op->last_output_height * resize_op->convolution_op->last_output_width) {
    const size_t indirection_buffer_size = sizeof(void*) * (output_height * output_width * 4);
    const size_t packed_weights_size = (output_height * output_width * 2) << log2_weight_element_size;

    const void** indirection_buffer = (const void**) xnn_reallocate_memory(resize_op->convolution_op->indirection_buffer, indirection_buffer_size);
    if (indirection_buffer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator indirection buffer",
        indirection_buffer_size, xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw));
      return xnn_status_out_of_memory;
    }
    resize_op->convolution_op->indirection_buffer = indirection_buffer;
    xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
      indirection_buffer_size, xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw));

    // Note: packed weights must be SIMD-aligned, so we can't use xnn_reallocate_memory
    xnn_release_simd_memory(resize_op->packed_weights.pointer);
    resize_op->packed_weights.pointer = xnn_allocate_simd_memory(packed_weights_size);
    if (resize_op->packed_weights.pointer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator packed weights",
        packed_weights_size, xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw));
      return xnn_status_out_of_memory;
    }
  }

  const size_t input_pixel_stride_in_bytes = 1 << log2_data_element_size; // Since the layout in CHW the pixels
  if (input_height != resize_op->convolution_op->last_input_height ||
      input_width != resize_op->convolution_op->last_input_width ||
      output_height != resize_op->convolution_op->last_output_height ||
      output_width != resize_op->convolution_op->last_output_width)
  {
    const uint32_t flags = resize_op->flags;
    // Set a dummy input first, the actual input offset is calculated in setup when we have the input pointer.
    void* dummy_input = (void*) XNN_ALLOCATION_ALIGNMENT;
    resize_op->ibilinear_chw_config->indirection_init(
        input_pixel_stride_in_bytes,
        input_height, input_width,
        output_height, output_width,
        dummy_input, resize_op->convolution_op->indirection_buffer, resize_op->packed_weights.pointer,
        !!(flags & XNN_FLAG_ALIGN_CORNERS),
        !!(flags & XNN_FLAG_TENSORFLOW_LEGACY_MODE));

    resize_op->convolution_op->last_input = dummy_input;
    resize_op->convolution_op->last_input_height = input_height;
    resize_op->convolution_op->last_input_width = input_width;
    resize_op->convolution_op->last_output_height = output_height;
    resize_op->convolution_op->last_output_width = output_width;
  }

  const struct xnn_ibilinear_chw_config* ibilinear_chw = resize_op->ibilinear_chw_config;
  // Resize bilinear packed weights can change when the operator is resized, we will not use weights cache.
  assert(resize_op->weights_cache == NULL);
  resize_op->context.resize_bilinear_chw = (struct resize_bilinear_chw_context) {
    .output_pixels = output_height * output_width,
    .channels = resize_op->channels,
    .input_channel_stride = (input_height * input_width) << log2_data_element_size,
    .indirect_input = resize_op->convolution_op->indirection_buffer,
    .input_batch_stride = (input_pixel_stride * input_height * input_width) << log2_data_element_size,
    .packed_weights = resize_op->packed_weights.pointer,
    .output_batch_stride = (output_pixel_stride * output_height * output_width) << log2_data_element_size,
    .output_channel_stride = (output_height * output_width) << log2_data_element_size,
    .ukernel = ibilinear_chw->ukernel,
  };

  size_t output_channel_tile = channels;
  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  if (num_threads > 1) {
    const size_t target_tiles_per_thread = 4;
    const size_t max_channel_tile = divide_round_up(output_channel_tile, num_threads * target_tiles_per_thread);
    if (max_channel_tile < output_channel_tile) {
      const uint32_t output_channel_subtile = ibilinear_chw->channel_tile;
      output_channel_tile =
        min(output_channel_tile,
          divide_round_up(output_channel_tile, max_channel_tile * output_channel_subtile) * output_channel_subtile);
    }
  }

  resize_op->compute[0].type = xnn_parallelization_type_2d_tile_1d;
  resize_op->compute[0].task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_resize_bilinear_chw;
  resize_op->compute[0].range[0] = batch_size;
  resize_op->compute[0].range[1] = channels;
  resize_op->compute[0].tile[0] = output_channel_tile;
  resize_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_setup_resize_bilinear2d_nchw(
    xnn_operator_t resize_op,
    const void* input,
    void* output)
{
  if (resize_op->type != xnn_operator_type_resize_bilinear_nchw) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(xnn_operator_type_resize_bilinear_nchw),
        xnn_operator_type_to_string_v2(resize_op));
    return xnn_status_invalid_parameter;
  }

  switch (resize_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(resize_op));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  resize_op->context.resize_bilinear_chw.input_offset =
    (size_t) ((uintptr_t) input - (uintptr_t) resize_op->convolution_op->last_input);
  resize_op->context.resize_bilinear_chw.output = output;

  resize_op->state = xnn_run_state_ready;

  return xnn_status_success;
}
