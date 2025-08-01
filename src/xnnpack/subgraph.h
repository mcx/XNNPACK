// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_SUBGRAPH_H_
#define XNNPACK_SRC_XNNPACK_SUBGRAPH_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "include/experimental.h"
#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/node-type.h"
#include <pthreadpool.h>

#if defined(EMSCRIPTEN)
#include <emscripten/emscripten.h>
#elif XNN_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <time.h>
#endif

#define XNN_MAX_INPUTS 5
#define XNN_MAX_OUTPUTS 4

#define XNN_INVALID_NODE_ID UINT32_MAX

#define XNN_MAX_OPERATOR_OBJECTS 5

/// Disable fusion of nodes in subgraph. Fusion is enabled by default, set this
/// flag to turn it off.
#define XNN_FLAG_NO_OPERATOR_FUSION 0x80000000

/// Assume tensors of rank > 2 will be squashed to 2 dimensions.
#define XNN_FLAG_SQUASH_GROUPS 0x00000100
#define XNN_VALUE_FLAG_ONE_CONSUMER 0x00000200
#define XNN_VALUE_FLAG_FP16_COMPATIBLE 0x00000400
#define XNN_VALUE_FLAG_LAYOUT_NCHW 0x00000800

/// Create explicit `pack-lh` nodes, instead of pack the data on the fly
/// in a temporary buffer in the consuming op. Inline packing reduces memory
/// consumption and improves memory locality but may lead to slower GEMMs
/// because of tiling.
#define XNN_FLAG_NO_INLINED_LHS_PACKING 0x00004000

#ifdef __cplusplus
extern "C" {
#endif

#ifdef XNN_SLINKY_AVAILABLE
/// Slinky interface -- unused unless XNN_FLAG_SLINKY_ENABLED is set
struct slinky_pipeline;
typedef struct slinky_pipeline* slinky_pipeline_t;

enum xnn_status slinky_init_pipeline(xnn_runtime_t runtime);
void slinky_setup_pipeline(xnn_runtime_t runtime);
void slinky_destroy_pipeline(xnn_runtime_t runtime);
enum xnn_status slinky_reshape_pipeline(xnn_runtime_t runtime);
enum xnn_status slinky_invoke_pipeline(xnn_runtime_t runtime);
#endif  // XNN_SLINKY_AVAILABLE

struct xnn_shape {
  size_t num_dims;
  size_t dim[XNN_MAX_TENSOR_DIMS];
};

enum xnn_value_type {
  xnn_value_type_invalid = 0,
  xnn_value_type_dense_tensor = 1,
};

enum xnn_layout_type {
  xnn_layout_type_nhwc = 0,
  xnn_layout_type_nchw = 1,
};

struct xnn_value_quantization {
  /// Offset from zero of the quantized elements.
  int32_t zero_point;
  union {
    /// Multiplication factor to convert quantized elements to real
    /// representation.
    float scale;
    struct {
      /// Per-channel multiplication factor to convert quantized elements to
      /// real representation.
      const float* channelwise_scale;
      /// Index of the channel dimension with per-channel quantization
      /// parameters.
      size_t channel_dimension;
    };
    struct {
      /// Per-channel-block multiplication factor to convert quantized
      /// elements to real representation, bf16 format.
      union {
        const xnn_bfloat16* bf16_scale;
        const xnn_float16* fp16_scale;
      } blockwise_scale;
      enum xnn_datatype scale_type;
      /// Index of the channel dimension with blockwise quantization
      /// parameters.
      size_t channel_dimension_blockwise;
      /// Block size.
      size_t block_size;
    };
    struct {
      /// Number of non-batch dimensions. 1 for FC, 3 for Conv2D.
      size_t num_nonbatch_dims;
      /// Per-batch quantization parameters factor to convert quantized
      /// elements to real representation.
      struct xnn_quantization_params* dynamic_params;
      /// Number of (struct xnn_quantization_params) * sizeof(struct
      /// xnn_quantization_params)
      size_t dynamic_params_size;
    };
  };
};

/// Abstraction for a collections of elements produced and consumed by nodes.
struct xnn_value {
  /// Unique ID for the value.
  uint32_t id;
  /// Type of the collection of elements.
  ///
  /// Currently only dense tensors are supported.
  /// Other types (e.g. sparse tensors) might be supported in the future.
  enum xnn_value_type type;
  /// Type of elements in the collection.
  enum xnn_datatype datatype;
  /// Per-value quantization parameters.
  struct xnn_value_quantization quantization;
  /// Tensor shape.
  struct xnn_shape shape;
  /// Size of tensor.
  size_t size;
  /// Type of allocation for this tensors' data.
  enum xnn_allocation_type allocation_type;
  /// Binary features of the tensor. Supported values are any combination of:
  /// - XNN_VALUE_FLAG_EXTERNAL_INPUT
  /// - XNN_VALUE_FLAG_EXTERNAL_OUTPUT
  uint32_t flags;
  /// Static initialization data. Must be null for non-static values.
  void* data;
  /// Index of the Subgraph node that produced the value, or XNN_INVALID_NODE_ID
  /// is the Value is an external input.
  uint32_t producer;
  /// Index of the first Node that consume the value, or XNN_INVALID_NODE_ID if
  /// the Value has no consumers within the graph (e.g. Value is an external
  /// output).
  uint32_t first_consumer;
  bool all_consumers_types_same;
  /// Number of Nodes that consume the value.
  /// If multiple inputs in a Node refer to this Value as input, the Node is
  /// counted as consumer multiple times. If the Value is an external output, it
  /// counts as having an extra consumer.
  uint32_t num_consumers;
  uint32_t num_nchw_compatible_consumers;
  enum xnn_layout_type layout;
  /// Set during analysis in xnn_subgraph_rewrite_for_fp16.
  /// Indicates that this value should be converted to FP16.
  bool fp16_compatible;
  /// Set during analysis in xnn_subgraph_rewrite_for_fp16.
  /// Indicates Value ID of the FP16 variant of this Value.
  uint32_t fp16_id;
  /// Set during analysis in xnn_subgraph_rewrite_for_fp16.
  /// Indicates Value ID of the FP32 variant of this Value.
  uint32_t fp32_id;
  /// Used during analysis in xnn_subgraph_rewrite_for_fp16.
  /// Temporary buffer to convert static data to FP16.
  void* fp16_temp_data;
  // Pointer to a `xnn_gemm_config` if this value is packed for a specific GEMM.
  const struct xnn_gemm_config* gemm_config;
  // Pointer to original fp32 data if this value was converted from fp32 to fp16
  // (only for static values). This is used for nodes like Convolution, where
  // the filter is expected to be kept as fp32, but could have been converted to
  // fp16 if another node (like Subtraction) also consumed the weights. If NULL,
  // no conversion to fp16 was done, use field `data`. If not NULL, points to
  // the original fp32 data, (which should be `data` before it was overwritten
  // to point to converted fp16 buffer.
  const void* fp32_data;
};

/// Abstraction for a collections of elements produced and consumed by nodes.
struct xnn_runtime_value {
  /// Static initialization data. Must be null for non-static values.
  void* data;
  // Pointer to a `xnn_gemm_config` if this value is packed for a specific GEMM.
  const struct xnn_gemm_config* gemm_config;
  // Pointer to original fp32 data if this value was converted from fp32 to fp16
  // (only for static values). This is used for nodes like Convolution, where
  // the filter is expected to be kept as fp32, but could have been converted to
  // fp16 if another node (like Subtraction) also consumed the weights. If NULL,
  // no conversion to fp16 was done, use field `data`. If not NULL, points to
  // the original fp32 data, (which should be `data` before it was overwritten
  // to point to converted fp16 buffer.
  const void* fp32_data;
  /// Tensor shape. 56 bytes.
  struct xnn_shape shape;
  /// Size of tensor.
  size_t size;
  /// Per-value quantization parameters. 40 bytes.
  struct xnn_value_quantization quantization;
  /// Unique ID for the value.
  uint32_t id;
  /// Type of the collection of elements.
  ///
  /// Currently only dense tensors are supported.
  /// Other types (e.g. sparse tensors) might be supported in the future.
  enum xnn_value_type type;
  /// Type of elements in the collection.
  enum xnn_datatype datatype;
  /// Type of allocation for this tensors' data.
  enum xnn_allocation_type allocation_type;
  /// Binary features of the tensor. Supported values are any combination of:
  /// - XNN_FLAG_SQUASH_GROUPS
  /// - XNN_VALUE_FLAG_NEEDS_CLEANUP
  /// - XNN_VALUE_FLAG_EXTERNAL_INPUT
  /// - XNN_VALUE_FLAG_EXTERNAL_OUTPUT
  /// - XNN_VALUE_FLAG_PERSISTENT
  /// - XNN_VALUE_FLAG_ONE_CONSUMER
  /// - XNN_VALUE_FLAG_LAYOUT_NCHW
  uint32_t flags;
  /// Index of the first Node that consume the value, or XNN_INVALID_NODE_ID if
  /// the Value has no consumers within the graph (e.g. Value is an external
  /// output).
  uint32_t first_consumer;
};

XNN_INLINE static bool xnn_value_is_external(uint32_t flags) {
  return (flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT |
                   XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0;
}

XNN_INLINE static bool xnn_value_is_external_output(uint32_t flags) {
  return (flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) != 0;
}

XNN_INLINE static bool xnn_value_is_external_input(uint32_t flags) {
  return (flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) != 0;
}

XNN_INLINE static bool xnn_value_is_internal(const struct xnn_value* value) {
  return (value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT |
                          XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0;
}

XNN_INLINE static bool xnn_value_is_valid(enum xnn_value_type value_type) {
  return value_type != xnn_value_type_invalid;
}

XNN_INLINE static bool xnn_value_is_static(
    enum xnn_allocation_type allocation_type) {
  return allocation_type == xnn_allocation_type_static;
}

struct xnn_node;
struct xnn_operator_data;

typedef enum xnn_status (*xnn_create_operator_fn)(
    const struct xnn_node* node, const struct xnn_runtime_value* values,
    size_t num_values, struct xnn_operator_data* opdata,
    xnn_weights_cache_t weights_cache);

typedef enum xnn_status (*xnn_reshape_operator_fn)(
    struct xnn_operator_data* opdata, struct xnn_runtime_value* values,
    size_t num_values, pthreadpool_t threadpool);

typedef enum xnn_status (*xnn_setup_operator_fn)(
    const struct xnn_operator_data* opdata,
    const struct xnn_runtime_value* values, size_t num_values,
    pthreadpool_t threadpool);

struct xnn_node {
  enum xnn_node_type type;
  union {
    enum xnn_binary_operator binary_operator;
    enum xnn_unary_operator unary_operator;
    // Used by the `fully-connected` and `batch-matrix-multiply` ops.
    enum xnn_datatype packed_input_datatype;
  };
  uint32_t id;
  /// Static parameters of the operator node.
  union {
    struct {
      int32_t axis;
    } concatenate;
    struct {
      uint32_t input_padding_top;
      uint32_t input_padding_right;
      uint32_t input_padding_bottom;
      uint32_t input_padding_left;
      uint32_t kernel_height;
      uint32_t kernel_width;
      uint32_t subsampling_height;
      uint32_t subsampling_width;
      uint32_t dilation_height;
      uint32_t dilation_width;
      uint32_t groups;
      size_t group_input_channels;
      size_t group_output_channels;
    } convolution_2d;
    struct {
      uint32_t padding_top;
      uint32_t padding_right;
      uint32_t padding_bottom;
      uint32_t padding_left;
      uint32_t adjustment_height;
      uint32_t adjustment_width;
      uint32_t kernel_height;
      uint32_t kernel_width;
      uint32_t upsampling_height;
      uint32_t upsampling_width;
      uint32_t dilation_height;
      uint32_t dilation_width;
      uint32_t groups;
      size_t group_input_channels;
      size_t group_output_channels;
    } deconvolution_2d;
    struct {
      uint32_t input_padding_top;
      uint32_t input_padding_right;
      uint32_t input_padding_bottom;
      uint32_t input_padding_left;
      uint32_t kernel_height;
      uint32_t kernel_width;
      uint32_t subsampling_height;
      uint32_t subsampling_width;
      uint32_t dilation_height;
      uint32_t dilation_width;
      uint32_t depth_multiplier;
      size_t input_channels;
    } depthwise_convolution_2d;
    struct {
      uint32_t block_size;
    } depth_to_space_2d;
    struct {
      int32_t axis;
    } even_split;
    struct {
      uint32_t padding_top;
      uint32_t padding_right;
      uint32_t padding_bottom;
      uint32_t padding_left;
      uint32_t pooling_height;
      uint32_t pooling_width;
      uint32_t stride_height;
      uint32_t stride_width;
      uint32_t dilation_height;
      uint32_t dilation_width;
    } pooling_2d;
    struct {
      size_t pre_paddings[XNN_MAX_TENSOR_DIMS];
      size_t post_paddings[XNN_MAX_TENSOR_DIMS];
      uint32_t padding_value;
    } static_pad;
    struct {
      struct xnn_shape new_shape;
      size_t axis;
    } static_reshape;
    struct {
      size_t new_height;
      size_t new_width;
    } static_resize;
    struct {
      size_t num_dims;
      int64_t begins[XNN_MAX_TENSOR_DIMS];
      int64_t ends[XNN_MAX_TENSOR_DIMS];
    } slice;
    struct {
      uint32_t block_size;
    } space_to_depth_2d;
    struct {
      size_t num_reduction_axes;
      int64_t reduction_axes[XNN_MAX_TENSOR_DIMS];
    } reduce;
    struct {
      size_t perm[XNN_MAX_TENSOR_DIMS];
      size_t num_dims;
    } transpose;
    union xnn_unary_params unary;
  } params;
  struct {
    float output_min;
    float output_max;
  } activation;
  /// Value IDs for node inputs.
  uint32_t inputs[XNN_MAX_INPUTS];
  uint32_t num_inputs;
  /// Value IDs for node outputs.
  uint32_t outputs[XNN_MAX_OUTPUTS];
  uint32_t num_outputs;
  uint32_t flags;
  uint32_t layout_flags;
  uint32_t cluster_leader;
  // Number of filter parameters in all 1x1 Convolutions of the sparse cluster.
  // This value is properly initialized only in sparse inference analysis of 1x1
  // Convolutions.
  size_t num_params;
  // Number of zero filter parameters in all 1x1 Convolutions of the sparse
  // cluster. This value is properly initialized only in sparse inference
  // analysis of 1x1 Convolutions.
  size_t num_zeroes;
  // Factory function to create an operator object from the node.
  xnn_create_operator_fn create;
  // Function to reshape an operator using opdata.
  xnn_reshape_operator_fn reshape;
  // Function to setup an operator using opdata.
  xnn_setup_operator_fn setup;
};

#ifdef __MACH__
typedef uint64_t xnn_timestamp;
#elif __EMSCRIPTEN__
typedef double xnn_timestamp;
#elif XNN_PLATFORM_WINDOWS
typedef LARGE_INTEGER xnn_timestamp;
#else
typedef struct timespec xnn_timestamp;
#endif

struct xnn_operator_data {
  enum xnn_node_type type;
  uint32_t id;
  xnn_operator_t operator_objects[XNN_MAX_OPERATOR_OBJECTS];
  xnn_reshape_operator_fn reshape;
  xnn_setup_operator_fn setup;
  union {
    // Used for reduction.
    struct {
      size_t num_reduction_axes;
      int64_t reduction_axes[XNN_MAX_TENSOR_DIMS];
    };
    // Used for reshape.
    struct {
      size_t num_reshape_dims;
      size_t reshape_dims[XNN_MAX_TENSOR_DIMS];
      // Used for concatenate, copy and static split dims.
      int32_t axis;
    };
    // Used for static constant pad.
    struct {
      size_t post_paddings[XNN_MAX_TENSOR_DIMS];
      size_t pre_paddings[XNN_MAX_TENSOR_DIMS];
    };
    // Used for static slice.
    struct {
      int64_t begins[XNN_MAX_TENSOR_DIMS];
      int64_t ends[XNN_MAX_TENSOR_DIMS];
    };
    struct {
      uint32_t adjustment_height;
      uint32_t adjustment_width;
    };
    struct xnn_shape shape1;
  };
  uint32_t num_inputs;
  uint32_t inputs[XNN_MAX_INPUTS];
  uint32_t num_outputs;
  uint32_t outputs[XNN_MAX_OUTPUTS];
  xnn_timestamp* end_ts;
  void* workspace;
  size_t workspace_size;
  uint32_t flags;
};

struct xnn_subgraph {
  /// Number of Value IDs reserved for communication with external graph
  /// representation. Values created during subgraph transformation avoid using
  /// IDs in [0, reserved_value_ids-1] range.
  uint32_t external_value_ids;

  uint32_t num_reserved_values;
  uint32_t num_values;
  struct xnn_value* values;

  uint32_t num_reserved_nodes;
  uint32_t num_nodes;
  struct xnn_node* nodes;
};

/// Runtime is a combination of an execution plan for subgraph Nodes and a
/// memory manager for subgraph Values.
struct xnn_runtime {
  uint32_t num_external_values;
  uint32_t flags;

  /// List of operators in the execution plan, in execution order.
  struct xnn_operator_data* opdata;
  /// Number of operators in the execution plan.
  size_t num_ops;

  struct xnn_runtime_value* values;
  size_t num_values;

  struct xnn_workspace* workspace;
  struct xnn_runtime* next_workspace_user;

  pthreadpool_t threadpool;

  bool profiling;
  // The start timestamp of the first operator in the subgraph. This is set when
  // profiling is true.
  xnn_timestamp start_ts;

  // True if runtime has ever been setup. If it has been setup, the pointers
  // inside of opdata need to be updated if workspace changes.
  bool has_been_setup;
  bool memory_planned;

#ifdef XNN_SLINKY_AVAILABLE
  // Fields used by Slinky -- unused unless XNN_FLAG_SLINKY_ENABLED is set
  slinky_pipeline_t slinky_pipeline;
  xnn_scheduler_t scheduler;
#endif  // XNN_SLINKY_AVAILABLE
};

enum xnn_status xnn_insert_clamp_node(xnn_subgraph_t subgraph, float output_min,
                                      float output_max, struct xnn_node* node);

enum xnn_status xnn_insert_pack_lh_node(xnn_subgraph_t subgraph,
                                        uint32_t input_id, uint32_t* new_id);

struct xnn_value* xnn_subgraph_new_internal_value(xnn_subgraph_t subgraph);

struct xnn_node* xnn_subgraph_new_node(xnn_subgraph_t subgraph);

enum xnn_status xnn_subgraph_add_nodes(xnn_subgraph_t subgraph,
                                       size_t num_nodes);

uint32_t xnn_subgraph_get_value_flags(xnn_subgraph_t subgraph,
                                      uint32_t value_id);
size_t xnn_subgraph_get_value_size(xnn_subgraph_t subgraph, uint32_t value_id);
uint32_t xnn_subgraph_get_num_external_values(xnn_subgraph_t subgraph);

// Get size of the tensor in bytes (based on dimensions of tensor).
size_t xnn_tensor_get_size(const struct xnn_value* value);
size_t xnn_runtime_tensor_get_size(const struct xnn_runtime_value* value);

size_t xnn_tensor_get_size_by_id(xnn_subgraph_t subgraph, uint32_t value_id);

XNN_INLINE static size_t xnn_get_rounded_size(size_t size) {
  // We round it to XNN_EXTRA_BYTES to ensure that we can read more than the
  // actual size of the tensor, and round it to allocation alignment to ensure
  // that all tensors and operator workspaces are aligned correctly.
  return round_up_po2(round_up_po2(size, XNN_EXTRA_BYTES),
                      XNN_ALLOCATION_ALIGNMENT);
}

// Returns the size of tensor rounded to appropriate extra bytes and allocation
// alignment.
XNN_INLINE static size_t xnn_tensor_get_rounded_size(
    const struct xnn_runtime_value* value) {
  return xnn_get_rounded_size(value->size);
}

// Product of all shape dimensions
size_t xnn_shape_multiply_all_dims(const struct xnn_shape* shape);

// Product of all shape dimensions, except for the specified number of the last
// dimensions
size_t xnn_shape_multiply_batch_dims(const struct xnn_shape* shape,
                                     size_t num_nonbatch_dims);

// Product of all shape dimensions, except for the last (channel) one
size_t xnn_shape_multiply_non_channel_dims(const struct xnn_shape* shape);

// Product of n leading dimensions.
size_t xnn_shape_multiply_leading_dims(const struct xnn_shape* shape,
                                       size_t num_leading_dims);

// Product of trailing dimensions starting from start_dim.
size_t xnn_shape_multiply_trailing_dims(const struct xnn_shape* shape,
                                        size_t start_dim);

// Get the size in bytes to hold dynamic quant params
size_t xnn_tensor_get_dynamic_quant_param_size(enum xnn_datatype datatype,
                                               const struct xnn_shape* shape,
                                               size_t num_nonbatch_dims);

XNN_INLINE static size_t xnn_tensor_get_rounded_dynamic_quant_param_size(
    const struct xnn_runtime_value* value) {
  assert(value->datatype == xnn_datatype_qdint8 ||
         value->datatype == xnn_datatype_qduint8);

  // We may read out of bounds for qparams.
  return xnn_get_rounded_size(value->quantization.dynamic_params_size +
                              XNN_EXTRA_QUANTIZATION_PARAMS *
                                  sizeof(struct xnn_quantization_params));
}

// Rewrites the subgraph such that values have exactly one producer.
void xnn_subgraph_rewrite_ssa(xnn_subgraph_t subgraph);

enum xnn_status xnn_subgraph_optimize(xnn_subgraph_t subgraph, uint32_t flags);

void xnn_subgraph_rewrite_for_nchw(xnn_subgraph_t subgraph);
// Rewrites subgraph for FP16, returns true if success, false if rewrite failed.
bool xnn_subgraph_rewrite_for_fp16(xnn_subgraph_t subgraph);

void xnn_node_clear(struct xnn_node* node);
void xnn_node_copy(struct xnn_node* dst_node, const struct xnn_node* src_node);
void xnn_value_clear(struct xnn_value* value);

void xnn_value_copy(struct xnn_value* dst_value,
                    const struct xnn_value* src_value);

void xnn_runtime_value_copy(struct xnn_runtime_value* dst_value,
                            const struct xnn_value* src_value);

void xnn_init_convert_node(struct xnn_node* node, uint32_t input_id,
                           uint32_t output_id, uint32_t flags);

struct xnn_workspace {
  void* data;
  size_t size;
  struct xnn_runtime* first_user;
  // Workspace will be destroyed in xnn_delete_runtime or xnn_delete_workspace
  // if num_users reaches 0.
  size_t ref_count;
};

void xnn_subgraph_analyze_consumers_and_producers(xnn_subgraph_t subgraph);

enum xnn_status resize_fully_connected_output_tensor(
    const struct xnn_operator_data* opdata, struct xnn_runtime_value* values,
    size_t num_values, size_t old_workspace_size, pthreadpool_t threadpool);

XNN_INTERNAL enum xnn_node_type xnn_reduce_operator_to_node_type(
    enum xnn_reduce_operator type);
XNN_INTERNAL enum xnn_reduce_operator xnn_node_type_to_reduce_operator(
    enum xnn_node_type type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_SUBGRAPH_H_
