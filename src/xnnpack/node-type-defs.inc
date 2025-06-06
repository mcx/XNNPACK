// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_ENUM_ITEM_0
#define XNN_ENUM_ITEM_0(enum_name, enum_string) \
  XNN_ENUM_ITEM(enum_name, enum_string)
#define XNN_DEFINED_ENUM_ITEM_0
#endif

XNN_ENUM_ITEM_0(xnn_node_type_invalid, "Invalid")
XNN_ENUM_ITEM(xnn_node_type_argmax_pooling_2d, "ArgMax Pooling 2D")
XNN_ENUM_ITEM(xnn_node_type_average_pooling_2d, "Average Pooling 2D")
XNN_ENUM_ITEM(xnn_node_type_batch_matrix_multiply, "Batch Matrix Multiply")
XNN_ENUM_ITEM(xnn_node_type_binary_elementwise, "Binary Elementwise")
XNN_ENUM_ITEM(xnn_node_type_concatenate, "Concatenate")
XNN_ENUM_ITEM(xnn_node_type_convert, "Convert")
XNN_ENUM_ITEM(xnn_node_type_convolution_2d, "Convolution 2D")
XNN_ENUM_ITEM(xnn_node_type_copy, "Copy")
XNN_ENUM_ITEM(xnn_node_type_deconvolution_2d, "Deconvolution 2D")
XNN_ENUM_ITEM(xnn_node_type_depth_to_space_2d, "Depth To Space 2D")
XNN_ENUM_ITEM(xnn_node_type_depthwise_convolution_2d,
              "Depthwise Convolution 2D")
XNN_ENUM_ITEM(xnn_node_type_even_split, "Even Split")
XNN_ENUM_ITEM(xnn_node_type_fully_connected, "Fully Connected")
XNN_ENUM_ITEM(xnn_node_type_fully_connected_sparse, "Fully Connected Sparse")
XNN_ENUM_ITEM(xnn_node_type_global_average_pooling_1d,
              "Global Average Pooling 1D")
XNN_ENUM_ITEM(xnn_node_type_global_average_pooling_2d,
              "Global Average Pooling 2D")
XNN_ENUM_ITEM(xnn_node_type_global_sum_pooling_1d, "Global Sum Pooling 1D")
XNN_ENUM_ITEM(xnn_node_type_global_sum_pooling_2d, "Global Sum Pooling 2D")
XNN_ENUM_ITEM(xnn_node_type_max_pooling_2d, "Max Pooling 2D")
XNN_ENUM_ITEM(xnn_node_type_pack_lh, "Pack LH")
XNN_ENUM_ITEM(xnn_node_type_rope, "RoPE")
XNN_ENUM_ITEM(xnn_node_type_softmax, "Softmax")
XNN_ENUM_ITEM(xnn_node_type_space_to_depth_2d, "Space To Depth 2D")
XNN_ENUM_ITEM(xnn_node_type_static_constant_pad, "Static Constant Pad")
XNN_ENUM_ITEM(xnn_node_type_static_expand_dims, "Static Expand Dims")
XNN_ENUM_ITEM(xnn_node_type_fuse_dims, "Fuse Dims")
XNN_ENUM_ITEM(xnn_node_type_split_dims, "Split Dims")
XNN_ENUM_ITEM(xnn_node_type_static_broadcast, "Static Broadcast")
XNN_ENUM_ITEM(xnn_node_type_static_mean, "Static Mean")
XNN_ENUM_ITEM(xnn_node_type_static_reduce_max, "Static Reduce Max")
XNN_ENUM_ITEM(xnn_node_type_static_reduce_min, "Static Reduce Min")
XNN_ENUM_ITEM(xnn_node_type_static_reshape, "Static Reshape")
XNN_ENUM_ITEM(xnn_node_type_static_resize_bilinear_2d,
              "Static Resize Bilinear 2D")
XNN_ENUM_ITEM(xnn_node_type_static_slice, "Static Slice")
XNN_ENUM_ITEM(xnn_node_type_static_sum, "Static Sum")
XNN_ENUM_ITEM(xnn_node_type_static_transpose, "Static Transpose")
XNN_ENUM_ITEM(xnn_node_type_unary_elementwise, "Unary Elementwise")
XNN_ENUM_ITEM(xnn_node_type_unpooling_2d, "Unpooling 2D")

#ifdef XNN_DEFINED_ENUM_ITEM_0
#undef XNN_DEFINED_ENUM_ITEM_0
#undef XNN_ENUM_ITEM_0
#endif
