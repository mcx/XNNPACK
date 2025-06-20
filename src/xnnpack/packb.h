// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_PACKB_H_
#define XNNPACK_SRC_XNNPACK_PACKB_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_subtile,     \
                    channel_round)                                          \
  XNN_INTERNAL void ukernel(size_t groups, size_t channels,                 \
                            const uint32_t* bias, uint32_t* packed_weights, \
                            size_t channel_tile_stride,                     \
                            size_t channel_subtile_stride,                  \
                            const struct xnn_x32_packb_params* params);

#include "src/x32-packb/x32-packb.inc"

#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_PACKB_H_
