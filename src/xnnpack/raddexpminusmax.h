// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_RADDEXPMINUSMAX_H_
#define XNNPACK_SRC_XNNPACK_RADDEXPMINUSMAX_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(fn_name)         \
  XNN_INTERNAL void fn_name(size_t n, const float* input, float* sum, \
                            float max);

DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u32)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u32_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u32_acc4)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u64)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u64_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u64_acc4)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u72)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u72_acc3)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc5)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u96)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u96_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u96_acc3)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u96_acc6)

DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u64)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u64_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u64_acc4)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc4)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u144)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u144_acc3)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u160)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u160_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u160_acc5)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u192)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u192_acc2)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u192_acc3)
DECLARE_F32_RADDEXPMINUSMAX_UKERNEL_FUNCTION(
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u192_acc6)

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // XNNPACK_SRC_XNNPACK_RADDEXPMINUSMAX_H_
