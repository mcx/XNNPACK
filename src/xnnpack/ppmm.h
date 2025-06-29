// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_PPMM_H_
#define XNNPACK_SRC_XNNPACK_PPMM_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(fn_name)                    \
  XNN_INTERNAL void fn_name(size_t mr, size_t nc, size_t kc, const float* a, \
                            const float* w, float* c, size_t cm_stride,      \
                            size_t cn_stride,                                \
                            const struct xnn_f32_minmax_params* params);

DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128_prfm)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_8x8__asm_aarch64_neonfma_ld128_prfm)

DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x8__aarch64_neonfma_prfm)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_4x8__neon)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x8__neon_prfm)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x16__aarch64_neonfma_prfm)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_4x16__neon)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x16__neon_prfm)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_8x8__aarch64_neonfma_prfm)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_8x8__neon)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_8x8__neon_prfm)

DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_ppmm_minmax_ukernel_4x8__sse)

DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_arm_splat)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x8__wasmsimd_x86_splat)

DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_2x4__scalar)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_3x3__scalar)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x2__scalar)
DECLARE_F32_PPMM_MINMAX_UKERNEL_FUNCTION(
    xnn_f32_ppmm_minmax_ukernel_4x4__scalar)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_PPMM_H_
