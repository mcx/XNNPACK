// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vcopysign/copysignc.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-neon.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vcopysignc_ukernel__neon_u4(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vsign = xnn_set1_f32(*sign);
  vsign = xnn_and_f32(vsign, vsign_mask);

  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysignc_ukernel__neon_u8(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vsign = xnn_set1_f32(*sign);
  vsign = xnn_and_f32(vsign, vsign_mask);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag + 0 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    mag += 8;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign, vmag_1);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 8;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysignc_ukernel__neon_u12(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vsign = xnn_set1_f32(*sign);
  vsign = xnn_and_f32(vsign, vsign_mask);

  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag + 0 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_2 = xnn_abs_f32(xnn_loadu_f32(mag + 2 * xnn_simd_size_f32));
    mag += 12;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign, vmag_1);
    xnn_simd_f32_t vy_2 = xnn_or_f32(vsign, vmag_2);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    output += 12;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vcopysignc_ukernel__neon_u16(
    size_t batch,
    const float* mag,
    const float* sign,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 4);

  XNN_SIMD_CONST_F32(vsign_mask, -0.f);
  xnn_simd_f32_t vsign = xnn_set1_f32(*sign);
  vsign = xnn_and_f32(vsign, vsign_mask);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {

    xnn_simd_f32_t vmag_0 = xnn_abs_f32(xnn_loadu_f32(mag + 0 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_1 = xnn_abs_f32(xnn_loadu_f32(mag + 1 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_2 = xnn_abs_f32(xnn_loadu_f32(mag + 2 * xnn_simd_size_f32));
    xnn_simd_f32_t vmag_3 = xnn_abs_f32(xnn_loadu_f32(mag + 3 * xnn_simd_size_f32));
    mag += 16;

    xnn_simd_f32_t vy_0 = xnn_or_f32(vsign, vmag_0);
    xnn_simd_f32_t vy_1 = xnn_or_f32(vsign, vmag_1);
    xnn_simd_f32_t vy_2 = xnn_or_f32(vsign, vmag_2);
    xnn_simd_f32_t vy_3 = xnn_or_f32(vsign, vmag_3);

    xnn_storeu_f32(output + 0 * xnn_simd_size_f32, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    xnn_storeu_f32(output + 2 * xnn_simd_size_f32, vy_2);
    xnn_storeu_f32(output + 3 * xnn_simd_size_f32, vy_3);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_loadu_f32(mag));
    mag += xnn_simd_size_f32;

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vmag = xnn_abs_f32(xnn_load_tail_f32(mag, batch >> XNN_LOG2_SIZEOF_FLOAT));

    xnn_simd_f32_t vy = xnn_or_f32(vsign, vmag);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
