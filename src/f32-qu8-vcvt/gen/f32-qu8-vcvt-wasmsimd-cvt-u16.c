// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/wasmsimd-cvt.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/vcvt.h"


void xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16(
    size_t batch,
    const float* input,
    uint8_t* output,
    const struct xnn_f32_qu8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vscale = wasm_v128_load32_splat(&params->scalar.scale);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    v128_t vx0123 = wasm_v128_load(input);
    v128_t vx4567 = wasm_v128_load(input + 4);
    v128_t vx89AB = wasm_v128_load(input + 8);
    v128_t vxCDEF = wasm_v128_load(input + 12);
    input += 16;

    vx0123 = wasm_f32x4_mul(vx0123, vscale);
    vx4567 = wasm_f32x4_mul(vx4567, vscale);
    vx89AB = wasm_f32x4_mul(vx89AB, vscale);
    vxCDEF = wasm_f32x4_mul(vxCDEF, vscale);

    vx0123 = wasm_f32x4_nearest(vx0123);
    vx4567 = wasm_f32x4_nearest(vx4567);
    vx89AB = wasm_f32x4_nearest(vx89AB);
    vxCDEF = wasm_f32x4_nearest(vxCDEF);

    v128_t vacc0123 = wasm_i32x4_trunc_sat_f32x4(vx0123);
    v128_t vacc4567 = wasm_i32x4_trunc_sat_f32x4(vx4567);
    v128_t vacc89AB = wasm_i32x4_trunc_sat_f32x4(vx89AB);
    v128_t vaccCDEF = wasm_i32x4_trunc_sat_f32x4(vxCDEF);

    v128_t vacc01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
    v128_t vacc89ABCDEF = wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF);

    vacc01234567 = wasm_i16x8_add_sat(vacc01234567, voutput_zero_point);
    vacc89ABCDEF = wasm_i16x8_add_sat(vacc89ABCDEF, voutput_zero_point);

    v128_t vy0123456789ABCDEF = wasm_u8x16_narrow_i16x8(vacc01234567, vacc89ABCDEF);

    wasm_v128_store(output, vy0123456789ABCDEF);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    v128_t vx_lo = wasm_v128_load(input);
    v128_t vx_hi = wasm_v128_load(input + 4);
    input += 8;

    vx_lo = wasm_f32x4_mul(vx_lo, vscale);
    vx_hi = wasm_f32x4_mul(vx_hi, vscale);

    vx_lo = wasm_f32x4_nearest(vx_lo);
    vx_hi = wasm_f32x4_nearest(vx_hi);

    v128_t vacc_lo = wasm_i32x4_trunc_sat_f32x4(vx_lo);
    v128_t vacc_hi = wasm_i32x4_trunc_sat_f32x4(vx_hi);

    v128_t vacc = wasm_i16x8_narrow_i32x4(vacc_lo, vacc_hi);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);

    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    v128_t vx_lo = wasm_v128_load(input);
    const float* x_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    v128_t vx_hi = wasm_v128_load(x_hi);

    vx_lo = wasm_f32x4_mul(vx_lo, vscale);
    vx_hi = wasm_f32x4_mul(vx_hi, vscale);

    vx_lo = wasm_f32x4_nearest(vx_lo);
    vx_hi = wasm_f32x4_nearest(vx_hi);

    v128_t vacc_lo = wasm_i32x4_trunc_sat_f32x4(vx_lo);
    v128_t vacc_hi = wasm_i32x4_trunc_sat_f32x4(vx_hi);

    v128_t vacc = wasm_i16x8_narrow_i32x4(vacc_lo, vacc_hi);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);

    if (batch & (4 * sizeof(float))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}
