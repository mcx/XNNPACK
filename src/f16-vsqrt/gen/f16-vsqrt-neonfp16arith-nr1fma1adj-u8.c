// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsqrt/neonfp16arith-nr1fma1adj.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16x8_t vpositive_infinity = vmovq_n_u16(UINT16_C(0x7C00));
  const float16x8_t vhalf = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3800)));  // 0.5h
  const uint16x8_t vexp4_mask = vmovq_n_u16(UINT16_C(0x7800));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx = vbslq_f16(vexp4_mask, vhalf, vi);
    const int16x8_t vexp4i = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi), vexp4_mask));

    const float16x8_t vrsqrtx = vrsqrteq_f16(vx);
    const int16x8_t vpostscale = vhsubq_s16(vexp4i, vreinterpretq_s16_f16(vhalf));

    float16x8_t vsqrtx = vmulq_f16(vrsqrtx, vx);
    const float16x8_t vhalfrsqrtx = vmulq_f16(vrsqrtx, vhalf);
    uint16x8_t vspecial_mask = vcgeq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vresidual = vfmsq_f16(vhalf, vsqrtx, vhalfrsqrtx);
    const uint16x8_t vzero_mask = vceqq_f16(vi, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value = vmovq_n_u16(UINT16_C(0x7E00));

    vsqrtx = vfmaq_f16(vsqrtx, vresidual, vsqrtx);
    vspecial_mask = vorrq_u16(vspecial_mask, vzero_mask);
    const uint16x8_t vinfinity_mask = vceqq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vadjustment = vfmsq_f16(vx, vsqrtx, vsqrtx);
    const uint16x8_t vinput_mask = vorrq_u16(vinfinity_mask, vzero_mask);

    vsqrtx = vfmaq_f16(vsqrtx, vhalfrsqrtx, vadjustment);
    vspecial_value = vbslq_u16(vinput_mask, vreinterpretq_u16_f16(vi), vspecial_value);

    float16x8_t vy = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx), vpostscale));

    vy = vbslq_f16(vspecial_mask, vreinterpretq_f16_u16(vspecial_value), vy);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i));

    const float16x8_t vx = vbslq_f16(vexp4_mask, vhalf, vi);
    const int16x8_t vexp4i = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi), vexp4_mask));

    const float16x8_t vrsqrtx = vrsqrteq_f16(vx);
    const int16x8_t vpostscale = vhsubq_s16(vexp4i, vreinterpretq_s16_f16(vhalf));

    float16x8_t vsqrtx = vmulq_f16(vrsqrtx, vx);
    const float16x8_t vhalfrsqrtx = vmulq_f16(vrsqrtx, vhalf);
    uint16x8_t vspecial_mask = vcgeq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vresidual = vfmsq_f16(vhalf, vsqrtx, vhalfrsqrtx);
    const uint16x8_t vzero_mask = vceqq_f16(vi, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value = vmovq_n_u16(UINT16_C(0x7E00));

    vsqrtx = vfmaq_f16(vsqrtx, vresidual, vsqrtx);
    vspecial_mask = vorrq_u16(vspecial_mask, vzero_mask);
    const uint16x8_t vinfinity_mask = vceqq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vadjustment = vfmsq_f16(vx, vsqrtx, vsqrtx);
    const uint16x8_t vinput_mask = vorrq_u16(vinfinity_mask, vzero_mask);

    vsqrtx = vfmaq_f16(vsqrtx, vhalfrsqrtx, vadjustment);
    vspecial_value = vbslq_u16(vinput_mask, vreinterpretq_u16_f16(vi), vspecial_value);

    float16x8_t vy = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx), vpostscale));

    vy = vbslq_f16(vspecial_mask, vreinterpretq_f16_u16(vspecial_value), vy);

    float16x4_t vy_lo = vget_low_f16(vy);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;
      vy_lo = vget_high_f16(vy);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o += 2;
      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}
