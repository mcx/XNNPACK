// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vclamp/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
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


void xnn_f16_vclamp_ukernel__neonfp16arith_u8(
    size_t batch,
    const xnn_float16* restrict input,
    xnn_float16* restrict output,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16((const uint16_t*) &params->scalar.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16((const uint16_t*) &params->scalar.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif

  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vmaxq_f16(vacc, vmin);
    vacc = vminq_f16(vacc, vmax);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    if (batch & (4 * sizeof(uint16_t))) {
      float16x4_t vacc = vreinterpret_f16_u16(vld1_u16(i)); i += 4;
      vacc = vmax_f16(vacc, vget_low_f16(vmin));
      vacc = vmin_f16(vacc, vget_low_f16(vmax));
      vst1_u16(o, vreinterpret_u16_f16(vacc)); o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      float16x4_t vacc = vreinterpret_f16_u32(vld1_dup_u32((const void*) i)); i += 2;
      vacc = vmax_f16(vacc, vget_low_f16(vmin));
      vacc = vmin_f16(vacc, vget_low_f16(vmax));
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc), 0); o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      float16x4_t vacc = vreinterpret_f16_u16(vld1_dup_u16(i));
      vacc = vmax_f16(vacc, vget_low_f16(vmin));
      vacc = vmin_f16(vacc, vget_low_f16(vmax));
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc), 0);
    }
  }
}
