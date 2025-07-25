// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-qs8-vcvt/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
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
#include "src/xnnpack/vcvt.h"


void xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u64(
    size_t batch,
    const xnn_float16* input,
    int8_t* output,
    const struct xnn_f16_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;

  const float16x8_t vscale = vreinterpretq_f16_u16(vdupq_n_u16(*(const uint16_t*) &params->scalar.scale));
  const int16x8_t voutput_zero_point = vdupq_n_s16(params->scalar.output_zero_point);
  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx8 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx16 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx24 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx32 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx40 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx48 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx56 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vx0 = vmulq_f16(vx0, vscale);
    vx8 = vmulq_f16(vx8, vscale);
    vx16 = vmulq_f16(vx16, vscale);
    vx24 = vmulq_f16(vx24, vscale);
    vx32 = vmulq_f16(vx32, vscale);
    vx40 = vmulq_f16(vx40, vscale);
    vx48 = vmulq_f16(vx48, vscale);
    vx56 = vmulq_f16(vx56, vscale);

    int16x8_t vacc0 = vcvtnq_s16_f16(vx0);
    int16x8_t vacc8 = vcvtnq_s16_f16(vx8);
    int16x8_t vacc16 = vcvtnq_s16_f16(vx16);
    int16x8_t vacc24 = vcvtnq_s16_f16(vx24);
    int16x8_t vacc32 = vcvtnq_s16_f16(vx32);
    int16x8_t vacc40 = vcvtnq_s16_f16(vx40);
    int16x8_t vacc48 = vcvtnq_s16_f16(vx48);
    int16x8_t vacc56 = vcvtnq_s16_f16(vx56);

    vacc0 = vqaddq_s16(vacc0, voutput_zero_point);
    vacc8 = vqaddq_s16(vacc8, voutput_zero_point);
    vacc16 = vqaddq_s16(vacc16, voutput_zero_point);
    vacc24 = vqaddq_s16(vacc24, voutput_zero_point);
    vacc32 = vqaddq_s16(vacc32, voutput_zero_point);
    vacc40 = vqaddq_s16(vacc40, voutput_zero_point);
    vacc48 = vqaddq_s16(vacc48, voutput_zero_point);
    vacc56 = vqaddq_s16(vacc56, voutput_zero_point);

    int8x16_t vy0 = vcombine_s8(vqmovn_s16(vacc0), vqmovn_s16(vacc8));
    int8x16_t vy16 = vcombine_s8(vqmovn_s16(vacc16), vqmovn_s16(vacc24));
    int8x16_t vy32 = vcombine_s8(vqmovn_s16(vacc32), vqmovn_s16(vacc40));
    int8x16_t vy48 = vcombine_s8(vqmovn_s16(vacc48), vqmovn_s16(vacc56));

    vst1q_s8(output, vy0); output += 16;
    vst1q_s8(output, vy16); output += 16;
    vst1q_s8(output, vy32); output += 16;
    vst1q_s8(output, vy48); output += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vx = vmulq_f16(vx, vscale);

    int16x8_t vacc = vcvtnq_s16_f16(vx);

    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));

    vx = vmulq_f16(vx, vscale);

    int16x8_t vacc = vcvtnq_s16_f16(vx);
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);

    if (batch & (4 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_s8(output, vy, 0);
    }
  }
}
