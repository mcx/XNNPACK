// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/neonfp16arith.c.in
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
#include "src/xnnpack/reduce.h"


void xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  float16x8_t vmax0 = vreinterpretq_f16_u16(vld1q_dup_u16((uint16_t*)((uintptr_t) o + 0 * sizeof(uint16_t))));
  float16x8_t vmax1 = vmax0;
  float16x8_t vmax2 = vmax0;
  float16x8_t vmax3 = vmax0;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vt0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vt3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vmax0 = vmaxq_f16(vmax0, vt0);
    vmax1 = vmaxq_f16(vmax1, vt1);
    vmax2 = vmaxq_f16(vmax2, vt2);
    vmax3 = vmaxq_f16(vmax3, vt3);
  }
  vmax0 = vmaxq_f16(vmax0, vmax1);
  vmax2 = vmaxq_f16(vmax2, vmax3);
  vmax0 = vmaxq_f16(vmax0, vmax2);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vt = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vmax0 = vmaxq_f16(vmax0, vt);
  }
  float16x4_t vmax_lo = vmax_f16(vget_low_f16(vmax0), vget_high_f16(vmax0));

  if (XNN_UNLIKELY(batch != 0)) {
    const float16x8_t vt = vreinterpretq_f16_u16(vld1q_u16(i));
    float16x4_t vt_lo = vget_low_f16(vt);
    if (batch & (4 * sizeof(uint16_t))) {
      vmax_lo = vmax_f16(vmax_lo, vt_lo);
      vt_lo = vget_high_f16(vt);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vmax_lo = vmax_f16(vmax_lo, vext_f16(vmax_lo, vt_lo, 2));
      vt_lo = vext_f16(vt_lo, vt_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vmax_lo = vmax_f16(vmax_lo, vext_f16(vmax_lo, vt_lo, 1));
    }
  }
  #if XNN_ARCH_ARM64 && defined(__GNUC__)
    *((__fp16*) o) = vmaxv_f16(vmax_lo);
  #else
    vmax_lo = vpmax_f16(vmax_lo, vmax_lo);
    vmax_lo = vpmax_f16(vmax_lo, vmax_lo);
    vst1_lane_u16(o, vreinterpret_u16_f16(vmax_lo), 0);
  #endif
}
