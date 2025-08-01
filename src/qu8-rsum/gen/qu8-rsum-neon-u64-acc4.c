// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"


void xnn_qu8_rsum_ukernel__neon_u64_acc4(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  XNN_ALIGN(16) static const uint8_t onemask_table[32] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  uint32x4_t vacc0 = vmovq_n_u32(0);
  uint32x4_t vacc1 = vmovq_n_u32(0);
  uint32x4_t vacc2 = vmovq_n_u32(0);
  uint32x4_t vacc3 = vmovq_n_u32(0);

  // 256 uint8s may be summed into an uint16 before overflowing.
  // Each register has 8 lanes and there are 4 accumulators so batch size is 8192

  for (; batch >= 8192; batch -= 8192) {
    uint16x8_t vacc16_0 = vmovq_n_u16(0);
    uint16x8_t vacc16_1 = vmovq_n_u16(0);
    uint16x8_t vacc16_2 = vmovq_n_u16(0);
    uint16x8_t vacc16_3 = vmovq_n_u16(0);
    for (size_t current_batch = 8192; current_batch > 0; current_batch -= 64) {
      const uint8x16_t vt0 = vld1q_u8(input); input += 16;
      const uint8x16_t vt1 = vld1q_u8(input); input += 16;
      const uint8x16_t vt2 = vld1q_u8(input); input += 16;
      const uint8x16_t vt3 = vld1q_u8(input); input += 16;
      vacc16_0 = vpadalq_u8(vacc16_0, vt0);
      vacc16_1 = vpadalq_u8(vacc16_1, vt1);
      vacc16_2 = vpadalq_u8(vacc16_2, vt2);
      vacc16_3 = vpadalq_u8(vacc16_3, vt3);
    }
    vacc0 = vpadalq_u16(vacc0, vacc16_0);
    vacc1 = vpadalq_u16(vacc1, vacc16_1);
    vacc2 = vpadalq_u16(vacc2, vacc16_2);
    vacc3 = vpadalq_u16(vacc3, vacc16_3);
  }

  if (XNN_LIKELY(batch >= 64)) {
    assert(batch >= 1 && batch < 8192);
    uint16x8_t vacc16_0 = vmovq_n_u16(0);
    uint16x8_t vacc16_1 = vmovq_n_u16(0);
    uint16x8_t vacc16_2 = vmovq_n_u16(0);
    uint16x8_t vacc16_3 = vmovq_n_u16(0);
    for (; batch >= 64; batch -= 64) {
      const uint8x16_t vt0 = vld1q_u8(input); input += 16;
      const uint8x16_t vt1 = vld1q_u8(input); input += 16;
      const uint8x16_t vt2 = vld1q_u8(input); input += 16;
      const uint8x16_t vt3 = vld1q_u8(input); input += 16;
      vacc16_0 = vpadalq_u8(vacc16_0, vt0);
      vacc16_1 = vpadalq_u8(vacc16_1, vt1);
      vacc16_2 = vpadalq_u8(vacc16_2, vt2);
      vacc16_3 = vpadalq_u8(vacc16_3, vt3);
    }
    vacc0 = vpadalq_u16(vacc0, vacc16_0);
    vacc1 = vpadalq_u16(vacc1, vacc16_1);
    vacc2 = vpadalq_u16(vacc2, vacc16_2);
    vacc3 = vpadalq_u16(vacc3, vacc16_3);
  }
  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 2048);
    uint16x8_t vacc16 = vmovq_n_u16(0);
    for (; batch >= 16; batch -= 16) {
      const uint8x16_t vt = vld1q_u8(input); input += 16;
      vacc16 = vpadalq_u8(vacc16, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      const uint8x16_t vt = vld1q_u8(input);
      const uint8x16_t vonemask = vld1q_u8(&onemask_table[16 - batch]);
      const uint8x16_t vtm = vmulq_u8(vt, vonemask);
      vacc16 = vpadalq_u8(vacc16, vtm);
    }
    vacc0 = vpadalq_u16(vacc0, vacc16);
  }
  vacc0 = vaddq_u32(vacc0, vacc1);
  vacc2 = vaddq_u32(vacc2, vacc3);
  vacc0 = vaddq_u32(vacc0, vacc2);
  #if XNN_ARCH_ARM64
    const uint32_t vacc = vaddvq_u32(vacc0);
  #else
    uint32x2_t vacc_lo = vadd_u32(vget_low_u32(vacc0), vget_high_u32(vacc0));
    vacc_lo = vpadd_u32(vacc_lo, vacc_lo);
    const uint32_t vacc = vget_lane_u32(vacc_lo, 0);
  #endif

  *output += vacc;
}
