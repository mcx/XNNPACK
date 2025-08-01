// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-neoni8mm.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }

  // Loop over groups of 32 columns.
  do {
    // Initialize accumulators with bias. 32 bias values are loaded from the
    // weight matrix, at the start of the group of 32 columns.
    const int32x4_t vinput_zero_point01 = vld1q_s32(&quantization_params[0].zero_point);
    const int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumGHIJ = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xGHIJ = vmulq_lane_s32(vksumGHIJ, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xGHIJ = vmulq_lane_s32(vksumGHIJ, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumKLMN = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xKLMN = vmulq_lane_s32(vksumKLMN, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xKLMN = vmulq_lane_s32(vksumKLMN, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumOPQR = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xOPQR = vmulq_lane_s32(vksumOPQR, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xOPQR = vmulq_lane_s32(vksumOPQR, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vksumSTUV = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vksumzp0xSTUV = vmulq_lane_s32(vksumSTUV, vget_low_s32(vinput_zero_point01), 0);
    const int32x4_t vksumzp1xSTUV = vmulq_lane_s32(vksumSTUV, vget_high_s32(vinput_zero_point01), 0);
    const int32x4_t vinput_zero_point23 = vld1q_s32(&quantization_params[2].zero_point);
    const int32x4_t vksumzp2x0123 = vmulq_lane_s32(vksum0123, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3x0123 = vmulq_lane_s32(vksum0123, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2x4567 = vmulq_lane_s32(vksum4567, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3x4567 = vmulq_lane_s32(vksum4567, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2x89AB = vmulq_lane_s32(vksum89AB, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3x89AB = vmulq_lane_s32(vksum89AB, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xCDEF = vmulq_lane_s32(vksumCDEF, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xCDEF = vmulq_lane_s32(vksumCDEF, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xGHIJ = vmulq_lane_s32(vksumGHIJ, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xGHIJ = vmulq_lane_s32(vksumGHIJ, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xKLMN = vmulq_lane_s32(vksumKLMN, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xKLMN = vmulq_lane_s32(vksumKLMN, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xOPQR = vmulq_lane_s32(vksumOPQR, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xOPQR = vmulq_lane_s32(vksumOPQR, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp2xSTUV = vmulq_lane_s32(vksumSTUV, vget_low_s32(vinput_zero_point23), 0);
    const int32x4_t vksumzp3xSTUV = vmulq_lane_s32(vksumSTUV, vget_high_s32(vinput_zero_point23), 0);
    const int32x4_t vinput_zero_point45 = vld1q_dup_s32(&quantization_params[4].zero_point);
    const int32x4_t vksumzp4x0123 = vmulq_s32(vksum0123, vinput_zero_point45);
    const int32x4_t vksumzp4x4567 = vmulq_s32(vksum4567, vinput_zero_point45);
    const int32x4_t vksumzp4x89AB = vmulq_s32(vksum89AB, vinput_zero_point45);
    const int32x4_t vksumzp4xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point45);
    const int32x4_t vksumzp4xGHIJ = vmulq_s32(vksumGHIJ, vinput_zero_point45);
    const int32x4_t vksumzp4xKLMN = vmulq_s32(vksumKLMN, vinput_zero_point45);
    const int32x4_t vksumzp4xOPQR = vmulq_s32(vksumOPQR, vinput_zero_point45);
    const int32x4_t vksumzp4xSTUV = vmulq_s32(vksumSTUV, vinput_zero_point45);

    int32x4_t vacc01x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp1x0123)));
    int32x4_t vacc01x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x0123), vreinterpretq_u64_s32(vksumzp1x0123)));
    int32x4_t vacc01x45 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x4567), vreinterpretq_u64_s32(vksumzp1x4567)));
    int32x4_t vacc01x67 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x4567), vreinterpretq_u64_s32(vksumzp1x4567)));
    int32x4_t vacc01x89 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0x89AB), vreinterpretq_u64_s32(vksumzp1x89AB)));
    int32x4_t vacc01xAB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0x89AB), vreinterpretq_u64_s32(vksumzp1x89AB)));
    int32x4_t vacc01xCD = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xCDEF), vreinterpretq_u64_s32(vksumzp1xCDEF)));
    int32x4_t vacc01xEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xCDEF), vreinterpretq_u64_s32(vksumzp1xCDEF)));
    int32x4_t vacc01xGH = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xGHIJ), vreinterpretq_u64_s32(vksumzp1xGHIJ)));
    int32x4_t vacc01xIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xGHIJ), vreinterpretq_u64_s32(vksumzp1xGHIJ)));
    int32x4_t vacc01xKL = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xKLMN), vreinterpretq_u64_s32(vksumzp1xKLMN)));
    int32x4_t vacc01xMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xKLMN), vreinterpretq_u64_s32(vksumzp1xKLMN)));
    int32x4_t vacc01xOP = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xOPQR), vreinterpretq_u64_s32(vksumzp1xOPQR)));
    int32x4_t vacc01xQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xOPQR), vreinterpretq_u64_s32(vksumzp1xOPQR)));
    int32x4_t vacc01xST = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp0xSTUV), vreinterpretq_u64_s32(vksumzp1xSTUV)));
    int32x4_t vacc01xUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp0xSTUV), vreinterpretq_u64_s32(vksumzp1xSTUV)));
    int32x4_t vacc23x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x0123), vreinterpretq_u64_s32(vksumzp3x0123)));
    int32x4_t vacc23x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x0123), vreinterpretq_u64_s32(vksumzp3x0123)));
    int32x4_t vacc23x45 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x4567), vreinterpretq_u64_s32(vksumzp3x4567)));
    int32x4_t vacc23x67 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x4567), vreinterpretq_u64_s32(vksumzp3x4567)));
    int32x4_t vacc23x89 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2x89AB), vreinterpretq_u64_s32(vksumzp3x89AB)));
    int32x4_t vacc23xAB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2x89AB), vreinterpretq_u64_s32(vksumzp3x89AB)));
    int32x4_t vacc23xCD = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xCDEF), vreinterpretq_u64_s32(vksumzp3xCDEF)));
    int32x4_t vacc23xEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xCDEF), vreinterpretq_u64_s32(vksumzp3xCDEF)));
    int32x4_t vacc23xGH = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xGHIJ), vreinterpretq_u64_s32(vksumzp3xGHIJ)));
    int32x4_t vacc23xIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xGHIJ), vreinterpretq_u64_s32(vksumzp3xGHIJ)));
    int32x4_t vacc23xKL = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xKLMN), vreinterpretq_u64_s32(vksumzp3xKLMN)));
    int32x4_t vacc23xMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xKLMN), vreinterpretq_u64_s32(vksumzp3xKLMN)));
    int32x4_t vacc23xOP = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xOPQR), vreinterpretq_u64_s32(vksumzp3xOPQR)));
    int32x4_t vacc23xQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xOPQR), vreinterpretq_u64_s32(vksumzp3xOPQR)));
    int32x4_t vacc23xST = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp2xSTUV), vreinterpretq_u64_s32(vksumzp3xSTUV)));
    int32x4_t vacc23xUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp2xSTUV), vreinterpretq_u64_s32(vksumzp3xSTUV)));
    int32x4_t vacc45x01 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4x0123), vreinterpretq_u64_s32(vksumzp4x0123)));
    int32x4_t vacc45x23 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4x0123), vreinterpretq_u64_s32(vksumzp4x0123)));
    int32x4_t vacc45x45 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4x4567), vreinterpretq_u64_s32(vksumzp4x4567)));
    int32x4_t vacc45x67 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4x4567), vreinterpretq_u64_s32(vksumzp4x4567)));
    int32x4_t vacc45x89 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4x89AB), vreinterpretq_u64_s32(vksumzp4x89AB)));
    int32x4_t vacc45xAB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4x89AB), vreinterpretq_u64_s32(vksumzp4x89AB)));
    int32x4_t vacc45xCD = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4xCDEF), vreinterpretq_u64_s32(vksumzp4xCDEF)));
    int32x4_t vacc45xEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4xCDEF), vreinterpretq_u64_s32(vksumzp4xCDEF)));
    int32x4_t vacc45xGH = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4xGHIJ), vreinterpretq_u64_s32(vksumzp4xGHIJ)));
    int32x4_t vacc45xIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4xGHIJ), vreinterpretq_u64_s32(vksumzp4xGHIJ)));
    int32x4_t vacc45xKL = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4xKLMN), vreinterpretq_u64_s32(vksumzp4xKLMN)));
    int32x4_t vacc45xMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4xKLMN), vreinterpretq_u64_s32(vksumzp4xKLMN)));
    int32x4_t vacc45xOP = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4xOPQR), vreinterpretq_u64_s32(vksumzp4xOPQR)));
    int32x4_t vacc45xQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4xOPQR), vreinterpretq_u64_s32(vksumzp4xOPQR)));
    int32x4_t vacc45xST = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vksumzp4xSTUV), vreinterpretq_u64_s32(vksumzp4xSTUV)));
    int32x4_t vacc45xUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vksumzp4xSTUV), vreinterpretq_u64_s32(vksumzp4xSTUV)));

    // Inner accumulation loop along the 32 columns.
    size_t k = kc;
    // 2x partial unrolled loop to load 8 bytes at a time.

    uint64x2x2_t va01x0123456789ABCDEF;
    va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
    va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
    uint64x2x2_t va23x0123456789ABCDEF;
    va23x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
    va23x0123456789ABCDEF.val[1] = vdupq_n_u64(0);
    uint64x2x2_t va45x0123456789ABCDEF;
    va45x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
    va45x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

    while (k >= 16 * sizeof(int8_t)) {
      // Load a 5x16 block of activations.
      va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a0, va01x0123456789ABCDEF, 0); a0 += 16;
      va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a2, va23x0123456789ABCDEF, 0); a2 += 16;
      va45x0123456789ABCDEF = vld2q_lane_u64((const void*) a4, va45x0123456789ABCDEF, 0); a4 += 16;
      va01x0123456789ABCDEF = vld2q_lane_u64((const void*) a1, va01x0123456789ABCDEF, 1); a1 += 16;
      va23x0123456789ABCDEF = vld2q_lane_u64((const void*) a3, va23x0123456789ABCDEF, 1); a3 += 16;

      const int8x16_t va01x01234567 = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]);
      const int8x16_t va01x89ABCDEF = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]);
      const int8x16_t va23x01234567 = vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[0]);
      const int8x16_t va23x89ABCDEF = vreinterpretq_s8_u64(va23x0123456789ABCDEF.val[1]);
      const int8x16_t va45x01234567 = vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[0]);
      const int8x16_t va45x89ABCDEF = vreinterpretq_s8_u64(va45x0123456789ABCDEF.val[1]);

      // Load a 16x32 block of weights.
      const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbGHx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbIJx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbKLx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbMNx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbOPx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbQRx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbSTx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbUVx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89x89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbGHx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbIJx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbKLx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbMNx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbOPx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbQRx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbSTx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbUVx89ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 5x8 * 8x32 --> 5x32.
      vacc01x01 = vmmlaq_s32(vacc01x01, va01x01234567, vb01x01234567);
      vacc01x23 = vmmlaq_s32(vacc01x23, va01x01234567, vb23x01234567);
      vacc01x45 = vmmlaq_s32(vacc01x45, va01x01234567, vb45x01234567);
      vacc01x67 = vmmlaq_s32(vacc01x67, va01x01234567, vb67x01234567);
      vacc01x89 = vmmlaq_s32(vacc01x89, va01x01234567, vb89x01234567);
      vacc01xAB = vmmlaq_s32(vacc01xAB, va01x01234567, vbABx01234567);
      vacc01xCD = vmmlaq_s32(vacc01xCD, va01x01234567, vbCDx01234567);
      vacc01xEF = vmmlaq_s32(vacc01xEF, va01x01234567, vbEFx01234567);
      vacc01xGH = vmmlaq_s32(vacc01xGH, va01x01234567, vbGHx01234567);
      vacc01xIJ = vmmlaq_s32(vacc01xIJ, va01x01234567, vbIJx01234567);
      vacc01xKL = vmmlaq_s32(vacc01xKL, va01x01234567, vbKLx01234567);
      vacc01xMN = vmmlaq_s32(vacc01xMN, va01x01234567, vbMNx01234567);
      vacc01xOP = vmmlaq_s32(vacc01xOP, va01x01234567, vbOPx01234567);
      vacc01xQR = vmmlaq_s32(vacc01xQR, va01x01234567, vbQRx01234567);
      vacc01xST = vmmlaq_s32(vacc01xST, va01x01234567, vbSTx01234567);
      vacc01xUV = vmmlaq_s32(vacc01xUV, va01x01234567, vbUVx01234567);
      vacc23x01 = vmmlaq_s32(vacc23x01, va23x01234567, vb01x01234567);
      vacc23x23 = vmmlaq_s32(vacc23x23, va23x01234567, vb23x01234567);
      vacc23x45 = vmmlaq_s32(vacc23x45, va23x01234567, vb45x01234567);
      vacc23x67 = vmmlaq_s32(vacc23x67, va23x01234567, vb67x01234567);
      vacc23x89 = vmmlaq_s32(vacc23x89, va23x01234567, vb89x01234567);
      vacc23xAB = vmmlaq_s32(vacc23xAB, va23x01234567, vbABx01234567);
      vacc23xCD = vmmlaq_s32(vacc23xCD, va23x01234567, vbCDx01234567);
      vacc23xEF = vmmlaq_s32(vacc23xEF, va23x01234567, vbEFx01234567);
      vacc23xGH = vmmlaq_s32(vacc23xGH, va23x01234567, vbGHx01234567);
      vacc23xIJ = vmmlaq_s32(vacc23xIJ, va23x01234567, vbIJx01234567);
      vacc23xKL = vmmlaq_s32(vacc23xKL, va23x01234567, vbKLx01234567);
      vacc23xMN = vmmlaq_s32(vacc23xMN, va23x01234567, vbMNx01234567);
      vacc23xOP = vmmlaq_s32(vacc23xOP, va23x01234567, vbOPx01234567);
      vacc23xQR = vmmlaq_s32(vacc23xQR, va23x01234567, vbQRx01234567);
      vacc23xST = vmmlaq_s32(vacc23xST, va23x01234567, vbSTx01234567);
      vacc23xUV = vmmlaq_s32(vacc23xUV, va23x01234567, vbUVx01234567);
      vacc45x01 = vmmlaq_s32(vacc45x01, va45x01234567, vb01x01234567);
      vacc45x23 = vmmlaq_s32(vacc45x23, va45x01234567, vb23x01234567);
      vacc45x45 = vmmlaq_s32(vacc45x45, va45x01234567, vb45x01234567);
      vacc45x67 = vmmlaq_s32(vacc45x67, va45x01234567, vb67x01234567);
      vacc45x89 = vmmlaq_s32(vacc45x89, va45x01234567, vb89x01234567);
      vacc45xAB = vmmlaq_s32(vacc45xAB, va45x01234567, vbABx01234567);
      vacc45xCD = vmmlaq_s32(vacc45xCD, va45x01234567, vbCDx01234567);
      vacc45xEF = vmmlaq_s32(vacc45xEF, va45x01234567, vbEFx01234567);
      vacc45xGH = vmmlaq_s32(vacc45xGH, va45x01234567, vbGHx01234567);
      vacc45xIJ = vmmlaq_s32(vacc45xIJ, va45x01234567, vbIJx01234567);
      vacc45xKL = vmmlaq_s32(vacc45xKL, va45x01234567, vbKLx01234567);
      vacc45xMN = vmmlaq_s32(vacc45xMN, va45x01234567, vbMNx01234567);
      vacc45xOP = vmmlaq_s32(vacc45xOP, va45x01234567, vbOPx01234567);
      vacc45xQR = vmmlaq_s32(vacc45xQR, va45x01234567, vbQRx01234567);
      vacc45xST = vmmlaq_s32(vacc45xST, va45x01234567, vbSTx01234567);
      vacc45xUV = vmmlaq_s32(vacc45xUV, va45x01234567, vbUVx01234567);
      vacc01x01 = vmmlaq_s32(vacc01x01, va01x89ABCDEF, vb01x89ABCDEF);
      vacc01x23 = vmmlaq_s32(vacc01x23, va01x89ABCDEF, vb23x89ABCDEF);
      vacc01x45 = vmmlaq_s32(vacc01x45, va01x89ABCDEF, vb45x89ABCDEF);
      vacc01x67 = vmmlaq_s32(vacc01x67, va01x89ABCDEF, vb67x89ABCDEF);
      vacc01x89 = vmmlaq_s32(vacc01x89, va01x89ABCDEF, vb89x89ABCDEF);
      vacc01xAB = vmmlaq_s32(vacc01xAB, va01x89ABCDEF, vbABx89ABCDEF);
      vacc01xCD = vmmlaq_s32(vacc01xCD, va01x89ABCDEF, vbCDx89ABCDEF);
      vacc01xEF = vmmlaq_s32(vacc01xEF, va01x89ABCDEF, vbEFx89ABCDEF);
      vacc01xGH = vmmlaq_s32(vacc01xGH, va01x89ABCDEF, vbGHx89ABCDEF);
      vacc01xIJ = vmmlaq_s32(vacc01xIJ, va01x89ABCDEF, vbIJx89ABCDEF);
      vacc01xKL = vmmlaq_s32(vacc01xKL, va01x89ABCDEF, vbKLx89ABCDEF);
      vacc01xMN = vmmlaq_s32(vacc01xMN, va01x89ABCDEF, vbMNx89ABCDEF);
      vacc01xOP = vmmlaq_s32(vacc01xOP, va01x89ABCDEF, vbOPx89ABCDEF);
      vacc01xQR = vmmlaq_s32(vacc01xQR, va01x89ABCDEF, vbQRx89ABCDEF);
      vacc01xST = vmmlaq_s32(vacc01xST, va01x89ABCDEF, vbSTx89ABCDEF);
      vacc01xUV = vmmlaq_s32(vacc01xUV, va01x89ABCDEF, vbUVx89ABCDEF);
      vacc23x01 = vmmlaq_s32(vacc23x01, va23x89ABCDEF, vb01x89ABCDEF);
      vacc23x23 = vmmlaq_s32(vacc23x23, va23x89ABCDEF, vb23x89ABCDEF);
      vacc23x45 = vmmlaq_s32(vacc23x45, va23x89ABCDEF, vb45x89ABCDEF);
      vacc23x67 = vmmlaq_s32(vacc23x67, va23x89ABCDEF, vb67x89ABCDEF);
      vacc23x89 = vmmlaq_s32(vacc23x89, va23x89ABCDEF, vb89x89ABCDEF);
      vacc23xAB = vmmlaq_s32(vacc23xAB, va23x89ABCDEF, vbABx89ABCDEF);
      vacc23xCD = vmmlaq_s32(vacc23xCD, va23x89ABCDEF, vbCDx89ABCDEF);
      vacc23xEF = vmmlaq_s32(vacc23xEF, va23x89ABCDEF, vbEFx89ABCDEF);
      vacc23xGH = vmmlaq_s32(vacc23xGH, va23x89ABCDEF, vbGHx89ABCDEF);
      vacc23xIJ = vmmlaq_s32(vacc23xIJ, va23x89ABCDEF, vbIJx89ABCDEF);
      vacc23xKL = vmmlaq_s32(vacc23xKL, va23x89ABCDEF, vbKLx89ABCDEF);
      vacc23xMN = vmmlaq_s32(vacc23xMN, va23x89ABCDEF, vbMNx89ABCDEF);
      vacc23xOP = vmmlaq_s32(vacc23xOP, va23x89ABCDEF, vbOPx89ABCDEF);
      vacc23xQR = vmmlaq_s32(vacc23xQR, va23x89ABCDEF, vbQRx89ABCDEF);
      vacc23xST = vmmlaq_s32(vacc23xST, va23x89ABCDEF, vbSTx89ABCDEF);
      vacc23xUV = vmmlaq_s32(vacc23xUV, va23x89ABCDEF, vbUVx89ABCDEF);
      vacc45x01 = vmmlaq_s32(vacc45x01, va45x89ABCDEF, vb01x89ABCDEF);
      vacc45x23 = vmmlaq_s32(vacc45x23, va45x89ABCDEF, vb23x89ABCDEF);
      vacc45x45 = vmmlaq_s32(vacc45x45, va45x89ABCDEF, vb45x89ABCDEF);
      vacc45x67 = vmmlaq_s32(vacc45x67, va45x89ABCDEF, vb67x89ABCDEF);
      vacc45x89 = vmmlaq_s32(vacc45x89, va45x89ABCDEF, vb89x89ABCDEF);
      vacc45xAB = vmmlaq_s32(vacc45xAB, va45x89ABCDEF, vbABx89ABCDEF);
      vacc45xCD = vmmlaq_s32(vacc45xCD, va45x89ABCDEF, vbCDx89ABCDEF);
      vacc45xEF = vmmlaq_s32(vacc45xEF, va45x89ABCDEF, vbEFx89ABCDEF);
      vacc45xGH = vmmlaq_s32(vacc45xGH, va45x89ABCDEF, vbGHx89ABCDEF);
      vacc45xIJ = vmmlaq_s32(vacc45xIJ, va45x89ABCDEF, vbIJx89ABCDEF);
      vacc45xKL = vmmlaq_s32(vacc45xKL, va45x89ABCDEF, vbKLx89ABCDEF);
      vacc45xMN = vmmlaq_s32(vacc45xMN, va45x89ABCDEF, vbMNx89ABCDEF);
      vacc45xOP = vmmlaq_s32(vacc45xOP, va45x89ABCDEF, vbOPx89ABCDEF);
      vacc45xQR = vmmlaq_s32(vacc45xQR, va45x89ABCDEF, vbQRx89ABCDEF);
      vacc45xST = vmmlaq_s32(vacc45xST, va45x89ABCDEF, vbSTx89ABCDEF);
      vacc45xUV = vmmlaq_s32(vacc45xUV, va45x89ABCDEF, vbUVx89ABCDEF);

      k -= 16 * sizeof(int8_t);
    }
    // Handle up to 8 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 5x8 block of activations.
      uint64x2_t va01x01234567 = vld1q_dup_u64((const void*) a0); a0 += 8;
      uint64x2_t va23x01234567 = vld1q_dup_u64((const void*) a2); a2 += 8;
      uint64x2_t va45x01234567 = vld1q_dup_u64((const void*) a4); a4 += 8;
      va01x01234567 = vld1q_lane_u64((const void*) a1, va01x01234567, 1); a1 += 8;
      va23x01234567 = vld1q_lane_u64((const void*) a3, va23x01234567, 1); a3 += 8;

      // Load a 16x32 block of weights.
      const int8x16_t vb01x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb23x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb45x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb67x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb89x01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbABx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbCDx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbEFx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbGHx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbIJx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbKLx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbMNx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbOPx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbQRx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbSTx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vbUVx01234567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 5x4 * 4x32 --> 5x32.
      vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x01234567), vb01x01234567);
      vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x01234567), vb23x01234567);
      vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x01234567), vb45x01234567);
      vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x01234567), vb67x01234567);
      vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x01234567), vb89x01234567);
      vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x01234567), vbABx01234567);
      vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x01234567), vbCDx01234567);
      vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x01234567), vbEFx01234567);
      vacc01xGH = vmmlaq_s32(vacc01xGH, vreinterpretq_s8_u64(va01x01234567), vbGHx01234567);
      vacc01xIJ = vmmlaq_s32(vacc01xIJ, vreinterpretq_s8_u64(va01x01234567), vbIJx01234567);
      vacc01xKL = vmmlaq_s32(vacc01xKL, vreinterpretq_s8_u64(va01x01234567), vbKLx01234567);
      vacc01xMN = vmmlaq_s32(vacc01xMN, vreinterpretq_s8_u64(va01x01234567), vbMNx01234567);
      vacc01xOP = vmmlaq_s32(vacc01xOP, vreinterpretq_s8_u64(va01x01234567), vbOPx01234567);
      vacc01xQR = vmmlaq_s32(vacc01xQR, vreinterpretq_s8_u64(va01x01234567), vbQRx01234567);
      vacc01xST = vmmlaq_s32(vacc01xST, vreinterpretq_s8_u64(va01x01234567), vbSTx01234567);
      vacc01xUV = vmmlaq_s32(vacc01xUV, vreinterpretq_s8_u64(va01x01234567), vbUVx01234567);
      vacc23x01 = vmmlaq_s32(vacc23x01, vreinterpretq_s8_u64(va23x01234567), vb01x01234567);
      vacc23x23 = vmmlaq_s32(vacc23x23, vreinterpretq_s8_u64(va23x01234567), vb23x01234567);
      vacc23x45 = vmmlaq_s32(vacc23x45, vreinterpretq_s8_u64(va23x01234567), vb45x01234567);
      vacc23x67 = vmmlaq_s32(vacc23x67, vreinterpretq_s8_u64(va23x01234567), vb67x01234567);
      vacc23x89 = vmmlaq_s32(vacc23x89, vreinterpretq_s8_u64(va23x01234567), vb89x01234567);
      vacc23xAB = vmmlaq_s32(vacc23xAB, vreinterpretq_s8_u64(va23x01234567), vbABx01234567);
      vacc23xCD = vmmlaq_s32(vacc23xCD, vreinterpretq_s8_u64(va23x01234567), vbCDx01234567);
      vacc23xEF = vmmlaq_s32(vacc23xEF, vreinterpretq_s8_u64(va23x01234567), vbEFx01234567);
      vacc23xGH = vmmlaq_s32(vacc23xGH, vreinterpretq_s8_u64(va23x01234567), vbGHx01234567);
      vacc23xIJ = vmmlaq_s32(vacc23xIJ, vreinterpretq_s8_u64(va23x01234567), vbIJx01234567);
      vacc23xKL = vmmlaq_s32(vacc23xKL, vreinterpretq_s8_u64(va23x01234567), vbKLx01234567);
      vacc23xMN = vmmlaq_s32(vacc23xMN, vreinterpretq_s8_u64(va23x01234567), vbMNx01234567);
      vacc23xOP = vmmlaq_s32(vacc23xOP, vreinterpretq_s8_u64(va23x01234567), vbOPx01234567);
      vacc23xQR = vmmlaq_s32(vacc23xQR, vreinterpretq_s8_u64(va23x01234567), vbQRx01234567);
      vacc23xST = vmmlaq_s32(vacc23xST, vreinterpretq_s8_u64(va23x01234567), vbSTx01234567);
      vacc23xUV = vmmlaq_s32(vacc23xUV, vreinterpretq_s8_u64(va23x01234567), vbUVx01234567);
      vacc45x01 = vmmlaq_s32(vacc45x01, vreinterpretq_s8_u64(va45x01234567), vb01x01234567);
      vacc45x23 = vmmlaq_s32(vacc45x23, vreinterpretq_s8_u64(va45x01234567), vb23x01234567);
      vacc45x45 = vmmlaq_s32(vacc45x45, vreinterpretq_s8_u64(va45x01234567), vb45x01234567);
      vacc45x67 = vmmlaq_s32(vacc45x67, vreinterpretq_s8_u64(va45x01234567), vb67x01234567);
      vacc45x89 = vmmlaq_s32(vacc45x89, vreinterpretq_s8_u64(va45x01234567), vb89x01234567);
      vacc45xAB = vmmlaq_s32(vacc45xAB, vreinterpretq_s8_u64(va45x01234567), vbABx01234567);
      vacc45xCD = vmmlaq_s32(vacc45xCD, vreinterpretq_s8_u64(va45x01234567), vbCDx01234567);
      vacc45xEF = vmmlaq_s32(vacc45xEF, vreinterpretq_s8_u64(va45x01234567), vbEFx01234567);
      vacc45xGH = vmmlaq_s32(vacc45xGH, vreinterpretq_s8_u64(va45x01234567), vbGHx01234567);
      vacc45xIJ = vmmlaq_s32(vacc45xIJ, vreinterpretq_s8_u64(va45x01234567), vbIJx01234567);
      vacc45xKL = vmmlaq_s32(vacc45xKL, vreinterpretq_s8_u64(va45x01234567), vbKLx01234567);
      vacc45xMN = vmmlaq_s32(vacc45xMN, vreinterpretq_s8_u64(va45x01234567), vbMNx01234567);
      vacc45xOP = vmmlaq_s32(vacc45xOP, vreinterpretq_s8_u64(va45x01234567), vbOPx01234567);
      vacc45xQR = vmmlaq_s32(vacc45xQR, vreinterpretq_s8_u64(va45x01234567), vbQRx01234567);
      vacc45xST = vmmlaq_s32(vacc45xST, vreinterpretq_s8_u64(va45x01234567), vbSTx01234567);
      vacc45xUV = vmmlaq_s32(vacc45xUV, vreinterpretq_s8_u64(va45x01234567), vbUVx01234567);
    }

    int32x4_t vacc0x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
    int32x4_t vacc1x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
    int32x4_t vacc0x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
    int32x4_t vacc1x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
    int32x4_t vacc0x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
    int32x4_t vacc1x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
    int32x4_t vacc0xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));
    int32x4_t vacc1xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));
    int32x4_t vacc0xGHIJ = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xGH), vreinterpretq_u64_s32(vacc01xIJ)));
    int32x4_t vacc1xGHIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xGH), vreinterpretq_u64_s32(vacc01xIJ)));
    int32x4_t vacc0xKLMN = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xKL), vreinterpretq_u64_s32(vacc01xMN)));
    int32x4_t vacc1xKLMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xKL), vreinterpretq_u64_s32(vacc01xMN)));
    int32x4_t vacc0xOPQR = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xOP), vreinterpretq_u64_s32(vacc01xQR)));
    int32x4_t vacc1xOPQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xOP), vreinterpretq_u64_s32(vacc01xQR)));
    int32x4_t vacc0xSTUV = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xST), vreinterpretq_u64_s32(vacc01xUV)));
    int32x4_t vacc1xSTUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc01xST), vreinterpretq_u64_s32(vacc01xUV)));
    int32x4_t vacc2x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x01), vreinterpretq_u64_s32(vacc23x23)));
    int32x4_t vacc3x0123 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x01), vreinterpretq_u64_s32(vacc23x23)));
    int32x4_t vacc2x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x45), vreinterpretq_u64_s32(vacc23x67)));
    int32x4_t vacc3x4567 = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x45), vreinterpretq_u64_s32(vacc23x67)));
    int32x4_t vacc2x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23x89), vreinterpretq_u64_s32(vacc23xAB)));
    int32x4_t vacc3x89AB = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23x89), vreinterpretq_u64_s32(vacc23xAB)));
    int32x4_t vacc2xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xCD), vreinterpretq_u64_s32(vacc23xEF)));
    int32x4_t vacc3xCDEF = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xCD), vreinterpretq_u64_s32(vacc23xEF)));
    int32x4_t vacc2xGHIJ = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xGH), vreinterpretq_u64_s32(vacc23xIJ)));
    int32x4_t vacc3xGHIJ = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xGH), vreinterpretq_u64_s32(vacc23xIJ)));
    int32x4_t vacc2xKLMN = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xKL), vreinterpretq_u64_s32(vacc23xMN)));
    int32x4_t vacc3xKLMN = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xKL), vreinterpretq_u64_s32(vacc23xMN)));
    int32x4_t vacc2xOPQR = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xOP), vreinterpretq_u64_s32(vacc23xQR)));
    int32x4_t vacc3xOPQR = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xOP), vreinterpretq_u64_s32(vacc23xQR)));
    int32x4_t vacc2xSTUV = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc23xST), vreinterpretq_u64_s32(vacc23xUV)));
    int32x4_t vacc3xSTUV = vreinterpretq_s32_u64(vtrn2q_u64(vreinterpretq_u64_s32(vacc23xST), vreinterpretq_u64_s32(vacc23xUV)));
    int32x4_t vacc4x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x01), vreinterpretq_u64_s32(vacc45x23)));
    int32x4_t vacc4x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x45), vreinterpretq_u64_s32(vacc45x67)));
    int32x4_t vacc4x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45x89), vreinterpretq_u64_s32(vacc45xAB)));
    int32x4_t vacc4xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45xCD), vreinterpretq_u64_s32(vacc45xEF)));
    int32x4_t vacc4xGHIJ = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45xGH), vreinterpretq_u64_s32(vacc45xIJ)));
    int32x4_t vacc4xKLMN = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45xKL), vreinterpretq_u64_s32(vacc45xMN)));
    int32x4_t vacc4xOPQR = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45xOP), vreinterpretq_u64_s32(vacc45xQR)));
    int32x4_t vacc4xSTUV = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc45xST), vreinterpretq_u64_s32(vacc45xUV)));
    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout0xGHIJ = vcvtq_f32_s32(vacc0xGHIJ);
    float32x4_t vout0xKLMN = vcvtq_f32_s32(vacc0xKLMN);
    float32x4_t vout0xOPQR = vcvtq_f32_s32(vacc0xOPQR);
    float32x4_t vout0xSTUV = vcvtq_f32_s32(vacc0xSTUV);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vout1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vout1xGHIJ = vcvtq_f32_s32(vacc1xGHIJ);
    float32x4_t vout1xKLMN = vcvtq_f32_s32(vacc1xKLMN);
    float32x4_t vout1xOPQR = vcvtq_f32_s32(vacc1xOPQR);
    float32x4_t vout1xSTUV = vcvtq_f32_s32(vacc1xSTUV);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vout2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vout2xGHIJ = vcvtq_f32_s32(vacc2xGHIJ);
    float32x4_t vout2xKLMN = vcvtq_f32_s32(vacc2xKLMN);
    float32x4_t vout2xOPQR = vcvtq_f32_s32(vacc2xOPQR);
    float32x4_t vout2xSTUV = vcvtq_f32_s32(vacc2xSTUV);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vout3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vout3xCDEF = vcvtq_f32_s32(vacc3xCDEF);
    float32x4_t vout3xGHIJ = vcvtq_f32_s32(vacc3xGHIJ);
    float32x4_t vout3xKLMN = vcvtq_f32_s32(vacc3xKLMN);
    float32x4_t vout3xOPQR = vcvtq_f32_s32(vacc3xOPQR);
    float32x4_t vout3xSTUV = vcvtq_f32_s32(vacc3xSTUV);
    float32x4_t vout4x0123 = vcvtq_f32_s32(vacc4x0123);
    float32x4_t vout4x4567 = vcvtq_f32_s32(vacc4x4567);
    float32x4_t vout4x89AB = vcvtq_f32_s32(vacc4x89AB);
    float32x4_t vout4xCDEF = vcvtq_f32_s32(vacc4xCDEF);
    float32x4_t vout4xGHIJ = vcvtq_f32_s32(vacc4xGHIJ);
    float32x4_t vout4xKLMN = vcvtq_f32_s32(vacc4xKLMN);
    float32x4_t vout4xOPQR = vcvtq_f32_s32(vacc4xOPQR);
    float32x4_t vout4xSTUV = vcvtq_f32_s32(vacc4xSTUV);

    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    vout0x0123 = vmulq_lane_f32(vout0x0123, vget_low_f32(vinput_scale01), 1);
    vout1x0123 = vmulq_lane_f32(vout1x0123, vget_high_f32(vinput_scale01), 1);
    vout0x4567 = vmulq_lane_f32(vout0x4567, vget_low_f32(vinput_scale01), 1);
    vout1x4567 = vmulq_lane_f32(vout1x4567, vget_high_f32(vinput_scale01), 1);
    vout0x89AB = vmulq_lane_f32(vout0x89AB, vget_low_f32(vinput_scale01), 1);
    vout1x89AB = vmulq_lane_f32(vout1x89AB, vget_high_f32(vinput_scale01), 1);
    vout0xCDEF = vmulq_lane_f32(vout0xCDEF, vget_low_f32(vinput_scale01), 1);
    vout1xCDEF = vmulq_lane_f32(vout1xCDEF, vget_high_f32(vinput_scale01), 1);
    vout0xGHIJ = vmulq_lane_f32(vout0xGHIJ, vget_low_f32(vinput_scale01), 1);
    vout1xGHIJ = vmulq_lane_f32(vout1xGHIJ, vget_high_f32(vinput_scale01), 1);
    vout0xKLMN = vmulq_lane_f32(vout0xKLMN, vget_low_f32(vinput_scale01), 1);
    vout1xKLMN = vmulq_lane_f32(vout1xKLMN, vget_high_f32(vinput_scale01), 1);
    vout0xOPQR = vmulq_lane_f32(vout0xOPQR, vget_low_f32(vinput_scale01), 1);
    vout1xOPQR = vmulq_lane_f32(vout1xOPQR, vget_high_f32(vinput_scale01), 1);
    vout0xSTUV = vmulq_lane_f32(vout0xSTUV, vget_low_f32(vinput_scale01), 1);
    vout1xSTUV = vmulq_lane_f32(vout1xSTUV, vget_high_f32(vinput_scale01), 1);
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    vout2x0123 = vmulq_lane_f32(vout2x0123, vget_low_f32(vinput_scale23), 1);
    vout3x0123 = vmulq_lane_f32(vout3x0123, vget_high_f32(vinput_scale23), 1);
    vout2x4567 = vmulq_lane_f32(vout2x4567, vget_low_f32(vinput_scale23), 1);
    vout3x4567 = vmulq_lane_f32(vout3x4567, vget_high_f32(vinput_scale23), 1);
    vout2x89AB = vmulq_lane_f32(vout2x89AB, vget_low_f32(vinput_scale23), 1);
    vout3x89AB = vmulq_lane_f32(vout3x89AB, vget_high_f32(vinput_scale23), 1);
    vout2xCDEF = vmulq_lane_f32(vout2xCDEF, vget_low_f32(vinput_scale23), 1);
    vout3xCDEF = vmulq_lane_f32(vout3xCDEF, vget_high_f32(vinput_scale23), 1);
    vout2xGHIJ = vmulq_lane_f32(vout2xGHIJ, vget_low_f32(vinput_scale23), 1);
    vout3xGHIJ = vmulq_lane_f32(vout3xGHIJ, vget_high_f32(vinput_scale23), 1);
    vout2xKLMN = vmulq_lane_f32(vout2xKLMN, vget_low_f32(vinput_scale23), 1);
    vout3xKLMN = vmulq_lane_f32(vout3xKLMN, vget_high_f32(vinput_scale23), 1);
    vout2xOPQR = vmulq_lane_f32(vout2xOPQR, vget_low_f32(vinput_scale23), 1);
    vout3xOPQR = vmulq_lane_f32(vout3xOPQR, vget_high_f32(vinput_scale23), 1);
    vout2xSTUV = vmulq_lane_f32(vout2xSTUV, vget_low_f32(vinput_scale23), 1);
    vout3xSTUV = vmulq_lane_f32(vout3xSTUV, vget_high_f32(vinput_scale23), 1);
    const float32x4_t vinput_scale4 = vld1q_dup_f32(&quantization_params[4].inv_scale);
    vout4x0123 = vmulq_f32(vout4x0123, vinput_scale4);
    vout4x4567 = vmulq_f32(vout4x4567, vinput_scale4);
    vout4x89AB = vmulq_f32(vout4x89AB, vinput_scale4);
    vout4xCDEF = vmulq_f32(vout4xCDEF, vinput_scale4);
    vout4xGHIJ = vmulq_f32(vout4xGHIJ, vinput_scale4);
    vout4xKLMN = vmulq_f32(vout4xKLMN, vinput_scale4);
    vout4xOPQR = vmulq_f32(vout4xOPQR, vinput_scale4);
    vout4xSTUV = vmulq_f32(vout4xSTUV, vinput_scale4);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleGHIJ = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleKLMN = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleOPQR = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleSTUV = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
    vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
    vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
    vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    vout4x0123 = vfmaq_f32(vbias0123, vout4x0123, vfilter_output_scale0123);
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
    vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
    vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
    vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    vout4x4567 = vfmaq_f32(vbias4567, vout4x4567, vfilter_output_scale4567);
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
    vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
    vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
    vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    vout4x89AB = vfmaq_f32(vbias89AB, vout4x89AB, vfilter_output_scale89AB);
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
    vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
    vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
    vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    vout4xCDEF = vfmaq_f32(vbiasCDEF, vout4xCDEF, vfilter_output_scaleCDEF);
    const float32x4_t vbiasGHIJ = vld1q_f32(w); w = (const float*) w + 4;
    vout0xGHIJ = vfmaq_f32(vbiasGHIJ, vout0xGHIJ, vfilter_output_scaleGHIJ);
    vout1xGHIJ = vfmaq_f32(vbiasGHIJ, vout1xGHIJ, vfilter_output_scaleGHIJ);
    vout2xGHIJ = vfmaq_f32(vbiasGHIJ, vout2xGHIJ, vfilter_output_scaleGHIJ);
    vout3xGHIJ = vfmaq_f32(vbiasGHIJ, vout3xGHIJ, vfilter_output_scaleGHIJ);
    vout4xGHIJ = vfmaq_f32(vbiasGHIJ, vout4xGHIJ, vfilter_output_scaleGHIJ);
    const float32x4_t vbiasKLMN = vld1q_f32(w); w = (const float*) w + 4;
    vout0xKLMN = vfmaq_f32(vbiasKLMN, vout0xKLMN, vfilter_output_scaleKLMN);
    vout1xKLMN = vfmaq_f32(vbiasKLMN, vout1xKLMN, vfilter_output_scaleKLMN);
    vout2xKLMN = vfmaq_f32(vbiasKLMN, vout2xKLMN, vfilter_output_scaleKLMN);
    vout3xKLMN = vfmaq_f32(vbiasKLMN, vout3xKLMN, vfilter_output_scaleKLMN);
    vout4xKLMN = vfmaq_f32(vbiasKLMN, vout4xKLMN, vfilter_output_scaleKLMN);
    const float32x4_t vbiasOPQR = vld1q_f32(w); w = (const float*) w + 4;
    vout0xOPQR = vfmaq_f32(vbiasOPQR, vout0xOPQR, vfilter_output_scaleOPQR);
    vout1xOPQR = vfmaq_f32(vbiasOPQR, vout1xOPQR, vfilter_output_scaleOPQR);
    vout2xOPQR = vfmaq_f32(vbiasOPQR, vout2xOPQR, vfilter_output_scaleOPQR);
    vout3xOPQR = vfmaq_f32(vbiasOPQR, vout3xOPQR, vfilter_output_scaleOPQR);
    vout4xOPQR = vfmaq_f32(vbiasOPQR, vout4xOPQR, vfilter_output_scaleOPQR);
    const float32x4_t vbiasSTUV = vld1q_f32(w); w = (const float*) w + 4;
    vout0xSTUV = vfmaq_f32(vbiasSTUV, vout0xSTUV, vfilter_output_scaleSTUV);
    vout1xSTUV = vfmaq_f32(vbiasSTUV, vout1xSTUV, vfilter_output_scaleSTUV);
    vout2xSTUV = vfmaq_f32(vbiasSTUV, vout2xSTUV, vfilter_output_scaleSTUV);
    vout3xSTUV = vfmaq_f32(vbiasSTUV, vout3xSTUV, vfilter_output_scaleSTUV);
    vout4xSTUV = vfmaq_f32(vbiasSTUV, vout4xSTUV, vfilter_output_scaleSTUV);

    const float32x4_t voutput_min = vdupq_n_f32(params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout0xGHIJ = vmaxq_f32(vout0xGHIJ, voutput_min);
    vout0xKLMN = vmaxq_f32(vout0xKLMN, voutput_min);
    vout0xOPQR = vmaxq_f32(vout0xOPQR, voutput_min);
    vout0xSTUV = vmaxq_f32(vout0xSTUV, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);
    vout1xGHIJ = vmaxq_f32(vout1xGHIJ, voutput_min);
    vout1xKLMN = vmaxq_f32(vout1xKLMN, voutput_min);
    vout1xOPQR = vmaxq_f32(vout1xOPQR, voutput_min);
    vout1xSTUV = vmaxq_f32(vout1xSTUV, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout2x89AB = vmaxq_f32(vout2x89AB, voutput_min);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, voutput_min);
    vout2xGHIJ = vmaxq_f32(vout2xGHIJ, voutput_min);
    vout2xKLMN = vmaxq_f32(vout2xKLMN, voutput_min);
    vout2xOPQR = vmaxq_f32(vout2xOPQR, voutput_min);
    vout2xSTUV = vmaxq_f32(vout2xSTUV, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);
    vout3x89AB = vmaxq_f32(vout3x89AB, voutput_min);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, voutput_min);
    vout3xGHIJ = vmaxq_f32(vout3xGHIJ, voutput_min);
    vout3xKLMN = vmaxq_f32(vout3xKLMN, voutput_min);
    vout3xOPQR = vmaxq_f32(vout3xOPQR, voutput_min);
    vout3xSTUV = vmaxq_f32(vout3xSTUV, voutput_min);
    vout4x0123 = vmaxq_f32(vout4x0123, voutput_min);
    vout4x4567 = vmaxq_f32(vout4x4567, voutput_min);
    vout4x89AB = vmaxq_f32(vout4x89AB, voutput_min);
    vout4xCDEF = vmaxq_f32(vout4xCDEF, voutput_min);
    vout4xGHIJ = vmaxq_f32(vout4xGHIJ, voutput_min);
    vout4xKLMN = vmaxq_f32(vout4xKLMN, voutput_min);
    vout4xOPQR = vmaxq_f32(vout4xOPQR, voutput_min);
    vout4xSTUV = vmaxq_f32(vout4xSTUV, voutput_min);

    const float32x4_t voutput_max = vdupq_n_f32(params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout0xGHIJ = vminq_f32(vout0xGHIJ, voutput_max);
    vout0xKLMN = vminq_f32(vout0xKLMN, voutput_max);
    vout0xOPQR = vminq_f32(vout0xOPQR, voutput_max);
    vout0xSTUV = vminq_f32(vout0xSTUV, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);
    vout1xGHIJ = vminq_f32(vout1xGHIJ, voutput_max);
    vout1xKLMN = vminq_f32(vout1xKLMN, voutput_max);
    vout1xOPQR = vminq_f32(vout1xOPQR, voutput_max);
    vout1xSTUV = vminq_f32(vout1xSTUV, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout2x89AB = vminq_f32(vout2x89AB, voutput_max);
    vout2xCDEF = vminq_f32(vout2xCDEF, voutput_max);
    vout2xGHIJ = vminq_f32(vout2xGHIJ, voutput_max);
    vout2xKLMN = vminq_f32(vout2xKLMN, voutput_max);
    vout2xOPQR = vminq_f32(vout2xOPQR, voutput_max);
    vout2xSTUV = vminq_f32(vout2xSTUV, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout3x89AB = vminq_f32(vout3x89AB, voutput_max);
    vout3xCDEF = vminq_f32(vout3xCDEF, voutput_max);
    vout3xGHIJ = vminq_f32(vout3xGHIJ, voutput_max);
    vout3xKLMN = vminq_f32(vout3xKLMN, voutput_max);
    vout3xOPQR = vminq_f32(vout3xOPQR, voutput_max);
    vout3xSTUV = vminq_f32(vout3xSTUV, voutput_max);
    vout4x0123 = vminq_f32(vout4x0123, voutput_max);
    vout4x4567 = vminq_f32(vout4x4567, voutput_max);
    vout4x89AB = vminq_f32(vout4x89AB, voutput_max);
    vout4xCDEF = vminq_f32(vout4xCDEF, voutput_max);
    vout4xGHIJ = vminq_f32(vout4xGHIJ, voutput_max);
    vout4xKLMN = vminq_f32(vout4xKLMN, voutput_max);
    vout4xOPQR = vminq_f32(vout4xOPQR, voutput_max);
    vout4xSTUV = vminq_f32(vout4xSTUV, voutput_max);

    if XNN_LIKELY(nc >= 32) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);
      vst1q_f32(c0 + 16, vout0xGHIJ);
      vst1q_f32(c0 + 20, vout0xKLMN);
      vst1q_f32(c0 + 24, vout0xOPQR);
      vst1q_f32(c0 + 28, vout0xSTUV);
      vst1q_f32(c1, vout1x0123);
      vst1q_f32(c1 + 4, vout1x4567);
      vst1q_f32(c1 + 8, vout1x89AB);
      vst1q_f32(c1 + 12, vout1xCDEF);
      vst1q_f32(c1 + 16, vout1xGHIJ);
      vst1q_f32(c1 + 20, vout1xKLMN);
      vst1q_f32(c1 + 24, vout1xOPQR);
      vst1q_f32(c1 + 28, vout1xSTUV);
      vst1q_f32(c2, vout2x0123);
      vst1q_f32(c2 + 4, vout2x4567);
      vst1q_f32(c2 + 8, vout2x89AB);
      vst1q_f32(c2 + 12, vout2xCDEF);
      vst1q_f32(c2 + 16, vout2xGHIJ);
      vst1q_f32(c2 + 20, vout2xKLMN);
      vst1q_f32(c2 + 24, vout2xOPQR);
      vst1q_f32(c2 + 28, vout2xSTUV);
      vst1q_f32(c3, vout3x0123);
      vst1q_f32(c3 + 4, vout3x4567);
      vst1q_f32(c3 + 8, vout3x89AB);
      vst1q_f32(c3 + 12, vout3xCDEF);
      vst1q_f32(c3 + 16, vout3xGHIJ);
      vst1q_f32(c3 + 20, vout3xKLMN);
      vst1q_f32(c3 + 24, vout3xOPQR);
      vst1q_f32(c3 + 28, vout3xSTUV);
      vst1q_f32(c4, vout4x0123);
      vst1q_f32(c4 + 4, vout4x4567);
      vst1q_f32(c4 + 8, vout4x89AB);
      vst1q_f32(c4 + 12, vout4xCDEF);
      vst1q_f32(c4 + 16, vout4xGHIJ);
      vst1q_f32(c4 + 20, vout4xKLMN);
      vst1q_f32(c4 + 24, vout4xOPQR);
      vst1q_f32(c4 + 28, vout4xSTUV);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);

      nc -= 32;
    } else {
     if (nc & 16) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0xGHIJ;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1xGHIJ;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2xGHIJ;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3xGHIJ;
       vst1q_f32(c4, vout4x0123); c4 += 4;
       vout4x0123 = vout4xGHIJ;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xKLMN;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xKLMN;
       vst1q_f32(c2, vout2x4567); c2 += 4;
       vout2x4567 = vout2xKLMN;
       vst1q_f32(c3, vout3x4567); c3 += 4;
       vout3x4567 = vout3xKLMN;
       vst1q_f32(c4, vout4x4567); c4 += 4;
       vout4x4567 = vout4xKLMN;
       vst1q_f32(c0, vout0x89AB); c0 += 4;
       vout0x89AB = vout0xOPQR;
       vst1q_f32(c1, vout1x89AB); c1 += 4;
       vout1x89AB = vout1xOPQR;
       vst1q_f32(c2, vout2x89AB); c2 += 4;
       vout2x89AB = vout2xOPQR;
       vst1q_f32(c3, vout3x89AB); c3 += 4;
       vout3x89AB = vout3xOPQR;
       vst1q_f32(c4, vout4x89AB); c4 += 4;
       vout4x89AB = vout4xOPQR;
       vst1q_f32(c0, vout0xCDEF); c0 += 4;
       vout0xCDEF = vout0xSTUV;
       vst1q_f32(c1, vout1xCDEF); c1 += 4;
       vout1xCDEF = vout1xSTUV;
       vst1q_f32(c2, vout2xCDEF); c2 += 4;
       vout2xCDEF = vout2xSTUV;
       vst1q_f32(c3, vout3xCDEF); c3 += 4;
       vout3xCDEF = vout3xSTUV;
       vst1q_f32(c4, vout4xCDEF); c4 += 4;
       vout4xCDEF = vout4xSTUV;
     }
     if (nc & 8) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x89AB;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x89AB;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x89AB;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x89AB;
       vst1q_f32(c4, vout4x0123); c4 += 4;
       vout4x0123 = vout4x89AB;
       vst1q_f32(c0, vout0x4567); c0 += 4;
       vout0x4567 = vout0xCDEF;
       vst1q_f32(c1, vout1x4567); c1 += 4;
       vout1x4567 = vout1xCDEF;
       vst1q_f32(c2, vout2x4567); c2 += 4;
       vout2x4567 = vout2xCDEF;
       vst1q_f32(c3, vout3x4567); c3 += 4;
       vout3x4567 = vout3xCDEF;
       vst1q_f32(c4, vout4x4567); c4 += 4;
       vout4x4567 = vout4xCDEF;
     }
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
       vst1q_f32(c1, vout1x0123); c1 += 4;
       vout1x0123 = vout1x4567;
       vst1q_f32(c2, vout2x0123); c2 += 4;
       vout2x0123 = vout2x4567;
       vst1q_f32(c3, vout3x0123); c3 += 4;
       vout3x0123 = vout3x4567;
       vst1q_f32(c4, vout4x0123); c4 += 4;
       vout4x0123 = vout4x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     float32x2_t vout1x01 = vget_low_f32(vout1x0123);
     float32x2_t vout2x01 = vget_low_f32(vout2x0123);
     float32x2_t vout3x01 = vget_low_f32(vout3x0123);
     float32x2_t vout4x01 = vget_low_f32(vout4x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vst1_f32(c1, vout1x01); c1 += 2;
       vst1_f32(c2, vout2x01); c2 += 2;
       vst1_f32(c3, vout3x01); c3 += 2;
       vst1_f32(c4, vout4x01); c4 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
       vout1x01 = vget_high_f32(vout1x0123);
       vout2x01 = vget_high_f32(vout2x0123);
       vout3x01 = vget_high_f32(vout3x0123);
       vout4x01 = vget_high_f32(vout4x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
       vst1_lane_f32(c1, vout1x01, 0);
       vst1_lane_f32(c2, vout2x01, 0);
       vst1_lane_f32(c3, vout3x01, 0);
       vst1_lane_f32(c4, vout4x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
