// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#include "src/xnnpack/unaligned.h"


void xnn_qu8_gemm_minmax_rndnu_ukernel_2x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const int32_t vmultiplier = params->rndnu_scalar.multiplier;
  const int64_t vrounding = params->rndnu_scalar.rounding;
  const uint32_t vshift = params->rndnu_scalar.shift;
  const int32_t voutput_min_less_zero_point = (int32_t) params->rndnu_scalar.output_min - (int32_t) params->rndnu_scalar.output_zero_point;
  const int32_t voutput_max_less_zero_point = (int32_t) params->rndnu_scalar.output_max - (int32_t) params->rndnu_scalar.output_zero_point;
  const int32_t voutput_zero_point = params->rndnu_scalar.output_zero_point;

  const int32_t vb_zero_point = params->rndnu_scalar.kernel_zero_point;
  do {
    int32_t vacc0x0 = unaligned_indexed_load_s32(w, 0);
    int32_t vacc0x1 = unaligned_indexed_load_s32(w, 1);
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    w = (const int32_t*) w + 2;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) (uint32_t) *a0++;
      const int32_t va1 = (int32_t) (uint32_t) *a1++;

      const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
      const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
      w = (const uint8_t*) w + 2;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;

      k -= sizeof(uint8_t);
    } while (k != 0);

    const int64_t vextacc0x0 = math_mulext_s32(vacc0x0, vmultiplier) + vrounding;
    const int64_t vextacc0x1 = math_mulext_s32(vacc0x1, vmultiplier) + vrounding;
    const int64_t vextacc1x0 = math_mulext_s32(vacc1x0, vmultiplier) + vrounding;
    const int64_t vextacc1x1 = math_mulext_s32(vacc1x1, vmultiplier) + vrounding;

    int32_t vout0x0 = (int32_t) math_asr_s64(vextacc0x0, vshift);
    int32_t vout0x1 = (int32_t) math_asr_s64(vextacc0x1, vshift);
    int32_t vout1x0 = (int32_t) math_asr_s64(vextacc1x0, vshift);
    int32_t vout1x1 = (int32_t) math_asr_s64(vextacc1x1, vshift);

    vout0x0 = math_max_s32(vout0x0, voutput_min_less_zero_point);
    vout0x1 = math_max_s32(vout0x1, voutput_min_less_zero_point);
    vout1x0 = math_max_s32(vout1x0, voutput_min_less_zero_point);
    vout1x1 = math_max_s32(vout1x1, voutput_min_less_zero_point);

    vout0x0 = math_min_s32(vout0x0, voutput_max_less_zero_point);
    vout0x1 = math_min_s32(vout0x1, voutput_max_less_zero_point);
    vout1x0 = math_min_s32(vout1x0, voutput_max_less_zero_point);
    vout1x1 = math_min_s32(vout1x1, voutput_max_less_zero_point);

    vout0x0 += voutput_zero_point;
    vout0x1 += voutput_zero_point;
    vout1x0 += voutput_zero_point;
    vout1x1 += voutput_zero_point;

    if XNN_LIKELY(nc >= 2) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c1[0] = (uint8_t) vout1x0;
      c1[1] = (uint8_t) vout1x1;

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
        c1[0] = (uint8_t) vout1x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
