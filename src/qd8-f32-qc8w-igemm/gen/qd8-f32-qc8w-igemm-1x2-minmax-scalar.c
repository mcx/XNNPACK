// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/scalar.c.in
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
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#include "src/xnnpack/unaligned.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;


  do {
    const int32_t vksum0 = unaligned_indexed_load_s32(w, 0);
    const int32_t vksum1 = unaligned_indexed_load_s32(w, 1);
    const int32_t vinput_zero_point = quantization_params->zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point;
    int32_t vacc0x1 = vksum1 * vinput_zero_point;
    w = (const void*) ((const int32_t*) w + 2);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        w = (const void*) ((const int8_t*) w + 2);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;

    const float vinput_scale = quantization_params->inv_scale;
    vout0x0 *= vinput_scale;
    vout0x1 *= vinput_scale;

    const float vfilter_output_scale0 = unaligned_indexed_load_f32(w, 0);
    vout0x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = unaligned_indexed_load_f32(w, 1);
    vout0x1 *= vfilter_output_scale1;

    const float vbias0 = unaligned_indexed_load_f32(w, 2);
    vout0x0 += vbias0;
    const float vbias1 = unaligned_indexed_load_f32(w, 3);
    vout0x1 += vbias1;

    w = (const float*) w + 4;

    const float voutput_min = params->scalar.min;
    vout0x0 = math_max_f32(vout0x0, voutput_min);
    vout0x1 = math_max_f32(vout0x1, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = math_min_f32(vout0x0, voutput_max);
    vout0x1 = math_min_f32(vout0x1, voutput_max);

    if XNN_LIKELY(nc >= 2) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
