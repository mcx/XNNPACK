// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx16c4-avx512vnni.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x16c4__avx512vnni(
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
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  float* c0 = c;

  const __m512i vinput_zero_point = _mm512_set1_epi32((int) quantization_params->zero_point);
  const __m512 vinput_inv_scale = _mm512_set1_ps(quantization_params->inv_scale);
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  // XNN_FORCE_REALIZATION(vinput_inv_scale);
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32(w);
    __m512i vacc0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point);
    __m512i vacc1x0x0123456789ABCDEF = _mm512_setzero_epi32();
    w = (const int32_t*) w + 16;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = kc;
      while (k >= 8 * sizeof(int8_t)) {
        const __m512i va0x0123 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
        const __m512i va0x4567 = _mm512_set1_epi32((int) unaligned_load_u32(a0 + 4));
        a0 += 8;

        const __m512i vb0123456789ABCDEFx0123 = _mm512_load_si512(w);
        const __m512i vb0123456789ABCDEFx4567 = _mm512_load_si512((const int8_t*) w + 64);

        vacc0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x0123456789ABCDEF, va0x0123, vb0123456789ABCDEFx0123);
        vacc1x0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x0x0123456789ABCDEF, va0x4567, vb0123456789ABCDEFx4567);

        w = (const int8_t*) w + 128;
        k -= 8 * sizeof(int8_t);
      }

      if (k != 0) {
        const __m512i va0x0123 = _mm512_set1_epi32((int) unaligned_load_u32(a0));
        a0 += 4;

        const __m512i vb0123456789ABCDEF = _mm512_load_si512(w);

        vacc0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x0123456789ABCDEF, va0x0123, vb0123456789ABCDEF);

        w = (const int8_t*) w + 64;
        k -= 4 * sizeof(int8_t);
      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, vacc1x0x0123456789ABCDEF);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vinput_inv_scale);

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
