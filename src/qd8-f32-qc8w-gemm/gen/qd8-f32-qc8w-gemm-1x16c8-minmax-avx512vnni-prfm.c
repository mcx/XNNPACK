// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c8-avx512vnni.c.in
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
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/prefetch.h"
#include "src/xnnpack/unaligned.h"


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point);
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32(w);
    __m512i vsum0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, vinput_zero_point0);
    __m512i vacc0x01234567 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum0x0123456789ABCDEF, 0));
    __m512i vacc0x89ABCDEF = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(vsum0x0123456789ABCDEF, 1));
    __m512i vacc1x0x01234567 = _mm512_setzero_epi32();
    __m512i vacc1x0x89ABCDEF = _mm512_setzero_epi32();
    w = (const int32_t*) w + 16;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m512i va0x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0));
      const __m512i va0x89ABCDEF = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0 + 8));
      a0 += 16;

      const __m512i vb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
      const __m512i vb01234567x89ABCDEF = _mm512_load_si512((const int8_t*) w + 128);
      const __m512i vb89ABCDEFx89ABCDEF = _mm512_load_si512((const int8_t*) w + 192);
      xnn_prefetch_to_l1((const int8_t*) w + 768);
      xnn_prefetch_to_l1((const int8_t*) w + 832);

      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
      xnn_prefetch_to_l1((const int8_t*) w + 896);
      xnn_prefetch_to_l1((const int8_t*) w + 960);
      vacc1x0x01234567 = _mm512_dpbusd_epi32(vacc1x0x01234567, va0x89ABCDEF, vb01234567x89ABCDEF);
      vacc1x0x89ABCDEF = _mm512_dpbusd_epi32(vacc1x0x89ABCDEF, va0x89ABCDEF, vb89ABCDEFx89ABCDEF);

      w = (const int8_t*) w + 256;
      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const __m512i va0x01234567 = _mm512_set1_epi64((int64_t) unaligned_load_u64(a0));
      a0 += 8;

      const __m512i vb01234567x01234567 = _mm512_load_si512(w);
      const __m512i vb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);

      vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
      vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
      xnn_prefetch_to_l1((const int8_t*) w + 896);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      w = (const int8_t*) w + 128;
      k -= 8 * sizeof(int8_t);
    }
    vacc0x01234567 = _mm512_add_epi32(vacc0x01234567, vacc1x0x01234567);
    vacc0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, vacc1x0x89ABCDEF);

    // Add adjacent pairs
    const __m512i vidx = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i vsum0x01234567 = _mm512_add_epi32(vacc0x01234567, _mm512_srli_epi64(vacc0x01234567, 32));
    const __m512i vsum0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, _mm512_srli_epi64(vacc0x89ABCDEF, 32));
    __m512i vacc0x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum0x01234567, vidx, vsum0x89ABCDEF);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
