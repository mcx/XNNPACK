// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx8c8-avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
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
#include "src/xnnpack/unaligned.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx256skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const __m256 voutput_max_less_zero_point = _mm256_set1_ps((int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point);
  const __m256i voutput_zero_point = _mm256_set1_epi16(params->fp32_scalar.output_zero_point);
  const __m256i voutput_min = _mm256_set1_epi8(params->fp32_scalar.output_min);
  XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);

  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    __m256i vacc1x01 = vacc0x01;
    __m256i vacc1x23 = vacc0x23;
    __m256i vacc1x45 = vacc0x45;
    __m256i vacc1x67 = vacc0x67;
    w = (const int32_t*) w + 8;

    size_t k = kc;

    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
      const __m256i vxa1 = _mm256_cvtepi8_epi16(va1);
      a1 += 8;

      const __m256i vxb01 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*) w));

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
      const __m256i vxb23 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*) ((const int8_t*) w + 16)));

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
      const __m256i vxb45 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*) ((const int8_t*) w + 32)));

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
      const __m256i vxb67 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*) ((const int8_t*) w + 48)));

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
      vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));

      w = (const int8_t*) w + 64;
      k -= 8 * sizeof(int8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);

    __m256 vfpacc0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vfpacc1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);

    const __m256 vscale01234567 = _mm256_load_ps(w);
    w = (const float*) w + 8;
    vfpacc0x01234567 = _mm256_mul_ps(vfpacc0x01234567, vscale01234567);
    vfpacc1x01234567 = _mm256_mul_ps(vfpacc1x01234567, vscale01234567);

    vfpacc0x01234567 = _mm256_min_ps(vfpacc0x01234567, voutput_max_less_zero_point);
    vfpacc1x01234567 = _mm256_min_ps(vfpacc1x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vfpacc0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vfpacc1x01234567);

    __m256i vacc01x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc1x01234567), voutput_zero_point);

    vacc01x01234567 = _mm256_permute4x64_epi64(vacc01x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packs_epi16(vacc01x01234567, vacc01x01234567);

    vout = _mm256_max_epi8(vout, voutput_min);

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);
      _mm_storel_epi64((__m128i*) c1, vout_hi);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);
        _mm_storeu_si32(c1, vout_hi);

        c0 += 4;
        c1 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout_hi, 0));

        c0 += 2;
        c1 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (int8_t) _mm_extract_epi8(vout_lo, 0);
        *c1 = (int8_t) _mm_extract_epi8(vout_hi, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
