// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx4c8-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
  #include <intrin.h>
#else
  #include <x86intrin.h>
#endif

#include "src/xnnpack/common.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c8__avx_ld64(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const __m128 vmin = _mm_set1_ps(params->scalar.min);
  const __m128 vmax = _mm_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  const __m128i vinput_zero_point = _mm_castps_si128(_mm_broadcast_ss((const float*) &quantization_params->zero_point));
  const __m128 vinput_scale = _mm_broadcast_ss(&quantization_params->inv_scale);


  do {
    const __m128i vksum = _mm_load_si128((const __m128i*) w);
    const __m128i vzero = _mm_setzero_si128();
    const __m128i vinit0 = _mm_mullo_epi32(vksum, vinput_zero_point);
    const __m128i vinit1 = _mm_mullo_epi32(vksum, vinput_zero_point);
    const __m128i vinit2 = _mm_mullo_epi32(vksum, vinput_zero_point);
    const __m128i vinit3 = _mm_mullo_epi32(vksum, vinput_zero_point);
    __m128i vacc0x0 = _mm_blend_epi16(vinit0, vzero, 0xFC);
    __m128i vacc0x1 = _mm_blend_epi16(vinit0, vzero, 0xF3);
    __m128i vacc0x2 = _mm_blend_epi16(vinit0, vzero, 0xCF);
    __m128i vacc0x3 = _mm_blend_epi16(vinit0, vzero, 0x3F);
    __m128i vacc1x0 = _mm_blend_epi16(vinit1, vzero, 0xFC);
    __m128i vacc1x1 = _mm_blend_epi16(vinit1, vzero, 0xF3);
    __m128i vacc1x2 = _mm_blend_epi16(vinit1, vzero, 0xCF);
    __m128i vacc1x3 = _mm_blend_epi16(vinit1, vzero, 0x3F);
    __m128i vacc2x0 = _mm_blend_epi16(vinit2, vzero, 0xFC);
    __m128i vacc2x1 = _mm_blend_epi16(vinit2, vzero, 0xF3);
    __m128i vacc2x2 = _mm_blend_epi16(vinit2, vzero, 0xCF);
    __m128i vacc2x3 = _mm_blend_epi16(vinit2, vzero, 0x3F);
    __m128i vacc3x0 = _mm_blend_epi16(vinit3, vzero, 0xFC);
    __m128i vacc3x1 = _mm_blend_epi16(vinit3, vzero, 0xF3);
    __m128i vacc3x2 = _mm_blend_epi16(vinit3, vzero, 0xCF);
    __m128i vacc3x3 = _mm_blend_epi16(vinit3, vzero, 0x3F);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_cvtepi8_epi16(va2);
        a2 += 8;
        const __m128i va3 = _mm_loadl_epi64((const __m128i*) a3);
        const __m128i vxa3 = _mm_cvtepi8_epi16(va3);
        a3 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));
        vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
    const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
    const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
    const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
    const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);
    const __m128i vacc3x01 = _mm_hadd_epi32(vacc3x0, vacc3x1);
    const __m128i vacc3x23 = _mm_hadd_epi32(vacc3x2, vacc3x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
    __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);
    __m128i vacc3x0123 = _mm_hadd_epi32(vacc3x01, vacc3x23);

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);
    __m128 vout2x0123 = _mm_cvtepi32_ps(vacc2x0123);
    __m128 vout3x0123 = _mm_cvtepi32_ps(vacc3x0123);

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale);
    vout3x0123 = _mm_mul_ps(vout3x0123, vinput_scale);

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    vout0x0123 = _mm_mul_ps(vout0x0123, vfilter_output_scale0123);
    vout1x0123 = _mm_mul_ps(vout1x0123, vfilter_output_scale0123);
    vout2x0123 = _mm_mul_ps(vout2x0123, vfilter_output_scale0123);
    vout3x0123 = _mm_mul_ps(vout3x0123, vfilter_output_scale0123);

    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);
    vout3x0123 = _mm_add_ps(vout3x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);
    vout3x0123 = _mm_max_ps(vout3x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);
    vout3x0123 = _mm_min_ps(vout3x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c3, vout3x0123);
      _mm_storeu_ps(c2, vout2x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c0, vout0x0123);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c3, vout3x0123);
        vout3x0123 = _mm_unpackhi_ps(vout3x0123, vout3x0123);
        c3 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c3, vout3x0123);
        _mm_store_ss(c2, vout2x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c0, vout0x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}
