// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx8c8-avxvnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
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


void xnn_qd8_f16_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_qc4w_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 14);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  uint16_t* c6 = (uint16_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const int8_t* a7 = (const int8_t*) ((uintptr_t) a6 + a_stride);
  uint16_t* c7 = (uint16_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const int8_t* a8 = (const int8_t*) ((uintptr_t) a7 + a_stride);
  uint16_t* c8 = (uint16_t*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }
  const int8_t* a9 = (const int8_t*) ((uintptr_t) a8 + a_stride);
  uint16_t* c9 = (uint16_t*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    a9 = a8;
    c9 = c8;
  }
  const int8_t* a10 = (const int8_t*) ((uintptr_t) a9 + a_stride);
  uint16_t* c10 = (uint16_t*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    a10 = a9;
    c10 = c9;
  }
  const int8_t* a11 = (const int8_t*) ((uintptr_t) a10 + a_stride);
  uint16_t* c11 = (uint16_t*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    a11 = a10;
    c11 = c10;
  }
  const int8_t* a12 = (const int8_t*) ((uintptr_t) a11 + a_stride);
  uint16_t* c12 = (uint16_t*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    a12 = a11;
    c12 = c11;
  }
  const int8_t* a13 = (const int8_t*) ((uintptr_t) a12 + a_stride);
  uint16_t* c13 = (uint16_t*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 14) {
    a13 = a12;
    c13 = c12;
  }

  const __m256i vinput_zero_point0 = _mm256_set1_epi32((int) quantization_params[0].zero_point);
  const __m256i vinput_zero_point1 = _mm256_set1_epi32((int) quantization_params[1].zero_point);
  const __m256i vinput_zero_point2 = _mm256_set1_epi32((int) quantization_params[2].zero_point);
  const __m256i vinput_zero_point3 = _mm256_set1_epi32((int) quantization_params[3].zero_point);
  const __m256i vinput_zero_point4 = _mm256_set1_epi32((int) quantization_params[4].zero_point);
  const __m256i vinput_zero_point5 = _mm256_set1_epi32((int) quantization_params[5].zero_point);
  const __m256i vinput_zero_point6 = _mm256_set1_epi32((int) quantization_params[6].zero_point);
  const __m256i vinput_zero_point7 = _mm256_set1_epi32((int) quantization_params[7].zero_point);
  const __m256i vinput_zero_point8 = _mm256_set1_epi32((int) quantization_params[8].zero_point);
  const __m256i vinput_zero_point9 = _mm256_set1_epi32((int) quantization_params[9].zero_point);
  const __m256i vinput_zero_point10 = _mm256_set1_epi32((int) quantization_params[10].zero_point);
  const __m256i vinput_zero_point11 = _mm256_set1_epi32((int) quantization_params[11].zero_point);
  const __m256i vinput_zero_point12 = _mm256_set1_epi32((int) quantization_params[12].zero_point);
  const __m256i vinput_zero_point13 = _mm256_set1_epi32((int) quantization_params[13].zero_point);
  const __m256 voutput_min = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m256 voutput_max = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.max));
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  const __m256i vmask = _mm256_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  const __m256i vshl4 = _mm256_set1_epi64x(0x01020408);
  XNN_FORCE_REALIZATION(vshl4);
  do {
    const __m256i vksum01234567 = _mm256_load_si256(w);
    __m256i vsum0x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point0);
    __m256i vacc0x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum0x01234567, 0));
    __m256i vacc0x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum0x01234567, 1));
    __m256i vsum1x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point1);
    __m256i vacc1x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum1x01234567, 0));
    __m256i vacc1x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum1x01234567, 1));
    __m256i vsum2x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point2);
    __m256i vacc2x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum2x01234567, 0));
    __m256i vacc2x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum2x01234567, 1));
    __m256i vsum3x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point3);
    __m256i vacc3x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum3x01234567, 0));
    __m256i vacc3x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum3x01234567, 1));
    __m256i vsum4x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point4);
    __m256i vacc4x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum4x01234567, 0));
    __m256i vacc4x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum4x01234567, 1));
    __m256i vsum5x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point5);
    __m256i vacc5x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum5x01234567, 0));
    __m256i vacc5x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum5x01234567, 1));
    __m256i vsum6x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point6);
    __m256i vacc6x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum6x01234567, 0));
    __m256i vacc6x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum6x01234567, 1));
    __m256i vsum7x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point7);
    __m256i vacc7x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum7x01234567, 0));
    __m256i vacc7x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum7x01234567, 1));
    __m256i vsum8x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point8);
    __m256i vacc8x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum8x01234567, 0));
    __m256i vacc8x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum8x01234567, 1));
    __m256i vsum9x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point9);
    __m256i vacc9x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum9x01234567, 0));
    __m256i vacc9x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum9x01234567, 1));
    __m256i vsum10x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point10);
    __m256i vacc10x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum10x01234567, 0));
    __m256i vacc10x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum10x01234567, 1));
    __m256i vsum11x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point11);
    __m256i vacc11x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum11x01234567, 0));
    __m256i vacc11x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum11x01234567, 1));
    __m256i vsum12x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point12);
    __m256i vacc12x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum12x01234567, 0));
    __m256i vacc12x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum12x01234567, 1));
    __m256i vsum13x01234567 = _mm256_mullo_epi32(vksum01234567, vinput_zero_point13);
    __m256i vacc13x0123 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum13x01234567, 0));
    __m256i vacc13x4567 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(vsum13x01234567, 1));
    w = (const int32_t*) w + 8;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m256i va0x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
      const __m256i va0x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8));
      a0 += 16;
      const __m256i va1x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1));
      const __m256i va1x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8));
      a1 += 16;
      const __m256i va2x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2));
      const __m256i va2x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2 + 8));
      a2 += 16;
      const __m256i va3x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3));
      const __m256i va3x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3 + 8));
      a3 += 16;
      const __m256i va4x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4));
      const __m256i va4x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4 + 8));
      a4 += 16;
      const __m256i va5x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5));
      const __m256i va5x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5 + 8));
      a5 += 16;
      const __m256i va6x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a6));
      const __m256i va6x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a6 + 8));
      a6 += 16;
      const __m256i va7x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a7));
      const __m256i va7x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a7 + 8));
      a7 += 16;
      const __m256i va8x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a8));
      const __m256i va8x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a8 + 8));
      a8 += 16;
      const __m256i va9x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a9));
      const __m256i va9x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a9 + 8));
      a9 += 16;
      const __m256i va10x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a10));
      const __m256i va10x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a10 + 8));
      a10 += 16;
      const __m256i va11x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a11));
      const __m256i va11x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a11 + 8));
      a11 += 16;
      const __m256i va12x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a12));
      const __m256i va12x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a12 + 8));
      a12 += 16;
      const __m256i va13x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a13));
      const __m256i va13x89ABCDEF = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a13 + 8));
      a13 += 16;

      const __m256i vbb01234567x01234567 = _mm256_load_si256(w);
      const __m256i vbb89ABCDEFx01234567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vb01234567x0123 = _mm256_gf2p8affine_epi64_epi8(vbb01234567x01234567, vshl4, 0);
      const __m256i vb89ABCDEFx0123 = _mm256_gf2p8affine_epi64_epi8(vbb89ABCDEFx01234567, vshl4, 0);
      const __m256i vb01234567x4567 = _mm256_and_si256(vbb01234567x01234567, vmask);
      const __m256i vb89ABCDEFx4567 = _mm256_and_si256(vbb89ABCDEFx01234567, vmask);

      vacc0x0123 = _mm256_dpbusd_epi32(vacc0x0123, va0x01234567, vb01234567x0123);
      vacc0x4567 = _mm256_dpbusd_epi32(vacc0x4567, va0x01234567, vb89ABCDEFx0123);
      vacc1x0123 = _mm256_dpbusd_epi32(vacc1x0123, va1x01234567, vb01234567x0123);
      vacc1x4567 = _mm256_dpbusd_epi32(vacc1x4567, va1x01234567, vb89ABCDEFx0123);
      vacc2x0123 = _mm256_dpbusd_epi32(vacc2x0123, va2x01234567, vb01234567x0123);
      vacc2x4567 = _mm256_dpbusd_epi32(vacc2x4567, va2x01234567, vb89ABCDEFx0123);
      vacc3x0123 = _mm256_dpbusd_epi32(vacc3x0123, va3x01234567, vb01234567x0123);
      vacc3x4567 = _mm256_dpbusd_epi32(vacc3x4567, va3x01234567, vb89ABCDEFx0123);
      vacc4x0123 = _mm256_dpbusd_epi32(vacc4x0123, va4x01234567, vb01234567x0123);
      vacc4x4567 = _mm256_dpbusd_epi32(vacc4x4567, va4x01234567, vb89ABCDEFx0123);
      vacc5x0123 = _mm256_dpbusd_epi32(vacc5x0123, va5x01234567, vb01234567x0123);
      vacc5x4567 = _mm256_dpbusd_epi32(vacc5x4567, va5x01234567, vb89ABCDEFx0123);
      vacc6x0123 = _mm256_dpbusd_epi32(vacc6x0123, va6x01234567, vb01234567x0123);
      vacc6x4567 = _mm256_dpbusd_epi32(vacc6x4567, va6x01234567, vb89ABCDEFx0123);
      vacc7x0123 = _mm256_dpbusd_epi32(vacc7x0123, va7x01234567, vb01234567x0123);
      vacc7x4567 = _mm256_dpbusd_epi32(vacc7x4567, va7x01234567, vb89ABCDEFx0123);
      vacc8x0123 = _mm256_dpbusd_epi32(vacc8x0123, va8x01234567, vb01234567x0123);
      vacc8x4567 = _mm256_dpbusd_epi32(vacc8x4567, va8x01234567, vb89ABCDEFx0123);
      vacc9x0123 = _mm256_dpbusd_epi32(vacc9x0123, va9x01234567, vb01234567x0123);
      vacc9x4567 = _mm256_dpbusd_epi32(vacc9x4567, va9x01234567, vb89ABCDEFx0123);
      vacc10x0123 = _mm256_dpbusd_epi32(vacc10x0123, va10x01234567, vb01234567x0123);
      vacc10x4567 = _mm256_dpbusd_epi32(vacc10x4567, va10x01234567, vb89ABCDEFx0123);
      vacc11x0123 = _mm256_dpbusd_epi32(vacc11x0123, va11x01234567, vb01234567x0123);
      vacc11x4567 = _mm256_dpbusd_epi32(vacc11x4567, va11x01234567, vb89ABCDEFx0123);
      vacc12x0123 = _mm256_dpbusd_epi32(vacc12x0123, va12x01234567, vb01234567x0123);
      vacc12x4567 = _mm256_dpbusd_epi32(vacc12x4567, va12x01234567, vb89ABCDEFx0123);
      vacc13x0123 = _mm256_dpbusd_epi32(vacc13x0123, va13x01234567, vb01234567x0123);
      vacc13x4567 = _mm256_dpbusd_epi32(vacc13x4567, va13x01234567, vb89ABCDEFx0123);
      vacc0x0123 = _mm256_dpbusd_epi32(vacc0x0123, va0x89ABCDEF, vb01234567x4567);
      vacc0x4567 = _mm256_dpbusd_epi32(vacc0x4567, va0x89ABCDEF, vb89ABCDEFx4567);
      vacc1x0123 = _mm256_dpbusd_epi32(vacc1x0123, va1x89ABCDEF, vb01234567x4567);
      vacc1x4567 = _mm256_dpbusd_epi32(vacc1x4567, va1x89ABCDEF, vb89ABCDEFx4567);
      vacc2x0123 = _mm256_dpbusd_epi32(vacc2x0123, va2x89ABCDEF, vb01234567x4567);
      vacc2x4567 = _mm256_dpbusd_epi32(vacc2x4567, va2x89ABCDEF, vb89ABCDEFx4567);
      vacc3x0123 = _mm256_dpbusd_epi32(vacc3x0123, va3x89ABCDEF, vb01234567x4567);
      vacc3x4567 = _mm256_dpbusd_epi32(vacc3x4567, va3x89ABCDEF, vb89ABCDEFx4567);
      vacc4x0123 = _mm256_dpbusd_epi32(vacc4x0123, va4x89ABCDEF, vb01234567x4567);
      vacc4x4567 = _mm256_dpbusd_epi32(vacc4x4567, va4x89ABCDEF, vb89ABCDEFx4567);
      vacc5x0123 = _mm256_dpbusd_epi32(vacc5x0123, va5x89ABCDEF, vb01234567x4567);
      vacc5x4567 = _mm256_dpbusd_epi32(vacc5x4567, va5x89ABCDEF, vb89ABCDEFx4567);
      vacc6x0123 = _mm256_dpbusd_epi32(vacc6x0123, va6x89ABCDEF, vb01234567x4567);
      vacc6x4567 = _mm256_dpbusd_epi32(vacc6x4567, va6x89ABCDEF, vb89ABCDEFx4567);
      vacc7x0123 = _mm256_dpbusd_epi32(vacc7x0123, va7x89ABCDEF, vb01234567x4567);
      vacc7x4567 = _mm256_dpbusd_epi32(vacc7x4567, va7x89ABCDEF, vb89ABCDEFx4567);
      vacc8x0123 = _mm256_dpbusd_epi32(vacc8x0123, va8x89ABCDEF, vb01234567x4567);
      vacc8x4567 = _mm256_dpbusd_epi32(vacc8x4567, va8x89ABCDEF, vb89ABCDEFx4567);
      vacc9x0123 = _mm256_dpbusd_epi32(vacc9x0123, va9x89ABCDEF, vb01234567x4567);
      vacc9x4567 = _mm256_dpbusd_epi32(vacc9x4567, va9x89ABCDEF, vb89ABCDEFx4567);
      vacc10x0123 = _mm256_dpbusd_epi32(vacc10x0123, va10x89ABCDEF, vb01234567x4567);
      vacc10x4567 = _mm256_dpbusd_epi32(vacc10x4567, va10x89ABCDEF, vb89ABCDEFx4567);
      vacc11x0123 = _mm256_dpbusd_epi32(vacc11x0123, va11x89ABCDEF, vb01234567x4567);
      vacc11x4567 = _mm256_dpbusd_epi32(vacc11x4567, va11x89ABCDEF, vb89ABCDEFx4567);
      vacc12x0123 = _mm256_dpbusd_epi32(vacc12x0123, va12x89ABCDEF, vb01234567x4567);
      vacc12x4567 = _mm256_dpbusd_epi32(vacc12x4567, va12x89ABCDEF, vb89ABCDEFx4567);
      vacc13x0123 = _mm256_dpbusd_epi32(vacc13x0123, va13x89ABCDEF, vb01234567x4567);
      vacc13x4567 = _mm256_dpbusd_epi32(vacc13x4567, va13x89ABCDEF, vb89ABCDEFx4567);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const __m256i va0x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a0));
      a0 += 8;
      const __m256i va1x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a1));
      a1 += 8;
      const __m256i va2x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a2));
      a2 += 8;
      const __m256i va3x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a3));
      a3 += 8;
      const __m256i va4x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a4));
      a4 += 8;
      const __m256i va5x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a5));
      a5 += 8;
      const __m256i va6x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a6));
      a6 += 8;
      const __m256i va7x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a7));
      a7 += 8;
      const __m256i va8x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a8));
      a8 += 8;
      const __m256i va9x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a9));
      a9 += 8;
      const __m256i va10x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a10));
      a10 += 8;
      const __m256i va11x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a11));
      a11 += 8;
      const __m256i va12x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a12));
      a12 += 8;
      const __m256i va13x01234567 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(a13));
      a13 += 8;

      const __m256i vbb01234567x01234567 = _mm256_load_si256(w);
      const __m256i vbb89ABCDEFx01234567 = _mm256_load_si256((const __m256i*) ((const int8_t*) w + 32));
      const __m256i vb01234567x0123 = _mm256_gf2p8affine_epi64_epi8(vbb01234567x01234567, vshl4, 0);
      const __m256i vb89ABCDEFx0123 = _mm256_gf2p8affine_epi64_epi8(vbb89ABCDEFx01234567, vshl4, 0);

      vacc0x0123 = _mm256_dpbusd_epi32(vacc0x0123, va0x01234567, vb01234567x0123);
      vacc0x4567 = _mm256_dpbusd_epi32(vacc0x4567, va0x01234567, vb89ABCDEFx0123);
      vacc1x0123 = _mm256_dpbusd_epi32(vacc1x0123, va1x01234567, vb01234567x0123);
      vacc1x4567 = _mm256_dpbusd_epi32(vacc1x4567, va1x01234567, vb89ABCDEFx0123);
      vacc2x0123 = _mm256_dpbusd_epi32(vacc2x0123, va2x01234567, vb01234567x0123);
      vacc2x4567 = _mm256_dpbusd_epi32(vacc2x4567, va2x01234567, vb89ABCDEFx0123);
      vacc3x0123 = _mm256_dpbusd_epi32(vacc3x0123, va3x01234567, vb01234567x0123);
      vacc3x4567 = _mm256_dpbusd_epi32(vacc3x4567, va3x01234567, vb89ABCDEFx0123);
      vacc4x0123 = _mm256_dpbusd_epi32(vacc4x0123, va4x01234567, vb01234567x0123);
      vacc4x4567 = _mm256_dpbusd_epi32(vacc4x4567, va4x01234567, vb89ABCDEFx0123);
      vacc5x0123 = _mm256_dpbusd_epi32(vacc5x0123, va5x01234567, vb01234567x0123);
      vacc5x4567 = _mm256_dpbusd_epi32(vacc5x4567, va5x01234567, vb89ABCDEFx0123);
      vacc6x0123 = _mm256_dpbusd_epi32(vacc6x0123, va6x01234567, vb01234567x0123);
      vacc6x4567 = _mm256_dpbusd_epi32(vacc6x4567, va6x01234567, vb89ABCDEFx0123);
      vacc7x0123 = _mm256_dpbusd_epi32(vacc7x0123, va7x01234567, vb01234567x0123);
      vacc7x4567 = _mm256_dpbusd_epi32(vacc7x4567, va7x01234567, vb89ABCDEFx0123);
      vacc8x0123 = _mm256_dpbusd_epi32(vacc8x0123, va8x01234567, vb01234567x0123);
      vacc8x4567 = _mm256_dpbusd_epi32(vacc8x4567, va8x01234567, vb89ABCDEFx0123);
      vacc9x0123 = _mm256_dpbusd_epi32(vacc9x0123, va9x01234567, vb01234567x0123);
      vacc9x4567 = _mm256_dpbusd_epi32(vacc9x4567, va9x01234567, vb89ABCDEFx0123);
      vacc10x0123 = _mm256_dpbusd_epi32(vacc10x0123, va10x01234567, vb01234567x0123);
      vacc10x4567 = _mm256_dpbusd_epi32(vacc10x4567, va10x01234567, vb89ABCDEFx0123);
      vacc11x0123 = _mm256_dpbusd_epi32(vacc11x0123, va11x01234567, vb01234567x0123);
      vacc11x4567 = _mm256_dpbusd_epi32(vacc11x4567, va11x01234567, vb89ABCDEFx0123);
      vacc12x0123 = _mm256_dpbusd_epi32(vacc12x0123, va12x01234567, vb01234567x0123);
      vacc12x4567 = _mm256_dpbusd_epi32(vacc12x4567, va12x01234567, vb89ABCDEFx0123);
      vacc13x0123 = _mm256_dpbusd_epi32(vacc13x0123, va13x01234567, vb01234567x0123);
      vacc13x4567 = _mm256_dpbusd_epi32(vacc13x4567, va13x01234567, vb89ABCDEFx0123);

      w = (const int8_t*) w + 64;
      k -= 8 * sizeof(int8_t);
    }

    // Add adjacent pairs
    const __m256i vsum0x02134657 = _mm256_hadd_epi32(vacc0x0123, vacc0x4567);
    __m256i vacc0x01234567 = _mm256_permute4x64_epi64(vsum0x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum1x02134657 = _mm256_hadd_epi32(vacc1x0123, vacc1x4567);
    __m256i vacc1x01234567 = _mm256_permute4x64_epi64(vsum1x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum2x02134657 = _mm256_hadd_epi32(vacc2x0123, vacc2x4567);
    __m256i vacc2x01234567 = _mm256_permute4x64_epi64(vsum2x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum3x02134657 = _mm256_hadd_epi32(vacc3x0123, vacc3x4567);
    __m256i vacc3x01234567 = _mm256_permute4x64_epi64(vsum3x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum4x02134657 = _mm256_hadd_epi32(vacc4x0123, vacc4x4567);
    __m256i vacc4x01234567 = _mm256_permute4x64_epi64(vsum4x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum5x02134657 = _mm256_hadd_epi32(vacc5x0123, vacc5x4567);
    __m256i vacc5x01234567 = _mm256_permute4x64_epi64(vsum5x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum6x02134657 = _mm256_hadd_epi32(vacc6x0123, vacc6x4567);
    __m256i vacc6x01234567 = _mm256_permute4x64_epi64(vsum6x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum7x02134657 = _mm256_hadd_epi32(vacc7x0123, vacc7x4567);
    __m256i vacc7x01234567 = _mm256_permute4x64_epi64(vsum7x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum8x02134657 = _mm256_hadd_epi32(vacc8x0123, vacc8x4567);
    __m256i vacc8x01234567 = _mm256_permute4x64_epi64(vsum8x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum9x02134657 = _mm256_hadd_epi32(vacc9x0123, vacc9x4567);
    __m256i vacc9x01234567 = _mm256_permute4x64_epi64(vsum9x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum10x02134657 = _mm256_hadd_epi32(vacc10x0123, vacc10x4567);
    __m256i vacc10x01234567 = _mm256_permute4x64_epi64(vsum10x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum11x02134657 = _mm256_hadd_epi32(vacc11x0123, vacc11x4567);
    __m256i vacc11x01234567 = _mm256_permute4x64_epi64(vsum11x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum12x02134657 = _mm256_hadd_epi32(vacc12x0123, vacc12x4567);
    __m256i vacc12x01234567 = _mm256_permute4x64_epi64(vsum12x02134657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m256i vsum13x02134657 = _mm256_hadd_epi32(vacc13x0123, vacc13x4567);
    __m256i vacc13x01234567 = _mm256_permute4x64_epi64(vsum13x02134657, _MM_SHUFFLE(3, 1, 2, 0));

    vacc0x01234567 = _mm256_srai_epi32(vacc0x01234567, 4);
    vacc1x01234567 = _mm256_srai_epi32(vacc1x01234567, 4);
    vacc2x01234567 = _mm256_srai_epi32(vacc2x01234567, 4);
    vacc3x01234567 = _mm256_srai_epi32(vacc3x01234567, 4);
    vacc4x01234567 = _mm256_srai_epi32(vacc4x01234567, 4);
    vacc5x01234567 = _mm256_srai_epi32(vacc5x01234567, 4);
    vacc6x01234567 = _mm256_srai_epi32(vacc6x01234567, 4);
    vacc7x01234567 = _mm256_srai_epi32(vacc7x01234567, 4);
    vacc8x01234567 = _mm256_srai_epi32(vacc8x01234567, 4);
    vacc9x01234567 = _mm256_srai_epi32(vacc9x01234567, 4);
    vacc10x01234567 = _mm256_srai_epi32(vacc10x01234567, 4);
    vacc11x01234567 = _mm256_srai_epi32(vacc11x01234567, 4);
    vacc12x01234567 = _mm256_srai_epi32(vacc12x01234567, 4);
    vacc13x01234567 = _mm256_srai_epi32(vacc13x01234567, 4);
    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vout1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vout2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);
    __m256 vout3x01234567 = _mm256_cvtepi32_ps(vacc3x01234567);
    __m256 vout4x01234567 = _mm256_cvtepi32_ps(vacc4x01234567);
    __m256 vout5x01234567 = _mm256_cvtepi32_ps(vacc5x01234567);
    __m256 vout6x01234567 = _mm256_cvtepi32_ps(vacc6x01234567);
    __m256 vout7x01234567 = _mm256_cvtepi32_ps(vacc7x01234567);
    __m256 vout8x01234567 = _mm256_cvtepi32_ps(vacc8x01234567);
    __m256 vout9x01234567 = _mm256_cvtepi32_ps(vacc9x01234567);
    __m256 vout10x01234567 = _mm256_cvtepi32_ps(vacc10x01234567);
    __m256 vout11x01234567 = _mm256_cvtepi32_ps(vacc11x01234567);
    __m256 vout12x01234567 = _mm256_cvtepi32_ps(vacc12x01234567);
    __m256 vout13x01234567 = _mm256_cvtepi32_ps(vacc13x01234567);

    vout0x01234567 = _mm256_mul_ps(vout0x01234567, _mm256_set1_ps(quantization_params[0].inv_scale));
    vout1x01234567 = _mm256_mul_ps(vout1x01234567, _mm256_set1_ps(quantization_params[1].inv_scale));
    vout2x01234567 = _mm256_mul_ps(vout2x01234567, _mm256_set1_ps(quantization_params[2].inv_scale));
    vout3x01234567 = _mm256_mul_ps(vout3x01234567, _mm256_set1_ps(quantization_params[3].inv_scale));
    vout4x01234567 = _mm256_mul_ps(vout4x01234567, _mm256_set1_ps(quantization_params[4].inv_scale));
    vout5x01234567 = _mm256_mul_ps(vout5x01234567, _mm256_set1_ps(quantization_params[5].inv_scale));
    vout6x01234567 = _mm256_mul_ps(vout6x01234567, _mm256_set1_ps(quantization_params[6].inv_scale));
    vout7x01234567 = _mm256_mul_ps(vout7x01234567, _mm256_set1_ps(quantization_params[7].inv_scale));
    vout8x01234567 = _mm256_mul_ps(vout8x01234567, _mm256_set1_ps(quantization_params[8].inv_scale));
    vout9x01234567 = _mm256_mul_ps(vout9x01234567, _mm256_set1_ps(quantization_params[9].inv_scale));
    vout10x01234567 = _mm256_mul_ps(vout10x01234567, _mm256_set1_ps(quantization_params[10].inv_scale));
    vout11x01234567 = _mm256_mul_ps(vout11x01234567, _mm256_set1_ps(quantization_params[11].inv_scale));
    vout12x01234567 = _mm256_mul_ps(vout12x01234567, _mm256_set1_ps(quantization_params[12].inv_scale));
    vout13x01234567 = _mm256_mul_ps(vout13x01234567, _mm256_set1_ps(quantization_params[13].inv_scale));

    const __m256 vfilter_output_scale01234567 = _mm256_load_ps((const float*) w);
    const __m256 vbias01234567 = _mm256_load_ps((const float*) w + 8);
    w = (const float*) w + 16;

    vout0x01234567 = _mm256_fmadd_ps(vout0x01234567, vfilter_output_scale01234567, vbias01234567);
    vout1x01234567 = _mm256_fmadd_ps(vout1x01234567, vfilter_output_scale01234567, vbias01234567);
    vout2x01234567 = _mm256_fmadd_ps(vout2x01234567, vfilter_output_scale01234567, vbias01234567);
    vout3x01234567 = _mm256_fmadd_ps(vout3x01234567, vfilter_output_scale01234567, vbias01234567);
    vout4x01234567 = _mm256_fmadd_ps(vout4x01234567, vfilter_output_scale01234567, vbias01234567);
    vout5x01234567 = _mm256_fmadd_ps(vout5x01234567, vfilter_output_scale01234567, vbias01234567);
    vout6x01234567 = _mm256_fmadd_ps(vout6x01234567, vfilter_output_scale01234567, vbias01234567);
    vout7x01234567 = _mm256_fmadd_ps(vout7x01234567, vfilter_output_scale01234567, vbias01234567);
    vout8x01234567 = _mm256_fmadd_ps(vout8x01234567, vfilter_output_scale01234567, vbias01234567);
    vout9x01234567 = _mm256_fmadd_ps(vout9x01234567, vfilter_output_scale01234567, vbias01234567);
    vout10x01234567 = _mm256_fmadd_ps(vout10x01234567, vfilter_output_scale01234567, vbias01234567);
    vout11x01234567 = _mm256_fmadd_ps(vout11x01234567, vfilter_output_scale01234567, vbias01234567);
    vout12x01234567 = _mm256_fmadd_ps(vout12x01234567, vfilter_output_scale01234567, vbias01234567);
    vout13x01234567 = _mm256_fmadd_ps(vout13x01234567, vfilter_output_scale01234567, vbias01234567);

    vout0x01234567 = _mm256_max_ps(vout0x01234567, voutput_min);
    vout1x01234567 = _mm256_max_ps(vout1x01234567, voutput_min);
    vout2x01234567 = _mm256_max_ps(vout2x01234567, voutput_min);
    vout3x01234567 = _mm256_max_ps(vout3x01234567, voutput_min);
    vout4x01234567 = _mm256_max_ps(vout4x01234567, voutput_min);
    vout5x01234567 = _mm256_max_ps(vout5x01234567, voutput_min);
    vout6x01234567 = _mm256_max_ps(vout6x01234567, voutput_min);
    vout7x01234567 = _mm256_max_ps(vout7x01234567, voutput_min);
    vout8x01234567 = _mm256_max_ps(vout8x01234567, voutput_min);
    vout9x01234567 = _mm256_max_ps(vout9x01234567, voutput_min);
    vout10x01234567 = _mm256_max_ps(vout10x01234567, voutput_min);
    vout11x01234567 = _mm256_max_ps(vout11x01234567, voutput_min);
    vout12x01234567 = _mm256_max_ps(vout12x01234567, voutput_min);
    vout13x01234567 = _mm256_max_ps(vout13x01234567, voutput_min);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, voutput_max);
    vout1x01234567 = _mm256_min_ps(vout1x01234567, voutput_max);
    vout2x01234567 = _mm256_min_ps(vout2x01234567, voutput_max);
    vout3x01234567 = _mm256_min_ps(vout3x01234567, voutput_max);
    vout4x01234567 = _mm256_min_ps(vout4x01234567, voutput_max);
    vout5x01234567 = _mm256_min_ps(vout5x01234567, voutput_max);
    vout6x01234567 = _mm256_min_ps(vout6x01234567, voutput_max);
    vout7x01234567 = _mm256_min_ps(vout7x01234567, voutput_max);
    vout8x01234567 = _mm256_min_ps(vout8x01234567, voutput_max);
    vout9x01234567 = _mm256_min_ps(vout9x01234567, voutput_max);
    vout10x01234567 = _mm256_min_ps(vout10x01234567, voutput_max);
    vout11x01234567 = _mm256_min_ps(vout11x01234567, voutput_max);
    vout12x01234567 = _mm256_min_ps(vout12x01234567, voutput_max);
    vout13x01234567 = _mm256_min_ps(vout13x01234567, voutput_max);

    __m128i vfp16out0x01234567 = _mm256_cvtps_ph(vout0x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out1x01234567 = _mm256_cvtps_ph(vout1x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out2x01234567 = _mm256_cvtps_ph(vout2x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out3x01234567 = _mm256_cvtps_ph(vout3x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out4x01234567 = _mm256_cvtps_ph(vout4x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out5x01234567 = _mm256_cvtps_ph(vout5x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out6x01234567 = _mm256_cvtps_ph(vout6x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out7x01234567 = _mm256_cvtps_ph(vout7x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out8x01234567 = _mm256_cvtps_ph(vout8x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out9x01234567 = _mm256_cvtps_ph(vout9x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out10x01234567 = _mm256_cvtps_ph(vout10x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out11x01234567 = _mm256_cvtps_ph(vout11x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out12x01234567 = _mm256_cvtps_ph(vout12x01234567, _MM_FROUND_TO_NEAREST_INT);
    __m128i vfp16out13x01234567 = _mm256_cvtps_ph(vout13x01234567, _MM_FROUND_TO_NEAREST_INT);
    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_si128((__m128i*) c0, vfp16out0x01234567);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_si128((__m128i*) c1, vfp16out1x01234567);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) c2, vfp16out2x01234567);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) c3, vfp16out3x01234567);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) c4, vfp16out4x01234567);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) c5, vfp16out5x01234567);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) c6, vfp16out6x01234567);
      a6 = (const int8_t*) ((uintptr_t) a6 - kc);
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);
      _mm_storeu_si128((__m128i*) c7, vfp16out7x01234567);
      a7 = (const int8_t*) ((uintptr_t) a7 - kc);
      c7 = (uint16_t*) ((uintptr_t) c7 + cn_stride);
      _mm_storeu_si128((__m128i*) c8, vfp16out8x01234567);
      a8 = (const int8_t*) ((uintptr_t) a8 - kc);
      c8 = (uint16_t*) ((uintptr_t) c8 + cn_stride);
      _mm_storeu_si128((__m128i*) c9, vfp16out9x01234567);
      a9 = (const int8_t*) ((uintptr_t) a9 - kc);
      c9 = (uint16_t*) ((uintptr_t) c9 + cn_stride);
      _mm_storeu_si128((__m128i*) c10, vfp16out10x01234567);
      a10 = (const int8_t*) ((uintptr_t) a10 - kc);
      c10 = (uint16_t*) ((uintptr_t) c10 + cn_stride);
      _mm_storeu_si128((__m128i*) c11, vfp16out11x01234567);
      a11 = (const int8_t*) ((uintptr_t) a11 - kc);
      c11 = (uint16_t*) ((uintptr_t) c11 + cn_stride);
      _mm_storeu_si128((__m128i*) c12, vfp16out12x01234567);
      a12 = (const int8_t*) ((uintptr_t) a12 - kc);
      c12 = (uint16_t*) ((uintptr_t) c12 + cn_stride);
      _mm_storeu_si128((__m128i*) c13, vfp16out13x01234567);
      a13 = (const int8_t*) ((uintptr_t) a13 - kc);
      c13 = (uint16_t*) ((uintptr_t) c13 + cn_stride);
      nc -= 8;
    } else {
      // Prepare mask for valid 16-bit elements (depends on nc).
      const __mmask8 vmask = _cvtu32_mask8((UINT32_C(1) << nc) - 1);
      _mm_mask_storeu_epi16(c0, vmask, vfp16out0x01234567);
      _mm_mask_storeu_epi16(c1, vmask, vfp16out1x01234567);
      _mm_mask_storeu_epi16(c2, vmask, vfp16out2x01234567);
      _mm_mask_storeu_epi16(c3, vmask, vfp16out3x01234567);
      _mm_mask_storeu_epi16(c4, vmask, vfp16out4x01234567);
      _mm_mask_storeu_epi16(c5, vmask, vfp16out5x01234567);
      _mm_mask_storeu_epi16(c6, vmask, vfp16out6x01234567);
      _mm_mask_storeu_epi16(c7, vmask, vfp16out7x01234567);
      _mm_mask_storeu_epi16(c8, vmask, vfp16out8x01234567);
      _mm_mask_storeu_epi16(c9, vmask, vfp16out9x01234567);
      _mm_mask_storeu_epi16(c10, vmask, vfp16out10x01234567);
      _mm_mask_storeu_epi16(c11, vmask, vfp16out11x01234567);
      _mm_mask_storeu_epi16(c12, vmask, vfp16out12x01234567);
      _mm_mask_storeu_epi16(c13, vmask, vfp16out13x01234567);
      nc = 0;
    }
  } while (nc != 0);
}
