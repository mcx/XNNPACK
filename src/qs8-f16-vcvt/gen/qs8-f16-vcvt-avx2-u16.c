// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-f16-vcvt/avx2.c.in
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
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"


void xnn_qs8_f16_vcvt_ukernel__avx2_u16(
    size_t batch,
    const int8_t* input,
    xnn_float16* output,
    const struct xnn_qs8_f16_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  const __m256i vzero_point = _mm256_set1_epi32(params->scalar.zero_point);
  const __m256 vscale = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.scale));
  XNN_FORCE_REALIZATION(vzero_point);
  XNN_FORCE_REALIZATION(vscale);
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m256i vx01234567 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input));
    __m256i vx89ABCDEF = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (input + 8)));
    input += 16;

    vx01234567 = _mm256_sub_epi32(vx01234567, vzero_point);
    vx89ABCDEF = _mm256_sub_epi32(vx89ABCDEF, vzero_point);

    __m256 vy01234567 = _mm256_cvtepi32_ps(vx01234567);
    __m256 vy89ABCDEF = _mm256_cvtepi32_ps(vx89ABCDEF);

    vy01234567 = _mm256_mul_ps(vy01234567, vscale);
    vy89ABCDEF = _mm256_mul_ps(vy89ABCDEF, vscale);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8) , _mm256_cvtps_ph(vy89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    __m256i vx = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input));
    vx = _mm256_sub_epi32(vx, vzero_point);
    input += 8;

    __m256 vy = _mm256_cvtepi32_ps(vx);
    vy = _mm256_mul_ps(vy, vscale);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    __m256i vx = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input));
    vx = _mm256_sub_epi32(vx, vzero_point);

    __m256 vy = _mm256_cvtepi32_ps(vx);
    vy = _mm256_mul_ps(vy, vscale);

    __m128i vhy = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(int8_t))) {
      _mm_storel_epi64((__m128i*) o, vhy);
      vhy = _mm_unpackhi_epi64(vhy, vhy);
      o += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      _mm_storeu_si32((__m64*) o, vhy);
      vhy = _mm_srli_epi64(vhy, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *o = (uint16_t) _mm_extract_epi16(vhy, 0);
    }
  }
}

