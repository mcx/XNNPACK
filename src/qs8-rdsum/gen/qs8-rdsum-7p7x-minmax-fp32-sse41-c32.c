// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/sse41.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <smmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/unaligned.h"


void xnn_qs8_rdsum_ukernel_7p7x__sse41_c32(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 32; channels -= 32) {
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

    __m128i vacc0 = _mm_setzero_si128();
    __m128i vacc4 = _mm_setzero_si128();
    __m128i vacc8 = _mm_setzero_si128();
    __m128i vacc12 = _mm_setzero_si128();
    __m128i vacc16 = _mm_setzero_si128();
    __m128i vacc20 = _mm_setzero_si128();
    __m128i vacc24 = _mm_setzero_si128();
    __m128i vacc28 = _mm_setzero_si128();

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    while (r > 0) {
      __m128i vacc16_0 = _mm_setzero_si128();
      __m128i vacc16_8 = _mm_setzero_si128();
      __m128i vacc16_16 = _mm_setzero_si128();
      __m128i vacc16_24 = _mm_setzero_si128();
      for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
        if XNN_UNPREDICTABLE(current_batch < 2) {
          i1 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 2) {
          i2 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 4) {
          i3 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 4) {
          i4 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 6) {
          i5 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 6) {
          i6 = zero;
        }
        __m128i vin0;
        __m128i vin8;
        __m128i vin16;
        __m128i vin24;
        vin0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[0]));
        vin8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[8]));
        vin16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[16]));
        vin24 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[24]));
        vacc16_0 = _mm_add_epi16(vacc16_0, vin0);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin8);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin16);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin24);
        vin0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[0]));
        vin8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[8]));
        vin16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[16]));
        vin24 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[24]));
        vacc16_0 = _mm_add_epi16(vacc16_0, vin0);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin8);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin16);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin24);
        vin0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[0]));
        vin8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[8]));
        vin16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[16]));
        vin24 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[24]));
        vacc16_0 = _mm_add_epi16(vacc16_0, vin0);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin8);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin16);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin24);
        vin0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[0]));
        vin8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[8]));
        vin16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[16]));
        vin24 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[24]));
        vacc16_0 = _mm_add_epi16(vacc16_0, vin0);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin8);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin16);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin24);
        vin0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[0]));
        vin8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[8]));
        vin16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[16]));
        vin24 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[24]));
        vacc16_0 = _mm_add_epi16(vacc16_0, vin0);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin8);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin16);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin24);
        vin0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[0]));
        vin8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[8]));
        vin16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[16]));
        vin24 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[24]));
        vacc16_0 = _mm_add_epi16(vacc16_0, vin0);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin8);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin16);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin24);
        vin0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[0]));
        vin8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[8]));
        vin16 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[16]));
        vin24 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[24]));
        vacc16_0 = _mm_add_epi16(vacc16_0, vin0);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin8);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin16);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin24);
        i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc0 = _mm_add_epi32(vacc0, _mm_cvtepi16_epi32(vacc16_0));
      vacc4 = _mm_add_epi32(vacc4, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_0, 8)));
      vacc8 = _mm_add_epi32(vacc8, _mm_cvtepi16_epi32(vacc16_8));
      vacc12 = _mm_add_epi32(vacc12, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_8, 8)));
      vacc16 = _mm_add_epi32(vacc16, _mm_cvtepi16_epi32(vacc16_16));
      vacc20 = _mm_add_epi32(vacc20, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_16, 8)));
      vacc24 = _mm_add_epi32(vacc24, _mm_cvtepi16_epi32(vacc16_24));
      vacc28 = _mm_add_epi32(vacc28, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_24, 8)));
      r = doz(r, 252);
    }

    __m128i vo0 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 0 * sizeof(int32_t)));
    __m128i vo4 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 4 * sizeof(int32_t)));
    __m128i vo8 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 8 * sizeof(int32_t)));
    __m128i vo12 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 12 * sizeof(int32_t)));
    __m128i vo16 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 16 * sizeof(int32_t)));
    __m128i vo20 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 20 * sizeof(int32_t)));
    __m128i vo24 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 24 * sizeof(int32_t)));
    __m128i vo28 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 28 * sizeof(int32_t)));
    vo0 = _mm_add_epi32(vo0, vacc0);
    vo4 = _mm_add_epi32(vo4, vacc4);
    vo8 = _mm_add_epi32(vo8, vacc8);
    vo12 = _mm_add_epi32(vo12, vacc12);
    vo16 = _mm_add_epi32(vo16, vacc16);
    vo20 = _mm_add_epi32(vo20, vacc20);
    vo24 = _mm_add_epi32(vo24, vacc24);
    vo28 = _mm_add_epi32(vo28, vacc28);
    _mm_storeu_si128((__m128i*) output, vo0); output += 4;
    _mm_storeu_si128((__m128i*) output, vo4); output += 4;
    _mm_storeu_si128((__m128i*) output, vo8); output += 4;
    _mm_storeu_si128((__m128i*) output, vo12); output += 4;
    _mm_storeu_si128((__m128i*) output, vo16); output += 4;
    _mm_storeu_si128((__m128i*) output, vo20); output += 4;
    _mm_storeu_si128((__m128i*) output, vo24); output += 4;
    _mm_storeu_si128((__m128i*) output, vo28); output += 4;

    input = (const int8_t*) ((uintptr_t) input + 32 * sizeof(int8_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    // 256 int8s may be summed into an int16 before overflowing.
    do {
      int num_batches = floor((rows + 251) / 252);
      int r = rows;
      const int8_t* i0 = input;
      const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
      const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
      const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
      const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
      const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
      const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

      __m128i vacc0 = _mm_setzero_si128();
      __m128i vacc1 = _mm_setzero_si128();

      for (; num_batches > 0; --num_batches) {
        __m128i vacc16 = _mm_setzero_si128();
        for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
          if XNN_UNPREDICTABLE(current_batch < 2) {
            i1 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 2) {
            i2 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 4) {
            i3 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 4) {
            i4 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 6) {
            i5 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 6) {
            i6 = zero;
          }

          __m128i vin0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[0]));
          __m128i vin1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[0]));
          __m128i vin2 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[0]));
          __m128i vin3 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[0]));
          __m128i vin4 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[0]));
          __m128i vin5 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[0]));
          __m128i vin6 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[0]));
          vacc16 = _mm_add_epi16(vacc16, vin0);
          vacc16 = _mm_add_epi16(vacc16, vin1);
          vacc16 = _mm_add_epi16(vacc16, vin2);
          vacc16 = _mm_add_epi16(vacc16, vin3);
          vacc16 = _mm_add_epi16(vacc16, vin4);
          vacc16 = _mm_add_epi16(vacc16, vin5);
          vacc16 = _mm_add_epi16(vacc16, vin6);
          i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc0 = _mm_add_epi32(vacc0, _mm_cvtepi16_epi32(vacc16));
        vacc1 = _mm_add_epi32(vacc1, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16, 8)));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 8) {
        __m128i vo0 = _mm_loadu_si128((const __m128i*) output);
        __m128i vo1 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 4 * sizeof(int32_t)));
        vo0 = _mm_add_epi32(vo0, vacc0);
        vo1 = _mm_add_epi32(vo1, vacc1);
        _mm_storeu_si128((__m128i*) output, vo0); output += 4;
        _mm_storeu_si128((__m128i*) output, vo1); output += 4;
        channels -= 8;
        input = (const int8_t*) ((uintptr_t) input + 8 * sizeof(int8_t));
      } else {
        if (channels & 4) {
          __m128i vo = _mm_loadu_si128((const __m128i*) output);
          vo = _mm_add_epi32(vo, vacc0);
          _mm_storeu_si128((__m128i*) output, vo); output += 4;
          vacc0 = vacc1;
        }
        if (channels & 2) {
          __m128i vo = _mm_loadl_epi64((const __m128i*) output);
          vo = _mm_add_epi32(vo, vacc0);
          _mm_storel_epi64((__m128i*) output, vo); output += 2;
          vacc0 = _mm_srli_si128(vacc0, 8);
        }
        if (channels & 1) {
          __m128i vo = _mm_cvtsi32_si128(unaligned_load_s32(output));
          vo = _mm_add_epi32(vo, vacc0);
          _mm_storeu_si32(output, vo);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
