// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qu8-rsum/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"

void xnn_qu8_rsum_ukernel__sse2_u64_acc4(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  XNN_ALIGN(16) static const int8_t mask_table[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  const __m128i vzero = _mm_setzero_si128();
  __m128i vacc0 = _mm_setzero_si128();
  __m128i vacc1 = _mm_setzero_si128();
  __m128i vacc2 = _mm_setzero_si128();
  __m128i vacc3 = _mm_setzero_si128();

  for (; batch >= 64; batch -= 64) {
    const __m128i vin0 = _mm_loadu_si128((const __m128i*) (input + 0));
    const __m128i vin1 = _mm_loadu_si128((const __m128i*) (input + 16));
    const __m128i vin2 = _mm_loadu_si128((const __m128i*) (input + 32));
    const __m128i vin3 = _mm_loadu_si128((const __m128i*) (input + 48));
    input += 64;
    const __m128i vt0 = _mm_sad_epu8(vin0, vzero);
    const __m128i vt1 = _mm_sad_epu8(vin1, vzero);
    const __m128i vt2 = _mm_sad_epu8(vin2, vzero);
    const __m128i vt3 = _mm_sad_epu8(vin3, vzero);
     vacc0 = _mm_add_epi32(vacc0, vt0);
     vacc1 = _mm_add_epi32(vacc1, vt1);
     vacc2 = _mm_add_epi32(vacc2, vt2);
     vacc3 = _mm_add_epi32(vacc3, vt3);
  }
  vacc0 = _mm_add_epi32(vacc0, vacc1);
  vacc2 = _mm_add_epi32(vacc2, vacc3);
  vacc0 = _mm_add_epi32(vacc0, vacc2);

  for (; batch >= 16; batch -= 16) {
    const __m128i vin = _mm_loadu_si128((const __m128i*) input);
    input += 16;
    const __m128i vt = _mm_sad_epu8(vin, vzero);
    vacc0 = _mm_add_epi32(vacc0, vt);
  }

  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch <= 15);
    const __m128i vmask = _mm_loadu_si128((const __m128i*) &mask_table[16 - batch]);
    const __m128i vt = _mm_sad_epu8(_mm_and_si128(_mm_loadu_si128((const __m128i*) input), vmask), vzero);
    vacc0 = _mm_add_epi32(vacc0, vt);
  }

  vacc0 = _mm_add_epi32(vacc0, _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(vacc0), _mm_castsi128_ps(vacc0))));
  vacc0 = _mm_add_epi32(vacc0, _mm_shuffle_epi32(vacc0, _MM_SHUFFLE(1, 1, 1, 1)));

  *output += (uint32_t)_mm_cvtsi128_si32(vacc0);
}
