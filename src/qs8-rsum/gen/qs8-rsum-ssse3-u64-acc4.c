// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/ssse3.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <tmmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"

void xnn_qs8_rsum_ukernel__ssse3_u64_acc4(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  XNN_ALIGN(16) static const int8_t onemask_table[32] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  const __m128i vone = _mm_load_si128((const __m128i*) &onemask_table[0]);
  const __m128i vone_16 = _mm_srli_epi16(vone, 8);
  __m128i vacc0 = _mm_setzero_si128();
  __m128i vacc1 = _mm_setzero_si128();
  __m128i vacc2 = _mm_setzero_si128();
  __m128i vacc3 = _mm_setzero_si128();

  // 256 int8s may be summed into an int16 before overflowing.
  // Each register has 16 lanes and there are 4 accumulators so batch size is 16384
  for (; batch >= 16384; batch -= 16384) {
    __m128i vacc16_0 = _mm_setzero_si128();
    __m128i vacc16_1 = _mm_setzero_si128();
    __m128i vacc16_2 = _mm_setzero_si128();
    __m128i vacc16_3 = _mm_setzero_si128();
    for (size_t current_batch = 16384; current_batch > 0; current_batch -= 64) {
      const __m128i vt0 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt1 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt2 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt3 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      vacc16_0 = _mm_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt1);
      vacc16_2 = _mm_add_epi16(vacc16_2, vt2);
      vacc16_3 = _mm_add_epi16(vacc16_3, vt3);
    }
    vacc0 = _mm_add_epi32(vacc0, _mm_madd_epi16(vone_16, vacc16_0));
    vacc1 = _mm_add_epi32(vacc1, _mm_madd_epi16(vone_16, vacc16_1));
    vacc2 = _mm_add_epi32(vacc2, _mm_madd_epi16(vone_16, vacc16_2));
    vacc3 = _mm_add_epi32(vacc3, _mm_madd_epi16(vone_16, vacc16_3));
  }

  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 16384);
    __m128i vacc16_0 = _mm_setzero_si128();
    __m128i vacc16_1 = _mm_setzero_si128();
    __m128i vacc16_2 = _mm_setzero_si128();
    __m128i vacc16_3 = _mm_setzero_si128();
    for (; batch >= 64; batch -= 64) {
      const __m128i vt0 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt1 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt2 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt3 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      vacc16_0 = _mm_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt1);
      vacc16_2 = _mm_add_epi16(vacc16_2, vt2);
      vacc16_3 = _mm_add_epi16(vacc16_3, vt3);
    }
    vacc0 = _mm_add_epi32(vacc0, _mm_madd_epi16(vone_16, vacc16_0));
    vacc1 = _mm_add_epi32(vacc1, _mm_madd_epi16(vone_16, vacc16_1));
    vacc2 = _mm_add_epi32(vacc2, _mm_madd_epi16(vone_16, vacc16_2));
    vacc3 = _mm_add_epi32(vacc3, _mm_madd_epi16(vone_16, vacc16_3));
  }
  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 4096);
    __m128i vacc16 = _mm_setzero_si128();
    for (; batch >= 16; batch -= 16) {
      const __m128i vt = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      vacc16 = _mm_add_epi16(vacc16, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 15);
      const __m128i vonemask = _mm_loadu_si128((const __m128i*) &onemask_table[16 - batch]);
      const __m128i vt = _mm_maddubs_epi16(vonemask, _mm_loadu_si128((const __m128i*) input));
      vacc16 = _mm_add_epi16(vacc16, vt);
    }
    vacc0 = _mm_add_epi32(vacc0, _mm_madd_epi16(vone_16, vacc16));
  }
  vacc0 = _mm_add_epi32(vacc0, vacc1);
  vacc2 = _mm_add_epi32(vacc2, vacc3);
  vacc0 = _mm_add_epi32(vacc0, vacc2);

  __m128i vacc_lo = _mm_hadd_epi32(vacc0, vacc0);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);
  const int32_t vacc = _mm_cvtsi128_si32(vacc_lo);

  *output += vacc;
}
