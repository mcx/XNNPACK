// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avx2.c.in
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
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"

void xnn_qs8_rsum_ukernel__avx256skx_u32(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  XNN_ALIGN(32) static const int8_t onemask_table[64] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  const __m256i vone = _mm256_load_si256((const __m256i*) &onemask_table[0]);
  const __m256i vone_16 = _mm256_srli_epi16(vone, 8);
  __m256i vacc0 = _mm256_setzero_si256();

  // 256 int8s may be summed into an int16 before overflowing.
  // Each register has 32 lanes and there are 1 accumulators so batch size is 8192
  for (; batch >= 8192; batch -= 8192) {
    __m256i vacc16_0 = _mm256_setzero_si256();
    for (size_t current_batch = 8192; current_batch > 0; current_batch -= 32) {
      const __m256i vt0 = _mm256_maddubs_epi16(vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;
      vacc16_0 = _mm256_add_epi16(vacc16_0, vt0);
    }
    vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(vone_16, vacc16_0));
  }

  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 8192);
    __m256i vacc16 = _mm256_setzero_si256();
    for (; batch >= 32; batch -= 32) {
      const __m256i vt = _mm256_maddubs_epi16(vone, _mm256_loadu_si256((const __m256i*) input)); input += 32;
      vacc16 = _mm256_add_epi16(vacc16, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 31);
      // Prepare mask for valid 8-bit elements (depends on batch).
      const __mmask32 vmask = _cvtu32_mask32((UINT32_C(1) << batch) - 1);
      const __m256i vt = _mm256_maddubs_epi16(vone, _mm256_maskz_loadu_epi8(vmask, input));
      vacc16 = _mm256_add_epi16(vacc16, vt);
    }
    vacc0 = _mm256_add_epi32(vacc0, _mm256_madd_epi16(vone_16, vacc16));
  }

  __m128i vacc_lo = _mm_add_epi32(_mm256_castsi256_si128(vacc0), _mm256_extractf128_si256(vacc0, 1));
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);

  *output += _mm_cvtsi128_si32(vacc_lo);
}
