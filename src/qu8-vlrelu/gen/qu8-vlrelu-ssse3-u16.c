// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/ssse3.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <tmmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"
#include "src/xnnpack/unaligned.h"


void xnn_qu8_vlrelu_ukernel__ssse3_u16(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_qu8_lrelu_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier_diff = _mm_set1_epi16(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const __m128i vmultiplier_base = _mm_set1_epi16(-params->scalar.negative_multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier_diff);
  XNN_FORCE_REALIZATION(vmultiplier_base);
  XNN_FORCE_REALIZATION(voutput_zero_point);

  const __m128i vzero = _mm_setzero_si128();
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vzero);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vzero);
    __m128i vmultiplier_lo = _mm_cmpgt_epi16(vacc_lo, vinput_zero_point);
    __m128i vmultiplier_hi = _mm_cmpgt_epi16(vacc_hi, vinput_zero_point);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vmultiplier_lo = _mm_and_si128(vmultiplier_lo, vmultiplier_diff);
    vmultiplier_hi = _mm_and_si128(vmultiplier_hi, vmultiplier_diff);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vmultiplier_lo = _mm_xor_si128(vmultiplier_lo, vmultiplier_base);
    vmultiplier_hi = _mm_xor_si128(vmultiplier_hi, vmultiplier_base);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier_lo);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier_hi);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    const __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vzero);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vzero);
    __m128i vmultiplier_lo = _mm_cmpgt_epi16(vacc_lo, vinput_zero_point);
    __m128i vmultiplier_hi = _mm_cmpgt_epi16(vacc_hi, vinput_zero_point);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vmultiplier_lo = _mm_and_si128(vmultiplier_lo, vmultiplier_diff);
    vmultiplier_hi = _mm_and_si128(vmultiplier_hi, vmultiplier_diff);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vmultiplier_lo = _mm_xor_si128(vmultiplier_lo, vmultiplier_base);
    vmultiplier_hi = _mm_xor_si128(vmultiplier_hi, vmultiplier_base);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier_lo);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier_hi);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(uint8_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) vy_lo;
    }
  }
}
