// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vsqrt/avx512fp16-sqrt.c.in
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
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u64(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(i);
    __m512h vacc1 = _mm512_loadu_ph(i + 32);
    i += 64;

    vacc0 = _mm512_sqrt_ph(vacc0);
    vacc1 = _mm512_sqrt_ph(vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(i);
    i += 32;
    vacc = _mm512_sqrt_ph(vacc);
    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i));
    vacc = _mm512_mask_sqrt_ph(vacc, vmask, vacc);

    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}
