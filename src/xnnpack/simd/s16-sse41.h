// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_S16_SSE2_H_
#define XNNPACK_SRC_XNNPACK_SIMD_S16_SSE2_H_

#include <assert.h>
#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/unaligned.h"

// SIMD vector type for s16 using SSE41.
typedef __m128i xnn_simd_s16_t;
#define xnn_simd_size_s16 8
#define xnn_simd_log2_size_s16 3
#define xnn_simd_bytes_s16 (xnn_simd_size_s16 * sizeof(int16_t))

#define XNN_SIMD_CONST_S16(var, val) \
  const xnn_simd_s16_t var = _mm_set1_epi16(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s16_t xnn_min_s16(xnn_simd_s16_t a,
                                             xnn_simd_s16_t b) {
  return _mm_min_epi16(a, b);
}

static XNN_INLINE xnn_simd_s16_t xnn_max_s16(xnn_simd_s16_t a,
                                             xnn_simd_s16_t b) {
  return _mm_max_epi16(a, b);
}

static XNN_INLINE xnn_simd_s16_t xnn_signcomplement_s16(xnn_simd_s16_t x) {
  XNN_SIMD_CONST_S16(nonsign_mask, 0x7FFF);
  return _mm_xor_si128(_mm_and_si128(x, nonsign_mask), _mm_srai_epi16(x, 15));
}

// Load/store operations.

static XNN_INLINE xnn_simd_s16_t xnn_loadu_s16(const int16_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_s16(const int16_t* ptr) {
  return _mm_load_si128((const __m128i*)ptr);
}

static XNN_INLINE void xnn_storeu_s16(int16_t* ptr, xnn_simd_s16_t v) {
  _mm_storeu_si128((__m128i*)ptr, v);
}

static XNN_INLINE void xnn_store_s16(int16_t* ptr, xnn_simd_s16_t v) {
  _mm_store_si128((__m128i*)ptr, v);
}

static XNN_INLINE xnn_simd_s16_t xnn_set1_s16(int16_t v) {
  return _mm_set1_epi16(v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s16_t
xnn_load_tail_s16(const int16_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);
  return _mm_loadu_si128((const __m128i*)input);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_tail_safe_s16(const int16_t* input,
                                                        size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);

  XNN_ALIGN(16) int16_t padded[8];
  int16_t* d = &padded[0];
  switch (num_elements) {
    case 7:
      *d++ = *input++;
    case 6:
      *d++ = *input++;
    case 5:
      *d++ = *input++;
    case 4:
      *d++ = *input++;
    case 3:
      *d++ = *input++;
    case 2:
      *d++ = *input++;
    case 1:
      *d++ = *input++;
  }
  return _mm_load_si128((const __m128i*)&padded[0]);
}

static XNN_INLINE void xnn_store_tail_s16(int16_t* output, xnn_simd_s16_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);

  if (num_elements & 4) {
    _mm_storel_epi64((__m128i*)output, v);
    v = _mm_unpackhi_epi64(v, v);
    output += 4;
  }
  if (num_elements & 2) {
    unaligned_store_u32(output, (uint32_t)_mm_cvtsi128_si32(v));
    v = _mm_srli_epi64(v, 32);
    output += 2;
  }
  if (num_elements & 1) {
    unaligned_store_u16(output, (uint16_t)_mm_extract_epi16(v, 0));
  }
}

#endif  // XNNPACK_SRC_XNNPACK_SIMD_S16_SSE2_H_
