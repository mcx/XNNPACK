// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/sse-dup.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xmmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/igemm.h"


void xnn_f32_igemm_minmax_ukernel_6x8__sse_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

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
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  const __m128 vmin = _mm_set1_ps(params->scalar.min);
  const __m128 vmax = _mm_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_load_ps(w);
    __m128 vacc0x4567 = _mm_load_ps(w + 4);
    __m128 vacc1x0123 = vacc0x0123;
    __m128 vacc1x4567 = vacc0x4567;
    __m128 vacc2x0123 = vacc0x0123;
    __m128 vacc2x4567 = vacc0x4567;
    __m128 vacc3x0123 = vacc0x0123;
    __m128 vacc3x4567 = vacc0x4567;
    __m128 vacc4x0123 = vacc0x0123;
    __m128 vacc4x4567 = vacc0x4567;
    __m128 vacc5x0123 = vacc0x0123;
    __m128 vacc5x4567 = vacc0x4567;
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const __m128 va0 = _mm_loadu_ps(a0);
        a0 += 4;
        const __m128 va1 = _mm_loadu_ps(a1);
        a1 += 4;
        const __m128 va2 = _mm_loadu_ps(a2);
        a2 += 4;
        const __m128 va3 = _mm_loadu_ps(a3);
        a3 += 4;
        const __m128 va4 = _mm_loadu_ps(a4);
        a4 += 4;
        const __m128 va5 = _mm_loadu_ps(a5);
        a5 += 4;


        const __m128 va0c0000 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 va1c0000 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 va2c0000 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 va3c0000 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 va4c0000 = _mm_shuffle_ps(va4, va4, _MM_SHUFFLE(0, 0, 0, 0));
        const __m128 va5c0000 = _mm_shuffle_ps(va5, va5, _MM_SHUFFLE(0, 0, 0, 0));

        const __m128 vb0123c0 = _mm_load_ps(w + 0);
        const __m128 vb4567c0 = _mm_load_ps(w + 4);

        vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
        vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c0000, vb0123c0));
        vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c0000, vb0123c0));
        vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c0000, vb0123c0));
        vacc4x0123 = _mm_add_ps(vacc4x0123, _mm_mul_ps(va4c0000, vb0123c0));
        vacc5x0123 = _mm_add_ps(vacc5x0123, _mm_mul_ps(va5c0000, vb0123c0));
        vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));
        vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c0000, vb4567c0));
        vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c0000, vb4567c0));
        vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c0000, vb4567c0));
        vacc4x4567 = _mm_add_ps(vacc4x4567, _mm_mul_ps(va4c0000, vb4567c0));
        vacc5x4567 = _mm_add_ps(vacc5x4567, _mm_mul_ps(va5c0000, vb4567c0));

        const __m128 va0c1111 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 va1c1111 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 va2c1111 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 va3c1111 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 va4c1111 = _mm_shuffle_ps(va4, va4, _MM_SHUFFLE(1, 1, 1, 1));
        const __m128 va5c1111 = _mm_shuffle_ps(va5, va5, _MM_SHUFFLE(1, 1, 1, 1));

        const __m128 vb0123c1 = _mm_load_ps(w + 8);
        const __m128 vb4567c1 = _mm_load_ps(w + 12);

        vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
        vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c1111, vb0123c1));
        vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c1111, vb0123c1));
        vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c1111, vb0123c1));
        vacc4x0123 = _mm_add_ps(vacc4x0123, _mm_mul_ps(va4c1111, vb0123c1));
        vacc5x0123 = _mm_add_ps(vacc5x0123, _mm_mul_ps(va5c1111, vb0123c1));
        vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));
        vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c1111, vb4567c1));
        vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c1111, vb4567c1));
        vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c1111, vb4567c1));
        vacc4x4567 = _mm_add_ps(vacc4x4567, _mm_mul_ps(va4c1111, vb4567c1));
        vacc5x4567 = _mm_add_ps(vacc5x4567, _mm_mul_ps(va5c1111, vb4567c1));

        const __m128 va0c2222 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 va1c2222 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 va2c2222 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 va3c2222 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 va4c2222 = _mm_shuffle_ps(va4, va4, _MM_SHUFFLE(2, 2, 2, 2));
        const __m128 va5c2222 = _mm_shuffle_ps(va5, va5, _MM_SHUFFLE(2, 2, 2, 2));

        const __m128 vb0123c2 = _mm_load_ps(w + 16);
        const __m128 vb4567c2 = _mm_load_ps(w + 20);

        vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c2222, vb0123c2));
        vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c2222, vb0123c2));
        vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c2222, vb0123c2));
        vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c2222, vb0123c2));
        vacc4x0123 = _mm_add_ps(vacc4x0123, _mm_mul_ps(va4c2222, vb0123c2));
        vacc5x0123 = _mm_add_ps(vacc5x0123, _mm_mul_ps(va5c2222, vb0123c2));
        vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c2222, vb4567c2));
        vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c2222, vb4567c2));
        vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c2222, vb4567c2));
        vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c2222, vb4567c2));
        vacc4x4567 = _mm_add_ps(vacc4x4567, _mm_mul_ps(va4c2222, vb4567c2));
        vacc5x4567 = _mm_add_ps(vacc5x4567, _mm_mul_ps(va5c2222, vb4567c2));

        const __m128 va0c3333 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(3, 3, 3, 3));
        const __m128 va1c3333 = _mm_shuffle_ps(va1, va1, _MM_SHUFFLE(3, 3, 3, 3));
        const __m128 va2c3333 = _mm_shuffle_ps(va2, va2, _MM_SHUFFLE(3, 3, 3, 3));
        const __m128 va3c3333 = _mm_shuffle_ps(va3, va3, _MM_SHUFFLE(3, 3, 3, 3));
        const __m128 va4c3333 = _mm_shuffle_ps(va4, va4, _MM_SHUFFLE(3, 3, 3, 3));
        const __m128 va5c3333 = _mm_shuffle_ps(va5, va5, _MM_SHUFFLE(3, 3, 3, 3));

        const __m128 vb0123c3 = _mm_load_ps(w + 24);
        const __m128 vb4567c3 = _mm_load_ps(w + 28);

        vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c3333, vb0123c3));
        vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1c3333, vb0123c3));
        vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2c3333, vb0123c3));
        vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3c3333, vb0123c3));
        vacc4x0123 = _mm_add_ps(vacc4x0123, _mm_mul_ps(va4c3333, vb0123c3));
        vacc5x0123 = _mm_add_ps(vacc5x0123, _mm_mul_ps(va5c3333, vb0123c3));
        vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c3333, vb4567c3));
        vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1c3333, vb4567c3));
        vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2c3333, vb4567c3));
        vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3c3333, vb4567c3));
        vacc4x4567 = _mm_add_ps(vacc4x4567, _mm_mul_ps(va4c3333, vb4567c3));
        vacc5x4567 = _mm_add_ps(vacc5x4567, _mm_mul_ps(va5c3333, vb4567c3));

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const __m128 vb0123 = _mm_load_ps(w);
          const __m128 vb4567 = _mm_load_ps(w + 4);
          w += 8;

          const __m128 va0 = _mm_load1_ps(a0);
          a0 += 1;
          const __m128 va1 = _mm_load1_ps(a1);
          a1 += 1;
          const __m128 va2 = _mm_load1_ps(a2);
          a2 += 1;
          const __m128 va3 = _mm_load1_ps(a3);
          a3 += 1;
          const __m128 va4 = _mm_load1_ps(a4);
          a4 += 1;
          const __m128 va5 = _mm_load1_ps(a5);
          a5 += 1;

          vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
          vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));
          vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(va1, vb0123));
          vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(va1, vb4567));
          vacc2x0123 = _mm_add_ps(vacc2x0123, _mm_mul_ps(va2, vb0123));
          vacc2x4567 = _mm_add_ps(vacc2x4567, _mm_mul_ps(va2, vb4567));
          vacc3x0123 = _mm_add_ps(vacc3x0123, _mm_mul_ps(va3, vb0123));
          vacc3x4567 = _mm_add_ps(vacc3x4567, _mm_mul_ps(va3, vb4567));
          vacc4x0123 = _mm_add_ps(vacc4x0123, _mm_mul_ps(va4, vb0123));
          vacc4x4567 = _mm_add_ps(vacc4x4567, _mm_mul_ps(va4, vb4567));
          vacc5x0123 = _mm_add_ps(vacc5x0123, _mm_mul_ps(va5, vb0123));
          vacc5x4567 = _mm_add_ps(vacc5x4567, _mm_mul_ps(va5, vb4567));
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc1x0123 = _mm_min_ps(vacc1x0123, vmax);
    vacc2x0123 = _mm_min_ps(vacc2x0123, vmax);
    vacc3x0123 = _mm_min_ps(vacc3x0123, vmax);
    vacc4x0123 = _mm_min_ps(vacc4x0123, vmax);
    vacc5x0123 = _mm_min_ps(vacc5x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);
    vacc1x4567 = _mm_min_ps(vacc1x4567, vmax);
    vacc2x4567 = _mm_min_ps(vacc2x4567, vmax);
    vacc3x4567 = _mm_min_ps(vacc3x4567, vmax);
    vacc4x4567 = _mm_min_ps(vacc4x4567, vmax);
    vacc5x4567 = _mm_min_ps(vacc5x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc1x0123 = _mm_max_ps(vacc1x0123, vmin);
    vacc2x0123 = _mm_max_ps(vacc2x0123, vmin);
    vacc3x0123 = _mm_max_ps(vacc3x0123, vmin);
    vacc4x0123 = _mm_max_ps(vacc4x0123, vmin);
    vacc5x0123 = _mm_max_ps(vacc5x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);
    vacc1x4567 = _mm_max_ps(vacc1x4567, vmin);
    vacc2x4567 = _mm_max_ps(vacc2x4567, vmin);
    vacc3x4567 = _mm_max_ps(vacc3x4567, vmin);
    vacc4x4567 = _mm_max_ps(vacc4x4567, vmin);
    vacc5x4567 = _mm_max_ps(vacc5x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c5, vacc5x0123);
      _mm_storeu_ps(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_ps(c4, vacc4x0123);
      _mm_storeu_ps(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_ps(c3, vacc3x0123);
      _mm_storeu_ps(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_ps(c2, vacc2x0123);
      _mm_storeu_ps(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_ps(c1, vacc1x0123);
      _mm_storeu_ps(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c5, vacc5x0123);
        _mm_storeu_ps(c4, vacc4x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc5x0123 = vacc5x4567;
        vacc4x0123 = vacc4x4567;
        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c5 += 4;
        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c5, vacc5x0123);
        _mm_storel_pi((__m64*) c4, vacc4x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc5x0123 = _mm_movehl_ps(vacc5x0123, vacc5x0123);
        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c5, vacc5x0123);
        _mm_store_ss(c4, vacc4x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
