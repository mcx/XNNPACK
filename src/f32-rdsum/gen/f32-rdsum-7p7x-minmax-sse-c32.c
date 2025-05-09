// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/math.h"


void xnn_f32_rdsum_ukernel_7p7x__sse_c32(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const struct xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vscale = _mm_set1_ps(params->scalar.scale);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 32; channels -= 32) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);

    __m128 vacc0 = _mm_setzero_ps();
    __m128 vacc1 = _mm_setzero_ps();
    __m128 vacc2 = _mm_setzero_ps();
    __m128 vacc3 = _mm_setzero_ps();
    __m128 vacc4 = _mm_setzero_ps();
    __m128 vacc5 = _mm_setzero_ps();
    __m128 vacc6 = _mm_setzero_ps();
    __m128 vacc7 = _mm_setzero_ps();

    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      __m128 vin0;
      __m128 vin1;
      __m128 vin2;
      __m128 vin3;
      __m128 vin4;
      __m128 vin5;
      __m128 vin6;
      __m128 vin7;
      vin0 = _mm_loadu_ps(&i0[0]);
      vin1 = _mm_loadu_ps(&i0[4]);
      vin2 = _mm_loadu_ps(&i0[8]);
      vin3 = _mm_loadu_ps(&i0[12]);
      vin4 = _mm_loadu_ps(&i0[16]);
      vin5 = _mm_loadu_ps(&i0[20]);
      vin6 = _mm_loadu_ps(&i0[24]);
      vin7 = _mm_loadu_ps(&i0[28]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vin0 = _mm_loadu_ps(&i1[0]);
      vin1 = _mm_loadu_ps(&i1[4]);
      vin2 = _mm_loadu_ps(&i1[8]);
      vin3 = _mm_loadu_ps(&i1[12]);
      vin4 = _mm_loadu_ps(&i1[16]);
      vin5 = _mm_loadu_ps(&i1[20]);
      vin6 = _mm_loadu_ps(&i1[24]);
      vin7 = _mm_loadu_ps(&i1[28]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vin0 = _mm_loadu_ps(&i2[0]);
      vin1 = _mm_loadu_ps(&i2[4]);
      vin2 = _mm_loadu_ps(&i2[8]);
      vin3 = _mm_loadu_ps(&i2[12]);
      vin4 = _mm_loadu_ps(&i2[16]);
      vin5 = _mm_loadu_ps(&i2[20]);
      vin6 = _mm_loadu_ps(&i2[24]);
      vin7 = _mm_loadu_ps(&i2[28]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vin0 = _mm_loadu_ps(&i3[0]);
      vin1 = _mm_loadu_ps(&i3[4]);
      vin2 = _mm_loadu_ps(&i3[8]);
      vin3 = _mm_loadu_ps(&i3[12]);
      vin4 = _mm_loadu_ps(&i3[16]);
      vin5 = _mm_loadu_ps(&i3[20]);
      vin6 = _mm_loadu_ps(&i3[24]);
      vin7 = _mm_loadu_ps(&i3[28]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vin0 = _mm_loadu_ps(&i4[0]);
      vin1 = _mm_loadu_ps(&i4[4]);
      vin2 = _mm_loadu_ps(&i4[8]);
      vin3 = _mm_loadu_ps(&i4[12]);
      vin4 = _mm_loadu_ps(&i4[16]);
      vin5 = _mm_loadu_ps(&i4[20]);
      vin6 = _mm_loadu_ps(&i4[24]);
      vin7 = _mm_loadu_ps(&i4[28]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vin0 = _mm_loadu_ps(&i5[0]);
      vin1 = _mm_loadu_ps(&i5[4]);
      vin2 = _mm_loadu_ps(&i5[8]);
      vin3 = _mm_loadu_ps(&i5[12]);
      vin4 = _mm_loadu_ps(&i5[16]);
      vin5 = _mm_loadu_ps(&i5[20]);
      vin6 = _mm_loadu_ps(&i5[24]);
      vin7 = _mm_loadu_ps(&i5[28]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vin0 = _mm_loadu_ps(&i6[0]);
      vin1 = _mm_loadu_ps(&i6[4]);
      vin2 = _mm_loadu_ps(&i6[8]);
      vin3 = _mm_loadu_ps(&i6[12]);
      vin4 = _mm_loadu_ps(&i6[16]);
      vin5 = _mm_loadu_ps(&i6[20]);
      vin6 = _mm_loadu_ps(&i6[24]);
      vin7 = _mm_loadu_ps(&i6[28]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = _mm_mul_ps(vacc0, vscale);
    vacc1 = _mm_mul_ps(vacc1, vscale);
    vacc2 = _mm_mul_ps(vacc2, vscale);
    vacc3 = _mm_mul_ps(vacc3, vscale);
    vacc4 = _mm_mul_ps(vacc4, vscale);
    vacc5 = _mm_mul_ps(vacc5, vscale);
    vacc6 = _mm_mul_ps(vacc6, vscale);
    vacc7 = _mm_mul_ps(vacc7, vscale);

    const float* o = output;
    __m128 vo0 = _mm_loadu_ps(o); o += 4;
    __m128 vo1 = _mm_loadu_ps(o); o += 4;
    __m128 vo2 = _mm_loadu_ps(o); o += 4;
    __m128 vo3 = _mm_loadu_ps(o); o += 4;
    __m128 vo4 = _mm_loadu_ps(o); o += 4;
    __m128 vo5 = _mm_loadu_ps(o); o += 4;
    __m128 vo6 = _mm_loadu_ps(o); o += 4;
    __m128 vo7 = _mm_loadu_ps(o); o += 4;
    vacc0 = _mm_add_ps(vo0, vacc0);
    vacc1 = _mm_add_ps(vo1, vacc1);
    vacc2 = _mm_add_ps(vo2, vacc2);
    vacc3 = _mm_add_ps(vo3, vacc3);
    vacc4 = _mm_add_ps(vo4, vacc4);
    vacc5 = _mm_add_ps(vo5, vacc5);
    vacc6 = _mm_add_ps(vo6, vacc6);
    vacc7 = _mm_add_ps(vo7, vacc7);
    _mm_storeu_ps(output, vacc0); output += 4;
    _mm_storeu_ps(output, vacc1); output += 4;
    _mm_storeu_ps(output, vacc2); output += 4;
    _mm_storeu_ps(output, vacc3); output += 4;
    _mm_storeu_ps(output, vacc4); output += 4;
    _mm_storeu_ps(output, vacc5); output += 4;
    _mm_storeu_ps(output, vacc6); output += 4;
    _mm_storeu_ps(output, vacc7); output += 4;

    input = (const float*) ((uintptr_t) input + 32 * sizeof(float));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);
    __m128 vacc[8];
    vacc[0] = _mm_setzero_ps();
    vacc[1] = _mm_setzero_ps();
    vacc[2] = _mm_setzero_ps();
    vacc[3] = _mm_setzero_ps();
    vacc[4] = _mm_setzero_ps();
    vacc[5] = _mm_setzero_ps();
    vacc[6] = _mm_setzero_ps();
    vacc[7] = _mm_setzero_ps();

    size_t num_chunks = round_up_po2(channels, 4) >> 2;
    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      for (int i = 0; i < num_chunks; ++i) {
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i0[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i1[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i2[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i3[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i4[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i5[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i6[i*4]), vacc[i]);
      }
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    for (int i = 0; i < num_chunks; ++i) {
      vacc[i] = _mm_mul_ps(vacc[i], vscale);
    }

    __m128 vo[8];
    const float* o = output;
    for (int i = 0; i < channels >> 2; ++i) {
      vo[i] = _mm_loadu_ps(o); o += 4;
    }
    for (int i = 0; i < channels >> 2; ++i) {
      vacc[i] = _mm_add_ps(vo[i], vacc[i]);
    }
    for (int i = 0; i < channels >> 2; ++i) {
      _mm_storeu_ps(output, vacc[i]); output += 4;
    }
    const size_t pos = channels >> 2;
    channels &= 0x3;
    __m128 vout = vacc[pos];
    if (channels & 2) {
      __m128 vo = _mm_loadl_pi(vscale, (__m64*) output);
      _mm_storel_pi((__m64*) output, _mm_add_ps(vo, vout));
      vout = _mm_movehl_ps(vout, vout);
      output += 2;
    }
    if (channels & 1) {
      __m128 vo = _mm_load_ss(output);
      _mm_store_ss(output, _mm_add_ps(vo, vout));
    }
  }
}
