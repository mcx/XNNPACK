// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-avx512f.h"


void xnn_f32_dwconv_minmax_ukernel_4p32c__avx512f_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(channels != 0);
  assert(output_width != 0);

  const xnn_simd_f32_t vmin = xnn_set1_f32(params->scalar.min);
  const xnn_simd_f32_t vmax = xnn_set1_f32(params->scalar.max);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 32; c -= 32) {
      xnn_simd_f32_t vacc0p0 = xnn_load_f32(w + 0);
      xnn_simd_f32_t vacc16p0 = xnn_load_f32(w + 16);


      const xnn_simd_f32_t vi0x0 = xnn_loadu_f32(i0 + 0);
      const xnn_simd_f32_t vi0x16 = xnn_loadu_f32(i0 + 16);
      i0 += 32;

      const xnn_simd_f32_t vk0x0 = xnn_load_f32(w + 32);
      const xnn_simd_f32_t vk0x16 = xnn_load_f32(w + 48);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi0x16, vk0x16, vacc16p0);

      const xnn_simd_f32_t vi1x0 = xnn_loadu_f32(i1 + 0);
      const xnn_simd_f32_t vi1x16 = xnn_loadu_f32(i1 + 16);
      i1 += 32;

      const xnn_simd_f32_t vk1x0 = xnn_load_f32(w + 64);
      const xnn_simd_f32_t vk1x16 = xnn_load_f32(w + 80);
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);
      xnn_simd_f32_t vacc16p1 = xnn_mul_f32(vi1x16, vk1x16);

      const xnn_simd_f32_t vi2x0 = xnn_loadu_f32(i2 + 0);
      const xnn_simd_f32_t vi2x16 = xnn_loadu_f32(i2 + 16);
      i2 += 32;

      const xnn_simd_f32_t vk2x0 = xnn_load_f32(w + 96);
      const xnn_simd_f32_t vk2x16 = xnn_load_f32(w + 112);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi2x16, vk2x16, vacc16p0);

      const xnn_simd_f32_t vi3x0 = xnn_loadu_f32(i3 + 0);
      const xnn_simd_f32_t vi3x16 = xnn_loadu_f32(i3 + 16);
      i3 += 32;

      const xnn_simd_f32_t vk3x0 = xnn_load_f32(w + 128);
      const xnn_simd_f32_t vk3x16 = xnn_load_f32(w + 144);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);
      vacc16p1 = xnn_fmadd_f32(vi3x16, vk3x16, vacc16p1);

      w += 160;

      // Add up all accumulators to vacc0p0
      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);
      vacc16p0 = xnn_add_f32(vacc16p0, vacc16p1);

      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      xnn_simd_f32_t vacc16 = xnn_max_f32(vmin, vacc16p0);
      vacc0 = xnn_min_f32(vmax, vacc0);
      vacc16 = xnn_min_f32(vmax, vacc16);

      xnn_storeu_f32(output + 0, vacc0);
      xnn_storeu_f32(output + 16, vacc16);
      output += 32;
    }
    for (; c >= 16; c -= 16) {
      xnn_simd_f32_t vacc0p0 = xnn_load_f32(w);

      const xnn_simd_f32_t vi0x0 = xnn_loadu_f32(i0);
      i0 += 16;

      const xnn_simd_f32_t vk0x0 = xnn_load_f32(w + 32);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);

      const xnn_simd_f32_t vi1x0 = xnn_loadu_f32(i1);
      i1 += 16;

      const xnn_simd_f32_t vk1x0 = xnn_load_f32(w + 64);
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);

      const xnn_simd_f32_t vi2x0 = xnn_loadu_f32(i2);
      i2 += 16;

      const xnn_simd_f32_t vk2x0 = xnn_load_f32(w + 96);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);

      const xnn_simd_f32_t vi3x0 = xnn_loadu_f32(i3);
      i3 += 16;

      const xnn_simd_f32_t vk3x0 = xnn_load_f32(w + 128);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);

      w += 16;

      // Add up all accumulators to vacc0p0
      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);

      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      vacc0 = xnn_min_f32(vmax, vacc0);

      xnn_storeu_f32(output, vacc0);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f32_t vacc0p0 = xnn_load_tail_f32(w, c);

      const xnn_simd_f32_t vi0x0 = xnn_load_tail_f32(i0, c);
      const xnn_simd_f32_t vk0x0 = xnn_load_tail_f32(w + 32, c);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);

      const xnn_simd_f32_t vi1x0 = xnn_load_tail_f32(i1, c);
      const xnn_simd_f32_t vk1x0 = xnn_load_tail_f32(w + 64, c);
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);

      const xnn_simd_f32_t vi2x0 = xnn_load_tail_f32(i2, c);
      const xnn_simd_f32_t vk2x0 = xnn_load_tail_f32(w + 96, c);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);

      const xnn_simd_f32_t vi3x0 = xnn_load_tail_f32(i3, c);
      const xnn_simd_f32_t vk3x0 = xnn_load_tail_f32(w + 128, c);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);

      // Add up all accumulators to vacc0p0
      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);

      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      vacc0 = xnn_min_f32(vmax, vacc0);

      xnn_store_tail_f32(output, vacc0, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
