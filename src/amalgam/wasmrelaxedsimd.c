// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/lut.h>
#include <xnnpack/vcvt.h>


void xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
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
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi0x4567, vk0x4567);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi1x4567, vk1x4567);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi2x4567, vk2x4567);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi3x4567, vk3x4567);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vi4x4567 = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      const v128_t vk4x4567 = wasm_v128_load(w + 44);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi4x4567, vk4x4567);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vi5x4567 = wasm_v128_load(i5 + 4);
      i5 += 8;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      const v128_t vk5x4567 = wasm_v128_load(w + 52);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi5x4567, vk5x4567);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vi6x4567 = wasm_v128_load(i6 + 4);
      i6 += 8;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      const v128_t vk6x4567 = wasm_v128_load(w + 60);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi6x4567, vk6x4567);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vi7x4567 = wasm_v128_load(i7 + 4);
      i7 += 8;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      const v128_t vk7x4567 = wasm_v128_load(w + 68);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi7x4567, vk7x4567);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vi8x4567 = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      const v128_t vk8x4567 = wasm_v128_load(w + 76);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi8x4567, vk8x4567);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      const v128_t vi9x4567 = wasm_v128_load(i9 + 4);
      i9 += 8;

      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      const v128_t vk9x4567 = wasm_v128_load(w + 84);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi9x0123, vk9x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi9x4567, vk9x4567);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      const v128_t vi10x4567 = wasm_v128_load(i10 + 4);
      i10 += 8;

      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      const v128_t vk10x4567 = wasm_v128_load(w + 92);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi10x0123, vk10x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi10x4567, vk10x4567);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      const v128_t vi11x4567 = wasm_v128_load(i11 + 4);
      i11 += 8;

      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      const v128_t vk11x4567 = wasm_v128_load(w + 100);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi11x0123, vk11x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi11x4567, vk11x4567);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      const v128_t vi12x4567 = wasm_v128_load(i12 + 4);
      i12 += 8;

      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      const v128_t vk12x4567 = wasm_v128_load(w + 108);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi12x0123, vk12x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi12x4567, vk12x4567);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      const v128_t vi13x4567 = wasm_v128_load(i13 + 4);
      i13 += 8;

      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      const v128_t vk13x4567 = wasm_v128_load(w + 116);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi13x0123, vk13x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi13x4567, vk13x4567);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      const v128_t vi14x4567 = wasm_v128_load(i14 + 4);
      i14 += 8;

      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      const v128_t vk14x4567 = wasm_v128_load(w + 124);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi14x0123, vk14x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi14x4567, vk14x4567);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      const v128_t vi15x4567 = wasm_v128_load(i15 + 4);
      i15 += 8;

      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      const v128_t vk15x4567 = wasm_v128_load(w + 132);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi15x0123, vk15x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi15x4567, vk15x4567);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      const v128_t vi16x4567 = wasm_v128_load(i16 + 4);
      i16 += 8;

      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      const v128_t vk16x4567 = wasm_v128_load(w + 140);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi16x0123, vk16x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi16x4567, vk16x4567);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      const v128_t vi17x4567 = wasm_v128_load(i17 + 4);
      i17 += 8;

      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      const v128_t vk17x4567 = wasm_v128_load(w + 148);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi17x0123, vk17x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi17x4567, vk17x4567);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      const v128_t vi18x4567 = wasm_v128_load(i18 + 4);
      i18 += 8;

      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      const v128_t vk18x4567 = wasm_v128_load(w + 156);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi18x0123, vk18x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi18x4567, vk18x4567);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      const v128_t vi19x4567 = wasm_v128_load(i19 + 4);
      i19 += 8;

      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      const v128_t vk19x4567 = wasm_v128_load(w + 164);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi19x0123, vk19x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi19x4567, vk19x4567);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      const v128_t vi20x4567 = wasm_v128_load(i20 + 4);
      i20 += 8;

      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      const v128_t vk20x4567 = wasm_v128_load(w + 172);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi20x0123, vk20x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi20x4567, vk20x4567);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      const v128_t vi21x4567 = wasm_v128_load(i21 + 4);
      i21 += 8;

      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      const v128_t vk21x4567 = wasm_v128_load(w + 180);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi21x0123, vk21x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi21x4567, vk21x4567);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      const v128_t vi22x4567 = wasm_v128_load(i22 + 4);
      i22 += 8;

      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      const v128_t vk22x4567 = wasm_v128_load(w + 188);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi22x0123, vk22x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi22x4567, vk22x4567);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      const v128_t vi23x4567 = wasm_v128_load(i23 + 4);
      i23 += 8;

      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      const v128_t vk23x4567 = wasm_v128_load(w + 196);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi23x0123, vk23x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi23x4567, vk23x4567);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      const v128_t vi24x4567 = wasm_v128_load(i24 + 4);
      i24 += 8;

      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      const v128_t vk24x4567 = wasm_v128_load(w + 204);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi24x0123, vk24x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi24x4567, vk24x4567);

      w += 208;


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      v128_t vacc4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc4567p0);

      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);
      vacc4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc4567);

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      i9 += 4;

      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi9x0123, vk9x0123);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      i10 += 4;

      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi10x0123, vk10x0123);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      i11 += 4;

      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi11x0123, vk11x0123);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      i12 += 4;

      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi12x0123, vk12x0123);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      i13 += 4;

      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi13x0123, vk13x0123);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      i14 += 4;

      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi14x0123, vk14x0123);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      i15 += 4;

      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi15x0123, vk15x0123);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      i16 += 4;

      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi16x0123, vk16x0123);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      i17 += 4;

      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi17x0123, vk17x0123);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      i18 += 4;

      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi18x0123, vk18x0123);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      i19 += 4;

      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi19x0123, vk19x0123);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      i20 += 4;

      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi20x0123, vk20x0123);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      i21 += 4;

      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi21x0123, vk21x0123);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      i22 += 4;

      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi22x0123, vk22x0123);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      i23 += 4;

      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi23x0123, vk23x0123);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      i24 += 4;

      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi24x0123, vk24x0123);

      w += 4;


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi9x0123, vk9x0123);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi10x0123, vk10x0123);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi11x0123, vk11x0123);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi12x0123, vk12x0123);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi13x0123, vk13x0123);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi14x0123, vk14x0123);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi15x0123, vk15x0123);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi16x0123, vk16x0123);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi17x0123, vk17x0123);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi18x0123, vk18x0123);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi19x0123, vk19x0123);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi20x0123, vk20x0123);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi21x0123, vk21x0123);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi22x0123, vk22x0123);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi23x0123, vk23x0123);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi24x0123, vk24x0123);


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

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
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi0x4567, vk0x4567);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi1x4567, vk1x4567);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi2x4567, vk2x4567);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi3x4567, vk3x4567);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vi4x4567 = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      const v128_t vk4x4567 = wasm_v128_load(w + 44);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi4x4567, vk4x4567);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vi5x4567 = wasm_v128_load(i5 + 4);
      i5 += 8;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      const v128_t vk5x4567 = wasm_v128_load(w + 52);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi5x4567, vk5x4567);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vi6x4567 = wasm_v128_load(i6 + 4);
      i6 += 8;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      const v128_t vk6x4567 = wasm_v128_load(w + 60);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi6x4567, vk6x4567);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vi7x4567 = wasm_v128_load(i7 + 4);
      i7 += 8;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      const v128_t vk7x4567 = wasm_v128_load(w + 68);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi7x4567, vk7x4567);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vi8x4567 = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      const v128_t vk8x4567 = wasm_v128_load(w + 76);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi8x4567, vk8x4567);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      const v128_t vi9x4567 = wasm_v128_load(i9 + 4);
      i9 += 8;

      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      const v128_t vk9x4567 = wasm_v128_load(w + 84);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi9x0123, vk9x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi9x4567, vk9x4567);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      const v128_t vi10x4567 = wasm_v128_load(i10 + 4);
      i10 += 8;

      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      const v128_t vk10x4567 = wasm_v128_load(w + 92);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi10x0123, vk10x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi10x4567, vk10x4567);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      const v128_t vi11x4567 = wasm_v128_load(i11 + 4);
      i11 += 8;

      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      const v128_t vk11x4567 = wasm_v128_load(w + 100);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi11x0123, vk11x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi11x4567, vk11x4567);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      const v128_t vi12x4567 = wasm_v128_load(i12 + 4);
      i12 += 8;

      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      const v128_t vk12x4567 = wasm_v128_load(w + 108);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi12x0123, vk12x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi12x4567, vk12x4567);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      const v128_t vi13x4567 = wasm_v128_load(i13 + 4);
      i13 += 8;

      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      const v128_t vk13x4567 = wasm_v128_load(w + 116);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi13x0123, vk13x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi13x4567, vk13x4567);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      const v128_t vi14x4567 = wasm_v128_load(i14 + 4);
      i14 += 8;

      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      const v128_t vk14x4567 = wasm_v128_load(w + 124);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi14x0123, vk14x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi14x4567, vk14x4567);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      const v128_t vi15x4567 = wasm_v128_load(i15 + 4);
      i15 += 8;

      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      const v128_t vk15x4567 = wasm_v128_load(w + 132);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi15x0123, vk15x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi15x4567, vk15x4567);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      const v128_t vi16x4567 = wasm_v128_load(i16 + 4);
      i16 += 8;

      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      const v128_t vk16x4567 = wasm_v128_load(w + 140);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi16x0123, vk16x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi16x4567, vk16x4567);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      const v128_t vi17x4567 = wasm_v128_load(i17 + 4);
      i17 += 8;

      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      const v128_t vk17x4567 = wasm_v128_load(w + 148);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi17x0123, vk17x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi17x4567, vk17x4567);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      const v128_t vi18x4567 = wasm_v128_load(i18 + 4);
      i18 += 8;

      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      const v128_t vk18x4567 = wasm_v128_load(w + 156);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi18x0123, vk18x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi18x4567, vk18x4567);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      const v128_t vi19x4567 = wasm_v128_load(i19 + 4);
      i19 += 8;

      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      const v128_t vk19x4567 = wasm_v128_load(w + 164);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi19x0123, vk19x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi19x4567, vk19x4567);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      const v128_t vi20x4567 = wasm_v128_load(i20 + 4);
      i20 += 8;

      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      const v128_t vk20x4567 = wasm_v128_load(w + 172);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi20x0123, vk20x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi20x4567, vk20x4567);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      const v128_t vi21x4567 = wasm_v128_load(i21 + 4);
      i21 += 8;

      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      const v128_t vk21x4567 = wasm_v128_load(w + 180);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi21x0123, vk21x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi21x4567, vk21x4567);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      const v128_t vi22x4567 = wasm_v128_load(i22 + 4);
      i22 += 8;

      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      const v128_t vk22x4567 = wasm_v128_load(w + 188);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi22x0123, vk22x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi22x4567, vk22x4567);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      const v128_t vi23x4567 = wasm_v128_load(i23 + 4);
      i23 += 8;

      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      const v128_t vk23x4567 = wasm_v128_load(w + 196);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi23x0123, vk23x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi23x4567, vk23x4567);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      const v128_t vi24x4567 = wasm_v128_load(i24 + 4);
      i24 += 8;

      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      const v128_t vk24x4567 = wasm_v128_load(w + 204);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi24x0123, vk24x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi24x4567, vk24x4567);

      w += 208;


      const v128_t vacc0123 = vacc0123p0;
      const v128_t vacc4567 = vacc4567p0;

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      i9 += 4;

      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi9x0123, vk9x0123);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      i10 += 4;

      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi10x0123, vk10x0123);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      i11 += 4;

      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi11x0123, vk11x0123);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      i12 += 4;

      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi12x0123, vk12x0123);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      i13 += 4;

      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi13x0123, vk13x0123);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      i14 += 4;

      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi14x0123, vk14x0123);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      i15 += 4;

      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi15x0123, vk15x0123);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      i16 += 4;

      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi16x0123, vk16x0123);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      i17 += 4;

      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi17x0123, vk17x0123);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      i18 += 4;

      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi18x0123, vk18x0123);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      i19 += 4;

      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi19x0123, vk19x0123);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      i20 += 4;

      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi20x0123, vk20x0123);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      i21 += 4;

      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi21x0123, vk21x0123);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      i22 += 4;

      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi22x0123, vk22x0123);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      i23 += 4;

      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi23x0123, vk23x0123);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      i24 += 4;

      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi24x0123, vk24x0123);

      w += 4;


      const v128_t vacc0123 = vacc0123p0;

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi9x0123, vk9x0123);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi10x0123, vk10x0123);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi11x0123, vk11x0123);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi12x0123, vk12x0123);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi13x0123, vk13x0123);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi14x0123, vk14x0123);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi15x0123, vk15x0123);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi16x0123, vk16x0123);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi17x0123, vk17x0123);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi18x0123, vk18x0123);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi19x0123, vk19x0123);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi20x0123, vk20x0123);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi21x0123, vk21x0123);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi22x0123, vk22x0123);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi23x0123, vk23x0123);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi24x0123, vk24x0123);


      v128_t vacc0123 = vacc0123p0;

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_3p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
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
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi0x4567, vk0x4567);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi1x4567, vk1x4567);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi2x4567, vk2x4567);

      w += 32;


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      v128_t vacc4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc4567p0);

      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);
      vacc4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc4567);

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      w += 4;


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

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
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi0x4567, vk0x4567);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi1x4567, vk1x4567);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi2x4567, vk2x4567);

      w += 32;


      const v128_t vacc0123 = vacc0123p0;
      const v128_t vacc4567 = vacc4567p0;

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      w += 4;


      const v128_t vacc0123 = vacc0123p0;

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);


      v128_t vacc0123 = vacc0123p0;

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_4p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
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
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi0x4567, vk0x4567);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi1x4567, vk1x4567);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi2x4567, vk2x4567);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi3x4567, vk3x4567);

      w += 40;


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      v128_t vacc4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc4567p0);

      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);
      vacc4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc4567);

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      w += 4;


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

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
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi0x4567, vk0x4567);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi1x4567, vk1x4567);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi2x4567, vk2x4567);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi3x4567, vk3x4567);

      w += 40;


      const v128_t vacc0123 = vacc0123p0;
      const v128_t vacc4567 = vacc4567p0;

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      w += 4;


      const v128_t vacc0123 = vacc0123p0;

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);


      v128_t vacc0123 = vacc0123p0;

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_9p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
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
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi0x4567, vk0x4567);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi1x4567, vk1x4567);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi2x4567, vk2x4567);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi3x4567, vk3x4567);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vi4x4567 = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      const v128_t vk4x4567 = wasm_v128_load(w + 44);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi4x4567, vk4x4567);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vi5x4567 = wasm_v128_load(i5 + 4);
      i5 += 8;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      const v128_t vk5x4567 = wasm_v128_load(w + 52);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi5x4567, vk5x4567);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vi6x4567 = wasm_v128_load(i6 + 4);
      i6 += 8;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      const v128_t vk6x4567 = wasm_v128_load(w + 60);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi6x4567, vk6x4567);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vi7x4567 = wasm_v128_load(i7 + 4);
      i7 += 8;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      const v128_t vk7x4567 = wasm_v128_load(w + 68);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi7x4567, vk7x4567);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vi8x4567 = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      const v128_t vk8x4567 = wasm_v128_load(w + 76);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi8x4567, vk8x4567);

      w += 80;


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      v128_t vacc4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc4567p0);

      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);
      vacc4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc4567);

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);

      w += 4;


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);


      v128_t vacc0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0123p0);
      vacc0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

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
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi0x4567, vk0x4567);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi1x4567, vk1x4567);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi2x4567, vk2x4567);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi3x4567, vk3x4567);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vi4x4567 = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      const v128_t vk4x4567 = wasm_v128_load(w + 44);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi4x4567, vk4x4567);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vi5x4567 = wasm_v128_load(i5 + 4);
      i5 += 8;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      const v128_t vk5x4567 = wasm_v128_load(w + 52);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi5x4567, vk5x4567);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vi6x4567 = wasm_v128_load(i6 + 4);
      i6 += 8;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      const v128_t vk6x4567 = wasm_v128_load(w + 60);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi6x4567, vk6x4567);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vi7x4567 = wasm_v128_load(i7 + 4);
      i7 += 8;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      const v128_t vk7x4567 = wasm_v128_load(w + 68);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi7x4567, vk7x4567);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vi8x4567 = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      const v128_t vk8x4567 = wasm_v128_load(w + 76);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);
      vacc4567p0 = __builtin_wasm_fma_f32x4(vacc4567p0, vi8x4567, vk8x4567);

      w += 80;


      const v128_t vacc0123 = vacc0123p0;
      const v128_t vacc4567 = vacc4567p0;

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);

      w += 4;


      const v128_t vacc0123 = vacc0123p0;

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi0x0123, vk0x0123);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi1x0123, vk1x0123);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi2x0123, vk2x0123);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi3x0123, vk3x0123);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi4x0123, vk4x0123);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi5x0123, vk5x0123);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi6x0123, vk6x0123);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi7x0123, vk7x0123);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = __builtin_wasm_fma_f32x4(vacc0123p0, vi8x0123, vk8x0123);


      v128_t vacc0123 = vacc0123p0;

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);

      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    vacc0x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x0123);
    vacc0x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x4567);

    vacc0x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x0123);
    vacc0x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);

      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);

      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    v128_t vacc0x0c4 = wasm_v128_load32_zero(w);
    v128_t vacc0x1c4 = wasm_v128_load32_zero(w + 1);
    v128_t vacc1x0c4 = vacc0x0c4;
    v128_t vacc1x1c4 = vacc0x1c4;
    v128_t vacc2x0c4 = vacc0x0c4;
    v128_t vacc2x1c4 = vacc0x1c4;
    v128_t vacc3x0c4 = vacc0x0c4;
    v128_t vacc3x1c4 = vacc0x1c4;
    w += 2;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0c4 = __builtin_wasm_fma_f32x4(vacc0x0c4, va0, vb0);
      vacc0x1c4 = __builtin_wasm_fma_f32x4(vacc0x1c4, va0, vb1);
      vacc1x0c4 = __builtin_wasm_fma_f32x4(vacc1x0c4, va1, vb0);
      vacc1x1c4 = __builtin_wasm_fma_f32x4(vacc1x1c4, va1, vb1);
      vacc2x0c4 = __builtin_wasm_fma_f32x4(vacc2x0c4, va2, vb0);
      vacc2x1c4 = __builtin_wasm_fma_f32x4(vacc2x1c4, va2, vb1);
      vacc3x0c4 = __builtin_wasm_fma_f32x4(vacc3x0c4, va3, vb0);
      vacc3x1c4 = __builtin_wasm_fma_f32x4(vacc3x1c4, va3, vb1);
    }
    if XNN_UNLIKELY(k != 0) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      const v128_t va1 = wasm_v128_load(a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      const v128_t va2 = wasm_v128_load(a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      const v128_t va3 = wasm_v128_load(a3);
      a3 = (const float*) ((uintptr_t) a3 + k);

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      const v128_t vzero = wasm_f32x4_const_splat(0.0f);
      const v128_t vmask0 = wasm_f32x4_eq(vb0, vzero);
      const v128_t vmask1 = wasm_f32x4_eq(vb1, vzero);

      vacc0x0c4 = __builtin_wasm_fma_f32x4(vacc0x0c4, wasm_v128_andnot(va0, vmask0), vb0);
      vacc0x1c4 = __builtin_wasm_fma_f32x4(vacc0x1c4, wasm_v128_andnot(va0, vmask1), vb1);
      vacc1x0c4 = __builtin_wasm_fma_f32x4(vacc1x0c4, wasm_v128_andnot(va1, vmask0), vb0);
      vacc1x1c4 = __builtin_wasm_fma_f32x4(vacc1x1c4, wasm_v128_andnot(va1, vmask1), vb1);
      vacc2x0c4 = __builtin_wasm_fma_f32x4(vacc2x0c4, wasm_v128_andnot(va2, vmask0), vb0);
      vacc2x1c4 = __builtin_wasm_fma_f32x4(vacc2x1c4, wasm_v128_andnot(va2, vmask1), vb1);
      vacc3x0c4 = __builtin_wasm_fma_f32x4(vacc3x0c4, wasm_v128_andnot(va3, vmask0), vb0);
      vacc3x1c4 = __builtin_wasm_fma_f32x4(vacc3x1c4, wasm_v128_andnot(va3, vmask1), vb1);
    }

    const v128_t vacc0x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 2, 6, 3, 7));
    const v128_t vacc1x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 2, 6, 3, 7));
    const v128_t vacc2x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 2, 6, 3, 7));
    const v128_t vacc3x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 2, 6, 3, 7));

    v128_t vacc01x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 2, 3, 6, 7));
    v128_t vacc23x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 2, 3, 6, 7));

    vacc01x01 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc01x01);
    vacc23x01 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc23x01);

    vacc01x01 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc01x01);
    vacc23x01 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc23x01);

    if XNN_LIKELY(nc >= 2) {
      wasm_v128_store64_lane(c2, vacc23x01, 0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      wasm_v128_store64_lane(c3, vacc23x01, 1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      wasm_v128_store64_lane(c0, vacc01x01, 0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a0 = (const float*) ((uintptr_t) a0 - kc);
      wasm_v128_store64_lane(c1, vacc01x01, 1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      a1 = (const float*) ((uintptr_t) a1 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      wasm_v128_store32_lane(c2, vacc23x01, 0);
      wasm_v128_store32_lane(c3, vacc23x01, 2);
      wasm_v128_store32_lane(c0, vacc01x01, 0);
      wasm_v128_store32_lane(c1, vacc01x01, 2);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0c4 = wasm_v128_load32_zero(w);
    v128_t vacc0x1c4 = wasm_v128_load32_zero(w + 1);
    v128_t vacc1x0c4 = vacc0x0c4;
    v128_t vacc1x1c4 = vacc0x1c4;
    v128_t vacc2x0c4 = vacc0x0c4;
    v128_t vacc2x1c4 = vacc0x1c4;
    v128_t vacc3x0c4 = vacc0x0c4;
    v128_t vacc3x1c4 = vacc0x1c4;
    w += 2;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0c4 = __builtin_wasm_fma_f32x4(vacc0x0c4, va0, vb0);
      vacc0x1c4 = __builtin_wasm_fma_f32x4(vacc0x1c4, va0, vb1);
      vacc1x0c4 = __builtin_wasm_fma_f32x4(vacc1x0c4, va1, vb0);
      vacc1x1c4 = __builtin_wasm_fma_f32x4(vacc1x1c4, va1, vb1);
      vacc2x0c4 = __builtin_wasm_fma_f32x4(vacc2x0c4, va2, vb0);
      vacc2x1c4 = __builtin_wasm_fma_f32x4(vacc2x1c4, va2, vb1);
      vacc3x0c4 = __builtin_wasm_fma_f32x4(vacc3x0c4, va3, vb0);
      vacc3x1c4 = __builtin_wasm_fma_f32x4(vacc3x1c4, va3, vb1);
    }
    if XNN_UNLIKELY(k != 0) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      const v128_t va1 = wasm_v128_load(a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      const v128_t va2 = wasm_v128_load(a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      const v128_t va3 = wasm_v128_load(a3);
      a3 = (const float*) ((uintptr_t) a3 + k);

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      const v128_t vzero = wasm_f32x4_const_splat(0.0f);
      const v128_t vmask0 = wasm_f32x4_eq(vb0, vzero);
      const v128_t vmask1 = wasm_f32x4_eq(vb1, vzero);

      vacc0x0c4 = __builtin_wasm_fma_f32x4(vacc0x0c4, wasm_v128_andnot(va0, vmask0), vb0);
      vacc0x1c4 = __builtin_wasm_fma_f32x4(vacc0x1c4, wasm_v128_andnot(va0, vmask1), vb1);
      vacc1x0c4 = __builtin_wasm_fma_f32x4(vacc1x0c4, wasm_v128_andnot(va1, vmask0), vb0);
      vacc1x1c4 = __builtin_wasm_fma_f32x4(vacc1x1c4, wasm_v128_andnot(va1, vmask1), vb1);
      vacc2x0c4 = __builtin_wasm_fma_f32x4(vacc2x0c4, wasm_v128_andnot(va2, vmask0), vb0);
      vacc2x1c4 = __builtin_wasm_fma_f32x4(vacc2x1c4, wasm_v128_andnot(va2, vmask1), vb1);
      vacc3x0c4 = __builtin_wasm_fma_f32x4(vacc3x0c4, wasm_v128_andnot(va3, vmask0), vb0);
      vacc3x1c4 = __builtin_wasm_fma_f32x4(vacc3x1c4, wasm_v128_andnot(va3, vmask1), vb1);
    }

    const v128_t vacc0x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 2, 6, 3, 7));
    const v128_t vacc1x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 2, 6, 3, 7));
    const v128_t vacc2x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 2, 6, 3, 7));
    const v128_t vacc3x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 2, 6, 3, 7));

    v128_t vacc01x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 2, 3, 6, 7));
    v128_t vacc23x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 2, 3, 6, 7));


    if XNN_LIKELY(nc >= 2) {
      wasm_v128_store64_lane(c2, vacc23x01, 0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      wasm_v128_store64_lane(c3, vacc23x01, 1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      wasm_v128_store64_lane(c0, vacc01x01, 0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a0 = (const float*) ((uintptr_t) a0 - kc);
      wasm_v128_store64_lane(c1, vacc01x01, 1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      a1 = (const float*) ((uintptr_t) a1 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      wasm_v128_store32_lane(c2, vacc23x01, 0);
      wasm_v128_store32_lane(c3, vacc23x01, 2);
      wasm_v128_store32_lane(c0, vacc01x01, 0);
      wasm_v128_store32_lane(c1, vacc01x01, 2);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);

      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    vacc0x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x0123);
    vacc1x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc1x0123);
    vacc2x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc2x0123);
    vacc3x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc3x0123);
    vacc0x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x4567);
    vacc1x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc1x4567);
    vacc2x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc2x4567);
    vacc3x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc3x4567);

    vacc0x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x0123);
    vacc1x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc1x0123);
    vacc2x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc2x0123);
    vacc3x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc3x0123);
    vacc0x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x4567);
    vacc1x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc1x4567);
    vacc2x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc2x4567);
    vacc3x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc3x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);

      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);

      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;
      const v128_t va4 = wasm_v128_load(a4);
      a4 += 4;
      const v128_t va5 = wasm_v128_load(a5);
      a5 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
      const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
      const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c0, vb0123c0);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c0, vb0123c0);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c0, vb4567c0);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c0, vb4567c0);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
      const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
      const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c1, vb0123c1);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c1, vb0123c1);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c1, vb4567c1);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c1, vb4567c1);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
      const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
      const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c2, vb0123c2);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c2, vb0123c2);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c2, vb4567c2);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c2, vb4567c2);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
      const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
      const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c3, vb0123c3);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c3, vb0123c3);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c3, vb4567c3);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c3, vb4567c3);

      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;
        const v128_t va4 = wasm_v128_load32_splat(a4);
        a4 += 1;
        const v128_t va5 = wasm_v128_load32_splat(a5);
        a5 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4, vb0123);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5, vb0123);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4, vb4567);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    vacc0x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x0123);
    vacc1x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc1x0123);
    vacc2x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc2x0123);
    vacc3x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc3x0123);
    vacc4x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc4x0123);
    vacc5x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc5x0123);
    vacc0x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x4567);
    vacc1x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc1x4567);
    vacc2x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc2x4567);
    vacc3x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc3x4567);
    vacc4x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc4x4567);
    vacc5x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc5x4567);

    vacc0x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x0123);
    vacc1x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc1x0123);
    vacc2x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc2x0123);
    vacc3x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc3x0123);
    vacc4x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc4x0123);
    vacc5x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc5x0123);
    vacc0x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x4567);
    vacc1x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc1x4567);
    vacc2x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc2x4567);
    vacc3x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc3x4567);
    vacc4x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc4x4567);
    vacc5x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc5x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a5 = (const float*) ((uintptr_t) a5 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c5, vacc5x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

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
        wasm_v128_store64_lane(c5, vacc5x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;
      const v128_t va4 = wasm_v128_load(a4);
      a4 += 4;
      const v128_t va5 = wasm_v128_load(a5);
      a5 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
      const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
      const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c0, vb0123c0);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c0, vb0123c0);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c0, vb4567c0);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c0, vb4567c0);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
      const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
      const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c1, vb0123c1);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c1, vb0123c1);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c1, vb4567c1);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c1, vb4567c1);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
      const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
      const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c2, vb0123c2);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c2, vb0123c2);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c2, vb4567c2);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c2, vb4567c2);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
      const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
      const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c3, vb0123c3);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c3, vb0123c3);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c3, vb4567c3);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c3, vb4567c3);

      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;
        const v128_t va4 = wasm_v128_load32_splat(a4);
        a4 += 1;
        const v128_t va5 = wasm_v128_load32_splat(a5);
        a5 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4, vb0123);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5, vb0123);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4, vb4567);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc4x0123 = wasm_i32x4_max(vacc4x0123, vzero);
    vacc5x0123 = wasm_i32x4_max(vacc5x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);
    vacc4x4567 = wasm_i32x4_max(vacc4x4567, vzero);
    vacc5x4567 = wasm_i32x4_max(vacc5x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a5 = (const float*) ((uintptr_t) a5 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c5, vacc5x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

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
        wasm_v128_store64_lane(c5, vacc5x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;
      const v128_t va4 = wasm_v128_load(a4);
      a4 += 4;
      const v128_t va5 = wasm_v128_load(a5);
      a5 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
      const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
      const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c0, vb0123c0);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c0, vb0123c0);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c0, vb4567c0);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c0, vb4567c0);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
      const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
      const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c1, vb0123c1);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c1, vb0123c1);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c1, vb4567c1);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c1, vb4567c1);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
      const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
      const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c2, vb0123c2);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c2, vb0123c2);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c2, vb4567c2);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c2, vb4567c2);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
      const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
      const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
      vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
      vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
      vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
      vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c3, vb0123c3);
      vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c3, vb0123c3);
      vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
      vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
      vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
      vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);
      vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c3, vb4567c3);
      vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c3, vb4567c3);

      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;
        const v128_t va4 = wasm_v128_load32_splat(a4);
        a4 += 1;
        const v128_t va5 = wasm_v128_load32_splat(a5);
        a5 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4, vb0123);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5, vb0123);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4, vb4567);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5, vb4567);

        k -= sizeof(float);
      } while (k != 0);
    }


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a5 = (const float*) ((uintptr_t) a5 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c5, vacc5x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

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
        wasm_v128_store64_lane(c5, vacc5x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;

          vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
          vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x0123);
    vacc0x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x4567);

    vacc0x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x0123);
    vacc0x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;

          vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
          vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;

          vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
          vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    v128_t vacc0x0c4 = wasm_v128_load32_zero(w);
    v128_t vacc0x1c4 = wasm_v128_load32_zero(w + 1);
    v128_t vacc1x0c4 = vacc0x0c4;
    v128_t vacc1x1c4 = vacc0x1c4;
    v128_t vacc2x0c4 = vacc0x0c4;
    v128_t vacc2x1c4 = vacc0x1c4;
    v128_t vacc3x0c4 = vacc0x0c4;
    v128_t vacc3x1c4 = vacc0x1c4;
    w += 2;

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
      a += 4;

      size_t k = kc;
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;

        const v128_t vb0 = wasm_v128_load(w);
        const v128_t vb1 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0c4 = __builtin_wasm_fma_f32x4(vacc0x0c4, va0, vb0);
        vacc0x1c4 = __builtin_wasm_fma_f32x4(vacc0x1c4, va0, vb1);
        vacc1x0c4 = __builtin_wasm_fma_f32x4(vacc1x0c4, va1, vb0);
        vacc1x1c4 = __builtin_wasm_fma_f32x4(vacc1x1c4, va1, vb1);
        vacc2x0c4 = __builtin_wasm_fma_f32x4(vacc2x0c4, va2, vb0);
        vacc2x1c4 = __builtin_wasm_fma_f32x4(vacc2x1c4, va2, vb1);
        vacc3x0c4 = __builtin_wasm_fma_f32x4(vacc3x0c4, va3, vb0);
        vacc3x1c4 = __builtin_wasm_fma_f32x4(vacc3x1c4, va3, vb1);
      }
      if XNN_UNLIKELY(k != 0) {
        const v128_t va0 = wasm_v128_load(a0);
        const v128_t va1 = wasm_v128_load(a1);
        const v128_t va2 = wasm_v128_load(a2);
        const v128_t va3 = wasm_v128_load(a3);

        const v128_t vb0 = wasm_v128_load(w);
        const v128_t vb1 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t vzero = wasm_f32x4_const_splat(0.0f);
        const v128_t vmask0 = wasm_f32x4_eq(vb0, vzero);
        const v128_t vmask1 = wasm_f32x4_eq(vb1, vzero);

        vacc0x0c4 = __builtin_wasm_fma_f32x4(vacc0x0c4, wasm_v128_andnot(va0, vmask0), vb0);
        vacc0x1c4 = __builtin_wasm_fma_f32x4(vacc0x1c4, wasm_v128_andnot(va0, vmask1), vb1);
        vacc1x0c4 = __builtin_wasm_fma_f32x4(vacc1x0c4, wasm_v128_andnot(va1, vmask0), vb0);
        vacc1x1c4 = __builtin_wasm_fma_f32x4(vacc1x1c4, wasm_v128_andnot(va1, vmask1), vb1);
        vacc2x0c4 = __builtin_wasm_fma_f32x4(vacc2x0c4, wasm_v128_andnot(va2, vmask0), vb0);
        vacc2x1c4 = __builtin_wasm_fma_f32x4(vacc2x1c4, wasm_v128_andnot(va2, vmask1), vb1);
        vacc3x0c4 = __builtin_wasm_fma_f32x4(vacc3x0c4, wasm_v128_andnot(va3, vmask0), vb0);
        vacc3x1c4 = __builtin_wasm_fma_f32x4(vacc3x1c4, wasm_v128_andnot(va3, vmask1), vb1);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 2, 6, 3, 7));
    const v128_t vacc1x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 2, 6, 3, 7));
    const v128_t vacc2x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 2, 6, 3, 7));
    const v128_t vacc3x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 2, 6, 3, 7));

    v128_t vacc01x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 2, 3, 6, 7));
    v128_t vacc23x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 2, 3, 6, 7));

    vacc01x01 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc01x01);
    vacc23x01 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc23x01);

    vacc01x01 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc01x01);
    vacc23x01 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc23x01);

    if XNN_LIKELY(nc >= 2) {
      wasm_v128_store64_lane(c3, vacc23x01, 1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store64_lane(c2, vacc23x01, 0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store64_lane(c1, vacc01x01, 1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store64_lane(c0, vacc01x01, 0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      wasm_v128_store32_lane(c3, vacc23x01, 2);
      wasm_v128_store32_lane(c2, vacc23x01, 0);
      wasm_v128_store32_lane(c1, vacc01x01, 2);
      wasm_v128_store32_lane(c0, vacc01x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_4x2c4__wasmrelaxedsimd_fma(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    v128_t vacc0x0c4 = wasm_v128_load32_zero(w);
    v128_t vacc0x1c4 = wasm_v128_load32_zero(w + 1);
    v128_t vacc1x0c4 = vacc0x0c4;
    v128_t vacc1x1c4 = vacc0x1c4;
    v128_t vacc2x0c4 = vacc0x0c4;
    v128_t vacc2x1c4 = vacc0x1c4;
    v128_t vacc3x0c4 = vacc0x0c4;
    v128_t vacc3x1c4 = vacc0x1c4;
    w += 2;

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
      a += 4;

      size_t k = kc;
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;

        const v128_t vb0 = wasm_v128_load(w);
        const v128_t vb1 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0c4 = __builtin_wasm_fma_f32x4(vacc0x0c4, va0, vb0);
        vacc0x1c4 = __builtin_wasm_fma_f32x4(vacc0x1c4, va0, vb1);
        vacc1x0c4 = __builtin_wasm_fma_f32x4(vacc1x0c4, va1, vb0);
        vacc1x1c4 = __builtin_wasm_fma_f32x4(vacc1x1c4, va1, vb1);
        vacc2x0c4 = __builtin_wasm_fma_f32x4(vacc2x0c4, va2, vb0);
        vacc2x1c4 = __builtin_wasm_fma_f32x4(vacc2x1c4, va2, vb1);
        vacc3x0c4 = __builtin_wasm_fma_f32x4(vacc3x0c4, va3, vb0);
        vacc3x1c4 = __builtin_wasm_fma_f32x4(vacc3x1c4, va3, vb1);
      }
      if XNN_UNLIKELY(k != 0) {
        const v128_t va0 = wasm_v128_load(a0);
        const v128_t va1 = wasm_v128_load(a1);
        const v128_t va2 = wasm_v128_load(a2);
        const v128_t va3 = wasm_v128_load(a3);

        const v128_t vb0 = wasm_v128_load(w);
        const v128_t vb1 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t vzero = wasm_f32x4_const_splat(0.0f);
        const v128_t vmask0 = wasm_f32x4_eq(vb0, vzero);
        const v128_t vmask1 = wasm_f32x4_eq(vb1, vzero);

        vacc0x0c4 = __builtin_wasm_fma_f32x4(vacc0x0c4, wasm_v128_andnot(va0, vmask0), vb0);
        vacc0x1c4 = __builtin_wasm_fma_f32x4(vacc0x1c4, wasm_v128_andnot(va0, vmask1), vb1);
        vacc1x0c4 = __builtin_wasm_fma_f32x4(vacc1x0c4, wasm_v128_andnot(va1, vmask0), vb0);
        vacc1x1c4 = __builtin_wasm_fma_f32x4(vacc1x1c4, wasm_v128_andnot(va1, vmask1), vb1);
        vacc2x0c4 = __builtin_wasm_fma_f32x4(vacc2x0c4, wasm_v128_andnot(va2, vmask0), vb0);
        vacc2x1c4 = __builtin_wasm_fma_f32x4(vacc2x1c4, wasm_v128_andnot(va2, vmask1), vb1);
        vacc3x0c4 = __builtin_wasm_fma_f32x4(vacc3x0c4, wasm_v128_andnot(va3, vmask0), vb0);
        vacc3x1c4 = __builtin_wasm_fma_f32x4(vacc3x1c4, wasm_v128_andnot(va3, vmask1), vb1);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 2, 6, 3, 7));
    const v128_t vacc1x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 2, 6, 3, 7));
    const v128_t vacc2x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 2, 6, 3, 7));
    const v128_t vacc3x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 2, 6, 3, 7));

    v128_t vacc01x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 2, 3, 6, 7));
    v128_t vacc23x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 2, 3, 6, 7));


    if XNN_LIKELY(nc >= 2) {
      wasm_v128_store64_lane(c3, vacc23x01, 1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store64_lane(c2, vacc23x01, 0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store64_lane(c1, vacc01x01, 1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store64_lane(c0, vacc01x01, 0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      wasm_v128_store32_lane(c3, vacc23x01, 2);
      wasm_v128_store32_lane(c2, vacc23x01, 0);
      wasm_v128_store32_lane(c1, vacc01x01, 2);
      wasm_v128_store32_lane(c0, vacc01x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
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
      a += 4;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
        const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
        const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
        const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
        const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
        const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
        const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
        const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
        const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
        const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
        const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
        const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
        const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;
          const v128_t va1 = wasm_v128_load32_splat(a1);
          a1 += 1;
          const v128_t va2 = wasm_v128_load32_splat(a2);
          a2 += 1;
          const v128_t va3 = wasm_v128_load32_splat(a3);
          a3 += 1;

          vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
          vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
          vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
          vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
          vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
          vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
          vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
          vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x0123);
    vacc1x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc1x0123);
    vacc2x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc2x0123);
    vacc3x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc3x0123);
    vacc0x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x4567);
    vacc1x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc1x4567);
    vacc2x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc2x4567);
    vacc3x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc3x4567);

    vacc0x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x0123);
    vacc1x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc1x0123);
    vacc2x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc2x0123);
    vacc3x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc3x0123);
    vacc0x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x4567);
    vacc1x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc1x4567);
    vacc2x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc2x4567);
    vacc3x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc3x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
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
      a += 4;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
        const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
        const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
        const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
        const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
        const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
        const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
        const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
        const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
        const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
        const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
        const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
        const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;
          const v128_t va1 = wasm_v128_load32_splat(a1);
          a1 += 1;
          const v128_t va2 = wasm_v128_load32_splat(a2);
          a2 += 1;
          const v128_t va3 = wasm_v128_load32_splat(a3);
          a3 += 1;

          vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
          vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
          vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
          vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
          vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
          vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
          vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
          vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_4x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
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
      a += 4;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
        const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
        const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
        const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
        const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
        const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
        const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
        const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
        const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
        const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
        const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
        const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
        const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;
          const v128_t va1 = wasm_v128_load32_splat(a1);
          a1 += 1;
          const v128_t va2 = wasm_v128_load32_splat(a2);
          a2 += 1;
          const v128_t va3 = wasm_v128_load32_splat(a3);
          a3 += 1;

          vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
          vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
          vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
          vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
          vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
          vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
          vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
          vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
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
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;
        const v128_t va4 = wasm_v128_load(a4);
        a4 += 4;
        const v128_t va5 = wasm_v128_load(a5);
        a5 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
        const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
        const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
        const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
        const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
        const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c0, vb0123c0);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c0, vb0123c0);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c0, vb4567c0);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c0, vb4567c0);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
        const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
        const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
        const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
        const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
        const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c1, vb0123c1);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c1, vb0123c1);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c1, vb4567c1);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c1, vb4567c1);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
        const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
        const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
        const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
        const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
        const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c2, vb0123c2);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c2, vb0123c2);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c2, vb4567c2);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c2, vb4567c2);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
        const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
        const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
        const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
        const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
        const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c3, vb0123c3);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c3, vb0123c3);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c3, vb4567c3);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;
          const v128_t va1 = wasm_v128_load32_splat(a1);
          a1 += 1;
          const v128_t va2 = wasm_v128_load32_splat(a2);
          a2 += 1;
          const v128_t va3 = wasm_v128_load32_splat(a3);
          a3 += 1;
          const v128_t va4 = wasm_v128_load32_splat(a4);
          a4 += 1;
          const v128_t va5 = wasm_v128_load32_splat(a5);
          a5 += 1;

          vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
          vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
          vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
          vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
          vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
          vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
          vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
          vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);
          vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4, vb0123);
          vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4, vb4567);
          vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5, vb0123);
          vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x0123);
    vacc1x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc1x0123);
    vacc2x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc2x0123);
    vacc3x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc3x0123);
    vacc4x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc4x0123);
    vacc5x0123 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc5x0123);
    vacc0x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc0x4567);
    vacc1x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc1x4567);
    vacc2x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc2x4567);
    vacc3x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc3x4567);
    vacc4x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc4x4567);
    vacc5x4567 = __builtin_wasm_relaxed_max_f32x4(vmin, vacc5x4567);

    vacc0x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x0123);
    vacc1x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc1x0123);
    vacc2x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc2x0123);
    vacc3x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc3x0123);
    vacc4x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc4x0123);
    vacc5x0123 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc5x0123);
    vacc0x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc0x4567);
    vacc1x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc1x4567);
    vacc2x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc2x4567);
    vacc3x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc3x4567);
    vacc4x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc4x4567);
    vacc5x4567 = __builtin_wasm_relaxed_min_f32x4(vmax, vacc5x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c5, vacc5x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

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
        wasm_v128_store64_lane(c5, vacc5x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
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
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;
        const v128_t va4 = wasm_v128_load(a4);
        a4 += 4;
        const v128_t va5 = wasm_v128_load(a5);
        a5 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
        const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
        const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
        const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
        const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
        const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c0, vb0123c0);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c0, vb0123c0);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c0, vb4567c0);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c0, vb4567c0);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
        const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
        const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
        const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
        const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
        const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c1, vb0123c1);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c1, vb0123c1);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c1, vb4567c1);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c1, vb4567c1);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
        const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
        const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
        const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
        const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
        const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c2, vb0123c2);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c2, vb0123c2);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c2, vb4567c2);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c2, vb4567c2);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
        const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
        const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
        const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
        const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
        const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c3, vb0123c3);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c3, vb0123c3);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c3, vb4567c3);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;
          const v128_t va1 = wasm_v128_load32_splat(a1);
          a1 += 1;
          const v128_t va2 = wasm_v128_load32_splat(a2);
          a2 += 1;
          const v128_t va3 = wasm_v128_load32_splat(a3);
          a3 += 1;
          const v128_t va4 = wasm_v128_load32_splat(a4);
          a4 += 1;
          const v128_t va5 = wasm_v128_load32_splat(a5);
          a5 += 1;

          vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
          vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
          vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
          vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
          vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
          vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
          vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
          vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);
          vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4, vb0123);
          vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4, vb4567);
          vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5, vb0123);
          vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc4x0123 = wasm_i32x4_max(vacc4x0123, vzero);
    vacc5x0123 = wasm_i32x4_max(vacc5x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);
    vacc4x4567 = wasm_i32x4_max(vacc4x4567, vzero);
    vacc5x4567 = wasm_i32x4_max(vacc5x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c5, vacc5x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

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
        wasm_v128_store64_lane(c5, vacc5x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
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
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;
        const v128_t va4 = wasm_v128_load(a4);
        a4 += 4;
        const v128_t va5 = wasm_v128_load(a5);
        a5 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
        const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
        const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
        const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
        const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
        const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c0, vb0123c0);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c0, vb0123c0);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c0, vb0123c0);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c0, vb0123c0);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c0, vb0123c0);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c0, vb0123c0);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c0, vb4567c0);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c0, vb4567c0);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c0, vb4567c0);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c0, vb4567c0);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c0, vb4567c0);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c0, vb4567c0);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
        const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
        const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
        const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
        const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
        const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c1, vb0123c1);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c1, vb0123c1);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c1, vb0123c1);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c1, vb0123c1);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c1, vb0123c1);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c1, vb0123c1);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c1, vb4567c1);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c1, vb4567c1);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c1, vb4567c1);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c1, vb4567c1);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c1, vb4567c1);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c1, vb4567c1);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
        const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
        const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
        const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
        const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
        const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c2, vb0123c2);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c2, vb0123c2);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c2, vb0123c2);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c2, vb0123c2);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c2, vb0123c2);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c2, vb0123c2);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c2, vb4567c2);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c2, vb4567c2);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c2, vb4567c2);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c2, vb4567c2);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c2, vb4567c2);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c2, vb4567c2);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
        const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
        const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
        const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
        const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
        const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0c3, vb0123c3);
        vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1c3, vb0123c3);
        vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2c3, vb0123c3);
        vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3c3, vb0123c3);
        vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4c3, vb0123c3);
        vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5c3, vb0123c3);
        vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0c3, vb4567c3);
        vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1c3, vb4567c3);
        vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2c3, vb4567c3);
        vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3c3, vb4567c3);
        vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4c3, vb4567c3);
        vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;
          const v128_t va1 = wasm_v128_load32_splat(a1);
          a1 += 1;
          const v128_t va2 = wasm_v128_load32_splat(a2);
          a2 += 1;
          const v128_t va3 = wasm_v128_load32_splat(a3);
          a3 += 1;
          const v128_t va4 = wasm_v128_load32_splat(a4);
          a4 += 1;
          const v128_t va5 = wasm_v128_load32_splat(a5);
          a5 += 1;

          vacc0x0123 = __builtin_wasm_fma_f32x4(vacc0x0123, va0, vb0123);
          vacc0x4567 = __builtin_wasm_fma_f32x4(vacc0x4567, va0, vb4567);
          vacc1x0123 = __builtin_wasm_fma_f32x4(vacc1x0123, va1, vb0123);
          vacc1x4567 = __builtin_wasm_fma_f32x4(vacc1x4567, va1, vb4567);
          vacc2x0123 = __builtin_wasm_fma_f32x4(vacc2x0123, va2, vb0123);
          vacc2x4567 = __builtin_wasm_fma_f32x4(vacc2x4567, va2, vb4567);
          vacc3x0123 = __builtin_wasm_fma_f32x4(vacc3x0123, va3, vb0123);
          vacc3x4567 = __builtin_wasm_fma_f32x4(vacc3x4567, va3, vb4567);
          vacc4x0123 = __builtin_wasm_fma_f32x4(vacc4x0123, va4, vb0123);
          vacc4x4567 = __builtin_wasm_fma_f32x4(vacc4x4567, va4, vb4567);
          vacc5x0123 = __builtin_wasm_fma_f32x4(vacc5x0123, va5, vb0123);
          vacc5x4567 = __builtin_wasm_fma_f32x4(vacc5x4567, va5, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c5, vacc5x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

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
        wasm_v128_store64_lane(c5, vacc5x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_x32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load64_splat(params->wasmsimd.input_zero_point);
  const v128_t vmultiplier = wasm_v128_load64_splat(params->wasmsimd.multiplier);
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd.output_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    v128_t vacc0 = wasm_i16x8_load8x8(input);
    v128_t vacc1 = wasm_i16x8_load8x8(input + 8);
    v128_t vacc2 = wasm_i16x8_load8x8(input + 16);
    v128_t vacc3 = wasm_i16x8_load8x8(input + 24);
    input += 32;

    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);
    vacc2 = wasm_i16x8_sub(vinput_zero_point, vacc2);
    vacc3 = wasm_i16x8_sub(vinput_zero_point, vacc3);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vacc3 = wasm_i16x8_shl(vacc3, 7);

    vacc0 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc0, vmultiplier);
    vacc1 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc1, vmultiplier);
    vacc2 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc2, vmultiplier);
    vacc3 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc3, vmultiplier);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_i8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    v128_t vacc = wasm_i16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    v128_t vacc = wasm_i16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load64_splat(params->wasmsimd_arm.input_zero_point);
  const v128_t vpositive_multiplier = wasm_v128_load64_splat(params->wasmsimd_arm.positive_multiplier);
  const v128_t vnegative_multiplier = wasm_v128_load64_splat(params->wasmsimd_arm.negative_multiplier);
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd_arm.output_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    v128_t vx0 = wasm_v128_load(input);
    v128_t vx1 = wasm_v128_load(input + 16);
    input += 32;

    v128_t vacc0 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_low_i8x16(vx0));
    v128_t vacc1 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_high_i8x16(vx0));
    v128_t vmultiplier0 = wasm_i16x8_shr(vacc0, 15);
    v128_t vmultiplier1 = wasm_i16x8_shr(vacc1, 15);
    v128_t vacc2 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_low_i8x16(vx1));
    v128_t vacc3 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_high_i8x16(vx1));
    v128_t vmultiplier2 = wasm_i16x8_shr(vacc2, 15);
    v128_t vmultiplier3 = wasm_i16x8_shr(vacc3, 15);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier0);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier1);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier2);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier3);

    vacc0 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc0, vmultiplier0);
    vacc1 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc1, vmultiplier1);
    vacc2 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc2, vmultiplier2);
    vacc3 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_i8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const v128_t vx = wasm_i16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const v128_t vx = wasm_i16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load64_splat(params->wasmsimd_x86.input_zero_point);
  const v128_t vmultiplier_diff = wasm_v128_load64_splat(params->wasmsimd_x86.multiplier_diff);
  const v128_t vmultiplier_base = wasm_v128_load64_splat(params->wasmsimd_x86.multiplier_base);
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd_x86.output_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    v128_t vacc0 = wasm_i16x8_load8x8(input);
    v128_t vacc1 = wasm_i16x8_load8x8(input + 8);
    v128_t vacc2 = wasm_i16x8_load8x8(input + 16);
    v128_t vacc3 = wasm_i16x8_load8x8(input + 24);
    input += 32;

    v128_t vmultiplier0 = wasm_i16x8_gt(vacc0, vinput_zero_point);
    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    v128_t vmultiplier1 = wasm_i16x8_gt(vacc1, vinput_zero_point);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);
    v128_t vmultiplier2 = wasm_i16x8_gt(vacc2, vinput_zero_point);
    vacc2 = wasm_i16x8_sub(vinput_zero_point, vacc2);
    v128_t vmultiplier3 = wasm_i16x8_gt(vacc3, vinput_zero_point);
    vacc3 = wasm_i16x8_sub(vinput_zero_point, vacc3);

    vmultiplier0 = wasm_v128_and(vmultiplier0, vmultiplier_diff);
    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_v128_xor(vmultiplier0, vmultiplier_base);
    vmultiplier1 = wasm_v128_and(vmultiplier1, vmultiplier_diff);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_v128_xor(vmultiplier1, vmultiplier_base);
    vmultiplier2 = wasm_v128_and(vmultiplier2, vmultiplier_diff);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = wasm_v128_xor(vmultiplier2, vmultiplier_base);
    vmultiplier3 = wasm_v128_and(vmultiplier3, vmultiplier_diff);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = wasm_v128_xor(vmultiplier3, vmultiplier_base);

    vacc0 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc0, vmultiplier0);
    vacc1 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc1, vmultiplier1);
    vacc2 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc2, vmultiplier2);
    vacc3 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_i8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    v128_t vacc = wasm_i16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    v128_t vacc = wasm_i16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_x32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load64_splat(params->wasmsimd.input_zero_point);
  const v128_t vmultiplier = wasm_v128_load64_splat(params->wasmsimd.multiplier);
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd.output_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vacc0 = wasm_u16x8_load8x8(input);
    v128_t vacc1 = wasm_u16x8_load8x8(input + 8);
    v128_t vacc2 = wasm_u16x8_load8x8(input + 16);
    v128_t vacc3 = wasm_u16x8_load8x8(input + 24);
    input += 32;

    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);
    vacc2 = wasm_i16x8_sub(vinput_zero_point, vacc2);
    vacc3 = wasm_i16x8_sub(vinput_zero_point, vacc3);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vacc3 = wasm_i16x8_shl(vacc3, 7);

    vacc0 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc0, vmultiplier);
    vacc1 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc1, vmultiplier);
    vacc2 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc2, vmultiplier);
    vacc3 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc3, vmultiplier);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_u8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    v128_t vacc = wasm_u16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    v128_t vacc = wasm_u16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load64_splat(params->wasmsimd_arm.input_zero_point);
  const v128_t vpositive_multiplier = wasm_v128_load64_splat(params->wasmsimd_arm.positive_multiplier);
  const v128_t vnegative_multiplier = wasm_v128_load64_splat(params->wasmsimd_arm.negative_multiplier);
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd_arm.output_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vx0 = wasm_v128_load(input);
    v128_t vx1 = wasm_v128_load(input + 16);
    input += 32;

    v128_t vacc0 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_low_u8x16(vx0));
    v128_t vacc1 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_high_u8x16(vx0));
    v128_t vmultiplier0 = wasm_i16x8_shr(vacc0, 15);
    v128_t vmultiplier1 = wasm_i16x8_shr(vacc1, 15);
    v128_t vacc2 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_low_u8x16(vx1));
    v128_t vacc3 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_high_u8x16(vx1));
    v128_t vmultiplier2 = wasm_i16x8_shr(vacc2, 15);
    v128_t vmultiplier3 = wasm_i16x8_shr(vacc3, 15);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier0);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier1);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier2);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier3);

    vacc0 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc0, vmultiplier0);
    vacc1 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc1, vmultiplier1);
    vacc2 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc2, vmultiplier2);
    vacc3 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_u8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const v128_t vx = wasm_u16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    const v128_t vx = wasm_u16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = __builtin_wasm_laneselect_i16x8(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load64_splat(params->wasmsimd_x86.input_zero_point);
  const v128_t vmultiplier_diff = wasm_v128_load64_splat(params->wasmsimd_x86.multiplier_diff);
  const v128_t vmultiplier_base = wasm_v128_load64_splat(params->wasmsimd_x86.multiplier_base);
  const v128_t voutput_zero_point = wasm_v128_load64_splat(params->wasmsimd_x86.output_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vacc0 = wasm_u16x8_load8x8(input);
    v128_t vacc1 = wasm_u16x8_load8x8(input + 8);
    v128_t vacc2 = wasm_u16x8_load8x8(input + 16);
    v128_t vacc3 = wasm_u16x8_load8x8(input + 24);
    input += 32;

    v128_t vmultiplier0 = wasm_i16x8_gt(vacc0, vinput_zero_point);
    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    v128_t vmultiplier1 = wasm_i16x8_gt(vacc1, vinput_zero_point);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);
    v128_t vmultiplier2 = wasm_i16x8_gt(vacc2, vinput_zero_point);
    vacc2 = wasm_i16x8_sub(vinput_zero_point, vacc2);
    v128_t vmultiplier3 = wasm_i16x8_gt(vacc3, vinput_zero_point);
    vacc3 = wasm_i16x8_sub(vinput_zero_point, vacc3);

    vmultiplier0 = wasm_v128_and(vmultiplier0, vmultiplier_diff);
    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_v128_xor(vmultiplier0, vmultiplier_base);
    vmultiplier1 = wasm_v128_and(vmultiplier1, vmultiplier_diff);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_v128_xor(vmultiplier1, vmultiplier_base);
    vmultiplier2 = wasm_v128_and(vmultiplier2, vmultiplier_diff);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = wasm_v128_xor(vmultiplier2, vmultiplier_base);
    vmultiplier3 = wasm_v128_and(vmultiplier3, vmultiplier_diff);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = wasm_v128_xor(vmultiplier3, vmultiplier_base);

    vacc0 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc0, vmultiplier0);
    vacc1 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc1, vmultiplier1);
    vacc2 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc2, vmultiplier2);
    vacc3 = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_u8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    v128_t vacc = wasm_u16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    v128_t vacc = wasm_u16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = __builtin_wasm_relaxed_q15mulr_s_i16x8(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_x8_lut_ukernel__wasmpshufb_x32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vt0 = wasm_v128_load(table);
  const v128_t vt1 = wasm_v128_load(table + 16);
  const v128_t vt2 = wasm_v128_load(table + 32);
  const v128_t vt3 = wasm_v128_load(table + 48);
  const v128_t vt4 = wasm_v128_load(table + 64);
  const v128_t vt5 = wasm_v128_load(table + 80);
  const v128_t vt6 = wasm_v128_load(table + 96);
  const v128_t vt7 = wasm_v128_load(table + 112);
  const v128_t vt8 = wasm_v128_load(table + 128);
  const v128_t vt9 = wasm_v128_load(table + 144);
  const v128_t vtA = wasm_v128_load(table + 160);
  const v128_t vtB = wasm_v128_load(table + 176);
  const v128_t vtC = wasm_v128_load(table + 192);
  const v128_t vtD = wasm_v128_load(table + 208);
  const v128_t vtE = wasm_v128_load(table + 224);
  const v128_t vtF = wasm_v128_load(table + 240);

  const v128_t vtable0 = vt0;
  const v128_t vtable1 = wasm_v128_xor(vt0, vt1);
  const v128_t vtable2 = wasm_v128_xor(vt1, vt2);
  const v128_t vtable3 = wasm_v128_xor(vt2, vt3);
  const v128_t vtable4 = wasm_v128_xor(vt3, vt4);
  const v128_t vtable5 = wasm_v128_xor(vt4, vt5);
  const v128_t vtable6 = wasm_v128_xor(vt5, vt6);
  const v128_t vtable7 = wasm_v128_xor(vt6, vt7);
  const v128_t vtable8 = wasm_v128_xor(wasm_v128_xor(vt7, vt8), vtable0);
  const v128_t vtable9 = wasm_v128_xor(wasm_v128_xor(vt8, vt9), vtable1);
  const v128_t vtableA = wasm_v128_xor(wasm_v128_xor(vt9, vtA), vtable2);
  const v128_t vtableB = wasm_v128_xor(wasm_v128_xor(vtA, vtB), vtable3);
  const v128_t vtableC = wasm_v128_xor(wasm_v128_xor(vtB, vtC), vtable4);
  const v128_t vtableD = wasm_v128_xor(wasm_v128_xor(vtC, vtD), vtable5);
  const v128_t vtableE = wasm_v128_xor(wasm_v128_xor(vtD, vtE), vtable6);
  const v128_t vtableF = wasm_v128_xor(wasm_v128_xor(vtE, vtF), vtable7);

  const v128_t voffset = wasm_i8x16_const_splat(16);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vx0 = wasm_v128_load((const v128_t*) input);
    v128_t vx1 = wasm_v128_load((const v128_t*) (input + 16));
    input += 32;

    v128_t vy0 = __builtin_wasm_relaxed_swizzle_i8x16(vtable0, vx0);
    v128_t vy1 = __builtin_wasm_relaxed_swizzle_i8x16(vtable0, vx1);

    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtable1, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtable1, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtable2, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtable2, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtable3, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtable3, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtable4, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtable4, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtable5, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtable5, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtable6, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtable6, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtable7, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtable7, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtable8, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtable8, vx1));

    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtable9, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtable9, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtableA, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtableA, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtableB, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtableB, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtableC, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtableC, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtableD, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtableD, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtableE, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtableE, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, __builtin_wasm_relaxed_swizzle_i8x16(vtableF, vx0));
    vy1 = wasm_v128_xor(vy1, __builtin_wasm_relaxed_swizzle_i8x16(vtableF, vx1));

    wasm_v128_store(output, vy0);
    wasm_v128_store(output + 16, vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    v128_t vx = wasm_v128_load(input);
    input += 16;

    v128_t vy = __builtin_wasm_relaxed_swizzle_i8x16(vtable0, vx);

    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable1, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable2, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable3, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable4, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable5, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable6, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable7, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable8, vx));

    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable9, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableA, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableB, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableC, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableD, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableE, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableF, vx));

    wasm_v128_store(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vx = wasm_v128_load((const v128_t*) input);

    v128_t vy = __builtin_wasm_relaxed_swizzle_i8x16(vtable0, vx);

    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable1, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable2, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable3, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable4, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable5, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable6, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable7, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable8, vx));

    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtable9, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableA, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableB, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableC, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableD, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableE, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, __builtin_wasm_relaxed_swizzle_i8x16(vtableF, vx));

    if (batch & (8 * sizeof(uint8_t))) {
      wasm_v128_store64_lane(output, vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}