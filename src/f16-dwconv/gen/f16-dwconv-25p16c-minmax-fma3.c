// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass-fma3.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f16_dwconv_minmax_ukernel_25p16c__fma3(
    size_t channels,
    size_t output_width,
    const xnn_float16** input,
    const xnn_float16* weights,
    xnn_float16* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const xnn_float16* zero,
    const struct xnn_f16_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmin = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m256 vmax = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.max));
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = (const uint16_t*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = (const uint16_t*) input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = (const uint16_t*) input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = (const uint16_t*) input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = (const uint16_t*) input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = (const uint16_t*) input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint16_t* i9 = (const uint16_t*) input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != (const uint16_t*) zero) {
      i9 = (const uint16_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint16_t* i10 = (const uint16_t*) input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != (const uint16_t*) zero) {
      i10 = (const uint16_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint16_t* i11 = (const uint16_t*) input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != (const uint16_t*) zero) {
      i11 = (const uint16_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint16_t* i12 = (const uint16_t*) input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != (const uint16_t*) zero) {
      i12 = (const uint16_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint16_t* i13 = (const uint16_t*) input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != (const uint16_t*) zero) {
      i13 = (const uint16_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint16_t* i14 = (const uint16_t*) input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != (const uint16_t*) zero) {
      i14 = (const uint16_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint16_t* i15 = (const uint16_t*) input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != (const uint16_t*) zero) {
      i15 = (const uint16_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint16_t* i16 = (const uint16_t*) input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != (const uint16_t*) zero) {
      i16 = (const uint16_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint16_t* i17 = (const uint16_t*) input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != (const uint16_t*) zero) {
      i17 = (const uint16_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint16_t* i18 = (const uint16_t*) input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != (const uint16_t*) zero) {
      i18 = (const uint16_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint16_t* i19 = (const uint16_t*) input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != (const uint16_t*) zero) {
      i19 = (const uint16_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint16_t* i20 = (const uint16_t*) input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != (const uint16_t*) zero) {
      i20 = (const uint16_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint16_t* i21 = (const uint16_t*) input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != (const uint16_t*) zero) {
      i21 = (const uint16_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint16_t* i22 = (const uint16_t*) input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != (const uint16_t*) zero) {
      i22 = (const uint16_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint16_t* i23 = (const uint16_t*) input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != (const uint16_t*) zero) {
      i23 = (const uint16_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint16_t* i24 = (const uint16_t*) input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != (const uint16_t*) zero) {
      i24 = (const uint16_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const xnn_float16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = (const uint16_t*)weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 64)));
      const __m256 vk3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 72)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 8)));
      i4 += 16;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 80)));
      const __m256 vk4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 88)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 8)));
      i5 += 16;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 96)));
      const __m256 vk5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 104)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 8)));
      i6 += 16;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 112)));
      const __m256 vk6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 120)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      const __m256 vi7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 8)));
      i7 += 16;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 128)));
      const __m256 vk7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 136)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      const __m256 vi8x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8 + 8)));
      i8 += 16;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 144)));
      const __m256 vk8x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 152)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x89ABCDEF, vk8x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));
      const __m256 vi9x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i9 + 8)));
      i9 += 16;

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 160)));
      const __m256 vk9x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 168)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x89ABCDEF, vk9x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));
      const __m256 vi10x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i10 + 8)));
      i10 += 16;

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 176)));
      const __m256 vk10x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 184)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x89ABCDEF, vk10x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));
      const __m256 vi11x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i11 + 8)));
      i11 += 16;

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 192)));
      const __m256 vk11x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 200)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x89ABCDEF, vk11x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));
      const __m256 vi12x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i12 + 8)));
      i12 += 16;

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 208)));
      const __m256 vk12x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 216)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x89ABCDEF, vk12x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));
      const __m256 vi13x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i13 + 8)));
      i13 += 16;

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 224)));
      const __m256 vk13x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 232)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x89ABCDEF, vk13x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));
      const __m256 vi14x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i14 + 8)));
      i14 += 16;

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 240)));
      const __m256 vk14x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 248)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x89ABCDEF, vk14x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));
      const __m256 vi15x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i15 + 8)));
      i15 += 16;

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 256)));
      const __m256 vk15x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 264)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x89ABCDEF, vk15x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));
      const __m256 vi16x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i16 + 8)));
      i16 += 16;

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 272)));
      const __m256 vk16x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 280)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x89ABCDEF, vk16x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));
      const __m256 vi17x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i17 + 8)));
      i17 += 16;

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 288)));
      const __m256 vk17x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 296)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x89ABCDEF, vk17x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));
      const __m256 vi18x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i18 + 8)));
      i18 += 16;

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 304)));
      const __m256 vk18x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 312)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x89ABCDEF, vk18x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));
      const __m256 vi19x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i19 + 8)));
      i19 += 16;

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 320)));
      const __m256 vk19x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 328)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x89ABCDEF, vk19x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));
      const __m256 vi20x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i20 + 8)));
      i20 += 16;

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 336)));
      const __m256 vk20x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 344)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x89ABCDEF, vk20x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));
      const __m256 vi21x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i21 + 8)));
      i21 += 16;

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 352)));
      const __m256 vk21x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 360)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x89ABCDEF, vk21x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));
      const __m256 vi22x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i22 + 8)));
      i22 += 16;

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 368)));
      const __m256 vk22x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 376)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x89ABCDEF, vk22x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));
      const __m256 vi23x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i23 + 8)));
      i23 += 16;

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 384)));
      const __m256 vk23x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 392)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x89ABCDEF, vk23x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));
      const __m256 vi24x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i24 + 8)));
      i24 += 16;

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 400)));
      const __m256 vk24x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 408)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x89ABCDEF, vk24x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      w += 416;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      o += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));
      i9 += 8;

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 160)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));
      i10 += 8;

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 176)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));
      i11 += 8;

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 192)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));
      i12 += 8;

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 208)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));
      i13 += 8;

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 224)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));
      i14 += 8;

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 240)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));
      i15 += 8;

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 256)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));
      i16 += 8;

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 272)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));
      i17 += 8;

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 288)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));
      i18 += 8;

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 304)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));
      i19 += 8;

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 320)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));
      i20 += 8;

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 336)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));
      i21 += 8;

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 352)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));
      i22 += 8;

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 368)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));
      i23 += 8;

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 384)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));
      i24 += 8;

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 400)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 160)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 176)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 192)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 208)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 224)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 240)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 256)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 272)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 288)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 304)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 320)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 336)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 352)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 368)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 384)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 400)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        _mm_storeu_si32(o, vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    input_offset += input_pixel_stride;
    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}
