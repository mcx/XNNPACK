// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-avx2-mul32.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/unaligned.h"


void xnn_qu8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256i vk_zero_point = _mm256_set1_epi32(params->fp32_scalar.kernel_zero_point);
  XNN_FORCE_REALIZATION(vk_zero_point);

  const __m256 vscale = _mm256_set1_ps(params->fp32_scalar.scale);
  XNN_FORCE_REALIZATION(vscale);

  const __m256 voutput_max_less_zero_point = _mm256_set1_ps((int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point);
  const __m256i voutput_zero_point = _mm256_set1_epi16(params->fp32_scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->fp32_scalar.output_min);
  XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);

  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);
      __m256i vacc89ABCDEF = _mm256_loadu_si256((const __m256i*) ((const int32_t*) w + 8));


      const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
      const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi0x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
      const __m256i vk0x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 8 * sizeof(uint8_t)))), vk_zero_point);
      i0 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi0x89ABCDEF, vk0x89ABCDEF));

      const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
      const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi1x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
      const __m256i vk1x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 24 * sizeof(uint8_t)))), vk_zero_point);
      i1 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi1x89ABCDEF, vk1x89ABCDEF));

      const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
      const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi2x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
      const __m256i vk2x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 40 * sizeof(uint8_t)))), vk_zero_point);
      i2 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi2x89ABCDEF, vk2x89ABCDEF));

      const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
      const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi3x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
      const __m256i vk3x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 56 * sizeof(uint8_t)))), vk_zero_point);
      i3 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi3x89ABCDEF, vk3x89ABCDEF));

      const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
      const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi4x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
      const __m256i vk4x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 72 * sizeof(uint8_t)))), vk_zero_point);
      i4 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi4x89ABCDEF, vk4x89ABCDEF));

      const __m256i vi5x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i5));
      const __m256i vk5x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 80 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi5x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i5 + 8)));
      const __m256i vk5x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 88 * sizeof(uint8_t)))), vk_zero_point);
      i5 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi5x89ABCDEF, vk5x89ABCDEF));

      const __m256i vi6x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i6));
      const __m256i vk6x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 96 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi6x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i6 + 8)));
      const __m256i vk6x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 104 * sizeof(uint8_t)))), vk_zero_point);
      i6 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi6x89ABCDEF, vk6x89ABCDEF));

      const __m256i vi7x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i7));
      const __m256i vk7x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 112 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi7x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i7 + 8)));
      const __m256i vk7x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 120 * sizeof(uint8_t)))), vk_zero_point);
      i7 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi7x89ABCDEF, vk7x89ABCDEF));

      const __m256i vi8x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i8));
      const __m256i vk8x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 128 * sizeof(uint8_t)))), vk_zero_point);
      const __m256i vi8x89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (i8 + 8)));
      const __m256i vk8x89ABCDEF = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t) + 136 * sizeof(uint8_t)))), vk_zero_point);
      i8 += 16;

      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_mullo_epi32(vi8x89ABCDEF, vk8x89ABCDEF));

      w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t) + 144 * sizeof(uint8_t));

      __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
      __m256 vscaled89ABCDEF = _mm256_cvtepi32_ps(vacc89ABCDEF);

      vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale);
      vscaled89ABCDEF = _mm256_mul_ps(vscaled89ABCDEF, vscale);

      vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
      vscaled89ABCDEF = _mm256_min_ps(vscaled89ABCDEF, voutput_max_less_zero_point);

      vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);
      vacc89ABCDEF = _mm256_cvtps_epi32(vscaled89ABCDEF);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(vacc01234567, vacc89ABCDEF), voutput_zero_point);

      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8_t* k = (const uint8_t*) ((const int32_t*) w + 16);
      do {
        __m256i vacc01234567 = _mm256_loadu_si256((const __m256i*) w);


        const __m256i vi0x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i0));
        const __m256i vk0x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) k)), vk_zero_point);
        i0 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi0x01234567, vk0x01234567));

        const __m256i vi1x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i1));
        const __m256i vk1x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 16))), vk_zero_point);
        i1 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi1x01234567, vk1x01234567));

        const __m256i vi2x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i2));
        const __m256i vk2x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 32))), vk_zero_point);
        i2 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi2x01234567, vk2x01234567));

        const __m256i vi3x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i3));
        const __m256i vk3x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 48))), vk_zero_point);
        i3 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi3x01234567, vk3x01234567));

        const __m256i vi4x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i4));
        const __m256i vk4x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 64))), vk_zero_point);
        i4 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi4x01234567, vk4x01234567));

        const __m256i vi5x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i5));
        const __m256i vk5x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 80))), vk_zero_point);
        i5 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi5x01234567, vk5x01234567));

        const __m256i vi6x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i6));
        const __m256i vk6x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 96))), vk_zero_point);
        i6 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi6x01234567, vk6x01234567));

        const __m256i vi7x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i7));
        const __m256i vk7x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 112))), vk_zero_point);
        i7 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi7x01234567, vk7x01234567));

        const __m256i vi8x01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) i8));
        const __m256i vk8x01234567 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (k + 128))), vk_zero_point);
        i8 += 8;

        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_mullo_epi32(vi8x01234567, vk8x01234567));

        k += 8;

        __m256 vscaled01234567 = _mm256_cvtepi32_ps(vacc01234567);
        vscaled01234567 = _mm256_mul_ps(vscaled01234567, vscale);
        vscaled01234567 = _mm256_min_ps(vscaled01234567, voutput_max_less_zero_point);
        vacc01234567 = _mm256_cvtps_epi32(vscaled01234567);

        w = (const void*) ((const int32_t*) w + 8);

        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(_mm256_castsi256_si128(vacc01234567), _mm256_extracti128_si256(vacc01234567, 1)), _mm256_castsi256_si128(voutput_zero_point));

        __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

        vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    input_offset += input_pixel_stride;
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
