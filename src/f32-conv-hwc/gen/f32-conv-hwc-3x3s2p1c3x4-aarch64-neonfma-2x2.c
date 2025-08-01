// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-conv-hwc/3x3s2p1c3-neon-x2.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/conv.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f32_conv_hwc_ukernel_3x3s2p1c3x4__aarch64_neonfma_2x2(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_width_stride,
    const struct xnn_f32_minmax_params* restrict params) XNN_OOB_READS
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_decrement = input_width * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_decrement = output_width * output_width_stride - 4 * sizeof(float);
  const size_t output_height_increment = output_height_stride * 2 - round_up_po2(output_channels, 4) * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input +
    input_height_stride * (output_y_start * 2 /* vertical stride */ - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  float* o0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);
  float* o1 = (float*) ((uintptr_t) o0 + output_height_stride);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
  const float32x4_t vmax = vdupq_n_f32(params->scalar.max);

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if XNN_UNPREDICTABLE(input_y2 > input_height) {
      i1 = zero;
    }
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 > input_height) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 >= input_height) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(output_y + 2 > output_y_end) {
      o1 = o0;
    }

    const float* w = weights;
    size_t c = output_channels;
    do {
      // viMx0 = ( iM0c2, iM0c1, iM0c0, --- )
      float32x4_t vi0x0 = vmovq_n_f32(0.0f);
      float32x4_t vi1x0 = vmovq_n_f32(0.0f);
      float32x4_t vi2x0 = vmovq_n_f32(0.0f);
      float32x4_t vi3x0 = vmovq_n_f32(0.0f);
      float32x4_t vi4x0 = vmovq_n_f32(0.0f);

      size_t iw = input_width;
      for (; iw >= 4; iw -= 4) {
        float32x4_t vo0x0c0123 = vld1q_f32(w);
        float32x4_t vo1x0c0123 = vo0x0c0123;
        float32x4_t vo0x1c0123 = vo0x0c0123;
        float32x4_t vo1x1c0123 = vo0x0c0123;

        const float32x4_t vk00c0x0123 = vld1q_f32(w + 4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        const float32x4_t vi0x1 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1x1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2x1 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3x1 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4x1 = vld1q_f32(i4); i4 += 4;

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk00c0x0123, vget_low_f32(vi0x0), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk00c0x0123, vget_low_f32(vi2x0), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk00c0x0123, vget_high_f32(vi0x1), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk00c0x0123, vget_high_f32(vi2x1), 1);

        const float32x4_t vk10c0x0123 = vld1q_f32(w + 8);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk10c0x0123, vget_low_f32(vi1x0), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk10c0x0123, vget_low_f32(vi3x0), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk10c0x0123, vget_high_f32(vi1x1), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk10c0x0123, vget_high_f32(vi3x1), 1);

        const float32x4_t vk20c0x0123 = vld1q_f32(w + 12);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk20c0x0123, vget_low_f32(vi2x0), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk20c0x0123, vget_low_f32(vi4x0), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk20c0x0123, vget_high_f32(vi2x1), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk20c0x0123, vget_high_f32(vi4x1), 1);

        const float32x4_t vk00c1x0123 = vld1q_f32(w + 16);

        // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
        const float32x4_t vi0x2 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1x2 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2x2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3x2 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4x2 = vld1q_f32(i4); i4 += 4;

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk00c1x0123, vget_high_f32(vi0x0), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk00c1x0123, vget_high_f32(vi2x0), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk00c1x0123, vget_low_f32(vi0x2), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk00c1x0123, vget_low_f32(vi2x2), 0);

        const float32x4_t vk10c1x0123 = vld1q_f32(w + 20);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk10c1x0123, vget_high_f32(vi1x0), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk10c1x0123, vget_high_f32(vi3x0), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk10c1x0123, vget_low_f32(vi1x2), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk10c1x0123, vget_low_f32(vi3x2), 0);

        const float32x4_t vk20c1x0123 = vld1q_f32(w + 24);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk20c1x0123, vget_high_f32(vi2x0), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk20c1x0123, vget_high_f32(vi4x0), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk20c1x0123, vget_low_f32(vi2x2), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk20c1x0123, vget_low_f32(vi4x2), 0);

        const float32x4_t vk00c2x0123 = vld1q_f32(w + 28);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk00c2x0123, vget_high_f32(vi0x0), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk00c2x0123, vget_high_f32(vi2x0), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk00c2x0123, vget_low_f32(vi0x2), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk00c2x0123, vget_low_f32(vi2x2), 1);

        const float32x4_t vk10c2x0123 = vld1q_f32(w + 32);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk10c2x0123, vget_high_f32(vi1x0), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk10c2x0123, vget_high_f32(vi3x0), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk10c2x0123, vget_low_f32(vi1x2), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk10c2x0123, vget_low_f32(vi3x2), 1);

        const float32x4_t vk20c2x0123 = vld1q_f32(w + 36);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk20c2x0123, vget_high_f32(vi2x0), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk20c2x0123, vget_high_f32(vi4x0), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk20c2x0123, vget_low_f32(vi2x2), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk20c2x0123, vget_low_f32(vi4x2), 1);

        const float32x4_t vk01c0x0123 = vld1q_f32(w + 40);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk01c0x0123, vget_low_f32(vi0x1), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk01c0x0123, vget_low_f32(vi2x1), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk01c0x0123, vget_high_f32(vi0x2), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk01c0x0123, vget_high_f32(vi2x2), 0);

        const float32x4_t vk11c0x0123 = vld1q_f32(w + 44);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk11c0x0123, vget_low_f32(vi1x1), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk11c0x0123, vget_low_f32(vi3x1), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk11c0x0123, vget_high_f32(vi1x2), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk11c0x0123, vget_high_f32(vi3x2), 0);

        const float32x4_t vk21c0x0123 = vld1q_f32(w + 48);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk21c0x0123, vget_low_f32(vi2x1), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk21c0x0123, vget_low_f32(vi4x1), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk21c0x0123, vget_high_f32(vi2x2), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk21c0x0123, vget_high_f32(vi4x2), 0);

        const float32x4_t vk01c1x0123 = vld1q_f32(w + 52);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk01c1x0123, vget_low_f32(vi0x1), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk01c1x0123, vget_low_f32(vi2x1), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk01c1x0123, vget_high_f32(vi0x2), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk01c1x0123, vget_high_f32(vi2x2), 1);

        const float32x4_t vk11c1x0123 = vld1q_f32(w + 56);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk11c1x0123, vget_low_f32(vi1x1), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk11c1x0123, vget_low_f32(vi3x1), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk11c1x0123, vget_high_f32(vi1x2), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk11c1x0123, vget_high_f32(vi3x2), 1);

        const float32x4_t vk21c1x0123 = vld1q_f32(w + 60);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk21c1x0123, vget_low_f32(vi2x1), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk21c1x0123, vget_low_f32(vi4x1), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk21c1x0123, vget_high_f32(vi2x2), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk21c1x0123, vget_high_f32(vi4x2), 1);

        const float32x4_t vk01c2x0123 = vld1q_f32(w + 64);

        // viMx3 = ( iM4c2, iM4c1, iM4c0, iM3c2 )
        const float32x4_t vi0x3 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1x3 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2x3 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3x3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4x3 = vld1q_f32(i4); i4 += 4;

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk01c2x0123, vget_high_f32(vi0x1), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk01c2x0123, vget_high_f32(vi2x1), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk01c2x0123, vget_low_f32(vi0x3), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk01c2x0123, vget_low_f32(vi2x3), 0);

        const float32x4_t vk11c2x0123 = vld1q_f32(w + 68);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk11c2x0123, vget_high_f32(vi1x1), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk11c2x0123, vget_high_f32(vi3x1), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk11c2x0123, vget_low_f32(vi1x3), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk11c2x0123, vget_low_f32(vi3x3), 0);

        const float32x4_t vk21c2x0123 = vld1q_f32(w + 72);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk21c2x0123, vget_high_f32(vi2x1), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk21c2x0123, vget_high_f32(vi4x1), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk21c2x0123, vget_low_f32(vi2x3), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk21c2x0123, vget_low_f32(vi4x3), 0);

        const float32x4_t vk02c0x0123 = vld1q_f32(w + 76);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk02c0x0123, vget_high_f32(vi0x1), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk02c0x0123, vget_high_f32(vi2x1), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk02c0x0123, vget_low_f32(vi0x3), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk02c0x0123, vget_low_f32(vi2x3), 1);

        const float32x4_t vk12c0x0123 = vld1q_f32(w + 80);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk12c0x0123, vget_high_f32(vi1x1), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk12c0x0123, vget_high_f32(vi3x1), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk12c0x0123, vget_low_f32(vi1x3), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk12c0x0123, vget_low_f32(vi3x3), 1);

        const float32x4_t vk22c0x0123 = vld1q_f32(w + 84);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk22c0x0123, vget_high_f32(vi2x1), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk22c0x0123, vget_high_f32(vi4x1), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk22c0x0123, vget_low_f32(vi2x3), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk22c0x0123, vget_low_f32(vi4x3), 1);

        const float32x4_t vk02c1x0123 = vld1q_f32(w + 88);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk02c1x0123, vget_low_f32(vi0x2), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk02c1x0123, vget_low_f32(vi2x2), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk02c1x0123, vget_high_f32(vi0x3), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk02c1x0123, vget_high_f32(vi2x3), 0);

        const float32x4_t vk12c1x0123 = vld1q_f32(w + 92);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk12c1x0123, vget_low_f32(vi1x2), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk12c1x0123, vget_low_f32(vi3x2), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk12c1x0123, vget_high_f32(vi1x3), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk12c1x0123, vget_high_f32(vi3x3), 0);

        const float32x4_t vk22c1x0123 = vld1q_f32(w + 96);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk22c1x0123, vget_low_f32(vi2x2), 0);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk22c1x0123, vget_low_f32(vi4x2), 0);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk22c1x0123, vget_high_f32(vi2x3), 0);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk22c1x0123, vget_high_f32(vi4x3), 0);

        const float32x4_t vk02c2x0123 = vld1q_f32(w + 100);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk02c2x0123, vget_low_f32(vi0x2), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk02c2x0123, vget_low_f32(vi2x2), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk02c2x0123, vget_high_f32(vi0x3), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk02c2x0123, vget_high_f32(vi2x3), 1);

        const float32x4_t vk12c2x0123 = vld1q_f32(w + 104);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk12c2x0123, vget_low_f32(vi1x2), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk12c2x0123, vget_low_f32(vi3x2), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk12c2x0123, vget_high_f32(vi1x3), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk12c2x0123, vget_high_f32(vi3x3), 1);

        const float32x4_t vk22c2x0123 = vld1q_f32(w + 108);

        vo0x0c0123 = vfmaq_lane_f32(vo0x0c0123, vk22c2x0123, vget_low_f32(vi2x2), 1);
        vo1x0c0123 = vfmaq_lane_f32(vo1x0c0123, vk22c2x0123, vget_low_f32(vi4x2), 1);

        vo0x1c0123 = vfmaq_lane_f32(vo0x1c0123, vk22c2x0123, vget_high_f32(vi2x3), 1);
        vo1x1c0123 = vfmaq_lane_f32(vo1x1c0123, vk22c2x0123, vget_high_f32(vi4x3), 1);

        vi0x0 = vi0x3;
        vi1x0 = vi1x3;
        vi2x0 = vi2x3;
        vi3x0 = vi3x3;
        vi4x0 = vi4x3;


        vo0x0c0123 = vmaxq_f32(vo0x0c0123, vmin);
        vo1x0c0123 = vmaxq_f32(vo1x0c0123, vmin);

        vo0x1c0123 = vmaxq_f32(vo0x1c0123, vmin);
        vo1x1c0123 = vmaxq_f32(vo1x1c0123, vmin);

        vo0x0c0123 = vminq_f32(vo0x0c0123, vmax);
        vo1x0c0123 = vminq_f32(vo1x0c0123, vmax);

        vo0x1c0123 = vminq_f32(vo0x1c0123, vmax);
        vo1x1c0123 = vminq_f32(vo1x1c0123, vmax);

        if XNN_LIKELY(c >= 4) {
          vst1q_f32(o1, vo1x0c0123);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride);
          vst1q_f32(o0, vo0x0c0123);
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);

          vst1q_f32(o1, vo1x1c0123);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride);
          vst1q_f32(o0, vo0x1c0123);
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        } else {
          float* o0_tmp = o0;
          float* o1_tmp = o1;
          float32x2_t vo0x0c01 = vget_low_f32(vo0x0c0123);
          float32x2_t vo1x0c01 = vget_low_f32(vo1x0c0123);
          float32x2_t vo0x1c01 = vget_low_f32(vo0x1c0123);
          float32x2_t vo1x1c01 = vget_low_f32(vo1x1c0123);
          if (c & 2) {
            vst1_f32((float*) ((uintptr_t) o1_tmp + output_width_stride), vo1x1c01);
            vo1x1c01 = vget_high_f32(vo1x1c0123);
            vst1_f32((float*) ((uintptr_t) o0_tmp + output_width_stride), vo0x1c01);
            vo0x1c01 = vget_high_f32(vo0x1c0123);

            vst1_f32(o1_tmp, vo1x0c01); o1_tmp += 2;
            vo1x0c01 = vget_high_f32(vo1x0c0123);
            vst1_f32(o0_tmp, vo0x0c01); o0_tmp += 2;
            vo0x0c01 = vget_high_f32(vo0x0c0123);
          }
          if (c & 1) {
            vst1_lane_f32(o1_tmp, vo1x0c01, 0);
            vst1_lane_f32(o0_tmp, vo0x0c01, 0);

            vst1_lane_f32((float*) ((uintptr_t) o1_tmp + output_width_stride), vo1x1c01, 0);
            vst1_lane_f32((float*) ((uintptr_t) o0_tmp + output_width_stride), vo0x1c01, 0);
          }

          o0 = (float*) ((uintptr_t) o0 + output_width_stride * 2);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride * 2);
        }
      }
      assert(iw < 4);
      if XNN_UNLIKELY(iw & 2) {
        float32x4_t vo0c0123 = vld1q_f32(w);
        float32x4_t vo1c0123 = vo0c0123;

        const float32x4_t vk00c0x0123 = vld1q_f32(w + 4);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk00c0x0123, vget_low_f32(vi0x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk00c0x0123, vget_low_f32(vi2x0), 1);

        const float32x4_t vk10c0x0123 = vld1q_f32(w + 8);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk10c0x0123, vget_low_f32(vi1x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk10c0x0123, vget_low_f32(vi3x0), 1);

        const float32x4_t vk20c0x0123 = vld1q_f32(w + 12);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk20c0x0123, vget_low_f32(vi2x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk20c0x0123, vget_low_f32(vi4x0), 1);

        const float32x4_t vk00c1x0123 = vld1q_f32(w + 16);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk00c1x0123, vget_high_f32(vi0x0), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk00c1x0123, vget_high_f32(vi2x0), 0);

        const float32x4_t vk10c1x0123 = vld1q_f32(w + 20);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk10c1x0123, vget_high_f32(vi1x0), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk10c1x0123, vget_high_f32(vi3x0), 0);

        const float32x4_t vk20c1x0123 = vld1q_f32(w + 24);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk20c1x0123, vget_high_f32(vi2x0), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk20c1x0123, vget_high_f32(vi4x0), 0);

        const float32x4_t vk00c2x0123 = vld1q_f32(w + 28);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk00c2x0123, vget_high_f32(vi0x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk00c2x0123, vget_high_f32(vi2x0), 1);

        const float32x4_t vk10c2x0123 = vld1q_f32(w + 32);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk10c2x0123, vget_high_f32(vi1x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk10c2x0123, vget_high_f32(vi3x0), 1);

        const float32x4_t vk20c2x0123 = vld1q_f32(w + 36);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk20c2x0123, vget_high_f32(vi2x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk20c2x0123, vget_high_f32(vi4x0), 1);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        const float32x4_t vi0x1 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1x1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2x1 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3x1 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4x1 = vld1q_f32(i4); i4 += 4;

        const float32x4_t vk01c0x0123 = vld1q_f32(w + 40);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk01c0x0123, vget_low_f32(vi0x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk01c0x0123, vget_low_f32(vi2x1), 0);

        const float32x4_t vk11c0x0123 = vld1q_f32(w + 44);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk11c0x0123, vget_low_f32(vi1x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk11c0x0123, vget_low_f32(vi3x1), 0);

        const float32x4_t vk21c0x0123 = vld1q_f32(w + 48);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk21c0x0123, vget_low_f32(vi2x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk21c0x0123, vget_low_f32(vi4x1), 0);

        const float32x4_t vk01c1x0123 = vld1q_f32(w + 52);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk01c1x0123, vget_low_f32(vi0x1), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk01c1x0123, vget_low_f32(vi2x1), 1);

        const float32x4_t vk11c1x0123 = vld1q_f32(w + 56);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk11c1x0123, vget_low_f32(vi1x1), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk11c1x0123, vget_low_f32(vi3x1), 1);

        const float32x4_t vk21c1x0123 = vld1q_f32(w + 60);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk21c1x0123, vget_low_f32(vi2x1), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk21c1x0123, vget_low_f32(vi4x1), 1);

        const float32x4_t vk01c2x0123 = vld1q_f32(w + 64);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk01c2x0123, vget_high_f32(vi0x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk01c2x0123, vget_high_f32(vi2x1), 0);

        const float32x4_t vk11c2x0123 = vld1q_f32(w + 68);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk11c2x0123, vget_high_f32(vi1x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk11c2x0123, vget_high_f32(vi3x1), 0);

        const float32x4_t vk21c2x0123 = vld1q_f32(w + 72);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk21c2x0123, vget_high_f32(vi2x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk21c2x0123, vget_high_f32(vi4x1), 0);

        const float32x4_t vk02c0x0123 = vld1q_f32(w + 76);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk02c0x0123, vget_high_f32(vi0x1), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk02c0x0123, vget_high_f32(vi2x1), 1);

        const float32x4_t vk12c0x0123 = vld1q_f32(w + 80);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk12c0x0123, vget_high_f32(vi1x1), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk12c0x0123, vget_high_f32(vi3x1), 1);

        const float32x4_t vk22c0x0123 = vld1q_f32(w + 84);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk22c0x0123, vget_high_f32(vi2x1), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk22c0x0123, vget_high_f32(vi4x1), 1);

        // viMx2 = ( iM2c2, iM2c1 )
        const float32x2_t vi0x2 = vld1_f32(i0); i0 += 2;
        const float32x2_t vi1x2 = vld1_f32(i1); i1 += 2;
        const float32x2_t vi2x2 = vld1_f32(i2); i2 += 2;
        const float32x2_t vi3x2 = vld1_f32(i3); i3 += 2;
        const float32x2_t vi4x2 = vld1_f32(i4); i4 += 2;

        const float32x4_t vk02c1x0123 = vld1q_f32(w + 88);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk02c1x0123, vi0x2, 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk02c1x0123, vi2x2, 0);

        const float32x4_t vk12c1x0123 = vld1q_f32(w + 92);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk12c1x0123, vi1x2, 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk12c1x0123, vi3x2, 0);

        const float32x4_t vk22c1x0123 = vld1q_f32(w + 96);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk22c1x0123, vi2x2, 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk22c1x0123, vi4x2, 0);

        const float32x4_t vk02c2x0123 = vld1q_f32(w + 100);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk02c2x0123, vi0x2, 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk02c2x0123, vi2x2, 1);

        const float32x4_t vk12c2x0123 = vld1q_f32(w + 104);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk12c2x0123, vi1x2, 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk12c2x0123, vi3x2, 1);

        const float32x4_t vk22c2x0123 = vld1q_f32(w + 108);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk22c2x0123, vi2x2, 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk22c2x0123, vi4x2, 1);

        vi0x0 = vcombine_f32(vget_high_f32(vi0x1), vi0x2);
        vi1x0 = vcombine_f32(vget_high_f32(vi1x1), vi1x2);
        vi2x0 = vcombine_f32(vget_high_f32(vi2x1), vi2x2);
        vi3x0 = vcombine_f32(vget_high_f32(vi3x1), vi3x2);
        vi4x0 = vcombine_f32(vget_high_f32(vi4x1), vi4x2);


        vo0c0123 = vmaxq_f32(vo0c0123, vmin);
        vo1c0123 = vmaxq_f32(vo1c0123, vmin);

        vo0c0123 = vminq_f32(vo0c0123, vmax);
        vo1c0123 = vminq_f32(vo1c0123, vmax);

        if XNN_LIKELY(c >= 4) {
          vst1q_f32(o1, vo1c0123);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride);
          vst1q_f32(o0, vo0c0123);
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        } else {
          float* o0_tmp = o0;
          float* o1_tmp = o1;
          float32x2_t vo0c01 = vget_low_f32(vo0c0123);
          float32x2_t vo1c01 = vget_low_f32(vo1c0123);
          if (c & 2) {
            vst1_f32(o1_tmp, vo1c01); o1_tmp += 2;
            vo1c01 = vget_high_f32(vo1c0123);
            vst1_f32(o0_tmp, vo0c01); o0_tmp += 2;
            vo0c01 = vget_high_f32(vo0c0123);
          }
          if (c & 1) {
            vst1_lane_f32(o1_tmp, vo1c01, 0);
            vst1_lane_f32(o0_tmp, vo0c01, 0);
          }

          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride);
        }
      }
      if XNN_UNLIKELY(iw & 1) {
        float32x4_t vo0c0123 = vld1q_f32(w);
        float32x4_t vo1c0123 = vo0c0123;

        const float32x4_t vk00c0x0123 = vld1q_f32(w + 4);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk00c0x0123, vget_low_f32(vi0x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk00c0x0123, vget_low_f32(vi2x0), 1);

        const float32x4_t vk10c0x0123 = vld1q_f32(w + 8);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk10c0x0123, vget_low_f32(vi1x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk10c0x0123, vget_low_f32(vi3x0), 1);

        const float32x4_t vk20c0x0123 = vld1q_f32(w + 12);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk20c0x0123, vget_low_f32(vi2x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk20c0x0123, vget_low_f32(vi4x0), 1);

        const float32x4_t vk00c1x0123 = vld1q_f32(w + 16);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk00c1x0123, vget_high_f32(vi0x0), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk00c1x0123, vget_high_f32(vi2x0), 0);

        const float32x4_t vk10c1x0123 = vld1q_f32(w + 20);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk10c1x0123, vget_high_f32(vi1x0), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk10c1x0123, vget_high_f32(vi3x0), 0);

        const float32x4_t vk20c1x0123 = vld1q_f32(w + 24);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk20c1x0123, vget_high_f32(vi2x0), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk20c1x0123, vget_high_f32(vi4x0), 0);

        const float32x4_t vk00c2x0123 = vld1q_f32(w + 28);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk00c2x0123, vget_high_f32(vi0x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk00c2x0123, vget_high_f32(vi2x0), 1);

        const float32x4_t vk10c2x0123 = vld1q_f32(w + 32);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk10c2x0123, vget_high_f32(vi1x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk10c2x0123, vget_high_f32(vi3x0), 1);

        const float32x4_t vk20c2x0123 = vld1q_f32(w + 36);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk20c2x0123, vget_high_f32(vi2x0), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk20c2x0123, vget_high_f32(vi4x0), 1);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        const float32x4_t vi0x1 = vld1q_f32(i0); i0 += 3;
        const float32x4_t vi1x1 = vld1q_f32(i1); i1 += 3;
        const float32x4_t vi2x1 = vld1q_f32(i2); i2 += 3;
        const float32x4_t vi3x1 = vld1q_f32(i3); i3 += 3;
        const float32x4_t vi4x1 = vld1q_f32(i4); i4 += 3;

        const float32x4_t vk01c0x0123 = vld1q_f32(w + 40);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk01c0x0123, vget_low_f32(vi0x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk01c0x0123, vget_low_f32(vi2x1), 0);

        const float32x4_t vk11c0x0123 = vld1q_f32(w + 44);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk11c0x0123, vget_low_f32(vi1x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk11c0x0123, vget_low_f32(vi3x1), 0);

        const float32x4_t vk21c0x0123 = vld1q_f32(w + 48);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk21c0x0123, vget_low_f32(vi2x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk21c0x0123, vget_low_f32(vi4x1), 0);

        const float32x4_t vk01c1x0123 = vld1q_f32(w + 52);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk01c1x0123, vget_low_f32(vi0x1), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk01c1x0123, vget_low_f32(vi2x1), 1);

        const float32x4_t vk11c1x0123 = vld1q_f32(w + 56);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk11c1x0123, vget_low_f32(vi1x1), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk11c1x0123, vget_low_f32(vi3x1), 1);

        const float32x4_t vk21c1x0123 = vld1q_f32(w + 60);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk21c1x0123, vget_low_f32(vi2x1), 1);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk21c1x0123, vget_low_f32(vi4x1), 1);

        const float32x4_t vk01c2x0123 = vld1q_f32(w + 64);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk01c2x0123, vget_high_f32(vi0x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk01c2x0123, vget_high_f32(vi2x1), 0);

        const float32x4_t vk11c2x0123 = vld1q_f32(w + 68);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk11c2x0123, vget_high_f32(vi1x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk11c2x0123, vget_high_f32(vi3x1), 0);

        const float32x4_t vk21c2x0123 = vld1q_f32(w + 72);

        vo0c0123 = vfmaq_lane_f32(vo0c0123, vk21c2x0123, vget_high_f32(vi2x1), 0);
        vo1c0123 = vfmaq_lane_f32(vo1c0123, vk21c2x0123, vget_high_f32(vi4x1), 0);


        vo0c0123 = vmaxq_f32(vo0c0123, vmin);
        vo1c0123 = vmaxq_f32(vo1c0123, vmin);

        vo0c0123 = vminq_f32(vo0c0123, vmax);
        vo1c0123 = vminq_f32(vo1c0123, vmax);

        if XNN_LIKELY(c >= 4) {
          vst1q_f32(o1, vo1c0123);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride);
          vst1q_f32(o0, vo0c0123);
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
        } else {
          float* o0_tmp = o0;
          float* o1_tmp = o1;
          float32x2_t vo0c01 = vget_low_f32(vo0c0123);
          float32x2_t vo1c01 = vget_low_f32(vo1c0123);
          if (c & 2) {
            vst1_f32(o1_tmp, vo1c01); o1_tmp += 2;
            vo1c01 = vget_high_f32(vo1c0123);
            vst1_f32(o0_tmp, vo0c01); o0_tmp += 2;
            vo0c01 = vget_high_f32(vo0c0123);
          }
          if (c & 1) {
            vst1_lane_f32(o1_tmp, vo1c01, 0);
            vst1_lane_f32(o0_tmp, vo0c01, 0);
          }
          o0 = (float*) ((uintptr_t) o0 + output_width_stride);
          o1 = (float*) ((uintptr_t) o1 + output_width_stride);
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels
      o0 = (float*) ((uintptr_t) o0 - output_channel_decrement);
      o1 = (float*) ((uintptr_t) o1 - output_channel_decrement);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_decrement);
      i1 = (const float*) ((uintptr_t) i1 - input_width_decrement);
      i2 = (const float*) ((uintptr_t) i2 - input_width_decrement);
      i3 = (const float*) ((uintptr_t) i3 - input_width_decrement);
      i4 = (const float*) ((uintptr_t) i4 - input_width_decrement);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers back to the position of the first channel, and forward to the next block of rows
    o0 = (float*) ((uintptr_t) o0 + output_height_increment);
    o1 = (float*) ((uintptr_t) o1 + output_height_increment);
    // Move input pointers forward to the next four rows
    i0 = i4;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  }
}
