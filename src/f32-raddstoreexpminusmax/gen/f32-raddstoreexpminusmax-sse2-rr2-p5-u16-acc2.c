// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/raddstoreexpminusmax.h"


void xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.8000FEp23f);
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E400p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(-0x1.7F7D1Cp-20f);
  const __m128 vc5 = _mm_set1_ps(0x1.0F9F9Cp-7f);
  const __m128 vc4 = _mm_set1_ps(0x1.573A1Ap-5f);
  const __m128 vc3 = _mm_set1_ps(0x1.555A80p-3f);
  const __m128 vc2 = _mm_set1_ps(0x1.FFFDC6p-2f);
  const __m128 vc1 = _mm_set1_ps(0x1.FFFFF6p-1f);
  const __m128 vdenorm_cutoff = _mm_set1_ps(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const __m128 vi_max = _mm_load1_ps(max);

  __m128 vacc0 = _mm_setzero_ps();
  __m128 vacc1 = _mm_setzero_ps();
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    // Load 16 (4x4) inputs at a time.
    const __m128 vi0 = _mm_loadu_ps(input);
    const __m128 vi1 = _mm_loadu_ps(input + 4);
    const __m128 vi2 = _mm_loadu_ps(input + 8);
    const __m128 vi3 = _mm_loadu_ps(input + 12);
    input += 16;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m128 vx0 = _mm_sub_ps(vi0, vi_max);
    const __m128 vx1 = _mm_sub_ps(vi1, vi_max);
    const __m128 vx2 = _mm_sub_ps(vi2, vi_max);
    const __m128 vx3 = _mm_sub_ps(vi3, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m128 vn0 = _mm_add_ps(_mm_mul_ps(vx0, vlog2e), vmagic_bias);
    __m128 vn1 = _mm_add_ps(_mm_mul_ps(vx1, vlog2e), vmagic_bias);
    __m128 vn2 = _mm_add_ps(_mm_mul_ps(vx2, vlog2e), vmagic_bias);
    __m128 vn3 = _mm_add_ps(_mm_mul_ps(vx3, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m128 vs0 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn0), 23));
    const __m128 vs1 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn1), 23));
    const __m128 vs2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn2), 23));
    const __m128 vs3 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn3), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn0 = _mm_sub_ps(vn0, vmagic_bias);
    vn1 = _mm_sub_ps(vn1, vmagic_bias);
    vn2 = _mm_sub_ps(vn2, vmagic_bias);
    vn3 = _mm_sub_ps(vn3, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt0 = _mm_add_ps(_mm_mul_ps(vn0, vminus_ln2_hi), vx0);
    __m128 vt1 = _mm_add_ps(_mm_mul_ps(vn1, vminus_ln2_hi), vx1);
    __m128 vt2 = _mm_add_ps(_mm_mul_ps(vn2, vminus_ln2_hi), vx2);
    __m128 vt3 = _mm_add_ps(_mm_mul_ps(vn3, vminus_ln2_hi), vx3);

    vt0 = _mm_add_ps(_mm_mul_ps(vn0, vminus_ln2_lo), vt0);
    vt1 = _mm_add_ps(_mm_mul_ps(vn1, vminus_ln2_lo), vt1);
    vt2 = _mm_add_ps(_mm_mul_ps(vn2, vminus_ln2_lo), vt2);
    vt3 = _mm_add_ps(_mm_mul_ps(vn3, vminus_ln2_lo), vt3);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp0 = _mm_add_ps(_mm_mul_ps(vc5, vt0), vc4);
    __m128 vp1 = _mm_add_ps(_mm_mul_ps(vc5, vt1), vc4);
    __m128 vp2 = _mm_add_ps(_mm_mul_ps(vc5, vt2), vc4);
    __m128 vp3 = _mm_add_ps(_mm_mul_ps(vc5, vt3), vc4);

    vp0 = _mm_add_ps(_mm_mul_ps(vp0, vt0), vc3);
    vp1 = _mm_add_ps(_mm_mul_ps(vp1, vt1), vc3);
    vp2 = _mm_add_ps(_mm_mul_ps(vp2, vt2), vc3);
    vp3 = _mm_add_ps(_mm_mul_ps(vp3, vt3), vc3);

    vp0 = _mm_add_ps(_mm_mul_ps(vp0, vt0), vc2);
    vp1 = _mm_add_ps(_mm_mul_ps(vp1, vt1), vc2);
    vp2 = _mm_add_ps(_mm_mul_ps(vp2, vt2), vc2);
    vp3 = _mm_add_ps(_mm_mul_ps(vp3, vt3), vc2);

    vp0 = _mm_add_ps(_mm_mul_ps(vp0, vt0), vc1);
    vp1 = _mm_add_ps(_mm_mul_ps(vp1, vt1), vc1);
    vp2 = _mm_add_ps(_mm_mul_ps(vp2, vt2), vc1);
    vp3 = _mm_add_ps(_mm_mul_ps(vp3, vt3), vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt0 = _mm_mul_ps(vt0, vs0);
    vt1 = _mm_mul_ps(vt1, vs1);
    vt2 = _mm_mul_ps(vt2, vs2);
    vt3 = _mm_mul_ps(vt3, vs3);

    __m128 vf0 = _mm_add_ps(_mm_mul_ps(vt0, vp0), vs0);
    __m128 vf1 = _mm_add_ps(_mm_mul_ps(vt1, vp1), vs1);
    __m128 vf2 = _mm_add_ps(_mm_mul_ps(vt2, vp2), vs2);
    __m128 vf3 = _mm_add_ps(_mm_mul_ps(vt3, vp3), vs3);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf0 = _mm_andnot_ps(_mm_cmplt_ps(vx0, vdenorm_cutoff), vf0);
    vf1 = _mm_andnot_ps(_mm_cmplt_ps(vx1, vdenorm_cutoff), vf1);
    vf2 = _mm_andnot_ps(_mm_cmplt_ps(vx2, vdenorm_cutoff), vf2);
    vf3 = _mm_andnot_ps(_mm_cmplt_ps(vx3, vdenorm_cutoff), vf3);

    // Store 16 (4x4) outputs at a time.
    _mm_storeu_ps(output, vf0);
    _mm_storeu_ps(output + 4, vf1);
    _mm_storeu_ps(output + 8, vf2);
    _mm_storeu_ps(output + 12, vf3);
    output += 16;

    // Accumulate computed exponents.
    vacc0 = _mm_add_ps(vacc0, vf0);
    vacc1 = _mm_add_ps(vacc1, vf1);
    vacc0 = _mm_add_ps(vacc0, vf2);
    vacc1 = _mm_add_ps(vacc1, vf3);
  }
  // Add up all accumulators to vacc0
  vacc0 = _mm_add_ps(vacc0, vacc1);

  __m128 vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    // Load 4 inputs at a time.
    const __m128 vi = _mm_loadu_ps(input);
    input += 4;

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m128 vx = _mm_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m128 vn = _mm_add_ps(_mm_mul_ps(vx, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vx);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc5, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm_mul_ps(vt, vs);
    __m128 vf = _mm_add_ps(_mm_mul_ps(vt, vp), vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm_andnot_ps(_mm_cmplt_ps(vx, vdenorm_cutoff), vf);

    // Store 4 outputs at a time.
    _mm_storeu_ps(output, vf);
    output += 4;

    // Accumulate computed exponents.
    vacc = _mm_add_ps(vacc, vf);
  }
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    // Load 4 inputs at a time.
    const __m128 vi = _mm_loadu_ps(input);

    // Subtract maximum input x := i - i_max. This implies x <= 0.
    const __m128 vx = _mm_sub_ps(vi, vi_max);

    // Compute reduced argument batch := round(x / log(2)).
    __m128 vn = _mm_add_ps(_mm_mul_ps(vx, vlog2e), vmagic_bias);

    // Create a floating-point number s (scale) such that s == 2**batch for inputs which don't cause underflow, i.e.
    // -87.33642 <= x <= 0.0, and -126 <= batch <= 0 accordingly.
    const __m128 vs = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn), 23));

    // Subtract the large number back to get final batch := round(x / log(2)).
    vn = _mm_sub_ps(vn, vmagic_bias);

    // Compute reduced argument t := x - batch * log(2).
    // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vx);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    // Compute degree-5 polynomial approximation for exp(t) on [-log(2)/2, log(2)/2].
    __m128 vp = _mm_add_ps(_mm_mul_ps(vc5, vt), vc4);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc3);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc2);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vc1);

    // Reconstruct the final f value:
    //   f = s * (1 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
    //     = s + (t * s) * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    //     = s + (t * s) * p
    vt = _mm_mul_ps(vt, vs);
    __m128 vf = _mm_add_ps(_mm_mul_ps(vt, vp), vs);

    // For inputs below zero cutoff, replace output with +0.0f.
    // Note that for NaN inputs, comparison result is false, and outputs are left unchanged.
    vf = _mm_andnot_ps(_mm_cmplt_ps(vx, vdenorm_cutoff), vf);

    if (batch & (2 * sizeof(float))) {
      // Store 2 outputs at a time.
      _mm_storel_pi((__m64*) output, vf);
      output += 2;

      // Accumulate 2 computed exponents.
      vacc = _mm_add_ps(vacc, _mm_movelh_ps(vf, _mm_setzero_ps()));

      vf = _mm_movehl_ps(vf, vf);
    }
    if (batch & (1 * sizeof(float))) {
      // Store 1 output at a time.
      _mm_store_ss(output, vf);

      // Accumulate 1 computed exponent.
      vacc = _mm_add_ss(vacc, vf);
    }
  }
  // Reduce 4 batch in the SIMD register
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_shuffle_ps(vacc, vacc, _MM_SHUFFLE(2, 3, 0, 1)));
  _mm_store_ss(sum, vacc);
}
