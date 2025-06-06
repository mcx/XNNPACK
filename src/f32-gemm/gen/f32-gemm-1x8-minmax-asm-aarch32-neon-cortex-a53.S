// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/1x8-aarch32-neon-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

.syntax unified

// void xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53(
//     size_t mr,                                r0
//     size_t nc,                                r1
//     size_t kc,                                r2 -> r0
//     const float* a,                           r3
//     size_t a_stride,               sp + 8  -> (unused)
//     const float* w,                sp + 12 -> r9
//     float* c,                      sp + 16 -> r12
//     size_t cm_stride,              sp + 20 -> (unused)
//     size_t cn_stride,              sp + 24 -> r7
//     xnn_f32_minmax_params params)  sp + 28 -> (r0)

// d8-d31, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0
// B    r9  d24, d25, d26, d27
// B        d28, d29, d30, d31
// C0  r12 d16-d17  q8  d18-d19  q9 q10 q11
// clamp  (r0) d4 d5 d6 d7

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53
        .arm
#ifndef __APPLE__
        .arch       armv7-a
        .fpu        neon
#endif
        # Push 8 bytes
        PUSH        {r7, r9}                // 8

        LDR         r0, [sp, 28]            // params
        LDR         r9, [sp, 12]            // w
        LDR         r12, [sp, 16]           // c

        # Load min/max values
        VLD1.32     {d4[], d5[]}, [r0]!
        LDR         r7, [sp, 24]            // cn_stride
        VLD1.32     {d6[], d7[]}, [r0]

0:
        # Load initial bias from w into accumulators
        VLDM        r9!, {d16-d19}          // Bias
        VMOV.I32    q10, 0                  // second set of C for pipelining VMLA
        SUBS        r0, r2, 8
        VMOV.I32    q11, 0

        BLO         3f                      // less than 2 channels?

        # Main loop - 2 floats of A (8 bytes)
1:
        VLDM        r9!, {d24-d27}          // B0
        VLD1.32     {d0}, [r3]!             // A0
        VLDM        r9!, {d28-d31}          // B1
        VMLA.F32    q8, q12, d0[0]
        VMLA.F32    q9, q13, d0[0]
        VMLA.F32    q10, q14, d0[1]
        VMLA.F32    q11, q15, d0[1]
        SUBS        r0, r0, 8
        BHS         1b

        # Is there a remainder?- 1 float of A (4 bytes)
        TST         r0, 4
        BNE         3f

2:
        VADD.F32    q8, q8, q10
        VADD.F32    q9, q9, q11

        # Clamp
        VMAX.F32    q8,  q8, q2
        SUBS        r1, r1, 8
        VMAX.F32    q9,  q9, q2
        VMIN.F32    q8,  q8, q3
        VMIN.F32    q9,  q9, q3

        # Store full 4 x 8
        BLO         4f
        VST1.32     {d16-d19}, [r12], r7
        SUB         r3, r3, r2
        BHI         0b

        POP         {r7, r9}
        BX          lr

3:
        # Remainder- 1 float of A (4 bytes)
        VLDM        r3!,  {s0}              // A0
        VLDM        r9!, {d24-d27}          // B0
        VMLA.F32    q8, q12, d0[0]
        VMLA.F32    q9, q13, d0[0]
        B           2b

        # Store odd width
4:
        TST         r1, 4
        BEQ         5f
        VST1.32     {d16-d17}, [r12]!
        VMOV        q8,  q9

5:
        TST         r1, 2
        BEQ         6f
        VST1.32     {d16}, [r12]!
        VMOV        d16, d17

6:
        TST         r1, 1
        BEQ         7f
        VST1.32     {d16[0]}, [r12]

7:
        POP         {r7, r9}
        BX          lr

END_FUNCTION xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
