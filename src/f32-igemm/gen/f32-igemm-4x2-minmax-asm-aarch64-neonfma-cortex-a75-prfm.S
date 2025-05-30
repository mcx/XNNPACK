// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/4x2-aarch64-neonfma-cortex-a75.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

# void xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm(
#     size_t mr,                         x0
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const float** restrict a,           x4
#     const float* restrict w,            x5
#     float* restrict c,                  x6
#     size_t cm_stride,                  x7
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const float* zero,                 [sp + 16] -> x12
#     const xnn_f32_minmax_params params [sp + 24] -> x8

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0 x20 v0  v4
// A1 x13 v1  v5
// A2 x14 v2  v6
// A3 x15 v3  v7
// B   x5 v16 v17 v18 v19 v20 v21 v22 v23
// C0  x6 v24 v25
// C1 x16 v26 v27
// C2 x17 v28 v29
// C3  x7 v30 v31
// clamp  v4 v5


BEGIN_FUNCTION xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm

        # Load cn_stride, a_offset
        LDP         x10, x11, [sp]

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Load min/max values
        LD2R        {v4.2s, v5.2s}, [x8]

        # Save x20 on stack
        STR         x20, [sp, -16]!

        # Clamp C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        CSEL        x16, x6, x16, LO        //   c1 = c0

        ADD         x17, x16, x7            // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x17, x16, x17, LS       //   c2 = c1

        CMP         x0, 4                   // if mr < 4
        ADD         x7, x17, x7             // c3 = c2 + cm_stride
        CSEL        x7, x17, x7, LO         //   c3 = c2

0:
        # Load initial bias from w into accumulators
        LDR         d24, [x5], 8
        MOV         v26.8b, v24.8b
        MOV         v28.8b, v24.8b
        MOV         v30.8b, v24.8b
        MOVI        v25.2s, 0
        PRFM        PLDL1KEEP, [x5, 64]
        MOVI        v27.2s, 0
        PRFM        PLDL1KEEP, [x5, 128]
        MOVI        v29.2s, 0
        PRFM        PLDL1KEEP, [x5, 192]
        MOVI        v31.2s, 0
        PRFM        PLDL1KEEP, [x5, 256]

        MOV         x9, x3                  // p = ks

1:
        # Load next 4 A pointers
        LDP         x20, x13, [x4], 16
        LDP         x14, x15, [x4], 16

        CMP         x20, x12                // if a0 == zero
        ADD         x20, x20, x11           // a0 += a_offset
        CSEL        x20, x12, x20, EQ       //   a0 = zero, else += a0 + a_offset
        CMP         x13, x12                // if a1 == zero
        ADD         x13, x13, x11           // a1 += a_offset
        CSEL        x13, x12, x13, EQ       //   a1 = zero, else += a1 + a_offset
        CMP         x14, x12                // if a2 == zero
        ADD         x14, x14, x11           // a2 += a_offset
        CSEL        x14, x12, x14, EQ       //   a2 = zero, else += a2 + a_offset
        CMP         x15, x12                // if a3 == zero
        ADD         x15, x15, x11           // a3 += a_offset
        CSEL        x15, x12, x15, EQ       //   a3 = zero, else += a3 + a_offset

        # Is there at least 8 floats (32 bytes) for prologue + epilogue?
        SUBS        x0, x2, 32              // k = kc - 32
        B.LO        5f

        # Prologue
        # Read first block of 4 A and B.
        LDR         q0, [x20], 16
        LDP         d20, d21, [x5], 16
        LDR         q1, [x13], 16
        LDR         q2, [x14], 16
        LDR         q3, [x15], 16
        LDP         d22, d23, [x5], 16

        # Is there at least 32.  yes do main loop
        SUBS        x0, x0, 32
        B.LO        3f

        # Main loop - 8 floats of A (32 bytes)
2:
        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA        v24.2s, v20.2s, v0.s[0]
        LDR         q4, [x20], 16
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        LDR         d16, [x5, 0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        FMLA        v25.2s, v21.2s, v0.s[1]
        LDR         q5, [x13], 16
        FMLA        v27.2s, v21.2s, v1.s[1]
        FMLA        v29.2s, v21.2s, v2.s[1]
        LDR         q6, [x14], 16
        FMLA        v31.2s, v21.2s, v3.s[1]
        FMLA        v24.2s, v22.2s, v0.s[2]
        LDR         q7, [x15], 16
        FMLA        v26.2s, v22.2s, v1.s[2]
        FMLA        v28.2s, v22.2s, v2.s[2]
        LDR         d17, [x5, 8]
        FMLA        v30.2s, v22.2s, v3.s[2]
        FMLA        v25.2s, v23.2s, v0.s[3]
        LDR         d18, [x5, 16]
        FMLA        v27.2s, v23.2s, v1.s[3]
        FMLA        v29.2s, v23.2s, v2.s[3]
        LDR         d19, [x5, 24]
        FMLA        v31.2s, v23.2s, v3.s[3]
        PRFM        PLDL1KEEP, [x5, 320]

        # Second block of 4.  FMA for second 4, loads for 1st block of 4.
        FMLA        v24.2s, v16.2s, v4.s[0]
        LDR         q0, [x20], 16
        FMLA        v26.2s, v16.2s, v5.s[0]
        FMLA        v28.2s, v16.2s, v6.s[0]
        LDR         d20, [x5, 32]
        FMLA        v30.2s, v16.2s, v7.s[0]
        FMLA        v25.2s, v17.2s, v4.s[1]
        LDR         q1, [x13], 16
        FMLA        v27.2s, v17.2s, v5.s[1]
        FMLA        v29.2s, v17.2s, v6.s[1]
        LDR         q2, [x14], 16
        FMLA        v31.2s, v17.2s, v7.s[1]
        FMLA        v24.2s, v18.2s, v4.s[2]
        LDR         q3, [x15], 16
        FMLA        v26.2s, v18.2s, v5.s[2]
        FMLA        v28.2s, v18.2s, v6.s[2]
        LDR         d21, [x5, 40]
        FMLA        v30.2s, v18.2s, v7.s[2]
        SUBS        x0, x0, 32
        FMLA        v25.2s, v19.2s, v4.s[3]
        LDR         d22, [x5, 48]
        FMLA        v27.2s, v19.2s, v5.s[3]
        LDR         d23, [x5, 56]
        FMLA        v29.2s, v19.2s, v6.s[3]
        ADD         x5, x5, 64
        FMLA        v31.2s, v19.2s, v7.s[3]
        B.HS        2b

3:
        # Epilogue
        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA        v24.2s, v20.2s, v0.s[0]
        LDR         q4, [x20], 16
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        LDR         d16, [x5, 0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        FMLA        v25.2s, v21.2s, v0.s[1]
        LDR         q5, [x13], 16
        FMLA        v27.2s, v21.2s, v1.s[1]
        FMLA        v29.2s, v21.2s, v2.s[1]
        LDR         q6, [x14], 16
        FMLA        v31.2s, v21.2s, v3.s[1]
        FMLA        v24.2s, v22.2s, v0.s[2]
        LDR         q7, [x15], 16
        FMLA        v26.2s, v22.2s, v1.s[2]
        FMLA        v28.2s, v22.2s, v2.s[2]
        LDR         d17, [x5, 8]
        FMLA        v30.2s, v22.2s, v3.s[2]
        FMLA        v25.2s, v23.2s, v0.s[3]
        LDR         d18, [x5, 16]
        FMLA        v27.2s, v23.2s, v1.s[3]
        FMLA        v29.2s, v23.2s, v2.s[3]
        LDR         d19, [x5, 24]
        FMLA        v31.2s, v23.2s, v3.s[3]
        PRFM        PLDL1KEEP, [x5, 320]

        # Second block of 4.  FMA for second 4, no loads
        FMLA        v24.2s, v16.2s, v4.s[0]
        FMLA        v26.2s, v16.2s, v5.s[0]
        FMLA        v28.2s, v16.2s, v6.s[0]
        FMLA        v30.2s, v16.2s, v7.s[0]
        FMLA        v25.2s, v17.2s, v4.s[1]
        FMLA        v27.2s, v17.2s, v5.s[1]
        FMLA        v29.2s, v17.2s, v6.s[1]
        FMLA        v31.2s, v17.2s, v7.s[1]
        FMLA        v24.2s, v18.2s, v4.s[2]
        FMLA        v26.2s, v18.2s, v5.s[2]
        FMLA        v28.2s, v18.2s, v6.s[2]
        ADDS        x0, x0, 32
        FMLA        v30.2s, v18.2s, v7.s[2]
        FMLA        v25.2s, v19.2s, v4.s[3]
        ADD         x5, x5, 32
        FMLA        v27.2s, v19.2s, v5.s[3]
        FMLA        v29.2s, v19.2s, v6.s[3]
        LD2R        {v4.2s, v5.2s}, [x8]        // Load min/max values
        FMLA        v31.2s, v19.2s, v7.s[3]

        # Is there a remainder? up to 8 floats (32 bytes)
        B.NE        5f

4:
        # ks loop
        SUBS        x9, x9, 32              // ks -= MR * sizeof(void*)
        B.HI        1b

        FADD        v24.2s, v24.2s, v25.2s
        FADD        v26.2s, v26.2s, v27.2s
        FADD        v28.2s, v28.2s, v29.2s
        FADD        v30.2s, v30.2s, v31.2s

        # Clamp
        FMAX        v24.2s, v24.2s, v4.2s
        FMAX        v26.2s, v26.2s, v4.2s
        FMAX        v28.2s, v28.2s, v4.2s
        FMAX        v30.2s, v30.2s, v4.2s
        SUBS        x1, x1, 2
        FMIN        v24.2s, v24.2s, v5.2s
        FMIN        v26.2s, v26.2s, v5.2s
        FMIN        v28.2s, v28.2s, v5.2s
        FMIN        v30.2s, v30.2s, v5.2s

        # Store full 4 x 2
        B.LO        8f

        STR         d30, [x7]
        ADD         x7,  x7, x10
        STR         d28, [x17]
        ADD         x17, x17, x10
        STR         d26, [x16]
        ADD         x16,  x16, x10
        STR         d24, [x6]
        ADD         x6,  x6, x10
        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b

        # Restore x20 from stack
        LDR         x20, [sp], 16
        RET

5:
        # Remainder- 4 floats of A (16 bytes)
        TBZ         x0, 4, 6f

        LDR         q0, [x20], 16
        LDP         d20, d21, [x5], 16
        LDR         q1, [x13], 16
        LDR         q2, [x14], 16
        LDR         q3, [x15], 16
        LDP         d22, d23, [x5], 16
        FMLA        v24.2s, v20.2s, v0.s[0]
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        FMLA        v25.2s, v21.2s, v0.s[1]
        FMLA        v27.2s, v21.2s, v1.s[1]
        FMLA        v29.2s, v21.2s, v2.s[1]
        FMLA        v31.2s, v21.2s, v3.s[1]
        FMLA        v24.2s, v22.2s, v0.s[2]
        FMLA        v26.2s, v22.2s, v1.s[2]
        FMLA        v28.2s, v22.2s, v2.s[2]
        FMLA        v30.2s, v22.2s, v3.s[2]
        FMLA        v25.2s, v23.2s, v0.s[3]
        FMLA        v27.2s, v23.2s, v1.s[3]
        FMLA        v29.2s, v23.2s, v2.s[3]
        FMLA        v31.2s, v23.2s, v3.s[3]

6:
        # Remainder- 2 floats of A (8 bytes)
        TBZ         x0, 3, 7f

        LDR         d0, [x20], 8
        LDP         d20, d21, [x5], 16
        LDR         d1, [x13], 8
        LDR         d2, [x14], 8
        LDR         d3, [x15], 8
        FMLA        v24.2s, v20.2s, v0.s[0]
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        FMLA        v25.2s, v21.2s, v0.s[1]
        FMLA        v27.2s, v21.2s, v1.s[1]
        FMLA        v29.2s, v21.2s, v2.s[1]
        FMLA        v31.2s, v21.2s, v3.s[1]

7:
        # Remainder- 1 float of A (4 bytes)
        TBZ         x0, 2, 4b

        LDR         s0, [x20], 4
        LDR         d20, [x5], 8
        LDR         s1, [x13], 4
        LDR         s2, [x14], 4
        LDR         s3, [x15], 4
        FMLA        v24.2s, v20.2s, v0.s[0]
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        B           4b

        # Store odd width
8:
        STR         s30,  [x7]
        STR         s28, [x17]
        STR         s26, [x16]
        STR         s24,  [x6]

        # Restore x20 from stack
        LDR         x20, [sp], 16
        RET

END_FUNCTION xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
