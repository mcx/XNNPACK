// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/1x12-aarch64-neonfma-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

# void xnn_f32_gemminc_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53(
#     size_t mr,                (x0) - unused.  mr = 1
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const float* a,           x3
#     size_t a_stride,          (x4) - unused
#     const float* w,           x5
#     float* c,                 x6
#     size_t cm_stride,         (x7) - unused
#     size_t cn_stride,         [sp] -> x14
#     const float* acc,         [sp + 8] -> x15
#     const xnn_f32_minmax_params* params)  [sp + 16] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x3 v0 v1
# B   x5 v20 v21 v22 v23
# B      v24 v25 v26 v27
# C0  x6 v16 v17 v18 v5 v6 v7
# Clamp v2 v3

# A53 based on LD128 with LDR.

BEGIN_FUNCTION xnn_f32_gemminc_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53

        # Load cn_stride, acc
        LDP         x14, x15, [sp]
        # Load params pointer
        LDR         x8, [sp, 16]

        # Load min/max values
        LD2R        {v2.4s, v3.4s}, [x8]
0:
        # Load initial accumulators
        LD1         {v16.16b, v17.16b, v18.16b}, [x15], 48

        MOVI        v5.4s, 0                // second set of C for pipelining FMLA
        PRFM        PLDL1KEEP, [x5]
        MOVI        v6.4s, 0
        PRFM        PLDL1KEEP, [x5, 64]
        MOVI        v7.4s, 0
        PRFM        PLDL1KEEP, [x5, 128]
        PRFM        PLDL1KEEP, [x5, 192]

        # Is there at least 8 floats (32 bytes) for prologue + epilogue?
        SUBS        x0, x2, 32              // k = kc - 32

        B.LO        3f

        # 16 prologue
        # Read first block of 1 A and B.
        LDP         q20, q21, [x5], 32
        LDP         q22, q23, [x5], 32
        LDP         q24, q25, [x5], 32
        LDP         q26, q27, [x5], 32
        LDP         q28, q29, [x5], 32
        LDP         q30, q31, [x5], 32
        LDR         q0, [x3], 16

        # Is there at least 32.  yes do main loop
        SUBS        x0, x0, 32
        B.LO        2f

        # Main loop - 8 floats of A (32 bytes)
1:
        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA        v16.4s, v20.4s, v0.s[0]
        LDR         q1, [x3], 16
        FMLA        v17.4s, v21.4s, v0.s[0]
        LDR         q20, [x5], 16
        FMLA        v18.4s, v22.4s, v0.s[0]
        LDR         q21, [x5], 16
        FMLA        v5.4s, v23.4s, v0.s[1]
        LDR         q22, [x5], 16
        FMLA        v6.4s, v24.4s, v0.s[1]
        LDR         q23, [x5], 16
        FMLA        v7.4s, v25.4s, v0.s[1]
        LDR         q24, [x5], 16
        FMLA        v16.4s, v26.4s, v0.s[2]
        LDR         q25, [x5], 16
        FMLA        v17.4s, v27.4s, v0.s[2]
        LDR         q26, [x5], 16
        FMLA        v18.4s, v28.4s, v0.s[2]
        LDR         q27, [x5], 16
        FMLA        v5.4s, v29.4s, v0.s[3]
        LDR         q28, [x5], 16
        FMLA        v6.4s, v30.4s, v0.s[3]
        LDR         q29, [x5], 16
        FMLA        v7.4s, v31.4s, v0.s[3]
        LDR         q30, [x5], 16
        LDR         q31, [x5], 16

        # Second block of 4.  FMA for second 4, loads for 1st block of 4.
        FMLA        v16.4s, v20.4s, v1.s[0]
        LDR         q0, [x3], 16
        FMLA        v17.4s, v21.4s, v1.s[0]
        LDR         q20, [x5], 16
        FMLA        v18.4s, v22.4s, v1.s[0]
        LDR         q21, [x5], 16
        FMLA        v5.4s, v23.4s, v1.s[1]
        LDR         q22, [x5], 16
        FMLA        v6.4s, v24.4s, v1.s[1]
        LDR         q23, [x5], 16
        FMLA        v7.4s, v25.4s, v1.s[1]
        LDR         q24, [x5], 16
        FMLA        v16.4s, v26.4s, v1.s[2]
        LDR         q25, [x5], 16
        FMLA        v17.4s, v27.4s, v1.s[2]
        LDR         q26, [x5], 16
        FMLA        v18.4s, v28.4s, v1.s[2]
        LDR         q27, [x5], 16
        FMLA        v5.4s, v29.4s, v1.s[3]
        LDR         q28, [x5], 16
        FMLA        v6.4s, v30.4s, v1.s[3]
        LDR         q29, [x5], 16
        FMLA        v7.4s, v31.4s, v1.s[3]
        LDR         q30, [x5], 16
        SUBS        x0, x0, 32
        LDR         q31, [x5], 16
        B.HS        1b

2:
        # Epilogue

        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA        v16.4s, v20.4s, v0.s[0]
        LDR         q1, [x3], 16
        FMLA        v17.4s, v21.4s, v0.s[0]
        LDR         q20, [x5], 16
        FMLA        v18.4s, v22.4s, v0.s[0]
        LDR         q21, [x5], 16
        FMLA        v5.4s, v23.4s, v0.s[1]
        LDR         q22, [x5], 16
        FMLA        v6.4s, v24.4s, v0.s[1]
        LDR         q23, [x5], 16
        FMLA        v7.4s, v25.4s, v0.s[1]
        LDR         q24, [x5], 16
        FMLA        v16.4s, v26.4s, v0.s[2]
        LDR         q25, [x5], 16
        FMLA        v17.4s, v27.4s, v0.s[2]
        LDR         q26, [x5], 16
        FMLA        v18.4s, v28.4s, v0.s[2]
        LDR         q27, [x5], 16
        FMLA        v5.4s, v29.4s, v0.s[3]
        LDR         q28, [x5], 16
        FMLA        v6.4s, v30.4s, v0.s[3]
        LDR         q29, [x5], 16
        FMLA        v7.4s, v31.4s, v0.s[3]
        LDR         q30, [x5], 16

        # Second block of 4.  FMA for second 4, no loads.
        FMLA        v16.4s, v20.4s, v1.s[0]
        LDR         q31, [x5], 16
        FMLA        v17.4s, v21.4s, v1.s[0]
        FMLA        v18.4s, v22.4s, v1.s[0]
        FMLA        v5.4s, v23.4s, v1.s[1]
        FMLA        v6.4s, v24.4s, v1.s[1]
        FMLA        v7.4s, v25.4s, v1.s[1]
        FMLA        v16.4s, v26.4s, v1.s[2]
        FMLA        v17.4s, v27.4s, v1.s[2]
        FMLA        v18.4s, v28.4s, v1.s[2]
        FMLA        v5.4s, v29.4s, v1.s[3]
        FMLA        v6.4s, v30.4s, v1.s[3]
        FMLA        v7.4s, v31.4s, v1.s[3]

3:
        # Is there a remainder?- 4 floats of A (16 bytes)
        TBNZ        x0, 4, 5f
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBNZ        x0, 3, 6f
        # Is there a remainder?- 1 float of A (4 bytes)
        TBNZ        x0, 2, 8f

4:
        FADD        v16.4s, v16.4s, v5.4s
        FADD        v17.4s, v17.4s, v6.4s
        FADD        v18.4s, v18.4s, v7.4s
        SUBS        x1, x1, 12

        # Clamp
        FMAX        v16.4s, v16.4s, v2.4s
        FMAX        v17.4s, v17.4s, v2.4s
        FMAX        v18.4s, v18.4s, v2.4s
        FMIN        v16.4s, v16.4s, v3.4s
        FMIN        v17.4s, v17.4s, v3.4s
        FMIN        v18.4s, v18.4s, v3.4s

        # Store full 1 x 12
        B.LO        9f

        ST1         {v16.16b, v17.16b, v18.16b}, [x6], x14
        SUB         x3,  x3, x2             // a0 -= kc

        B.HI        0b

        RET

5:
        # Remainder- 4 floats of A (16 bytes)
        LDR         q0, [x3], 16
        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         q22, [x5], 16
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v17.4s, v21.4s, v0.s[0]
        FMLA        v18.4s, v22.4s, v0.s[0]

        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         q22, [x5], 16
        FMLA        v16.4s, v20.4s, v0.s[1]
        FMLA        v17.4s, v21.4s, v0.s[1]
        FMLA        v18.4s, v22.4s, v0.s[1]

        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         q22, [x5], 16
        FMLA        v16.4s, v20.4s, v0.s[2]
        FMLA        v17.4s, v21.4s, v0.s[2]
        FMLA        v18.4s, v22.4s, v0.s[2]

        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         q22, [x5], 16
        FMLA        v16.4s, v20.4s, v0.s[3]
        FMLA        v17.4s, v21.4s, v0.s[3]
        FMLA        v18.4s, v22.4s, v0.s[3]

        TBZ         x0, 3, 7f
6:
        # Remainder- 2 floats of A (8 bytes)
        LDR         d0, [x3], 8
        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         q22, [x5], 16
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v17.4s, v21.4s, v0.s[0]
        FMLA        v18.4s, v22.4s, v0.s[0]

        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         q22, [x5], 16
        FMLA        v16.4s, v20.4s, v0.s[1]
        FMLA        v17.4s, v21.4s, v0.s[1]
        FMLA        v18.4s, v22.4s, v0.s[1]
7:
        TBZ         x0, 2, 4b
8:
        # Remainder- 1 float of A (4 bytes)
        LDR         s0, [x3], 4
        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         q22, [x5], 16
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v17.4s, v21.4s, v0.s[0]
        FMLA        v18.4s, v22.4s, v0.s[0]
        B           4b

        # Store odd channels
9:
        ADD         x1, x1, 12
        TBZ         x1, 3, 10f
        STP         q16, q17, [x6], 32
        MOV         v16.16b, v18.16b

10:
        TBZ         x1, 2, 11f
        STR         q16, [x6], 16
        MOV         v16.16b, v17.16b

11:
        TBZ         x1, 1, 12f
        STR         d16, [x6], 8
        DUP         d16, v16.d[1]

12:
        TBZ         x1, 0, 13f
        STR         s16, [x6]
13:
        RET

END_FUNCTION xnn_f32_gemminc_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
