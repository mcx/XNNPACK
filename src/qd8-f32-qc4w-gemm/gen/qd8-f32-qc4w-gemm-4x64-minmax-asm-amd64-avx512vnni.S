// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"
.MASK:
      .quad   -1085102592571150096

BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x64c4__asm_amd64_avx512vnni

      .intel_syntax noprefix
      # Free up GP registers.
      # Save register arguments for tail call to msan annotation helper.
      push rdi
      push rsi
      push rbx
      push rbp
      push r15
      push r14
      push r13
      push r12

      # load params to free up GP registers
      mov r13, [rsp + 96] # params
      vbroadcastss zmm0, dword ptr [r13]
      vbroadcastss zmm1, dword ptr [r13 + 4]

      # Load c pointer.
      mov r10, [rsp + 72]
      # Load cm_stride.
      mov r11, [rsp + 80]

      add rdx, 3
      and rdx, -4

      # Move stack parameters which have not yet been loaded
      mov r12, [rsp + 104]

      # Align the stack pointer.
      mov r13, rsp
      sub rsp, 64
      and rsp, 0xFFFFFFFFFFFFFFC0
      # Store the old stack pointer containing the return address
      mov [rsp], r13
      # Push additional stack parameters to the new stack
      mov [rsp + 8], r12

      # Allocate some space on the stack.
      sub rsp, 384

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

      # Clamp a & c pointers if mr <= 2
      mov r15, rax
      add r15, r8
      mov rbx, r13
      add rbx, r11
      cmp rdi, 2
      cmovle r15, rax
      cmovle rbx, r13

      # Clamp a & c pointers if mr <= 3
      mov r14, r15
      add r14, r8
      mov rbp, rbx
      add rbp, r11
      cmp rdi, 3
      cmovle r14, r15
      cmovle rbp, rbx
      # Load quantization_params pointer from stack
      mov r11, [rsp + 392]
      mov edi, [r11 + 0]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 128], zmm6
      mov edi, [r11 + 8]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 192], zmm6
      mov edi, [r11 + 16]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 256], zmm6
      mov edi, [r11 + 24]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 320], zmm6

      mov r11, [rsp + 88]
      # Load 0xF0 for masking the weights
      vbroadcastsd  zmm13, qword ptr [rip + .MASK]


.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      vmovaps  zmm8, [r9 + 128]
      vmovaps  zmm9, [r9 + 192]
      vpmulld zmm5, zmm6, zmmword ptr [rsp + 128]
      vpmulld zmm12, zmm6, zmmword ptr [rsp + 192]
      vpmulld zmm14, zmm6, zmmword ptr [rsp + 256]
      vpmulld zmm15, zmm6, zmmword ptr [rsp + 320]
      vpmulld zmm16, zmm7, zmmword ptr [rsp + 128]
      vpmulld zmm17, zmm7, zmmword ptr [rsp + 192]
      vpmulld zmm18, zmm7, zmmword ptr [rsp + 256]
      vpmulld zmm19, zmm7, zmmword ptr [rsp + 320]
      vpmulld zmm20, zmm8, zmmword ptr [rsp + 128]
      vpmulld zmm21, zmm8, zmmword ptr [rsp + 192]
      vpmulld zmm22, zmm8, zmmword ptr [rsp + 256]
      vpmulld zmm23, zmm8, zmmword ptr [rsp + 320]
      vpmulld zmm24, zmm9, zmmword ptr [rsp + 128]
      vpmulld zmm25, zmm9, zmmword ptr [rsp + 192]
      vpmulld zmm26, zmm9, zmmword ptr [rsp + 256]
      vpmulld zmm27, zmm9, zmmword ptr [rsp + 320]
      add r9, 256

.Linner_loop:
      vmovaps zmm7, [r9 + 0]
      vpslld zmm6, zmm7, 4
      vpandd zmm6, zmm6, zmm13
      vpandd zmm7, zmm7, zmm13
      vmovaps zmm9, [r9 + 64]
      vpslld zmm8, zmm9, 4
      vpandd zmm8, zmm8, zmm13
      vpandd zmm9, zmm9, zmm13
      add r9, 128
      vpbroadcastd zmm2, [rcx + r11]
      vpdpbusd  zmm5, zmm2, zmm6
      vpdpbusd  zmm16, zmm2, zmm7
      vpdpbusd  zmm20, zmm2, zmm8
      vpdpbusd  zmm24, zmm2, zmm9
      vpbroadcastd zmm2, [rax + r11]
      vpdpbusd  zmm12, zmm2, zmm6
      vpdpbusd  zmm17, zmm2, zmm7
      vpdpbusd  zmm21, zmm2, zmm8
      vpdpbusd  zmm25, zmm2, zmm9
      vpbroadcastd zmm2, [r15 + r11]
      vpdpbusd  zmm14, zmm2, zmm6
      vpdpbusd  zmm18, zmm2, zmm7
      vpdpbusd  zmm22, zmm2, zmm8
      vpdpbusd  zmm26, zmm2, zmm9
      vpbroadcastd zmm2, [r14 + r11]
      vpdpbusd  zmm15, zmm2, zmm6
      vpdpbusd  zmm19, zmm2, zmm7
      vpdpbusd  zmm23, zmm2, zmm8
      vpdpbusd  zmm27, zmm2, zmm9

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:

      # Convert from int32 to float.
      vpsrad zmm5, zmm5, 4
      vcvtdq2ps zmm5, zmm5
      vpsrad zmm12, zmm12, 4
      vcvtdq2ps zmm12, zmm12
      vpsrad zmm14, zmm14, 4
      vcvtdq2ps zmm14, zmm14
      vpsrad zmm15, zmm15, 4
      vcvtdq2ps zmm15, zmm15
      vpsrad zmm16, zmm16, 4
      vcvtdq2ps zmm16, zmm16
      vpsrad zmm17, zmm17, 4
      vcvtdq2ps zmm17, zmm17
      vpsrad zmm18, zmm18, 4
      vcvtdq2ps zmm18, zmm18
      vpsrad zmm19, zmm19, 4
      vcvtdq2ps zmm19, zmm19
      vpsrad zmm20, zmm20, 4
      vcvtdq2ps zmm20, zmm20
      vpsrad zmm21, zmm21, 4
      vcvtdq2ps zmm21, zmm21
      vpsrad zmm22, zmm22, 4
      vcvtdq2ps zmm22, zmm22
      vpsrad zmm23, zmm23, 4
      vcvtdq2ps zmm23, zmm23
      vpsrad zmm24, zmm24, 4
      vcvtdq2ps zmm24, zmm24
      vpsrad zmm25, zmm25, 4
      vcvtdq2ps zmm25, zmm25
      vpsrad zmm26, zmm26, 4
      vcvtdq2ps zmm26, zmm26
      vpsrad zmm27, zmm27, 4
      vcvtdq2ps zmm27, zmm27
      # Load quantization_params pointer from stack
      mov r11, [rsp + 392]
      vmulps zmm5, zmm5, dword ptr [r11 + 4]{1to16}
      vmulps zmm12, zmm12, dword ptr [r11 + 12]{1to16}
      vmulps zmm14, zmm14, dword ptr [r11 + 20]{1to16}
      vmulps zmm15, zmm15, dword ptr [r11 + 28]{1to16}
      vmulps zmm16, zmm16, dword ptr [r11 + 4]{1to16}
      vmulps zmm17, zmm17, dword ptr [r11 + 12]{1to16}
      vmulps zmm18, zmm18, dword ptr [r11 + 20]{1to16}
      vmulps zmm19, zmm19, dword ptr [r11 + 28]{1to16}
      vmulps zmm20, zmm20, dword ptr [r11 + 4]{1to16}
      vmulps zmm21, zmm21, dword ptr [r11 + 12]{1to16}
      vmulps zmm22, zmm22, dword ptr [r11 + 20]{1to16}
      vmulps zmm23, zmm23, dword ptr [r11 + 28]{1to16}
      vmulps zmm24, zmm24, dword ptr [r11 + 4]{1to16}
      vmulps zmm25, zmm25, dword ptr [r11 + 12]{1to16}
      vmulps zmm26, zmm26, dword ptr [r11 + 20]{1to16}
      vmulps zmm27, zmm27, dword ptr [r11 + 28]{1to16}
      vmovaps zmm10, [r9 + 0]
      vmovaps zmm11, [r9 + 64]
      vmovaps zmm2, [r9 + 128]
      vmovaps zmm3, [r9 + 192]
      add r9, 256
      vmovaps zmm6, [r9 + 0]
      vmovaps zmm7, [r9 + 64]
      vmovaps zmm8, [r9 + 128]
      vmovaps zmm9, [r9 + 192]
      add r9, 256
      vfmadd213ps zmm5, zmm10, zmm6
      vfmadd213ps zmm12, zmm10, zmm6
      vfmadd213ps zmm14, zmm10, zmm6
      vfmadd213ps zmm15, zmm10, zmm6
      vfmadd213ps zmm16, zmm11, zmm7
      vfmadd213ps zmm17, zmm11, zmm7
      vfmadd213ps zmm18, zmm11, zmm7
      vfmadd213ps zmm19, zmm11, zmm7
      vfmadd213ps zmm20, zmm2, zmm8
      vfmadd213ps zmm21, zmm2, zmm8
      vfmadd213ps zmm22, zmm2, zmm8
      vfmadd213ps zmm23, zmm2, zmm8
      vfmadd213ps zmm24, zmm3, zmm9
      vfmadd213ps zmm25, zmm3, zmm9
      vfmadd213ps zmm26, zmm3, zmm9
      vfmadd213ps zmm27, zmm3, zmm9
      # Min/max clamping.
      vminps  zmm5, zmm1, zmm5
      vminps  zmm16, zmm1, zmm16
      vminps  zmm20, zmm1, zmm20
      vminps  zmm24, zmm1, zmm24
      vminps  zmm12, zmm1, zmm12
      vminps  zmm17, zmm1, zmm17
      vminps  zmm21, zmm1, zmm21
      vminps  zmm25, zmm1, zmm25
      vminps  zmm14, zmm1, zmm14
      vminps  zmm18, zmm1, zmm18
      vminps  zmm22, zmm1, zmm22
      vminps  zmm26, zmm1, zmm26
      vminps  zmm15, zmm1, zmm15
      vminps  zmm19, zmm1, zmm19
      vminps  zmm23, zmm1, zmm23
      vminps  zmm27, zmm1, zmm27
      vmaxps  zmm5, zmm0, zmm5
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm20, zmm0, zmm20
      vmaxps  zmm24, zmm0, zmm24
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm21, zmm0, zmm21
      vmaxps  zmm25, zmm0, zmm25
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm18, zmm0, zmm18
      vmaxps  zmm22, zmm0, zmm22
      vmaxps  zmm26, zmm0, zmm26
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm19, zmm0, zmm19
      vmaxps  zmm23, zmm0, zmm23
      vmaxps  zmm27, zmm0, zmm27

      # Check whether full or partial store.
      cmp rsi, 64
      jl .Ltail

      vmovups  [r10], zmm5
      vmovups  [r10 + 64], zmm16
      vmovups  [r10 + 128], zmm20
      vmovups  [r10 + 192], zmm24
      vmovups  [r13], zmm12
      vmovups  [r13 + 64], zmm17
      vmovups  [r13 + 128], zmm21
      vmovups  [r13 + 192], zmm25
      vmovups  [rbx], zmm14
      vmovups  [rbx + 64], zmm18
      vmovups  [rbx + 128], zmm22
      vmovups  [rbx + 192], zmm26
      vmovups  [rbp], zmm15
      vmovups  [rbp + 64], zmm19
      vmovups  [rbp + 128], zmm23
      vmovups  [rbp + 192], zmm27
      add r10, 256
      add r13, 256
      add rbx, 256
      add rbp, 256

      sub rsi, 64
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      shr r11, 16
      kmovw k2, r11d
      shr r11, 16
      kmovw k3, r11d
      shr r11, 16
      kmovw k4, r11d

      vmovups  zmmword ptr [r10]{k1}, zmm5
      vmovups  zmmword ptr [r10 + 64]{k2}, zmm16
      vmovups  zmmword ptr [r10 + 128]{k3}, zmm20
      vmovups  zmmword ptr [r10 + 192]{k4}, zmm24
      vmovups  zmmword ptr [r13]{k1}, zmm12
      vmovups  zmmword ptr [r13 + 64]{k2}, zmm17
      vmovups  zmmword ptr [r13 + 128]{k3}, zmm21
      vmovups  zmmword ptr [r13 + 192]{k4}, zmm25
      vmovups  zmmword ptr [rbx]{k1}, zmm14
      vmovups  zmmword ptr [rbx + 64]{k2}, zmm18
      vmovups  zmmword ptr [rbx + 128]{k3}, zmm22
      vmovups  zmmword ptr [rbx + 192]{k4}, zmm26
      vmovups  zmmword ptr [rbp]{k1}, zmm15
      vmovups  zmmword ptr [rbp + 64]{k2}, zmm19
      vmovups  zmmword ptr [rbp + 128]{k3}, zmm23
      vmovups  zmmword ptr [rbp + 192]{k4}, zmm27

.Lreturn:
      add rsp, 384
      mov r13, [rsp]
      mov rsp, r13
      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      pop rsi
      pop rdi
      #if XNN_HAS_FEATURE(memory_sanitizer)
      jmp xnn_gemm_ukernel_msan_sizeof_c_4
      #else
      ret
      #endif
END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x64c4__asm_amd64_avx512vnni

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x64c4__asm_amd64_avx512vnni.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x64c4__asm_amd64_avx512vnni.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__