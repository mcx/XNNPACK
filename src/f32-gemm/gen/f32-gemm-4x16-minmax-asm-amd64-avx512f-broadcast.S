// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_avx512f_broadcast

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

      # Align the stack pointer.
      mov r13, rsp
      sub rsp, 64
      and rsp, 0xFFFFFFFFFFFFFFC0
      # Store the old stack pointer containing the return address
      mov [rsp], r13

      # Allocate some space on the stack.
      sub rsp, 128

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

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with the biases.
      vmovaps  zmm11, [r9 + 0]
      vmovaps zmm12, zmm11
      vmovaps zmm13, zmm11
      vmovaps zmm14, zmm11
      add r9, 64

.Linner_loop:
      vmovaps  zmm7, [r9 + 0]
      add r9, 64
      vbroadcastss zmm2, dword ptr [rcx + r11]
      vfmadd231ps  zmm11, zmm2, zmm7
      vbroadcastss zmm3, dword ptr [rax + r11]
      vfmadd231ps  zmm12, zmm3, zmm7
      vbroadcastss zmm4, dword ptr [r15 + r11]
      vfmadd231ps  zmm13, zmm4, zmm7
      vbroadcastss zmm5, dword ptr [r14 + r11]
      vfmadd231ps  zmm14, zmm5, zmm7

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:
      # Min/max clamping.
      vminps  zmm11, zmm1, zmm11
      vminps  zmm12, zmm1, zmm12
      vminps  zmm13, zmm1, zmm13
      vminps  zmm14, zmm1, zmm14
      vmaxps  zmm11, zmm0, zmm11
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm13, zmm0, zmm13
      vmaxps  zmm14, zmm0, zmm14

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail

      vmovups  [r10], zmm11
      vmovups  [r13], zmm12
      vmovups  [rbx], zmm13
      vmovups  [rbp], zmm14
      add r10, 64
      add r13, 64
      add rbx, 64
      add rbp, 64

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  zmmword ptr [r10]{k1}, zmm11
      vmovups  zmmword ptr [r13]{k1}, zmm12
      vmovups  zmmword ptr [rbx]{k1}, zmm13
      vmovups  zmmword ptr [rbp]{k1}, zmm14

.Lreturn:
      add rsp, 128
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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_avx512f_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_avx512f_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_avx512f_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__