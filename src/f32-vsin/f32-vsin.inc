// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__scalar_rational_5_4_div_u1, 1, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__scalar_rational_5_4_div_u2, 2, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__scalar_rational_5_4_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__scalar_rational_5_4_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__sse2_rational_5_4_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__sse2_rational_5_4_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__sse2_rational_5_4_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__sse2_rational_5_4_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__sse2fma_rational_5_4_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__sse2fma_rational_5_4_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsin_ukernel__avx_rational_5_4_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsin_ukernel__avx_rational_5_4_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsin_ukernel__avx_rational_5_4_div_u24, 24, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_vsin_ukernel__avx_rational_5_4_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_fma3, xnn_f32_vsin_ukernel__fma3_rational_5_4_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_fma3, xnn_f32_vsin_ukernel__fma3_rational_5_4_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_fma3, xnn_f32_vsin_ukernel__fma3_rational_5_4_div_u24, 24, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_fma3, xnn_f32_vsin_ukernel__fma3_rational_5_4_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsin_ukernel__avx512f_rational_5_4_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsin_ukernel__avx512f_rational_5_4_div_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsin_ukernel__avx512f_rational_5_4_div_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsin_ukernel__avx512f_rational_5_4_div_u64, 64, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsin_ukernel__avx512f_rational_5_4_nr_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsin_ukernel__avx512f_rational_5_4_nr_u32, 32, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsin_ukernel__avx512f_rational_5_4_nr_u48, 48, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vsin_ukernel__avx512f_rational_5_4_nr_u64, 64, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__wasmsimd_rational_5_4_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__wasmsimd_rational_5_4_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__wasmsimd_rational_5_4_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__wasmsimd_rational_5_4_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__hvx_rational_5_4_div_u32,  32,  false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__hvx_rational_5_4_div_u64,  64,  false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__hvx_rational_5_4_div_u96,  96,  false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vsin_ukernel__hvx_rational_5_4_div_u128, 128, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_HEXAGON && XNN_ENABLE_HVX

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsin_ukernel__neon_rational_5_4_div_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsin_ukernel__neon_rational_5_4_div_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsin_ukernel__neon_rational_5_4_div_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsin_ukernel__neon_rational_5_4_div_u16, 16, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsin_ukernel__neon_rational_5_4_nr_u4, 4, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsin_ukernel__neon_rational_5_4_nr_u8, 8, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsin_ukernel__neon_rational_5_4_nr_u12, 12, false, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vsin_ukernel__neon_rational_5_4_nr_u16, 16, false, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

