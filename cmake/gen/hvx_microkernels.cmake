# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for hvx
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_HVX_MICROKERNEL_SRCS
  src/f32-gemm/gen/f32-gemm-1x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-1x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-5x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-8x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-1x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-5x64-minmax-hvx-broadcast.c
  src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u256.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u128-acc2.c
  src/f32-rdminmax/gen/f32-rdmax-2p2x-hvx-c32.c
  src/f32-rdminmax/gen/f32-rdmin-2p2x-hvx-c32.c
  src/f32-rminmax/gen/f32-rmax-hvx-u64-acc2.c
  src/f32-rminmax/gen/f32-rmin-hvx-u64-acc2.c
  src/f32-rminmax/gen/f32-rminmax-hvx-u64-acc2.c
  src/f32-rsum/gen/f32-rsum-hvx-u64-acc2.c
  src/f32-vbinary/gen/f32-vadd-hvx-u128.c
  src/f32-vbinary/gen/f32-vaddc-hvx-u128.c
  src/f32-vbinary/gen/f32-vmax-hvx-u128.c
  src/f32-vbinary/gen/f32-vmaxc-hvx-u128.c
  src/f32-vbinary/gen/f32-vmin-hvx-u128.c
  src/f32-vbinary/gen/f32-vminc-hvx-u128.c
  src/f32-vbinary/gen/f32-vmul-hvx-u128.c
  src/f32-vbinary/gen/f32-vmulc-hvx-u128.c
  src/f32-vbinary/gen/f32-vrsubc-hvx-u128.c
  src/f32-vbinary/gen/f32-vsqrdiff-hvx-u128.c
  src/f32-vbinary/gen/f32-vsqrdiffc-hvx-u128.c
  src/f32-vbinary/gen/f32-vsub-hvx-u128.c
  src/f32-vbinary/gen/f32-vsubc-hvx-u128.c
  src/f32-vunary/gen/f32-vabs-hvx.c
  src/f32-vunary/gen/f32-vneg-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x128c4-minmax-fp32-hvx.c
  src/s8-rminmax/gen/s8-rmax-hvx-u256-acc2.c
  src/s8-rminmax/gen/s8-rmin-hvx-u256-acc2.c
  src/s8-rminmax/gen/s8-rminmax-hvx-u256-acc2.c
  src/u8-rminmax/gen/u8-rmax-hvx-u256-acc2.c
  src/u8-rminmax/gen/u8-rmin-hvx-u256-acc2.c
  src/u8-rminmax/gen/u8-rminmax-hvx-u256-acc2.c
  src/x32-packw/gen/x32-packw-gio-hvx-u2.c
  src/x32-transposec/gen/x32-transposec-32x32-multi-multi-hvx.c)

SET(NON_PROD_HVX_MICROKERNEL_SRCS
  src/f32-gemm/gen/f32-gemm-1x128-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-2x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-2x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-2x128-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-3x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-3x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-3x128-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-4x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-4x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-4x128-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-5x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-5x128-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-6x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-6x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-6x128-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-7x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-7x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-8x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-9x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-9x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-10x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-10x64-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-16x32-minmax-hvx-broadcast.c
  src/f32-gemm/gen/f32-gemm-16x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-1x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-1x128-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-2x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-2x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-2x128-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-3x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-3x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-3x128-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-4x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-4x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-4x128-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-5x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-5x128-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-6x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-6x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-6x128-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-7x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-7x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-8x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-8x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-9x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-9x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-10x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-10x64-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-16x32-minmax-hvx-broadcast.c
  src/f32-igemm/gen/f32-igemm-16x64-minmax-hvx-broadcast.c
  src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u32.c
  src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u64.c
  src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u96.c
  src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u128.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u32.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u64-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u128-acc4.c
  src/f32-rminmax/gen/f32-rmax-hvx-u32.c
  src/f32-rminmax/gen/f32-rmax-hvx-u96-acc3.c
  src/f32-rminmax/gen/f32-rmax-hvx-u128-acc2.c
  src/f32-rminmax/gen/f32-rmax-hvx-u128-acc4.c
  src/f32-rminmax/gen/f32-rmin-hvx-u32.c
  src/f32-rminmax/gen/f32-rmin-hvx-u96-acc3.c
  src/f32-rminmax/gen/f32-rmin-hvx-u128-acc2.c
  src/f32-rminmax/gen/f32-rmin-hvx-u128-acc4.c
  src/f32-rminmax/gen/f32-rminmax-hvx-u32.c
  src/f32-rminmax/gen/f32-rminmax-hvx-u96-acc3.c
  src/f32-rminmax/gen/f32-rminmax-hvx-u128-acc2.c
  src/f32-rminmax/gen/f32-rminmax-hvx-u128-acc4.c
  src/f32-rsum/gen/f32-rsum-hvx-u32.c
  src/f32-rsum/gen/f32-rsum-hvx-u96-acc3.c
  src/f32-rsum/gen/f32-rsum-hvx-u128-acc2.c
  src/f32-rsum/gen/f32-rsum-hvx-u128-acc4.c
  src/f32-rsum/gen/f32-rsum-hvx-u256-acc4.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-hvx-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-hvx-pipelined-x4.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-hvx-pipelined.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-hvx-x2.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-hvx-x4.c
  src/f32-spmm/gen/f32-spmm-32x1-minmax-hvx.c
  src/f32-spmm/gen/f32-spmm-64x1-minmax-hvx-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-64x1-minmax-hvx-pipelined-x4.c
  src/f32-spmm/gen/f32-spmm-64x1-minmax-hvx-pipelined.c
  src/f32-spmm/gen/f32-spmm-64x1-minmax-hvx-x2.c
  src/f32-spmm/gen/f32-spmm-64x1-minmax-hvx-x4.c
  src/f32-spmm/gen/f32-spmm-64x1-minmax-hvx.c
  src/f32-spmm/gen/f32-spmm-128x1-minmax-hvx-pipelined-x2.c
  src/f32-spmm/gen/f32-spmm-128x1-minmax-hvx-pipelined-x4.c
  src/f32-spmm/gen/f32-spmm-128x1-minmax-hvx-pipelined.c
  src/f32-spmm/gen/f32-spmm-128x1-minmax-hvx-x2.c
  src/f32-spmm/gen/f32-spmm-128x1-minmax-hvx-x4.c
  src/f32-spmm/gen/f32-spmm-128x1-minmax-hvx.c
  src/f32-vapproxgelu/gen/f32-vapproxgelu-hvx-rational-12-10-div.c
  src/f32-vbinary/gen/f32-vadd-hvx-u32.c
  src/f32-vbinary/gen/f32-vadd-hvx-u64.c
  src/f32-vbinary/gen/f32-vaddc-hvx-u32.c
  src/f32-vbinary/gen/f32-vaddc-hvx-u64.c
  src/f32-vbinary/gen/f32-vmax-hvx-u32.c
  src/f32-vbinary/gen/f32-vmax-hvx-u64.c
  src/f32-vbinary/gen/f32-vmaxc-hvx-u32.c
  src/f32-vbinary/gen/f32-vmaxc-hvx-u64.c
  src/f32-vbinary/gen/f32-vmin-hvx-u32.c
  src/f32-vbinary/gen/f32-vmin-hvx-u64.c
  src/f32-vbinary/gen/f32-vminc-hvx-u32.c
  src/f32-vbinary/gen/f32-vminc-hvx-u64.c
  src/f32-vbinary/gen/f32-vmul-hvx-u32.c
  src/f32-vbinary/gen/f32-vmul-hvx-u64.c
  src/f32-vbinary/gen/f32-vmulc-hvx-u32.c
  src/f32-vbinary/gen/f32-vmulc-hvx-u64.c
  src/f32-vbinary/gen/f32-vrsubc-hvx-u32.c
  src/f32-vbinary/gen/f32-vrsubc-hvx-u64.c
  src/f32-vbinary/gen/f32-vsqrdiff-hvx-u32.c
  src/f32-vbinary/gen/f32-vsqrdiff-hvx-u64.c
  src/f32-vbinary/gen/f32-vsqrdiffc-hvx-u32.c
  src/f32-vbinary/gen/f32-vsqrdiffc-hvx-u64.c
  src/f32-vbinary/gen/f32-vsub-hvx-u32.c
  src/f32-vbinary/gen/f32-vsub-hvx-u64.c
  src/f32-vbinary/gen/f32-vsubc-hvx-u32.c
  src/f32-vbinary/gen/f32-vsubc-hvx-u64.c
  src/f32-vexp/gen/f32-vexp-hvx-rational-3-2-div.c
  src/f32-vgelu/gen/f32-vgelu-hvx-rational-12-10-div.c
  src/f32-vunary/gen/f32-vsqr-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x128c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x32c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x64c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x96c4-minmax-fp32-hvx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x128c4-minmax-fp32-hvx.c
  src/qs8-vadd/gen/qs8-vadd-minmax-hvx-u32.c
  src/qs8-vadd/gen/qs8-vadd-minmax-hvx-u64.c
  src/qs8-vadd/gen/qs8-vadd-minmax-hvx-u96.c
  src/qs8-vadd/gen/qs8-vadd-minmax-hvx-u128.c)

SET(ALL_HVX_MICROKERNEL_SRCS ${PROD_HVX_MICROKERNEL_SRCS} + ${NON_PROD_HVX_MICROKERNEL_SRCS})
