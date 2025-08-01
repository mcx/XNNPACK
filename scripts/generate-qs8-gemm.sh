#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4-minmax-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=1 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x1-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4-minmax-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F32 -D WASM=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F32 -D WASM=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F32 -D WASM=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F32 -D WASM=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F32 -D WASM=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F32 -D WASM=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F32 -D WASM=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x4-minmax-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QB4_F16 -D WASM=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x4-minmax-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x2-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x2-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x2-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x2-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x2-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x2-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x2-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x2-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x2-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x2-minmax-rndnu-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x4-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x4-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x4-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x4-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x4-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x4-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x4-minmax-rndnu-scalar.c &

################################## ARMv6 SIMD #################################
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x2c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x2c4-minmax-fp32-armsimd32.c &

tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x2c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x2c4-minmax-fp32-armsimd32.c &

################################ RISC-V Vector ################################
tools/xngen src/qs8-gemm/rvv.c.in -D MR=1 -D NR=m4 -D  -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=2 -D NR=m4 -D  -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=3 -D NR=m4 -D  -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=4 -D NR=m4 -D  -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=5 -D NR=m4 -D  -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=6 -D NR=m4 -D  -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=7 -D NR=m4 -D  -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=8 -D NR=m4 -D  -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x4v-minmax-rvv.c &

tools/xngen src/qs8-gemm/rvv.c.in -D MR=1 -D NR=m4 -D  -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=2 -D NR=m4 -D  -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=3 -D NR=m4 -D  -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=4 -D NR=m4 -D  -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=5 -D NR=m4 -D  -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=6 -D NR=m4 -D  -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=7 -D NR=m4 -D  -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x4v-minmax-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=8 -D NR=m4 -D  -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x4v-minmax-rvv.c &

tools/xngen src/qs8-gemm/rvv.c.in -D MR=1 -D NR=m4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4v-minmax-fp32-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=4 -D NR=m4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4v-minmax-fp32-rvv.c &
tools/xngen src/qs8-gemm/rvv.c.in -D MR=7 -D NR=m4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x4v-minmax-fp32-rvv.c &

################################## WAsm SIMD ##################################
### C2 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c2-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c2-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c2-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c2-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c2-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c2-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c2-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c2-minmax-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx16c2-wasmsimd-dot16x2.c.in -D ACCUMULATORS=1 -D MR=1 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c2-minmax-fp32-wasmsimd-dot16x2.c &
tools/xngen src/qs8-gemm/MRx16c2-wasmsimd-dot16x2.c.in -D ACCUMULATORS=1 -D MR=4 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c2-minmax-fp32-wasmsimd-dot16x2.c &

tools/xngen src/qs8-gemm/MRx16c2s2-wasmsimd-dot16x2.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c2s2-minmax-fp32-wasmsimd-dot16x2.c &
tools/xngen src/qs8-gemm/MRx16c2s2-wasmsimd-dot16x2.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c2s2-minmax-fp32-wasmsimd-dot16x2.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

############################## WAsm Relaxed SIMD ##############################
### C16 micro-kernels
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=1 NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=2 NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=3 NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=4 NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c16-minmax-fp32-wasmsdot.c &

tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=1 NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=2 NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=3 NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=4 NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c16-minmax-fp32-wasmsdot.c &

tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=1 NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c16-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=2 NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c16-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=3 NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c16-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=4 NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c16-minmax-wasmsdot.c &

tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=1 NR=8 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c16-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=2 NR=8 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c16-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=3 NR=8 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c16-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=4 NR=8 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c16-minmax-wasmsdot.c &

tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=1 NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=2 NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=3 NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=4 NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c16-minmax-fp32-wasmusdot.c &

tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=1 NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=2 NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=3 NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=4 NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c16-minmax-fp32-wasmusdot.c &

tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=1 NR=8 -D UNROLL=0 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=2 NR=8 -D UNROLL=0 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=3 NR=8 -D UNROLL=0 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=4 NR=8 -D UNROLL=0 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-wasmusdot.c &

tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=1 NR=8 -D UNROLL=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-wasmusdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=2 NR=8 -D UNROLL=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-wasmusdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=3 NR=8 -D UNROLL=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-wasmusdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=4 NR=8 -D UNROLL=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-wasmusdot-u2.c &

tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=1 NR=8 -D UNROLL=0 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=2 NR=8 -D UNROLL=0 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=3 NR=8 -D UNROLL=0 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=4 NR=8 -D UNROLL=0 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-wasmsdot.c &

tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=1 NR=8 -D UNROLL=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-wasmsdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=2 NR=8 -D UNROLL=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-wasmsdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=3 NR=8 -D UNROLL=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-wasmsdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=4 NR=8 -D UNROLL=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-wasmsdot-u2.c &

tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=1 NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c16-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=2 NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c16-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=3 NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c16-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=4 NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c16-minmax-wasmusdot.c &

tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=1 NR=8 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c16-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=2 NR=8 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c16-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=3 NR=8 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c16-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmdot.c.in -D MR=4 NR=8 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c16-minmax-wasmusdot.c &

tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=1 NR=8 -D UNROLL=0 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=2 NR=8 -D UNROLL=0 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=3 NR=8 -D UNROLL=0 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=4 NR=8 -D UNROLL=0 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-wasmusdot.c &

tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=1 NR=8 -D UNROLL=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-wasmusdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=2 NR=8 -D UNROLL=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-wasmusdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=3 NR=8 -D UNROLL=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-wasmusdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=4 NR=8 -D UNROLL=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-wasmusdot-u2.c &

tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=1 NR=8 -D UNROLL=0 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=2 NR=8 -D UNROLL=0 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=3 NR=8 -D UNROLL=0 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=4 NR=8 -D UNROLL=0 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-wasmsdot.c &

tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=1 NR=8 -D UNROLL=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-wasmsdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=2 NR=8 -D UNROLL=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-wasmsdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=3 NR=8 -D UNROLL=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-wasmsdot-u2.c &
tools/xngen src/qs8-gemm/c8-wasmdot.c.in -D MR=4 NR=8 -D UNROLL=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-wasmsdot-u2.c &

### C4 micro-kernels
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D MR=1 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D MR=3 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c4-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D MR=4 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-wasmusdot.c &

tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=1 -D MR=1 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-wasmusdot-u2.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D UNROLL=2 -D MR=3 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c4-minmax-fp32-wasmusdot-u2.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D UNROLL=2 -D MR=4 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-wasmusdot-u2.c &

tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=1 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-wasmusdot-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=3 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c4-minmax-fp32-wasmusdot-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=4 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-wasmusdot-u2-acc2.c &

tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D MR=1 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D MR=3 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c4-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D MR=4 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-wasmsdot.c &

tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=1 -D MR=1 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-wasmsdot-u2.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D UNROLL=2 -D MR=3 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c4-minmax-fp32-wasmsdot-u2.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D ACCUMULATORS=1 -D UNROLL=2 -D MR=4 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-wasmsdot-u2.c &

tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=1 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-wasmsdot-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=3 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c4-minmax-fp32-wasmsdot-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=4 NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-wasmsdot-u2-acc2.c &

tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D MR=1 NR=16 -D ACCUMULATORS=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-wasmusdot.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D MR=4 NR=16 -D ACCUMULATORS=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-wasmusdot.c &

tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=1 -D MR=1 NR=16 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-wasmusdot-u2.c &
tools/xngen src/qs8-gemm/MRx16c4-wasmdot.c.in -D UNROLL=2 -D ACCUMULATORS=1 -D MR=4 NR=16 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-wasmusdot-u2.c &

################################### ARM NEON ##################################
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16-minmax-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16-minmax-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x16-minmax-neonfp16arith-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x16-minmax-neonfp16arith-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x16-minmax-neonfp16arith-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F16 -D ARMV8=0 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x16-minmax-neonfp16arith-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16-minmax-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16-minmax-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x16-minmax-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QB4_F32 -D ARMV8=0 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x16-minmax-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16-minmax-fp32-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16-minmax-fp32-neonv8-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16-minmax-fp32-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16-minmax-fp32-neonv8-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-2x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-3x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-6x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x16-minmax-rndnu-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU16    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x16-minmax-rndnu16-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-2x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-3x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-6x16-minmax-rndnu-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=      -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=      -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-fp32-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gemm/gen/qu8-gemm-1x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-fp32-neonv8-mlal-lane.c &

### C2 micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neonv8-mlal-dup.c &

### C2 LD1R micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neonv8-mlal-ld1r.c &

### C2 LD2R micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neonv8-mlal-ld2r.c &

### C2 LD4R micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neon-mlal-ld4r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD4R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neonv8-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD4R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neonv8-mlal-ld4r.c &

### C2S4 micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -D ARMV8=0 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c2s4-minmax-neonfp16arith.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -D ARMV8=0 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c2s4-minmax-neonfp16arith.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c2s4-minmax-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c2s4-minmax-neon-mlal.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2s4-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2s4-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2s4-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2s4-minmax-fp32-neonv8-mlal.c &

### C4 micro-kernels
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neonv8-mlal-dup.c &

### C4 LD1R micro-kernels
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neonv8-mlal-ld1r.c &

### C4 LD2R micro-kernels
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neonv8-mlal-ld2r.c &

### C4S2 micro-kernels
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4s2-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4s2-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4s2-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4s2-minmax-fp32-neonv8-mlal.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-neonv8-mlal.c &

### C4 micro-kernels
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c4-minmax-neondotfp16arith.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x16c4-minmax-neondotfp16arith.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c4-minmax-neondot.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c4-minmax-neondot.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c4-minmax-neondot.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x8c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x8c4-minmax-neondot.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x16c4-minmax-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x16c4-minmax-neondot.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c4-minmax-neondotfp16arith.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x16c4-minmax-neondotfp16arith.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-5x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x8c4-minmax-neondotfp16arith.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-5x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x16c4-minmax-neondotfp16arith.c &


tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c4-minmax-fp32-neondot.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-neondot-ld64.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-neondot-ld64.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-neondot-ld64.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-neondot-ld64.c &

tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-aarch64-neondot-ld128.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-aarch64-neondot-ld128.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-aarch64-neondot-ld128.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-aarch64-neondot-ld128.c &

############################### AArch32 assembly ##############################
### Cortex-A53 lane micro-kernels
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64-prfm.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64-prfm.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S &

tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &
tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S &
tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S &

### QU8 micro-kernels
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64-prfm.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &

tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &

### C4 micro-kernels
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c4-minmax-asm-aarch32-neondotfp16arith-cortex-a55.S &
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c4-minmax-asm-aarch32-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-ld64.S.in                           -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c4-minmax-fp32-asm-aarch32-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c4-minmax-fp32-asm-aarch32-neondot-cortex-a55.S &

############################### AArch64 assembly ##############################
### Cortex-A53 lane micro-kernels
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64-prfm.S &

### QU8 micro-kernels
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU16 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu16-asm-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU16 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu16-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64-prfm.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a75.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a75.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75-prfm.S &

### C4 micro-kernels
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=         -D DATATYPE=QD8_F16     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondotfp16arith-cortex-a55.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=         -D DATATYPE=QD8_F32     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=         -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=         -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=         -D DATATYPE=QD8     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld64.S &

tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld128.S &

### C8 / C16 micro-kernels
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mull.S.in                          -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mull.S &
tools/xngen src/qs8-gemm/2x8c16-aarch64-neon-mlal.S.in                         -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c16-minmax-fp32-asm-aarch64-neon-mlal.S &

### NEON I8MM C8 micro-kernels
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x32c8-minmax-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x32c8-minmax-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-5x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-6x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-7x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F16 -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-8x32c8-minmax-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x32c8-minmax-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x32-minmax-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-6x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QB4_F32 -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x32c8-minmax-neoni8mm.c &

################################### x86 SSE ###################################
### C2 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-avx-ld128.c &

################################### x86 SSE ###################################
### C2S4 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-avx-ld128.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x4c8-minmax-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x4c8-minmax-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x4c8-minmax-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x4c8-minmax-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x4c8-minmax-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QB4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x4c8-minmax-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-avx-ld128.c &

################################### x86 AVX2 ##################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QB4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QB4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QB4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QB4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QB4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QB4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QB4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QB4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-1x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-2x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-3x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-4x8c8-minmax-fp32-avx2.c &

################################### x86 AVX256SKX ##################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avx256skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avx256skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx256skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx256skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avx256skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avx256skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QU8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-1x8c8-minmax-fp32-avx256skx.c &

################################## x86 AVX512 #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-1x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-5x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-7x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-8x16c8-minmax-fp32-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512skx-prfm.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-1x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-5x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-7x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-8x16c8-minmax-fp32-avx512skx-prfm.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512skx-prfm.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512skx-prfm.c &

################################## x86 AVX512VNNI #################################
### C4 micro-kernels
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-12x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-14x16c4-minmax-fp32-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-9x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-10x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-12x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-14x16c4-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c4-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c4-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c4-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c4-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c4-minmax-avx512skx-madd.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-12x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-14x16c4-minmax-fp32-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-9x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-10x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-12x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-14x16c4-minmax-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c4-minmax-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c4-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c4-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c4-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c4-minmax-avx512skx-madd-prfm.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-9x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-10x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-12x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-14x16c8-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-9x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-10x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-12x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QB4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-14x16c8-minmax-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-12x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-14x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-9x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-10x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-12x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-14x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c8-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c8-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c8-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c8-minmax-avx512skx-madd.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-12x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-14x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-9x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-10x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-12x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QD8 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-14x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c8-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c8-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c8-minmax-avx512skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c8-minmax-avx512skx-madd-prfm.c &

################################## x86 AVX512VNNI GFNI #################################
### C4 micro-kernels
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c4-minmax-avx512vnnigfni.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c4-minmax-avx512vnnigfni-prfm.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-9x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-10x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-12x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-14x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-5x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-7x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-8x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-9x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-10x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-12x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QB4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-14x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC4 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x16c8-minmax-avx512vnnigfni-prfm.c &

################################## x86 SSSE3 #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=1  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=2  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=3  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=4  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=5  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=6  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x4c8-minmax-ssse3-madd.c &

tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=1  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=2  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=3  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=4  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=5  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=6  -D DATATYPE=QC4_F32 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x4c8-minmax-ssse3-madd-prfm.c &

tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=1  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-1x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=2  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-2x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=3  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-3x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=4  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-4x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=5  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-5x4c8-minmax-ssse3-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=6  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-6x4c8-minmax-ssse3-madd.c &

tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=1  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-1x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=2  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-2x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=3  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-3x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=4  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-4x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=5  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-5x4c8-minmax-ssse3-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=6  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=0 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-6x4c8-minmax-ssse3-madd-prfm.c &

tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=1  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-1x4c8-minmax-avx-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=2  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-2x4c8-minmax-avx-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=3  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-3x4c8-minmax-avx-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=4  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-4x4c8-minmax-avx-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=5  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-5x4c8-minmax-avx-madd.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=6  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-6x4c8-minmax-avx-madd.c &

tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=1  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-1x4c8-minmax-avx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=2  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-2x4c8-minmax-avx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=3  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-3x4c8-minmax-avx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=4  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-4x4c8-minmax-avx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=5  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-5x4c8-minmax-avx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx4c8-ssevnni.c.in -D MR=6  -D DATATYPE=QS8_QC4 -D SSE=3 -D AVX=1 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-6x4c8-minmax-avx-madd-prfm.c &

################################## x86 AVX256 VNNI EVEX #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-12x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-14x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-9x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-10x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-12x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-14x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-9x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-10x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-12x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-14x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-9x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-10x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-12x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-14x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-9x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-10x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-12x8c8-minmax-avx256skx-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-14x8c8-minmax-avx256skx-madd.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-12x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC8     -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-14x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-9x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-10x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-12x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QD8_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-14x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-9x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-10x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-12x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QD8_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-14x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-9x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-10x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-12x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-14x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-9x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-10x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-12x8c8-minmax-avx256skx-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-14x8c8-minmax-avx256skx-madd-prfm.c &

################################## x86 AVX256 VNNI GFNI EVEX #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-9x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-10x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-12x8c8-minmax-avx256vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-14x8c8-minmax-avx256vnnigfni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-9x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-10x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-12x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F32 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-14x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-9x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-10x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-12x8c8-minmax-avx256vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC4_F16 -D AVX=10 -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-14x8c8-minmax-avx256vnnigfni-prfm.c &

################################## x86 AVX2/AVXVNNI #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-1x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-2x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-3x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-4x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-5x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-6x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-7x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-8x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx2-madd.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx2-madd.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=AVXVNNIINT8     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avxvnniint8-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC8     -D AVX=2 -D VARIANT=AVXVNNIINT8     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avxvnniint8-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-1x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-2x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-3x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-4x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-5x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-6x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-7x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QS8_QC4 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-8x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=     -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx2-madd-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=2 -D VARIANT=MADD -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx2-madd-prfm.c &

### C4 micro-kernels
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=1 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c4-minmax-avxvnni-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=2 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c4-minmax-avxvnni-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=3 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c4-minmax-avxvnni-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=4 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c4-minmax-avxvnni-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=5 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c4-minmax-avxvnni-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=6 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c4-minmax-avxvnni-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=7 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c4-minmax-avxvnni-u2-acc2.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=8 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c4-minmax-avxvnni-u2-acc2.c &

tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=1 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c4-minmax-avxvnni-u4-acc4.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=2 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c4-minmax-avxvnni-u4-acc4.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=3 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c4-minmax-avxvnni-u4-acc4.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=4 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c4-minmax-avxvnni-u4-acc4.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=5 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c4-minmax-avxvnni-u4-acc4.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=6 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c4-minmax-avxvnni-u4-acc4.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=7 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c4-minmax-avxvnni-u4-acc4.c &
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=8 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c4-minmax-avxvnni-u4-acc4.c &

################################## x86 AVX512 AMX #################################
### C4 micro-kernels
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=16 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=16 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=16 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x16c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=16 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x16c4-minmax-fp32-avx512amx-prfm.c &

tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=32 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x32c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=32 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x32c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=32 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x32c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=32 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x32c4-minmax-fp32-avx512amx-prfm.c &

tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=64 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x64c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=64 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x64c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=64 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x64c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=64 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x64c4-minmax-fp32-avx512amx-prfm.c &

tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=16 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=16 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=16 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x16c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=16 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x16c4-minmax-avx512amx-prfm.c &

tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=32 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x32c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=32 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x32c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=32 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x32c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=32 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x32c4-minmax-avx512amx-prfm.c &

tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=64 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=64 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=64 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=64 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x64c4-minmax-avx512amx-prfm.c &

tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=64 -D DATATYPE=QD8_F16 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=64 -D DATATYPE=QD8_F16 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=64 -D DATATYPE=QD8_F16 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-16x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=64 -D DATATYPE=QD8_F16 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-16x64c4-minmax-avx512amx-prfm.c &

tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D MR=1  -D NR=16 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D MR=7  -D NR=16 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D MR=16 -D NR=16 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x16c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D MR=16 -D NR=16 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x16c4-minmax-avx512amx-prfm.c &

tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D MR=1  -D NR=32 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x32c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D MR=7  -D NR=32 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x32c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D MR=16 -D NR=32 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x32c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D MR=16 -D NR=32 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x32c4-minmax-avx512amx-prfm.c &

tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D MR=1  -D NR=64 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D MR=7  -D NR=64 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=0 -D MR=16 -D NR=64 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-gemm/c4-avx512amx.c.in -D VARIANT= -D GFNI=1 -D PREFETCH=1 -D MR=16 -D NR=64 -D DATATYPE=QC4_F32 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x64c4-minmax-avx512amx-prfm.c &

################################## Hexagon HVX #################################
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=1  -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x32c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=2  -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x32c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=3  -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x32c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=4  -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x32c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=5  -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x32c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=6  -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x32c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=7  -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x32c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=8  -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x32c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=9  -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x32c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=10 -D NR=32  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x32c4-minmax-fp32-hvx.c &

tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=1  -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x64c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=2  -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x64c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=3  -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x64c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=4  -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x64c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=5  -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x64c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=6  -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x64c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=7  -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x64c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=8  -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x64c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=9  -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x64c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=10 -D NR=64  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x64c4-minmax-fp32-hvx.c &

tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=1  -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x96c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=2  -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x96c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=3  -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x96c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=4  -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x96c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=5  -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x96c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=6  -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x96c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=7  -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x96c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=8  -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x96c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=9  -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x96c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=10 -D NR=96  -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x96c4-minmax-fp32-hvx.c &

tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=1  -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x128c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=2  -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x128c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=3  -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x128c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=4  -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x128c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=5  -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x128c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=6  -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x128c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=7  -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x128c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=8  -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x128c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=9  -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-9x128c4-minmax-fp32-hvx.c &
tools/xngen src/qs8-gemm/c4-hvx.c.in -D MR=10 -D NR=128 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-10x128c4-minmax-fp32-hvx.c &

wait
