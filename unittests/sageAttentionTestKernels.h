/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

//! Launches FP16 reference attention for BSHD Q/K/V tensors and writes BSHD output for accuracy comparison.
void launchSageReferenceAttention(half const* q, half const* k, half const* v, half* output, int32_t batchSize,
    int32_t qoLen, int32_t kvLen, int32_t numQoHeads, int32_t numKvHeads, int32_t headDim, bool causal,
    cudaStream_t stream);

//! Launches per-warp INT8 quantization for BSHD query or key tensors and writes per-CTA warp scales.
void launchQuantizeToInt8PerWarp(half const* input, int8_t* output, float* scales, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headDim, bool isQuery, cudaStream_t stream);

//! FP8 roundtrip verification: encode→decode for values near 448
void launchFp8RoundtripTest(float* output, cudaStream_t stream);

//! FP8 smem→TensorCore roundtrip: store FP8 in swizzled smem, read via compute_fp8_sv
void launchFp8SmemRoundtripTest(float* output, cudaStream_t stream);

//! Launches FP8 quantization for BSHD value tensors and stores output in SageAttention transposed layout.
void launchQuantizeToFp8Transposed(half const* input, int8_t* output, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headDim, int32_t paddedSeqLen, cudaStream_t stream);
