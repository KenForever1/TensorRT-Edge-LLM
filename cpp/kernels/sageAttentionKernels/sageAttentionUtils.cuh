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

// This file provides device-side utilities for SageAttention CUDA kernels
// For host-side utilities, see sageAttentionHostUtils.h

#include "sageAttentionHostUtils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

// FP8 types (E4M3 and E5M2) for SM89+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#include <cuda_fp8.h>
#endif

namespace trt_edgellm
{
namespace sage
{

// Device helper functions (only available in CUDA device code)
// Note: __CUDA_ARCH__ is only defined during device code compilation, not host code
#if defined(__CUDA_ARCH__)
__device__ __forceinline__ uint32_t getWarpId()
{
    return threadIdx.y;
}

__device__ __forceinline__ uint32_t getLaneId()
{
    return threadIdx.x;
}

template<uint32_t numWarpsQ, uint32_t numWarpsK>
__device__ __forceinline__ uint32_t getWarpIdxQ()
{
    return getWarpId() / numWarpsK;
}

template<uint32_t numWarpsQ, uint32_t numWarpsK>
__device__ __forceinline__ uint32_t getWarpIdxK()
{
    return getWarpId() % numWarpsK;
}
#endif // defined(__CUDA_ARCH__)

} // namespace sage
} // namespace trt_edgellm
