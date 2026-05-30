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

// This file provides host-side utilities for SageAttention integration
// For device-side utilities, see sageAttentionDeviceUtils.cuh

#include "sageAttentionParams.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <string>

namespace trt_edgellm
{
namespace sage
{

// FP8 offset constants for SageAttention quantization
constexpr float S_FP8_OFFSET = 8.807f;
constexpr float S_FP8_OFFSET_EXP = 6680.8477f;
constexpr float S_FP8_OFFSET_EXP_INV = 0.0022326917f;

// Div ceiling helper
constexpr int32_t divCeil(int32_t M, int32_t N)
{
    return (M + N - 1) / N;
}

// Warp size constant
constexpr uint32_t WARP_SIZE = 32;

// Host-side error checking macro
#define SAGE_CHECK_CUDA(call)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            std::ostringstream oss;                                                                \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err); \
            throw std::runtime_error(oss.str());                                                   \
        }                                                                                          \
    } while (0)

#define SAGE_CHECK(cond, msg)                                                                      \
    do                                                                                             \
    {                                                                                              \
        if (!(cond))                                                                               \
        {                                                                                          \
            throw std::runtime_error(msg);                                                         \
        }                                                                                          \
    } while (0)

// Type dispatch macros for host code
#define SAGE_DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                                           \
    do                                                                                             \
    {                                                                                              \
        if (head_dim == 64)                                                                        \
        {                                                                                          \
            constexpr int HEAD_DIM = 64;                                                           \
            __VA_ARGS__                                                                            \
        }                                                                                          \
        else if (head_dim == 128)                                                                  \
        {                                                                                          \
            constexpr int HEAD_DIM = 128;                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
        else if (head_dim == 256)                                                                  \
        {                                                                                          \
            constexpr int HEAD_DIM = 256;                                                          \
            __VA_ARGS__                                                                            \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            throw std::runtime_error("Unsupported head_dim: " + std::to_string(head_dim));         \
        }                                                                                          \
    } while (0)

#define SAGE_DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)                                           \
    do                                                                                             \
    {                                                                                              \
        if (is_causal)                                                                             \
        {                                                                                          \
            constexpr bool IS_CAUSAL = true;                                                       \
            __VA_ARGS__                                                                            \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            constexpr bool IS_CAUSAL = false;                                                      \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } while (0)

#define SAGE_DISPATCH_QUANT_GRAN(quant_gran, QK_QUANT_GRAN, ...)                                  \
    do                                                                                             \
    {                                                                                              \
        if (quant_gran == static_cast<int>(SageQuantGranularity::kPerWarp))                        \
        {                                                                                          \
            constexpr int QK_QUANT_GRAN = static_cast<int>(SageQuantGranularity::kPerWarp);        \
            __VA_ARGS__                                                                            \
        }                                                                                          \
        else if (quant_gran == static_cast<int>(SageQuantGranularity::kPerThread))                 \
        {                                                                                          \
            constexpr int QK_QUANT_GRAN = static_cast<int>(SageQuantGranularity::kPerThread);      \
            __VA_ARGS__                                                                            \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            throw std::runtime_error("Unsupported SageAttention Q/K quantization granularity");    \
        }                                                                                          \
    } while (0)

#define SAGE_DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, ...)                                     \
    do                                                                                             \
    {                                                                                              \
        if (return_lse)                                                                            \
        {                                                                                          \
            constexpr bool RETURN_LSE = true;                                                      \
            __VA_ARGS__                                                                            \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            constexpr bool RETURN_LSE = false;                                                     \
            __VA_ARGS__                                                                            \
        }                                                                                          \
    } while (0)

#define SAGE_DISPATCH_DTYPE_TO_CTYPE_FP16(dtype, DTYPE, ...)                                      \
    do                                                                                             \
    {                                                                                              \
        if (dtype == nvinfer1::DataType::kHALF)                                                    \
        {                                                                                          \
            using DTYPE = half;                                                                    \
            __VA_ARGS__                                                                            \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            throw std::runtime_error("Unsupported output dtype (only FP16 supported for SM89)");   \
        }                                                                                          \
    } while (0)

// Forward declaration of kernel launcher (implemented in sageAttentionKernels.cu)
template<int32_t HEAD_DIM, bool IS_CAUSAL, int32_t QK_QUANT_GRAN, bool RETURN_LSE, typename DTypeOut>
void launchSageAttentionKernel(
    SageAttentionParams& params,
    int32_t gridSizeX,
    int32_t gridSizeY,
    int32_t gridSizeZ,
    size_t smemSize,
    cudaStream_t stream);

} // namespace sage
} // namespace trt_edgellm
