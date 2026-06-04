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

#include "common/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace sage
{

//! \brief Get SageAttention padded KV length for FP8 V transposed layout.
int32_t getSagePaddedKvLen(int32_t kvLen) noexcept;

//! \brief Get Q per-warp scale tensor size for SageAttention runtime quantization.
int32_t getSageQScaleSize(int32_t batchSize, int32_t qoLen, int32_t numHeads) noexcept;

//! \brief Get K per-warp scale tensor size for SageAttention runtime quantization.
int32_t getSageKScaleSize(int32_t batchSize, int32_t kvLen, int32_t numHeads) noexcept;

//! \brief Get V per-channel scale tensor size for SageAttention runtime quantization.
int32_t getSageVScaleSize(int32_t batchSize, int32_t numKVHeads, int32_t headDim) noexcept;

//! \brief Quantize FP16 Q tensor in BSHD layout to INT8 with SageAttention per-warp scales.
void launchSageQuantizeQToInt8(rt::Tensor const& q, rt::Tensor& qInt8, rt::Tensor& qScale, cudaStream_t stream);

//! \brief Convert FP16 KV cache into SageAttention INT8 K and FP8 transposed V tensors
//! with per-channel V scaling and K-mean centering (smooth_k).
//! vScale: [batchSize, numKVHeads, headDim] — per-channel V scales.
//! kMean:  [batchSize, numKVHeads, headDim] — per-channel K means.
void launchSageConvertKVCacheToInt8AndFp8(rt::Tensor const& kvCache, rt::Tensor const& sequenceLengths,
    rt::Tensor& kInt8, rt::Tensor& vFp8, rt::Tensor& kScale, rt::Tensor& vScale, rt::Tensor& kMean,
    int32_t kvLen, cudaStream_t stream);

} // namespace sage
} // namespace trt_edgellm
