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

#include "sageAttentionParams.h"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{

/*!
 * @brief Runner for SageAttention kernels
 *
 * SageAttention is an efficient attention implementation using INT8 quantized Q/K
 * and FP8 quantized V. It provides significant speedup over standard FP16 attention
 * while maintaining accuracy through careful quantization strategies.
 *
 * Supported features:
 * - INT8 quantized Q/K with configurable granularity (per-block, per-warp, per-thread)
 * - FP8 quantized V
 * - Causal and non-causal attention
 * - Grouped-query attention (GQA) with different Q/KV head ratios
 * - FP16 or BF16 output
 *
 * Requirements:
 * - SM 8.9+ (Ada Lovelace or newer, e.g., RTX 4090)
 * - Head dimension must be multiple of 64
 */
class SageAttentionRunner
{
public:
    /*!
     * @brief Construct a SageAttention runner
     * @param dataType Output data type (FP16 or BF16)
     * @param batchSize Batch size
     * @param qoLen Query/output sequence length
     * @param kvLen Key/value sequence length
     * @param numQoHeads Number of query/output heads
     * @param numKvHeads Number of key/value heads
     * @param headDim Head dimension (must be multiple of 64)
     * @param smVersion CUDA compute capability (e.g., 89 for SM 8.9)
     * @param tensorLayout Input tensor layout
     * @param maskMode Attention mask mode
     * @param quantGran Quantization granularity for Q/K
     * @throws std::runtime_error if configuration is not supported
     */
    SageAttentionRunner(nvinfer1::DataType dataType, int32_t batchSize, int32_t qoLen, int32_t kvLen,
        int32_t numQoHeads, int32_t numKvHeads, int32_t headDim, int32_t smVersion,
        SageTensorLayout tensorLayout = SageTensorLayout::kBSHD,
        SageMaskMode maskMode = SageMaskMode::kCausal,
        SageQuantGranularity quantGran = SageQuantGranularity::kPerWarp);

    //! @brief Deleted default constructor
    SageAttentionRunner() = delete;

    //! @brief Destructor
    ~SageAttentionRunner() noexcept = default;

    //! @brief Get required workspace size in bytes
    //! @return Workspace size
    size_t getWorkspaceSize() const noexcept;

    /*!
     * @brief Run SageAttention kernel
     * @param params Attention parameters with device pointers set
     * @param workspace Workspace memory (can be nullptr if not needed)
     * @param stream CUDA stream for kernel launch
     * @throws std::runtime_error if device pointers are invalid or CUDA error occurs
     */
    void run(SageAttentionParams& params, void* workspace, cudaStream_t stream);

    /*!
     * @brief Check if SageAttention can be implemented for given configuration
     * @param headDim Head dimension
     * @param sm CUDA compute capability
     * @param dataType Output data type
     * @return True if implementation is available
     */
    static bool canImplement(
        int32_t headDim, int32_t sm, nvinfer1::DataType dataType) noexcept;

private:
    nvinfer1::DataType mDataType;       //!< Output data type
    int32_t mBatchSize;                 //!< Batch size
    int32_t mQoLen;                     //!< Query/output sequence length
    int32_t mKvLen;                     //!< Key/value sequence length
    int32_t mNumQoHeads;                //!< Number of query/output heads
    int32_t mNumKvHeads;                //!< Number of key/value heads
    int32_t mHeadDim;                   //!< Head dimension
    int32_t mSmVersion;                 //!< CUDA compute capability
    SageTensorLayout mTensorLayout;     //!< Input tensor layout
    SageMaskMode mMaskMode;             //!< Attention mask mode
    SageQuantGranularity mQuantGran;    //!< Quantization granularity

    size_t mWorkspaceSize{};            //!< Cached workspace size
};

} // namespace trt_edgellm
