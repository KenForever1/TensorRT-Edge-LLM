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

#include "sageAttentionRunner.h"
#include "sageAttentionHostUtils.h"

#include "common/checkMacros.h"

#include <cmath>
#include <sstream>

namespace trt_edgellm
{

SageAttentionRunner::SageAttentionRunner(nvinfer1::DataType dataType, int32_t batchSize, int32_t qoLen,
    int32_t kvLen, int32_t numQoHeads, int32_t numKvHeads, int32_t headDim, int32_t smVersion,
    SageTensorLayout tensorLayout, SageMaskMode maskMode, SageQuantGranularity quantGran)
    : mDataType(dataType)
    , mBatchSize(batchSize)
    , mQoLen(qoLen)
    , mKvLen(kvLen)
    , mNumQoHeads(numQoHeads)
    , mNumKvHeads(numKvHeads)
    , mHeadDim(headDim)
    , mSmVersion(smVersion)
    , mTensorLayout(tensorLayout)
    , mMaskMode(maskMode)
    , mQuantGran(quantGran)
{
    // Validate configuration
    if (!canImplement(headDim, smVersion, dataType))
    {
        std::ostringstream oss;
        oss << "SageAttention not supported for headDim=" << headDim << ", SM=" << smVersion
            << ", dataType=" << static_cast<int>(dataType);
        throw std::runtime_error(oss.str());
    }

    // Validate head dimension
    if (headDim % 64 != 0)
    {
        throw std::runtime_error("Head dimension must be a multiple of 64 for SageAttention");
    }

    // Validate GQA ratio
    if (numQoHeads % numKvHeads != 0)
    {
        throw std::runtime_error("numQoHeads must be divisible by numKvHeads for GQA support");
    }

    // Calculate workspace size (currently no workspace needed)
    mWorkspaceSize = 0;
}

size_t SageAttentionRunner::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

bool SageAttentionRunner::canImplement(
    int32_t headDim, int32_t sm, nvinfer1::DataType dataType) noexcept
{
    // SageAttention SM89 requires:
    // - SM 8.9+ (Ada Lovelace or newer)
    // - Head dimension: 64, 128, or 256
    // - Output dtype: FP16 (BF16 support depends on kernel variant)

    if (sm < 89)
    {
        return false;
    }

    if (headDim != 64 && headDim != 128 && headDim != 256)
    {
        return false;
    }

    if (dataType != nvinfer1::DataType::kHALF)
    {
        // Currently only FP16 output is supported
        // BF16 support can be added with different kernel variants
        return false;
    }

    return true;
}

void SageAttentionRunner::run(
    SageAttentionParams& params, void* /* workspace */, cudaStream_t stream)
{
    // Validate parameters
    SAGE_CHECK(params.q_ptr != nullptr, "Query pointer is null");
    SAGE_CHECK(params.k_ptr != nullptr, "Key pointer is null");
    SAGE_CHECK(params.v_ptr != nullptr, "Value pointer is null");
    SAGE_CHECK(params.o_ptr != nullptr, "Output pointer is null");
    SAGE_CHECK(params.q_scale_ptr != nullptr, "Query scale pointer is null");
    SAGE_CHECK(params.k_scale_ptr != nullptr, "Key scale pointer is null");

    // Fill in default parameters if not set
    if (params.batch_size == 0)
        params.batch_size = mBatchSize;
    if (params.qo_len == 0)
        params.qo_len = mQoLen;
    if (params.kv_len == 0)
        params.kv_len = mKvLen;
    if (params.num_qo_heads == 0)
        params.num_qo_heads = mNumQoHeads;
    if (params.num_kv_heads == 0)
        params.num_kv_heads = mNumKvHeads;
    if (params.head_dim == 0)
        params.head_dim = mHeadDim;

    // Set default sm_scale if not provided
    if (params.sm_scale == 0.0f)
    {
        params.sm_scale = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
    }

    // Set default tensor layout
    if (static_cast<int>(params.tensor_layout) == 0 && mTensorLayout != SageTensorLayout::kBSHD)
    {
        params.tensor_layout = mTensorLayout;
    }

    // Set default mask mode
    if (static_cast<int>(params.mask_mode) == 0)
    {
        params.mask_mode = mMaskMode;
    }

    // Set default quantization granularity
    if (static_cast<int>(params.qk_quant_gran) == 0)
    {
        params.qk_quant_gran = mQuantGran;
    }

    // Calculate strides if not provided (assuming BSHD layout by default)
    if (params.stride_bz_q == 0)
    {
        if (mTensorLayout == SageTensorLayout::kBSHD)
        {
            params.stride_bz_q = params.qo_len * params.num_qo_heads * params.head_dim;
            params.stride_seq_q = params.num_qo_heads * params.head_dim;
            params.stride_h_q = params.head_dim;

            params.stride_bz_k = params.kv_len * params.num_kv_heads * params.head_dim;
            params.stride_seq_k = params.num_kv_heads * params.head_dim;
            params.stride_h_k = params.head_dim;

            // V has transposed layout: [B, Dv, H_kv, S] for FP8 efficiency
            params.stride_bz_v = params.kv_len * params.num_kv_heads * params.head_dim;
            params.stride_h_v = params.kv_len;
            params.stride_d_v = params.num_kv_heads * params.kv_len;

            params.stride_bz_o = params.qo_len * params.num_qo_heads * params.head_dim;
            params.stride_seq_o = params.num_qo_heads * params.head_dim;
            params.stride_h_o = params.head_dim;
        }
        else
        {
            // BHSD layout
            params.stride_bz_q = params.num_qo_heads * params.qo_len * params.head_dim;
            params.stride_h_q = params.qo_len * params.head_dim;
            params.stride_seq_q = params.head_dim;

            params.stride_bz_k = params.num_kv_heads * params.kv_len * params.head_dim;
            params.stride_h_k = params.kv_len * params.head_dim;
            params.stride_seq_k = params.head_dim;

            params.stride_bz_v = params.num_kv_heads * params.kv_len * params.head_dim;
            params.stride_h_v = params.kv_len;
            params.stride_d_v = params.num_kv_heads * params.kv_len;

            params.stride_bz_o = params.num_qo_heads * params.qo_len * params.head_dim;
            params.stride_h_o = params.qo_len * params.head_dim;
            params.stride_seq_o = params.head_dim;
        }
    }

    // Launch kernel based on configuration
    int32_t const isCausal = (params.mask_mode == SageMaskMode::kCausal) ? 1 : 0;
    int32_t const qkQuantGranInt = static_cast<int32_t>(params.qk_quant_gran);
    int32_t const returnLseInt = params.return_lse ? 1 : 0;

    // Number of KV groups for GQA
    // int32_t const numKvGroups = params.num_qo_heads / params.num_kv_heads;

    // Grid and block dimensions
    // CTA_Q = 128, CTA_K = 64 for SM89 kernels
    constexpr int32_t CTA_Q = 128;
    constexpr int32_t CTA_K = 64;

    int32_t const numBlocksQ = sage::divCeil(params.qo_len, CTA_Q);
    int32_t const gridSizeX = numBlocksQ;
    int32_t const gridSizeY = params.num_qo_heads;
    int32_t const gridSizeZ = params.batch_size;

    // Shared memory size calculation
    // smem_max = max(CTA_Q * HEAD_DIM * sizeof(int8_t) + CTA_K * HEAD_DIM * sizeof(int8_t) +
    //                CTA_K * HEAD_DIM * sizeof(int8_t),
    //                CTA_Q * HEAD_DIM * sizeof(half))
    size_t smemSize = std::max(
        static_cast<size_t>(CTA_Q * params.head_dim + CTA_K * params.head_dim * 2),
        static_cast<size_t>(CTA_Q * params.head_dim * sizeof(half)));

    // Dispatch to appropriate kernel based on head dimension
    SAGE_DISPATCH_HEAD_DIM(params.head_dim, HEAD_DIM,
    {
        SAGE_DISPATCH_CAUSAL(isCausal, IS_CAUSAL,
        {
            SAGE_DISPATCH_QUANT_GRAN(qkQuantGranInt, QK_QUANT_GRAN,
            {
                SAGE_DISPATCH_RETURN_LSE(returnLseInt, RETURN_LSE,
                {
                    SAGE_DISPATCH_DTYPE_TO_CTYPE_FP16(mDataType, DTypeOut,
                    {
                        sage::launchSageAttentionKernel<HEAD_DIM, IS_CAUSAL, QK_QUANT_GRAN, RETURN_LSE, DTypeOut>(
                            params, gridSizeX, gridSizeY, gridSizeZ, smemSize, stream);
                    });
                });
            });
        });
    });

    // Check for kernel launch errors
    SAGE_CHECK_CUDA(cudaGetLastError());
}

} // namespace trt_edgellm
