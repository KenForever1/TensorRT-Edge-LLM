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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/sageAttentionKernels/sageAttentionRunner.h"
#include "sageAttentionTestKernels.h"
#include "testUtils.h"

using namespace nvinfer1;
using namespace trt_edgellm;

void TestSageAttentionAccuracy(int32_t batchSize, int32_t qoLen, int32_t kvLen, int32_t numQoHeads, int32_t numKvHeads,
    int32_t headDim, bool causal = true)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    // Check if SageAttention is supported
    if (!SageAttentionRunner::canImplement(headDim, smVersion, DataType::kHALF))
    {
        GTEST_SKIP() << "SageAttention not supported for headSize=" << headDim << ", SM=" << smVersion;
    }

    // Tensor sizes
    size_t const qSize = static_cast<size_t>(batchSize) * qoLen * numQoHeads * headDim;
    size_t const kSize = static_cast<size_t>(batchSize) * kvLen * numKvHeads * headDim;
    size_t const vSize = static_cast<size_t>(batchSize) * kvLen * numKvHeads * headDim;
    size_t const oSize = static_cast<size_t>(batchSize) * qoLen * numQoHeads * headDim;

    // Initialize input data (FP16)
    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kSize);
    std::vector<half> vInput(vSize);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    // Create tensors
    rt::Tensor qTensor({batchSize, qoLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({batchSize, kvLen, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({batchSize, kvLen, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor oTensorRef({batchSize, qoLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor oTensorSage({batchSize, qoLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    // Copy input to device
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMemcpy(qTensor.rawPointer(), qInput.data(), qSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kTensor.rawPointer(), kInput.data(), kSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vTensor.rawPointer(), vInput.data(), vSize * sizeof(half), cudaMemcpyHostToDevice));

    // Compute reference output (standard FP16 attention)
    launchSageReferenceAttention(static_cast<half const*>(qTensor.rawPointer()),
        static_cast<half const*>(kTensor.rawPointer()), static_cast<half const*>(vTensor.rawPointer()),
        static_cast<half*>(oTensorRef.rawPointer()), batchSize, qoLen, kvLen, numQoHeads, numKvHeads, headDim, causal,
        stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Create quantized tensors for SageAttention
    constexpr int32_t kCTA_Q = 128;
    constexpr int32_t kCTA_K = 64;
    constexpr int32_t kWARP_Q = 32;
    constexpr int32_t kWARP_K = 64;
    int32_t const numWarpsQ = kCTA_Q / kWARP_Q;
    int32_t const numWarpsK = kCTA_K / kWARP_K;
    int32_t const numBlocksQ = (qoLen + kCTA_Q - 1) / kCTA_Q;
    int32_t const numBlocksK = (kvLen + kCTA_K - 1) / kCTA_K;
    int32_t const paddedKvLen = numBlocksK * kCTA_K;
    int32_t const qScaleSize = batchSize * numQoHeads * numBlocksQ * numWarpsQ;
    int32_t const kScaleSize = batchSize * numKvHeads * numBlocksK * numWarpsK;

    // Q, K: INT8 with per-warp scales
    // V: FP8 (stored as int8_t) in transposed [B, D, H_kv, padded_S] layout
    rt::Tensor qInt8Tensor({batchSize, qoLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor kInt8Tensor({batchSize, kvLen, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor vFp8Tensor({batchSize, headDim, numKvHeads, paddedKvLen}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor qScaleTensor({qScaleSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor kScaleTensor({kScaleSize}, rt::DeviceType::kGPU, DataType::kFLOAT);

    // Quantize inputs
    launchQuantizeToInt8PerWarp(static_cast<half const*>(qTensor.rawPointer()),
        static_cast<int8_t*>(qInt8Tensor.rawPointer()), static_cast<float*>(qScaleTensor.rawPointer()), batchSize,
        qoLen, numQoHeads, headDim, true, stream);
    launchQuantizeToInt8PerWarp(static_cast<half const*>(kTensor.rawPointer()),
        static_cast<int8_t*>(kInt8Tensor.rawPointer()), static_cast<float*>(kScaleTensor.rawPointer()), batchSize,
        kvLen, numKvHeads, headDim, false, stream);
    launchQuantizeToFp8Transposed(static_cast<half const*>(vTensor.rawPointer()),
        static_cast<int8_t*>(vFp8Tensor.rawPointer()), batchSize, kvLen, numKvHeads, headDim, paddedKvLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Create SageAttention runner
    SageAttentionRunner runner(DataType::kHALF, batchSize, qoLen, kvLen, numQoHeads, numKvHeads, headDim, smVersion,
        SageTensorLayout::kBSHD, causal ? SageMaskMode::kCausal : SageMaskMode::kNone, SageQuantGranularity::kPerWarp);

    // Setup parameters
    SageAttentionParams params;
    params.q_ptr = static_cast<int8_t*>(qInt8Tensor.rawPointer());
    params.k_ptr = static_cast<int8_t*>(kInt8Tensor.rawPointer());
    params.v_ptr = static_cast<int8_t*>(vFp8Tensor.rawPointer());
    params.o_ptr = oTensorSage.rawPointer();
    params.q_scale_ptr = static_cast<float*>(qScaleTensor.rawPointer());
    params.k_scale_ptr = static_cast<float*>(kScaleTensor.rawPointer());
    params.batch_size = batchSize;
    params.qo_len = qoLen;
    params.kv_len = kvLen;
    params.num_qo_heads = numQoHeads;
    params.num_kv_heads = numKvHeads;
    params.head_dim = headDim;
    params.tensor_layout = SageTensorLayout::kBSHD;
    params.mask_mode = causal ? SageMaskMode::kCausal : SageMaskMode::kNone;
    params.qk_quant_gran = SageQuantGranularity::kPerWarp;
    params.stride_bz_q = qoLen * numQoHeads * headDim;
    params.stride_seq_q = numQoHeads * headDim;
    params.stride_h_q = headDim;
    params.stride_bz_k = kvLen * numKvHeads * headDim;
    params.stride_seq_k = numKvHeads * headDim;
    params.stride_h_k = headDim;
    params.stride_bz_v = headDim * numKvHeads * paddedKvLen;
    params.stride_h_v = paddedKvLen;
    params.stride_d_v = numKvHeads * paddedKvLen;
    params.stride_bz_o = qoLen * numQoHeads * headDim;
    params.stride_seq_o = numQoHeads * headDim;
    params.stride_h_o = headDim;

    // Run SageAttention
    runner.run(params, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // Copy outputs to host
    std::vector<half> outRef(oSize);
    std::vector<half> outSage(oSize);
    CUDA_CHECK(cudaMemcpy(outRef.data(), oTensorRef.rawPointer(), oSize * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outSage.data(), oTensorSage.rawPointer(), oSize * sizeof(half), cudaMemcpyDeviceToHost));

    // Check accuracy (SageAttention uses quantization, so tolerance is higher)
    int64_t const totalElements = static_cast<int64_t>(oSize);
    int32_t numClose = 0;
    float totalError = 0.0f;
    bool nanValueDetected = false;

    for (int64_t i = 0; i < totalElements; ++i)
    {
        float const ref = __half2float(outRef[i]);
        float const sage = __half2float(outSage[i]);
        if (std::isnan(sage))
        {
            nanValueDetected = true;
        }
        float const error = std::abs(ref - sage);
        totalError += error;

        // Allow 10% relative error or 0.2 absolute error for INT8/FP8 quantized attention.
        float const tolerance = std::max(std::abs(ref) * 0.10f, 0.20f);
        if (error <= tolerance)
        {
            numClose++;
        }
    }

    float const passRate = static_cast<float>(numClose) / totalElements;
    float const avgError = totalError / totalElements;

    std::cout << "SageAttention test: batch=" << batchSize << " qo_len=" << qoLen << " kv_len=" << kvLen
              << " num_qo_heads=" << numQoHeads << " num_kv_heads=" << numKvHeads << " head_dim=" << headDim
              << " causal=" << causal << " pass_rate=" << passRate << " avg_error=" << avgError << std::endl;

    EXPECT_FALSE(nanValueDetected);
    EXPECT_GT(passRate, 0.90);
}

// Test cases for different configurations

TEST(SageAttentionTest, BasicMHA)
{
    // Multi-head attention: numQoHeads == numKvHeads
    TestSageAttentionAccuracy(1, 512, 512, 8, 8, 128, true);
}

TEST(SageAttentionTest, GQARatio2)
{
    // Grouped-query attention with ratio 2
    TestSageAttentionAccuracy(1, 512, 512, 16, 8, 128, true);
}

TEST(SageAttentionTest, GQARatio4)
{
    // Grouped-query attention with ratio 4 (common in LLaMA-style models)
    TestSageAttentionAccuracy(1, 512, 512, 32, 8, 128, true);
}

TEST(SageAttentionTest, GQARatio8)
{
    // Grouped-query attention with ratio 8
    TestSageAttentionAccuracy(1, 512, 512, 32, 4, 128, true);
}

TEST(SageAttentionTest, HeadDim64)
{
    // Head dimension 64
    TestSageAttentionAccuracy(1, 256, 256, 16, 8, 64, true);
}

TEST(SageAttentionTest, HeadDim256)
{
    // Head dimension 256
    TestSageAttentionAccuracy(1, 256, 256, 8, 4, 256, true);
}

TEST(SageAttentionTest, NonCausal)
{
    // Non-causal attention (for encoder or VIT-style models)
    TestSageAttentionAccuracy(1, 256, 256, 16, 16, 128, false);
}

TEST(SageAttentionTest, LongSequence)
{
    // Longer sequence
    TestSageAttentionAccuracy(1, 1024, 1024, 16, 4, 128, true);
}

TEST(SageAttentionTest, BatchSize2)
{
    // Batch size > 1
    TestSageAttentionAccuracy(2, 256, 256, 16, 8, 128, true);
}

TEST(SageAttentionTest, DecodeQOLenLessThanKVLen)
{
    // Decode/chunked decode: queries are the latest tokens and attend to a longer KV cache.
    TestSageAttentionAccuracy(1, 128, 512, 16, 8, 128, true);
}
