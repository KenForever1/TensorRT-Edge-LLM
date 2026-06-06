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
#include "kernels/sageAttentionKernels/sageAttentionRuntimeKernels.h"
#include "kernels/sageAttentionKernels/sageAttentionRunner.h"
#include "kernels/sageAttentionKernels/sageAttentionHostUtils.h"
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

void TestSageAttentionDecodeKvCacheAccuracy(
    int32_t batchSize, int32_t qoLen, int32_t kvLen, int32_t numQoHeads, int32_t numKvHeads, int32_t headDim)
{
    ASSERT_LE(qoLen, kvLen);
    // SageAttention causal masking only works when qo_len == kv_len; for decode use non-causal.
    bool const causal = (qoLen == kvLen);
    TestSageAttentionAccuracy(batchSize, qoLen, kvLen, numQoHeads, numKvHeads, headDim, causal);
}

void TestSageAttentionRuntimeKvCacheHelperAccuracy(
    int32_t batchSize, int32_t kvLen, int32_t numQoHeads, int32_t numKvHeads, int32_t headDim)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    if (!SageAttentionRunner::canImplement(headDim, smVersion, DataType::kHALF))
    {
        GTEST_SKIP() << "SageAttention not supported for headSize=" << headDim << ", SM=" << smVersion;
    }

    constexpr int32_t kQOLen = 1;
    size_t const qSize = static_cast<size_t>(batchSize) * kQOLen * numQoHeads * headDim;
    size_t const kvSize = static_cast<size_t>(batchSize) * kvLen * numKvHeads * headDim;
    size_t const oSize = static_cast<size_t>(batchSize) * kQOLen * numQoHeads * headDim;
    size_t const kvCacheSize = static_cast<size_t>(batchSize) * 2 * numKvHeads * kvLen * headDim;

    std::vector<half> qInput(qSize);
    std::vector<half> kInput(kvSize);
    std::vector<half> vInput(kvSize);
    std::vector<half> kvCacheInput(kvCacheSize);
    std::vector<int32_t> sequenceLengths(batchSize, kvLen);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t seq = 0; seq < kvLen; ++seq)
        {
            for (int32_t head = 0; head < numKvHeads; ++head)
            {
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    int64_t const kvIdx
                        = (((static_cast<int64_t>(batch) * kvLen + seq) * numKvHeads + head) * headDim + dim);
                    int64_t const kCacheIdx
                        = (((static_cast<int64_t>(batch) * 2 * numKvHeads + head) * kvLen + seq) * headDim + dim);
                    int64_t const vCacheIdx
                        = (((static_cast<int64_t>(batch) * 2 * numKvHeads + numKvHeads + head) * kvLen + seq)
                              * headDim
                            + dim);
                    kvCacheInput[kCacheIdx] = kInput[kvIdx];
                    kvCacheInput[vCacheIdx] = vInput[kvIdx];
                }
            }
        }
    }

    rt::Tensor qTensor({batchSize, kQOLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kTensor({batchSize, kvLen, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({batchSize, kvLen, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvCacheTensor({batchSize, 2, numKvHeads, kvLen, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor sequenceLengthsTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor oTensorRef({batchSize, kQOLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor oTensorSage({batchSize, kQOLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMemcpy(qTensor.rawPointer(), qInput.data(), qSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kTensor.rawPointer(), kInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vTensor.rawPointer(), vInput.data(), kvSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(kvCacheTensor.rawPointer(), kvCacheInput.data(), kvCacheSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sequenceLengthsTensor.rawPointer(), sequenceLengths.data(), sequenceLengths.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));

    // SageAttention causal masking only works when qo_len == kv_len; for decode use non-causal.
    launchSageReferenceAttention(static_cast<half const*>(qTensor.rawPointer()),
        static_cast<half const*>(kTensor.rawPointer()), static_cast<half const*>(vTensor.rawPointer()),
        static_cast<half*>(oTensorRef.rawPointer()), batchSize, kQOLen, kvLen, numQoHeads, numKvHeads, headDim, false,
        stream);

    int32_t const paddedKvLen = sage::getSagePaddedKvLen(kvLen);
    rt::Tensor qInt8Tensor({batchSize, kQOLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor kInt8Tensor({batchSize, kvLen, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor vFp8Tensor({batchSize, headDim, numKvHeads, paddedKvLen}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor qScaleTensor({sage::getSageQScaleSize(batchSize, kQOLen, numQoHeads)}, rt::DeviceType::kGPU,
        DataType::kFLOAT);
    rt::Tensor kScaleTensor(
        {sage::getSageKScaleSize(batchSize, kvLen, numKvHeads)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor vScaleTensor(
        {sage::getSageVScaleSize(batchSize, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor kMeanTensor(
        {sage::getSageVScaleSize(batchSize, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor partialVMaxTensor(
        {sage::getSageStatsPartialsSize(batchSize, kvLen, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor partialKSumTensor(
        {sage::getSageStatsPartialsSize(batchSize, kvLen, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);

    sage::launchSageQuantizeQToInt8(qTensor, qInt8Tensor, qScaleTensor, stream);
    sage::launchSageConvertKVCacheToInt8AndFp8(
        kvCacheTensor, sequenceLengthsTensor, kInt8Tensor, vFp8Tensor, kScaleTensor, vScaleTensor, kMeanTensor, partialVMaxTensor, partialKSumTensor, kvLen, stream);

    // ----- DIAGNOSTIC: produce reference K-Int8/V-Fp8 with the proven test kernels and compare -----
    rt::Tensor kInt8Ref({batchSize, kvLen, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor vFp8Ref({batchSize, headDim, numKvHeads, paddedKvLen}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor kScaleRef(
        {sage::getSageKScaleSize(batchSize, kvLen, numKvHeads)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor qInt8Ref({batchSize, kQOLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor qScaleRef(
        {sage::getSageQScaleSize(batchSize, kQOLen, numQoHeads)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    launchQuantizeToInt8PerWarp(static_cast<half const*>(qTensor.rawPointer()),
        static_cast<int8_t*>(qInt8Ref.rawPointer()), static_cast<float*>(qScaleRef.rawPointer()), batchSize, kQOLen,
        numQoHeads, headDim, true, stream);
    launchQuantizeToInt8PerWarp(static_cast<half const*>(kTensor.rawPointer()),
        static_cast<int8_t*>(kInt8Ref.rawPointer()), static_cast<float*>(kScaleRef.rawPointer()), batchSize, kvLen,
        numKvHeads, headDim, false, stream);
    launchQuantizeToFp8Transposed(static_cast<half const*>(vTensor.rawPointer()),
        static_cast<int8_t*>(vFp8Ref.rawPointer()), batchSize, kvLen, numKvHeads, headDim, paddedKvLen, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    {
        size_t const kInt8Bytes = static_cast<size_t>(batchSize) * kvLen * numKvHeads * headDim;
        size_t const vFp8Bytes = static_cast<size_t>(batchSize) * headDim * numKvHeads * paddedKvLen;
        size_t const kScaleN = static_cast<size_t>(sage::getSageKScaleSize(batchSize, kvLen, numKvHeads));
        size_t const qInt8Bytes = static_cast<size_t>(batchSize) * kQOLen * numQoHeads * headDim;
        size_t const qScaleN = static_cast<size_t>(sage::getSageQScaleSize(batchSize, kQOLen, numQoHeads));
        std::vector<int8_t> kRtH(kInt8Bytes), kRefH(kInt8Bytes);
        std::vector<int8_t> vRtH(vFp8Bytes), vRefH(vFp8Bytes);
        std::vector<float> kSRtH(kScaleN), kSRefH(kScaleN);
        std::vector<int8_t> qRtH(qInt8Bytes), qRefH(qInt8Bytes);
        std::vector<float> qSRtH(qScaleN), qSRefH(qScaleN);
        CUDA_CHECK(cudaMemcpy(kRtH.data(), kInt8Tensor.rawPointer(), kInt8Bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kRefH.data(), kInt8Ref.rawPointer(), kInt8Bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vRtH.data(), vFp8Tensor.rawPointer(), vFp8Bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vRefH.data(), vFp8Ref.rawPointer(), vFp8Bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kSRtH.data(), kScaleTensor.rawPointer(), kScaleN * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kSRefH.data(), kScaleRef.rawPointer(), kScaleN * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qRtH.data(), qInt8Tensor.rawPointer(), qInt8Bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qRefH.data(), qInt8Ref.rawPointer(), qInt8Bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qSRtH.data(), qScaleTensor.rawPointer(), qScaleN * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qSRefH.data(), qScaleRef.rawPointer(), qScaleN * sizeof(float), cudaMemcpyDeviceToHost));
        size_t kDiff = 0, vDiff = 0, kSDiff = 0, qDiff = 0, qSDiff = 0;
        for (size_t i = 0; i < kInt8Bytes; ++i) if (kRtH[i] != kRefH[i]) ++kDiff;
        for (size_t i = 0; i < vFp8Bytes; ++i) if (vRtH[i] != vRefH[i]) ++vDiff;
        for (size_t i = 0; i < kScaleN; ++i) if (std::abs(kSRtH[i] - kSRefH[i]) > 1e-6f) ++kSDiff;
        for (size_t i = 0; i < qInt8Bytes; ++i) if (qRtH[i] != qRefH[i]) ++qDiff;
        for (size_t i = 0; i < qScaleN; ++i) if (std::abs(qSRtH[i] - qSRefH[i]) > 1e-6f) ++qSDiff;
        std::cout << "[DIAG] kInt8 mismatches=" << kDiff << "/" << kInt8Bytes
                  << " vFp8 mismatches=" << vDiff << "/" << vFp8Bytes
                  << " kScale mismatches=" << kSDiff << "/" << kScaleN
                  << " qInt8 mismatches=" << qDiff << "/" << qInt8Bytes
                  << " qScale mismatches=" << qSDiff << "/" << qScaleN
                  << std::endl;
        if (kSDiff > 0)
        {
            std::cout << "[DIAG] kScale first few: rt vs ref" << std::endl;
            for (size_t i = 0; i < std::min<size_t>(8, kScaleN); ++i)
            {
                std::cout << "  [" << i << "] " << kSRtH[i] << " vs " << kSRefH[i] << std::endl;
            }
        }
        if (qSDiff > 0)
        {
            std::cout << "[DIAG] qScale first few: rt vs ref" << std::endl;
            for (size_t i = 0; i < std::min<size_t>(8, qScaleN); ++i)
            {
                std::cout << "  [" << i << "] " << qSRtH[i] << " vs " << qSRefH[i] << std::endl;
            }
        }
        if (vDiff > 0)
        {
            // Sample V FP8 bytes at various (batch, dim, head, seq) positions to understand the diff pattern.
            std::cout << "[DIAG] vFp8 sample bytes (rt vs ref) at (b,d,h,s) -> linear_idx:" << std::endl;
            int32_t const samplePoints[][4] = {
                {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}, {0, 0, 0, 7},
                {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 64}, {0, 0, 0, 128},
                {0, 5, 3, 17}, {0, 64, 4, 100}, {0, 127, 7, 511},
            };
            for (auto const& pt : samplePoints)
            {
                int32_t const b = pt[0], d = pt[1], h = pt[2], s = pt[3];
                size_t const idx = ((static_cast<size_t>(b) * headDim + d) * numKvHeads + h) * paddedKvLen + s;
                if (idx < vFp8Bytes)
                {
                    std::cout << "  (b=" << b << ",d=" << d << ",h=" << h << ",s=" << s << ") idx=" << idx
                              << " rt=0x" << std::hex << (int) (uint8_t) vRtH[idx] << " ref=0x"
                              << (int) (uint8_t) vRefH[idx] << std::dec << std::endl;
                }
            }
            // Count how many runtime bytes are exactly zero (potential symptom of broken FP8 conversion or unwritten).
            size_t rtZeroCount = 0;
            size_t refZeroCount = 0;
            for (size_t i = 0; i < vFp8Bytes; ++i)
            {
                if (vRtH[i] == 0) ++rtZeroCount;
                if (vRefH[i] == 0) ++refZeroCount;
            }
            std::cout << "[DIAG] vFp8 zero bytes: rt=" << rtZeroCount << " ref=" << refZeroCount << " of "
                      << vFp8Bytes << std::endl;
        }
    }

    SageAttentionParams params;
    params.q_ptr = static_cast<int8_t*>(qInt8Tensor.rawPointer());
    params.k_ptr = static_cast<int8_t*>(kInt8Tensor.rawPointer());
    params.v_ptr = static_cast<int8_t*>(vFp8Tensor.rawPointer());
    params.o_ptr = oTensorSage.rawPointer();
    params.q_scale_ptr = static_cast<float*>(qScaleTensor.rawPointer());
    params.k_scale_ptr = static_cast<float*>(kScaleTensor.rawPointer());
    params.v_scale_ptr = static_cast<float*>(vScaleTensor.rawPointer());
    params.fuse_v_scale = true;
    params.batch_size = batchSize;
    params.qo_len = kQOLen;
    params.kv_len = kvLen;
    params.num_qo_heads = numQoHeads;
    params.num_kv_heads = numKvHeads;
    params.head_dim = headDim;
    params.tensor_layout = SageTensorLayout::kBSHD;
    params.mask_mode = SageMaskMode::kNone;
    params.qk_quant_gran = SageQuantGranularity::kPerWarp;
    params.stride_bz_q = kQOLen * numQoHeads * headDim;
    params.stride_seq_q = numQoHeads * headDim;
    params.stride_h_q = headDim;
    params.stride_bz_k = kvLen * numKvHeads * headDim;
    params.stride_seq_k = numKvHeads * headDim;
    params.stride_h_k = headDim;
    params.stride_bz_v = headDim * numKvHeads * paddedKvLen;
    params.stride_h_v = paddedKvLen;
    params.stride_d_v = numKvHeads * paddedKvLen;
    params.stride_bz_o = kQOLen * numQoHeads * headDim;
    params.stride_seq_o = numQoHeads * headDim;
    params.stride_h_o = headDim;

    SageAttentionRunner runner(DataType::kHALF, batchSize, kQOLen, kvLen, numQoHeads, numKvHeads, headDim, smVersion,
        SageTensorLayout::kBSHD, SageMaskMode::kNone, SageQuantGranularity::kPerWarp);
    runner.run(params, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> outRef(oSize);
    std::vector<half> outSage(oSize);
    CUDA_CHECK(cudaMemcpy(outRef.data(), oTensorRef.rawPointer(), oSize * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outSage.data(), oTensorSage.rawPointer(), oSize * sizeof(half), cudaMemcpyDeviceToHost));

    int32_t numClose = 0;
    float totalError = 0.0f;
    bool nanValueDetected = false;
    for (size_t i = 0; i < oSize; ++i)
    {
        float const ref = __half2float(outRef[i]);
        float const sageOutput = __half2float(outSage[i]);
        nanValueDetected |= std::isnan(sageOutput);
        float const error = std::abs(ref - sageOutput);
        totalError += error;
        float const tolerance = std::max(std::abs(ref) * 0.10f, 0.20f);
        if (error <= tolerance)
        {
            numClose++;
        }
    }

    float const passRate = static_cast<float>(numClose) / static_cast<float>(oSize);
    float const avgError = totalError / static_cast<float>(oSize);
    std::cout << "SageAttention runtime helper test: batch=" << batchSize << " kv_len=" << kvLen
              << " num_qo_heads=" << numQoHeads << " num_kv_heads=" << numKvHeads << " head_dim=" << headDim
              << " pass_rate=" << passRate << " avg_error=" << avgError << std::endl;

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

TEST(SageAttentionTest, SingleTokenDecode)
{
    // Token-by-token decode: one query token attends to the full KV cache.
    // SageAttention causal masking only works when qo_len == kv_len; for decode use non-causal.
    TestSageAttentionAccuracy(1, 1, 512, 16, 8, 128, false);
}

TEST(SageAttentionTest, SingleTokenDecodeKvCache)
{
    // Runtime-style decode: one query token attends to pre-existing KV cache tensors.
    TestSageAttentionDecodeKvCacheAccuracy(1, 1, 512, 16, 8, 128);
}

TEST(SageAttentionTest, BatchDecodeKvCache)
{
    // Runtime-style batched decode: each batch has one query token and its own KV cache.
    TestSageAttentionDecodeKvCacheAccuracy(2, 1, 512, 16, 8, 128);
}

TEST(SageAttentionTest, RuntimeKvCacheHelperSingleTokenDecode)
{
    // Runtime helper path: convert plugin-style KV cache then run SageAttention decode.
    TestSageAttentionRuntimeKvCacheHelperAccuracy(1, 512, 16, 8, 128);
}

TEST(SageAttentionTest, RuntimeKvCacheHelperBatchDecode)
{
    // Runtime helper path with batch > 1.
    TestSageAttentionRuntimeKvCacheHelperAccuracy(2, 512, 16, 8, 128);
}

namespace
{
// Verify that the SageAttention decode kernel honours device-side sequence_lengths
// so that capacity (layout kv_len) can exceed the effective kv_len. This exercises
// the CUDA-graph-compatible single-graph code path used by the attention plugin.
void TestSageAttentionDeviceSeqLensAccuracy(
    int32_t batchSize, int32_t kvCapacity, int32_t effectiveKvLen, int32_t numQoHeads, int32_t numKvHeads, int32_t headDim)
{
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);

    if (!SageAttentionRunner::canImplement(headDim, smVersion, DataType::kHALF))
    {
        GTEST_SKIP() << "SageAttention not supported for headSize=" << headDim << ", SM=" << smVersion;
    }
    ASSERT_LT(effectiveKvLen, kvCapacity);

    constexpr int32_t kQOLen = 1;
    size_t const qSize = static_cast<size_t>(batchSize) * kQOLen * numQoHeads * headDim;
    size_t const kvEffSize = static_cast<size_t>(batchSize) * effectiveKvLen * numKvHeads * headDim;
    size_t const oSize = static_cast<size_t>(batchSize) * kQOLen * numQoHeads * headDim;
    size_t const kvCacheSize = static_cast<size_t>(batchSize) * 2 * numKvHeads * kvCapacity * headDim;

    std::vector<half> qInput(qSize);
    std::vector<half> kEffInput(kvEffSize);
    std::vector<half> vEffInput(kvEffSize);
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    std::vector<int32_t> sequenceLengths(batchSize, effectiveKvLen);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kEffInput, -1.0f, 1.0f);
    uniformFloatInitialization(vEffInput, -1.0f, 1.0f);

    // Populate kv cache only at [0, effectiveKvLen); positions [effectiveKvLen, kvCapacity)
    // are left zero so we can also confirm the convert-kernel zero-fill doesn't leak through.
    for (int32_t batch = 0; batch < batchSize; ++batch)
    {
        for (int32_t seq = 0; seq < effectiveKvLen; ++seq)
        {
            for (int32_t head = 0; head < numKvHeads; ++head)
            {
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    int64_t const kvIdx = (((static_cast<int64_t>(batch) * effectiveKvLen + seq) * numKvHeads + head)
                            * headDim + dim);
                    int64_t const kCacheIdx = (((static_cast<int64_t>(batch) * 2 * numKvHeads + head) * kvCapacity + seq)
                            * headDim + dim);
                    int64_t const vCacheIdx
                        = (((static_cast<int64_t>(batch) * 2 * numKvHeads + numKvHeads + head) * kvCapacity + seq)
                              * headDim + dim);
                    kvCacheInput[kCacheIdx] = kEffInput[kvIdx];
                    kvCacheInput[vCacheIdx] = vEffInput[kvIdx];
                }
            }
        }
    }

    rt::Tensor qTensor({batchSize, kQOLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kRefTensor({batchSize, effectiveKvLen, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vRefTensor({batchSize, effectiveKvLen, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvCacheTensor({batchSize, 2, numKvHeads, kvCapacity, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor sequenceLengthsTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor oTensorRef({batchSize, kQOLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor oTensorSage({batchSize, kQOLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMemcpy(qTensor.rawPointer(), qInput.data(), qSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kRefTensor.rawPointer(), kEffInput.data(), kvEffSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vRefTensor.rawPointer(), vEffInput.data(), kvEffSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        kvCacheTensor.rawPointer(), kvCacheInput.data(), kvCacheSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sequenceLengthsTensor.rawPointer(), sequenceLengths.data(),
        sequenceLengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Reference attends only over [0, effectiveKvLen) so the SageAttention path must mask
    // out positions [effectiveKvLen, kvCapacity) to match.
    launchSageReferenceAttention(static_cast<half const*>(qTensor.rawPointer()),
        static_cast<half const*>(kRefTensor.rawPointer()), static_cast<half const*>(vRefTensor.rawPointer()),
        static_cast<half*>(oTensorRef.rawPointer()), batchSize, kQOLen, effectiveKvLen, numQoHeads, numKvHeads, headDim,
        false, stream);

    int32_t const paddedKvLen = sage::getSagePaddedKvLen(kvCapacity);
    rt::Tensor qInt8Tensor({batchSize, kQOLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor kInt8Tensor({batchSize, kvCapacity, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor vFp8Tensor({batchSize, headDim, numKvHeads, paddedKvLen}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor qScaleTensor(
        {sage::getSageQScaleSize(batchSize, kQOLen, numQoHeads)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor kScaleTensor(
        {sage::getSageKScaleSize(batchSize, kvCapacity, numKvHeads)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor vScaleTensor(
        {sage::getSageVScaleSize(batchSize, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor kMeanTensor(
        {sage::getSageVScaleSize(batchSize, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor partialVMaxTensor(
        {sage::getSageStatsPartialsSize(batchSize, kvCapacity, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor partialKSumTensor(
        {sage::getSageStatsPartialsSize(batchSize, kvCapacity, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);

    sage::launchSageQuantizeQToInt8(qTensor, qInt8Tensor, qScaleTensor, stream);
    sage::launchSageConvertKVCacheToInt8AndFp8(
        kvCacheTensor, sequenceLengthsTensor, kInt8Tensor, vFp8Tensor, kScaleTensor, vScaleTensor, kMeanTensor, partialVMaxTensor, partialKSumTensor, kvCapacity, stream);

    SageAttentionParams params;
    params.q_ptr = static_cast<int8_t*>(qInt8Tensor.rawPointer());
    params.k_ptr = static_cast<int8_t*>(kInt8Tensor.rawPointer());
    params.v_ptr = static_cast<int8_t*>(vFp8Tensor.rawPointer());
    params.o_ptr = oTensorSage.rawPointer();
    params.q_scale_ptr = static_cast<float*>(qScaleTensor.rawPointer());
    params.k_scale_ptr = static_cast<float*>(kScaleTensor.rawPointer());
    params.v_scale_ptr = static_cast<float*>(vScaleTensor.rawPointer());
    params.fuse_v_scale = true;
    params.sequence_lengths = static_cast<int32_t const*>(sequenceLengthsTensor.rawPointer());
    params.batch_size = batchSize;
    params.qo_len = kQOLen;
    params.kv_len = kvCapacity;
    params.num_qo_heads = numQoHeads;
    params.num_kv_heads = numKvHeads;
    params.head_dim = headDim;
    params.tensor_layout = SageTensorLayout::kBSHD;
    params.mask_mode = SageMaskMode::kNone;
    params.qk_quant_gran = SageQuantGranularity::kPerWarp;
    params.stride_bz_q = kQOLen * numQoHeads * headDim;
    params.stride_seq_q = numQoHeads * headDim;
    params.stride_h_q = headDim;
    params.stride_bz_k = kvCapacity * numKvHeads * headDim;
    params.stride_seq_k = numKvHeads * headDim;
    params.stride_h_k = headDim;
    params.stride_bz_v = headDim * numKvHeads * paddedKvLen;
    params.stride_h_v = paddedKvLen;
    params.stride_d_v = numKvHeads * paddedKvLen;
    params.stride_bz_o = kQOLen * numQoHeads * headDim;
    params.stride_seq_o = numQoHeads * headDim;
    params.stride_h_o = headDim;

    SageAttentionRunner runner(DataType::kHALF, batchSize, kQOLen, kvCapacity, numQoHeads, numKvHeads, headDim, smVersion,
        SageTensorLayout::kBSHD, SageMaskMode::kNone, SageQuantGranularity::kPerWarp);
    runner.run(params, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    std::vector<half> outRef(oSize);
    std::vector<half> outSage(oSize);
    CUDA_CHECK(cudaMemcpy(outRef.data(), oTensorRef.rawPointer(), oSize * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outSage.data(), oTensorSage.rawPointer(), oSize * sizeof(half), cudaMemcpyDeviceToHost));

    int32_t numClose = 0;
    float totalError = 0.0f;
    bool nanValueDetected = false;
    for (size_t i = 0; i < oSize; ++i)
    {
        float const ref = __half2float(outRef[i]);
        float const sageOutput = __half2float(outSage[i]);
        nanValueDetected |= std::isnan(sageOutput);
        float const error = std::abs(ref - sageOutput);
        totalError += error;
        float const tolerance = std::max(std::abs(ref) * 0.10f, 0.20f);
        if (error <= tolerance)
        {
            numClose++;
        }
    }

    float const passRate = static_cast<float>(numClose) / static_cast<float>(oSize);
    float const avgError = totalError / static_cast<float>(oSize);
    std::cout << "SageAttention device-seq-lens test: batch=" << batchSize << " capacity=" << kvCapacity
              << " effective_kv_len=" << effectiveKvLen << " pass_rate=" << passRate << " avg_error=" << avgError
              << std::endl;

    EXPECT_FALSE(nanValueDetected);
    EXPECT_GT(passRate, 0.90);
}
} // namespace

TEST(SageAttentionTest, DeviceSequenceLengthsCapacityPadded)
{
    // capacity (layout kv_len) > effective kv_len read from device pointer.
    // This is the CUDA-graph-compatible decode path.
    TestSageAttentionDeviceSeqLensAccuracy(1, /*kvCapacity=*/512, /*effectiveKvLen=*/384, 16, 8, 128);
}

TEST(SageAttentionTest, DeviceSequenceLengthsBatchCapacityPadded)
{
    TestSageAttentionDeviceSeqLensAccuracy(2, /*kvCapacity=*/512, /*effectiveKvLen=*/384, 16, 8, 128);
}

// Compare K INT8 from fused kernel vs convert pipeline
TEST(SageAttentionTest, FusedKernelKInt8Compare)
{
    int32_t const batchSize = 1, qoLen = 1, kvLen = 64, numQoHeads = 4, numKvHeads = 4, headDim = 128;
    int32_t kvCacheCapacity = kvLen;

    // Generate test data
    std::vector<half> qInput(batchSize*qoLen*numQoHeads*headDim), kInput(batchSize*kvLen*numKvHeads*headDim),
                       vInput(kInput.size());
    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    // Populate KV cache
    int32_t kvCacheSize = batchSize * 2 * numKvHeads * kvCacheCapacity * headDim;
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    for (int32_t b = 0; b < batchSize; b++) for (int32_t h = 0; h < numKvHeads; h++)
        for (int32_t s = 0; s < kvLen; s++) for (int32_t d = 0; d < headDim; d++) {
            int64_t src = ((b*kvLen+s)*numKvHeads+h)*headDim+d;
            kvCacheInput[(((b*2*numKvHeads+h)*kvCacheCapacity+s)*headDim+d)] = kInput[src];
            kvCacheInput[(((b*2*numKvHeads+numKvHeads+h)*kvCacheCapacity+s)*headDim+d)] = vInput[src];
        }

    std::vector<int32_t> seqLens(batchSize, kvLen);

    // GPU tensors
    rt::Tensor qTensor({batchSize, qoLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor kvTensor({batchSize, 2, numKvHeads, kvCacheCapacity, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor slTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMemcpy(qTensor.rawPointer(), qInput.data(), qInput.size()*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kvTensor.rawPointer(), kvCacheInput.data(), kvCacheSize*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(slTensor.rawPointer(), seqLens.data(), batchSize*sizeof(int32_t), cudaMemcpyHostToDevice));

    // Reference: convert pipeline K INT8
    int32_t paddedKvLen = sage::getSagePaddedKvLen(kvCacheCapacity);
    rt::Tensor qInt8R({batchSize, qoLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor kInt8R({batchSize, kvCacheCapacity, numKvHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor vFp8R({batchSize, headDim, numKvHeads, paddedKvLen}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor qScaleR({sage::getSageQScaleSize(batchSize, qoLen, numQoHeads)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor kScaleR({sage::getSageKScaleSize(batchSize, kvCacheCapacity, numKvHeads)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor vScaleR({sage::getSageVScaleSize(batchSize, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor kMeanR({sage::getSageVScaleSize(batchSize, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor pvmR({sage::getSageStatsPartialsSize(batchSize, kvCacheCapacity, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor pksR({sage::getSageStatsPartialsSize(batchSize, kvCacheCapacity, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qTensor, qInt8R, qScaleR, stream);
    sage::launchSageConvertKVCacheToInt8AndFp8(kvTensor, slTensor, kInt8R, vFp8R, kScaleR, vScaleR, kMeanR, pvmR, pksR, kvCacheCapacity, stream);

    // Fused kernel: K INT8 debug output
    rt::Tensor qInt8F({batchSize, qoLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor qScaleF({sage::getSageQScaleSize(batchSize, qoLen, numQoHeads)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor vScaleCh({sage::getSageVScaleSize(batchSize, numKvHeads, headDim)}, rt::DeviceType::kGPU, DataType::kFLOAT);
    int32_t kDebugSize = batchSize * kvCacheCapacity * numKvHeads * headDim;
    rt::Tensor kDebug({kDebugSize}, rt::DeviceType::kGPU, DataType::kINT8);
    int32_t ksDebugSize = batchSize * numKvHeads;
    rt::Tensor ksDebug({ksDebugSize}, rt::DeviceType::kGPU, DataType::kFLOAT);
    int32_t vDebugSize = batchSize * kvCacheCapacity * numKvHeads * headDim;
    rt::Tensor vDebug({vDebugSize}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor oF({batchSize, qoLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);

    sage::launchSageQuantizeQToInt8(qTensor, qInt8F, qScaleF, stream);
    sage::launchSageComputeVChannelScales(kvTensor, slTensor, vScaleCh, kvCacheCapacity, stream);
    int32_t stBz = qoLen * numQoHeads * headDim, stSeq = numQoHeads * headDim;
    float smS = 1.0f / std::sqrt((float)headDim);
    sage::launchSageAttentionFused(
        static_cast<int8_t*>(qInt8F.rawPointer()), static_cast<half const*>(kvTensor.rawPointer()),
        static_cast<half*>(oF.rawPointer()), static_cast<float*>(qScaleF.rawPointer()),
        static_cast<float*>(vScaleCh.rawPointer()), nullptr, static_cast<int32_t const*>(slTensor.rawPointer()),
        batchSize, numQoHeads, numKvHeads, kvCacheCapacity,
        stBz, stSeq, headDim, stBz, stSeq, headDim, smS, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare K scales
    {
        int32_t ksN = sage::getSageKScaleSize(batchSize, kvCacheCapacity, numKvHeads);
        std::vector<float> ksRef(ksN);
        CUDA_CHECK(cudaMemcpy(ksRef.data(), kScaleR.rawPointer(), ksN*sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "K scales (ref): ";
        for (int32_t i = 0; i < std::min(5, ksN); i++) std::cout << ksRef[i] << " ";
        std::cout << std::endl;
        std::vector<float> ksFused(ksDebugSize);
        CUDA_CHECK(cudaMemcpy(ksFused.data(), ksDebug.rawPointer(), ksDebugSize*sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "K scales (fused): ";
        for (int32_t i = 0; i < std::min(5, ksDebugSize); i++) std::cout << ksFused[i] << " ";
        std::cout << std::endl;
    }

    // Compare K INT8
    std::vector<int8_t> kRef(kDebugSize), kFused(kDebugSize);
    CUDA_CHECK(cudaMemcpy(kRef.data(), kInt8R.rawPointer(), kDebugSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kFused.data(), kDebug.rawPointer(), kDebugSize, cudaMemcpyDeviceToHost));

    int32_t mismatches = 0, total = 0;
    for (int32_t b = 0; b < batchSize; b++) for (int32_t h = 0; h < numKvHeads; h++)
        for (int32_t s = 0; s < kvLen; s++) for (int32_t d = 0; d < headDim; d++) {
            // Both in [B][S][H][D] layout
            int64_t idx = ((b*kvCacheCapacity + s)*numKvHeads + h)*headDim + d;
            total++;
            if (kRef[idx] != kFused[idx]) mismatches++;
        }

    std::cout << "K INT8 comparison: mismatches=" << mismatches << "/" << total
              << " (" << (100.0f*mismatches/total) << "%)" << std::endl;
    if (mismatches > 0) {
        std::cout << "First 10 mismatches (b,h,s,d): ref vs fused" << std::endl;
        int cnt = 0;
        for (int32_t b = 0; b < batchSize && cnt < 10; b++) for (int32_t h = 0; h < numKvHeads && cnt < 10; h++)
            for (int32_t s = 0; s < kvLen && cnt < 10; s++) for (int32_t d = 0; d < headDim && cnt < 10; d++) {
                int64_t idx = ((b*kvCacheCapacity + s)*numKvHeads + h)*headDim + d;
                if (kRef[idx] != kFused[idx]) {
                    std::cout << "  (" << b << "," << h << "," << s << "," << d
                              << ") ref=" << (int)kRef[idx] << " fused=" << (int)kFused[idx] << std::endl;
                    cnt++;
                }
            }
    }
    // K INT8 differs due to K-mean (fused kernel doesn't apply K-mean)
    // K-mean is softmax-invariant, so this doesn't affect attention output
    std::cout << "  (expected: K-mean not applied in fused kernel)" << std::endl;

    // Compare V-FP8 (fused bypass smem vs reference, accounting for K-mean effect on ref)
    {
        std::vector<int8_t> vRef(vDebugSize), vFused(vDebugSize);
        CUDA_CHECK(cudaMemcpy(vRef.data(), vFp8R.rawPointer(), vFp8R.getMemoryCapacity(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vFused.data(), vDebug.rawPointer(), vDebugSize, cudaMemcpyDeviceToHost));
        int32_t vMismatches = 0, vTotal = 0;
        for (int32_t b = 0; b < batchSize; b++) for (int32_t h = 0; h < numKvHeads; h++)
            for (int32_t s = 0; s < kvLen; s++) for (int32_t d = 0; d < headDim; d++) {
                int32_t mod16 = s%16, ps = (mod16/8)*2 + ((mod16/2)%4)*4 + (mod16%2);
                int32_t permS = (s/16)*16 + ps;
                int64_t rIdx = ((b*headDim+d)*numKvHeads+h)*paddedKvLen + permS;
                int64_t fIdx = ((b*kvCacheCapacity+s)*numKvHeads+h)*headDim + d;
                vTotal++;
                if (vRef[rIdx] != vFused[fIdx]) vMismatches++;
            }
        std::cout << "V-FP8 comparison (bypass smem): mismatches=" << vMismatches << "/" << vTotal
                  << " (" << (100.0f*vMismatches/vTotal) << "%)" << std::endl;
        if (vMismatches > 0) {
            int cnt = 0;
            for (int32_t b = 0; b < batchSize && cnt < 5; b++) for (int32_t h = 0; h < numKvHeads && cnt < 5; h++)
                for (int32_t s = 0; s < kvLen && cnt < 5; s++) for (int32_t d = 0; d < headDim && cnt < 5; d++) {
                    int32_t mod16 = s%16, ps = (mod16/8)*2 + ((mod16/2)%4)*4 + (mod16%2);
                    int64_t rIdx = ((b*headDim+d)*numKvHeads+h)*paddedKvLen + (s/16)*16+ps;
                    int64_t fIdx = ((b*kvCacheCapacity+s)*numKvHeads+h)*headDim+d;
                    if (vRef[rIdx] != vFused[fIdx]) {
                        std::cout << "  (" << b << "," << h << "," << s << "," << d
                                  << ") ref=" << (int)vRef[rIdx] << " fused=" << (int)vFused[fIdx] << std::endl;
                        cnt++;
                    }
                }
        }
    }

    // Compare attention output
    {
        std::vector<half> oRef(batchSize*qoLen*numQoHeads*headDim), oFused(oRef.size());
        rt::Tensor oRefT({batchSize, qoLen, numQoHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
        SageAttentionParams rp{};
        rp.q_ptr = static_cast<int8_t*>(qInt8R.rawPointer());
        rp.k_ptr = static_cast<int8_t*>(kInt8R.rawPointer());
        rp.v_ptr = static_cast<int8_t*>(vFp8R.rawPointer());
        rp.o_ptr = oRefT.rawPointer();
        rp.q_scale_ptr = static_cast<float*>(qScaleR.rawPointer());
        rp.k_scale_ptr = static_cast<float*>(kScaleR.rawPointer());
        rp.v_scale_ptr = static_cast<float*>(vScaleR.rawPointer());
        rp.fuse_v_scale = true;
        rp.batch_size = batchSize; rp.qo_len = qoLen; rp.kv_len = kvCacheCapacity;
        rp.num_qo_heads = numQoHeads; rp.num_kv_heads = numKvHeads; rp.head_dim = headDim;
        rp.tensor_layout = SageTensorLayout::kBSHD; rp.mask_mode = SageMaskMode::kNone;
        rp.qk_quant_gran = SageQuantGranularity::kPerWarp;
        rp.stride_bz_q = qoLen*numQoHeads*headDim; rp.stride_seq_q = numQoHeads*headDim; rp.stride_h_q = headDim;
        rp.stride_bz_k = kvCacheCapacity*numKvHeads*headDim; rp.stride_seq_k = numKvHeads*headDim; rp.stride_h_k = headDim;
        rp.stride_bz_v = headDim*numKvHeads*paddedKvLen; rp.stride_h_v = paddedKvLen; rp.stride_d_v = numKvHeads*paddedKvLen;
        rp.stride_bz_o = qoLen*numQoHeads*headDim; rp.stride_seq_o = numQoHeads*headDim; rp.stride_h_o = headDim;
        SageAttentionRunner rr(DataType::kHALF, batchSize, qoLen, kvCacheCapacity, numQoHeads, numKvHeads, headDim, 89);
        rr.run(rp, nullptr, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(oRef.data(), oRefT.rawPointer(), oRef.size()*sizeof(half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(oFused.data(), oF.rawPointer(), oFused.size()*sizeof(half), cudaMemcpyDeviceToHost));
        int32_t oM = 0; float maxD = 0, sumD = 0, sumR = 0;
        for (size_t i = 0; i < oRef.size(); i++) {
            float r = __half2float(oRef[i]), f = __half2float(oFused[i]);
            float d = fabsf(r-f); sumD += d; sumR += fabsf(r); maxD = fmaxf(maxD, d);
            if (d > 0.001f*fmaxf(fabsf(r),0.001f)) oM++;
        }
        std::cout << "Attention output: mismatches=" << oM << "/" << oRef.size()
                  << " maxDiff=" << maxD << " avgRelErr=" << (sumR>0?sumD/sumR:0) << std::endl;
    }
}

// Compare fused vs original kernel with IDENTICAL K/V quantization (no K-mean, single FP8)
TEST(SageAttentionTest, FusedKernelMatchedQuant)
{
    int32_t const B=1,qL=1,kL=64,Hq=4,Hkv=4,D=128,pkL=sage::getSagePaddedKvLen(kL);
    // Generate data
    std::vector<half> qH(B*qL*Hq*D),kH(B*kL*Hkv*D),vH(kH.size()),kvH(B*2*Hkv*kL*D,__float2half(0));
    uniformFloatInitialization(qH,-1,1); uniformFloatInitialization(kH,-1,1); uniformFloatInitialization(vH,-1,1);
    for(int b=0;b<B;b++)for(int h=0;h<Hkv;h++)for(int s=0;s<kL;s++)for(int d=0;d<D;d++){
        auto src=((b*kL+s)*Hkv+h)*D+d;
        kvH[(((b*2*Hkv+h)*kL+s)*D+d)]=kH[src];
        kvH[(((b*2*Hkv+Hkv+h)*kL+s)*D+d)]=vH[src];
    }
    std::vector<int32_t> sl(B,kL);
    cudaStream_t s=nullptr;
    auto mt=[&](auto sh,auto dt){return rt::Tensor(sh,rt::DeviceType::kGPU,dt);};
    auto dt_sz=[](nvinfer1::DataType t)->size_t{return t==DataType::kHALF?2:t==DataType::kFLOAT?4:1;};
    auto cp=[&](auto&h,auto&d){CUDA_CHECK(cudaMemcpy(d.rawPointer(),h.data(),h.size()*dt_sz(d.getDataType()),cudaMemcpyHostToDevice));};

    rt::Tensor qT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF); cp(qH,qT);
    rt::Tensor kvT=mt(rt::Coords{B,2,Hkv,kL,D},DataType::kHALF); cp(kvH,kvT);
    rt::Tensor slT=mt(rt::Coords{B},DataType::kINT32); cp(sl,slT);
    rt::Tensor qI=mt(rt::Coords{B,qL,Hq,D},DataType::kINT8);
    rt::Tensor qS=mt(rt::Coords{sage::getSageQScaleSize(B,qL,Hq)},DataType::kFLOAT);
    rt::Tensor vsC=mt(rt::Coords{sage::getSageVScaleSize(B,Hkv,D)},DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT,qI,qS,s);
    sage::launchSageComputeVChannelScales(kvT,slT,vsC,kL,s);

    // Get K-scales from convert pipeline
    rt::Tensor kSR=mt(rt::Coords{sage::getSageKScaleSize(B,kL,Hkv)},DataType::kFLOAT);
    {rt::Tensor kIR=mt(rt::Coords{B,kL,Hkv,D},DataType::kINT8);
     rt::Tensor vFR=mt(rt::Coords{B,D,Hkv,pkL},DataType::kINT8);
     rt::Tensor vSR=mt(rt::Coords{sage::getSageVScaleSize(B,Hkv,D)},DataType::kFLOAT);
     rt::Tensor kMR=mt(rt::Coords{sage::getSageVScaleSize(B,Hkv,D)},DataType::kFLOAT);
     rt::Tensor pvR=mt(rt::Coords{sage::getSageStatsPartialsSize(B,kL,Hkv,D)},DataType::kFLOAT);
     rt::Tensor pkR=mt(rt::Coords{sage::getSageStatsPartialsSize(B,kL,Hkv,D)},DataType::kFLOAT);
     sage::launchSageConvertKVCacheToInt8AndFp8(kvT,slT,kIR,vFR,kSR,vSR,kMR,pvR,pkR,kL,s);}
    CUDA_CHECK(cudaStreamSynchronize(s));
    auto kSH=std::vector<float>(B*Hkv*1); CUDA_CHECK(cudaMemcpy(kSH.data(),kSR.rawPointer(),kSH.size()*4,cudaMemcpyDeviceToHost));
    auto vsH=std::vector<float>(B*Hkv*D); CUDA_CHECK(cudaMemcpy(vsH.data(),vsC.rawPointer(),vsH.size()*4,cudaMemcpyDeviceToHost));

    // Host-compute K_INT8 (no K-mean!)
    auto kI=std::vector<int8_t>(B*kL*Hkv*D,0);
    for(int b=0;b<B;b++)for(int h=0;h<Hkv;h++)for(int s=0;s<kL;s++)for(int d=0;d<D;d++){
        int64_t src=((b*kL+s)*Hkv+h)*D+d; float fv=__half2float(kH[src]);
        int qi=(int)nearbyintf(fv/kSH[b*Hkv+h]); kI[((b*kL+s)*Hkv+h)*D+d]=(int8_t)std::min(127,std::max(-127,qi));
    }
    // Host-compute V_FP8 (single conversion)
    auto vF8=std::vector<int8_t>(B*D*Hkv*pkL,0);
    for(int b=0;b<B;b++)for(int h=0;h<Hkv;h++)for(int s=0;s<kL;s++)for(int d=0;d<D;d++){
        float fv=__half2float(vH[((b*kL+s)*Hkv+h)*D+d]); float vs=vsH[(b*Hkv+h)*D+d];
        __nv_fp8_e4m3 f8(fv/vs); int m=s%16,ps=(m/8)*2+((m/2)%4)*4+(m%2);
        vF8[((b*D+d)*Hkv+h)*pkL+(s/16)*16+ps]=*(int8_t*)&f8;
    }

    rt::Tensor kIT=mt(rt::Coords{B,kL,Hkv,D},DataType::kINT8); cp(kI,kIT);
    rt::Tensor vFT=mt(rt::Coords{B,D,Hkv,pkL},DataType::kINT8); cp(vF8,vFT);
    rt::Tensor oRT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF);
    rt::Tensor oFT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF);

    // Original kernel with host-computed K/V
    {SageAttentionParams rp{}; rp.q_ptr=(int8_t*)qI.rawPointer(); rp.k_ptr=kIT.dataPointer<int8_t>(); rp.v_ptr=vFT.dataPointer<int8_t>();
     rp.o_ptr=oRT.rawPointer(); rp.q_scale_ptr=(float*)qS.rawPointer(); rp.k_scale_ptr=(float*)kSR.rawPointer(); rp.v_scale_ptr=(float*)vsC.rawPointer();
     rp.fuse_v_scale=true; rp.batch_size=B;rp.qo_len=qL;rp.kv_len=kL;rp.num_qo_heads=Hq;rp.num_kv_heads=Hkv;rp.head_dim=D;
     rp.tensor_layout=SageTensorLayout::kBSHD;rp.mask_mode=SageMaskMode::kNone;rp.qk_quant_gran=SageQuantGranularity::kPerWarp;
     rp.stride_bz_q=qL*Hq*D;rp.stride_seq_q=Hq*D;rp.stride_h_q=D;rp.stride_bz_k=kL*Hkv*D;rp.stride_seq_k=Hkv*D;rp.stride_h_k=D;
     rp.stride_bz_v=D*Hkv*pkL;rp.stride_h_v=pkL;rp.stride_d_v=Hkv*pkL;rp.stride_bz_o=qL*Hq*D;rp.stride_seq_o=Hq*D;rp.stride_h_o=D;
     SageAttentionRunner rr(DataType::kHALF,B,qL,kL,Hq,Hkv,D,89); rr.run(rp,nullptr,s);}

    // Fused kernel
    {int sb=qL*Hq*D,ss=Hq*D; float sm=1.0f/sqrtf((float)D);
     sage::launchSageAttentionFused((int8_t*)qI.rawPointer(),(half const*)kvT.rawPointer(),(half*)oFT.rawPointer(),
         (float*)qS.rawPointer(),(float*)vsC.rawPointer(),nullptr,(int32_t const*)slT.rawPointer(),B,Hq,Hkv,kL,sb,ss,D,sb,ss,D,sm,s);}
    CUDA_CHECK(cudaStreamSynchronize(s));

    // Compare
    auto oR=std::vector<half>(B*qL*Hq*D),oF=oR; cp(oR,oRT); cp(oF,oFT);
    int mM=0;float md=0,sd=0,sr=0;
    for(size_t i=0;i<oR.size();i++){float r=__half2float(oR[i]),f=__half2float(oF[i]);float d=fabsf(r-f);sd+=d;sr+=fabsf(r);md=fmaxf(md,d);if(d>0.001f*fmaxf(fabsf(r),0.001f))mM++;}
    std::cout<<"MATCHED-QUANT: mismatches="<<mM<<"/"<<oR.size()<<" maxDiff="<<md<<" avgRelErr="<<(sr>0?sd/sr:0)<<std::endl;
    if(mM>0){for(int i=0;i<5&&i<(int)oR.size();i++)std::cout<<" ["<<i<<"] ref="<<__half2float(oR[i])<<" fused="<<__half2float(oF[i])<<" diff="<<fabsf(__half2float(oR[i])-__half2float(oF[i]))<<std::endl;}
    EXPECT_LT(mM,(int)oR.size()/2);
}

TEST(SageAttentionTest, MatchedQuant256H8) {
    int32_t const B=1,qL=1,kL=256,Hq=8,Hkv=8,D=128,pkL=sage::getSagePaddedKvLen(kL);
    std::vector<half> qH(B*qL*Hq*D),kH(B*kL*Hkv*D),vH(kH.size()),kvH(int64_t(B)*2*Hkv*kL*D,__float2half(0.0f));
    uniformFloatInitialization(qH,-1,1); uniformFloatInitialization(kH,-1,1); uniformFloatInitialization(vH,-1,1);
    for(int b=0;b<B;b++)for(int h=0;h<Hkv;h++)for(int s=0;s<kL;s++)for(int d=0;d<D;d++){
        auto src=((int64_t(b)*kL+s)*Hkv+h)*D+d;
        kvH[(((int64_t(b)*2*Hkv+h)*kL+s)*D+d)]=kH[src];
        kvH[(((int64_t(b)*2*Hkv+Hkv+h)*kL+s)*D+d)]=vH[src];
    }
    std::vector<int32_t> sl(B,kL); cudaStream_t s=nullptr;
    auto mt=[&](auto sh,auto dt){return rt::Tensor(sh,rt::DeviceType::kGPU,dt);};
    auto dsz=[](nvinfer1::DataType t)->size_t{return t==DataType::kHALF?2:t==DataType::kFLOAT?4:1;};
    auto cp=[&](auto&h,auto&d){CUDA_CHECK(cudaMemcpy(d.rawPointer(),h.data(),h.size()*dsz(d.getDataType()),cudaMemcpyHostToDevice));};
    rt::Tensor qT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF); cp(qH,qT);
    rt::Tensor kvT=mt(rt::Coords{B,2,Hkv,kL,D},DataType::kHALF); cp(kvH,kvT);
    rt::Tensor slT=mt(rt::Coords{B},DataType::kINT32); cp(sl,slT);
    rt::Tensor qI=mt(rt::Coords{B,qL,Hq,D},DataType::kINT8);
    rt::Tensor qS=mt(rt::Coords{sage::getSageQScaleSize(B,qL,Hq)},DataType::kFLOAT);
    rt::Tensor vsC=mt(rt::Coords{sage::getSageVScaleSize(B,Hkv,D)},DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT,qI,qS,s);
    sage::launchSageComputeVChannelScales(kvT,slT,vsC,kL,s);
    rt::Tensor kSR=mt(rt::Coords{sage::getSageKScaleSize(B,kL,Hkv)},DataType::kFLOAT);
    {rt::Tensor kIR=mt(rt::Coords{B,kL,Hkv,D},DataType::kINT8);rt::Tensor vFR=mt(rt::Coords{B,D,Hkv,pkL},DataType::kINT8);rt::Tensor vSR=mt(rt::Coords{sage::getSageVScaleSize(B,Hkv,D)},DataType::kFLOAT);rt::Tensor kMR=mt(rt::Coords{sage::getSageVScaleSize(B,Hkv,D)},DataType::kFLOAT);rt::Tensor pvR=mt(rt::Coords{sage::getSageStatsPartialsSize(B,kL,Hkv,D)},DataType::kFLOAT);rt::Tensor pkR=mt(rt::Coords{sage::getSageStatsPartialsSize(B,kL,Hkv,D)},DataType::kFLOAT);sage::launchSageConvertKVCacheToInt8AndFp8(kvT,slT,kIR,vFR,kSR,vSR,kMR,pvR,pkR,kL,s);}
    CUDA_CHECK(cudaStreamSynchronize(s));
    auto kSH=std::vector<float>(B*Hkv*(kL/64)); CUDA_CHECK(cudaMemcpy(kSH.data(),kSR.rawPointer(),kSH.size()*4,cudaMemcpyDeviceToHost));
    auto vsH=std::vector<float>(B*Hkv*D); CUDA_CHECK(cudaMemcpy(vsH.data(),vsC.rawPointer(),vsH.size()*4,cudaMemcpyDeviceToHost));
    auto kI=std::vector<int8_t>(int64_t(B)*kL*Hkv*D,0);
    for(int b=0;b<B;b++)for(int h=0;h<Hkv;h++)for(int s=0;s<kL;s++)for(int d=0;d<D;d++){
        int64_t src=((int64_t(b)*kL+s)*Hkv+h)*D+d; float fv=__half2float(kH[src]);
        int blk=s/64; int qi=(int)nearbyintf(fv/kSH[int64_t(b)*Hkv*(kL/64)+h*(kL/64)+blk]);
        kI[((int64_t(b)*kL+s)*Hkv+h)*D+d]=(int8_t)std::min(127,std::max(-127,qi));
    }
    auto vF8=std::vector<int8_t>(int64_t(B)*D*Hkv*pkL,0);
    for(int b=0;b<B;b++)for(int h=0;h<Hkv;h++)for(int s=0;s<kL;s++)for(int d=0;d<D;d++){
        float fv=__half2float(vH[((int64_t(b)*kL+s)*Hkv+h)*D+d]); float vs=vsH[(int64_t(b)*Hkv+h)*D+d];
        __nv_fp8_e4m3 f8(fv/vs); int m=s%16,ps=(m/8)*2+((m/2)%4)*4+(m%2);
        vF8[((int64_t(b)*D+d)*Hkv+h)*pkL+(s/16)*16+ps]=*(int8_t*)&f8;
    }
    rt::Tensor kIT=mt(rt::Coords{B,kL,Hkv,D},DataType::kINT8); cp(kI,kIT);
    rt::Tensor vFT=mt(rt::Coords{B,D,Hkv,pkL},DataType::kINT8); cp(vF8,vFT);
    rt::Tensor oRT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF); rt::Tensor oFT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF);
    {SageAttentionParams rp{}; rp.q_ptr=(int8_t*)qI.rawPointer(); rp.k_ptr=kIT.dataPointer<int8_t>(); rp.v_ptr=vFT.dataPointer<int8_t>();rp.o_ptr=oRT.rawPointer();rp.q_scale_ptr=(float*)qS.rawPointer();rp.k_scale_ptr=(float*)kSR.rawPointer();rp.v_scale_ptr=(float*)vsC.rawPointer();rp.fuse_v_scale=true;rp.batch_size=B;rp.qo_len=qL;rp.kv_len=kL;rp.num_qo_heads=Hq;rp.num_kv_heads=Hkv;rp.head_dim=D;rp.tensor_layout=SageTensorLayout::kBSHD;rp.mask_mode=SageMaskMode::kNone;rp.qk_quant_gran=SageQuantGranularity::kPerWarp;rp.stride_bz_q=qL*Hq*D;rp.stride_seq_q=Hq*D;rp.stride_h_q=D;rp.stride_bz_k=kL*Hkv*D;rp.stride_seq_k=Hkv*D;rp.stride_h_k=D;rp.stride_bz_v=D*Hkv*pkL;rp.stride_h_v=pkL;rp.stride_d_v=Hkv*pkL;rp.stride_bz_o=qL*Hq*D;rp.stride_seq_o=Hq*D;rp.stride_h_o=D;SageAttentionRunner rr(DataType::kHALF,B,qL,kL,Hq,Hkv,D,89);rr.run(rp,nullptr,s);}
    {int sb=qL*Hq*D,ss=Hq*D;float sm=1.0f/sqrtf((float)D);sage::launchSageAttentionFused((int8_t*)qI.rawPointer(),(half const*)kvT.rawPointer(),(half*)oFT.rawPointer(),(float*)qS.rawPointer(),(float*)vsC.rawPointer(),nullptr,(int32_t const*)slT.rawPointer(),B,Hq,Hkv,kL,sb,ss,D,sb,ss,D,sm,s);}
    CUDA_CHECK(cudaStreamSynchronize(s));
    auto oR=std::vector<half>(qH.size()),oF=oR; cp(oR,oRT); cp(oF,oFT);
    int mM=0;float md=0,sd=0,sr=0;
    for(size_t i=0;i<oR.size();i++){float r=__half2float(oR[i]),f=__half2float(oF[i]);float d=fabsf(r-f);sd+=d;sr+=fabsf(r);md=fmaxf(md,d);if(d>0.001f*fmaxf(fabsf(r),0.001f))mM++;}
    std::cout<<"MATCHED_k256_H8: m="<<mM<<"/"<<oR.size()<<" md="<<md<<" ae="<<(sr>0?sd/sr:0)<<std::endl;
    EXPECT_LT(mM,(int)oR.size()/2);
}

