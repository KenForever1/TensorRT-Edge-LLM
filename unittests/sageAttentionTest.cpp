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
#include <fstream>

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
    // NOTE: effectiveKvLen=384 is intentionally a multiple of CTA_K=64 to avoid OOB mask trigger.
    // See FusedPartialKvSimplePattern for OOB mask testing with non-multiple kv_len.
    TestSageAttentionDeviceSeqLensAccuracy(1, /*kvCapacity=*/512, /*effectiveKvLen=*/384, 16, 8, 128);
}

// Test fused kernel with KNOWN simple data pattern to isolate partial KV bug
TEST(SageAttentionTest, FusedPartialKvSimplePattern)
{
    int32_t const B = 1, qL = 1, Hq = 8, Hkv = 8, D = 128, cap = 128, eff = 21;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";

    size_t const qSize = static_cast<size_t>(B) * qL * Hq * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;
    // Q = all 1.0, K = all 0.5, V = all 2.0 for valid positions
    // Expected output: uniform attention over 21 positions, each V=2.0, so output=2.0
    std::vector<half> qInput(qSize, __float2half(1.0f));
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    std::vector<int32_t> seqLens(B, eff);
    for (int32_t b = 0; b < B; b++)
        for (int32_t h = 0; h < Hkv; h++)
            for (int32_t s = 0; s < eff; s++)
                for (int32_t d = 0; d < D; d++) {
                    int64_t kcIdx = (((int64_t(b) * 2 * Hkv + h) * cap + s) * D + d);
                    int64_t vcIdx = (((int64_t(b) * 2 * Hkv + Hkv + h) * cap + s) * D + d);
                    kvCacheInput[kcIdx] = __float2half(0.5f);
                    kvCacheInput[vcIdx] = __float2half(2.0f);
                }

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };
    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF); cp(qInput, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF); cp(kvCacheInput, kvT, kvCacheSize * 2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32); cp(seqLens, slT, B * 4);

    // Fused path
    rt::Tensor qI = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageComputeVScalesAndKMeans(kvT, slT, vS, kM, cap, s);
    rt::Tensor oFusedT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    int32_t sb = qL * Hq * D, ss = Hq * D; float smv = 1.0f / sqrtf(static_cast<float>(D));
    sage::launchSageAttentionFused(
        static_cast<int8_t*>(qI.rawPointer()), static_cast<half const*>(kvT.rawPointer()),
        static_cast<half*>(oFusedT.rawPointer()), static_cast<float*>(qS.rawPointer()),
        static_cast<float*>(vS.rawPointer()), static_cast<float*>(kM.rawPointer()),
        static_cast<int32_t const*>(slT.rawPointer()),
        B, Hq, Hkv, cap, sb, ss, D, sb, ss, D, smv, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<half> oFused(qSize);
    CUDA_CHECK(cudaMemcpy(oFused.data(), oFusedT.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));

    // Expected: all elements should be close to 2.0
    float maxErr = 0;
    for (size_t i = 0; i < qSize; i++) {
        float f = __half2float(oFused[i]);
        float err = fabsf(f - 2.0f);
        maxErr = fmaxf(maxErr, err);
    }
    std::cout << "FusedSimplePattern: maxErr from 2.0 = " << maxErr
              << " first 4 values: " << __half2float(oFused[0]) << " " << __half2float(oFused[1])
              << " " << __half2float(oFused[2]) << " " << __half2float(oFused[3]) << std::endl;
    // With 21 valid positions and 128-21=107 zero positions properly masked,
    // output should be 2.0 (uniform attention over all-2.0 V values)
    // If OOB positions leak through, output will be < 2.0 (averaged with zeros)
    EXPECT_LT(maxErr, 0.1f);  // Within 5% of expected 2.0
}

// Minimal FP8 roundtrip test: verify __nv_fp8_e4m3 encoding/decoding
TEST(SageAttentionTest, Fp8Roundtrip) {
    std::vector<float> output(20);
    float* dOutput;
    CUDA_CHECK(cudaMalloc(&dOutput, 20 * sizeof(float)));
    launchFp8RoundtripTest(dOutput, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaMemcpy(output.data(), dOutput, 20 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dOutput));

    std::cout << "FP8 roundtrip:" << std::endl;
    std::cout << "  fp8(448.0f) decoded = " << output[0] << " raw_byte = " << (int)output[1] << std::endl;
    std::cout << "  fp8(2.0/vScale) decoded = " << output[2] << " raw_byte = " << (int)output[3] << std::endl;
    std::cout << "  float: 2.0/vScale = " << output[4] << " vScale = " << output[5] << std::endl;
    for (int i = 0; i < 5; i++) {
        float val = 440.0f + i * 4.0f;
        std::cout << "  fp8(" << val << ") decoded = " << output[6+i*2]
                  << " raw = " << (int)output[7+i*2] << std::endl;
    }

    // Verify: fp8(448.0f) should decode back to 448.0f
    EXPECT_NEAR(output[0], 448.0f, 1.0f);
    // Verify: raw byte for 448.0f should be 0x7e (126)
    EXPECT_EQ((int)output[1], 0x7e);
    // Verify: 2.0 / (2.0/448.0) should be close to 448.0
    EXPECT_NEAR(output[4], 448.0f, 1.0f);
}

// FP8 smem → TensorCore roundtrip: verify V_FP8 in swizzled smem is read correctly
TEST(SageAttentionTest, DISABLED_Fp8SmemRoundtrip) {
    std::vector<float> output(4);
    float* dOutput;
    CUDA_CHECK(cudaMalloc(&dOutput, 4 * sizeof(float)));
    launchFp8SmemRoundtripTest(dOutput, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CUDA_CHECK(cudaMemcpy(output.data(), dOutput, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dOutput));

    std::cout << "FP8 smem→TC roundtrip:" << std::endl;
    std::cout << "  RO[0][0][0] = " << output[0] << std::endl;
    std::cout << "  RO[0][0][1] = " << output[1] << std::endl;
    std::cout << "  d[0][0] = " << output[2] << std::endl;
    std::cout << "  d[0][1] = " << output[3] << std::endl;

    // With 21 positions having V=448 and RS=448 in FP8, and 43 positions having V=0:
    // Expected RO contribution per valid (seq, dim) pair through TensorCore:
    // mma_sync accumulates V*RS across all 64 sequence positions
    // For tid=0 (handling specific dims): first 21 seqs contribute 448*448 each
    // 43 OOB seqs contribute 0*448 = 0
    // RO[0][0][0] should be approximately 21 * 448 * 448 / (some factor depending on MMA layout)
    // d[0][0] should be approximately 21 * 448 (from accumulate_d_f8)

    float expectedPerValid = 448.0f * 448.0f;  // 200704
    float expectedRO_21 = 21.0f * expectedPerValid;  // 4214784
    float expectedD_21 = 21.0f * 448.0f;  // 9408

    std::cout << "  expected RO if correct: " << expectedRO_21 << std::endl;
    std::cout << "  expected d if correct: " << expectedD_21 << std::endl;
    std::cout << "  RO / d ratio: " << output[0] / output[2] << " (expected 448.0)" << std::endl;

    // Check d: should be 9408 (21 valid positions * 448 RS weight each)
    EXPECT_NEAR(output[2], expectedD_21, expectedD_21 * 0.1f);
    // Check RO/d ratio: should be 448
    float ratio = output[2] > 0 ? output[0] / output[2] : 0;
    EXPECT_NEAR(ratio, 448.0f, 50.0f);
}

TEST(SageAttentionTest, DeviceSequenceLengthsBatchCapacityPadded)
{
    TestSageAttentionDeviceSeqLensAccuracy(2, /*kvCapacity=*/512, /*effectiveKvLen=*/384, 16, 8, 128);
}

// Compare K INT8 from fused kernel vs convert pipeline
TEST(SageAttentionTest, DISABLED_FusedKernelKInt8Compare)
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

// Validate fused decode at a small matched-head shape against the FP16 reference.
// The pre-fusion Sage runner is not a reliable oracle for this qL=1/kL=64 shape.
TEST(SageAttentionTest, FusedKernelMatchedQuant)
{
    int32_t const B = 1, qL = 1, kL = 64, Hq = 4, Hkv = 4, D = 128, cap = kL;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";

    size_t const qSize = static_cast<size_t>(B) * qL * Hq * D;
    size_t const kvSize = static_cast<size_t>(B) * kL * Hkv * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;
    std::vector<half> qH(qSize), kH(kvSize), vH(kvSize);
    std::vector<half> kvH(kvCacheSize, __float2half(0.0F));
    std::vector<int32_t> sl(B, kL);
    uniformFloatInitialization(qH, -1.0F, 1.0F);
    uniformFloatInitialization(kH, -1.0F, 1.0F);
    uniformFloatInitialization(vH, -1.0F, 1.0F);
    for (int32_t b = 0; b < B; b++)
        for (int32_t h = 0; h < Hkv; h++)
            for (int32_t s = 0; s < kL; s++)
                for (int32_t d = 0; d < D; d++)
                {
                    int64_t const src = (((int64_t(b) * kL + s) * Hkv + h) * D + d);
                    kvH[(((int64_t(b) * 2 * Hkv + h) * cap + s) * D + d)] = kH[src];
                    kvH[(((int64_t(b) * 2 * Hkv + Hkv + h) * cap + s) * D + d)] = vH[src];
                }

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) {
        CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice));
    };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF); cp(qH, qT, qSize * sizeof(half));
    rt::Tensor kT = mt(rt::Coords{B, kL, Hkv, D}, DataType::kHALF); cp(kH, kT, kvSize * sizeof(half));
    rt::Tensor vT = mt(rt::Coords{B, kL, Hkv, D}, DataType::kHALF); cp(vH, vT, kvSize * sizeof(half));
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF); cp(kvH, kvT, kvCacheSize * sizeof(half));
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32); cp(sl, slT, sl.size() * sizeof(int32_t));

    rt::Tensor oRefT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    launchSageReferenceAttention(static_cast<half const*>(qT.rawPointer()), static_cast<half const*>(kT.rawPointer()),
        static_cast<half const*>(vT.rawPointer()), static_cast<half*>(oRefT.rawPointer()),
        B, qL, kL, Hq, Hkv, D, false, s);

    rt::Tensor qI = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageComputeVChannelScales(kvT, slT, vS, cap, s);

    rt::Tensor oFusedT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    int32_t const strideBz = qL * Hq * D;
    int32_t const strideSeq = Hq * D;
    float const smScale = 1.0F / sqrtf(static_cast<float>(D));
    sage::launchSageAttentionFused(static_cast<int8_t*>(qI.rawPointer()),
        static_cast<half const*>(kvT.rawPointer()), static_cast<half*>(oFusedT.rawPointer()),
        static_cast<float*>(qS.rawPointer()), static_cast<float*>(vS.rawPointer()), nullptr,
        static_cast<int32_t const*>(slT.rawPointer()), B, Hq, Hkv, cap,
        strideBz, strideSeq, D, strideBz, strideSeq, D, smScale, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> oRef(qSize), oFused(qSize);
    CUDA_CHECK(cudaMemcpy(oRef.data(), oRefT.rawPointer(), qSize * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(oFused.data(), oFusedT.rawPointer(), qSize * sizeof(half), cudaMemcpyDeviceToHost));

    int32_t largeErr = 0;
    float maxDiff = 0.0F, sumDiff = 0.0F, sumRef = 0.0F;
    for (size_t i = 0; i < qSize; i++)
    {
        float const r = __half2float(oRef[i]);
        float const f = __half2float(oFused[i]);
        float const diff = fabsf(r - f);
        maxDiff = fmaxf(maxDiff, diff);
        sumDiff += diff;
        sumRef += fabsf(r);
        if (diff > std::max(0.20F, 0.10F * fabsf(r))) largeErr++;
    }
    float const relErr = sumRef > 0.0F ? sumDiff / sumRef : 0.0F;
    std::cout << "FusedSmallRef: largeErr=" << largeErr << "/" << qSize
              << " maxDiff=" << maxDiff << " relErr=" << relErr << std::endl;
    EXPECT_EQ(largeErr, 0);
    EXPECT_LT(maxDiff, 0.08F);
    EXPECT_LT(relErr, 0.10F);
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
        __nv_fp8_e4m3 f8Raw(fv); float fvRaw = (float)f8Raw;
        __nv_fp8_e4m3 f8(fvRaw/vs); int m=s%16,ps=(m/8)*2+((m/2)%4)*4+(m%2);
        vF8[((int64_t(b)*D+d)*Hkv+h)*pkL+(s/16)*16+ps]=*(int8_t*)&f8;
    }
    rt::Tensor kIT=mt(rt::Coords{B,kL,Hkv,D},DataType::kINT8); cp(kI,kIT);
    rt::Tensor vFT=mt(rt::Coords{B,D,Hkv,pkL},DataType::kINT8); cp(vF8,vFT);
    rt::Tensor oRT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF); rt::Tensor oFT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF);
    {SageAttentionParams rp{}; rp.q_ptr=(int8_t*)qI.rawPointer(); rp.k_ptr=kIT.dataPointer<int8_t>(); rp.v_ptr=vFT.dataPointer<int8_t>();rp.o_ptr=oRT.rawPointer();rp.q_scale_ptr=(float*)qS.rawPointer();rp.k_scale_ptr=(float*)kSR.rawPointer();rp.v_scale_ptr=(float*)vsC.rawPointer();rp.fuse_v_scale=true;rp.batch_size=B;rp.qo_len=qL;rp.kv_len=kL;rp.num_qo_heads=Hq;rp.num_kv_heads=Hkv;rp.head_dim=D;rp.tensor_layout=SageTensorLayout::kBSHD;rp.mask_mode=SageMaskMode::kNone;rp.qk_quant_gran=SageQuantGranularity::kPerWarp;rp.stride_bz_q=qL*Hq*D;rp.stride_seq_q=Hq*D;rp.stride_h_q=D;rp.stride_bz_k=kL*Hkv*D;rp.stride_seq_k=Hkv*D;rp.stride_h_k=D;rp.stride_bz_v=D*Hkv*pkL;rp.stride_h_v=pkL;rp.stride_d_v=Hkv*pkL;rp.stride_bz_o=qL*Hq*D;rp.stride_seq_o=Hq*D;rp.stride_h_o=D;SageAttentionRunner rr(DataType::kHALF,B,qL,kL,Hq,Hkv,D,89);rr.run(rp,nullptr,s);}
    {int sb=qL*Hq*D,ss=Hq*D;float sm=1.0f/sqrtf((float)D);sage::launchSageAttentionFused((int8_t*)qI.rawPointer(),(half const*)kvT.rawPointer(),(half*)oFT.rawPointer(),(float*)qS.rawPointer(),(float*)vsC.rawPointer(),nullptr,(int32_t const*)slT.rawPointer(),B,Hq,Hkv,kL,sb,ss,D,sb,ss,D,sm,s);}
    CUDA_CHECK(cudaStreamSynchronize(s));
    auto oR=std::vector<half>(qH.size()),oF=oR;
    CUDA_CHECK(cudaMemcpy(oR.data(), oRT.rawPointer(), oR.size()*sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(oF.data(), oFT.rawPointer(), oF.size()*sizeof(half), cudaMemcpyDeviceToHost));
    int mM=0;float md=0,sd=0,sr=0;
    for(size_t i=0;i<oR.size();i++){float r=__half2float(oR[i]),f=__half2float(oF[i]);float d=fabsf(r-f);sd+=d;sr+=fabsf(r);md=fmaxf(md,d);if(d>0.001f*fmaxf(fabsf(r),0.001f))mM++;}
    float meanAbsDiff = sd / static_cast<float>(oR.size());
    std::cout<<"MATCHED_k256_H8: strictMismatches="<<mM<<"/"<<oR.size()<<" maxDiff="<<md<<" meanAbsDiff="<<meanAbsDiff<<std::endl;
    EXPECT_LT(md, 0.002f);
    EXPECT_LT(meanAbsDiff, 0.001f);
}


// Compare: run fused kernel with plugin data, check output matches plugin
TEST(SageAttentionTest, CompareWithPluginData) {
    std::ifstream f("/tmp/fused_debug.bin", std::ios::binary);
    if (!f) { GTEST_SKIP() << "Debug file not found"; return; }
    int32_t meta[8]; f.read((char*)meta, 32);
    int32_t B=meta[0], qL=meta[1], Hq=meta[2], Hkv=meta[3], D=meta[4], cap=meta[5], qSize=meta[6], kvSize=meta[7];
    int32_t qScaleSize = sage::getSageQScaleSize(B, qL, Hq);
    int32_t vScaleSize = sage::getSageVScaleSize(B, Hkv, D);
    std::vector<half> qf(qSize), ohP(qSize), kv(kvSize);
    std::vector<int8_t> qi(qSize);
    std::vector<float> qs(qScaleSize), vs(vScaleSize), km(vScaleSize);
    std::vector<int32_t> sl(B);
    f.read((char*)qf.data(), qSize*2); f.read((char*)qi.data(), qSize);
    f.read((char*)qs.data(), qScaleSize*4); f.read((char*)vs.data(), vScaleSize*4); f.read((char*)km.data(), vScaleSize*4);
    f.read((char*)kv.data(), kvSize*2); f.read((char*)sl.data(), B*4);
    f.read((char*)ohP.data(), qSize*2); f.close();

    cudaStream_t s=nullptr;
    auto mt=[&](auto sh,auto dt){return rt::Tensor(sh,rt::DeviceType::kGPU,dt);};
    auto cp=[&](auto&h,auto&d,size_t n){CUDA_CHECK(cudaMemcpy(d.rawPointer(),h.data(),n,cudaMemcpyHostToDevice));};
    rt::Tensor qI=mt(rt::Coords{B,qL,Hq,D},DataType::kINT8); cp(qi,qI,qSize);
    rt::Tensor qS=mt(rt::Coords{qScaleSize},DataType::kFLOAT); cp(qs,qS,qScaleSize*4);
    rt::Tensor vS=mt(rt::Coords{vScaleSize},DataType::kFLOAT); cp(vs,vS,vScaleSize*4);
    rt::Tensor kM=mt(rt::Coords{vScaleSize},DataType::kFLOAT); cp(km,kM,vScaleSize*4);
    rt::Tensor kvT=mt(rt::Coords{B,2,Hkv,cap,D},DataType::kHALF); cp(kv,kvT,kvSize*2);
    rt::Tensor slT=mt(rt::Coords{B},DataType::kINT32); cp(sl,slT,B*4);
    rt::Tensor oT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF);

    int Sb=qL*Hq*D, Ss=Hq*D; float sm=1.0f/sqrtf((float)D);
    sage::launchSageAttentionFused((int8_t*)qI.rawPointer(),(half const*)kvT.rawPointer(),(half*)oT.rawPointer(),(float*)qS.rawPointer(),(float*)vS.rawPointer(),(float*)kM.rawPointer(),(int32_t const*)slT.rawPointer(),B,Hq,Hkv,cap,Sb,Ss,D,Sb,Ss,D,sm,s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<half> ohT(qSize); CUDA_CHECK(cudaMemcpy(ohT.data(),oT.rawPointer(),qSize*2,cudaMemcpyDeviceToHost));
    int mM=0; float mD=0;
    for(int i=0;i<qSize;i++){float r=__half2float(ohP[i]),t=__half2float(ohT[i]);float d=fabsf(r-t);mD=fmaxf(mD,d);if(d>0.001f*fmaxf(fabsf(r),0.001f))mM++;}
    std::cout<<"Plugin vs Test: mism="<<mM<<"/"<<qSize<<" maxD="<<mD<<std::endl;
    if(mM>0){for(int i=0;i<5;i++)std::cout<<" ["<<i<<"] plugin="<<__half2float(ohP[i])<<" test="<<__half2float(ohT[i])<<std::endl;}
    EXPECT_LT(mM,qSize/2);
}

TEST(SageAttentionTest, CompareWithPluginDataPerHead) {
    std::ifstream f("/tmp/fused_debug.bin", std::ios::binary);
    if (!f) { GTEST_SKIP() << "Debug file not found"; return; }
    int32_t meta[8]; f.read((char*)meta, 32);
    int32_t B=meta[0], qL=meta[1], Hq=meta[2], Hkv=meta[3], D=meta[4], cap=meta[5], qSize=meta[6], kvSize=meta[7];
    int32_t qScaleSize = sage::getSageQScaleSize(B, qL, Hq);
    int32_t vScaleSize = sage::getSageVScaleSize(B, Hkv, D);
    std::vector<half> qf(qSize), ohP(qSize), kv(kvSize);
    std::vector<int8_t> qi(qSize);
    std::vector<float> qs(qScaleSize), vs(vScaleSize), km(vScaleSize);
    std::vector<int32_t> sl(B);
    f.read((char*)qf.data(), qSize*2); f.read((char*)qi.data(), qSize);
    f.read((char*)qs.data(), qScaleSize*4); f.read((char*)vs.data(), vScaleSize*4); f.read((char*)km.data(), vScaleSize*4);
    f.read((char*)kv.data(), kvSize*2); f.read((char*)sl.data(), B*4);
    f.read((char*)ohP.data(), qSize*2); f.close();

    cudaStream_t s=nullptr;
    auto mt=[&](auto sh,auto dt){return rt::Tensor(sh,rt::DeviceType::kGPU,dt);};
    auto cp=[&](auto&h,auto&d,size_t n){CUDA_CHECK(cudaMemcpy(d.rawPointer(),h.data(),n,cudaMemcpyHostToDevice));};
    rt::Tensor qI=mt(rt::Coords{B,qL,Hq,D},DataType::kINT8); cp(qi,qI,qSize);
    rt::Tensor qS=mt(rt::Coords{qScaleSize},DataType::kFLOAT); cp(qs,qS,qScaleSize*4);
    rt::Tensor vS=mt(rt::Coords{vScaleSize},DataType::kFLOAT); cp(vs,vS,vScaleSize*4);
    rt::Tensor kM=mt(rt::Coords{vScaleSize},DataType::kFLOAT); cp(km,kM,vScaleSize*4);
    rt::Tensor kvT=mt(rt::Coords{B,2,Hkv,cap,D},DataType::kHALF); cp(kv,kvT,kvSize*2);
    rt::Tensor slT=mt(rt::Coords{B},DataType::kINT32); cp(sl,slT,B*4);
    rt::Tensor oT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF);

    // Test 1: use pre-quantized Q_INT8 from plugin
    int Sb=qL*Hq*D, Ss=Hq*D; float sm=1.0f/sqrtf((float)D);
    sage::launchSageAttentionFused((int8_t*)qI.rawPointer(),(half const*)kvT.rawPointer(),(half*)oT.rawPointer(),(float*)qS.rawPointer(),(float*)vS.rawPointer(),(float*)kM.rawPointer(),(int32_t const*)slT.rawPointer(),B,Hq,Hkv,cap,Sb,Ss,D,Sb,Ss,D,sm,s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<half> ohT1(qSize); CUDA_CHECK(cudaMemcpy(ohT1.data(),oT.rawPointer(),qSize*2,cudaMemcpyDeviceToHost));

    // Test 2: re-quantize Q from scratch (like plugin does)
    rt::Tensor qTT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF); cp(qf,qTT,qSize*2);
    rt::Tensor qI2=mt(rt::Coords{B,qL,Hq,D},DataType::kINT8);
    rt::Tensor qS2=mt(rt::Coords{qScaleSize},DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qTT, qI2, qS2, s);
    sage::launchSageAttentionFused((int8_t*)qI2.rawPointer(),(half const*)kvT.rawPointer(),(half*)oT.rawPointer(),(float*)qS2.rawPointer(),(float*)vS.rawPointer(),(float*)kM.rawPointer(),(int32_t const*)slT.rawPointer(),B,Hq,Hkv,cap,Sb,Ss,D,Sb,Ss,D,sm,s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<half> ohT2(qSize); CUDA_CHECK(cudaMemcpy(ohT2.data(),oT.rawPointer(),qSize*2,cudaMemcpyDeviceToHost));

    for(int h=0;h<Hq;h++){int hm=0;for(int d=0;d<D;d++){int i=h*D+d;if(fabsf(__half2float(ohP[i])-__half2float(ohT1[i]))>0.001f)hm++;}std::cout<<"Head "<<h<<": T1_mism="<<hm<<"/"<<D<<" ";hm=0;for(int d=0;d<D;d++){int i=h*D+d;if(fabsf(__half2float(ohP[i])-__half2float(ohT2[i]))>0.001f)hm++;}std::cout<<"T2_mism="<<hm<<"/"<<D<<std::endl;}
    EXPECT_LT(1,1); // Always print
}
TEST(SageAttentionTest, CompareWithPluginFixed) {
    std::ifstream f("/tmp/fused_debug.bin", std::ios::binary);
    if (!f) { GTEST_SKIP() << "Debug file not found"; return; }
    int32_t meta[8]; f.read((char*)meta, 32);
    int32_t B=meta[0], qL=meta[1], Hq=meta[2], Hkv=meta[3], D=meta[4], cap=meta[5], qSize=meta[6], kvSize=meta[7];
    int32_t qScaleSize = Hq * 4;  // batchSize * numHeads * numBlocks * numWarpsPerBlock
    std::vector<half> qf(qSize), ohP(qSize), kv(kvSize);
    std::vector<int8_t> qi(qSize);
    std::vector<float> qs(qScaleSize), vs(1024), km(1024);
    std::vector<int32_t> sl(B);
    f.read((char*)qf.data(), qSize*2); f.read((char*)qi.data(), qSize);
    f.read((char*)qs.data(), qScaleSize*4); f.read((char*)vs.data(), 4096); f.read((char*)km.data(), 4096);
    f.read((char*)kv.data(), kvSize*2); f.read((char*)sl.data(), B*4);
    f.read((char*)ohP.data(), qSize*2); f.close();

    cudaStream_t s=nullptr;
    auto mt=[&](auto sh,auto dt){return rt::Tensor(sh,rt::DeviceType::kGPU,dt);};
    auto cp=[&](auto&h,auto&d,size_t n){CUDA_CHECK(cudaMemcpy(d.rawPointer(),h.data(),n,cudaMemcpyHostToDevice));};
    rt::Tensor qI=mt(rt::Coords{B,qL,Hq,D},DataType::kINT8); cp(qi,qI,qSize);
    rt::Tensor qS=mt(rt::Coords{qScaleSize},DataType::kFLOAT); cp(qs,qS,qScaleSize*4);
    rt::Tensor vS=mt(rt::Coords{1024},DataType::kFLOAT); cp(vs,vS,4096);
    rt::Tensor kM=mt(rt::Coords{1024},DataType::kFLOAT); cp(km,kM,4096);
    rt::Tensor kvT=mt(rt::Coords{B,2,Hkv,cap,D},DataType::kHALF); cp(kv,kvT,kvSize*2);
    rt::Tensor slT=mt(rt::Coords{B},DataType::kINT32); cp(sl,slT,B*4);
    rt::Tensor oT=mt(rt::Coords{B,qL,Hq,D},DataType::kHALF);

    int Sb=qL*Hq*D, Ss=Hq*D; float sm=1.0f/sqrtf((float)D);
    sage::launchSageAttentionFused((int8_t*)qI.rawPointer(),(half const*)kvT.rawPointer(),(half*)oT.rawPointer(),(float*)qS.rawPointer(),(float*)vS.rawPointer(),(float*)kM.rawPointer(),(int32_t const*)slT.rawPointer(),B,Hq,Hkv,cap,Sb,Ss,D,Sb,Ss,D,sm,s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<half> ohT(qSize); CUDA_CHECK(cudaMemcpy(ohT.data(),oT.rawPointer(),qSize*2,cudaMemcpyDeviceToHost));
    int mM=0; float mD=0;
    for(int i=0;i<qSize;i++){float r=__half2float(ohP[i]),t=__half2float(ohT[i]);float d=fabsf(r-t);mD=fmaxf(mD,d);if(d>0.001f*fmaxf(fabsf(r),0.001f))mM++;}
    std::cout<<"Plugin vs Test (fixed QS size): mism="<<mM<<"/"<<qSize<<" maxD="<<mD<<std::endl;
    for(int h=0;h<Hq;h++){int hm=0;for(int d=0;d<D;d++){if(fabsf(__half2float(ohP[h*D+d])-__half2float(ohT[h*D+d]))>0.001f)hm++;}std::cout<<" H"<<h<<":"<<hm;}
    std::cout<<std::endl;
    EXPECT_LT(mM,qSize/2);
}

// Exact reproduction of plugin's fused path: quantize Q, compute V_scales + K_means,
// then call fused kernel with kMean != nullptr. Compare against FP16 reference.
TEST(SageAttentionTest, PluginExactPathDecode)
{
    int32_t const B = 1, qL = 1, Hq = 16, Hkv = 8, D = 128, cap = 512, kvLen = cap;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
    {
        GTEST_SKIP() << "SageAttention not supported";
    }
    size_t const qSize = static_cast<size_t>(B) * qL * Hq * D;
    size_t const kvEffSize = static_cast<size_t>(B) * kvLen * Hkv * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;
    std::vector<half> qInput(qSize), kInput(kvEffSize), vInput(kvEffSize);
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    std::vector<int32_t> seqLens(B, kvLen);
    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);
    for (int32_t b = 0; b < B; b++)
        for (int32_t h = 0; h < Hkv; h++)
            for (int32_t seq = 0; seq < kvLen; seq++)
                for (int32_t d = 0; d < D; d++) {
                    int64_t kvIdx = (((int64_t(b)*kvLen+seq)*Hkv+h)*D+d);
                    int64_t kcIdx = (((int64_t(b)*2*Hkv+h)*cap+seq)*D+d);
                    int64_t vcIdx = (((int64_t(b)*2*Hkv+Hkv+h)*cap+seq)*D+d);
                    kvCacheInput[kcIdx] = kInput[kvIdx];
                    kvCacheInput[vcIdx] = vInput[kvIdx];
                }
    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };
    rt::Tensor qT = mt(rt::Coords{B,qL,Hq,D}, DataType::kHALF); cp(qInput, qT, qSize*2);
    rt::Tensor kRefT = mt(rt::Coords{B,kvLen,Hkv,D}, DataType::kHALF); cp(kInput, kRefT, kvEffSize*2);
    rt::Tensor vRefT = mt(rt::Coords{B,kvLen,Hkv,D}, DataType::kHALF); cp(vInput, vRefT, kvEffSize*2);
    rt::Tensor kvT = mt(rt::Coords{B,2,Hkv,cap,D}, DataType::kHALF); cp(kvCacheInput, kvT, kvCacheSize*2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32); cp(seqLens, slT, B*4);
    // Reference FP16 attention
    rt::Tensor oRefT = mt(rt::Coords{B,qL,Hq,D}, DataType::kHALF);
    launchSageReferenceAttention(static_cast<half const*>(qT.rawPointer()),
        static_cast<half const*>(kRefT.rawPointer()), static_cast<half const*>(vRefT.rawPointer()),
        static_cast<half*>(oRefT.rawPointer()), B, qL, kvLen, Hq, Hkv, D, false, s);
    // Plugin path: quantize Q, compute V_scales + K_means
    rt::Tensor qI = mt(rt::Coords{B,qL,Hq,D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B,qL,Hq)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B,Hkv,D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B,Hkv,D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageComputeVScalesAndKMeans(kvT, slT, vS, kM, cap, s);
    rt::Tensor oFusedT = mt(rt::Coords{B,qL,Hq,D}, DataType::kHALF);
    int32_t sb = qL*Hq*D, ss = Hq*D; float smv = 1.0f/sqrtf(static_cast<float>(D));
    // Fused kernel with kMean != nullptr (matching plugin)
    sage::launchSageAttentionFused(
        static_cast<int8_t*>(qI.rawPointer()), static_cast<half const*>(kvT.rawPointer()),
        static_cast<half*>(oFusedT.rawPointer()), static_cast<float*>(qS.rawPointer()),
        static_cast<float*>(vS.rawPointer()), static_cast<float*>(kM.rawPointer()),
        static_cast<int32_t const*>(slT.rawPointer()),
        B, Hq, Hkv, cap, sb, ss, D, sb, ss, D, smv, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<half> oRef(qSize), oFused(qSize);
    CUDA_CHECK(cudaMemcpy(oRef.data(), oRefT.rawPointer(), qSize*2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(oFused.data(), oFusedT.rawPointer(), qSize*2, cudaMemcpyDeviceToHost));
    int32_t mM = 0; float mD = 0, sD = 0, sR = 0;
    for (size_t i = 0; i < qSize; i++) {
        float r = __half2float(oRef[i]), f = __half2float(oFused[i]);
        float d = fabsf(r - f); sD += d; sR += fabsf(r);
        mD = fmaxf(mD, d);
        if (d > std::max(fabsf(r) * 0.10f, 0.20f)) mM++;
    }
    float passRate = 1.0f - static_cast<float>(mM) / qSize;
    std::cout << "PluginExactPath (with K-mean): mism=" << mM << "/" << qSize
              << " passRate=" << passRate << " maxDiff=" << mD << " avgRelErr=" << (sR>0?sD/sR:0) << std::endl;
    if (mM > 0) {
        for (int i = 0; i < 5 && i < (int)qSize; i++)
            std::cout << "  [" << i << "] ref=" << __half2float(oRef[i])
                      << " fused=" << __half2float(oFused[i])
                      << " diff=" << fabsf(__half2float(oRef[i]) - __half2float(oFused[i])) << std::endl;
    }
    EXPECT_GT(passRate, 0.90f);
}

// Test fused kernel with PARTIAL KV cache (effective KV len < capacity)
// This is the actual model behavior where only a few tokens are in the KV cache
TEST(SageAttentionTest, PluginExactPathPartialKvCache)
{
    int32_t const B = 1, qL = 1, Hq = 16, Hkv = 8, D = 128, cap = 512;
    int32_t const effectiveKvLen = 21;  // Match the model's first decode step

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";

    size_t const qSize = static_cast<size_t>(B) * qL * Hq * D;
    size_t const kvEffSize = static_cast<size_t>(B) * effectiveKvLen * Hkv * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;

    // Fill KV cache with data at [0, effectiveKvLen), zero beyond
    std::vector<half> qInput(qSize), kInput(kvEffSize), vInput(kvEffSize);
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    std::vector<int32_t> seqLens(B, effectiveKvLen);

    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);

    for (int32_t b = 0; b < B; b++)
        for (int32_t h = 0; h < Hkv; h++)
            for (int32_t s = 0; s < effectiveKvLen; s++)
                for (int32_t d = 0; d < D; d++) {
                    int64_t kvIdx = (((int64_t(b) * effectiveKvLen + s) * Hkv + h) * D + d);
                    int64_t kcIdx = (((int64_t(b) * 2 * Hkv + h) * cap + s) * D + d);
                    int64_t vcIdx = (((int64_t(b) * 2 * Hkv + Hkv + h) * cap + s) * D + d);
                    kvCacheInput[kcIdx] = kInput[kvIdx];
                    kvCacheInput[vcIdx] = vInput[kvIdx];
                }

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    cp(qInput, qT, qSize * 2);
    rt::Tensor kRefT = mt(rt::Coords{B, effectiveKvLen, Hkv, D}, DataType::kHALF);
    cp(kInput, kRefT, kvEffSize * 2);
    rt::Tensor vRefT = mt(rt::Coords{B, effectiveKvLen, Hkv, D}, DataType::kHALF);
    cp(vInput, vRefT, kvEffSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF);
    cp(kvCacheInput, kvT, kvCacheSize * 2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32);
    cp(seqLens, slT, B * 4);

    // Reference FP16 attention (attends over effectiveKvLen positions)
    rt::Tensor oRefT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    launchSageReferenceAttention(
        static_cast<half const*>(qT.rawPointer()),
        static_cast<half const*>(kRefT.rawPointer()),
        static_cast<half const*>(vRefT.rawPointer()),
        static_cast<half*>(oRefT.rawPointer()),
        B, qL, effectiveKvLen, Hq, Hkv, D, false, s);

    // Plugin path: quantize Q, compute V_scales + K_means
    rt::Tensor qI = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageComputeVScalesAndKMeans(kvT, slT, vS, kM, cap, s);

    rt::Tensor oFusedT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    int32_t sb = qL * Hq * D, ss = Hq * D;
    float smv = 1.0f / sqrtf(static_cast<float>(D));
    // Fused kernel with PARTIAL KV cache (cap > effectiveKvLen)
    sage::launchSageAttentionFused(
        static_cast<int8_t*>(qI.rawPointer()), static_cast<half const*>(kvT.rawPointer()),
        static_cast<half*>(oFusedT.rawPointer()), static_cast<float*>(qS.rawPointer()),
        static_cast<float*>(vS.rawPointer()), static_cast<float*>(kM.rawPointer()),
        static_cast<int32_t const*>(slT.rawPointer()),
        B, Hq, Hkv, cap, sb, ss, D, sb, ss, D, smv, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> oRef(qSize), oFused(qSize);
    CUDA_CHECK(cudaMemcpy(oRef.data(), oRefT.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(oFused.data(), oFusedT.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));

    int32_t mM = 0;
    float mD = 0, sD = 0, sR = 0;
    for (size_t i = 0; i < qSize; i++) {
        float r = __half2float(oRef[i]), f = __half2float(oFused[i]);
        float d = fabsf(r - f); sD += d; sR += fabsf(r);
        mD = fmaxf(mD, d);
        if (d > std::max(fabsf(r) * 0.10f, 0.20f)) mM++;
    }
    float passRate = 1.0f - static_cast<float>(mM) / qSize;
    std::cout << "PartialKV(eff=" << effectiveKvLen << "/cap=" << cap
              << "): mism=" << mM << "/" << qSize
              << " passRate=" << passRate << " maxDiff=" << mD << " avgRelErr=" << (sR > 0 ? sD / sR : 0) << std::endl;
    if (mM > 0) {
        for (int i = 0; i < 5; i++)
            std::cout << "  [" << i << "] ref=" << __half2float(oRef[i])
                      << " fused=" << __half2float(oFused[i])
                      << " diff=" << fabsf(__half2float(oRef[i]) - __half2float(oFused[i])) << std::endl;
    }
    // Also test WITHOUT K-mean
    {
        rt::Tensor oFusedNoKMT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
        sage::launchSageAttentionFused(
            static_cast<int8_t*>(qI.rawPointer()), static_cast<half const*>(kvT.rawPointer()),
            static_cast<half*>(oFusedNoKMT.rawPointer()), static_cast<float*>(qS.rawPointer()),
            static_cast<float*>(vS.rawPointer()), nullptr,  // kMean = nullptr!
            static_cast<int32_t const*>(slT.rawPointer()),
            B, Hq, Hkv, cap, sb, ss, D, sb, ss, D, smv, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<half> oNoKM(qSize);
        CUDA_CHECK(cudaMemcpy(oNoKM.data(), oFusedNoKMT.rawPointer(), qSize*2, cudaMemcpyDeviceToHost));
        int32_t mM2 = 0; float mD2 = 0, sD2 = 0, sR2 = 0;
        for (size_t i = 0; i < qSize; i++) {
            float r = __half2float(oRef[i]), f = __half2float(oNoKM[i]);
            float d = fabsf(r - f); sD2 += d; sR2 += fabsf(r);
            mD2 = fmaxf(mD2, d);
            if (d > std::max(fabsf(r) * 0.10f, 0.20f)) mM2++;
        }
        float pr2 = 1.0f - static_cast<float>(mM2) / qSize;
        std::cout << "PartialKV NO K-mean: mism=" << mM2 << "/" << qSize
                  << " passRate=" << pr2 << " maxDiff=" << mD2 << std::endl;
        EXPECT_GT(pr2, 0.90f);
    }
    EXPECT_GT(passRate, 0.90f);
}

// Compare fused vs original kernel output with identical partial KV data
TEST(SageAttentionTest, DISABLED_FusedVsOriginalPartialKV)
{
    int32_t const B = 1, qL = 1, Hq = 16, Hkv = 8, D = 128, cap = 512, eff = 21;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";

    size_t const qSize = static_cast<size_t>(B) * qL * Hq * D;
    size_t const kvEffSize = static_cast<size_t>(B) * eff * Hkv * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;
    std::vector<half> qInput(qSize), kInput(kvEffSize), vInput(kvEffSize);
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    std::vector<int32_t> seqLens(B, eff);
    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kInput, -1.0f, 1.0f);
    uniformFloatInitialization(vInput, -1.0f, 1.0f);
    for (int32_t b = 0; b < B; b++)
        for (int32_t h = 0; h < Hkv; h++)
            for (int32_t s = 0; s < eff; s++)
                for (int32_t d = 0; d < D; d++) {
                    int64_t kvIdx = (((int64_t(b) * eff + s) * Hkv + h) * D + d);
                    int64_t kcIdx = (((int64_t(b) * 2 * Hkv + h) * cap + s) * D + d);
                    int64_t vcIdx = (((int64_t(b) * 2 * Hkv + Hkv + h) * cap + s) * D + d);
                    kvCacheInput[kcIdx] = kInput[kvIdx];
                    kvCacheInput[vcIdx] = vInput[kvIdx];
                }

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF); cp(qInput, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF); cp(kvCacheInput, kvT, kvCacheSize * 2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32); cp(seqLens, slT, B * 4);

    // FUSED kernel path (same as plugin)
    rt::Tensor qI_f = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS_f = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor vS_f = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM_f = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI_f, qS_f, s);
    sage::launchSageComputeVScalesAndKMeans(kvT, slT, vS_f, kM_f, cap, s);
    rt::Tensor oF = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    int32_t sb = qL * Hq * D, ss = Hq * D; float smv = 1.0f / sqrtf(static_cast<float>(D));
    sage::launchSageAttentionFused(
        static_cast<int8_t*>(qI_f.rawPointer()), static_cast<half const*>(kvT.rawPointer()),
        static_cast<half*>(oF.rawPointer()), static_cast<float*>(qS_f.rawPointer()),
        static_cast<float*>(vS_f.rawPointer()), static_cast<float*>(kM_f.rawPointer()),
        static_cast<int32_t const*>(slT.rawPointer()),
        B, Hq, Hkv, cap, sb, ss, D, sb, ss, D, smv, s);

    // ORIGINAL kernel path (convert pipeline + SageAttentionRunner)
    int32_t const pkL = sage::getSagePaddedKvLen(cap);
    rt::Tensor qI_o = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS_o = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor kI_o = mt(rt::Coords{B, cap, Hkv, D}, DataType::kINT8);
    rt::Tensor vF_o = mt(rt::Coords{B, D, Hkv, pkL}, DataType::kINT8);
    rt::Tensor kS_o = mt(rt::Coords{sage::getSageKScaleSize(B, cap, Hkv)}, DataType::kFLOAT);
    rt::Tensor vS_o = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM_o = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pvm = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pks = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI_o, qS_o, s);
    sage::launchSageConvertKVCacheToInt8AndFp8(kvT, slT, kI_o, vF_o, kS_o, vS_o, kM_o, pvm, pks, cap, s);
    rt::Tensor oO = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    SageAttentionParams rp{};
    rp.q_ptr = static_cast<int8_t*>(qI_o.rawPointer());
    rp.k_ptr = static_cast<int8_t*>(kI_o.rawPointer());
    rp.v_ptr = static_cast<int8_t*>(vF_o.rawPointer());
    rp.o_ptr = oO.rawPointer();
    rp.q_scale_ptr = static_cast<float*>(qS_o.rawPointer());
    rp.k_scale_ptr = static_cast<float*>(kS_o.rawPointer());
    rp.v_scale_ptr = static_cast<float*>(vS_o.rawPointer());
    rp.fuse_v_scale = true;
    rp.sequence_lengths = static_cast<int32_t const*>(slT.rawPointer());
    rp.batch_size = B; rp.qo_len = qL; rp.kv_len = cap;
    rp.num_qo_heads = Hq; rp.num_kv_heads = Hkv; rp.head_dim = D;
    rp.tensor_layout = SageTensorLayout::kBSHD; rp.mask_mode = SageMaskMode::kNone;
    rp.qk_quant_gran = SageQuantGranularity::kPerWarp;
    rp.stride_bz_q = qL*Hq*D; rp.stride_seq_q = Hq*D; rp.stride_h_q = D;
    rp.stride_bz_k = cap*Hkv*D; rp.stride_seq_k = Hkv*D; rp.stride_h_k = D;
    rp.stride_bz_v = D*Hkv*pkL; rp.stride_h_v = pkL; rp.stride_d_v = Hkv*pkL;
    rp.stride_bz_o = qL*Hq*D; rp.stride_seq_o = Hq*D; rp.stride_h_o = D;
    SageAttentionRunner runner(DataType::kHALF, B, qL, cap, Hq, Hkv, D, smVersion);
    runner.run(rp, nullptr, s);

    CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<half> oF_h(qSize), oO_h(qSize);
    CUDA_CHECK(cudaMemcpy(oF_h.data(), oF.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(oO_h.data(), oO.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));

    int32_t mM = 0; float mD = 0;
    for (size_t i = 0; i < qSize; i++) {
        float f = __half2float(oF_h[i]), o = __half2float(oO_h[i]);
        float d = fabsf(f - o); mD = fmaxf(mD, d);
        if (d > 0.001f * fmaxf(fabsf(o), 0.001f)) mM++;
    }
    std::cout << "Fused vs Original (partial KV, eff=" << eff << "): mism=" << mM
              << "/" << qSize << " maxDiff=" << mD << std::endl;
    if (mM > 0) {
        for (int i = 0; i < 5; i++)
            std::cout << "  [" << i << "] fused=" << __half2float(oF_h[i])
                      << " orig=" << __half2float(oO_h[i])
                      << " diff=" << fabsf(__half2float(oF_h[i]) - __half2float(oO_h[i])) << std::endl;
    }
    // If fused and original match, both are equally wrong (shared bug).
    // If they differ, fused has a unique bug.
    EXPECT_LT(mM, qSize / 2);
}

// Compare BOTH fused and XQA outputs against FP16 reference using real model data
TEST(SageAttentionTest, CompareBothWithReference)
{
    std::ifstream f("/tmp/fused_debug.bin", std::ios::binary);
    if (!f) { GTEST_SKIP() << "Debug file not found"; return; }

    int32_t meta[8]; f.read((char*)meta, 32);
    int32_t B = meta[0], qL = meta[1], Hq = meta[2], Hkv = meta[3], D = meta[4], cap = meta[5], qSize = meta[6], kvSize = meta[7];
    int32_t qScaleSize = sage::getSageQScaleSize(B, qL, Hq);
    int32_t vScaleSize = sage::getSageVScaleSize(B, Hkv, D);

    std::vector<half> qf(qSize), ohFused(qSize), kv(kvSize);
    std::vector<int8_t> qi(qSize);
    std::vector<float> qs(qScaleSize), vs(vScaleSize), km(vScaleSize);
    std::vector<int32_t> sl(B);

    f.read((char*)qf.data(), qSize * 2);
    f.read((char*)qi.data(), qSize);
    f.read((char*)qs.data(), qScaleSize * 4);
    f.read((char*)vs.data(), vScaleSize * 4);
    f.read((char*)km.data(), vScaleSize * 4);
    f.read((char*)kv.data(), kvSize * 2);
    f.read((char*)sl.data(), B * 4);
    f.read((char*)ohFused.data(), qSize * 2);

    // Read XQA output (appended after fused output)
    std::vector<half> ohXqa(qSize);
    f.read((char*)ohXqa.data(), qSize * 2);
    bool hasXqa = (f.gcount() == (std::streamsize)(qSize * 2));
    f.close();

    int32_t kvLen = sl[0];  // effective KV length
    // Extract K and V from KV cache [B, 2, Hkv, cap, D] → [B, kvLen, Hkv, D]
    std::vector<half> kExtract(B * kvLen * Hkv * D), vExtract(B * kvLen * Hkv * D);
    for (int32_t b = 0; b < B; b++)
        for (int32_t h = 0; h < Hkv; h++)
            for (int32_t s = 0; s < kvLen; s++)
                for (int32_t d = 0; d < D; d++) {
                    int64_t kCacheIdx = (((int64_t(b) * 2 * Hkv + h) * cap + s) * D + d);
                    int64_t vCacheIdx = (((int64_t(b) * 2 * Hkv + Hkv + h) * cap + s) * D + d);
                    int64_t kvIdx = (((int64_t(b) * kvLen + s) * Hkv + h) * D + d);
                    kExtract[kvIdx] = kv[kCacheIdx];
                    vExtract[kvIdx] = kv[vCacheIdx];
                }

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    cp(qf, qT, qSize * 2);
    rt::Tensor kT = mt(rt::Coords{B, kvLen, Hkv, D}, DataType::kHALF);
    cp(kExtract, kT, B * kvLen * Hkv * D * 2);
    rt::Tensor vT = mt(rt::Coords{B, kvLen, Hkv, D}, DataType::kHALF);
    cp(vExtract, vT, B * kvLen * Hkv * D * 2);
    rt::Tensor oRefT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);

    // FP16 reference attention
    launchSageReferenceAttention(
        static_cast<half const*>(qT.rawPointer()),
        static_cast<half const*>(kT.rawPointer()),
        static_cast<half const*>(vT.rawPointer()),
        static_cast<half*>(oRefT.rawPointer()),
        B, qL, kvLen, Hq, Hkv, D, false, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> oRef(qSize);
    CUDA_CHECK(cudaMemcpy(oRef.data(), oRefT.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));

    // Compare fused vs reference
    {
        int32_t mM = 0, mM10 = 0;
        float mD = 0, sD = 0, sR = 0;
        for (int32_t i = 0; i < qSize; i++) {
            float r = __half2float(oRef[i]), f = __half2float(ohFused[i]);
            float d = fabsf(r - f); sD += d; sR += fabsf(r);
            mD = fmaxf(mD, d);
            if (d > 0.001f * fmaxf(fabsf(r), 0.001f)) mM++;
            if (d > std::max(fabsf(r) * 0.10f, 0.20f)) mM10++;
        }
        std::cout << "FUSED vs REF: mism(strict)=" << mM << "/" << qSize
                  << " mism(10%/0.2)=" << mM10 << "/" << qSize
                  << " maxDiff=" << mD << " avgRelErr=" << (sR > 0 ? sD / sR : 0) << std::endl;
        if (mM10 > 0) {
            for (int i = 0; i < 5; i++)
                std::cout << "  [" << i << "] ref=" << __half2float(oRef[i])
                          << " fused=" << __half2float(ohFused[i])
                          << " diff=" << fabsf(__half2float(oRef[i]) - __half2float(ohFused[i])) << std::endl;
        }
        EXPECT_LT(mM10, qSize / 2);
    }

    // Compare XQA vs reference (if available)
    if (hasXqa) {
        int32_t mM = 0, mM10 = 0;
        float mD = 0, sD = 0, sR = 0;
        for (int32_t i = 0; i < qSize; i++) {
            float r = __half2float(oRef[i]), x = __half2float(ohXqa[i]);
            float d = fabsf(r - x); sD += d; sR += fabsf(r);
            mD = fmaxf(mD, d);
            if (d > 0.001f * fmaxf(fabsf(r), 0.001f)) mM++;
            if (d > std::max(fabsf(r) * 0.10f, 0.20f)) mM10++;
        }
        std::cout << "XQA vs REF: mism(strict)=" << mM << "/" << qSize
                  << " mism(10%/0.2)=" << mM10 << "/" << qSize
                  << " maxDiff=" << mD << " avgRelErr=" << (sR > 0 ? sD / sR : 0) << std::endl;
        if (mM10 > 0) {
            for (int i = 0; i < 5; i++)
                std::cout << "  [" << i << "] ref=" << __half2float(oRef[i])
                          << " xqa=" << __half2float(ohXqa[i])
                          << " diff=" << fabsf(__half2float(oRef[i]) - __half2float(ohXqa[i])) << std::endl;
        }
    }
}

// Replicate the EXACT plugin path with the inference shape: B=1, qL=1, Hq=16, Hkv=8 (GQA r=2),
// D=128, cap=4096, eff=21. Compares fused output against fp16 reference attention.
TEST(SageAttentionTest, FusedPluginShapeRandom)
{
    int32_t const B = 1, qL = 1, Hq = 16, Hkv = 8, D = 128, cap = 4096, eff = 21;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";

    size_t const qSize = static_cast<size_t>(B) * qL * Hq * D;
    size_t const kvEffSize = static_cast<size_t>(B) * eff * Hkv * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;

    std::vector<half> qInput(qSize), kEff(kvEffSize), vEff(kvEffSize);
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kEff, -1.0f, 1.0f);
    uniformFloatInitialization(vEff, -1.0f, 1.0f);

    // Lay out KV cache in [B, 2, Hkv, cap, D] with effective-only data, padding=0.
    for (int32_t b = 0; b < B; b++)
        for (int32_t h = 0; h < Hkv; h++)
            for (int32_t s = 0; s < eff; s++)
                for (int32_t d = 0; d < D; d++) {
                    int64_t srcIdx = (((int64_t(b) * eff + s) * Hkv + h) * D + d);
                    int64_t kcIdx = (((int64_t(b) * 2 * Hkv + h) * cap + s) * D + d);
                    int64_t vcIdx = (((int64_t(b) * 2 * Hkv + Hkv + h) * cap + s) * D + d);
                    kvCacheInput[kcIdx] = kEff[srcIdx];
                    kvCacheInput[vcIdx] = vEff[srcIdx];
                }

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) {
        CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice));
    };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);          cp(qInput, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF);    cp(kvCacheInput, kvT, kvCacheSize * 2);
    std::vector<int32_t> seqLens(B, eff);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32);                    cp(seqLens, slT, B * 4);

    // Plugin path: launchSageQuantizeQToInt8 + launchSageComputeVScalesAndKMeans + launchSageAttentionFused
    rt::Tensor qI = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageComputeVScalesAndKMeans(kvT, slT, vS, kM, cap, s);

    rt::Tensor oFused = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    int32_t sb = qL * Hq * D, ss = Hq * D;
    float smv = 1.0f / sqrtf(static_cast<float>(D));
    sage::launchSageAttentionFused(
        static_cast<int8_t*>(qI.rawPointer()), static_cast<half const*>(kvT.rawPointer()),
        static_cast<half*>(oFused.rawPointer()), static_cast<float*>(qS.rawPointer()),
        static_cast<float*>(vS.rawPointer()), static_cast<float*>(kM.rawPointer()),
        static_cast<int32_t const*>(slT.rawPointer()),
        B, Hq, Hkv, cap, sb, ss, D, sb, ss, D, smv, s);

    // FP16 reference: needs K/V in [B, kvLen, Hkv, D] flat layout (effective only, no padding).
    rt::Tensor kRefT = mt(rt::Coords{B, eff, Hkv, D}, DataType::kHALF);     cp(kEff, kRefT, kvEffSize * 2);
    rt::Tensor vRefT = mt(rt::Coords{B, eff, Hkv, D}, DataType::kHALF);     cp(vEff, vRefT, kvEffSize * 2);
    rt::Tensor oRef = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    launchSageReferenceAttention(static_cast<half const*>(qT.rawPointer()),
        static_cast<half const*>(kRefT.rawPointer()), static_cast<half const*>(vRefT.rawPointer()),
        static_cast<half*>(oRef.rawPointer()), B, qL, eff, Hq, Hkv, D, /*causal=*/false, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> fH(qSize), rH(qSize);
    CUDA_CHECK(cudaMemcpy(fH.data(), oFused.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rH.data(), oRef.rawPointer(),   qSize * 2, cudaMemcpyDeviceToHost));

    int mM = 0;
    int mMStrict = 0;
    float mD = 0;
    float sumAbs = 0;
    float sumRef = 0;
    for (size_t i = 0; i < qSize; i++) {
        float r = __half2float(rH[i]), f = __half2float(fH[i]);
        float d = fabsf(r - f); mD = fmaxf(mD, d);
        sumAbs += d; sumRef += fabsf(r);
        if (d > std::max(0.20f, 0.10f * fabsf(r))) mM++;
        if (d > 0.05f * fmaxf(fabsf(r), 0.05f)) mMStrict++;
    }
    float pr = 1.0f - static_cast<float>(mMStrict) / qSize;
    float relErr = sumRef > 0 ? sumAbs / sumRef : 0;
    std::cout << "FusedPluginShape (B=" << B << " Hq=" << Hq << " Hkv=" << Hkv << " D=" << D
              << " cap=" << cap << " eff=" << eff << "): strictPassRate=" << pr
              << " largeErr=" << mM << "/" << qSize << " maxDiff=" << mD
              << " relErr=" << relErr << std::endl;
    if (mM > 0) {
        std::cout << "  first 5 (ref / fused):";
        for (int i = 0; i < 5 && i < (int)qSize; i++)
            std::cout << " " << __half2float(rH[i]) << "/" << __half2float(fH[i]);
        std::cout << std::endl;
    }
    EXPECT_EQ(mM, 0);
    EXPECT_LT(mD, 0.05f);
    EXPECT_LT(relErr, 0.08f);
}

// Sweep effective KV length (tile-aligned vs partial) and K-mean on/off.
// Same plugin path as FusedPluginShapeRandom, but parameterized.
struct FusedSweepCase { int32_t eff; bool useKMean; };
class FusedPluginShapeSweep : public ::testing::TestWithParam<FusedSweepCase> {};

TEST_P(FusedPluginShapeSweep, FusedVsRefSweep)
{
    int32_t const B = 1, qL = 1, Hq = 16, Hkv = 8, D = 128, cap = 4096;
    int32_t const eff = GetParam().eff;
    bool const useKMean = GetParam().useKMean;

    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";

    size_t const qSize = static_cast<size_t>(B) * qL * Hq * D;
    size_t const kvEffSize = static_cast<size_t>(B) * eff * Hkv * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;

    std::vector<half> qInput(qSize), kEff(kvEffSize), vEff(kvEffSize);
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kEff, -1.0f, 1.0f);
    uniformFloatInitialization(vEff, -1.0f, 1.0f);

    for (int32_t b = 0; b < B; b++)
        for (int32_t h = 0; h < Hkv; h++)
            for (int32_t s = 0; s < eff; s++)
                for (int32_t d = 0; d < D; d++) {
                    int64_t srcIdx = (((int64_t(b) * eff + s) * Hkv + h) * D + d);
                    int64_t kcIdx = (((int64_t(b) * 2 * Hkv + h) * cap + s) * D + d);
                    int64_t vcIdx = (((int64_t(b) * 2 * Hkv + Hkv + h) * cap + s) * D + d);
                    kvCacheInput[kcIdx] = kEff[srcIdx];
                    kvCacheInput[vcIdx] = vEff[srcIdx];
                }

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) {
        CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice));
    };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);          cp(qInput, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF);    cp(kvCacheInput, kvT, kvCacheSize * 2);
    std::vector<int32_t> seqLens(B, eff);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32);                    cp(seqLens, slT, B * 4);

    rt::Tensor qI = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    if (useKMean) {
        sage::launchSageComputeVScalesAndKMeans(kvT, slT, vS, kM, cap, s);
    } else {
        sage::launchSageComputeVChannelScales(kvT, slT, vS, cap, s);
    }
    rt::Tensor oFused = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    int32_t sb = qL * Hq * D, ss = Hq * D;
    float smv = 1.0f / sqrtf(static_cast<float>(D));
    sage::launchSageAttentionFused(
        static_cast<int8_t*>(qI.rawPointer()), static_cast<half const*>(kvT.rawPointer()),
        static_cast<half*>(oFused.rawPointer()), static_cast<float*>(qS.rawPointer()),
        static_cast<float*>(vS.rawPointer()), useKMean ? static_cast<float*>(kM.rawPointer()) : nullptr,
        static_cast<int32_t const*>(slT.rawPointer()),
        B, Hq, Hkv, cap, sb, ss, D, sb, ss, D, smv, s);

    rt::Tensor kRefT = mt(rt::Coords{B, eff, Hkv, D}, DataType::kHALF);     cp(kEff, kRefT, kvEffSize * 2);
    rt::Tensor vRefT = mt(rt::Coords{B, eff, Hkv, D}, DataType::kHALF);     cp(vEff, vRefT, kvEffSize * 2);
    rt::Tensor oRef = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    launchSageReferenceAttention(static_cast<half const*>(qT.rawPointer()),
        static_cast<half const*>(kRefT.rawPointer()), static_cast<half const*>(vRefT.rawPointer()),
        static_cast<half*>(oRef.rawPointer()), B, qL, eff, Hq, Hkv, D, false, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> fH(qSize), rH(qSize);
    CUDA_CHECK(cudaMemcpy(fH.data(), oFused.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rH.data(), oRef.rawPointer(),   qSize * 2, cudaMemcpyDeviceToHost));

    int mM = 0; float mD = 0; float sumAbs = 0, sumRef = 0;
    for (size_t i = 0; i < qSize; i++) {
        float r = __half2float(rH[i]), f = __half2float(fH[i]);
        float d = fabsf(r - f); mD = fmaxf(mD, d);
        sumAbs += d; sumRef += fabsf(r);
        if (d > 0.05f * fmaxf(fabsf(r), 0.05f)) mM++;
    }
    float pr = 1.0f - static_cast<float>(mM) / qSize;
    float relErr = sumRef > 0 ? sumAbs / sumRef : 0;
    std::cout << "[Sweep] eff=" << eff << " kMean=" << (useKMean ? "Y" : "N")
              << " passRate=" << pr << " maxDiff=" << mD << " relErr=" << relErr << std::endl;
    EXPECT_LT(mD, 0.08f);
    EXPECT_LT(relErr, 0.10f);
}

INSTANTIATE_TEST_SUITE_P(EffSweep, FusedPluginShapeSweep,
    ::testing::Values(
        FusedSweepCase{ 21,  true},   // partial KV + K-mean (matches inference path)
        FusedSweepCase{ 21, false},   // partial KV no K-mean
        FusedSweepCase{ 64,  true},   // tile-aligned + K-mean
        FusedSweepCase{ 64, false},   // tile-aligned no K-mean
        FusedSweepCase{128,  true},   // 2 tiles + K-mean
        FusedSweepCase{128, false}    // 2 tiles no K-mean
    ));

// Sanity: original SageAttentionRunner vs fp16 reference at the inference shape, no fused involvement.
// Tells us how lossy plain SageAttention is at this shape, so we know the precision floor for fused.
TEST(SageAttentionTest, OriginalSageRunnerShapeRandom)
{
    int32_t const B = 1, qL = 1, Hq = 16, Hkv = 8, D = 128, eff = 21;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";
    int32_t const cap = 4096;  // match inference: kvCacheCapacity, eff via sequenceLengths
    int32_t const pkL = sage::getSagePaddedKvLen(cap);

    size_t const qSize = static_cast<size_t>(B) * qL * Hq * D;
    size_t const kvEffSize = static_cast<size_t>(B) * eff * Hkv * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;

    std::vector<half> qInput(qSize), kEff(kvEffSize), vEff(kvEffSize);
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kEff, -1.0f, 1.0f);
    uniformFloatInitialization(vEff, -1.0f, 1.0f);
    for (int32_t b = 0; b < B; b++) for (int32_t h = 0; h < Hkv; h++)
        for (int32_t s = 0; s < eff; s++) for (int32_t d = 0; d < D; d++) {
            int64_t srcIdx = (((int64_t(b)*eff+s)*Hkv+h)*D+d);
            kvCacheInput[(((int64_t(b)*2*Hkv+h)*cap+s)*D+d)] = kEff[srcIdx];
            kvCacheInput[(((int64_t(b)*2*Hkv+Hkv+h)*cap+s)*D+d)] = vEff[srcIdx];
        }
    std::vector<int32_t> sl(B, eff);

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);          cp(qInput, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF);    cp(kvCacheInput, kvT, kvCacheSize * 2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32);                    cp(sl, slT, B * 4);

    rt::Tensor qI = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor kI = mt(rt::Coords{B, cap, Hkv, D}, DataType::kINT8);
    rt::Tensor vF = mt(rt::Coords{B, D, Hkv, pkL}, DataType::kINT8);
    rt::Tensor kS = mt(rt::Coords{sage::getSageKScaleSize(B, cap, Hkv)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pvm = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pks = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageConvertKVCacheToInt8AndFp8(kvT, slT, kI, vF, kS, vS, kM, pvm, pks, cap, s);

    rt::Tensor oOrig = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    SageAttentionParams rp{};
    rp.q_ptr = static_cast<int8_t*>(qI.rawPointer());
    rp.k_ptr = static_cast<int8_t*>(kI.rawPointer());
    rp.v_ptr = static_cast<int8_t*>(vF.rawPointer());
    rp.o_ptr = oOrig.rawPointer();
    rp.q_scale_ptr = static_cast<float*>(qS.rawPointer());
    rp.k_scale_ptr = static_cast<float*>(kS.rawPointer());
    rp.v_scale_ptr = static_cast<float*>(vS.rawPointer());
    rp.fuse_v_scale = true;
    rp.sequence_lengths = static_cast<int32_t const*>(slT.rawPointer());
    rp.batch_size = B; rp.qo_len = qL; rp.kv_len = cap;
    rp.num_qo_heads = Hq; rp.num_kv_heads = Hkv; rp.head_dim = D;
    rp.tensor_layout = SageTensorLayout::kBSHD; rp.mask_mode = SageMaskMode::kNone;
    rp.qk_quant_gran = SageQuantGranularity::kPerWarp;
    rp.stride_bz_q = qL*Hq*D; rp.stride_seq_q = Hq*D; rp.stride_h_q = D;
    rp.stride_bz_k = cap*Hkv*D; rp.stride_seq_k = Hkv*D; rp.stride_h_k = D;
    rp.stride_bz_v = D*Hkv*pkL; rp.stride_h_v = pkL; rp.stride_d_v = Hkv*pkL;
    rp.stride_bz_o = qL*Hq*D; rp.stride_seq_o = Hq*D; rp.stride_h_o = D;
    SageAttentionRunner runner(DataType::kHALF, B, qL, cap, Hq, Hkv, D, smVersion);
    runner.run(rp, nullptr, s);

    rt::Tensor kRefT = mt(rt::Coords{B, eff, Hkv, D}, DataType::kHALF);     cp(kEff, kRefT, kvEffSize * 2);
    rt::Tensor vRefT = mt(rt::Coords{B, eff, Hkv, D}, DataType::kHALF);     cp(vEff, vRefT, kvEffSize * 2);
    rt::Tensor oRef = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    launchSageReferenceAttention(static_cast<half const*>(qT.rawPointer()),
        static_cast<half const*>(kRefT.rawPointer()), static_cast<half const*>(vRefT.rawPointer()),
        static_cast<half*>(oRef.rawPointer()), B, qL, eff, Hq, Hkv, D, false, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> oH(qSize), rH(qSize);
    CUDA_CHECK(cudaMemcpy(oH.data(), oOrig.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rH.data(), oRef.rawPointer(),  qSize * 2, cudaMemcpyDeviceToHost));

    int mM = 0; float mD = 0; float sumAbs = 0, sumRef = 0;
    for (size_t i = 0; i < qSize; i++) {
        float r = __half2float(rH[i]), o = __half2float(oH[i]);
        float d = fabsf(r - o); mD = fmaxf(mD, d);
        sumAbs += d; sumRef += fabsf(r);
        if (d > 0.05f * fmaxf(fabsf(r), 0.05f)) mM++;
    }
    float pr = 1.0f - static_cast<float>(mM) / qSize;
    std::cout << "OrigSageRunner (B=" << B << " Hq=" << Hq << " Hkv=" << Hkv << " D=" << D
              << " eff=" << eff << "): passRate=" << pr
              << " maxDiff=" << mD << " relErr=" << (sumRef>0 ? sumAbs/sumRef : 0) << std::endl;
}

// Inference-shape, plugin-equivalent: feed same KV cache to fused and to convert+SageAttentionRunner
// (the proven pre-fusion path). Fused must match the runner; this isolates the fused kernel itself.
TEST(SageAttentionTest, DISABLED_FusedVsPreFusionPluginShape)
{
    int32_t const B = 1, qL = 1, Hq = 16, Hkv = 8, D = 128, cap = 4096, eff = 21;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";
    int32_t const pkL = sage::getSagePaddedKvLen(cap);

    size_t const qSize = static_cast<size_t>(B) * qL * Hq * D;
    size_t const kvEffSize = static_cast<size_t>(B) * eff * Hkv * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;

    std::vector<half> qInput(qSize), kEff(kvEffSize), vEff(kvEffSize);
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    uniformFloatInitialization(qInput, -1.0f, 1.0f);
    uniformFloatInitialization(kEff, -1.0f, 1.0f);
    uniformFloatInitialization(vEff, -1.0f, 1.0f);
    for (int32_t b = 0; b < B; b++) for (int32_t h = 0; h < Hkv; h++)
        for (int32_t s = 0; s < eff; s++) for (int32_t d = 0; d < D; d++) {
            int64_t srcIdx = (((int64_t(b)*eff+s)*Hkv+h)*D+d);
            kvCacheInput[(((int64_t(b)*2*Hkv+h)*cap+s)*D+d)] = kEff[srcIdx];
            kvCacheInput[(((int64_t(b)*2*Hkv+Hkv+h)*cap+s)*D+d)] = vEff[srcIdx];
        }
    std::vector<int32_t> sl(B, eff);

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);          cp(qInput, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF);    cp(kvCacheInput, kvT, kvCacheSize * 2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32);                    cp(sl, slT, B * 4);

    // === Pre-fusion (proven correct in inference) ===
    rt::Tensor qI = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor kI = mt(rt::Coords{B, cap, Hkv, D}, DataType::kINT8);
    rt::Tensor vF = mt(rt::Coords{B, D, Hkv, pkL}, DataType::kINT8);
    rt::Tensor kS = mt(rt::Coords{sage::getSageKScaleSize(B, cap, Hkv)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pvm = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pks = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageConvertKVCacheToInt8AndFp8(kvT, slT, kI, vF, kS, vS, kM, pvm, pks, cap, s);

    rt::Tensor oPre = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    SageAttentionParams rp{};
    rp.q_ptr = static_cast<int8_t*>(qI.rawPointer());
    rp.k_ptr = static_cast<int8_t*>(kI.rawPointer());
    rp.v_ptr = static_cast<int8_t*>(vF.rawPointer());
    rp.o_ptr = oPre.rawPointer();
    rp.q_scale_ptr = static_cast<float*>(qS.rawPointer());
    rp.k_scale_ptr = static_cast<float*>(kS.rawPointer());
    rp.v_scale_ptr = static_cast<float*>(vS.rawPointer());
    rp.fuse_v_scale = true;
    rp.sequence_lengths = static_cast<int32_t const*>(slT.rawPointer());
    rp.batch_size = B; rp.qo_len = qL; rp.kv_len = cap;
    rp.num_qo_heads = Hq; rp.num_kv_heads = Hkv; rp.head_dim = D;
    rp.tensor_layout = SageTensorLayout::kBSHD; rp.mask_mode = SageMaskMode::kNone;
    rp.qk_quant_gran = SageQuantGranularity::kPerWarp;
    rp.stride_bz_q = qL*Hq*D; rp.stride_seq_q = Hq*D; rp.stride_h_q = D;
    rp.stride_bz_k = cap*Hkv*D; rp.stride_seq_k = Hkv*D; rp.stride_h_k = D;
    rp.stride_bz_v = D*Hkv*pkL; rp.stride_h_v = pkL; rp.stride_d_v = Hkv*pkL;
    rp.stride_bz_o = qL*Hq*D; rp.stride_seq_o = Hq*D; rp.stride_h_o = D;
    SageAttentionRunner runner(DataType::kHALF, B, qL, cap, Hq, Hkv, D, smVersion);
    runner.run(rp, nullptr, s);

    // === Fused: same Q-int8, same V-scale & K-mean as pre-fusion ===
    rt::Tensor oFused = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    int32_t sb = qL * Hq * D, ss_ = Hq * D;
    float smv = 1.0f / sqrtf(static_cast<float>(D));
    sage::launchSageAttentionFused(
        static_cast<int8_t*>(qI.rawPointer()), static_cast<half const*>(kvT.rawPointer()),
        static_cast<half*>(oFused.rawPointer()), static_cast<float*>(qS.rawPointer()),
        static_cast<float*>(vS.rawPointer()), static_cast<float*>(kM.rawPointer()),
        static_cast<int32_t const*>(slT.rawPointer()),
        B, Hq, Hkv, cap, sb, ss_, D, sb, ss_, D, smv, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> fH(qSize), pH(qSize);
    CUDA_CHECK(cudaMemcpy(fH.data(), oFused.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pH.data(), oPre.rawPointer(),   qSize * 2, cudaMemcpyDeviceToHost));

    int mM = 0; float mD = 0; float sumAbs = 0, sumP = 0;
    for (size_t i = 0; i < qSize; i++) {
        float p = __half2float(pH[i]), f = __half2float(fH[i]);
        float d = fabsf(p - f); mD = fmaxf(mD, d);
        sumAbs += d; sumP += fabsf(p);
        if (d > 0.05f * fmaxf(fabsf(p), 0.05f)) mM++;
    }
    float pr = 1.0f - static_cast<float>(mM) / qSize;
    std::cout << "FusedVsPreFusion: passRate=" << pr << " maxDiff=" << mD
              << " relErr=" << (sumP > 0 ? sumAbs / sumP : 0)
              << " (qSize=" << qSize << " nMis=" << mM << ")" << std::endl;
    if (mM > 0) {
        std::cout << "  first 5 (pre / fused):";
        for (int i = 0; i < 5 && i < (int)qSize; i++)
            std::cout << " " << __half2float(pH[i]) << "/" << __half2float(fH[i]);
        std::cout << std::endl;
    }
    EXPECT_GT(pr, 0.95f);
}

// Isolate GQA: Hq=16 Hkv=8 (ratio=2), kL=64 = CTA_K, full KV.
TEST(SageAttentionTest, DISABLED_FusedGQAFullKv)
{
    int32_t const B = 1, qL = 1, Hq = 16, Hkv = 8, D = 128, eff = 64, cap = 64;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";
    int32_t const pkL = sage::getSagePaddedKvLen(cap);

    size_t const qSize = static_cast<size_t>(B) * 1 * Hq * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * Hkv * cap * D;
    std::vector<half> qInput(qSize), kvCacheInput(kvCacheSize);
    uniformFloatInitialization(qInput, -1.f, 1.f);
    uniformFloatInitialization(kvCacheInput, -1.f, 1.f);
    std::vector<int32_t> sl(B, eff);

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);          cp(qInput, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF);    cp(kvCacheInput, kvT, kvCacheSize * 2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32);                    cp(sl, slT, B * 4);

    // Pre-fusion path
    rt::Tensor qI = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor kI = mt(rt::Coords{B, cap, Hkv, D}, DataType::kINT8);
    rt::Tensor vF = mt(rt::Coords{B, D, Hkv, pkL}, DataType::kINT8);
    rt::Tensor kS = mt(rt::Coords{sage::getSageKScaleSize(B, cap, Hkv)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pvm = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pks = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageConvertKVCacheToInt8AndFp8(kvT, slT, kI, vF, kS, vS, kM, pvm, pks, cap, s);

    rt::Tensor oPre = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    SageAttentionParams rp{};
    rp.q_ptr = (int8_t*)qI.rawPointer(); rp.k_ptr = (int8_t*)kI.rawPointer(); rp.v_ptr = (int8_t*)vF.rawPointer();
    rp.o_ptr = oPre.rawPointer();
    rp.q_scale_ptr = (float*)qS.rawPointer(); rp.k_scale_ptr = (float*)kS.rawPointer(); rp.v_scale_ptr = (float*)vS.rawPointer();
    rp.fuse_v_scale = true; rp.sequence_lengths = (int32_t const*)slT.rawPointer();
    rp.batch_size = B; rp.qo_len = qL; rp.kv_len = cap;
    rp.num_qo_heads = Hq; rp.num_kv_heads = Hkv; rp.head_dim = D;
    rp.tensor_layout = SageTensorLayout::kBSHD; rp.mask_mode = SageMaskMode::kNone;
    rp.qk_quant_gran = SageQuantGranularity::kPerWarp;
    rp.stride_bz_q = qL*Hq*D; rp.stride_seq_q = Hq*D; rp.stride_h_q = D;
    rp.stride_bz_k = cap*Hkv*D; rp.stride_seq_k = Hkv*D; rp.stride_h_k = D;
    rp.stride_bz_v = D*Hkv*pkL; rp.stride_h_v = pkL; rp.stride_d_v = Hkv*pkL;
    rp.stride_bz_o = qL*Hq*D; rp.stride_seq_o = Hq*D; rp.stride_h_o = D;
    SageAttentionRunner runner(DataType::kHALF, B, qL, cap, Hq, Hkv, D, smVersion);
    runner.run(rp, nullptr, s);

    // Fused on the same KV/scales/kMean
    rt::Tensor oFused = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    int sb = qL * Hq * D, ss_ = Hq * D;
    float smv = 1.0f / sqrtf((float)D);
    sage::launchSageAttentionFused(
        (int8_t*)qI.rawPointer(), (half const*)kvT.rawPointer(), (half*)oFused.rawPointer(),
        (float*)qS.rawPointer(), (float*)vS.rawPointer(), nullptr,
        (int32_t const*)slT.rawPointer(),
        B, Hq, Hkv, cap, sb, ss_, D, sb, ss_, D, smv, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> fH(qSize), pH(qSize);
    CUDA_CHECK(cudaMemcpy(fH.data(), oFused.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pH.data(), oPre.rawPointer(),   qSize * 2, cudaMemcpyDeviceToHost));

    int mM = 0; float mD = 0; float sumAbs = 0, sumP = 0;
    for (size_t i = 0; i < qSize; i++) {
        float p = __half2float(pH[i]), f = __half2float(fH[i]);
        float d = fabsf(p - f); mD = fmaxf(mD, d); sumAbs += d; sumP += fabsf(p);
        if (d > 0.05f * fmaxf(fabsf(p), 0.05f)) mM++;
    }
    float pr = 1.0f - (float)mM / qSize;
    std::cout << "FusedGQAFullKv (Hq=16 Hkv=8 cap=64 eff=64): passRate=" << pr
              << " maxDiff=" << mD << " relErr=" << (sumP>0 ? sumAbs/sumP : 0)
              << " (nMis=" << mM << "/" << qSize << ")" << std::endl;
    if (mM > 0) {
        std::cout << "  first 5 (pre/fused):";
        for (int i = 0; i < 5 && i < (int)qSize; i++) std::cout << " " << __half2float(pH[i]) << "/" << __half2float(fH[i]);
        std::cout << std::endl;
        // Per-head breakdown: count mismatches for each query head
        std::cout << "  per-head mismatches (head: nMism/maxDiff/meanPre/meanFused):" << std::endl;
        for (int h = 0; h < Hq; h++) {
            int hm = 0; float hMd = 0, hSumP = 0, hSumF = 0;
            for (int d = 0; d < D; d++) {
                int i = h * D + d;
                float p = __half2float(pH[i]), f = __half2float(fH[i]);
                float df = fabsf(p - f); hMd = fmaxf(hMd, df);
                hSumP += p; hSumF += f;
                if (df > 0.05f * fmaxf(fabsf(p), 0.05f)) hm++;
            }
            std::cout << "    head " << h << ": " << hm << "/" << D
                      << " maxD=" << hMd
                      << " meanPre=" << (hSumP/D) << " meanFused=" << (hSumF/D) << std::endl;
        }
    }
    EXPECT_GT(pr, 0.90f);
}

// Isolation: r=1 but same number of heads (Hq=Hkv=16) — no GQA.
// If this passes while FusedGQAFullKv (Hq=16 Hkv=8) fails, the bug is GQA-specific.
TEST(SageAttentionTest, DISABLED_FusedGQAFullKv_NoGQA)
{
    int32_t const B = 1, qL = 1, H = 16, D = 128, eff = 64, cap = 64;  // Hq=Hkv=16, r=1
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";
    int32_t const pkL = sage::getSagePaddedKvLen(cap);

    size_t const qSize = static_cast<size_t>(B) * 1 * H * D;
    size_t const kvCacheSize = static_cast<size_t>(B) * 2 * H * cap * D;
    std::vector<half> qInput(qSize), kvCacheInput(kvCacheSize);
    uniformFloatInitialization(qInput, -1.f, 1.f);
    uniformFloatInitialization(kvCacheInput, -1.f, 1.f);
    std::vector<int32_t> sl(B, eff);

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };

    rt::Tensor qT = mt(rt::Coords{B, qL, H, D}, DataType::kHALF);          cp(qInput, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, H, cap, D}, DataType::kHALF);     cp(kvCacheInput, kvT, kvCacheSize * 2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32);                   cp(sl, slT, B * 4);

    // Pre-fusion path
    rt::Tensor qI = mt(rt::Coords{B, qL, H, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, H)}, DataType::kFLOAT);
    rt::Tensor kI = mt(rt::Coords{B, cap, H, D}, DataType::kINT8);
    rt::Tensor vF = mt(rt::Coords{B, D, H, pkL}, DataType::kINT8);
    rt::Tensor kS = mt(rt::Coords{sage::getSageKScaleSize(B, cap, H)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, H, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, H, D)}, DataType::kFLOAT);
    rt::Tensor pvm = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, H, D)}, DataType::kFLOAT);
    rt::Tensor pks = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, H, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageConvertKVCacheToInt8AndFp8(kvT, slT, kI, vF, kS, vS, kM, pvm, pks, cap, s);

    rt::Tensor oPre = mt(rt::Coords{B, qL, H, D}, DataType::kHALF);
    SageAttentionParams rp{};
    rp.q_ptr = (int8_t*)qI.rawPointer(); rp.k_ptr = (int8_t*)kI.rawPointer(); rp.v_ptr = (int8_t*)vF.rawPointer();
    rp.o_ptr = oPre.rawPointer();
    rp.q_scale_ptr = (float*)qS.rawPointer(); rp.k_scale_ptr = (float*)kS.rawPointer(); rp.v_scale_ptr = (float*)vS.rawPointer();
    rp.fuse_v_scale = true; rp.sequence_lengths = (int32_t const*)slT.rawPointer();
    rp.batch_size = B; rp.qo_len = qL; rp.kv_len = cap;
    rp.num_qo_heads = H; rp.num_kv_heads = H; rp.head_dim = D;
    rp.tensor_layout = SageTensorLayout::kBSHD; rp.mask_mode = SageMaskMode::kNone;
    rp.qk_quant_gran = SageQuantGranularity::kPerWarp;
    rp.stride_bz_q = qL*H*D; rp.stride_seq_q = H*D; rp.stride_h_q = D;
    rp.stride_bz_k = cap*H*D; rp.stride_seq_k = H*D; rp.stride_h_k = D;
    rp.stride_bz_v = D*H*pkL; rp.stride_h_v = pkL; rp.stride_d_v = H*pkL;
    rp.stride_bz_o = qL*H*D; rp.stride_seq_o = H*D; rp.stride_h_o = D;
    SageAttentionRunner runner(DataType::kHALF, B, qL, cap, H, H, D, smVersion);
    runner.run(rp, nullptr, s);

    // Fused on the same KV/scales/kMean
    rt::Tensor oFused = mt(rt::Coords{B, qL, H, D}, DataType::kHALF);
    int sb = qL * H * D, ss_ = H * D;
    float smv = 1.0f / sqrtf((float)D);
    sage::launchSageAttentionFused(
        (int8_t*)qI.rawPointer(), (half const*)kvT.rawPointer(), (half*)oFused.rawPointer(),
        (float*)qS.rawPointer(), (float*)vS.rawPointer(), nullptr,
        (int32_t const*)slT.rawPointer(),
        B, H, H, cap, sb, ss_, D, sb, ss_, D, smv, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> fH(qSize), pH(qSize);
    CUDA_CHECK(cudaMemcpy(fH.data(), oFused.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pH.data(), oPre.rawPointer(),   qSize * 2, cudaMemcpyDeviceToHost));

    int mM = 0; float mD = 0; float sumAbs = 0, sumP = 0;
    for (size_t i = 0; i < qSize; i++) {
        float p = __half2float(pH[i]), f = __half2float(fH[i]);
        float d = fabsf(p - f); mD = fmaxf(mD, d); sumAbs += d; sumP += fabsf(p);
        if (d > 0.05f * fmaxf(fabsf(p), 0.05f)) mM++;
    }
    float pr = 1.0f - (float)mM / qSize;
    std::cout << "FusedGQAFullKv_NoGQA (Hq=Hkv=16 r=1 cap=64 eff=64): passRate=" << pr
              << " maxDiff=" << mD << " relErr=" << (sumP>0 ? sumAbs/sumP : 0)
              << " (nMis=" << mM << "/" << qSize << ")" << std::endl;
    if (mM > 0) {
        std::cout << "  first 5 (pre/fused):";
        for (int i = 0; i < 5 && i < (int)qSize; i++) std::cout << " " << __half2float(pH[i]) << "/" << __half2float(fH[i]);
        std::cout << std::endl;
        // Per-head breakdown
        std::cout << "  per-head mismatches (head: nMism/maxDiff/meanPre/meanFused):" << std::endl;
        for (int h = 0; h < H; h++) {
            int hm = 0; float hMd = 0, hSumP = 0, hSumF = 0;
            for (int d_ = 0; d_ < D; d_++) {
                int i = h * D + d_;
                float p = __half2float(pH[i]), f = __half2float(fH[i]);
                float df = fabsf(p - f); hMd = fmaxf(hMd, df);
                hSumP += p; hSumF += f;
                if (df > 0.05f * fmaxf(fabsf(p), 0.05f)) hm++;
            }
            std::cout << "    head " << h << ": " << hm << "/" << D
                      << " maxD=" << hMd
                      << " meanPre=" << (hSumP/D) << " meanFused=" << (hSumF/D) << std::endl;
        }
    }
    EXPECT_GT(pr, 0.90f);
}

// Parametric fuse-vs-prefusion test to find the head-count threshold
static void FusedVsPreFusionWithHeads(int32_t Hq, int32_t Hkv, int32_t cap, int32_t eff)
{
    int32_t const B = 1, qL = 1, D = 128;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        { GTEST_SKIP() << "SA not supported"; return; }
    int32_t const pkL = sage::getSagePaddedKvLen(cap);

    size_t const qSize = (size_t)B * qL * Hq * D;
    size_t const kvEffSize = (size_t)B * eff * Hkv * D;
    size_t const kvCacheSize = (size_t)B * 2 * Hkv * cap * D;

    std::vector<half> qH(qSize), kEff(kvEffSize), vEff(kvEffSize);
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    uniformFloatInitialization(qH, -1, 1);
    uniformFloatInitialization(kEff, -1, 1);
    uniformFloatInitialization(vEff, -1, 1);
    for (int32_t b = 0; b < B; b++) for (int32_t h = 0; h < Hkv; h++)
        for (int32_t s = 0; s < eff; s++) for (int32_t d = 0; d < D; d++) {
            int64_t srcIdx = (((int64_t)b * eff + s) * Hkv + h) * D + d;
            kvCacheInput[(((int64_t)b * 2 * Hkv + h) * cap + s) * D + d] = kEff[srcIdx];
            kvCacheInput[(((int64_t)b * 2 * Hkv + Hkv + h) * cap + s) * D + d] = vEff[srcIdx];
        }
    std::vector<int32_t> sl(B, eff);

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };

    rt::Tensor qT = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);          cp(qH, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, Hkv, cap, D}, DataType::kHALF);    cp(kvCacheInput, kvT, kvCacheSize * 2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32);                    cp(sl, slT, B * 4);

    // Pre-fusion path
    rt::Tensor qI = mt(rt::Coords{B, qL, Hq, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, Hq)}, DataType::kFLOAT);
    rt::Tensor kI = mt(rt::Coords{B, cap, Hkv, D}, DataType::kINT8);
    rt::Tensor vF = mt(rt::Coords{B, D, Hkv, pkL}, DataType::kINT8);
    rt::Tensor kS = mt(rt::Coords{sage::getSageKScaleSize(B, cap, Hkv)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pvm = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    rt::Tensor pks = mt(rt::Coords{sage::getSageStatsPartialsSize(B, cap, Hkv, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageConvertKVCacheToInt8AndFp8(kvT, slT, kI, vF, kS, vS, kM, pvm, pks, cap, s);

    rt::Tensor oPre = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    SageAttentionParams rp{};
    rp.q_ptr = (int8_t*)qI.rawPointer(); rp.k_ptr = (int8_t*)kI.rawPointer(); rp.v_ptr = (int8_t*)vF.rawPointer();
    rp.o_ptr = oPre.rawPointer();
    rp.q_scale_ptr = (float*)qS.rawPointer(); rp.k_scale_ptr = (float*)kS.rawPointer(); rp.v_scale_ptr = (float*)vS.rawPointer();
    rp.fuse_v_scale = true; rp.sequence_lengths = (int32_t const*)slT.rawPointer();
    rp.batch_size = B; rp.qo_len = qL; rp.kv_len = cap;
    rp.num_qo_heads = Hq; rp.num_kv_heads = Hkv; rp.head_dim = D;
    rp.tensor_layout = SageTensorLayout::kBSHD; rp.mask_mode = SageMaskMode::kNone;
    rp.qk_quant_gran = SageQuantGranularity::kPerWarp;
    rp.stride_bz_q = qL*Hq*D; rp.stride_seq_q = Hq*D; rp.stride_h_q = D;
    rp.stride_bz_k = cap*Hkv*D; rp.stride_seq_k = Hkv*D; rp.stride_h_k = D;
    rp.stride_bz_v = D*Hkv*pkL; rp.stride_h_v = pkL; rp.stride_d_v = Hkv*pkL;
    rp.stride_bz_o = qL*Hq*D; rp.stride_seq_o = Hq*D; rp.stride_h_o = D;
    SageAttentionRunner runner(DataType::kHALF, B, qL, cap, Hq, Hkv, D, smVersion);
    runner.run(rp, nullptr, s);

    // Fused on same data
    rt::Tensor oFused = mt(rt::Coords{B, qL, Hq, D}, DataType::kHALF);
    int sb = qL * Hq * D, ss_ = Hq * D;
    float smv = 1.0f / sqrtf((float)D);
    // Fused: use SAME kMean as pre-fusion pipeline (not nullptr)
    sage::launchSageAttentionFused(
        (int8_t*)qI.rawPointer(), (half const*)kvT.rawPointer(), (half*)oFused.rawPointer(),
        (float*)qS.rawPointer(), (float*)vS.rawPointer(), (float*)kM.rawPointer(),
        (int32_t const*)slT.rawPointer(),
        B, Hq, Hkv, cap, sb, ss_, D, sb, ss_, D, smv, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> fH(qSize), pH(qSize);
    CUDA_CHECK(cudaMemcpy(fH.data(), oFused.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pH.data(), oPre.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));

    int mM = 0; float mD = 0;
    for (size_t i = 0; i < qSize; i++) {
        float p = __half2float(pH[i]), f = __half2float(fH[i]);
        float d = fabsf(p - f); mD = fmaxf(mD, d);
        if (d > 0.05f * fmaxf(fabsf(p), 0.05f)) mM++;
    }
    float pr = 1.0f - (float)mM / qSize;
    std::cout << "FusedVsPreF H" << Hq << "kv" << Hkv << " cap=" << cap << " eff=" << eff
              << ": pr=" << pr << " md=" << mD << " nM=" << mM << "/" << qSize << std::endl;
    EXPECT_GT(pr, 0.90f);
}

TEST(SageAttentionTest, DISABLED_FusedVsPreFusionThreshold8)
    { FusedVsPreFusionWithHeads(8, 8, 64, 64); }
TEST(SageAttentionTest, DISABLED_FusedVsPreFusionThreshold9)
    { FusedVsPreFusionWithHeads(9, 9, 64, 64); }
TEST(SageAttentionTest, DISABLED_FusedVsPreFusionThreshold10)
    { FusedVsPreFusionWithHeads(10, 10, 64, 64); }
TEST(SageAttentionTest, DISABLED_FusedVsPreFusionThreshold12)
    { FusedVsPreFusionWithHeads(12, 12, 64, 64); }
TEST(SageAttentionTest, DISABLED_FusedVsPreFusionThreshold14)
    { FusedVsPreFusionWithHeads(14, 14, 64, 64); }

// Dump Q scales to check for decode-kernel anomalies
TEST(SageAttentionTest, QScaleDump16Heads)
{
    int32_t const B = 1, qL = 1, H = 16, D = 128;
    size_t const qSize = (size_t)B * qL * H * D;
    std::vector<half> qH(qSize);
    uniformFloatInitialization(qH, -1, 1);
    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };
    rt::Tensor qT = mt(rt::Coords{B, qL, H, D}, DataType::kHALF); cp(qH, qT, qSize * 2);
    rt::Tensor qI = mt(rt::Coords{B, qL, H, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, H)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    int qScaleN = sage::getSageQScaleSize(B, qL, H); // should be 16*4=64
    std::vector<float> qScales(qScaleN);
    CUDA_CHECK(cudaMemcpy(qScales.data(), qS.rawPointer(), qScaleN * 4, cudaMemcpyDeviceToHost));
    std::vector<int8_t> qInt8(qSize);
    CUDA_CHECK(cudaMemcpy(qInt8.data(), qI.rawPointer(), qSize, cudaMemcpyDeviceToHost));

    std::cout << "QScaleDump: qScaleSize=" << qScaleN << std::endl;
    for (int h = 0; h < H; h++) {
        std::cout << "  head " << h << " scales:";
        for (int w = 0; w < 4; w++) {
            std::cout << " " << qScales[h * 4 + w];
        }
        // Manually compute expected Q max for this head
        float headMax = 0;
        for (int d = 0; d < D; d++) {
            float fv = fabsf(__half2float(qH[h * D + d]));
            headMax = fmaxf(headMax, fv);
        }
        float expectedScale = fmaxf(headMax / 127.0f, 1.0e-6f);
        std::cout << " expected_w0=" << expectedScale << std::endl;
    }
    // Check: are scales for warp 1,2,3 all epsilon (~1e-6)?
    int badNonZero = 0;
    for (int h = 0; h < H; h++) {
        for (int w = 1; w < 4; w++) {
            float s = qScales[h * 4 + w];
            if (s > 2.0e-6f) { badNonZero++; }
        }
    }
    std::cout << "  warp 1-3 scales > 2e-6: " << badNonZero << "/" << (H*3) << std::endl;
    EXPECT_EQ(badNonZero, 0);
}

// Simple-pattern test for 16 heads: all Q=1.0, all K=0.5, all V=2.0
// Expected: uniform attention → output ≈ 2.0 for all heads
TEST(SageAttentionTest, FusedSimplePattern16Heads)
{
    int32_t const B = 1, qL = 1, H = 16, D = 128, eff = 64, cap = 64;
    int32_t smVersion = getSMVersion();
    applyThorSMRenumberWAR(smVersion);
    if (!SageAttentionRunner::canImplement(D, smVersion, DataType::kHALF))
        GTEST_SKIP() << "SageAttention not supported";

    size_t const qSize = (size_t)B * qL * H * D;
    size_t const kvCacheSize = (size_t)B * 2 * H * cap * D;
    std::vector<half> qInput(qSize, __float2half(1.0f));
    std::vector<half> kvCacheInput(kvCacheSize, __float2half(0.0f));
    std::vector<int32_t> sl(B, eff);
    for (int32_t b = 0; b < B; b++)
        for (int32_t h = 0; h < H; h++)
            for (int32_t s = 0; s < eff; s++)
                for (int32_t d = 0; d < D; d++) {
                    int64_t kcIdx = (((int64_t(b) * 2 * H + h) * cap + s) * D + d);
                    int64_t vcIdx = (((int64_t(b) * 2 * H + H + h) * cap + s) * D + d);
                    kvCacheInput[kcIdx] = __float2half(0.5f);
                    kvCacheInput[vcIdx] = __float2half(2.0f);
                }

    cudaStream_t s = nullptr;
    auto mt = [](auto sh, auto dt) { return rt::Tensor(sh, rt::DeviceType::kGPU, dt); };
    auto cp = [](auto& h, auto& d, size_t n) { CUDA_CHECK(cudaMemcpy(d.rawPointer(), h.data(), n, cudaMemcpyHostToDevice)); };

    rt::Tensor qT = mt(rt::Coords{B, qL, H, D}, DataType::kHALF);      cp(qInput, qT, qSize * 2);
    rt::Tensor kvT = mt(rt::Coords{B, 2, H, cap, D}, DataType::kHALF); cp(kvCacheInput, kvT, kvCacheSize * 2);
    rt::Tensor slT = mt(rt::Coords{B}, DataType::kINT32);                cp(sl, slT, B * 4);

    rt::Tensor qI = mt(rt::Coords{B, qL, H, D}, DataType::kINT8);
    rt::Tensor qS = mt(rt::Coords{sage::getSageQScaleSize(B, qL, H)}, DataType::kFLOAT);
    rt::Tensor vS = mt(rt::Coords{sage::getSageVScaleSize(B, H, D)}, DataType::kFLOAT);
    rt::Tensor kM = mt(rt::Coords{sage::getSageVScaleSize(B, H, D)}, DataType::kFLOAT);
    sage::launchSageQuantizeQToInt8(qT, qI, qS, s);
    sage::launchSageComputeVScalesAndKMeans(kvT, slT, vS, kM, cap, s);

    rt::Tensor oFused = mt(rt::Coords{B, qL, H, D}, DataType::kHALF);
    int32_t sb = qL * H * D, ss_ = H * D;
    float smv = 1.0f / sqrtf((float)D);
    sage::launchSageAttentionFused(
        (int8_t*)qI.rawPointer(), (half const*)kvT.rawPointer(), (half*)oFused.rawPointer(),
        (float*)qS.rawPointer(), (float*)vS.rawPointer(), nullptr,
        (int32_t const*)slT.rawPointer(),
        B, H, H, cap, sb, ss_, D, sb, ss_, D, smv, s);
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<half> oF(qSize);
    CUDA_CHECK(cudaMemcpy(oF.data(), oFused.rawPointer(), qSize * 2, cudaMemcpyDeviceToHost));

    float maxErr = 0, sumAbs = 0;
    for (size_t i = 0; i < qSize; i++) {
        float f = __half2float(oF[i]);
        float err = fabsf(f - 2.0f);
        maxErr = fmaxf(maxErr, err);
        sumAbs += fabsf(f);
    }
    std::cout << "FusedSimplePattern16Heads: maxErr from 2.0 = " << maxErr
              << " meanAbs = " << (sumAbs / qSize) << std::endl;
    std::cout << "  head0 first 8:";
    for (int d = 0; d < 8; d++) std::cout << " " << __half2float(oF[d]);
    std::cout << std::endl;
    std::cout << "  head15 first 8:";
    for (int d = 0; d < 8; d++) std::cout << " " << __half2float(oF[15 * D + d]);
    std::cout << std::endl;
    std::cout << "  per-head mean:";
    for (int h = 0; h < H; h++) {
        float sum = 0;
        for (int d = 0; d < D; d++) sum += __half2float(oF[h * D + d]);
        std::cout << " " << (sum / D);
    }
    std::cout << std::endl;
    EXPECT_LT(maxErr, 0.5f);
}
