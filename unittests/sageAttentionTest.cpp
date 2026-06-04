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

    sage::launchSageQuantizeQToInt8(qTensor, qInt8Tensor, qScaleTensor, stream);
    sage::launchSageConvertKVCacheToInt8AndFp8(
        kvCacheTensor, sequenceLengthsTensor, kInt8Tensor, vFp8Tensor, kScaleTensor, vScaleTensor, kMeanTensor, kvLen, stream);

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

    sage::launchSageQuantizeQToInt8(qTensor, qInt8Tensor, qScaleTensor, stream);
    sage::launchSageConvertKVCacheToInt8AndFp8(
        kvCacheTensor, sequenceLengthsTensor, kInt8Tensor, vFp8Tensor, kScaleTensor, vScaleTensor, kMeanTensor, kvCapacity, stream);

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
