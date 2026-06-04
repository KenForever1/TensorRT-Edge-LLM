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

#include "sageAttentionRuntimeKernels.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#endif

namespace trt_edgellm
{
namespace sage
{
namespace
{
constexpr int32_t kSAGE_CTA_Q = 128;
constexpr int32_t kSAGE_CTA_K = 64;
constexpr int32_t kSAGE_WARP_Q = 32;
constexpr int32_t kSAGE_WARP_K = 64;
constexpr int32_t kSAGE_THREADS_PER_BLOCK = 256;
constexpr float kINT8_MAX = 127.0F;
constexpr float kMIN_SCALE = 1.0e-6F;
constexpr float kFP8_MAX = 448.0F; // Max representable value for FP8 e4m3

__device__ int8_t convertFloatToFp8Byte(float const value)
{
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    __nv_fp8_e4m3 const fp8Value(value);
    return *reinterpret_cast<int8_t const*>(&fp8Value);
#else
    return 0;
#endif
}

__global__ void quantizeBshdToInt8PerWarpKernel(half const* input, int8_t* output, float* scales,
    int32_t const batchSize, int32_t const seqLen, int32_t const numHeads, int32_t const headDim,
    int32_t const ctaLen, int32_t const warpLen, int32_t const numBlocks, int32_t const numWarpsPerBlock)
{
    int32_t const scaleIdx = blockIdx.x;
    int32_t const warpInBlock = scaleIdx % numWarpsPerBlock;
    int32_t const blockSeq = (scaleIdx / numWarpsPerBlock) % numBlocks;
    int32_t const head = (scaleIdx / (numWarpsPerBlock * numBlocks)) % numHeads;
    int32_t const batch = scaleIdx / (numWarpsPerBlock * numBlocks * numHeads);
    int32_t const seqBegin = blockSeq * ctaLen + warpInBlock * warpLen;
    int32_t const seqEnd = min(seqBegin + warpLen, seqLen);

    __shared__ float blockMax[kSAGE_THREADS_PER_BLOCK];
    float localMax = 0.0F;

    for (int32_t offset = threadIdx.x; offset < (seqEnd - seqBegin) * headDim; offset += blockDim.x)
    {
        int32_t const seqOffset = offset / headDim;
        int32_t const dim = offset % headDim;
        int32_t const seq = seqBegin + seqOffset;
        int64_t const inputIdx
            = (((static_cast<int64_t>(batch) * seqLen + seq) * numHeads + head) * headDim + dim);
        localMax = fmaxf(localMax, fabsf(__half2float(input[inputIdx])));
    }

    blockMax[threadIdx.x] = localMax;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            blockMax[threadIdx.x] = fmaxf(blockMax[threadIdx.x], blockMax[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float const scale = fmaxf(blockMax[0] / kINT8_MAX, kMIN_SCALE);
    if (threadIdx.x == 0)
    {
        scales[scaleIdx] = scale;
    }

    for (int32_t offset = threadIdx.x; offset < (seqEnd - seqBegin) * headDim; offset += blockDim.x)
    {
        int32_t const seqOffset = offset / headDim;
        int32_t const dim = offset % headDim;
        int32_t const seq = seqBegin + seqOffset;
        int64_t const inputIdx
            = (((static_cast<int64_t>(batch) * seqLen + seq) * numHeads + head) * headDim + dim);
        float const quantized = nearbyintf(__half2float(input[inputIdx]) / scale);
        output[inputIdx] = static_cast<int8_t>(min(kINT8_MAX, fmaxf(-kINT8_MAX, quantized)));
    }

    for (int32_t offset = threadIdx.x + (seqEnd - seqBegin) * headDim; offset < warpLen * headDim;
         offset += blockDim.x)
    {
        int32_t const seqOffset = offset / headDim;
        int32_t const dim = offset % headDim;
        int32_t const seq = seqBegin + seqOffset;
        if (seq < seqLen)
        {
            int64_t const inputIdx
                = (((static_cast<int64_t>(batch) * seqLen + seq) * numHeads + head) * headDim + dim);
            output[inputIdx] = 0;
        }
    }
}

// Kernel to compute per-channel V scales and K-means over valid sequence positions.
__global__ void computePerChannelStatsKernel(half const* kvCache, int32_t const* sequenceLengths,
    float* vScale, float* kMean, int32_t const batchSize, int32_t const numKVHeads, int32_t const headDim,
    int32_t const capacity)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const totalElements = batchSize * numKVHeads * headDim;
    if (idx >= totalElements) return;

    int32_t const dim = idx % headDim;
    int32_t const head = (idx / headDim) % numKVHeads;
    int32_t const batch = idx / (numKVHeads * headDim);
    int32_t const effectiveKvLen = sequenceLengths == nullptr ? capacity : min(sequenceLengths[batch], capacity);

    float maxAbsV = 0.0F;
    float sumK = 0.0F;
    for (int32_t s = 0; s < effectiveKvLen; ++s)
    {
        // K cache at kvCache[batch][0][head][s][dim]
        int64_t const kCacheIdx = (((static_cast<int64_t>(batch) * 2 * numKVHeads + head) * capacity + s) * headDim + dim);
        sumK += __half2float(kvCache[kCacheIdx]);
        // V cache at kvCache[batch][1][head][s][dim]
        int64_t const vCacheIdx = (((static_cast<int64_t>(batch) * 2 * numKVHeads + numKVHeads + head) * capacity + s) * headDim + dim);
        maxAbsV = fmaxf(maxAbsV, fabsf(__half2float(kvCache[vCacheIdx])));
    }
    float const count = static_cast<float>(effectiveKvLen > 0 ? effectiveKvLen : 1);
    // K mean: mean over sequence dim
    kMean[idx] = sumK / count;
    // V scale: scale = max(|V|) / 448.0; V_fp8 = fp8(V / scale); kernel does O *= scale
    vScale[idx] = fmaxf(maxAbsV / kFP8_MAX, kMIN_SCALE);
}

__global__ void convertKvCacheToSageKernel(half const* kvCache, int32_t const* sequenceLengths, int8_t* kInt8,
    int8_t* vFp8, float* kScale, float const* vScale, float const* kMean, int32_t const batchSize,
    int32_t const kvLen, int32_t const numHeads, int32_t const headDim, int32_t const paddedKvLen,
    int32_t const capacity, int32_t const numBlocks, int32_t const numWarpsPerBlock)
{
    int32_t const scaleIdx = blockIdx.x;
    int32_t const warpInBlock = scaleIdx % numWarpsPerBlock;
    int32_t const blockSeq = (scaleIdx / numWarpsPerBlock) % numBlocks;
    int32_t const head = (scaleIdx / (numWarpsPerBlock * numBlocks)) % numHeads;
    int32_t const batch = scaleIdx / (numWarpsPerBlock * numBlocks * numHeads);
    int32_t const effectiveKvLen = sequenceLengths == nullptr ? kvLen : min(sequenceLengths[batch], kvLen);
    int32_t const seqBegin = blockSeq * kSAGE_CTA_K + warpInBlock * kSAGE_WARP_K;
    int32_t const seqEnd = min(seqBegin + kSAGE_WARP_K, effectiveKvLen);

    __shared__ float blockMax[kSAGE_THREADS_PER_BLOCK];
    float localMax = 0.0F;

    for (int32_t offset = threadIdx.x; offset < (seqEnd - seqBegin) * headDim; offset += blockDim.x)
    {
        int32_t const seqOffset = offset / headDim;
        int32_t const dim = offset % headDim;
        int32_t const seq = seqBegin + seqOffset;
        int64_t const cacheIdx
            = (((static_cast<int64_t>(batch) * 2 * numHeads + head) * capacity + seq) * headDim + dim);
        localMax = fmaxf(localMax, fabsf(__half2float(kvCache[cacheIdx])));
    }

    blockMax[threadIdx.x] = localMax;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            blockMax[threadIdx.x] = fmaxf(blockMax[threadIdx.x], blockMax[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float const scale = fmaxf(blockMax[0] / kINT8_MAX, kMIN_SCALE);
    if (threadIdx.x == 0)
    {
        kScale[scaleIdx] = scale;
    }

    for (int32_t offset = threadIdx.x; offset < kSAGE_WARP_K * headDim; offset += blockDim.x)
    {
        int32_t const seqOffset = offset / headDim;
        int32_t const dim = offset % headDim;
        int32_t const seq = seqBegin + seqOffset;
        int64_t const kOutputIdx
            = (((static_cast<int64_t>(batch) * kvLen + seq) * numHeads + head) * headDim + dim);
        if (seq < kvLen)
        {
            // SageAttention V sequence permutation [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]
            // from fused.cu TransposePadPermuteKernel: smem_load_row = (seq/16)*16 + (mod/8)*2 + ((mod/2)%4)*4 + (mod%2)
            int32_t const mod16 = seq % 16;
            int32_t const ps = (mod16 / 8) * 2 + ((mod16 / 2) % 4) * 4 + (mod16 % 2);
            int32_t const permutedSeq = (seq / 16) * 16 + ps;
            int64_t const vOutputIdx = (((static_cast<int64_t>(batch) * headDim + dim) * numHeads + head)
                * paddedKvLen + permutedSeq);

            if (seq < effectiveKvLen)
            {
                int64_t const kCacheIdx
                    = (((static_cast<int64_t>(batch) * 2 * numHeads + head) * capacity + seq) * headDim + dim);
                int64_t const vCacheIdx
                    = (((static_cast<int64_t>(batch) * 2 * numHeads + numHeads + head) * capacity + seq) * headDim
                        + dim);
                float const kValue = __half2float(kvCache[kCacheIdx]);
                float const vValue = __half2float(kvCache[vCacheIdx]);
                // Smooth_k: subtract K-mean before quantization (softmax is invariant to constant shift)
                float const km = (kMean != nullptr)
                    ? kMean[batch * numHeads * headDim + head * headDim + dim]
                    : 0.0F;
                float const quantized = nearbyintf((kValue - km) / scale);
                kInt8[kOutputIdx] = static_cast<int8_t>(min(kINT8_MAX, fmaxf(-kINT8_MAX, quantized)));

                // Apply per-channel V scale before FP8 conversion
                float const vs = (vScale != nullptr)
                    ? vScale[batch * numHeads * headDim + head * headDim + dim]
                    : 1.0F;
                vFp8[vOutputIdx] = convertFloatToFp8Byte(vValue / vs);
            }
            else
            {
                kInt8[kOutputIdx] = 0;
                vFp8[vOutputIdx] = convertFloatToFp8Byte(0.0F);
            }
        }
    }
}
} // namespace

int32_t getSagePaddedKvLen(int32_t kvLen) noexcept
{
    return ((kvLen + kSAGE_CTA_K - 1) / kSAGE_CTA_K) * kSAGE_CTA_K;
}

int32_t getSageQScaleSize(int32_t batchSize, int32_t qoLen, int32_t numHeads) noexcept
{
    int32_t const numBlocks = (qoLen + kSAGE_CTA_Q - 1) / kSAGE_CTA_Q;
    int32_t const numWarpsPerBlock = kSAGE_CTA_Q / kSAGE_WARP_Q;
    return batchSize * numHeads * numBlocks * numWarpsPerBlock;
}

int32_t getSageKScaleSize(int32_t batchSize, int32_t kvLen, int32_t numHeads) noexcept
{
    int32_t const numBlocks = (kvLen + kSAGE_CTA_K - 1) / kSAGE_CTA_K;
    int32_t const numWarpsPerBlock = kSAGE_CTA_K / kSAGE_WARP_K;
    return batchSize * numHeads * numBlocks * numWarpsPerBlock;
}

int32_t getSageVScaleSize(int32_t batchSize, int32_t numKVHeads, int32_t headDim) noexcept
{
    return batchSize * numKVHeads * headDim;
}

void launchSageQuantizeQToInt8(rt::Tensor const& q, rt::Tensor& qInt8, rt::Tensor& qScale, cudaStream_t stream)
{
    rt::Coords const qShape = q.getShape();
    int32_t const batchSize = static_cast<int32_t>(qShape[0]);
    int32_t const qoLen = static_cast<int32_t>(qShape[1]);
    int32_t const numHeads = static_cast<int32_t>(qShape[2]);
    int32_t const headDim = static_cast<int32_t>(qShape[3]);
    int32_t const numBlocks = (qoLen + kSAGE_CTA_Q - 1) / kSAGE_CTA_Q;
    int32_t const numWarpsPerBlock = kSAGE_CTA_Q / kSAGE_WARP_Q;
    int32_t const scaleSize = getSageQScaleSize(batchSize, qoLen, numHeads);

    check::check(q.getDataType() == nvinfer1::DataType::kHALF, "SageAttention Q input must be FP16.");
    check::check(qInt8.getDataType() == nvinfer1::DataType::kINT8, "SageAttention Q output must be INT8.");
    check::check(qScale.getDataType() == nvinfer1::DataType::kFLOAT, "SageAttention Q scale must be FP32.");

    quantizeBshdToInt8PerWarpKernel<<<scaleSize, kSAGE_THREADS_PER_BLOCK, 0, stream>>>(q.dataPointer<half>(),
        qInt8.dataPointer<int8_t>(), qScale.dataPointer<float>(), batchSize, qoLen, numHeads, headDim, kSAGE_CTA_Q,
        kSAGE_WARP_Q, numBlocks, numWarpsPerBlock);
}

void launchSageConvertKVCacheToInt8AndFp8(rt::Tensor const& kvCache, rt::Tensor const& sequenceLengths,
    rt::Tensor& kInt8, rt::Tensor& vFp8, rt::Tensor& kScale, rt::Tensor& vScale, rt::Tensor& kMean,
    int32_t kvLen, cudaStream_t stream)
{
    rt::Coords const kvCacheShape = kvCache.getShape();
    int32_t const batchSize = static_cast<int32_t>(kvCacheShape[0]);
    int32_t const numKVHeads = static_cast<int32_t>(kvCacheShape[2]);
    int32_t const capacity = static_cast<int32_t>(kvCacheShape[3]);
    int32_t const headDim = static_cast<int32_t>(kvCacheShape[4]);
    int32_t const paddedKvLen = getSagePaddedKvLen(kvLen);
    int32_t const numBlocks = (kvLen + kSAGE_CTA_K - 1) / kSAGE_CTA_K;
    int32_t const numWarpsPerBlock = kSAGE_CTA_K / kSAGE_WARP_K;
    int32_t const scaleSize = getSageKScaleSize(batchSize, kvLen, numKVHeads);

    check::check(kvCache.getDataType() == nvinfer1::DataType::kHALF, "SageAttention KV cache input must be FP16.");
    check::check(kInt8.getDataType() == nvinfer1::DataType::kINT8, "SageAttention K output must be INT8.");
    check::check(vFp8.getDataType() == nvinfer1::DataType::kINT8, "SageAttention V output must be INT8.");
    check::check(kScale.getDataType() == nvinfer1::DataType::kFLOAT, "SageAttention K scale must be FP32.");
    check::check(kvLen <= capacity, "SageAttention KV length must not exceed KV cache capacity.");

    // First, compute per-channel V scales and K-means in one pass
    int32_t const perChannelSize = getSageVScaleSize(batchSize, numKVHeads, headDim);
    int32_t const statBlocks = (perChannelSize + kSAGE_THREADS_PER_BLOCK - 1) / kSAGE_THREADS_PER_BLOCK;
    computePerChannelStatsKernel<<<statBlocks, kSAGE_THREADS_PER_BLOCK, 0, stream>>>(
        kvCache.dataPointer<half>(),
        sequenceLengths.isEmpty() ? nullptr : sequenceLengths.dataPointer<int32_t>(),
        vScale.dataPointer<float>(), kMean.dataPointer<float>(),
        batchSize, numKVHeads, headDim, capacity);

    // Zero out V-FP8 tensor (padding area)
    CUDA_CHECK(cudaMemsetAsync(vFp8.rawPointer(), 0,
        static_cast<size_t>(batchSize) * headDim * numKVHeads * paddedKvLen * sizeof(int8_t), stream));
    // Convert KV cache to SageAttention quantized format with V scaling and K-mean centering
    float const* vScalePtr = vScale.isEmpty() ? nullptr : vScale.dataPointer<float>();
    float const* kMeanPtr = kMean.isEmpty() ? nullptr : kMean.dataPointer<float>();
    convertKvCacheToSageKernel<<<scaleSize, kSAGE_THREADS_PER_BLOCK, 0, stream>>>(kvCache.dataPointer<half>(),
        sequenceLengths.isEmpty() ? nullptr : sequenceLengths.dataPointer<int32_t>(), kInt8.dataPointer<int8_t>(),
        vFp8.dataPointer<int8_t>(), kScale.dataPointer<float>(), vScalePtr, kMeanPtr, batchSize, kvLen, numKVHeads,
        headDim, paddedKvLen, capacity, numBlocks, numWarpsPerBlock);
}

} // namespace sage
} // namespace trt_edgellm
