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

// Decode-optimized Q quantization kernel: for small qoLen (<= kSAGE_WARP_Q),
// uses 1 block per (batch, head) instead of numWarpsPerBlock blocks.
// Each block computes all per-warp scales and quantizes Q in a single pass.
// Reduces kernel launch count and thread overhead by 4x for the common decode case.
__global__ void quantizeBshdToInt8DecodeKernel(half const* input, int8_t* output, float* scales,
    int32_t const batchSize, int32_t const qoLen, int32_t const numHeads, int32_t const headDim)
{
    int32_t const head = blockIdx.x % numHeads;
    int32_t const batch = blockIdx.x / numHeads;
    int32_t const numWarpsPerBlock = kSAGE_CTA_Q / kSAGE_WARP_Q; // 4
    int32_t const numBlocks = 1; // single block for small qoLen

    // Compute per-warp max for this (batch, head)
    __shared__ float sMax[kSAGE_WARP_Q]; // one max per warp in the CTA_Q range
    __shared__ float sBlockMax[kSAGE_THREADS_PER_BLOCK];

    // Initialize per-warp max
    sMax[threadIdx.x % kSAGE_WARP_Q] = 0.0F;
    if (threadIdx.x < kSAGE_WARP_Q)
    {
        for (int32_t w = threadIdx.x; w < numWarpsPerBlock * kSAGE_WARP_Q; w += kSAGE_WARP_Q)
        {
            // Not needed — only kSAGE_WARP_Q entries used for max storage
        }
    }
    for (int32_t w = 0; w < numWarpsPerBlock; ++w)
    {
        // Initialize via first few threads
        if (threadIdx.x == 0)
        {
            sMax[w] = 0.0F;
        }
    }
    __syncthreads();

    // Each thread computes local max for the dims it handles, per sequence position
    float localMax[kSAGE_WARP_Q] = {0.0F};

    for (int32_t s = 0; s < qoLen; ++s)
    {
        int32_t const warpIdx = s / kSAGE_WARP_Q;
        for (int32_t d = threadIdx.x; d < headDim; d += blockDim.x)
        {
            int64_t const inputIdx
                = (((static_cast<int64_t>(batch) * qoLen + s) * numHeads + head) * headDim + d);
            float const absVal = fabsf(__half2float(input[inputIdx]));
            localMax[warpIdx] = fmaxf(localMax[warpIdx], absVal);
        }
    }

    // Reduce across threads for each warp group
    for (int32_t w = 0; w < numWarpsPerBlock; ++w)
    {
        sBlockMax[threadIdx.x] = localMax[w];
        __syncthreads();
        for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                sBlockMax[threadIdx.x] = fmaxf(sBlockMax[threadIdx.x], sBlockMax[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        float const scale = fmaxf(sBlockMax[0] / kINT8_MAX, kMIN_SCALE);
        if (threadIdx.x == 0)
        {
            sMax[w] = scale;
            // Write scale for this warp. Scale index layout: [batch, head, blockSeq=0, warp=w]
            int32_t const scaleIdx = (batch * numHeads + head) * numWarpsPerBlock + w;
            scales[scaleIdx] = scale;
        }
        __syncthreads();
    }

    // Quantize Q using per-warp scales
    for (int32_t s = 0; s < qoLen; ++s)
    {
        int32_t const warpIdx = s / kSAGE_WARP_Q;
        float const scale = sMax[warpIdx];

        for (int32_t d = threadIdx.x; d < headDim; d += blockDim.x)
        {
            int64_t const inputIdx
                = (((static_cast<int64_t>(batch) * qoLen + s) * numHeads + head) * headDim + d);
            float const quantized = nearbyintf(__half2float(input[inputIdx]) / scale);
            output[inputIdx] = static_cast<int8_t>(min(kINT8_MAX, fmaxf(-kINT8_MAX, quantized)));
        }
    }
}
// Each thread handles one (batch, head, dim) element with serial reduction over seq.
// This is efficient because: (a) memory access is coalesced when threads read the same
// seq position for different dims, (b) no shared memory or sync overhead.
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
        int64_t const kIdx = (((static_cast<int64_t>(batch) * 2 * numKVHeads + head) * capacity + s) * headDim + dim);
        int64_t const vIdx = (((static_cast<int64_t>(batch) * 2 * numKVHeads + numKVHeads + head) * capacity + s) * headDim + dim);
        sumK += __half2float(kvCache[kIdx]);
        maxAbsV = fmaxf(maxAbsV, fabsf(__half2float(kvCache[vIdx])));
    }
    float const count = static_cast<float>(effectiveKvLen > 0 ? effectiveKvLen : 1);
    kMean[idx] = sumK / count;
    vScale[idx] = fmaxf(maxAbsV / kFP8_MAX, kMIN_SCALE);
}

// Fused convert kernel: reads KV cache once, quantizes K (INT8) and V (FP8),
// and simultaneously computes per-warp per-dim partial stats for V-max and K-sum.
// Uses half2 vectorized loads for doubled memory throughput.
// Outputs raw K_int8 (without kMean centering) and raw V_fp8 (without per-channel V scale).
// Partial stats are later reduced to produce kMean and vScale, then applied via
// applyKMeanAdjustmentKernel and applyVScaleAdjustmentKernel.
__global__ void convertKvCacheToSageKernelFused(half const* kvCache, int32_t const* sequenceLengths, int8_t* kInt8,
    int8_t* vFp8, float* kScale, float* partialVMax, float* partialKSum, int32_t const batchSize,
    int32_t const kvLen, int32_t const numHeads, int32_t const headDim, int32_t const paddedKvLen,
    int32_t const capacity, int32_t const numBlocks, int32_t const numWarpsPerBlock)
{
    // Block indexing: one block per (batch, head, seqChunk), same as original convert kernel
    int32_t const scaleIdx = blockIdx.x;
    int32_t const warpInBlock = scaleIdx % numWarpsPerBlock;
    int32_t const blockSeq = (scaleIdx / numWarpsPerBlock) % numBlocks;
    int32_t const head = (scaleIdx / (numWarpsPerBlock * numBlocks)) % numHeads;
    int32_t const batch = scaleIdx / (numWarpsPerBlock * numBlocks * numHeads);
    int32_t const effectiveKvLen = sequenceLengths == nullptr ? kvLen : min(sequenceLengths[batch], kvLen);
    int32_t const seqBegin = blockSeq * kSAGE_CTA_K + warpInBlock * kSAGE_WARP_K;
    int32_t const seqEnd = min(seqBegin + kSAGE_WARP_K, effectiveKvLen);
    int32_t const halfHeadDim = headDim / 2;

    // === Loop 1: Compute per-warp K max (for K scale) with half2 vectorized reads ===
    __shared__ float blockMax[kSAGE_THREADS_PER_BLOCK];
    float localMax = 0.0F;

    // Use half2 to read 2 K elements per iteration — halves loop iterations
    for (int32_t offset = threadIdx.x; offset < (seqEnd - seqBegin) * halfHeadDim; offset += blockDim.x)
    {
        int32_t const seqOffset = offset / halfHeadDim;
        int32_t const pair = offset % halfHeadDim;
        int32_t const dim0 = 2 * pair;
        int32_t const seq = seqBegin + seqOffset;
        int64_t const cacheIdx
            = (((static_cast<int64_t>(batch) * 2 * numHeads + head) * capacity + seq) * headDim + dim0);
        half2 const kVal = *reinterpret_cast<half2 const*>(&kvCache[cacheIdx]);
        localMax = fmaxf(localMax, fabsf(__half2float(kVal.x)));
        localMax = fmaxf(localMax, fabsf(__half2float(kVal.y)));
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

    // === Loop 2: Quantize K and V, accumulate per-dim partial stats, using half2 ===
    // Each thread handles 2 adjacent dims (dim0, dim1) via half2.
    // The dim pair is determined by threadIdx.x % halfHeadDim and is FIXED across all
    // loop iterations (since blockDim.x is a multiple of halfHeadDim for headDim in {64,128,256}).
    // Per-thread accumulators for partial V-max and K-sum.
    float threadSumK0 = 0.0F, threadSumK1 = 0.0F;
    float threadMaxAbsV0 = 0.0F, threadMaxAbsV1 = 0.0F;
    int32_t const pair = threadIdx.x % halfHeadDim;
    int32_t const dim0 = 2 * pair;
    int32_t const dim1 = 2 * pair + 1;

    for (int32_t offset = threadIdx.x; offset < kSAGE_WARP_K * halfHeadDim; offset += blockDim.x)
    {
        int32_t const seqOffset = offset / halfHeadDim;
        int32_t const seq = seqBegin + seqOffset;

        // K output (BSHD layout): consecutive dims → int16 store
        int64_t const kOutputIdx
            = (((static_cast<int64_t>(batch) * kvLen + seq) * numHeads + head) * headDim + dim0);

        // V output (BDHSP layout with SageAttention permutation):
        // permutedSeq = (seq/16)*16 + (seq%16/8)*2 + ((seq%16/2)%4)*4 + (seq%16%2)
        int32_t const mod16 = seq % 16;
        int32_t const ps = (mod16 / 8) * 2 + ((mod16 / 2) % 4) * 4 + (mod16 % 2);
        int32_t const permutedSeq = (seq / 16) * 16 + ps;
        int64_t const vOutputBase = (((static_cast<int64_t>(batch) * headDim + dim0) * numHeads + head)
            * paddedKvLen + permutedSeq);

        // V dim1 output index: same as dim0 but offset by 1 in the dim dimension
        int64_t const vOutputBase1 = (((static_cast<int64_t>(batch) * headDim + dim1) * numHeads + head)
            * paddedKvLen + permutedSeq);

        if (seq < kvLen)
        {
            if (seq < effectiveKvLen)
            {
                // half2 loads for K and V from KV cache
                int64_t const kCacheIdx
                    = (((static_cast<int64_t>(batch) * 2 * numHeads + head) * capacity + seq) * headDim + dim0);
                int64_t const vCacheIdx
                    = (((static_cast<int64_t>(batch) * 2 * numHeads + numHeads + head) * capacity + seq) * headDim
                        + dim0);

                half2 const kVal = *reinterpret_cast<half2 const*>(&kvCache[kCacheIdx]);
                half2 const vVal = *reinterpret_cast<half2 const*>(&kvCache[vCacheIdx]);

                float const kValue0 = __half2float(kVal.x);
                float const kValue1 = __half2float(kVal.y);
                float const vValue0 = __half2float(vVal.x);
                float const vValue1 = __half2float(vVal.y);

                // Quantize K to INT8 (without kMean — applied later by adjust kernel)
                // Use int16_t to write 2 adjacent INT8 values at once
                float const q0 = nearbyintf(kValue0 / scale);
                float const q1 = nearbyintf(kValue1 / scale);
                int8_t const ki0 = static_cast<int8_t>(min(kINT8_MAX, fmaxf(-kINT8_MAX, q0)));
                int8_t const ki1 = static_cast<int8_t>(min(kINT8_MAX, fmaxf(-kINT8_MAX, q1)));
                int16_t const kPacked = (static_cast<int16_t>(static_cast<uint8_t>(ki1)) << 8)
                    | static_cast<int16_t>(static_cast<uint8_t>(ki0));
                *reinterpret_cast<int16_t*>(&kInt8[kOutputIdx]) = kPacked;

                // Quantize V to FP8 (without per-channel V scale — applied later by adjust kernel)
                vFp8[vOutputBase] = convertFloatToFp8Byte(vValue0);
                vFp8[vOutputBase1] = convertFloatToFp8Byte(vValue1);

                // Accumulate per-dim partial stats
                threadSumK0 += kValue0;
                threadSumK1 += kValue1;
                threadMaxAbsV0 = fmaxf(threadMaxAbsV0, fabsf(vValue0));
                threadMaxAbsV1 = fmaxf(threadMaxAbsV1, fabsf(vValue1));
            }
            else
            {
                // Padding within kvLen: seq >= effectiveKvLen but < kvLen
                kInt8[kOutputIdx] = 0;
                kInt8[kOutputIdx + 1] = 0;
                vFp8[vOutputBase] = convertFloatToFp8Byte(0.0F);
                vFp8[vOutputBase1] = convertFloatToFp8Byte(0.0F);
            }
        }
    }

    // === Intra-block reduction: combine per-thread per-dim partials using atomic ops ===
    __shared__ float sMaxAbsV[kSAGE_THREADS_PER_BLOCK];
    __shared__ float sSumK[kSAGE_THREADS_PER_BLOCK];

    // Initialize shared memory for headDim slots
    for (int32_t i = threadIdx.x; i < headDim; i += blockDim.x)
    {
        sMaxAbsV[i] = 0.0F;
        sSumK[i] = 0.0F;
    }
    __syncthreads();

    // Each thread's dim0 and dim1 are within [0, headDim). Use atomic ops to combine.
    // atomicMax via int-reinterpretation: safe for non-negative float values.
    atomicMax(reinterpret_cast<int*>(&sMaxAbsV[dim0]),
        __float_as_int(threadMaxAbsV0));
    atomicMax(reinterpret_cast<int*>(&sMaxAbsV[dim1]),
        __float_as_int(threadMaxAbsV1));
    atomicAdd(&sSumK[dim0], threadSumK0);
    atomicAdd(&sSumK[dim1], threadSumK1);
    __syncthreads();

    // Write per-dim per-block partials to global buffers
    if (threadIdx.x < headDim)
    {
        int32_t const dim = threadIdx.x;
        int32_t const partialIdx = ((batch * numHeads + head) * numBlocks + blockSeq) * headDim + dim;
        partialVMax[partialIdx] = sMaxAbsV[dim];
        partialKSum[partialIdx] = sSumK[dim];
    }
}

// Reduction kernel: combines per-warp-block partial V max and K sum across all blocks
// to produce final per-channel vScale and kMean.
__global__ void reducePartialChannelStatsKernel(float const* partialVMax, float const* partialKSum,
    float* vScale, float* kMean, int32_t const* sequenceLengths, int32_t const batchSize,
    int32_t const numKVHeads, int32_t const headDim, int32_t const numBlocks, int32_t const kvLen)
{
    int32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t const totalElements = batchSize * numKVHeads * headDim;
    if (idx >= totalElements) return;

    int32_t const dim = idx % headDim;
    int32_t const head = (idx / headDim) % numKVHeads;
    int32_t const batch = idx / (numKVHeads * headDim);

    // Reduce across all blocks for this (batch, head, dim)
    float maxV = 0.0F;
    float sumK = 0.0F;
    for (int32_t b = 0; b < numBlocks; ++b)
    {
        int32_t const partialIdx = ((batch * numKVHeads + head) * numBlocks + b) * headDim + dim;
        maxV = fmaxf(maxV, partialVMax[partialIdx]);
        sumK += partialKSum[partialIdx];
    }

    int32_t const effectiveKvLen = sequenceLengths == nullptr ? kvLen : min(sequenceLengths[batch], kvLen);
    float const count = static_cast<float>(effectiveKvLen > 0 ? effectiveKvLen : 1);
    kMean[idx] = sumK / count;
    vScale[idx] = fmaxf(maxV / kFP8_MAX, kMIN_SCALE);
}

// K-mean adjustment kernel: applies kMean centering to pre-quantized K_int8.
// Each block processes the same (batch, head, warpChunk) as the convert kernel.
// K_int8 = K_int8_raw - round(kMean[dim] / kScale[warp])
// This matches the original algorithm's K-mean subtraction, with
// potential ±1 INT8 difference due to rounding order.
__global__ void applyKMeanAdjustmentKernel(int8_t* kInt8, float const* kScale, float const* kMean,
    int32_t const batchSize, int32_t const kvLen, int32_t const numHeads, int32_t const headDim,
    int32_t const numBlocks, int32_t const numWarpsPerBlock)
{
    int32_t const scaleIdx = blockIdx.x;
    int32_t const warpInBlock = scaleIdx % numWarpsPerBlock;
    int32_t const blockSeq = (scaleIdx / numWarpsPerBlock) % numBlocks;
    int32_t const head = (scaleIdx / (numWarpsPerBlock * numBlocks)) % numHeads;
    int32_t const batch = scaleIdx / (numWarpsPerBlock * numBlocks * numHeads);
    int32_t const seqBegin = blockSeq * kSAGE_CTA_K + warpInBlock * kSAGE_WARP_K;
    int32_t const seqEnd = min(seqBegin + kSAGE_WARP_K, kvLen);

    float const kScaleVal = kScale[scaleIdx];
    int32_t const kMeanBase = batch * numHeads * headDim + head * headDim;

    // Vectorize using halfHeadDim for half2-like processing of int8 pairs
    int32_t const halfHeadDim = headDim / 2;

    for (int32_t offset = threadIdx.x; offset < (seqEnd - seqBegin) * halfHeadDim; offset += blockDim.x)
    {
        int32_t const seqOffset = offset / halfHeadDim;
        int32_t const pair = offset % halfHeadDim;
        int32_t const dim0 = 2 * pair;
        int32_t const dim1 = 2 * pair + 1;
        int32_t const seq = seqBegin + seqOffset;

        int64_t const kIdx = (((static_cast<int64_t>(batch) * kvLen + seq) * numHeads + head) * headDim + dim0);

        float const km0 = kMean[kMeanBase + dim0];
        float const km1 = kMean[kMeanBase + dim1];

        int32_t const adj0 = static_cast<int32_t>(nearbyintf(km0 / kScaleVal));
        int32_t const adj1 = static_cast<int32_t>(nearbyintf(km1 / kScaleVal));

        // Read existing K_int8 pair, adjust, clamp, write back
        int32_t const ki0 = static_cast<int32_t>(kInt8[kIdx]) - adj0;
        int32_t const ki1 = static_cast<int32_t>(kInt8[kIdx + 1]) - adj1;

        kInt8[kIdx] = static_cast<int8_t>(min(127, max(-127, ki0)));
        kInt8[kIdx + 1] = static_cast<int8_t>(min(127, max(-127, ki1)));
    }
}

// V-scale adjustment kernel: applies per-channel V scaling to pre-quantized V_fp8.
// V_fp8 = convertFloatToFp8(convertFp8ToFloat(V_fp8_raw) / vScale[dim])
// This restores per-channel V scaling after the fused convert kernel stored raw values.
__global__ void applyVScaleAdjustmentKernel(int8_t* vFp8, float const* vScale, int32_t const batchSize,
    int32_t const kvLen, int32_t const numHeads, int32_t const headDim, int32_t const paddedKvLen,
    int32_t const numBlocks, int32_t const numWarpsPerBlock)
{
    int32_t const scaleIdx = blockIdx.x;
    int32_t const warpInBlock = scaleIdx % numWarpsPerBlock;
    int32_t const blockSeq = (scaleIdx / numWarpsPerBlock) % numBlocks;
    int32_t const head = (scaleIdx / (numWarpsPerBlock * numBlocks)) % numHeads;
    int32_t const batch = scaleIdx / (numWarpsPerBlock * numBlocks * numHeads);
    int32_t const seqBegin = blockSeq * kSAGE_CTA_K + warpInBlock * kSAGE_WARP_K;
    int32_t const seqEnd = min(seqBegin + kSAGE_WARP_K, kvLen);

    int32_t const vScaleBase = batch * numHeads * headDim + head * headDim;
    int32_t const halfHeadDim = headDim / 2;

    for (int32_t offset = threadIdx.x; offset < (seqEnd - seqBegin) * halfHeadDim; offset += blockDim.x)
    {
        int32_t const seqOffset = offset / halfHeadDim;
        int32_t const pair = offset % halfHeadDim;
        int32_t const dim0 = 2 * pair;
        int32_t const dim1 = 2 * pair + 1;
        int32_t const seq = seqBegin + seqOffset;

        // SageAttention V permutation
        int32_t const mod16 = seq % 16;
        int32_t const ps = (mod16 / 8) * 2 + ((mod16 / 2) % 4) * 4 + (mod16 % 2);
        int32_t const permutedSeq = (seq / 16) * 16 + ps;

        int64_t const vIdx0 = (((static_cast<int64_t>(batch) * headDim + dim0) * numHeads + head)
            * paddedKvLen + permutedSeq);
        int64_t const vIdx1 = (((static_cast<int64_t>(batch) * headDim + dim1) * numHeads + head)
            * paddedKvLen + permutedSeq);

        float const vs0 = vScale[vScaleBase + dim0];
        float const vs1 = vScale[vScaleBase + dim1];

        // Read raw FP8, dequantize, scale, requantize
        // This double-conversion introduces minor precision loss (bounded by FP8 E4M3 precision)
        __nv_fp8_e4m3 const vFp8Raw0 = *reinterpret_cast<__nv_fp8_e4m3 const*>(&vFp8[vIdx0]);
        __nv_fp8_e4m3 const vFp8Raw1 = *reinterpret_cast<__nv_fp8_e4m3 const*>(&vFp8[vIdx1]);

        vFp8[vIdx0] = convertFloatToFp8Byte(static_cast<float>(vFp8Raw0) / vs0);
        vFp8[vIdx1] = convertFloatToFp8Byte(static_cast<float>(vFp8Raw1) / vs1);
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

int32_t getSageStatsPartialsSize(int32_t batchSize, int32_t kvLen, int32_t numKVHeads, int32_t headDim) noexcept
{
    int32_t const numBlocks = (kvLen + kSAGE_CTA_K - 1) / kSAGE_CTA_K;
    return batchSize * numKVHeads * headDim * numBlocks;
}

void launchSageQuantizeQToInt8(rt::Tensor const& q, rt::Tensor& qInt8, rt::Tensor& qScale, cudaStream_t stream)
{
    rt::Coords const qShape = q.getShape();
    int32_t const batchSize = static_cast<int32_t>(qShape[0]);
    int32_t const qoLen = static_cast<int32_t>(qShape[1]);
    int32_t const numHeads = static_cast<int32_t>(qShape[2]);
    int32_t const headDim = static_cast<int32_t>(qShape[3]);

    check::check(q.getDataType() == nvinfer1::DataType::kHALF, "SageAttention Q input must be FP16.");
    check::check(qInt8.getDataType() == nvinfer1::DataType::kINT8, "SageAttention Q output must be INT8.");
    check::check(qScale.getDataType() == nvinfer1::DataType::kFLOAT, "SageAttention Q scale must be FP32.");

    // Use decode-optimized kernel for small qoLen (common SageAttention decode case qoLen=1)
    // Reduces grid size from batchSize*numHeads*4 to batchSize*numHeads blocks
    if (qoLen <= kSAGE_WARP_Q)
    {
        int32_t const gridSize = batchSize * numHeads;
        quantizeBshdToInt8DecodeKernel<<<gridSize, kSAGE_THREADS_PER_BLOCK, 0, stream>>>(
            q.dataPointer<half>(), qInt8.dataPointer<int8_t>(), qScale.dataPointer<float>(),
            batchSize, qoLen, numHeads, headDim);
        return;
    }

    int32_t const numBlocks = (qoLen + kSAGE_CTA_Q - 1) / kSAGE_CTA_Q;
    int32_t const numWarpsPerBlock = kSAGE_CTA_Q / kSAGE_WARP_Q;
    int32_t const scaleSize = getSageQScaleSize(batchSize, qoLen, numHeads);

    quantizeBshdToInt8PerWarpKernel<<<scaleSize, kSAGE_THREADS_PER_BLOCK, 0, stream>>>(q.dataPointer<half>(),
        qInt8.dataPointer<int8_t>(), qScale.dataPointer<float>(), batchSize, qoLen, numHeads, headDim, kSAGE_CTA_Q,
        kSAGE_WARP_Q, numBlocks, numWarpsPerBlock);
}

void launchSageConvertKVCacheToInt8AndFp8(rt::Tensor const& kvCache, rt::Tensor const& sequenceLengths,
    rt::Tensor& kInt8, rt::Tensor& vFp8, rt::Tensor& kScale, rt::Tensor& vScale, rt::Tensor& kMean,
    rt::Tensor& partialVMax, rt::Tensor& partialKSum, int32_t kvLen, cudaStream_t stream)
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

    // Step 1: Fused convert kernel — read KV cache once, quantize K/V, compute partial stats
    convertKvCacheToSageKernelFused<<<scaleSize, kSAGE_THREADS_PER_BLOCK, 0, stream>>>(kvCache.dataPointer<half>(),
        sequenceLengths.isEmpty() ? nullptr : sequenceLengths.dataPointer<int32_t>(), kInt8.dataPointer<int8_t>(),
        vFp8.dataPointer<int8_t>(), kScale.dataPointer<float>(), partialVMax.dataPointer<float>(),
        partialKSum.dataPointer<float>(), batchSize, kvLen, numKVHeads, headDim, paddedKvLen, capacity, numBlocks,
        numWarpsPerBlock);

    // Step 2: Reduce partial stats to produce final vScale and kMean
    int32_t const perChannelSize = getSageVScaleSize(batchSize, numKVHeads, headDim);
    int32_t const statBlocks = (perChannelSize + kSAGE_THREADS_PER_BLOCK - 1) / kSAGE_THREADS_PER_BLOCK;
    reducePartialChannelStatsKernel<<<statBlocks, kSAGE_THREADS_PER_BLOCK, 0, stream>>>(
        partialVMax.dataPointer<float>(), partialKSum.dataPointer<float>(), vScale.dataPointer<float>(),
        kMean.dataPointer<float>(),
        sequenceLengths.isEmpty() ? nullptr : sequenceLengths.dataPointer<int32_t>(),
        batchSize, numKVHeads, headDim, numBlocks, kvLen);

    // Step 3: Apply K-mean centering to pre-quantized K_int8
    applyKMeanAdjustmentKernel<<<scaleSize, kSAGE_THREADS_PER_BLOCK, 0, stream>>>(
        kInt8.dataPointer<int8_t>(), kScale.dataPointer<float>(), kMean.dataPointer<float>(),
        batchSize, kvLen, numKVHeads, headDim, numBlocks, numWarpsPerBlock);

    // Step 4: Apply per-channel V scaling to pre-quantized V_fp8
    applyVScaleAdjustmentKernel<<<scaleSize, kSAGE_THREADS_PER_BLOCK, 0, stream>>>(
        vFp8.dataPointer<int8_t>(), vScale.dataPointer<float>(),
        batchSize, kvLen, numKVHeads, headDim, paddedKvLen, numBlocks, numWarpsPerBlock);

    // Zero out V-FP8 tensor padding area (only beyond kvLen).
    // When kvLen == paddedKvLen (kvLen is multiple of CTA_K), no memset needed.
    if (paddedKvLen > kvLen)
    {
        int32_t const vFp8Numel = static_cast<int32_t>(batchSize) * headDim * numKVHeads * paddedKvLen;
        int32_t const kvLenRegion = static_cast<int32_t>(batchSize) * headDim * numKVHeads * kvLen;
        int32_t const padBytes = (vFp8Numel - kvLenRegion) * static_cast<int32_t>(sizeof(int8_t));
        CUDA_CHECK(cudaMemsetAsync(
            static_cast<int8_t*>(vFp8.rawPointer()) + kvLenRegion, 0, padBytes, stream));
    }
}

} // namespace sage
} // namespace trt_edgellm
