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

#include "sageAttentionTestKernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include <cuda_fp8.h>
#endif

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace
{

constexpr int32_t kSageCtaQ = 128;
constexpr int32_t kSageCtaK = 64;
constexpr int32_t kSageWarpQ = 32;
constexpr int32_t kSageWarpK = 64;
constexpr float kInt8Max = 127.0F;
constexpr float kMinScale = 1.0e-6F;
constexpr int32_t kThreadsPerBlock = 256;

__device__ int64_t getBshdOffset(int32_t const batch, int32_t const seq, int32_t const head, int32_t const dim,
    int32_t const seqLen, int32_t const numHeads, int32_t const headDim)
{
    return (((static_cast<int64_t>(batch) * seqLen + seq) * numHeads + head) * headDim + dim);
}

__global__ void sageReferenceAttentionKernel(half const* q, half const* k, half const* v, half* output,
    int32_t const batchSize, int32_t const qoLen, int32_t const kvLen, int32_t const numQoHeads,
    int32_t const numKvHeads, int32_t const headDim, bool const causal)
{
    int64_t const totalElements = static_cast<int64_t>(batchSize) * qoLen * numQoHeads * headDim;
    int64_t const outputIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (outputIdx >= totalElements)
    {
        return;
    }

    int32_t const dim = static_cast<int32_t>(outputIdx % headDim);
    int64_t const tmp0 = outputIdx / headDim;
    int32_t const qHead = static_cast<int32_t>(tmp0 % numQoHeads);
    int64_t const tmp1 = tmp0 / numQoHeads;
    int32_t const qSeq = static_cast<int32_t>(tmp1 % qoLen);
    int32_t const batch = static_cast<int32_t>(tmp1 / qoLen);

    int32_t const groupSize = numQoHeads / numKvHeads;
    int32_t const kvHead = min(qHead / groupSize, numKvHeads - 1);
    int32_t const qAbsSeq = qSeq + kvLen - qoLen;
    float const scale = rsqrtf(static_cast<float>(headDim));

    float maxLogit = -FLT_MAX;
    for (int32_t kvSeq = 0; kvSeq < kvLen; ++kvSeq)
    {
        if (causal && kvSeq > qAbsSeq)
        {
            continue;
        }

        float dot = 0.0F;
        for (int32_t d = 0; d < headDim; ++d)
        {
            int64_t const qOffset = getBshdOffset(batch, qSeq, qHead, d, qoLen, numQoHeads, headDim);
            int64_t const kOffset = getBshdOffset(batch, kvSeq, kvHead, d, kvLen, numKvHeads, headDim);
            dot += __half2float(q[qOffset]) * __half2float(k[kOffset]);
        }
        maxLogit = fmaxf(maxLogit, dot * scale);
    }

    float denom = 0.0F;
    float accum = 0.0F;
    for (int32_t kvSeq = 0; kvSeq < kvLen; ++kvSeq)
    {
        if (causal && kvSeq > qAbsSeq)
        {
            continue;
        }

        float dot = 0.0F;
        for (int32_t d = 0; d < headDim; ++d)
        {
            int64_t const qOffset = getBshdOffset(batch, qSeq, qHead, d, qoLen, numQoHeads, headDim);
            int64_t const kOffset = getBshdOffset(batch, kvSeq, kvHead, d, kvLen, numKvHeads, headDim);
            dot += __half2float(q[qOffset]) * __half2float(k[kOffset]);
        }

        float const weight = expf(dot * scale - maxLogit);
        int64_t const vOffset = getBshdOffset(batch, kvSeq, kvHead, dim, kvLen, numKvHeads, headDim);
        denom += weight;
        accum += weight * __half2float(v[vOffset]);
    }

    output[outputIdx] = __float2half(accum / denom);
}

__global__ void quantizeToInt8PerWarpKernel(half const* input, int8_t* output, float* scales, int32_t const seqLen,
    int32_t const numHeads, int32_t const headDim, int32_t const ctaLen, int32_t const warpLen, int32_t const numBlocks,
    int32_t const numWarpsPerBlock)
{
    int32_t const scaleIdx = blockIdx.x;
    int32_t const warpInBlock = scaleIdx % numWarpsPerBlock;
    int32_t const blockSeq = (scaleIdx / numWarpsPerBlock) % numBlocks;
    int32_t const head = (scaleIdx / (numWarpsPerBlock * numBlocks)) % numHeads;
    int32_t const batch = scaleIdx / (numWarpsPerBlock * numBlocks * numHeads);
    int32_t const seqBegin = blockSeq * ctaLen + warpInBlock * warpLen;
    int32_t const seqEnd = min(seqBegin + warpLen, seqLen);

    __shared__ float blockMax[kThreadsPerBlock];
    float maxAbs = 0.0F;
    for (int32_t linearIdx = threadIdx.x; linearIdx < (seqEnd - seqBegin) * headDim; linearIdx += blockDim.x)
    {
        int32_t const localSeq = linearIdx / headDim;
        int32_t const dim = linearIdx % headDim;
        int32_t const seq = seqBegin + localSeq;
        int64_t const inputIdx = getBshdOffset(batch, seq, head, dim, seqLen, numHeads, headDim);
        maxAbs = fmaxf(maxAbs, fabsf(__half2float(input[inputIdx])));
    }

    blockMax[threadIdx.x] = maxAbs;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            blockMax[threadIdx.x] = fmaxf(blockMax[threadIdx.x], blockMax[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float const scale = fmaxf(blockMax[0] / kInt8Max, kMinScale);
    if (threadIdx.x == 0)
    {
        scales[scaleIdx] = scale;
    }
    __syncthreads();

    for (int32_t linearIdx = threadIdx.x; linearIdx < (seqEnd - seqBegin) * headDim; linearIdx += blockDim.x)
    {
        int32_t const localSeq = linearIdx / headDim;
        int32_t const dim = linearIdx % headDim;
        int32_t const seq = seqBegin + localSeq;
        int64_t const inputIdx = getBshdOffset(batch, seq, head, dim, seqLen, numHeads, headDim);
        float const quantized = nearbyintf(__half2float(input[inputIdx]) / scale);
        output[inputIdx] = static_cast<int8_t>(min(kInt8Max, fmaxf(-kInt8Max, quantized)));
    }
}

__device__ int8_t convertToFp8Byte(float const value)
{
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_e4m3 const fp8Value(value);
    return *reinterpret_cast<int8_t const*>(&fp8Value);
#else
    // This helper is only valid for the SM89 SageAttention path. The fallback is only present so lower-arch
    // host/device compilation can succeed; runtime tests skip before launching on unsupported GPUs.
    return 0;
#endif
}

__global__ void quantizeToFp8TransposedKernel(half const* input, int8_t* output, int32_t const batchSize,
    int32_t const seqLen, int32_t const numHeads, int32_t const headDim, int32_t const paddedSeqLen)
{
    int64_t const totalElements = static_cast<int64_t>(batchSize) * seqLen * numHeads * headDim;
    int64_t const inputIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (inputIdx >= totalElements)
    {
        return;
    }

    int32_t const dim = static_cast<int32_t>(inputIdx % headDim);
    int64_t const tmp0 = inputIdx / headDim;
    int32_t const head = static_cast<int32_t>(tmp0 % numHeads);
    int64_t const tmp1 = tmp0 / numHeads;
    int32_t const seq = static_cast<int32_t>(tmp1 % seqLen);
    int32_t const batch = static_cast<int32_t>(tmp1 / seqLen);
    int64_t const outputIdx = (((static_cast<int64_t>(batch) * headDim + dim) * numHeads + head) * paddedSeqLen + seq);

    output[outputIdx] = convertToFp8Byte(__half2float(input[inputIdx]));
}

} // namespace

void launchSageReferenceAttention(half const* q, half const* k, half const* v, half* output, int32_t batchSize,
    int32_t qoLen, int32_t kvLen, int32_t numQoHeads, int32_t numKvHeads, int32_t headDim, bool causal,
    cudaStream_t stream)
{
    int64_t const totalElements = static_cast<int64_t>(batchSize) * qoLen * numQoHeads * headDim;
    int32_t const gridSize = static_cast<int32_t>((totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    sageReferenceAttentionKernel<<<gridSize, kThreadsPerBlock, 0, stream>>>(
        q, k, v, output, batchSize, qoLen, kvLen, numQoHeads, numKvHeads, headDim, causal);
}

void launchQuantizeToInt8PerWarp(half const* input, int8_t* output, float* scales, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headDim, bool isQuery, cudaStream_t stream)
{
    int32_t const ctaLen = isQuery ? kSageCtaQ : kSageCtaK;
    int32_t const warpLen = isQuery ? kSageWarpQ : kSageWarpK;
    int32_t const numWarpsPerBlock = ctaLen / warpLen;
    int32_t const numBlocks = (seqLen + ctaLen - 1) / ctaLen;
    int32_t const gridSize = batchSize * numHeads * numBlocks * numWarpsPerBlock;
    quantizeToInt8PerWarpKernel<<<gridSize, kThreadsPerBlock, 0, stream>>>(
        input, output, scales, seqLen, numHeads, headDim, ctaLen, warpLen, numBlocks, numWarpsPerBlock);
}

void launchQuantizeToFp8Transposed(half const* input, int8_t* output, int32_t batchSize, int32_t seqLen,
    int32_t numHeads, int32_t headDim, int32_t paddedSeqLen, cudaStream_t stream)
{
    size_t const outputSize = static_cast<size_t>(batchSize) * headDim * numHeads * paddedSeqLen * sizeof(int8_t);
    cudaMemsetAsync(output, 0, outputSize, stream);

    int64_t const totalElements = static_cast<int64_t>(batchSize) * seqLen * numHeads * headDim;
    int32_t const gridSize = static_cast<int32_t>((totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    quantizeToFp8TransposedKernel<<<gridSize, kThreadsPerBlock, 0, stream>>>(
        input, output, batchSize, seqLen, numHeads, headDim, paddedSeqLen);
}
