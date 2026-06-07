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
#include <cuda_fp8.h>

#include "kernels/sageAttentionKernels/csrc/permuted_smem.cuh"
#include "kernels/sageAttentionKernels/csrc/mma.cuh"
#include "kernels/sageAttentionKernels/csrc/math.cuh"
#include "kernels/sageAttentionKernels/csrc/qattn/attn_utils.cuh"

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
    __nv_fp8_e4m3 const fp8Value(value);
    return *reinterpret_cast<int8_t const*>(&fp8Value);
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

// ===== FP8 roundtrip verification kernels =====

__global__ void fp8RoundtripKernel(float* output) {
    // Test 1: encode 448.0f to __nv_fp8_e4m3 and decode back
    __nv_fp8_e4m3 fp8_448(448.0f);
    int8_t raw = *(int8_t*)&fp8_448;
    float decoded = (float)fp8_448;
    output[0] = decoded;
    output[1] = (float)(int)raw;

    // Test 2: V/vScale scenario: 2.0 / (2.0/448.0)
    __nv_fp8_e4m3 fp8_v(2.0f / (2.0f / 448.0f));
    output[2] = (float)fp8_v;
    output[3] = (float)(int)*(int8_t*)&fp8_v;

    // Test 3: float precision: 2.0 / (2.0/448.0) in float32
    float vScale = 2.0f / 448.0f;  // 0.0044642857...
    float ratio = 2.0f / vScale;    // should be 448.0
    output[4] = ratio;
    output[5] = vScale;

    // Test 4: encode various values near 448
    for (int i = 0; i < 5; i++) {
        float val = 440.0f + i * 4.0f;  // 440, 444, 448, 452, 456
        __nv_fp8_e4m3 fp8_t(val);
        output[6 + i*2] = (float)fp8_t;
        output[7 + i*2] = (float)(int)*(int8_t*)&fp8_t;
    }
}

void launchFp8RoundtripTest(float* output, cudaStream_t stream) {
    fp8RoundtripKernel<<<1, 1, 0, stream>>>(output);
}

// Kernel that stores FP8 V into smem using the same swizzle as the fused kernel,
// then reads back via ldmatrix to check the roundtrip through smem.
__global__ void fp8SmemRoundtripKernel(float* output) {
    using DTypeOut = half;
    constexpr uint32_t CTA_Q = 128, CTA_K = 64, WARP_Q = 32, WARP_K = 64;
    constexpr uint32_t HEAD_DIM = 128;
    constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;  // 4
    constexpr uint32_t num_warps_k = CTA_K / WARP_K;  // 1
    constexpr uint32_t num_warps = num_warps_q * num_warps_k; // 4
    constexpr uint32_t num_tiles_q = WARP_Q / 16;     // 2
    constexpr uint32_t num_tiles_k = WARP_K / 16;     // 4
    constexpr uint32_t num_tiles_v = HEAD_DIM / 16;   // 8

    constexpr SwizzleMode sw_v = SwizzleMode::k64B;
    constexpr uint32_t V_SMEM_STRIDE = CTA_K;          // 64
    constexpr uint32_t stride = V_SMEM_STRIDE / 16;    // 4

    extern __shared__ int8_t smem[];
    smem_t<sw_v, stride> smem_V(smem);

    uint32_t tid = threadIdx.y * 32 + threadIdx.x;
    uint32_t warp_id = threadIdx.y;
    uint32_t lane_id = threadIdx.x;

    // Zero-init smem
    for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += 32 * num_warps) {
        smem[i] = 0;
    }
    __syncthreads();

    // Write known FP8 value (448 = 0x7e) to ALL V positions in swizzled smem
    constexpr uint32_t copy_lines_v = 8;
    constexpr uint32_t line_lanes_v = 4;
    constexpr uint32_t v_smem_col_iters = HEAD_DIM / (num_warps * copy_lines_v); // 128/(4*8)=4
    constexpr uint32_t v_smem_row_iters = V_SMEM_STRIDE / (line_lanes_v * 16);    // 64/(4*16)=1

    uint32_t smem_off = smem_V.get_permuted_offset(
        warp_id * copy_lines_v * v_smem_col_iters + lane_id / line_lanes_v,
        lane_id % line_lanes_v);

    for (uint32_t ci = 0; ci < v_smem_col_iters; ci++) {
        for (uint32_t ri = 0; ri < v_smem_row_iters; ri++) {
            uint32_t dim_group = warp_id * copy_lines_v * v_smem_col_iters + ci * copy_lines_v + ri;
            uint32_t seq_col = lane_id % line_lanes_v;

            uint32_t packed[4];
            for (uint32_t e = 0; e < 4; e++) {
                uint32_t word = 0;
                for (uint32_t b = 0; b < 4; b++) {
                    uint32_t logical_seq = seq_col * 16 + e * 4 + b;
                    // Store known value 448.0f (0x7e in FP8) for first 21 seq positions,
                    // 0 for the rest
                    int8_t f8v;
                    if (logical_seq < 21) {
                        __nv_fp8_e4m3 fp8_v(448.0f);
                        f8v = *(int8_t*)&fp8_v;
                    } else {
                        f8v = 0;
                    }
                    word |= ((uint32_t)(uint8_t)f8v) << (b * 8);
                }
                packed[e] = word;
            }
            smem_V.base[smem_off] = *(uint4*)packed;
            smem_off = smem_V.advance_offset_by_column<line_lanes_v>(smem_off);
        }
        smem_off = smem_V.advance_offset_by_row<copy_lines_v>(
            smem_off - (v_smem_row_iters * line_lanes_v));
    }
    __syncthreads();

    // Set up RS_f8 in registers: match fused kernel's post-softmax layout.
    // For tid=0 (lane_id=0): K_idx_lane_base=0, handles positions 0,1,8,9,16,17,24,25...
    // Positions 0,1,8,9,16,17 are valid (<21), 24,25,... are OOB.
    // Post-softmax: valid RS_exp ≈ 448 (FP8), OOB RS_exp ≈ 0.
    // RS_f8 layout: fk 0..1 (packs 2 logical K tiles per register tile),
    // k 0..3 (4 uint32 words), b 0..3 (4 FP8 values per word).
    constexpr uint32_t K_IDX_BASE = 2 * (0 % 4); // lane_id%4=0 → base=0
    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
        for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++)
            for (uint32_t k = 0; k < 4; k++) {
                uint32_t word = 0;
                for (uint32_t b = 0; b < 4; b++) {
                    uint32_t kv_idx = K_IDX_BASE + fk * 16 + 8 * (k / 4) + k % 2;
                    bool valid = kv_idx < 21; // Only first 21 positions valid
                    __nv_fp8_e4m3 fp8_rs(valid ? 448.0f : 0.0f);
                    word |= ((uint32_t)(uint8_t)*(int8_t*)&fp8_rs) << (b * 8);
                }
                RS_f8[fq][fk][k] = word;
            }

    // RO and d accumulators
    float RO[num_tiles_q][num_tiles_v][8];
    float d[num_tiles_q][2];
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
        for (uint32_t fv = 0; fv < num_tiles_v; fv++)
            for (uint32_t k = 0; k < 8; k++) RO[fq][fv][k] = 0.0f;
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
        d[fq][0] = d[fq][1] = 0.0f;

    // First accumulate denominator (same order as fused kernel)
    accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
    __syncthreads();

    // Run the actual PV matmul (same as fused kernel)
    compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v,
        sw_v, stride, float>(smem_V, RS_f8, RO, d);

    // Output RO[0][0][0] for tid=0, warp=0
    if (tid == 0) {
        output[0] = RO[0][0][0];
        output[1] = RO[0][0][1];
        output[2] = d[0][0];
        output[3] = d[0][1];
    }
}

void launchFp8SmemRoundtripTest(float* output, cudaStream_t stream) {
    constexpr uint32_t smemSize = 64 * 128;  // CTA_K * HEAD_DIM bytes for V smem
    fp8SmemRoundtripKernel<<<dim3(1, 8, 1), dim3(32, 4), smemSize, stream>>>(output);
}
