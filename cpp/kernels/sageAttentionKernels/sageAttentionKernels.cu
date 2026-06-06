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

#include "sageAttentionParams.h"
#include "sageAttentionHostUtils.h"

#include "csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh"
#include "csrc/qattn/qk_int_sv_f8_cuda_sm89_fused.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace trt_edgellm
{
namespace sage
{

namespace
{

template<int32_t HEAD_DIM, bool IS_CAUSAL, int32_t QK_QUANT_GRAN, bool RETURN_LSE, typename DTypeOut, bool FUSE_V_SCALE>
void launchSageAttentionSm89Kernel(SageAttentionParams& params, size_t smemSize, cudaStream_t stream)
{
    constexpr int32_t CTA_Q = 128;
    constexpr int32_t CTA_K = 64;
    constexpr int32_t WARP_Q = 32;
    constexpr int32_t WARP_K = 64;
    constexpr int32_t NUM_WARPS = (CTA_Q / WARP_Q) * (CTA_K / WARP_K);

    static_assert(QK_QUANT_GRAN == static_cast<int32_t>(SageQuantGranularity::kPerWarp)
            || QK_QUANT_GRAN == static_cast<int32_t>(SageQuantGranularity::kPerThread),
        "SageAttention SM89 only supports per-warp or per-thread Q/K quantization");

    constexpr auto quantGran = static_cast<QuantGranularity>(QK_QUANT_GRAN);
    constexpr MaskMode maskMode = IS_CAUSAL ? MaskMode::kCausal : MaskMode::kNone;
    using DTypeSVAccum = float;
    constexpr bool USE_INST_BUFFER = false;
    constexpr bool FUSE_V_MEAN = false;
    constexpr bool USE_PV_FP16_ACCU = false;

    auto kernelFunc = qk_int_sv_f8_attn_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM,
        DataType::kInt8, quantGran, quantGran, DTypeSVAccum, USE_INST_BUFFER, DTypeOut,
        ComputeUnit::kCudaCore, maskMode, RETURN_LSE, FUSE_V_SCALE, FUSE_V_MEAN, USE_PV_FP16_ACCU>;

    cudaError_t attrErr = cudaFuncSetAttribute(
        kernelFunc, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize);
    if (attrErr != cudaSuccess)
    {
        throw std::runtime_error(std::string("cudaFuncSetAttribute failed for SageAttention kernel: ")
            + cudaGetErrorString(attrErr));
    }

    dim3 const gridSize((params.qo_len + CTA_Q - 1) / CTA_Q, params.num_qo_heads, params.batch_size);
    dim3 const blockSize(32, NUM_WARPS);
    uint32_t const numKvGroups = static_cast<uint32_t>(params.num_qo_heads / params.num_kv_heads);

    kernelFunc<<<gridSize, blockSize, smemSize, stream>>>(params.q_ptr, params.k_ptr, params.v_ptr,
        static_cast<DTypeOut*>(params.o_ptr), params.lse_ptr, params.q_scale_ptr, params.k_scale_ptr,
        params.v_scale_ptr, params.v_mean_ptr, params.sequence_lengths, static_cast<uint32_t>(params.qo_len),
        static_cast<uint32_t>(params.kv_len), numKvGroups, static_cast<uint32_t>(params.stride_bz_q),
        static_cast<uint32_t>(params.stride_seq_q), static_cast<uint32_t>(params.stride_h_q),
        static_cast<uint32_t>(params.stride_bz_k), static_cast<uint32_t>(params.stride_seq_k),
        static_cast<uint32_t>(params.stride_h_k), static_cast<uint32_t>(params.stride_bz_v),
        static_cast<uint32_t>(params.stride_h_v), static_cast<uint32_t>(params.stride_d_v),
        static_cast<uint32_t>(params.stride_bz_o), static_cast<uint32_t>(params.stride_seq_o),
        static_cast<uint32_t>(params.stride_h_o), params.sm_scale);
}

template<int32_t HEAD_DIM, bool IS_CAUSAL, int32_t QK_QUANT_GRAN, bool RETURN_LSE, typename DTypeOut>
void launchSageAttentionSm89Dispatched(SageAttentionParams& params, size_t smemSize, cudaStream_t stream)
{
    if (params.fuse_v_scale)
    {
        launchSageAttentionSm89Kernel<HEAD_DIM, IS_CAUSAL, QK_QUANT_GRAN, RETURN_LSE, DTypeOut, true>(
            params, smemSize, stream);
    }
    else
    {
        launchSageAttentionSm89Kernel<HEAD_DIM, IS_CAUSAL, QK_QUANT_GRAN, RETURN_LSE, DTypeOut, false>(
            params, smemSize, stream);
    }
}

} // namespace

template<int32_t HEAD_DIM, bool IS_CAUSAL, int32_t QK_QUANT_GRAN, bool RETURN_LSE, typename DTypeOut>
void launchSageAttentionKernel(
    SageAttentionParams& params,
    int32_t gridSizeX,
    int32_t gridSizeY,
    int32_t gridSizeZ,
    size_t smemSize,
    cudaStream_t stream)
{
    (void) gridSizeX;
    (void) gridSizeY;
    (void) gridSizeZ;
    launchSageAttentionSm89Dispatched<HEAD_DIM, IS_CAUSAL, QK_QUANT_GRAN, RETURN_LSE, DTypeOut>(
        params, smemSize, stream);
}

// Instantiations for head_dim=64, 128, 256 × causal/non-causal × per_warp/per_thread × return_lse/none × FP16
// The fuse_v_scale dispatch happens at runtime inside launchSageAttentionSm89Dispatched

template void launchSageAttentionKernel<64, true, 2, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<64, false, 2, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<64, true, 3, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<64, false, 3, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<64, true, 2, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<64, false, 2, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<64, true, 3, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<64, false, 3, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);

template void launchSageAttentionKernel<128, true, 2, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<128, false, 2, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<128, true, 3, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<128, false, 3, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<128, true, 2, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<128, false, 2, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<128, true, 3, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<128, false, 3, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);

template void launchSageAttentionKernel<256, true, 2, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<256, false, 2, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<256, true, 3, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<256, false, 3, false, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<256, true, 2, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<256, false, 2, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<256, true, 3, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);
template void launchSageAttentionKernel<256, false, 3, true, half>(
    SageAttentionParams&, int32_t, int32_t, int32_t, size_t, cudaStream_t);

// Fused kernel launcher: loads K/V from FP16 KV cache, quantizes in-kernel.
// Only for headDim=128, decode case.
void launchSageAttentionFused(
    int8_t* qPtr, half const* kvCache, half* oPtr,
    float* qScale, float* vScalePerChannel, float* kMeanPerChannel,
    int32_t const* sequenceLengths,
    int32_t batchSize, int32_t numQoHeads, int32_t numKvHeads,
    int32_t kvCacheCapacity,
    int32_t strideBzQ, int32_t strideSeqQ, int32_t strideHQ,
    int32_t strideBzO, int32_t strideSeqO, int32_t strideHO,
    float smScale, cudaStream_t stream)
{
    constexpr int32_t HEAD_DIM = 128;
    constexpr int32_t CTA_Q = 128;
    constexpr int32_t CTA_K = 64;
    constexpr int32_t WARP_Q = 32;
    constexpr int32_t WARP_K = 64;
    constexpr int32_t NUM_WARPS = (CTA_Q / WARP_Q) * (CTA_K / WARP_K); // 4

    auto kernelFunc = sage_attn_fused_kernel<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, half, true>;

    constexpr uint32_t Q_BYTES = CTA_Q * HEAD_DIM;
    constexpr uint32_t K_BYTES = CTA_K * HEAD_DIM;
    constexpr uint32_t V_BYTES = HEAD_DIM * CTA_K;
    constexpr uint32_t TEMP_BYTES = CTA_K * HEAD_DIM * 2;
    size_t smemSize = Q_BYTES + K_BYTES + V_BYTES + TEMP_BYTES;

    cudaError_t attrErr = cudaFuncSetAttribute(kernelFunc,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize);
    if (attrErr != cudaSuccess)
    {
        throw std::runtime_error(std::string("cudaFuncSetAttribute failed for fused SageAttention: ")
            + cudaGetErrorString(attrErr));
    }

    dim3 grid(1, numQoHeads, batchSize);
    dim3 block(32, NUM_WARPS);
    uint32_t numKvGroups = static_cast<uint32_t>(numQoHeads / numKvHeads);

    kernelFunc<<<grid, block, smemSize, stream>>>(
        qPtr, kvCache, oPtr, qScale, vScalePerChannel, kMeanPerChannel, sequenceLengths,
        static_cast<uint32_t>(1), static_cast<uint32_t>(kvCacheCapacity), numKvGroups,
        static_cast<uint32_t>(kvCacheCapacity),
        static_cast<uint32_t>(strideBzQ), static_cast<uint32_t>(strideSeqQ), static_cast<uint32_t>(strideHQ),
        static_cast<uint32_t>(strideBzO), static_cast<uint32_t>(strideSeqO), static_cast<uint32_t>(strideHO),
        smScale);
}

} // namespace sage
} // namespace trt_edgellm
