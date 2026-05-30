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

#pragma once

#include <cstdint>

namespace trt_edgellm
{

//! \brief Quantization granularity for SageAttention
enum class SageQuantGranularity
{
    kPerTensor = 0, //!< Per-tensor quantization
    kPerBlock = 1,  //!< Per-block quantization
    kPerWarp = 2,   //!< Per-warp quantization
    kPerThread = 3  //!< Per-thread quantization
};

//! \brief Mask mode for SageAttention
enum class SageMaskMode
{
    kNone = 0,   //!< No mask (non-causal attention)
    kCausal = 1  //!< Causal mask
};

//! \brief Tensor layout for SageAttention inputs
enum class SageTensorLayout
{
    kBSHD = 0, //!< [Batch, Seq, Heads, Dim] layout
    kBHSD = 1  //!< [Batch, Heads, Seq, Dim] layout
};

//! \brief Parameters for SageAttention kernel
//!
//! SageAttention uses INT8 quantized Q/K and FP8 quantized V for efficient
//! inference with minimal accuracy loss. It supports various quantization
//! granularities for flexibility.
struct SageAttentionParams
{
    //! \brief Input tensors (device pointers)
    int8_t* q_ptr{};     //!< Quantized query tensor [B, S, H, D] or [B, H, S, D]
    int8_t* k_ptr{};     //!< Quantized key tensor [B, S, H_kv, D] or [B, H_kv, S, D]
    int8_t* v_ptr{};     //!< Quantized value tensor (FP8 stored as int8_t) [B, S, H_kv, Dv] or [B, H_kv, S, Dv]
    void* o_ptr{};       //!< Output tensor [B, S, H, D] or [B, H, S, D] (FP16 or BF16)

    //! \brief Quantization scales (device pointers)
    float* q_scale_ptr{}; //!< Query quantization scales
    float* k_scale_ptr{}; //!< Key quantization scales
    float* v_scale_ptr{}; //!< Value quantization scales (optional, for fused V scale)
    float* v_mean_ptr{};  //!< Value mean (optional, for fused V mean subtraction)

    //! \brief Output LSE (log-sum-exp) pointer (optional)
    float* lse_ptr{}; //!< Log-sum-exp output [B, H, S] (optional)

    //! \brief Dimensions
    int32_t batch_size{};    //!< Batch size
    int32_t qo_len{};        //!< Query/output sequence length
    int32_t kv_len{};        //!< Key/value sequence length
    int32_t num_qo_heads{};  //!< Number of query/output heads
    int32_t num_kv_heads{};  //!< Number of key/value heads
    int32_t head_dim{};      //!< Head dimension (must be multiple of 64)

    //! \brief Strides (in elements, not bytes)
    int32_t stride_bz_q{};   //!< Batch stride for Q
    int32_t stride_seq_q{};  //!< Sequence stride for Q
    int32_t stride_h_q{};    //!< Head stride for Q
    int32_t stride_bz_k{};   //!< Batch stride for K
    int32_t stride_seq_k{};  //!< Sequence stride for K
    int32_t stride_h_k{};    //!< Head stride for K
    int32_t stride_bz_v{};   //!< Batch stride for V
    int32_t stride_h_v{};    //!< Head stride for V
    int32_t stride_d_v{};    //!< Dimension stride for V (for transposed V layout)
    int32_t stride_bz_o{};   //!< Batch stride for O
    int32_t stride_seq_o{};  //!< Sequence stride for O
    int32_t stride_h_o{};    //!< Head stride for O

    //! \brief Kernel configuration
    float sm_scale{};                          //!< Softmax scale (typically 1/sqrt(head_dim))
    SageTensorLayout tensor_layout{};          //!< Input tensor layout
    SageMaskMode mask_mode{};                  //!< Attention mask mode
    SageQuantGranularity qk_quant_gran{};      //!< Q/K quantization granularity
    bool return_lse{};                         //!< Whether to return log-sum-exp
    bool fuse_v_scale{};                       //!< Whether to fuse V scale
    bool fuse_v_mean{};                        //!< Whether to fuse V mean subtraction
};

//! \brief Configuration for SageAttention runner
struct SageAttentionConfig
{
    int32_t sm_version{};                //!< CUDA compute capability (e.g., 89 for SM 8.9)
    SageQuantGranularity quant_gran{};   //!< Default quantization granularity
    bool use_inst_buffer{};              //!< Use instruction buffer optimization
};

} // namespace trt_edgellm
