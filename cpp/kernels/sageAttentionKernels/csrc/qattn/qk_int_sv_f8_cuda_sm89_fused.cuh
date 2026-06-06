/*
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused SageAttention kernel for decode: loads FP16 K/V directly from KV cache,
 * quantizes INT8/FP8 in shared memory. Eliminates all intermediate quantized buffers.
 * Only supports headDim=128, non-causal, qoLen=1 (single-token decode).
 */

// This kernel is designed for SM89 (Ada Lovelace).
#include "attn_utils.cuh"
#include "../permuted_smem.cuh"
#include "../numeric_conversion.cuh"
#include "../mma.cuh"
#include "../math.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t HEAD_DIM,
         typename DTypeOut = half, bool FUSE_V_SCALE = true>
__global__ void sage_attn_fused_kernel(
    int8_t *__restrict__ Q,               // pre-quantized Q INT8 [B,1,Hq,D]
    half const *__restrict__ kv_cache,     // FP16 KV cache [B,2,Hkv,capacity,D]
    DTypeOut *__restrict__ O,              // output FP16 [B,1,Hq,D]
    float *__restrict__ Q_scale,           // Q per-warp scales
    float *__restrict__ V_scale_ch,        // per-channel V scales [B*Hkv*D]
    float *__restrict__ k_mean,            // per-channel K means [B*Hkv*D] for K-mean centering
    const int32_t *__restrict__ seq_lens,  // per-batch effective KV lengths
    const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
    const uint32_t kv_cache_capacity,       // capacity stride in KV cache
    const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
    const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
    float sm_scale)
{
    constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
    constexpr uint32_t num_warps_k = CTA_K / WARP_K;
    constexpr uint32_t num_warps = num_warps_q * num_warps_k;
    constexpr uint32_t num_tiles_q = WARP_Q / 16;
    constexpr uint32_t num_tiles_k = WARP_K / 16;
    constexpr uint32_t num_tiles_qk_inner = HEAD_DIM / 32;
    constexpr uint32_t num_tiles_v = HEAD_DIM / 16;
    constexpr uint32_t NUM_THREADS = 32 * num_warps;

    constexpr uint32_t QK_SMEM_STRIDE = HEAD_DIM;
    constexpr uint32_t V_SMEM_STRIDE = CTA_K;
    constexpr uint32_t O_SMEM_STRIDE = HEAD_DIM;

    // Smem regions (bytes)
    constexpr uint32_t Q_BYTES = CTA_Q * QK_SMEM_STRIDE;
    constexpr uint32_t K_OFFSET = Q_BYTES;
    constexpr uint32_t K_BYTES = CTA_K * QK_SMEM_STRIDE;
    constexpr uint32_t V_OFFSET = K_OFFSET + K_BYTES;
    constexpr uint32_t V_BYTES = HEAD_DIM * V_SMEM_STRIDE;
    constexpr uint32_t TEMP_OFFSET = V_OFFSET + V_BYTES;

    extern __shared__ int8_t smem[];

    uint32_t tid = threadIdx.y * 32 + threadIdx.x;
    uint32_t lane_id = threadIdx.x;
    uint32_t warp_id = threadIdx.y;
    uint32_t batch_id = blockIdx.z;
    uint32_t head_id = blockIdx.y;
    uint32_t num_qo_heads = gridDim.y;

    uint32_t effective_kv_len = (seq_lens != nullptr)
        ? min((uint32_t)seq_lens[batch_id], kv_len) : kv_len;

    sm_scale *= math::log2e;

    // Swizzled smem wrappers
    constexpr SwizzleMode sw_qk = SwizzleMode::k128B; // HEAD_DIM=128 -> k128B
    smem_t<sw_qk, QK_SMEM_STRIDE / 16> smem_Q(smem);
    smem_t<sw_qk, QK_SMEM_STRIDE / 16> smem_K(smem + K_OFFSET);
    constexpr SwizzleMode sw_v = SwizzleMode::k64B; // CTA_K=64 -> k64B
    smem_t<sw_v, V_SMEM_STRIDE / 16> smem_V(smem + V_OFFSET);

    half* temp = reinterpret_cast<half*>(smem + TEMP_OFFSET);

    // Parameters for lane-to-element mapping
    constexpr uint32_t line_lanes_qk = 8; // HEAD_DIM=128 -> 8
    constexpr uint32_t copy_lines_qk = 4; // 32/8=4
    constexpr uint32_t line_lanes_v = 4;
    constexpr uint32_t copy_lines_v = 8;
    constexpr uint32_t q_smem_col_iters = CTA_Q / (num_warps * copy_lines_qk);
    constexpr uint32_t k_smem_col_iters = CTA_K / (num_warps * copy_lines_qk);
    constexpr uint32_t v_smem_col_iters = HEAD_DIM / (num_warps * copy_lines_v);
    constexpr uint32_t qk_smem_row_iters = QK_SMEM_STRIDE / (line_lanes_qk * 16);
    constexpr uint32_t v_smem_row_iters = V_SMEM_STRIDE / (line_lanes_v * 16);

    // Q lane pointer and smem offsets (same as original)
    int8_t *Q_lane = Q + batch_id * stride_bz_q + head_id * stride_h_q
        + (CTA_Q / num_warps * warp_id + lane_id / line_lanes_qk) * stride_seq_q
        + (lane_id % line_lanes_qk) * 16;

    uint32_t Q_smem_off_load = smem_Q.get_permuted_offset(
        warp_id * copy_lines_qk * q_smem_col_iters + lane_id / line_lanes_qk,
        lane_id % line_lanes_qk);
    uint32_t K_smem_offset_load = smem_K.get_permuted_offset(
        warp_id * copy_lines_qk * k_smem_col_iters + lane_id / line_lanes_qk,
        lane_id % line_lanes_qk);
    uint32_t V_smem_offset_load = smem_V.get_permuted_offset(
        warp_id * copy_lines_v * v_smem_col_iters + lane_id / line_lanes_v,
        lane_id % line_lanes_v);

    // Load Q (same as original)
    load_global_to_share<line_lanes_qk, copy_lines_qk, 1, q_smem_col_iters,
        sw_qk, QK_SMEM_STRIDE / 16, CTA_Q>(
        &Q_lane, Q_smem_off_load, stride_seq_q, smem_Q, 0u, qo_len);
    cp_async::commit_group(); cp_async::wait_group<0>(); __syncthreads();

    uint32_t Q_mma_off = smem_Q.get_permuted_offset(
        get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
    uint32_t K_mma_off = smem_K.get_permuted_offset(
        get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8,
        (lane_id / 8) % 2);

    // Preload Q to registers (num_tiles_qk_inner=4 for headDim=128)
    uint32_t RQ[num_tiles_q][4];
    uint32_t QK_off_Q = Q_mma_off;
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        smem_Q.ldmatrix_m8n8x4(QK_off_Q, RQ[fq]);
        QK_off_Q = smem_Q.advance_offset_by_row<16>(QK_off_Q);
    }

    // KV cache addressing
    uint32_t kv_head = head_id / num_kv_groups;
    uint32_t numKVHeads = num_qo_heads / num_kv_groups;
    uint64_t kv_k_base = (uint64_t)batch_id * 2 * numKVHeads * kv_cache_capacity * HEAD_DIM
        + (uint64_t)kv_head * kv_cache_capacity * HEAD_DIM;
    uint64_t kv_v_base = kv_k_base + (uint64_t)numKVHeads * kv_cache_capacity * HEAD_DIM;

    // Q scale index
    uint32_t q_scale_idx = batch_id * num_qo_heads * (1 * num_warps_q)
        + head_id * (1 * num_warps_q) + get_warp_idx_q<num_warps_q, num_warps_k>();

    float q_scale = Q_scale[q_scale_idx];
    float orig_sm = sm_scale;

    // Registers
    int32_t RS[num_tiles_q][num_tiles_k][8];
    float RO[num_tiles_q][num_tiles_v][8];
    float m[num_tiles_q][2], d[num_tiles_q][2];

    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
        for (uint32_t fv = 0; fv < num_tiles_v; fv++)
            for (uint32_t k = 0; k < 8; k++) RO[fq][fv][k] = 0.0f;
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
        { m[fq][0] = m[fq][1] = -5000000.0f; d[fq][0] = d[fq][1] = 1.0f; }

    uint32_t num_iter = div_ceil(effective_kv_len, CTA_K);

    // ===== MAIN LOOP =====
    for (uint32_t iter = 0; iter < num_iter; iter++)
    {
        uint32_t seq_start = iter * CTA_K;
        uint32_t num_valid = min(CTA_K, effective_kv_len - seq_start);

        // --- Load FP16 K from KV cache → linear temp smem ---
        for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += NUM_THREADS)
        {
            uint32_t seq_in_tile = i / HEAD_DIM;
            uint32_t dim = i % HEAD_DIM;
            uint32_t seq = seq_start + seq_in_tile;
            bool valid = seq < effective_kv_len;
            temp[i] = valid ? kv_cache[kv_k_base + (uint64_t)seq * HEAD_DIM + dim]
                            : __float2half(0.0f);
        }
        __syncthreads();

        // --- Compute per-tile K scale ---
        __shared__ float sKMax;
        if (tid == 0) sKMax = 0.0f;
        __syncthreads();

        float threadKMax = 0.0f;
        for (uint32_t i = tid; i < num_valid * HEAD_DIM; i += NUM_THREADS)
        {
            float v = __half2float(temp[i]);
            threadKMax = fmaxf(threadKMax, fabsf(v));
        }

        __shared__ float sKRed[NUM_THREADS];
        sKRed[tid] = threadKMax;
        __syncthreads();
        for (uint32_t s = NUM_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) sKRed[tid] = fmaxf(sKRed[tid], sKRed[tid + s]);
            __syncthreads();
        }
        float kScale = fmaxf(sKRed[0] / 127.0f, 1.0e-6f);

        // --- Quantize K: temp FP16 → swizzled K smem INT8 ---
        // Must match load_global_to_share lane-to-element mapping exactly
        {
            uint32_t smem_off = K_smem_offset_load;
            for (uint32_t ci = 0; ci < k_smem_col_iters; ci++)
            {
                for (uint32_t ri = 0; ri < qk_smem_row_iters; ri++)
                {
                    uint32_t row = warp_id * copy_lines_qk * k_smem_col_iters + ci * copy_lines_qk + ri;
                    uint32_t col = lane_id % line_lanes_qk;
                    uint32_t dim0 = col * 16;

                    uint32_t packed[4];
                    for (uint32_t e = 0; e < 4; e++) {
                        uint32_t word = 0;
                        for (uint32_t b = 0; b < 4; b++) {
                            uint32_t dim = dim0 + e * 4 + b;
                            int8_t qi = 0;
                            if (row < CTA_K && dim < HEAD_DIM) {
                                float fv = __half2float(temp[row * HEAD_DIM + dim]);
                                // K-mean centering: subtract per-channel mean (smooth_k)
                                float km = (k_mean != nullptr)
                                    ? k_mean[(uint64_t)batch_id * numKVHeads * HEAD_DIM + (uint64_t)kv_head * HEAD_DIM + dim]
                                    : 0.0f;
                                qi = (int8_t)min(127.0f, max(-127.0f, nearbyintf((fv - km) / kScale)));
                            }
                            word |= ((uint32_t)(uint8_t)qi) << (b * 8);
                        }
                        packed[e] = word;
                    }
                    smem_K.base[smem_off] = *(uint4*)packed;
                    smem_off = smem_K.advance_offset_by_column<line_lanes_qk>(smem_off);
                }
                smem_off = smem_K.advance_offset_by_row<copy_lines_qk>(smem_off - (qk_smem_row_iters * line_lanes_qk));
            }
        }
        __syncthreads();

        float dequant_scale = q_scale * kScale;
        sm_scale = orig_sm * dequant_scale;

        // --- Load FP16 V from KV cache → same temp smem ---
        for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += NUM_THREADS)
        {
            uint32_t seq_in_tile = i / HEAD_DIM;
            uint32_t dim = i % HEAD_DIM;
            uint32_t seq = seq_start + seq_in_tile;
            bool valid = seq < effective_kv_len;
            temp[i] = valid ? kv_cache[kv_v_base + (uint64_t)seq * HEAD_DIM + dim]
                            : __float2half(0.0f);
        }
        __syncthreads();

        // --- Quantize V: temp FP16 → swizzled V smem FP8 ---
        // V per-channel scales: [B, numKVHeads, D]. numKVHeads = num_qo_heads / num_kv_groups
        uint32_t actualNumKvHeads = num_qo_heads / num_kv_groups;
        const float* v_scale_head = V_scale_ch
            + (uint64_t)batch_id * actualNumKvHeads * HEAD_DIM
            + (uint64_t)kv_head * HEAD_DIM;
        // Match load_fp8_V_global_to_share lane mapping exactly
        {
            uint32_t smem_off = V_smem_offset_load;
            for (uint32_t ci = 0; ci < v_smem_col_iters; ci++)
            {
                for (uint32_t ri = 0; ri < v_smem_row_iters; ri++)
                {
                    uint32_t dim_group = warp_id * copy_lines_v * v_smem_col_iters + ci * copy_lines_v + ri;
                    uint32_t seq_col = lane_id % line_lanes_v;

                    uint32_t packed[4];
                    for (uint32_t e = 0; e < 4; e++) {
                        uint32_t word = 0;
                        for (uint32_t b = 0; b < 4; b++) {
                            uint32_t logical_seq = seq_col * 16 + e * 4 + b;
                            // Apply SageAttention V sequence permutation to match RS_32_to_8 order
                            // [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]
                            uint32_t mod16 = logical_seq % 16;
                            uint32_t ps = (mod16 / 8) * 2 + ((mod16 / 2) % 4) * 4 + (mod16 % 2);
                            uint32_t seq_in_tile = (logical_seq / 16) * 16 + ps;
                            int8_t f8v = 0;
                            if (dim_group < HEAD_DIM && seq_in_tile < CTA_K) {
                                float fv = __half2float(temp[seq_in_tile * HEAD_DIM + dim_group]);
                                float vs = v_scale_head[dim_group];
                                __nv_fp8_e4m3 fp8(fv / vs);
                                f8v = *(int8_t*)&fp8;
                            }
                            word |= ((uint32_t)(uint8_t)f8v) << (b * 8);
                        }
                        packed[e] = word;
                    }
                    smem_V.base[smem_off] = *(uint4*)packed;
                    smem_off = smem_V.advance_offset_by_column<line_lanes_v>(smem_off);
                }
                smem_off = smem_V.advance_offset_by_row<copy_lines_v>(smem_off - (v_smem_row_iters * line_lanes_v));
            }
        }
        __syncthreads();

        // --- QK Matmul (same as original) ---
        {
            uint32_t QK_off_K = K_mma_off;
            compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner,
                sw_qk, QK_SMEM_STRIDE / 16, DataType::kInt8>(
                smem_Q, smem_K, RS, Q_mma_off, QK_off_K);
        }

        // --- Convert int32 → float with dequant ---
        float RS_f32[num_tiles_q][num_tiles_k][8];
        for (uint32_t fq = 0; fq < num_tiles_q; fq++)
            for (uint32_t fk = 0; fk < num_tiles_k; fk++)
                for (uint32_t k = 0; k < 8; k++)
                    RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;

        // --- Out-of-bound mask for last tile ---
        if (num_valid < CTA_K)
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k, float>(
                get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4),
                RS_f32, num_valid);

        // --- Softmax ---
        update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false, float>(
            RS_f32, RO, m, d, orig_sm);

        // --- RS float → FP8, accumulate denominator ---
        uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
        RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
        accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);

        __syncthreads();

        // --- PV Matmul ---
        compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v,
            sw_v, V_SMEM_STRIDE / 16, float>(smem_V, RS_f8, RO, d);
        __syncthreads();
    }

    // === Normalize ===
    normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kTensorCore, float, float>(RO, m, d);

    // === fuse_v_scale ===
    if constexpr (FUSE_V_SCALE)
    {
        uint32_t actualNumKvHeads = num_qo_heads / num_kv_groups;
        float v_scale_reg[4];
        float *vs_ptr = V_scale_ch + batch_id * actualNumKvHeads * HEAD_DIM
            + (head_id / num_kv_groups) * HEAD_DIM + (lane_id % 4) * 2;
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            ((float2*)v_scale_reg)[0] = *((float2*)(vs_ptr + fv * 16));
            ((float2*)v_scale_reg)[1] = *((float2*)(vs_ptr + fv * 16 + 8));
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
                RO[fq][fv][0] *= v_scale_reg[0]; RO[fq][fv][1] *= v_scale_reg[1];
                RO[fq][fv][2] *= v_scale_reg[0]; RO[fq][fv][3] *= v_scale_reg[1];
                RO[fq][fv][4] *= v_scale_reg[2]; RO[fq][fv][5] *= v_scale_reg[3];
                RO[fq][fv][6] *= v_scale_reg[2]; RO[fq][fv][7] *= v_scale_reg[3];
            }
        }
    }

    // === Output: RO float → smem_O half → global O ===
    constexpr SwizzleMode sw_o = SwizzleMode::k128B;
    smem_t<sw_o, O_SMEM_STRIDE / 8> smem_O(smem); // reuse Q smem
    constexpr uint32_t line_lanes_o = 8;
    constexpr uint32_t copy_lines_o = 4;
    constexpr uint32_t o_smem_row_iters = O_SMEM_STRIDE / (line_lanes_o * 8);
    constexpr uint32_t o_smem_col_iters = CTA_Q / (num_warps * copy_lines_o);

    // Store RO to smem_O
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
            uint32_t sw_off = smem_O.get_permuted_offset(
                get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + fq * 16 + lane_id % 16,
                lane_id / 16 + fv * 2);
            half packed[8];
            for (uint32_t k = 0; k < 8; k++) packed[k] = __float2half_rn(RO[fq][fv][k]);
            smem_O.base[sw_off] = *(uint4*)packed;
        }
    }
    __syncthreads();

    // Copy smem_O → global O
    DTypeOut *O_lane = O + batch_id * stride_bz_o + head_id * stride_h_o
        + (CTA_Q / num_warps * warp_id + lane_id / line_lanes_o) * stride_seq_o
        + (lane_id % line_lanes_o) * 8;

    for (uint32_t ci = 0; ci < o_smem_col_iters; ci++) {
        for (uint32_t ri = 0; ri < o_smem_row_iters; ri++) {
            uint32_t sw_off = smem_O.get_permuted_offset(
                warp_id * copy_lines_o * o_smem_col_iters + ci * copy_lines_o + ri,
                lane_id % line_lanes_o);
            *(uint4*)O_lane = smem_O.base[sw_off];
            O_lane += line_lanes_o * 8;
        }
        O_lane += (copy_lines_o * stride_seq_o) - (o_smem_row_iters * line_lanes_o * 8);
    }
}
