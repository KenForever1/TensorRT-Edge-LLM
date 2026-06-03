# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test comparing C++ SageAttention runtime path vs original SageAttention reference.

This test isolates the decode-path bug by:
1. Running original SageAttention (sageattn_qk_int8_pv_fp8_cuda) as reference
2. Reproducing the C++ runtime's quantization path in Python (no V permutation, no V scale)
3. Calling the SAME underlying SM89 kernel with the C++ runtime's quantized tensors
4. Comparing outputs to identify where the divergence occurs

The C++ runtime differs from original SageAttention in V quantization:
- Original: per_channel_fp8() -> transpose + pad + PERMUTE + per-channel SCALE -> kernel with FUSE_V_SCALE=True
- C++ runtime: simple FP16->FP8 conversion per element, no permute, no scale -> kernel with FUSE_V_SCALE=False

This test verifies whether the kernel produces correct results when V is quantized
the way the C++ runtime does it (without permutation and without per-channel scaling).
"""

import sys
import os
import pytest
import torch
import numpy as np


def requires_sm89():
    """Check if current GPU is SM89 (Ada Lovelace)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return major * 10 + minor >= 89


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not requires_sm89(), reason="Requires SM89+ GPU")
class TestSageAttentionVsReference:
    """Compare C++ runtime SageAttention path against original SageAttention."""

    @staticmethod
    def _cpp_runtime_quantize_v(v_fp16, num_kv_heads, kv_len, head_dim, padded_kv_len):
        """
        Reproduce the C++ runtime's V quantization in Python.

        The C++ runtime (convertKvCacheToSageKernel) does:
        - Simple FP16 -> FP8 conversion per element (no scaling)
        - Layout: v_fp8[batch * headDim + dim][numHeads * paddedKvLen + head * paddedKvLen + seq]
          i.e., [B, D, Hkv, paddedKvLen]
        - Zero-padding for seq positions >= kvLen

        Input v_fp16: [B, Hkv, kvLen, D] (HND layout)
        Output: [B, D, Hkv, paddedKvLen] as float8_e4m3fn
        """
        batch_size = v_fp16.shape[0]
        # Create output in the transposed layout [B, D, Hkv, paddedKvLen]
        v_fp8 = torch.zeros(
            (batch_size, head_dim, num_kv_heads, padded_kv_len),
            dtype=torch.float8_e4m3fn,
            device=v_fp16.device
        )
        # Fill in values: simple element-wise FP16->FP8 cast, no scaling
        for b in range(batch_size):
            for h in range(num_kv_heads):
                for s in range(kv_len):
                    for d in range(head_dim):
                        v_fp8[b, d, h, s] = v_fp16[b, h, s, d].to(torch.float8_e4m3fn)
        return v_fp8

    @staticmethod
    def _cpp_runtime_quantize_v_fast(v_fp16, padded_kv_len):
        """
        Fast vectorized version of C++ runtime V quantization.

        Input v_fp16: [B, Hkv, kvLen, D] (HND layout)
        Output: [B, D, Hkv, paddedKvLen] as float8_e4m3fn
        """
        batch_size, num_kv_heads, kv_len, head_dim = v_fp16.shape
        # Transpose to [B, D, Hkv, kvLen]
        v_transposed = v_fp16.permute(0, 3, 1, 2).contiguous()
        # Pad seq dim to paddedKvLen
        if padded_kv_len > kv_len:
            v_padded = torch.nn.functional.pad(
                v_transposed, (0, padded_kv_len - kv_len), value=0.0
            )
        else:
            v_padded = v_transposed
        # Convert to FP8 directly (no per-channel scaling!)
        v_fp8 = v_padded.to(torch.float8_e4m3fn)
        return v_fp8

    def test_original_sage_attention_decode(self):
        """
        Run original SageAttention on decode scenario and verify it produces
        reasonable output (not attenuated).
        """
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        torch.manual_seed(42)
        batch_size = 1
        num_qo_heads = 16
        num_kv_heads = 8
        head_dim = 128
        qo_len = 1
        kv_len = 21

        q = torch.randn(batch_size, num_qo_heads, qo_len, head_dim,
                        dtype=torch.float16, device="cuda")
        k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                        dtype=torch.float16, device="cuda")
        v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                        dtype=torch.float16, device="cuda")

        # Original SageAttention with per_warp quantization (matching C++ runtime config)
        output = sageattn_qk_int8_pv_fp8_cuda(
            q, k, v,
            tensor_layout="HND",
            is_causal=False,
            qk_quant_gran="per_warp",
            sm_scale=None,
            pv_accum_dtype="fp32",
            smooth_k=False,
            smooth_v=False,
            return_lse=False,
        )

        # Compute FP16 reference with standard attention
        scale = head_dim ** -0.5
        # Expand K/V for GQA
        k_expanded = k.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
        v_expanded = v.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        ref_output = torch.matmul(attn_weights, v_expanded)

        # SageAttention output should be close to FP16 reference
        # (within quantization tolerance, not 5-30x attenuated)
        output_norm = output.float().norm()
        ref_norm = ref_output.float().norm()
        ratio = output_norm / ref_norm

        print(f"\n[Original SageAttention] output norm: {output_norm:.4f}")
        print(f"[FP16 Reference]         output norm: {ref_norm:.4f}")
        print(f"[Ratio]                  sage/ref:    {ratio:.4f}")

        # The ratio should be close to 1.0 (within ~20% for FP8 quantization)
        assert 0.5 < ratio < 2.0, (
            f"Original SageAttention output is severely off: ratio={ratio:.4f}"
        )

    def test_cpp_runtime_v_quantization_vs_original(self):
        """
        Compare V quantization between C++ runtime path and original SageAttention.

        The C++ runtime does NOT apply:
        1. Per-channel scaling (v_scale)
        2. Sequence permutation [0,1,8,9,2,3,10,11,...]

        This test shows whether these differences cause the output attenuation.
        """
        from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
        from sageattention.quant import per_channel_fp8

        torch.manual_seed(42)
        batch_size = 1
        num_qo_heads = 16
        num_kv_heads = 8
        head_dim = 128
        qo_len = 1
        kv_len = 21
        CTA_K = 64
        padded_kv_len = ((kv_len + CTA_K - 1) // CTA_K) * CTA_K  # 64

        q = torch.randn(batch_size, num_qo_heads, qo_len, head_dim,
                        dtype=torch.float16, device="cuda")
        k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                        dtype=torch.float16, device="cuda")
        v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                        dtype=torch.float16, device="cuda")

        # --- Original SageAttention quantization path ---
        # Q/K: per-warp INT8 quantization
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(
            q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, tensor_layout="HND"
        )
        # V: per-channel FP8 with permutation (scale_max=448.0 for fp32 accum)
        v_fp8_original, v_scale_original, _ = per_channel_fp8(
            v, tensor_layout="HND", scale_max=448.0, smooth_v=False
        )

        # --- C++ runtime quantization path ---
        # V: simple transpose + FP8 cast, NO permutation, NO per-channel scale
        v_fp8_cpp = self._cpp_runtime_quantize_v_fast(v, padded_kv_len)

        # Compare V tensors
        # Original v_fp8 shape: [B, Hkv, D, paddedKvLen] (HND per_channel_fp8 output)
        # C++ v_fp8 shape: [B, D, Hkv, paddedKvLen]
        print(f"\n[V quantization comparison]")
        print(f"  Original v_fp8 shape: {v_fp8_original.shape}")
        print(f"  C++ runtime v_fp8 shape: {v_fp8_cpp.shape}")
        print(f"  Original v_scale shape: {v_scale_original.shape}")
        print(f"  v_scale range: [{v_scale_original.min():.6f}, {v_scale_original.max():.6f}]")

        # Check if the raw values match (ignoring permutation and scaling)
        # Original does per-channel scaling: v_fp8_original = v / v_scale * 448
        # C++ does: v_fp8_cpp = cast_to_fp8(v)
        # These CANNOT match because the original scales V values before FP8 cast
        v_cpp_float = v_fp8_cpp.float()  # [B, D, Hkv, paddedKvLen]
        v_orig_float = v_fp8_original.float()  # [B, Hkv, D, paddedKvLen]
        # Transpose original to match C++ layout for comparison
        v_orig_transposed = v_orig_float.permute(0, 2, 1, 3)  # [B, D, Hkv, paddedKvLen]

        # At kv_len positions (0..20), compare actual FP8 values
        max_diff = (v_cpp_float[:, :, :, :kv_len] - v_orig_transposed[:, :, :, :kv_len]).abs().max()
        print(f"  Max diff between C++ and original FP8 values: {max_diff:.4f}")
        print(f"  (Expected to differ due to per-channel scaling in original)")

    def test_cpp_runtime_kernel_call_vs_original(self):
        """
        Call the SM89 kernel with BOTH quantization paths and compare outputs.

        This is the definitive test: same kernel, different V preparation.
        - Path A (original): per_channel_fp8 V (permuted+scaled) + fuse_v_scale kernel
        - Path B (C++ runtime): simple FP8 V (transposed, no permute, no scale) + non-fuse kernel

        The kernel binding expects V shape [B, Hkv, head_dim, paddedKvLen] for HND layout.
        """
        from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
        from sageattention.quant import per_channel_fp8
        from sageattention import sm89_compile

        torch.manual_seed(42)
        batch_size = 1
        num_qo_heads = 16
        num_kv_heads = 8
        head_dim = 128
        qo_len = 1
        kv_len = 21
        CTA_K = 64
        padded_kv_len = ((kv_len + CTA_K - 1) // CTA_K) * CTA_K  # 64
        sm_scale = head_dim ** -0.5

        q = torch.randn(batch_size, num_qo_heads, qo_len, head_dim,
                        dtype=torch.float16, device="cuda")
        k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                        dtype=torch.float16, device="cuda")
        v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                        dtype=torch.float16, device="cuda")

        # Shared Q/K quantization (per-warp)
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(
            q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, tensor_layout="HND"
        )

        _tensor_layout = 1  # HND
        _is_causal = 0
        _qk_quant_gran = 2  # per_warp
        _return_lse = 0

        # --- Path A: Original SageAttention (with V scale + permutation) ---
        # per_channel_fp8 produces [B, Hkv, head_dim, paddedKvLen] with permutation+scaling
        v_fp8_orig, v_scale_orig, _ = per_channel_fp8(
            v, tensor_layout="HND", scale_max=448.0, smooth_v=False
        )
        o_orig = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse_a = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
            q_int8, k_int8, v_fp8_orig, o_orig,
            q_scale, k_scale, v_scale_orig,
            _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse
        )

        # --- Path B: C++ runtime path (no V scale, no permutation) ---
        # C++ runtime produces V in [B, D, Hkv, paddedKvLen] layout
        # Kernel binding expects [B, Hkv, D, paddedKvLen], so we permute to match
        v_fp8_cpp_raw = self._cpp_runtime_quantize_v_fast(v, padded_kv_len)
        # Rearrange from [B, D, Hkv, paddedKvLen] -> [B, Hkv, D, paddedKvLen]
        v_fp8_cpp = v_fp8_cpp_raw.permute(0, 2, 1, 3).contiguous()

        # sm89_compile only exposes fuse_v_scale variants, so we pass v_scale=1.0
        # to simulate "no scaling" (C++ runtime doesn't scale V)
        v_scale_ones = torch.ones(
            (batch_size, num_kv_heads, head_dim),
            dtype=torch.float32, device="cuda"
        )

        o_cpp = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        # Use fuse_v_scale kernel with v_scale=1.0 (effectively no scaling)
        lse_b = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
            q_int8, k_int8, v_fp8_cpp, o_cpp,
            q_scale, k_scale, v_scale_ones,
            _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse
        )

        # Compare outputs
        o_orig_norm = o_orig.float().norm()
        o_cpp_norm = o_cpp.float().norm()
        ratio = o_cpp_norm / o_orig_norm

        print(f"\n[Kernel output comparison]")
        print(f"  Original (with v_scale+permute) output norm: {o_orig_norm:.4f}")
        print(f"  C++ runtime (no v_scale, no permute) output norm: {o_cpp_norm:.4f}")
        print(f"  Ratio (cpp/original): {ratio:.4f}")

        # Also compare both against FP16 reference
        scale = head_dim ** -0.5
        k_expanded = k.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
        v_expanded = v.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        ref_out = torch.matmul(attn_weights, v_expanded)
        ref_norm = ref_out.float().norm()

        print(f"  FP16 reference output norm: {ref_norm:.4f}")
        print(f"  Original vs FP16 ratio: {o_orig_norm / ref_norm:.4f}")
        print(f"  C++ runtime vs FP16 ratio: {o_cpp_norm / ref_norm:.4f}")

        max_abs_diff = (o_orig.float() - o_cpp.float()).abs().max()
        print(f"  Max absolute difference (orig vs cpp): {max_abs_diff:.6f}")

        # If ratio is severely off (5-30x attenuation), this confirms the bug
        if ratio < 0.5 or ratio > 2.0:
            print(f"  *** BUG CONFIRMED: C++ runtime V path produces "
                  f"{1.0/ratio:.1f}x attenuated output ***")
        else:
            print(f"  Outputs are within acceptable range")

    def test_v_permutation_effect(self):
        """
        Test the effect of the sequence permutation on kernel output.

        Original SageAttention applies permutation [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]
        to the seq dim after transposing V. The C++ runtime skips this permutation.

        This test shows how much the permutation matters for correctness.
        """
        from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
        from sageattention.quant import per_channel_fp8
        from sageattention import sm89_compile

        torch.manual_seed(42)
        batch_size = 1
        num_qo_heads = 16
        num_kv_heads = 8
        head_dim = 128
        qo_len = 1
        kv_len = 21
        CTA_K = 64
        padded_kv_len = ((kv_len + CTA_K - 1) // CTA_K) * CTA_K
        sm_scale = head_dim ** -0.5

        q = torch.randn(batch_size, num_qo_heads, qo_len, head_dim,
                        dtype=torch.float16, device="cuda")
        k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                        dtype=torch.float16, device="cuda")
        v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                        dtype=torch.float16, device="cuda")

        # Shared Q/K quantization
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(
            q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, tensor_layout="HND"
        )

        # --- Path A: WITH permutation (original) ---
        v_fp8_with_permute, v_scale, _ = per_channel_fp8(
            v, tensor_layout="HND", scale_max=448.0, smooth_v=False
        )
        o_with_permute = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse_a = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
            q_int8, k_int8, v_fp8_with_permute, o_with_permute,
            q_scale, k_scale, v_scale,
            1, 0, 2, sm_scale, 0
        )

        # --- Path B: WITHOUT permutation but WITH per-channel scaling ---
        # Manually transpose V without using transpose_pad_permute_cuda
        # This simulates what happens if C++ runtime adds scaling but skips permutation
        v_transposed = v.permute(0, 1, 3, 2).contiguous()  # [B, Hkv, D, kvLen]
        # Pad to padded_kv_len
        v_padded = torch.nn.functional.pad(
            v_transposed, (0, padded_kv_len - kv_len), value=0.0
        )  # [B, Hkv, D, paddedKvLen]

        # Apply per-channel scaling (same as per_channel_fp8 but without permute)
        v_abs_max = v_padded[:, :, :, :kv_len].float().abs().amax(dim=-1)  # [B, Hkv, D]
        v_scale_no_permute = v_abs_max / 448.0
        v_scale_no_permute = v_scale_no_permute.clamp(min=1e-6)
        v_scaled = v_padded.float() / v_scale_no_permute.unsqueeze(-1)
        v_fp8_no_permute = v_scaled.to(torch.float8_e4m3fn)

        o_no_permute = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse_b = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
            q_int8, k_int8, v_fp8_no_permute, o_no_permute,
            q_scale, k_scale, v_scale_no_permute,
            1, 0, 2, sm_scale, 0
        )

        # Compare
        norm_a = o_with_permute.float().norm()
        norm_b = o_no_permute.float().norm()
        ratio = norm_b / norm_a
        max_diff = (o_with_permute.float() - o_no_permute.float()).abs().max()

        print(f"\n[Permutation effect test]")
        print(f"  With permutation output norm:    {norm_a:.4f}")
        print(f"  Without permutation output norm: {norm_b:.4f}")
        print(f"  Ratio (no_permute/permute):      {ratio:.4f}")
        print(f"  Max absolute difference:         {max_diff:.6f}")

        if abs(ratio - 1.0) > 0.2:
            print(f"  *** Permutation significantly affects output! ***")
        else:
            print(f"  Permutation has minimal effect on output")

    def test_full_comparison_decode(self):
        """
        Full end-to-end comparison: original sageattn_qk_int8_pv_fp8_cuda vs
        manual reproduction of C++ runtime path, for multiple kv_len values.
        """
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        torch.manual_seed(42)
        batch_size = 1
        num_qo_heads = 16
        num_kv_heads = 8
        head_dim = 128
        qo_len = 1

        kv_lens_to_test = [21, 32, 64, 65, 128]

        print(f"\n{'='*60}")
        print(f"Full decode comparison: original SageAttention vs FP16 reference")
        print(f"{'='*60}")
        print(f"{'kv_len':<10} {'sage_norm':<12} {'ref_norm':<12} {'ratio':<10} {'max_diff':<12}")
        print(f"{'-'*60}")

        for kv_len in kv_lens_to_test:
            q = torch.randn(batch_size, num_qo_heads, qo_len, head_dim,
                            dtype=torch.float16, device="cuda")
            k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                            dtype=torch.float16, device="cuda")
            v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim,
                            dtype=torch.float16, device="cuda")

            # Original SageAttention (per_warp, fp32 accum, no smooth)
            sage_out = sageattn_qk_int8_pv_fp8_cuda(
                q, k, v,
                tensor_layout="HND",
                is_causal=False,
                qk_quant_gran="per_warp",
                sm_scale=None,
                pv_accum_dtype="fp32",
                smooth_k=False,
                smooth_v=False,
            )

            # FP16 reference
            scale = head_dim ** -0.5
            k_expanded = k.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
            v_expanded = v.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
            scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores, dim=-1)
            ref_out = torch.matmul(attn_weights, v_expanded)

            sage_norm = sage_out.float().norm().item()
            ref_norm = ref_out.float().norm().item()
            ratio = sage_norm / ref_norm if ref_norm > 0 else float('inf')
            max_diff = (sage_out.float() - ref_out.float()).abs().max().item()

            print(f"{kv_len:<10} {sage_norm:<12.4f} {ref_norm:<12.4f} {ratio:<10.4f} {max_diff:<12.6f}")

            # Sanity check: original SageAttention should NOT be attenuated
            assert ratio > 0.5, (
                f"Original SageAttention is attenuated at kv_len={kv_len}: ratio={ratio:.4f}"
            )


if __name__ == "__main__":
    # Allow running directly for quick debugging
    test = TestSageAttentionVsReference()
    print("=" * 60)
    print("Test 1: Original SageAttention decode sanity check")
    print("=" * 60)
    test.test_original_sage_attention_decode()

    print("\n" + "=" * 60)
    print("Test 2: V quantization comparison")
    print("=" * 60)
    test.test_cpp_runtime_v_quantization_vs_original()

    print("\n" + "=" * 60)
    print("Test 3: Kernel call comparison (C++ path vs original)")
    print("=" * 60)
    test.test_cpp_runtime_kernel_call_vs_original()

    print("\n" + "=" * 60)
    print("Test 4: Permutation effect")
    print("=" * 60)
    test.test_v_permutation_effect()

    print("\n" + "=" * 60)
    print("Test 5: Full decode comparison across kv_lens")
    print("=" * 60)
    test.test_full_comparison_decode()


# python tests/python-unittests/test_sage_attention_vs_reference.py============================================================
# Test 1: Original SageAttention decode sanity check
# ============================================================

# [Original SageAttention] output norm: 14.4303
# [FP16 Reference]         output norm: 14.4297
# [Ratio]                  sage/ref:    1.0000

# ============================================================
# Test 2: V quantization comparison
# ============================================================

# [V quantization comparison]
#   Original v_fp8 shape: torch.Size([1, 8, 128, 64])
#   C++ runtime v_fp8 shape: torch.Size([1, 128, 8, 64])
#   Original v_scale shape: torch.Size([1, 8, 128])
#   v_scale range: [0.002631, 0.009731]
#   Max diff between C++ and original FP8 values: 450.5000
#   (Expected to differ due to per-channel scaling in original)

# ============================================================
# Test 3: Kernel call comparison (C++ path vs original)
# ============================================================

# [Kernel output comparison]
#   Original (with v_scale+permute) output norm: 14.4303
#   C++ runtime (no v_scale, no permute) output norm: 13.9600
#   Ratio (cpp/original): 0.9674
#   FP16 reference output norm: 14.4297
#   Original vs FP16 ratio: 1.0000
#   C++ runtime vs FP16 ratio: 0.9675
#   Max absolute difference (orig vs cpp): 1.442627
#   Outputs are within acceptable range

# ============================================================
# Test 4: Permutation effect
# ============================================================

# [Permutation effect test]
#   With permutation output norm:    14.4303
#   Without permutation output norm: 13.9731
#   Ratio (no_permute/permute):      0.9683
#   Max absolute difference:         1.457275
#   Permutation has minimal effect on output

# ============================================================
# Test 5: Full decode comparison across kv_lens
# ============================================================

# ============================================================
# Full decode comparison: original SageAttention vs FP16 reference
# ============================================================
# kv_len     sage_norm    ref_norm     ratio      max_diff    
# ------------------------------------------------------------
# 21         14.4303      14.4297      1.0000     0.043457    
# 32         12.2423      12.2757      0.9973     0.050293    
# 64         8.9397       8.9523       0.9986     0.035645    
# 65         9.1265       9.1373       0.9988     0.031738    
# 128        6.3652       6.3658       0.9999     0.023438
