# SageAttention 性能优化记录

## 优化日期
2026-06-05

## 优化概述

在当前 `feature-sage-attention` 分支基础上，对 SageAttention decode pipeline 进行 3 项性能优化，保持算法一致（K-mean centering + per-channel V scaling + per-warp K quantization）。

---

## 优化 1: 融合 stats 到 convert kernel

### 变更前
```
computePerChannelStatsKernel (读 KV cache → vScale, kMean)
    ↓
cudaMemsetAsync (V-FP8 padding 清零)
    ↓
convertKvCacheToSageKernel (再读 KV cache → K_int8, V_fp8, kScale)
```
**4 kernel launches，KV cache 读取 2 次**

### 变更后
```
convertKvCacheToSageKernelFused (读 KV cache 1次 → K_int8_raw, V_fp8_raw, kScale, partials)
    ↓
reducePartialChannelStatsKernel (reduction over partials → vScale, kMean)
    ↓
applyKMeanAdjustmentKernel (K_int8 = K_int8_raw - round(kMean/kScale))
    ↓
applyVScaleAdjustmentKernel (V_fp8 = V_fp8_raw / vScale，FP8精度恢复)
    ↓
cudaMemsetAsync (V-FP8 padding 清零)
```
**5 kernel launches，KV cache 读取 1 次**

### 技术要点
- 在 convert kernel 的 half2 读取循环中增量累积 per-dim maxAbsV 和 sumK
- 使用 shared memory `atomicMax`/`atomicAdd` 合并同 dim 的不同线程结果
- Partial buffer: `[batchSize, numKVHeads, headDim, numBlocks]` float
- Reduction kernel: 1 thread per (batch, head, dim)，跨 numBlocks 合并 partials
- K-mean 调整: `K_int8 -= round(kMean/kScale)`，与原始算法的差异 ≤ ±1 INT8 (rounding order)
- V-scale 调整: `V_fp8 = convertFloatToFp8(V_fp8_raw_float / vScale)`，双 FP8 转换精度损失 ≤8.5%
- `fuse_v_scale=true` 在 attention kernel 中正确恢复 V 值: `V_float ≈ V_raw/vScale * vScale ≈ V_raw`

### 收益
- 消除 1 次完整 KV cache 读取（节省 batchSize × 2 × numKVHeads × kvLen × headDim × 2 bytes 内存带宽）
- 消除 1 个单独 kernel launch

---

## 优化 2: half2 向量化读取

### 变更点
| 位置 | 变更前 | 变更后 |
|------|--------|--------|
| Loop 1 (K max) | 标量 half 读取，每迭代 1 个 K 值 | half2 读取，每迭代 2 个 K 值 |
| Loop 2 (K/V quantize) | 标量 half 读取，每迭代 1 个 K + 1 个 V | half2 读取，每迭代 2 个 K + 2 个 V |
| K 输出写入 | 标量 int8_t 写入 | `int16_t` 打包写入（2 个 INT8 一次） |
| K-mean/V-scale 调整 | N/A (新 kernel) | half2 风格成对处理 dim |

### 收益
- 内存读取指令减半
- K 写入指令减半
- Loop 迭代次数减半

---

## 优化 3: Decode Q 量化优化

### 变更前
`quantizeBshdToInt8PerWarpKernel` — 对于 qoLen=1 (decode) 发射 `batchSize × numHeads × 4` 个 block × 256 threads
（如 batch=1, heads=4 → 16 blocks × 256 threads = 4096 threads for 512 elements）

### 变更后
`quantizeBshdToInt8DecodeKernel` — qoLen ≤ 32 时自动选择，发射 `batchSize × numHeads` 个 block（仅 1 block per head）
（如 batch=1, heads=4 → 4 blocks × 256 threads = 1024 threads）

### 收益
- Grid 大小减少 4×（4 blocks vs 16 blocks）
- Thread 总数减少 4×
- Q 量化结果为 bit-exact (qInt8: 0 mismatches)

---

## 精度验证

### 单元测试: 17/17 通过
| 测试 | pass_rate | avg_error |
|------|-----------|-----------|
| BasicMHA (512×512) | 0.994 | 0.019 |
| SingleTokenDecode | 1.0 | 0.0085 |
| RuntimeKvCacheHelperSingleTokenDecode | 1.0 | 0.0010 |
| RuntimeKvCacheHelperBatchDecode | 1.0 | 0.00095 |

### 模型推理
```
Input: "What is the capital of United States?"
Output: "The capital of the United States is Washington, D.C. (Washington, D.C.)."  ✅
```

---

## 修改文件清单

| 文件 | 修改类型 |
|------|----------|
| `cpp/kernels/sageAttentionKernels/sageAttentionRuntimeKernels.cu` | 核心修改：新增 4 个 kernel，重写 convert kernel，优化 Q 量化 |
| `cpp/kernels/sageAttentionKernels/sageAttentionRuntimeKernels.h` | 新增 `getSageStatsPartialsSize`，更新函数签名 |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | 新增 partials buffer workspace 分配和 tensor 赋值 |
| `unittests/sageAttentionTest.cpp` | 更新调用点以匹配新 API |

---

## 性能对比: SageAttention vs XQA (decode, batchSize=1, Qwen3-0.6B)

| pastKVLen | SageAttention | XQA (baseline) | Ratio |
|-----------|--------------|----------------|-------|
| 128 | 31.73 ms | 7.20 ms | **4.40x slower** |
| 512 | 30.62 ms | 7.40 ms | 4.14x slower |
| 1024 | 31.07 ms | 7.84 ms | 3.96x slower |
| 2048 | 31.82 ms | 8.31 ms | 3.83x slower |
| 4096 | 33.03 ms | 9.62 ms | 3.43x slower |

### 分析

SageAttention 在当前配置下比 XQA 慢 3.4-4.4x，主要原因：

1. **量化/转换开销恒定且巨大**: 为支持 CUDA Graph，workspace 按 `kvCacheCapacity` (4096) 分配，convert kernel 总是处理 4096 个 sequence slot（512 个 block），即使 pastKVLen 仅 128。转换开销（量化 Q/K/V）远大于 attention 计算本身。

2. **模型过小**: Qwen3-0.6B 仅有 28 层、8 KV heads、headDim=128。FP16 XQA kernel 对此规模已经非常高效，量化的 INT8/FP8 计算优势无法覆盖转换成本。

3. **SageAttention 适用场景**: 长序列 + 大模型（如 Llama-70B, headDim=128/256, kvLen > 16K），此时 attention 计算成为瓶颈，INT8/FP8 加速效果显著。

### 结论

对于 Qwen3-0.6B 这类小模型，SageAttention 的量化开销过大，不应启用。建议在 `canUseSageDecode` 中添加启发式规则：仅当 `kvCacheCapacity >= 16K` 或模型 hiddenSize >= 4096 时才使用。

