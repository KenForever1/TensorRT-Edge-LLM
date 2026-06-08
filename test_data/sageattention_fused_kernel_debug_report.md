# SageAttention Fused Kernel 调试报告

## 目标

将 SageAttention decode 路径中的 K/V 量化操作融合到 attention kernel 内部，直接从 FP16 KV cache 读取数据并量化，消除中间 INT8/FP8 buffer，使 kernel 原生兼容 CUDA Graph（无需按 capacity 预分配大 buffer）。

## 架构变更

```
变更前 (Plan D):
  KV cache → computeStats → convertKvCacheToSageKernel → 3 adjust kernels
           → K_INT8 [B,capacity,H,D] (4MB)
           → V_FP8  [B,D,H,paddedS] (4MB)
           → 原始 SageAttention kernel

变更后 (Plan 1 Fused):
  KV cache → computeVScalesAndKMeans (1 kernel, vScale+kMean 仅 4KB each)
           → fused attention kernel (直接从 cache 读 FP16 → quantize in smem → mma)
           → 零中间 buffer
```

## 已修复的 Bug

| # | 问题 | 位置 | 症状 |
|---|------|------|------|
| 1 | KV cache 索引多除了 sizeof(half) | `qk_int_sv_f8_cuda_sm89_fused.cuh` | V 数据加载完全错误 |
| 2 | V scale GQA 索引 (numQoHeads vs numKVHeads) | 同上 | GQA 模型 V 量化错误 |
| 3 | Softmax 使用了 `fuse_scale=true` 导致 `original_sm_scale` 未应用 | 同上 | 所有 attention weight 错误 |
| 4 | K scale 循环 bound `CTA_K*num_valid` 应为 `num_valid*HEAD_DIM` | 同上 | K scale 只算了半个 tile |
| 5 | QK mma offset `QK_off_Q` 在 Q preload 后被消耗 | 同上 | QK mma 读到垃圾数据 |
| 6 | K smem 写入未对齐原始 `load_global_to_share` 的 lane mapping | 同上 | ldmatrix 读到错位数据 |
| 7 | V smem 写入未对齐原始 `load_fp8_V_global_to_share` 的 lane mapping | 同上 | ldmatrix 读到错位数据 |
| 8 | V 序列 permutation 缺失 (RS_32_to_8 产出的 permuted RS 需要 matched permuted V) | 同上 | PV mma 输入 RS/V 不对齐 |
| 9 | K-mean centering 缺失 | 同上 | K 量化结果与原始路径不同 |

## 验证方法

### 1. MMA 路径验证 (Matched-Quant Test)

创建独立测试，两路径使用相同的 Q/K/V 量化值：
- 路径 A: 原始 kernel + host 计算的 K_INT8 + V_FP8 (无 K-mean，单次 FP8 转换)
- 路径 B: fused kernel (从 KV cache 加载并量化)

**结果**:
- kL=64, H=4: **0/512 错误** (bit-exact)
- kL=256, H=8: **0/1024 错误** (bit-exact)
- 证明: fused kernel 的 MMA/softmax/PV/output 路径完全正确

### 2. Per-Layer Attention 对比 (Plugin 内双路径测试)

在 `attentionPlugin::enqueue` 中同时运行原始路径和 fused 路径，比较 attention output：

```cpp
// 1. Fused 路径 → attentionOutputTensor
// 2. 同步 stream
// 3. 复制 fused output 到 host
// 4. 用独立 buffer 运行原始路径 → 单独 output buffer
// 5. 同步 stream
// 6. 复制原始 output 到 host
// 7. 逐元素对比
```

**关键教训**: 调试用 `rt::Tensor` 在 graph capture 期间 cudaMalloc，导致 graph 捕获了临时指针，replay 时指针失效，输出乱码。**这个错误多次误导了调试方向！**

最终方法：在 graph 启用前的 warmup 阶段做对比（`cmpCount < N`），或直接禁用 graph 测试。

**结果**: 所有 28 层，多个 decode step: **0/2048 错误 per layer** (bit-exact)

### 3. CUDA Graph 影响测试

禁用 CUDA Graph (`captureDecodingCUDAGraph` 返回 false)，测试 fused kernel：

- 禁用 graph: **输出完全正确** ✅
- 启用 graph: 输出乱码（实际是调试代码的 cudaMalloc 导致，不是 fused kernel 的问题）

**最终确认**: 移除调试代码后，启用 graph，输出完全正确。

## 关键教训

### GPU 内存与 CUDA Graph

> **CUDA Graph 黄金法则**: 在 graph capture 期间，所有 GPU 内存必须是预先分配的（workspace）或持久存在的。不能在 capture 期间 cudaMalloc + 使用 + cudaFree，因为 graph 会捕获临时指针，replay 时这些指针已失效。

```cpp
// ❌ 错误做法 (导致 graph 问题):
SageAttentionTest::enqueue(...) {
    rt::Tensor tmpBuf({...});  // cudaMalloc 临时地址
    launchKernel(tmpBuf.rawPointer());  // graph 捕获了这个临时地址!
} // ~tmpBuf() → cudaFree → replay 时野指针!

// ✅ 正确做法:
// 所有 buffer 必须在 getWorkspaceSize() 中声明
// 在 enqueue() 中通过 assignTensorFromWorkspace() 获取
```

### 调试策略总结

1. **先验证单组件**: 用 matched-quant test 验证 MMA 路径独立正确
2. **再验证集成**: 用双路径对比验证 per-layer 输出
3. **隔离变量**: 遇到异常时先禁用 CUDA Graph 排除 graph 交互
4. **注意副作用**: 调试代码本身可能改变行为 (memory allocation, timing)

## 性能

| 指标 | XQA (基线) | Fused SageAttention |
|------|-----------|-------------------|
| pastKVLen=128 | 7.20 ms | 8.20 ms |
| pastKVLen=512 | 7.40 ms | ~11 ms |
| pastKVLen=4096 | 9.62 ms | ~31 ms |

## 文件清单

| 文件 | 说明 |
|------|------|
| `cpp/kernels/sageAttentionKernels/csrc/qattn/qk_int_sv_f8_cuda_sm89_fused.cuh` | Fused kernel 主体 (~380行) |
| `cpp/kernels/sageAttentionKernels/sageAttentionKernels.cu` | Launcher + 模板实例化 |
| `cpp/kernels/sageAttentionKernels/sageAttentionHostUtils.h` | Fused kernel 声明 |
| `cpp/kernels/sageAttentionKernels/sageAttentionRuntimeKernels.cu/.h` | Q 量化 + V scale/K mean 计算 |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | Plugin 集成 (fused 路径选择) |

## 提交历史

```
b4fc26e fused kernel: verified bit-exact attention output
4189214 feat: add K-mean centering
07c42e5 WIP: mma verified at multiple scales
dca5d39 WIP: matched-quant 0/512
296123c WIP: V permutation + smem lane mapping
88b7ddf WIP: initial fused kernel, 5 bugs fixed
```
