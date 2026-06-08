
## 问题描述

在/home/ken/cc_workspace/TensorRT-Edge-LLM/cpp/plugins/attentionPlugin/attentionPlugin.cpp的159行，如果改了return false就是不使用我添加的sageattention execute vanilla decoding.
```
bool canUseSageDecode(AttentionExecutionMode executionMode, int32_t runtimeSeqLen, int32_t smVersion,
    nvinfer1::DataType dataType, int32_t headSize, int32_t numQHeads, int32_t numKVHeads, int32_t slidingWindowSize,
    int32_t enableTreeAttention, int32_t enableFp8KVCache) noexcept
{
    // return executionMode == AttentionExecutionMode::kVANILLA_DECODING && runtimeSeqLen == 1 && enableTreeAttention == 0
    //     && enableFp8KVCache == 0 && slidingWindowSize <= 0 && numKVHeads > 0 && numQHeads % numKVHeads == 0
    //     && SageAttentionRunner::canImplement(headSize, smVersion, dataType);

    return false;
}
```
如果取消注释，就是使用我添加的sageattention execute vanilla decoding.

现在我添加的sageattion在decode时出现了精度问题，生成的/home/ken/cc_workspace/TensorRT-Edge-LLM/test_data/output_err.json是错误的。正确的DecoderXQARunner生成的/home/ken/cc_workspace/TensorRT-Edge-LLM/test_data/output.json是正常的。


输入是/home/ken/cc_workspace/TensorRT-Edge-LLM/test_data/input.json。执行命令是：
```
cd /home/ken/cc_workspace/TensorRT-Edge-LLM
bash run_qwen3.sh
```
## 编译方法
如果修改了代码编译命令是：
```bash
bash compile.sh
```

## sageattention的测试：
```bash
/home/ken/cc_workspace/TensorRT-Edge-LLM/unittests/sageAttentionTest.cpp
```
执行命令：
```bash
./build/unitTest --gtest_filter="SageAttentionTest.*"
```
## sageattention python测试
python环境：
```
source ~/cpp_idioms/.venv/bin/activate.fish
```
测试的精度是通过的。
```bash
python3 tests/python-unittests/test_sage_attention_vs_reference.py
```

## 参考sageattention仓库源码位置
```
/home/ken/cc_workspace/SageAttention
```
请排查这个精度问题？

你可以通过添加打印、gdb等方式排查，将排查的核心过程记录到项目的markdown中，避免重复排查排查过的内容，这是个比较难排查的内容，你是个llm、算子开发的专家，相信你。

## 排查记录

### 已完成的修改

#### 1. V per-channel scaling (fuse_v_scale=true) — 已实施
**文件**: `cpp/kernels/sageAttentionKernels/sageAttentionRuntimeKernels.cu`
- 添加了 `computePerChannelStatsKernel` 内核，计算 per-channel V scale
- V scale = max(|V|) / 448.0 per (batch, kv_head, dim)
- V 在 FP8 转换前除以 V scale: V_fp8 = fp8(V / v_scale)
- 在 SageAttention 内核中启用 fuse_v_scale=true，内核在 S*V 矩阵乘法后乘以 v_scale 进行补偿

**文件**: `cpp/kernels/sageAttentionKernels/sageAttentionKernels.cu`
- 添加了 FUSE_V_SCALE 模板分派，在 `launchSageAttentionSm89Dispatched` 中运行时选择

**文件**: `cpp/plugins/attentionPlugin/attentionPlugin.cpp`
- 添加 V scale workspace 分配
- 将 v_scale_ptr 和 fuse_v_scale=true 传递给内核

**文件**: `unittests/sageAttentionTest.cpp`
- 更新所有测试以传递 vScale 和 kMean 张量

**效果**: 所有 17 个单元测试通过 (pass_rate=1.0)

#### 2. K-mean centering (smooth_k) — 已实施
**文件**: `cpp/kernels/sageAttentionKernels/sageAttentionRuntimeKernels.cu`
- 在 `computePerChannelStatsKernel` 中计算 per-channel K 均值
- K-mean = mean(K) over sequence per (batch, kv_head, dim)
- K_int8 = round((K - K_mean) / K_scale)
- softmax 对 K 均值的常数偏移是不变的（每行减去相同值不影响 softmax）

**效果**: 所有 17 个单元测试通过 (pass_rate=1.0)

#### 3. V sequence permutation — 不需要
- 原始 SageAttention 应用排列 [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] 到 V 序列维度
- 此排列是为原始 V 布局设计的: [B, Hkv, D, paddedS]
- 我们的布局是: [B, D, Hkv, paddedS]，stride 计算不同
- 在我们的布局中应用排列会使模型输出更差
- 已还原此更改

#### 4. CUDA Graph 捕获 — 已禁用（用于调试）
**文件**: `examples/llm/llm_inference.cpp`
- 暂时注释掉了 `captureDecodingCUDAGraph` 以在不使用 CUDA graph 的情况下进行测试

### 当前状态
- 模型输出: 部分连贯（提到 "United States"）但严重重复（"is is is..."）
- 与原来的完全乱码（output_err.json）相比有所改善，但仍然不正确
- 所有单元测试以 pass_rate=1.0 通过

### 模型输出对比
1. **正确输出 (XQA)**: "The capital of the United States is **Washington, D.C.**."
2. **原始 SageAttention（无修复）**: "The,\n is\n\n\nTheThe the\n\n\nThe question is about the..."
3. **当前（V scale + K-mean）**: "Theius is iship... The United States is a... is is is country..."

### 反思
    V的permute从原始sage attention来看应该需要。对齐我们的layout。排列是排列的seq维度，我们的layout排列seq维度，逻辑上也是等价的，因为seq维快速的排列是一样。需要再看下这个排列的具体实现和原始sage attention是如何实现的排列。



## ✅ 问题已解决

### 根本原因
C++ runtime 的 SageAttention 集成缺少了原始 SageAttention 中的三个关键精度优化：

1. **V per-channel scaling** — 未实现
2. **K-mean centering (smooth_k)** — 未实现
3. **V sequence permutation** — 使用了错误的排列

### 修改的文件
| 文件 | 修改内容 |
|------|---------|
| `cpp/kernels/sageAttentionKernels/sageAttentionRuntimeKernels.cu` | 添加 `computePerChannelStatsKernel`（V scale + K mean），在 convert 内核中添加 V scale/K mean/V permutation |
| `cpp/kernels/sageAttentionKernels/sageAttentionRuntimeKernels.h` | 添加 `getSageVScaleSize`，更新 `launchSageConvertKVCacheToInt8AndFp8` 签名 |
| `cpp/kernels/sageAttentionKernels/sageAttentionKernels.cu` | 添加 `FUSE_V_SCALE` 运行时分派 |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | 添加 V scale + K mean workspace，传递 fuse_v_scale=true + v_scale_ptr |
| `unittests/sageAttentionTest.cpp` | 更新测试以传递 vScale/kMean 参数 |

### 最终验证
- 所有 17 个 SageAttention 单元测试通过
- CUDA graph 启用时模型输出正确："The capital of the United States is **Washington, D.C.**."

---

## 性能分析与优化

### 当前性能 (Qwen3-0.6B, batch=1, decode noCudaGraph)

| pastKVLen | XQA (ms) | XQA (tok/s) | SageAttention (ms) | SageAttention (tok/s) | Sage/XQA |
|-----------|----------|-------------|-------------------|---------------------|----------|
| 128       | 9.03     | 110.8       | 21.06             | 47.5                | 2.33x   |
| 512       | 9.21     | 108.6       | 21.98             | 45.5                | 2.39x   |
| 1024      | 9.59     | 104.3       | 23.80             | 42.0                | 2.48x   |
| 2048      | 10.16    | 98.4        | 26.67             | 37.5                | 2.63x   |
| 4096      | 11.50    | 87.0        | 32.28             | 31.0                | 2.81x   |

### 性能瓶颈分析

每个 decode step，SageAttention 路径额外执行 4 个 kernel launch：
1. `computePerChannelStatsKernel` — 计算 V scale 和 K mean
2. `cudaMemsetAsync` — 清零 V-FP8 tensor (kvLen 为 64 倍数时可省略)
3. `convertKvCacheToSageKernel` — 量化 K/V
4. SageAttention kernel — 注意力计算

XQA 仅执行 1 个 kernel launch。

#### 主要瓶颈：`computePerChannelStatsKernel`
- 每个线程串行遍历所有 effectiveKvLen 个位置: `for (s = 0; s < effectiveKvLen; ++s)`
- 对 effectiveKvLen=4096，每个线程 4096 次迭代 × 2 次全局内存读取 = 8192 次读取
- 总工作量: O(batch × numKVHeads × headDim × effectiveKvLen)
- 线程不协作 — 每个 (batch, head, dim) 由一个线程独立完成

### 优化方案（按优先级）

#### 1. 融合 stats 和 convert kernel（高优先级）
- `computePerChannelStatsKernel` 和 `convertKvCacheToSageKernel` 都读取 KV cache
- 融合为一个 kernel：先计算 per-block stats，再用共享内存做跨 block 的 max/sum reduction
- 预期减少 1 个 kernel launch 和 50% 的 KV cache 全局内存读取

#### 2. 消除冗余的 cudaMemsetAsync（已实施）
- 当 kvLen 是 CTA_K(64) 的倍数时，convert kernel 已覆盖所有 V-FP8 位置
- 仅在 paddedKvLen > kvLen 时 memset padding 区域
- kvCacheCapacity=4096 时 kvLen 总是 64 倍数，可完全跳过 memset

### 优化后性能对比 (decode, noCudaGraph)

| pastKVLen | XQA (ms) | SageAttention 优化前 (ms) | SageAttention 优化后 (ms) |
|-----------|----------|--------------------------|--------------------------|
| 128       | 9.03     | 21.06                    | 22.31*                   |
| 512       | 9.21     | 21.98                    | —                        |
| 1024      | 9.59     | 23.80                    | —                        |
| 2048      | 10.16    | 26.67                    | —                        |
| 4096      | 11.50    | 32.28                    | 31.89                    |

* 128 时因并行 reduction 的 __syncthreads() 开销反而变慢，已回退该优化。

### 后续优化方向（保持算法一致）
1. **融合 stats 到 convert kernel**：消除 1 次 kernel launch 和 KV cache 重复读取，需要 partials buffer + reduction kernel
2. **向量化 stats 读取**：使用 half2 一次读取 2 个 half，加倍读取吞吐
3. **预计算 Q scale**：Q quantization 不依赖 KV cache，可以在 applyRopeWriteKV 时顺便完成
优化步骤：测试精度，结果正确->测试性能，性能提升，将优化结果记录到markdown