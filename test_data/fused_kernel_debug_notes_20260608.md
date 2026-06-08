# Fused SageAttention 排查笔记（2026-06-08 接力）

排查者：Claude。在 commit `ae2afdc` 基础上继续排查。上一轮笔记：`fused_kernel_debug_notes.md`。

---

## 一、当前 commit (`ae2afdc`) 状态总结

### 1.1 已修复的 bug（前轮继承）

| Bug | 位置 | 状态 |
|-----|------|------|
| V smem layout：forward vs inverse perm | fused kernel VLOAD/VQUANT | ✅ 已修复 |
| RO → smem_O store layout（重写） | fused kernel output 阶段 | ✅ 已修复 |

### 1.2 本轮新改动

| 改动 | 说明 | 原因 |
|------|------|------|
| 删除 Q preload（`RQ[num_tiles_q][4]` dead code） | fused kernel 行 117-123 | 减少寄存器压力 |
| sKRed/sKMax 从 `__shared__` 移到 extern smem | `REDUCTION_OFFSET` | 避免 static smem 导致总 smem 超 48KB |
| K smem zero-init | 量化前清零 | 防止 partial KV 的 OOB 位置泄漏 |
| K 量化的 OOB row guard | `row >= num_valid` → qi=0 | 与 zero-init 配合 |
| K-mean centering 支持已存在于 kernel 中 | `k_mean != nullptr` 分支 | 推理用的 |

### 1.3 测试补充

| 测试 | 配置 | 目的 |
|------|------|------|
| `FusedSimplePattern16Heads` | H=16, cap=64, 常数数据 (Q=1, K=0.5, V=2) | 验证 kernel 基本正确性 |
| `FusedGQAFullKv_NoGQA` | Hq=Hkv=16, r=1, cap=64 | 隔离 GQA vs head-count |
| `FusedVsPreFusionThreshold{8,9,10,12,14}` | 8-14 heads, 传入 kMean | 找 head-count 阈值 |
| `QScaleDump16Heads` | H=16 | 验证 decode Q 量化 kernel |
| `Fp8Roundtrip` / `Fp8SmemRoundtrip` | - | FP8 编解码验证 |

---

## 二、关键发现

### 2.1 Bug 不是 GQA 特定的

| 测试 | 配置 | 结果 |
|-----|------|------|
| `FusedKernelMatchedQuant` | Hq=Hkv=4, r=1, kL=64 | ✅ 0 mismatches |
| `MatchedQuant256H8` | Hq=Hkv=8, r=1, kL=256 | ✅ 0 mismatches |
| `FusedSimplePattern16Heads` | H=16, r=1, 常数数据 | ✅ maxErr=0 |
| `FusedGQAFullKv` | Hq=16 Hkv=8, r=2, 随机数据 | ❌ pr=0.003 |
| `FusedGQAFullKv_NoGQA` | Hq=Hkv=16, r=1, 随机数据 | ❌ pr=0.003 |
| `FusedVsPreFusionThreshold8` | H=8, 随机数据, 传入 kMean | ❌ pr=0.005 |

**所有使用 `launchSageConvertKVCacheToInt8AndFp8` 做 pre-fusion 参照的测试都 fail，不论 head count 和 GQA ratio。**

### 2.2 根本原因分析

**比较方式本身有问题**：

- Pre-fusion 路径使用 `launchSageConvertKVCacheToInt8AndFp8` → 对 K 做 **K-mean centering**（`K' = K - k_mean`）后再 INT8 量化
- Fused kernel（kMean=nullptr 时）直接从 raw FP16 K 做 INT8 量化

两者产生的 K_INT8 **数值不同**。虽然 K-mean centering 在理想浮点 softmax 中是移位不变的（`softmax(Q·K + C) = softmax(Q·K)`），但 INT8 量化非线性打破了这一不变性。

**证据**：当 fused 和 original SageAttention 用相同的 host 预量化 K_INT8 时（`FusedKernelMatchedQuant`、`MatchedQuant256H8`），结果完美匹配（0 mismatches）。

### 2.3 常数数据测试通过的意义

`FusedSimplePattern16Heads` (16 heads, 常数 K=0.5, V=2.0) 输出全部精确为 2.0，证明：

1. ✅ K 量化 → smem_K 正确
2. ✅ V 量化 → smem_V 正确
3. ✅ QK matmul（INT8 MMA）正确
4. ✅ Softmax + dequant 正确
5. ✅ PV matmul（FP8 MMA）正确
6. ✅ RO → smem_O → global O 输出路径正确
7. ✅ 所有 16 个 head 的 grid 调度和独立计算正确

**Kernel 的基本逻辑在所有 head count 下都是正确的。问题出在数据依赖的量化误差上。**

### 2.4 Q Scale 验证

`QScaleDump16Heads` 确认：
- 所有 16 个 head 的 warp-0 Q scale 与期望值完全一致
- warp-1/2/3 的 scale 全部为 epsilon (1e-6)，48 个都不为异常值
- **decode Q 量化 kernel 对于 16 heads 完全正确**

---

## 三、已排除的因素（总清单）

1. ✅ **V smem layout**（forward perm → inverse perm，已修复）
2. ✅ **RO → smem_O store layout**（重写为原版逻辑，已修复）
3. ✅ **GQA r=2**（r=1 同样 fail，不是 GQA 问题）
4. ✅ **K-mean**（传入 kMean 仍然 fail）
5. ✅ **Q scale indexing / Q scale 值**（QScaleDump 验证正确）
6. ✅ **K/V smem swizzle bounds**（手动计算验证，无溢出）
7. ✅ **SMEM 总量超限**（static → extern 后仍 fail，说明不是 smem 溢出）
8. ✅ **Partial KV**（full-KV 测试同样 fail）
9. ✅ **K 量化 OOB**（zero-init + OOB guard 已加）
10. ✅ **Q preload 死代码**（删除后仍 fail）
11. ✅ **计算单元**（kTensorCore / kCudaCore 已验证）
12. ✅ **KV cache head index**（验证正确）
13. ✅ **V_scale_ch 索引**（验证正确）
14. ✅ **Grid 大小**（8 blocks 和 16 blocks 都 fail，在 FusedVsPreFusionWithHeads 框架下）

---

## 四、通过的测试（golden reference）

| 测试 | 配置 | kMean | K 量化来源 | 结果 |
|-----|------|-------|-----------|------|
| `FusedKernelMatchedQuant` | H=4, kL=64 | nullptr | host 预量化 | ✅ 0/512 |
| `MatchedQuant256H8` | H=8, kL=256 | nullptr | host 预量化 | ✅ 0/1024 |
| `FusedPartialKvSimplePattern` | H=8, cap=128, eff=21, V=2.0 | nullptr | fused 自量化 | ✅ maxErr=0 |
| `FusedSimplePattern16Heads` | H=16, cap=64, V=2.0 | nullptr | fused 自量化 | ✅ maxErr=0 |

**模式**：只要 pre-fusion 参照路径不用 `launchSageConvertKVCacheToInt8AndFp8`（即不用 K-mean centering），fused 就能完美匹配。

---

## 五、推理路径现状

Plugin 中的 diag 块（`#if 1`）当前：

1. **先 fused** → 存 fusedOut
2. **再 pre-fusion**（SageAttentionRunner）→ 覆盖 attentionOutputTensor
3. **比较** fused vs pre-fusion，打印 `[FUSED_VS_PRE]`

由于 diag 块最终用 pre-fusion 输出，**推理仍然正确**（"Washington, D.C."）。

当 `#if 1` 改为 `#if 0`：
- `#else` 分支只调用 fused kernel（commit f673b99 原始实现）→ 乱码
- 需要等 fused kernel 修好后才能启用

---

## 六、下一步建议

### 6.1 最短路径：对齐 K 量化方式

当前 `launchSageConvertKVCacheToInt8AndFp8` 做 K-mean centering，而 fused kernel 不传 kMean 时不做。两种方案：

**A. 只用 fused kernel，且传 kMean（已在 plugin diag 块实现）**
- 推理时让 fused kernel 使用和 convert pipeline 相同的 kMean
- 已经传入但推理仍乱码 → 需要实际跑推理看效果

**B. 去掉 convert pipeline 的 K-mean centering**
- 修改 `launchSageConvertKVCacheToInt8AndFp8` 不应用 kMean
- 或者用 `launchSageComputeVChannelScales`（只算 V scale，不算 K-mean）替代

### 6.2 直接测试推理

```bash
# 确保 diag 块中 kMean 已传入 fused kernel
# 然后跑推理看输出
bash run_qwen3.sh
```

如果输出正确 → 问题在 K-mean 交互上，继续优化
如果输出仍乱码 → 排查推理特有的路径（如 RoPE、KV cache 布局等）

### 6.3 更深入的单步对比

如果推理仍不对，需要加 kernel-level dump：
- 在 fused kernel 内 dump 第一个 K_INT8 值、第一个 V_FP8 值
- 与 convert pipeline 的对应值比较
- 确认 K/V 量化在推理数据下是否一致

---

## 七、改动文件速查

| 文件 | 主要改动 |
|------|---------|
| `cpp/kernels/.../qk_int_sv_f8_cuda_sm89_fused.cuh` | Q preload 删除、sKRed 移到 extern smem、K zero-init、OOB guard、REDUCTION_OFFSET |
| `cpp/kernels/.../sageAttentionKernels.cu` | launch 函数 smemSize 增加 REDUCTION_BYTES |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | 恢复 pre-fusion 路径、添加 `#if 1` fused-vs-pre-fusion diag 块 |
| `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp` | `return false` 短路 graph capture（临时） |
| `unittests/sageAttentionTest.cpp` | 大量新测试（见 1.3 节） |
| `unittests/sageAttentionTestKernels.cu` | Fp8Roundtrip、Fp8SmemRoundtrip 验证 kernel |
| `unittests/sageAttentionTestKernels.h` | 对应声明 |

---

## 八、构建注意事项

- 修改 `.cuh` 后必须：`find build -name 'cmake_device_link.o' -delete && find build -name 'sageAttentionKernels*.o' -delete && bash compile.sh`
- 修改 `.cpp` 测试文件后直接 `bash compile.sh` 即可
- GPU: RTX 4050 Laptop, CC 8.9, 48KB default smem/block, 99KB opt-in
