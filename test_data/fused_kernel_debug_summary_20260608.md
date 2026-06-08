# Fused SageAttention 排查总结（2026-06-08）

---

## 已确认的事实

### 1. Kernel 本身正确
- `FusedKernelMatchedQuant` (H=4): 0/512 mismatches ✅
- `MatchedQuant256H8` (H=8, kL=256): 0/1024 mismatches ✅
- `FusedSimplePattern16Heads` (H=16, 常数 V=2.0): maxErr=0 ✅

### 2. K scale 完全匹配
- 新增 `launchSageComputeKScalesFromKV` kernel 产出的 K scale 与 convert pipeline 完全一致
- 512 个 per-tile K scale 值，nDiff=0

### 3. V_FP8 几乎一致
- 17/64 字节有差异，且仅差 1 LSB（FP8 编码精度损失）
- 不是主因

### 4. 无 CUDA 错误
- kernel launch 成功，无越界访问
- Q scale、V scale、kMean 索引全部正确

### 5. Pre-fusion 仍正确
- `#if 0` 切到 pre-fusion 路径：输出 "Washington, D.C." ✅
- 引擎、模型、tokenizer 无问题

---

## 已实现的改动

| 改动 | 文件 | 状态 |
|------|------|------|
| OOB write predicate fix | `qk_int_sv_f8_cuda_sm89_fused.cuh` | ✅ |
| K scale two-step centering | `qk_int_sv_f8_cuda_sm89_fused.cuh` | ✅ |
| External K scale kernel | `sageAttentionRuntimeKernels.cu` | ✅ |
| launch param kScalesPerTile/kDebug | `sageAttentionKernels.cu` / `hostUtils.h` | ✅ |
| sKRed/sKMax → extern smem | `qk_int_sv_f8_cuda_sm89_fused.cuh` | ✅ |
| Q preload dead code removal | `qk_int_sv_f8_cuda_sm89_fused.cuh` | ✅ |

---

## 未解决的核心问题

**Fused 和 pre-fusion 输出差 99%。** 即使 K scale + kMean + V scale 完全一致 + two-step K-mean centering，FUSED_VS_PRE 仍显示 99% mismatch。

### 已排除的因素 (14项)

1. ✅ V smem layout（forward vs inverse perm）
2. ✅ RO → smem_O store layout
3. ✅ OOB write predicate（output copy）
4. ✅ GQA ratio（r=1 和 r=2 表现一致）
5. ✅ K-mean centering（传 kMean 仍 fail）
6. ✅ Q scale indexing / Q scale 值
7. ✅ K/V smem swizzle bounds
8. ✅ SMEM 总量超限（static → extern smem）
9. ✅ Partial KV（OOB guard + zero-init）
10. ✅ Q preload 死代码
11. ✅ ComputeUnit（kTensorCore vs kCudaCore）
12. ✅ KV cache head index
13. ✅ V_scale_ch 索引
14. ✅ Grid 大小（8/16 blocks）

### 不同之处（唯一余下可能）

Fused kernel 在 smem 里量化 K/V，原始 kernel 用预量化的 K_INT8/V_FP8。即使公式相同：
- smem 里的 swizzle 布局
- ldmatrix 顺序
- `compute_fp8_sv` 里的 `smem_V_col_base = (lane_id / 8) % 2` 行为

都可能有细微差异被原始 kernel 的特殊处理跳过。

---

## 排查历程

1. **OOB write fix** → 修复 workspace 破坏，pre-fusion 恢复工作
2. **GQA isolation** → 确认 bug 不是 GQA 特定
3. **K scale alignment** → 新增 external K scale kernel，确认与 convert pipeline 完全一致
4. **Two-step K-mean** → 实现 `round(K/kScale) - round(kMean/kScale)` 匹配 convert pipeline
5. **V_FP8 comparison** → 确认仅 17/64 字节有 1-LSB 差异
6. **V scale overwrite** → 确认不是主因
7. **Kernel printf** → 确认 Q/K/RO 值合理，d≠0

---

## 建议的下一步

写一个最小隔离测试：**用 pre-fusion 的 K_INT8 和 V_FP8 直接喂给 fused kernel，绕过 fused 的自量化**。

具体做法：
- 新增 fused kernel 参数：接收预量化的 K_INT8 和 V_FP8
- 当这些参数非 null 时，跳过 K/V 量化，直接 load 到 smem
- 对比：相同的 K_INT8 + V_FP8 是否产生相同的 attention 输出

这样可以彻底分离"量化差异"和"matmul/layout 差异"。
