# Fused SageAttention 排查笔记（2026-06-07 接力）

排查者：Ducc。在 commit `cf220df` 基础上排查 fused SageAttention（plan 1 fusion）精度问题，目标是让 fused 在 graph / no-graph 模式下都能正确推理。

---

## 一、最终目标

1. **fused SageAttention 精度正确**：no-graph 模式下推理输出与 XQA / pre-fusion SageAttention 一致（"Washington, D.C."）。
2. **graph capture 模式也正确**：plugin 路径不能有同步 D2H/H2D（cudaMemcpy）破坏 graph capture，输出与 no-graph 一致。
3. **fused vs XQA 性能对比**：在两种模式下分别测吞吐 / latency，用一致的 prompt。

---

## 二、关键 baseline（已经验证的事实）

### 2.1 黄金对照 — pre-fusion SageAttention 路径（commit f673b99）

把 plugin 改回 f673b99 的实现（`launchSageConvertKVCacheToInt8AndFp8` → `SageAttentionRunner`）后，no-graph 推理：

```
The capital of the United States is **Washington, D.C.** (Washington, D.C.).
```

这条路径**就是 fused 应该对齐的目标**。代码在 `cpp/plugins/attentionPlugin/attentionPlugin.cpp` 当前文件，包在 `if (useSageDecode && kvCacheCapacity > 0) { ... }` 块里，f673b99 原版可以 `git show f673b99d9:cpp/plugins/attentionPlugin/attentionPlugin.cpp` 拿到全文备份在 `/tmp/attentionPlugin_f673b99.cpp`。

### 2.2 XQA 路径

把 `useFused = false`（强制走 XQA），no-graph 和 graph 都输出 `"The capital of the United States is **Washington, D.C.**."`。XQA 路径不依赖 fused，可作为另一个对照。

### 2.3 当前 plugin 的 diag 模式

`#if 1` 包着的"双跑"诊断：先 fused → 存 host fusedOut → 再 SageAttentionRunner（pre-fusion）→ 比较，**最终 attentionOutputTensor 留的是 pre-fusion 输出**，所以即便 fused 出问题推理仍能产出正确文本。每次 attention 调用打印 `[FUSED_VS_PRE] call=N effKv=M maxAbsDiff=... relErr=... mismatchPct=... firstBad=I (fused=F pre=P)`。

调试结束后把该 `#if 1` 改 `#if 0` 即可关闭诊断（保留 pre-fusion 行为）。fused 修对后整个 diag 块可以删掉，只留 fused 调用并把 `#else` 部分（pre-fusion runner）删除。

---

## 三、已修复的 bug

### 3.1 V smem layout：forward perm vs inverse perm

**位置**：`cpp/kernels/sageAttentionKernels/csrc/qattn/qk_int_sv_f8_cuda_sm89_fused.cuh`，VLOAD（行 ~159）+ VQUANT（行 ~280）

**之前的错误状态**：原始 commit cf220df 里 VLOAD 把 V 按 forward perm 写入 temp，VQUANT 又用 forward perm 读 → 两次相同的 forward perm 等于做了一次 perm⊙perm，**不是 involution，不抵消**。

**正确推理**：
- 原版 SageAttention `transpose_pad_permute` 把 V tensor 写出时按 **forward perm** 存：`V_global[d, perm(token_i)] = V_logical[d, token_i]`，等价 `V_global[d, j] = V_logical[d, perm⁻¹(j)]`
- 该 V tensor 通过 `load_fp8_V_global_to_share` 线性 cp_async 进 smem，所以 PV matmul 期望 `smem_V[d, j] = V_logical[d, perm⁻¹(j)]`
- **forward perm 不是 involution**：`perm = [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]`，`perm(2)=4, perm(4)=8 ≠ 2`
- 位运算视角：forward 是 `(b3 b2 b1 b0) → (b2 b1 b3 b0)`，逆是 `(c3 c2 c1 c0) → (c1 c3 c2 c0)`

**修复**：
- VLOAD 改回 linear（`temp[i, d] = V[seq_start+i, d]`）
- VQUANT 用 inverse perm 读 temp：
  ```cpp
  uint32_t mod16 = logical_seq % 16;
  uint32_t inv_ps = ((mod16 >> 1) & 1) * 8
                  + ((mod16 >> 3) & 1) * 4
                  + ((mod16 >> 2) & 1) * 2
                  + (mod16 & 1);
  uint32_t seq_in_tile = (logical_seq / 16) * 16 + inv_ps;
  ```

**验证**：`SageAttentionTest.FusedPartialKvSimplePattern` 通过，`maxErr=0`，`RO=21×448²=4214784`。

### 3.2 RO → smem_O 的 store layout（**重大 bug**）

**位置**：`qk_int_sv_f8_cuda_sm89_fused.cuh` 行 ~376（output 阶段）

**之前的错误状态**：fused kernel 把 RO[8] 当连续 8 个相邻 col 写：

```cpp
uint32_t sw_off = smem_O.get_permuted_offset(
    warp_q*WARP_Q + fq*16 + lane_id % 16,
    lane_id / 16 + fv * 2);
half packed[8];
for (uint32_t k = 0; k < 8; k++) packed[k] = __float2half_rn(RO[fq][fv][k]);
smem_O.base[sw_off] = *(uint4*)packed;
```

但 `mma_sync_m16n16k32_row_col_f8f8f32` 的 output 实际上是两个 m16n8k32 的拼接：
- `RO[0..3]` = 第一个 n=8 fragment（col 0..7）
- `RO[4..7]` = 第二个 n=8 fragment（col 8..15）
- 每 lane 拥有 4 个元素，分布在 (row=lane/4, col_pair=lane%4*2..*2+1) 和 (row=lane/4+8, col_pair=lane%4*2..*2+1) 上

**修复**：抄原版 SageAttention `qk_int_sv_f8_cuda_sm89.cuh` 的 store 逻辑：

```cpp
uint32_t smem_O_row_base = warp_q * WARP_Q + lane_id / 4;
for fq, fv:
    offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq*16, fv*2);
    RO_f16[0] = pack(RO[0], RO[1])
    RO_f16[1] = pack(RO[2], RO[3])
    RO_f16[2] = pack(RO[4], RO[5])
    RO_f16[3] = pack(RO[6], RO[7])
    ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0]
    ((uint32_t*)(smem_O.base + offset_O + 8 * (D/8)))[lane_id % 4] = RO_f16[1]
    offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq*16, fv*2 + 1);
    ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[2]
    ((uint32_t*)(smem_O.base + offset_O + 8 * (D/8)))[lane_id % 4] = RO_f16[3]
```

**验证**：
- `FusedPartialKvSimplePattern` 仍通过 ✓
- `FusedKernelMatchedQuant`（Hq=4 Hkv=4 GQA r=1, kL=64 full kv，host pre-quantized K/V）：**mismatches=0/512, maxDiff=0** 完美对齐 ✓
- `FusedVsPreFusionPluginShape`（Hq=16 Hkv=8 GQA r=2, partial kv eff=21 cap=4096）：仍 fail（接下来要排查的）

---

## 四、当前**仍未解决**的 bug — GQA r=2 路径

### 4.1 现状

| 测试 | 配置 | 结果 |
|-----|------|------|
| `FusedPartialKvSimplePattern` | Hq=Hkv=8 (r=1), cap=128, eff=21, all-V=2.0 | ✅ pass, maxErr=0 |
| `FusedKernelMatchedQuant` | Hq=Hkv=4 (r=1), kL=64=cap, no K-mean | ✅ pass, mismatches=0/512 |
| `FusedGQAFullKv`（新加） | Hq=16 Hkv=8 (r=2), cap=64=eff, no K-mean | ❌ pass rate 0.004, fused 输出几乎是常数 |
| `FusedVsPreFusionPluginShape` | Hq=16 Hkv=8 (r=2), cap=4096 eff=21 | ❌ pass rate 0.006, fused 偏离 ~10× |

**典型现象**：fused 第 0~4 元素几乎相同（如 `0.157, 0.157, 0.155, 0.150, 0.154`），pre-fusion 输出多样化（`0.78, 0.57, -0.84, -0.41, 0.11`）。

### 4.2 已经排除的因素

1. **不是 K-mean**：`FusedGQAFullKv` 关掉 K-mean（传 `nullptr`）依然 fail。
2. **不是 V-scale / K-mean 计算路径不一致**：在 plugin 里让 fused 使用 pre-fusion 的同一组 `vScaleTensor` 和 `kMeanTensor`（`launchSageConvertKVCacheToInt8AndFp8` 算的），结果与让 fused 自己用 `launchSageComputeVScalesAndKMeans` 算的几乎一致。
3. **不是 partial KV**：`FusedGQAFullKv` 用 cap=64=eff 没有 partial 也 fail。
4. **不是 normalize_d 的 ComputeUnit**：试过把 `kTensorCore` 改 `kCudaCore`，反而把 `FusedPartialKvSimplePattern` 也跑挂（maxErr 1.5）；说明 fused 的 `accumulate_d_f8` + `normalize_d<kTensorCore>` 是对的。
5. **不是 KV cache 的 head index**：fused 里 `kv_head = head_id / num_kv_groups`，`numKVHeads = num_qo_heads / num_kv_groups`，与原版 `qk_int_sv_f8_cuda_sm89.cuh` 一致；`launchSageAttentionFused` 里 `numKvGroups = numQoHeads / numKvHeads` 算正确。
6. **不是 V_scale_ch 索引**：`V_scale_ch + batch * Hkv * D + kv_head * D + dim`，与 `getSageVScaleSize(B, Hkv, D)` 的 `[B, Hkv, D]` layout 一致。
7. **不是 Q scale 索引**：`q_scale_idx = batch * Hq * num_warps_q + head_id * num_warps_q + warp_idx_q`，与 `getSageQScaleSize` 的 `[B, Hq, numBlocks * numWarpsPerBlock]` layout 一致（qL=1 时 numBlocks=1, numWarpsPerBlock=4）。

### 4.3 头号怀疑：**fused 在 GQA r=2 时不同 query head 之间共享了 KV smem 数据**

**理由**：fused 输出在第 0~4 dim 几乎是常数，意味着不同 dim 看到了几乎相同的 attention 权重 × V，而这通常是 **K/V 没载入到 smem**（出现全 0/未初始化），或者 **V_smem 被覆盖**，或者 **multi-warp 之间 K_smem 写入冲突**。

`FusedKernelMatchedQuant`（GQA r=1）和 `FusedPartialKvSimplePattern`（all-V=2.0 + Hq=Hkv=8）都通过，但前者 Hq=Hkv=4 太小可能藏住问题，后者数据太单一。

**关键问题**：fused kernel 的 **grid 是 `(1, numQoHeads, batchSize)`**，每个 block 处理一个 query head。一个 block 内 4 个 warp 协作（CTA_K=64=WARP_K，所以是 num_warps_k=1, num_warps_q=4）。每个 block 自己加载自己 KV head 的 K/V，**不同 block 不共享 smem**——所以理论上 GQA r=2 没区别。

但有一个**可疑点**：`grid(1, numQoHeads, batchSize)` 表示**每个 query head 独占一个 block，每个 block 自己重新做 K 量化和 V 量化**。GQA r=2 时 head 0 和 head 1 共享 KV head 0，它们各自的 block 都做一次 K 量化 → smem_K，然后 PV matmul。重复但不应错误。

**唯一与 r 相关的代码**：
- `kv_head = head_id / num_kv_groups`：r=2 时 head 0,1 → kv 0；head 2,3 → kv 1
- 没有其他 r 相关分支

### 4.4 第二怀疑：**单元测试自身的 KV cache 写入 layout 错误**

`FusedGQAFullKv` 的 KV cache 填充：
```cpp
uniformFloatInitialization(kvCacheInput, -1.f, 1.f);
```
**没有按 `[B, 2, Hkv, cap, D]` layout 摆放**，而是直接整段 random fill。这跟我之前的 `OriginalSageRunnerShapeRandom`（cap=4096 eff=21 pass rate 0.004）问题一样——但**`FusedKernelMatchedQuant` 是按 layout 摆的**，pass。

但**注意**：pre-fusion runner 在 `FusedGQAFullKv` 也吃同一份 KV cache，它和 fused 都基于同一 layout。如果 layout 错，两者**应该都错或者都对**——不会出现 pre-fusion 看起来像正常 attention（5 个值 0.78, 0.57, -0.84, -0.41, 0.11 这种多样化），而 fused 几乎是常数。

**所以第二怀疑站不住**，主要 bug 还在 fused kernel 内部。

### 4.5 第三怀疑：**K_smem zero-init + K 量化的 scope**

fused kernel 里 K_smem 在 K 量化前会 zero-init：

```cpp
for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += NUM_THREADS) {
    *reinterpret_cast<int8_t*>(smem + K_OFFSET + i) = 0;
}
```

但 `smem + K_OFFSET` 是 `uint32_t* base` 在 swizzled 上的视图，**直接 byte-stride 写 zero 与 swizzle 不一致**，可能写到错误位置。但 `FusedPartialKvSimplePattern` 通过说明这块（至少在 simple data 下）是可工作的。

### 4.6 接下来要做的最小化排查（按顺序）

1. **加一个 `FusedGQAFullKv_NoGQA` 变体**：Hq=Hkv=16, cap=64=eff，看 r=1 通过（应该通过，但 head 数大）；如果通过 → 锁定问题在 r>1 的处理上。
2. **打印 fused 内部 K_smem 内容**：在 head_id=0 和 head_id=1 各打一行 `K_smem[row=0, col=0..7]`，看 GQA r=2 时是不是一致（应该一致，因为同 kv_head=0）。head_id=2 应不同。
3. **打印 fused 内部 V_smem 内容**：同上。
4. **对比 fused 第一个 PV mm 的输出 RO[0][0]**：看 layer 0 step 0 head 0 dim 0 的值与 pre-fusion runner 在同点位的值。
5. **验证我的 RO store 修复是否对 GQA 生效**：用 `compute_fp8_sv` 内部某个固定值替换 V，跑 fused 看 RO 值能否还原 V_known。
6. **怀疑方向：`compute_fp8_sv` 里 `smem_V_col_base = (lane_id / 8) % 2`** —— 因为 V_SMEM_STRIDE=CTA_K=64，col 范围 0..3（每 16-byte cell），col_base 取值 0 或 1。如果不同 query head 同 KV head 的 V_smem 错位，这块会 leak。
7. **完整 dump fused 写入的 attentionOutputTensor[head=0, dim=0..127]**：看 dim 维度上的方差，如果接近 0 → fused 实际计算到的是个常数 vector。

### 4.7 单元测试的有效性问题

我自己写的几个 unit test 在 `cap=4096, eff=21` 时连 pre-fusion runner 都 pass rate ~0.004。这是因为测试的 reference 是 fp16 标准 attention（`launchSageReferenceAttention`），而 SageAttention 在 random data + 短 KV 下量化误差极大，**但与"runner 正确"不矛盾**：在真实推理时模型权重 + RoPE 的 K/V 分布让量化误差小很多（pre-fusion 推理输出 "Washington D.C." 就是证据）。

所以**用 fp16 reference 比较 fused 有干扰**。**正确的对比是 `fused vs pre-fusion runner`** —— 它们都吃同一组量化误差，差异只反映 fused 自己的 bug。这就是 `FusedVsPreFusionPluginShape` 的设计目的。

---

## 五、所有改动落点速查

| 文件 | 改动 | 状态 |
|------|------|------|
| `cpp/kernels/sageAttentionKernels/csrc/qattn/qk_int_sv_f8_cuda_sm89_fused.cuh` | VLOAD 还原 linear、VQUANT 改 inverse perm、RO→smem_O 抄原版、删 TRACE printf | 已应用，simple-pattern + matched-quant pass，GQA r=2 仍 fail |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | 删 effKvLen>=64 判定 + cudaMemcpy D2H + dump、删 `<fstream>`、把 fused 路径改成 pre-fusion 路径（commit f673b99 实现），用 `#if 1` 包裹的 fused-vs-pre-fusion diag 块 | 已应用，no-graph 推理输出正确（"Washington, D.C."） |
| `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp::captureDecodingCUDAGraph` | 当前 `return false;` 短路（专心 no-graph）；恢复需删除该行 | 临时 |
| `unittests/sageAttentionTest.cpp` | 新增 `FusedPluginShapeRandom`, `FusedPluginShapeSweep`, `OriginalSageRunnerShapeRandom`, `FusedVsPreFusionPluginShape`, `FusedGQAFullKv` | 已应用 |

---

## 六、Test 矩阵速查

| 测试名 | shape | KV layout | pre/fused 期望 | 实际 |
|--------|-------|-----------|---------------|------|
| FusedPartialKvSimplePattern | Hq=Hkv=8 cap=128 eff=21 V=2.0 | 整齐 | maxErr=0 | ✅ |
| FusedKernelMatchedQuant | Hq=Hkv=4 kL=cap=64 | host pre-quant | mismatches < N/2 | ✅ 0 mismatch |
| FusedKernelKInt8Compare | Hq=Hkv=4 kL=cap=64 | full | K_INT8 一致 | ✅ |
| Fp8Roundtrip | tiny | - | bit exact | ✅ |
| Fp8SmemRoundtrip | tiny | - | RO/d ≈ 448 | ❌ test 自身 RS_f8 构造 bug（与 fused 无关），优先级低 |
| FusedPluginShapeRandom | Hq=16 Hkv=8 cap=4096 eff=21 | plugin path | pr > 0.9 vs fp16 ref | ❌ pr 0.026（也包含 SageAttention 量化损失，不能孤立判断 fused） |
| FusedPluginShapeSweep × 6 | eff∈{21,64,128} × kMean∈{Y,N} | plugin path | 信息量参考 | 全部 0.02~0.04 vs fp16 ref（同上） |
| OriginalSageRunnerShapeRandom | Hq=16 Hkv=8 cap=4096 eff=21 | pre-fusion runner | 0.5+ vs fp16 ref | ❌ 0.004 — **测试自身用 fp16 ref 对 SageAttention 不公平** |
| FusedVsPreFusionPluginShape | Hq=16 Hkv=8 cap=4096 eff=21 | fused vs runner | pr > 0.95 | ❌ 0.006 — **目标修这个** |
| FusedGQAFullKv | Hq=16 Hkv=8 cap=eff=64 no K-mean | fused vs runner | pr > 0.9 | ❌ 0.004 — **比上一个更小化的复现** |

---

## 七、推理路径下的现状

无 K-mean diag、纯 fused（plugin diag 块改回 `#if 0` 后）：

| 模式 | XQA | Pre-fusion (f673b99) | Fused (当前) |
|------|-----|----------------------|--------------|
| no-graph | "Washington, D.C." ✅ | "Washington, D.C. (Washington, D.C.)" ✅ | 乱码 ❌ |
| graph capture | "Washington, D.C." ✅ | 应该可以（理论上无 cudaMemcpy 阻塞），未实测 | 当前因 `captureDecodingCUDAGraph` 短路成 `return false` 而**未启用 graph capture** — 待 fused 对了再开 |

---

## 八、回到正题的最短路径

1. 写 minimal 复现脚本 `FusedGQAR2Minimal`：Hq=2 Hkv=1 cap=64=eff，看是否还会出现"fused 输出是常数"——如果 r=2 + 最小 head 数还能复现，**bug 与 head 数无关，只与 r 有关**。
2. 在 fused kernel 内 head_id=0 和 head_id=1 各 dump 第一个 K_smem 字节、第一个 V_smem 字节、第一组 RO 值（warp 0 lane 0），与 pre-fusion runner 同样位置对比。
3. 核对 fused kernel 里**所有依赖 head_id 的索引**：
   - `kv_head = head_id / num_kv_groups`
   - `kv_k_base, kv_v_base`：`batch_id * 2 * numKVHeads * cap * D + kv_head * cap * D`
   - `q_scale_idx`：`batch * Hq * num_warps_q + head_id * num_warps_q + warp_idx_q`
   - K-mean: `k_mean[batch * numKVHeads * D + kv_head * D + dim]`
   - V_scale_ch: 同 kMean layout
   - fuse_v_scale: `vs_ptr = V_scale_ch + batch * actualNumKvHeads * D + (head_id / num_kv_groups) * D + ...`
   - 输出 stride: 用 `head_id`（query head），无 r 影响
4. 检查 K 量化 / V 量化里有没有用 query-head index 但应该是 kv-head index 的（或反之）。

---

## 九、注意事项 / 警告

- 每次改 `.cuh` 都要 `find build -name 'cmake_device_link.o' -delete && find build -name 'sageAttentionKernels*.o' -delete && bash compile.sh`，否则 device link 会用旧 kernel。
- `qk_int_sv_f8_cuda_sm89_fused.cuh` 现已无 printf，可放心 build；diag 全在 plugin 里。
- `unittests/sageAttentionTest.cpp` 末尾新增了一些 test，行号会随未来 edit 飘移；用 grep `FusedVsPreFusionPluginShape` / `FusedGQAFullKv` / `FusedPluginShapeSweep` 定位。
- `test_data/排查fused kernel精度问题.md` 是上一轮的笔记（V perm 修复部分），本文件是接力。
