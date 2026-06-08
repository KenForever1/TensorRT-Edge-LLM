# 排查 fused kernel 精度问题 — 最终报告

## 现象

| 场景 | 结果 |
|------|------|
| Fused, full KV (eff=cap) | ✅ 0 错误 |
| Fused, partial KV (eff=21, cap=512) | ❌ 25% 错误率 |
| 简单模式 Q=1,K=0.5,V=2, eff=21 | ❌ output=1.81 (期望 2.0) |
| XQA vs Reference | ✅ 0 错误 |
| Original SageAttention vs Reference (eff=21) | ✅ pass_rate=1.0 |

---

## 根因定位

### Trace 数据

```
d = 9408 = 21 × 448        ← RS_FP8=448 per valid position ✅
RO = 3813376 = 19 × 448²   ← 期望 21 × 448² ❌ (差了 2 个)
RO/d = 405.33              ← 期望 448
```

**d=21×448 说明 21 个有效位置的 RS（softmax 权重）正确**，但 **RO 只有 19 个有效位置贡献了 V×RS**，2 个位置的 V_FP8 被写成了 0。

### FP8 硬件验证

```
__nv_fp8_e4m3(448.0f) → 0x7e → decode 448.0 ✅
V smem → TensorCore PV matmul → RO 精确匹配 ✅
```

FP8 编码/解码和 TensorCore WMMA 硬件完全正确，bug 在 V 值写入 smem 之前的逻辑。

### 根因：V 量化时 permuted index 读取错误

Fused kernel 的 V 量化代码（`qk_int_sv_f8_cuda_sm89_fused.cuh` 行 ~285）：

```cpp
// logical_seq = 当前线程处理的逻辑序列位置 (0-63)
// 应用 SageAttention permutation: [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]
uint32_t mod16 = logical_seq % 16;
uint32_t ps = (mod16 / 8) * 2 + ((mod16 / 2) % 4) * 4 + (mod16 % 2);
uint32_t seq_in_tile = (logical_seq / 16) * 16 + ps;  // permuted position

// 🔴 BUG: 从 temp[permuted] 读，但 temp 是 LINEAR 顺序
float fv = __half2float(temp[seq_in_tile * HEAD_DIM + dim_group]);
```

**V 从 KV cache 加载到 temp 时是 linear 顺序**（temp[0]=V(seq0), temp[1*128]=V(seq1), ...）。

但量化时**用 permuted index `seq_in_tile` 去读**。对于 `logical_seq=19`：
- permutation: 19 → seq_in_tile = 21
- temp[21*128+dim] = V(seq21) = **0** (OOB 位置，超出 effective_kv_len=21)
- 期望读到 V(19) = 2.0

**有效位置 19 的 V 值被错误读成了 OOB 位置 21 的 V=0。**

原版 SageAttention kernel 不受影响，因为它用 `load_fp8_V_global_to_share` 从 global memory 加载——global memory 里 V_FP8 是 convert pipeline 写入的 **permuted 顺序**，所以 permuted index 读取是正确的。

---

## 修复方案

### 方案：V 加载时应用 permutation

在 V 从 KV cache 加载到 temp 时，就按照 permuted 顺序存储。这样量化时用 `seq_in_tile`（permuted index）读就能读到正确的 V 值。

```cpp
// 修改前（linear 加载）:
for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += NUM_THREADS) {
    uint32_t seq_in_tile = i / HEAD_DIM;
    uint32_t dim = i % HEAD_DIM;
    uint32_t seq = seq_start + seq_in_tile;
    bool valid = seq < effective_kv_len;
    temp[i] = valid ? kv_cache[seq * HEAD_DIM + dim] : 0;  // linear
}

// 修改后（permuted 加载）:
for (uint32_t i = tid; i < CTA_K * HEAD_DIM; i += NUM_THREADS) {
    uint32_t seq_in_tile = i / HEAD_DIM;
    uint32_t dim = i % HEAD_DIM;
    uint32_t seq = seq_start + seq_in_tile;
    bool valid = seq < effective_kv_len;
    half val = valid ? kv_cache[seq * HEAD_DIM + dim] : 0;
    // Apply same permutation to storage position
    uint32_t mod16 = seq_in_tile % 16;
    uint32_t ps = (mod16 / 8) * 2 + ((mod16 / 2) % 4) * 4 + (mod16 % 2);
    uint32_t permuted_seq = (seq_in_tile / 16) * 16 + ps;
    temp[permuted_seq * HEAD_DIM + dim] = val;  // permuted storage
}
```

这样 `temp[permuted_seq]` = V(seq)，量化时读 `temp[seq_in_tile]` = V(logical_seq)。✓

### 为什么之前的测试没通过

1. 编译后设备链接对象 (`cmake_device_link.o`) 可能未重新生成，导致旧 kernel 仍被使用
2. 需要 `find build -name "cmake_device_link.o" -delete` 强制重链

### 验证方法

```
./build/unitTest --gtest_filter="SageAttentionTest.FusedPartialKvSimplePattern"
```

期望输出：`maxErr from 2.0 = 0` 或很小。

---

## 涉及的 K 量化问题（次要）

K 量化有类似问题（OOB 位置的 K=0 经过 K-mean 减法后被量化到饱和值 ±127），但由于 K 使用 `load_global_to_share` 不同的 swizzle 模式（k128B vs k64B），且 OOB mask 进一步修正，影响相对较小。

当前有两个 K 侧修复已在代码中但并非根因：
1. `if (row >= num_valid) qi = 0` — OOB 位置强制 K_INT8=0
2. K smem zero-init — 确保未覆盖位置为 0

---

## 工作区代码改动

### `qk_int_sv_f8_cuda_sm89_fused.cuh`
- V 量化：`seq_in_tile` → `logical_seq` 条件检查和读取（**已 revert**，待用方案修复）
- K OOB fix + zero-init（保留）
- OOB mask: unconditional + global effective_kv_len（保留）
- **TRACE** 和 **VFP8** debug printf（待清理）
- **`<iostream>` include via ofstream?  No, this is a .cuh file**

### `attentionPlugin.cpp`
- workaround: `effKvLen >= 64` 才启用 fused（每 step 读一次 GPU context length）
- debug dump: meta[8] 32 bytes
- 移除 FUSEDvsXQA 对比代码和 static debug 变量
- 移除 FUSED_BLOCK_ENTERED printf

### `sageAttentionTest.cpp`
- 新增: PluginExactPathDecode, PluginExactPathPartialKvCache, FusedVsOriginalPartialKV, FusedPartialKvSimplePattern, CompareBothWithReference, CompareWithPluginData/PerHead (修复 qScaleSize/vScaleSize)
- 新增: Fp8Roundtrip, Fp8SmemRoundtrip (FP8 验证测试)

### `sageAttentionTestKernels.cu/h`
- 新增: launchFp8RoundtripTest, launchFp8SmemRoundtripTest

---

## 2026-06-07 接力排查（Ducc）

### 之前报告的不足

之前的报告把根因定位在「V 量化用 permuted index 读，但 temp 是 linear 顺序」，并把"修复"理解为「让 V 加载时也按 permuted 顺序写 temp」。把改动应用到 `qk_int_sv_f8_cuda_sm89_fused.cuh` 后，simple-pattern 测试的 maxErr 仍是 0.19（和未改前一致），`RO=3813376=19×448²`、`d=9408=21×448`。这说明那一版"修复"是错的——在 VLOAD 和 VQUANT 两端各加一次 forward permutation 等价于互相抵消，结果 `smem_V[d, j] = V_logical[j]`。

### 真正的对齐关系

PV matmul 对应的 V smem 行/列设计是 **row=dim, col=seq**（与原版 SageAttention `qk_int_sv_f8_cuda_sm89.cuh` 的 V_lane 路径一致，使用 `stride_d_v`）。原版仓库的 `transpose_pad_permute` 在写出 V tensor 时用的是 forward permutation（`fused.cu:290`）：

```
V_global[d, perm(token_i)] = V_logical[d, token_i]
```

这个 V tensor 通过 `load_fp8_V_global_to_share` 直接 cp_async 进 smem，**线性加载**意味着：

```
smem_V[d, j] = V_global[d, j] = V_logical[d, perm⁻¹(j)]
```

也就是说 **PV matmul 期望 `smem_V[d, j] = V_logical[d, perm⁻¹(j)]`**，注意是 `perm⁻¹`，不是 `perm`。

而 fused kernel 必须自己在 smem 内写出这种布局：
- VLOAD：linear 写 temp，`temp[i, d] = V_logical[seq_start+i, d]`
- VQUANT：写 `smem_V[d, j] = fp8(temp[perm⁻¹(j), d])`

### Permutation 的位运算理解

forward perm（mod16，4 bit `b3 b2 b1 b0`）→ `b2 b1 b3 b0`：
```
ps = (mod16 / 8) * 2 + ((mod16 / 2) % 4) * 4 + (mod16 % 2)
```

逆 perm（mod16 `c3 c2 c1 c0`）→ `c1 c3 c2 c0`：
```
inv_ps = ((mod16 >> 1) & 1) * 8
       + ((mod16 >> 3) & 1) * 4
       + ((mod16 >> 2) & 1) * 2
       + (mod16 & 1)
```

forward 不是 involution（perm(perm(2)) = perm(4) = 8 ≠ 2），所以两次 forward 会出错。

### 修复

`qk_int_sv_f8_cuda_sm89_fused.cuh`：

1. **VLOAD 改回 linear**（`temp[i, d] = V_logical[seq_start + i, d]`）。
2. **VQUANT 用 inv_ps 读 temp**（之前是 forward perm）。

关键 diff：

```cpp
// VQUANT loop (around line 285)
uint32_t mod16 = logical_seq % 16;
// 旧：forward perm（错误）
// uint32_t ps = (mod16 / 8) * 2 + ((mod16 / 2) % 4) * 4 + (mod16 % 2);
// 新：inverse perm
uint32_t inv_ps = ((mod16 >> 1) & 1) * 8
                + ((mod16 >> 3) & 1) * 4
                + ((mod16 >> 2) & 1) * 2
                + (mod16 & 1);
uint32_t seq_in_tile = (logical_seq / 16) * 16 + inv_ps;
```

### 验证

强制重链：

```bash
find build -name 'cmake_device_link.o' -delete
find build -name 'sageAttentionKernels*.o' -delete
bash compile.sh
```

跑 simple pattern：

```
./build/unitTest --gtest_filter='SageAttentionTest.FusedPartialKvSimplePattern'
```

输出：
```
TRACE post-pv: RO[0][0]=4214784.0 d[0]=9408.0          ← 21×448² 全部贡献 ✓
TRACE post-norm: RO[0][0]=448.0000 d[0]=9408.0000
FusedSimplePattern: maxErr from 2.0 = 0 first 4 values: 2 2 2 2
[       OK ] SageAttentionTest.FusedPartialKvSimplePattern (510 ms)
```

完整套件：

```
./build/unitTest --gtest_filter='SageAttentionTest.*'
[  PASSED  ] 23 tests.
[  SKIPPED ] 4 tests   ← /tmp/fused_debug.bin 缺失，跟修复无关
[  FAILED  ] 3 tests
```

### 仍未解决的 3 个测试

**1. `SageAttentionTest.Fp8SmemRoundtrip`**：与 V permutation 修复无关，是测试 kernel 自身的 bug。
- 输出 `d = 28672 = 64 × 448` 而期望 `9408 = 21 × 448`。
- 看 `unittests/sageAttentionTestKernels.cu:343-358`，构造 `RS_f8` 时 `kv_idx = K_IDX_BASE + fk * 16 + 8 * (k / 4) + k % 2` —— 因 `k` 只到 3，`k/4 == 0`，所以 `kv_idx ∈ {0,1,16,17}` 全部 < 21，导致 64 个位置都被填了 448。
- 修法：构造 RS_f8 时应该把 `kv_idx >= 21` 的位置填 0，或者按真正的 RS 输出 lane→element 映射重写测试。
- 优先级：低（测试 bug，不影响 kernel 正确性）。

**2. `SageAttentionTest.PluginExactPathPartialKvCache`**：random data, eff=21, cap=512，pass rate 0.776（期望 > 0.90）。
- 这个测试 ref 是 fp16 reference（且包含 K-mean centering），fused 路径不传 k_mean。
- pass rate 0.78 比 修复前明显改善，但仍未到 0.90。
- 可能的剩余因素：
  - **K-mean 缺失**：`launchSageAttentionFused` 调用 `nullptr` 当 k_mean。开 K-mean 后 K 量化 dynamic range 更窄，fp8 精度更高。
  - **OOB K 经过 K-mean 减法**：如果开 K-mean，OOB 位置（kv_cache=0）量化后是 `(0 - km) / kScale`，非 0，可能干扰 RS。kernel 里有 `if (row >= num_valid) qi = 0` 强置，应已防护。
  - **kScale 跨 tile 与 reference 不一致**：fused 在 partial KV 下 kScale 是「`num_valid * HEAD_DIM` 的 max」，reference convert pipeline 在 partial KV 下也只对 effective 内做 reduce，应该一致。
  - **OOB mask 阈值**：`apply_out_of_bound_mask` 用 `effective_kv_len` 全局判定，应该正确。
- 建议下一步：分两组对比——
  - (a) fused 与 ref（K-mean + apply_out_of_bound_mask 等价开/关），看是否 K-mean 是主要差距。
  - (b) 把 fused 的 K_INT8 dump 出来，跟 host 用相同 kScale + km 量化得到的 K_INT8 对比，定位 K 量化偏差。
- 优先级：中。

**3. `SageAttentionTest.FusedVsOriginalPartialKV`**：random data, eff=21, cap=512，fused vs original kernel 有 2048/2048 mismatch。
- 这个测试 expectation 是 `mM < qSize/2`（不强制收敛），目前 mM=2048 > 1024。
- 期望 fused 与 original kernel 在同一组 K-mean / V-scale 下输出近似一致。但调用差异：
  - fused: `launchSageComputeVScalesAndKMeans` + `launchSageAttentionFused`（k_mean 传给 kernel，kernel 内部走 K-mean 路径）。
  - original: `launchSageConvertKVCacheToInt8AndFp8` 在 host 侧做 K-mean 减法 + INT8 量化，传 K_INT8 给 kernel。
- 看起来两条路径对 K-mean 的处理时机不同：fused 在 tile 内对 raw fp16 K 做 `(fv - km) / kScale`，且 kScale 是从 raw fp16 K 算出的；convert pipeline 是对 `(K - mean)` 整体 reduce 算 kScale。两个 kScale 不等价 → 量化误差不同。
- 同样，V scale 一边是 `launchSageComputeVScalesAndKMeans`（include K-mean stats？），一边是 `launchSageConvertKVCacheToInt8AndFp8`，需要核对两个 V scale 计算路径是否一致。
- 优先级：中（暴露的是 fused vs original kernel 的算法差异，不一定是 fused 自己的 bug）。

### 工作区改动（本次）

- `cpp/kernels/sageAttentionKernels/csrc/qattn/qk_int_sv_f8_cuda_sm89_fused.cuh`
  - VLOAD 还原为 linear 加载，去掉之前的 permuted 写法和 `VLOAD seq=...` debug printf。
  - VQUANT 把 forward perm 改成 inverse perm，去掉 `VQUANT logical=...` debug printf。
  - 仍保留：K OOB 强置 0 / K smem zero-init / OOB mask 用 global effective_kv_len / `TRACE post-acc_d/post-pv/post-norm` printf。

### 备忘：debug 残留

`qk_int_sv_f8_cuda_sm89_fused.cuh` 里仍有 3 个 `printf("TRACE ...")`（行 ~340, 353, 363），等剩余两个测试也排查清后再清理。
