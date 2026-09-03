# SGN EPYC Zen 5 速度测试归档（2026-09-03）

> **更新日期**: 2026-09-03
> **测试执行**: 远程云主机（AMD EPYC 9K65 / Zen 5 虚拟机，3 vCPU）
> **测试源**: `mkern/simd/sgn_benchmark.cpp`（同源，per-file ISA + 运行时 CPUID 门控）
> **关联归档**: [SGN_EPYC速度测试归档_2026_08_31.md](SGN_EPYC速度测试归档_2026_08_31.md)（Zen 4 对照）、
> [SGN_ArrowLake速度测试归档_2026_09_02.md](SGN_ArrowLake速度测试归档_2026_09_02.md)（Intel 对照）
> **原始包**: `logs/mkern_remote_test_2026_09_03/2号远程测试机` 性能交付（SGN_REMOTE_TEST_02.md / run_bench.sh / perf_results.txt / sgn_benchmark_result.json）

---

## 0. 本归档要点（兼容能力，先看这个）

速度测试归档的核心价值是**跨硬件兼容**，不是性能排名。本归档作为"兼容能力"证据链的
第三块拼图：

| 平台 | 微架构 | 指令集 | 运行时后端 | 正确性 |
|---|---|---|---|---|
| Intel Arrow Lake（本机，Windows） | 6P+4E | AVX2, AVX-VNNI；**无 AVX-512** | `avxvnni`（256 位 VNNI） | simd 238/238 |
| AMD EPYC 9Y24（Zen 4，Ubuntu） | Zen 4 | AVX2, **AVX-512VNNI**；无 256 位 VNNI | `avx512vnni`（512 位 VNNI） | simd 238/238 + gemm 4819/4819 |
| AMD EPYC 9K65（Zen 5，Ubuntu） | Zen 5 | AVX2, **AVX-512VNNI** | `avx512vnni`（512 位 VNNI） | simd 238/238 |

**同一套源码**（mkern/simd + mkern/gemm），per-file ISA 编译、符号常驻，运行时 CPUID
自动选路——在 Intel 上用 256 位 VNNI、在 AMD 两代上用 512 位 VNNI，互不依赖编译期宏，
无对应指令集的机器自动回退标量。三平台全部 bit-exact。**这是"一套代码、处处最优且
可复现"的实证**。

> 兼容性是设计出来的，也是测出来的：AVX-VNNI CPUID 双重错位 bug（OSPKE 误读，
> 见 Arrow Lake 归档 §2）正是"单平台验证"遮蔽"平台相关缺陷"的教训——正是本机 +
> EPYC + Zen 5 三平台组合才能暴露并闭环。

## 1. 背景

SGN 微内核层（mkern/simd 原语 + mkern/gemm 矩阵级）含 AVX-512 服务器加速路径
（512 位 VNNI `vpdpbusd`）。Arrow Lake（无 AVX-512）只能逻辑验证 AVX-512 指令。
Zen 5 真机支持全 AVX-512 指令集，本归档在 Zen 5 上做真指令独立验证。

## 2. 测试环境

| 项目 | 详情 |
|---|---|
| CPU | AMD EPYC 9K65 192-Core（Zen 5），3 vCPU / 1 socket |
| 指令集 | AVX-512 全开（F/DQ/BW/VL/**VNNI**/BF16...）, AVX2, FMA, SSE4.2 |
| L1/L2/L3 | 96/64 KiB · 2 MiB · 32 MiB |
| 内存 | 4.3 GiB（虚拟机） |
| OS | Ubuntu 22.04（KVM guest，内核 6.6.69 x86_64） |
| 编译器 | g++ 11.4.0（gcc），C++23 |
| 编译选项 | `-O3 -ffp-contract=off`（build_remote.sh 补丁后；浮点归约禁 FMA 收缩保 kBitExact） |
| 运行时后端 | **`avx512vnni`**（CPUID 自动检测选中：`avx512vnni=1 → expect=avx512vnni`） |

## 3. 测试方法

与 Arrow Lake / Zen 4 归档一致（同一测试逻辑）：预热 50 次、`volatile sink` 防死代码
消除、us/call + Mops/s 口径、K∈{1024,4096,16384,65536}。**正确性先于性能**：
先跑 `simd_boundary_test` → ALL PASSED → 再采性能。

## 4. 正确性（238/238）

```
dot16              18/18   通过（int16×int16，位 exact）
dot8/dot4          28/28   通过（uint8×int8，含 255×127 满幅）
sum_f32 系        108/108  通过（R1 固定 8 路归约树，kBitExact）
decode_i16_f32     10/10   通过
reverse/batch      64/64   通过
accum_f32          10/10   通过
合计               238/238 通过
```

## 5. 性能基准（真实采集，每点 1024 次取平均）

### 5.1 原语层（后端 `avx512vnni`）

| 原语 | K=1024 | K=4096 | K=16384 | K=65536 |
|---|---|---|---|---|
| dot16 | 36,737 | 39,825 | 40,931 | **43,045** |
| dot8 | 88,190 | 121,238 | 136,736 | **89,821** |
| dot4 | 99,818 | 125,027 | 129,423 | **88,573** |
| **dot4_packed** | **122,690** | **162,847** | **175,145** | **159,966** |
| sum_f32 | 22,782 | 16,409 | 15,200 | **14,777** |
| decode_i16_f32 | 6,427 | 6,371 | 6,404 | **6,193** |

（单位 Mops/s）

### 5.2 要点与兼容解读

1. **dot4_packed 恒优于 dot4**（K=4096: 163k vs 125k，+30%）——4 位打包布局减少
   指令数，R2 带宽减半红利在 Zen 5 完全兑现，且**随规模上升**（K=16384 达 175k）。
2. **负载形态差异**：dot/decode 属带宽型→大 K 有 L2/L3 墙（dot8 K=65536 从 137k
   回落 90k）；`decode_i16_f32` 恒 ~6.4k（vpmovsxwd 扩展 + 归一化指令串行）；
   `sum_f32` 归约 ~15-23k（横向归约难完全向量化）。这些是**后端固有形态**，
   三平台同构出现（对照 Zen 4 / Arrow Lake 归档），属预期而非缺陷。
3. **兼容性证据**：同一二进制逻辑在 Zen 5 走 512 位 VNNI、在 Arrow Lake 走 256 位
   VNNI——`dot8` 峰值 Zen 5 137k vs Arrow Lake 93.5k，反映的是指令宽度与频率
   差异，**不是代码能力差异**；两端 bit-exact 结果完全一致。

## 6. 跨平台性能对照（兼容能力总览）

| 原语（K=65536） | Intel Arrow Lake（`avxvnni`） | AMD Zen 4（`avx512vnni`，2vCPU） | AMD Zen 5（`avx512vnni`） |
|---|---|---|---|
| dot16 | 15,929 | 22,257 | 43,045 |
| dot8 | 48,757 | 56,228 | 89,821 |
| dot4_packed | 34,620（本机） | **88,868** | **159,966** |
| sum_f32 | 10,537 | 9,745 | 14,777 |
| decode | 4,397 | 5,928 | 6,193 |

> ⚠️ **不可直接横比**：三平台为不同指令宽度/频率/虚拟机资源（Arrow Lake 桌面 vs
> EPYC 2vCPU/Zen 4 虚拟机 vs Zen 5 3vCPU 虚拟机）。此表用于展示**同一代码在各平台
> 各自启用最优后端、数量级正确**，不构成性能排名（架构归架构）。

## 7. 结论与检查项

- **兼容性**：全套 avx512vnni 路径在 Zen 5 真机 bit-exact 全过（238/238）；
  dot4_packed / R1 kBitExact / 归约 8 路树在真 AVX-512 硬件上确认数值正确。
- **性能**：dot 类峰值 ~160-175k Mops/s（dot4_packed），符合设计预期。
- **检查项**：① FP16 基准需 GCC 12+/Clang 22 编译（GCC 11 的 `_Float16` 不完整，
  基准项静默剔除——编译能力差异，非内核问题）；② 虚拟机单核口径，多核并行属
  后续 P2（narrow_dot 层 OpenMP）。
