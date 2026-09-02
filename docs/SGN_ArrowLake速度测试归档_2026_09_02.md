# SGN Arrow Lake 速度测试归档（2026-09-02，含 AVX-VNNI CPUID 潜伏 bug 修复）

> **更新日期**: 2026-09-02
> **测试执行**: 本机（Intel Core Ultra 5 225 / Arrow Lake，Windows）
> **测试源**: `engine/sgn/simd/sgn_benchmark.cpp`（本轮收编进 simd 主构建的 portable 版本）
> **关联归档**: [SGN_EPYC速度测试归档_2026_08_31.md](SGN_EPYC速度测试归档_2026_08_31.md)（远程 EPYC 对照）
> **速览版**: [engine/sgn/simd/simd速度测试_ArrowLake_2026_09_02.md](../engine/sgn/simd/simd速度测试_ArrowLake_2026_09_02.md)
> **本轮背景**: int8 对（h,l）梯度载体引擎侧吞吐基准（主报告 §八，统一数学框架 #32）

---

## 1. 背景

EPYC 远程验证（2026-08-31）完成了 AVX-512/512-VNNI 路径的真指令验证，但存在
**单平台验证盲区**：本机（Arrow Lake，无 AVX-512）此前只能做逻辑验证。本轮将
EPYC 归档的 `sgn_benchmark.cpp` 收编进 `simd/CMakeLists.txt` 正式 target
（portable 化：chrono 计时 + CWD JSON 路径，计算逻辑逐行一致），使原语基准可在
任意机器一键构建——**收编后本机首轮运行即暴露一个潜伏 CPUID bug**。

## 2. 潜伏 bug：AVX-VNNI CPUID 检测双重错位

### 2.1 错误内容

`simd_dispatch.cpp` 旧代码以 **leaf7 sub0 ECX[4]** 判定 AVX-VNNI。两处错误叠加：

| 层面 | 旧代码 | 正确值 | 依据 |
|------|--------|--------|------|
| subleaf | sub0 | **sub1** | sub0 ECX[4] 实为 **OSPKE**（OS 启用保护密钥）位 |
| 寄存器 | ECX | **EAX** | Intel SDM：CPUID.7.1.EAX[4]=AVX-VNNI（同寄存器 bit5=AVX512-BF16）；Rust std 检测源码同确认 |

### 2.2 为什么 EPYC 验证没有发现

| 平台 | OSPKE（sub0 ECX[4] 实际值） | 旧代码判定 | dot8/dot4 实际后端 |
|------|------|------|------|
| EPYC / Ubuntu（远程） | 1（Linux 使能了保护密钥） | 误判为"有 VNNI" ✓ 侥幸 | 有 AVX-512VNNI，走 512 路径，256 位 VNNI 从未被真正考验 |
| Arrow Lake / Windows（本机） | 0 | 判定"无 VNNI" | 静默落标量（~2.3k Mops/s） |

即：旧 bug 在 EPYC 上**方向凑巧无害**（OSPKE=1 → 误 true，而真 VNNI 由 512 位路径
掩盖），在本机上**方向保守**（误 false → 静默落标量，性能损失但不崩溃）。若某台
OSPKE=0 且只有 256 位 VNNI 的机器曾被误判为 true 才会崩溃——本组合未出现，故潜伏。
**与 2026-08-29 batch_get_all SSSE3 潜伏 bug 同型：单平台验证遮蔽平台相关缺陷。**

> ⚠️ 勘误连带：EPYC 归档 §2 注记"Zen 4 的 AVX-VNNI 在 CPUID leaf7 sub0 ECX[4]=1"
> 沿用了同一错误位定义（该位是 OSPKE）。EPYC 上实测数值不受影响（走 512 路径），
> 但该注记的位定义解读作废。

### 2.3 修复与验证

- `cpuid_leaf` 增加显式 subleaf 参数（`__cpuidex`/`__cpuid_count`）；
  `c.avx_vnni = sub1 EAX[4]`；位定义注释同步修正。
- 验证链：sgn_benchmark 238/238 bit-exact（修复前后各一轮）→ simd_boundary_test
  （UBSan trap）ALL PASSED → `SGN_KERNEL_BACKEND=scalar` 强制标量回归正常 →
  .pyd 重编后 Python 端 `dot_fused(16,8)` 对 int64 参考 bit-exact。
- 后端名演变：`avx2`（dot8 落标量）→ **`avxvnni`**。

## 3. 测试环境

| 项目 | 详情 |
|---|---|
| CPU | Intel Core Ultra 5 225（Arrow Lake-S，6P+4E，family 6 model 198） |
| 关键指令集 | AVX2, AVX-VNNI（修复后正确检出）, GFNI, VAES；无 AVX-512/F/BW/VL，无 AVX512-VNNI；OSPKE=0 |
| OS | Windows（PowerShell 7 执行） |
| 编译器 | Clang 22.1.8（c:\kaffj\clang+llvm-22.1.8-x86_64-pc-windows-msvc） |
| 构建 | CMake + Ninja，Release，`-O3`；per-file ISA：avx2(-mavx2) / avxvnni(-mavx2 -mavxvnni) / ssse3(-mssse3 -mavx2) / avx512 系各自成套；dispatch/scalar 基础编译（运行时 CPUID 门控） |
| 运行时后端 | 修复前 `avx2`（dot8/dot4 静默落标量）→ 修复后 **`avxvnni`** |

## 4. 测试方法

与 EPYC 归档 §3 完全一致（同一测试逻辑，chrono 计时替代 clock_gettime，
steady_clock 语义等价）：预热 50 次、volatile sink 防死代码消除、
us/call + Mops/s 口径、K∈{1024,4096,16384,65536}。
修复前后两轮数据同机同编译，**唯一差异 = CPUID 读法**，构成干净的 A/B 对照。

## 5. 性能结果（修复前 vs 修复后 A/B 对照）

### 5.1 原语层（sgn_benchmark）

| 原语 | K | 修复前 us/call | 修复前 Mops/s | 修复后 us/call | 修复后 Mops/s | 加速比 |
|---|---|---|---|---|---|---|
| dot8 | 1024 | 0.433 | 2,364 | 0.014 | 71,111 | ~31× |
| dot8 | 4096 | 1.811 | 2,261 | 0.044 | 93,516 | ~41× |
| dot8 | 16384 | 6.206 | 2,640 | 0.188 | 87,120 | ~33× |
| dot8 | 65536 | 28.758 | 2,279 | 1.344 | 48,757 | ~21× |
| dot4 | 1024 | 0.512 | 2,001 | 0.015 | 69,957 | ~35× |
| dot4 | 65536 | 27.166 | 2,412 | 1.251 | 52,366 | ~22× |
| dot16 | 65536 | 4.399 | 14,896 | 4.114 | 15,929 | ~1.07×（走 avx2，不受影响 ✓） |
| sum_f32 | 65536 | 7.199 | 9,103 | 6.153 | 10,537 | ~1.17×（波动内） |
| decode | 65536 | 15.412 | 4,252 | 14.905 | 4,397 | ~1.03×（波动内） |

要点：
- **修复前 dot8/dot4 ≈ 2.0–2.6k Mops/s = 标量锚点水平**，与"静默落标量"推断一致；
  dot16/sum/decode（avx2/ssse3 后端）前后一致，确认 A/B 差异仅来自 VNNI 检测。
- 修复后 vpdpbusd 峰值 **93.5k Mops/s**（K=4096，驻 L2），K=65536 降至 48.8k
  （L2→DRAM 带宽墙，同 EPYC 归档形态）。
- 与 EPYC 对照：本机 256 位 VNNI 峰值（93.5k）> EPYC 虚拟机 512 位 VNNI（57.6k @2vCPU），
  主因是 EPYC 为 2 vCPU 虚拟机 + 频率差；架构归架构，**不构成跨平台性能结论**。

### 5.2 消费端端到端（.pyd：dot_split(16,8)，bench_grad_int8_pair_dot8.py）

| K | 修复前（标量回退） | 修复后（avxvnni） | 端到端加速 |
|---|---|---|---|
| 1024 | 0.115 ms | 0.036 ms | 3.2× |
| 4096 | 0.525 ms | 0.146 ms | 3.6× |
| 16384 | 2.573 ms | 0.650 ms | 4.0× |
| 65536 | 10.105 ms | 3.515 ms | 2.9× |

### 5.3 关键定性：入口决定瓶颈（对 int8 对载体落地的约束）

原语层 ~30× 在 Python list 入口仅兑现 2.9–4.0×：K=16384 端到端 650 µs 中 dot8 计算
仅 0.75 µs（4×0.188 µs，<0.2%），其余 99.8% 为 per-element 拆分+打包
（split_parts_fixed + pack_narrow_value）与 Python 绑定转换。

- pair-fine"省一半 dot8"（4→2 次）在 Python list 入口端到端不可见（<0.1%）；
- §七 成本模型（fine 2 次/coarse 1 次 dot8）在 **C++ 内部预打包消费路径**
  （narrow_dot 直调，x 侧打包跨 M 行摊销）才完全兑现；
- **落地约束**：引擎侧 pair 路径（反向末端"写 pair"+ StrategyContext 开关）须
  C++ 内部闭环，避免 Python 往返；打包成本按 prepare_nibble_from_raw 同型优化
  （连续预分配 + 直写）处理。

## 6. 验证体汇总

| 资产 | 位置 | 状态 |
|------|------|------|
| 原语基准（收编版） | simd/sgn_benchmark.cpp + simd/CMakeLists.txt target | 238/238 bit-exact ×2 轮 |
| EPYC 归档原件 | simd/_remote_test_decompressed/sgn_benchmark/ | 保留不动 |
| 边界测试 | simd/simd_boundary_test.cpp（UBSan） | ALL PASSED |
| 消费端基准 | engine/sgn/tests/architecture/bench_grad_int8_pair_dot8.py | 修复前后双轮 + scalar 回归 |
| CPUID 修复 | simd/simd_dispatch.cpp | commit 9041faf |
| 数学层（§七） | math_verification/verification/validate_math_grad_int8_pair_dot8.py | V1-V7 双库 7/7 |

## 7. 遗留事项

1. **远程 EPYC 复核**：在 EPYC 上用收编版基准复跑一轮，确认 sub1 EAX[4] 读法在 AMD
   （Zen 4 有真 256 位 VNNI）上同样正确（低成本闭环，消除 2.2 节盲区）。
2. CPUID 位定义测试化：将 subleaf/寄存器期望值做成断言性单测（防再次读错位）。
