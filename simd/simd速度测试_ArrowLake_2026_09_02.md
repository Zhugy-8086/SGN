# SIMD 原语层速度测试（Arrow Lake 本机实测 + AVX-VNNI CPUID 修复）

> **更新日期**: 2026-09-02
> **测试执行**: 本机（Intel Core Ultra 5 225 / Arrow Lake，Windows）
> **测试源**: `simd/sgn_benchmark.cpp`（CMake target，收编自 `_remote_test_decompressed/sgn_benchmark/` EPYC 归档，计算逻辑逐行一致）
> **本文件**: 速度测试结果速览；完整方法/分析见 [docs/SGN_ArrowLake速度测试归档_2026_09_02.md](../../docs/SGN_ArrowLake速度测试归档_2026_09_02.md)
> **关联**: [EPYC 归档 2026-08-31](../../docs/SGN_EPYC速度测试归档_2026_08_31.md)、主报告 §八（fixes_相关修复/msint_int8_pair_grad_carrier_2026_08_31.md）

---

## 本轮核心事件：暴露并修复 AVX-VNNI CPUID 检测双重错位

收编 sgn_benchmark 进 simd 主构建后，**本机首轮运行即暴露潜伏 bug**：

- 旧代码读 leaf7 **sub0** ECX[4] 判定 AVX-VNNI——该位实为 **OSPKE**；
  正确位置 = leaf7 **sub1 EAX[4]**（Intel SDM + Rust std 检测源码双确认）。
- **后果**：OSPKE=0 机器（本机 Arrow Lake/Windows）上 dot8/dot4 永不选入 VNNI 后端、
  静默落标量；EPYC/Linux 因 OSPKE=1 侥幸误判为 true 掩盖（单平台验证遮蔽平台相关
  缺陷，与 2026-08-29 batch_get_all SSSE3 潜伏 bug 同型）。
- **修复后后端名变化**：`avx2`（dot8 落标量）→ **`avxvnni`**。

## 测试环境

| 项目 | 详情 |
|---|---|
| CPU | Intel Core Ultra 5 225（Arrow Lake，6P+4E，family 6 model 198），无 AVX-512，OSPKE=0 |
| 系统 | Windows，PowerShell 7 |
| 编译 | Clang 22.1.8（c:\kaffj\），CMake + Ninja Release，`-O3`，per-file ISA（同 simd/CMakeLists.txt） |
| 运行时后端 | 修复前 `avx2`（dot8/dot4 静默落标量）→ 修复后 **`avxvnni`** |

## 正确性：238 项全部通过（修复前后各一轮）

与 EPYC 归档同一测试集（dot16 18 / dot8·dot4 28 / sum_f32 系 108 / decode 10 /
reverse·batch 64 / accum 10），全部 bit-exact（sum 系 kRounding 容差口径）。
simd_boundary_test（UBSan trap 模式）ALL PASSED（修复后复跑）。

## 性能：修复前 vs 修复后（同机同编译，唯一差异 = CPUID 读法）

| 原语 | K | 修复前 us/call（标量回退） | 修复后 us/call（avxvnni） | Mops/s 修复后 | 加速比 |
|---|---|---|---|---|---|
| dot8 | 1024 | 0.433 | 0.014 | 71,111 | ~31× |
| dot8 | 4096 | 1.811 | 0.044 | 93,516 | ~41× |
| dot8 | 16384 | 6.206 | 0.188 | 87,120 | ~33× |
| dot8 | 65536 | 28.758 | 1.344 | 48,757 | ~21× |
| dot4 | 65536 | 27.166 | 1.251 | 52,366 | ~22× |
| dot16 | 65536 | 4.399 | 4.114 | 15,929 | ~1.07×（dot16 走 avx2，不受影响 ✓） |

> 交叉核对：修复前 dot8 ≈ 2,279 Mops/s ≈ 标量锚点水平——与"静默落标量"的推断一致；
> 修复后 vpdpbusd 256 位 VNNI 峰值 93.5k Mops/s（K=4096，驻 L2）。
> 本机无 AVX-512，dot8/dot4 峰值低于 EPYC 512 位路径（57.6k Mops/s @2vCPU 虚拟机），
> 属硬件差异而非实现差异。

## 复现

```powershell
cd engine/sgn/simd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sgn_benchmark
.\build\sgn_benchmark.exe              # 人类可读 + CWD sgn_benchmark_result.json
# 强制后端对比：$env:SGN_KERNEL_BACKEND="scalar"; .\build\sgn_benchmark.exe
```
