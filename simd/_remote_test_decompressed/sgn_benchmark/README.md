# SGN SIMD 原语层测试与基准

对 GitHub `Zhugy-8086/SGN` 仓库 `simd/` 目录的独立测试：正确性验证 + 性能基准。

## 测试环境

| 项目 | 详情 |
|---|---|
| **CPU 型号** | AMD EPYC 9Y24 96-Core Processor |
| **架构** | AMD Zen 4（AuthenticAMD, family 25, model 17, stepping 1） |
| **虚拟化** | KVM 完全虚拟化（full virtualization） |
| **分配 vCPU** | 2 核（2 socket × 1 core/socket，无超线程） |
| **物理机规格** | 96 核 192 线程（型号名所示，虚拟机仅分配 2 vCPU） |
| **运行频率** | 2596 MHz（约 2.6 GHz） |
| **内存** | 4 GB（4130368 kB，虚拟机分配；底层为 EPYC 平台 DDR5） |
| **内存类型/频率** | 无法获取（虚拟机环境无 dmidecode 权限） |
| **操作系统** | Ubuntu 22.04 LTS，内核 6.6.95 |
| **编译器** | g++ 11.4.0 |
| **编译选项** | `-O3 -std=c++23 -mavx2 -mavxvnni -mavx512f -mavx512bw -mavx512vl -mavx512vnni` |
| **运行时后端** | `avx512vnni`（CPUID 运行时自动检测选中） |

### CPU 支持的关键指令集

AVX2, AVX-512F, AVX-512DQ, AVX-512CD, AVX-512BW, AVX-512VL, AVX-512IFMA, AVX-512VBMI, AVX-512VBMI2, **AVX-512VNNI**, AVX-512BITALG, AVX-512VPOPCNTDQ, AVX-512BF16, GFNI, VAES, VPCLMULQDQ, SHA-NI, BMI1, BMI2

> **注意**：本机为 AMD Zen 4 架构，AVX-512 为双 256 位拼接实现（不降频）。256 位 AVX-VNNI（VEX 编码）在本机 /proc/cpuinfo 中未列出，但 CPUID leaf7 sub0 ECX[4] 读取为 1；运行时因优先选中 AVX512-VNNI（EVEX 编码），实际不触发 256 位 VNNI 路径。

## 覆盖原语

| 原语 | 类型 | 正确性 | 性能 |
|---|---|---|---|
| `dot16` | int16 点积（含 madd 快速路径） | ✅ 18 项 | ✅ |
| `dot8` | uint8×int8 点积（AVX512-VNNI） | ✅ 14 项 | ✅ |
| `dot4` | 4 位预解包点积（复用 dot8） | ✅ 14 项 | ✅ |
| `sum_f32` | 浮点等步长求和 | ✅ 36 项 | ✅ |
| `sum_sq_dev_f32` | 浮点偏差平方和 | ✅ 36 项 | ✅ |
| `sum_sumprod_f32` | 单遍双累加 | ✅ 36 项 | ✅ |
| `accum_f32` | 逐元素累加 | ✅ 10 项 | — |
| `decode_i16_f32` | packed int16 → float 解码 | ✅ 10 项 | ✅ |
| `reverse_bytes8` | 单 uint64 字节反转 | ✅ 8 项 | — |
| `batch_reverse_u8` | 批量 uint64 字节反转 | ✅ 56 项 | — |

**正确性合计：238 项，全部通过。**

> sum 系为 kRounding 档位：SIMD 分块累加与标量顺序累加的浮点舍入顺序不同，属预期差异（代码注释已标注），测试使用相对容差（1e-4）而非精确相等。其余原语均为 bit-exact 精确对比。

## 编译与运行

```bash
# 在 SGN 仓库根目录执行
g++ -O3 -std=c++23 -mavx2 -mavxvnni -mavx512f -mavx512bw -mavx512vl -mavx512vnni \
    -I. sgn_benchmark/sgn_benchmark.cpp \
    simd/scalar.cpp simd/simd_dispatch.cpp \
    simd/x86/avx2.cpp simd/x86/avxvnni.cpp simd/x86/ssse3.cpp \
    simd/x86/avx512.cpp simd/x86/avx512vnni.cpp simd/arm/neon.cpp \
    -o sgn_benchmark/sgn_benchmark

# 运行（人类可读输出 + result.json）
./sgn_benchmark/sgn_benchmark

# 仅输出 JSON
./sgn_benchmark/sgn_benchmark --json-only
```

## 性能基准结果（K=65536）

| 原语 | us/call | Mops/s |
|---|---|---|
| dot16 | 2.835 | 23,120 |
| dot8 | 1.138 | 57,570 |
| dot4 | 1.141 | 57,426 |
| sum_f32 | 3.327 | 19,696 |
| decode_i16_f32 | 10.759 | 6,091 |

完整多 K 值数据见 `result.json`。

## 文件清单

- `sgn_benchmark.cpp` — 测试与基准源码
- `README.md` — 本文件
- `result.json` — 本次运行的结构化结果
