# SGN — Structured Gradient Network

![License](https://img.shields.io/badge/License-Apache--2.0-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![CMake](https://img.shields.io/badge/CMake-%E2%89%A53.20-064F8C)

SGN（Structured Gradient Network）是一个以**整数 / 量化路径为特色**的独立神经网络框架：
tape 自动微分、标准层与损失、STE / GEF / SR 量化反向传播、MSint 多精度拆分编码、
Level 动态精度调度，CPU SIMD（AVX2 / AVX-VNNI / AVX-512）加速——**完全不依赖 PyTorch**。

它既不是一个"只能用整数"的框架，也不是 PyTorch 的复刻：默认 FLOAT32 反向
（开箱即用），同时保留完整的整数 / 量化路径，用于探索低精度、整数域的深度学习。

---

## 本仓库当前开放内容

```
mkern/
├── simd/   一维原语层：整型点积 / 解码 / 浮点归约 / 字节操作 / 精度档位机制
├── gemm/   矩阵级微内核：整数 GEMM / GEMV（VNNI 面板 tile + 寄存器分块）
└── docs/   性能白皮书、用户手册、速度测试归档（见 docs/ 目录）
```

### mkern/simd — 一维原语层

| 类别 | 原语 | 精度档 |
|---|---|---|
| 整型点积 | `dot16` / `dot8` / `dot4` / `dot4_packed`（4 位打包） | kBitExact |
| 解码 | `decode_i16_f32` / `decode_i16_f32_packed16` | kBitExact |
| 浮点归约 | `sum_f32` / `sum_sq_dev_f32` / `sum_sumprod_f32` / `accum_f32` | **kBitExact** |
| 字节操作 | `reverse_bytes8` / `batch_reverse_u8` | kBitExact |
| nibble 解包 | `unpack_nibble_u` / `unpack_nibble_s` | — |

- **所有原语 kBitExact**：浮点归约采用固定 8 路累加器 + 固定归约树，跨后端
  （scalar / AVX2 / AVX-512）逐位一致——为断点续训的可复现性而设计。
- **后端选择链**：`scalar → avx2 → avx-vnni → avx512-vnni`，per-file ISA 编译 +
  运行时 CPUID 门控，同一二进制按 CPU 能力自动选路、自动安全回退。

### mkern/gemm — 矩阵级微内核

| 原语 | 语义 | 要点 |
|---|---|---|
| `gemm_i8` | `C[M,N] int32 = A uint8 × B int8` | VNNI 面板布局（dpbusd 的 lane 恰好对应输出列，免水平归约）；4×16/32 列寄存器 tile |
| `gemm_i16` | `C[M,N] int64 = A int16 × B int16` | k-pair madd + 列 quad 交错；溢出守卫数学收窄到"a 双 -32768"，预扫描 + 无守卫热路径 |
| `gemv_i8` | `y[M] int64 = A × x` | 逐行消费 simd::dot8（矩阵层消费原语层的分层示范） |
| `pack_b_i8` | B → VNNI 面板布局 | 一次打包摊销，同 B 多次调用可复用 |

---

## 性能（实测，Arrow Lake / Core Ultra 5 225，单线程）

| 基准 | 结果 |
|---|---|
| gemm_i8 吞吐 | **34–53 MACs/cycle**（保守 roofline = 32，实测证明 dpbusd 吞吐 >1 条/周期） |
| gemm_i8 vs 现役 float matmul 内核（同口径） | **1.34×–23.7×**（出 L2 → L1，详见性能白皮书） |
| gemm_i16（缓存内） | 1.36–1.66× 逐输出 dot16 基线 |
| dot8 原语（AVX-VNNI） | 48,757–93,516 Mops/s |
| dot4_packed（4 位打包，K=65536 出 L1） | 反超预解包路径 1.30× |

> 完整数据、roofline 标注与测量方法见
> [性能白皮书](docs/SGN_性能白皮书.md) 与
> [Arrow Lake 速度测试归档](docs/SGN_ArrowLake速度测试归档_2026_09_02.md)。

## 快速开始

环境：CMake ≥ 3.20、Ninja、Clang（推荐 22.x；GCC 亦可用）。

```bash
# 矩阵级微内核（含 4819 项 bit-exact 对拍测试 + roofline 基准）
cd mkern/gemm
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
cmake --build build
./build/gemm_boundary_test      # ALL GEMM BOUNDARY TESTS PASSED
./build/gemm_benchmark          # 性能基准（roofline 标注 + 同口径对照）

# 一维原语层（boundary 测试 + 基准）
cd ../simd
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
cmake --build build
./build/simd_boundary_test      # ALL BOUNDARY TESTS PASSED
./build/sgn_benchmark
```

可选：`SGN_GEMM_BACKEND=scalar` / `SGN_KERNEL_BACKEND=scalar` 强制回退标量锚点
（测试钩子——所有 SIMD 路径都有逐位一致的标量参照）。

## 设计纪律

- **bit-exact 契约**：每个 SIMD 实现都有逐位一致的标量锚点；整型原语任意分块 /
  多累加器展开不改变结果（整数加法可交换 / 结合）；boundary 测试做三方对拍
  （dispatch vs 标量锚点 vs 朴素参照），覆盖尺寸边界、满幅值、K 上界、accum 双态。
- **per-file ISA + 运行时 CPUID**：编译期不假设指令集（无全局 AVX 宏），实现文件
  显式声明所需 ISA、符号常驻产出，运行时检测选路——无对应指令集的 CPU 安全回退。
- **浮点文件一律 `-ffp-contract=off`**：FMA 收缩产生单舍入结果，会静默破坏跨后端
  逐位一致（显式 `_mm256_fmadd` intrinsic 不受限）。
- **数据驱动**：性能结论必须扫全尺寸段并核对实现结构；每个原语有
  warmup + 多轮均值 + roofline 标注的基准。

## 文档

- [SGN 性能白皮书](docs/SGN_性能白皮书.md) — 性能基线、优化路线、测量方法
- [用户操作手册](docs/SGN_用户使用手册.md) / [Autograd 用户操作手册](docs/SGN_Autograd_用户操作手册.md)
- [Arrow Lake 速度测试归档](docs/SGN_ArrowLake速度测试归档_2026_09_02.md) /
  [EPYC 速度测试归档](docs/SGN_EPYC速度测试归档_2026_08_31.md)

## 路线图

- [ ] 核心引擎开源：tape 自动微分、标准层 / 损失 / 优化器、MSint 编码、Level 调度器、
      完整 Python 绑定（`pip install -e` 可装，344 项 pytest 回归）
- [ ] AVX-512-VNNI 路径真机验证（`vpdpbusd` 512 位 / `vpdpwssd` int16 GEMM）
- [ ] gemm_i16 大形状 B 面板预打包（出 L2 场景）

## License

Apache-2.0
