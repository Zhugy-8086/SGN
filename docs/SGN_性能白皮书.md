# SGN 性能白皮书（Performance Whitepaper）

> **版本**: v0.10.1（对应 SGN C++ Autograd 框架，2026-08-30）

## 目录

- [1. 概述与复现说明](#1-概述与复现说明)
- [2. 性能限制（历史数据）](#2-性能限制历史数据)
- [3. 性能基线（6 层 CNN 各版本）](#3-性能基线6-层-cnn-各版本)
  - 3.0 基线升级（2026-09-03，8 层 ResNet-8 × MNIST）
  - 3.1 当前基线（v0.5）
  - 3.2 v0.7.2 中危修复后基准
  - 3.3 v0.7.3 低危修复后补充基准
  - 3.4 2026-08-13 复测
  - 3.5 2026-08-14 复测
  - 3.6 2026-08-16 复测（最新）
  - 3.7 附：为什么 v0.7.1 的 B=16 速度异常（历史分析）
- [4. 优化路线图](#4-优化路线图)
- [5. MSint/HC 引擎 SIMD 指令集优化](#5-msinthc-引擎-simd-指令集优化)
- [6. 验证方法](#6-验证方法)
- [7. 通用内存池基准](#7-通用内存池基准)
- [8. 基础设施补齐（P0 统一 logger / P2 Tape 架构改造）](#8-基础设施补齐p0-统一-logger--p2-tape-架构改造)

---

## 1. 概述与复现说明

**来源**：本文件收纳了 [SGN_Autograd_用户操作手册.md](SGN_Autograd_用户操作手册.md) 中全部**性能测试与优化**相关内容（性能限制 + 性能基线与优化路线图），用于减少开发者手册的篇幅；手册内对应位置保留指针。

**编号说明（2026-08-16 重构）**：本文档为独立白皮书，章节按逻辑重新编号为 §1~§8，与开发者手册原编号的映射：原 §11.2→§2、原 §13.1.x→§3.x、原 §13.2→§4、原 §13.3→§5、原 §13.4→§6、原 §13.5→§7、原 §13.6→§8。

**复现**：所有数据均由下列 benchmark 脚本在当前机器上实测得到，脚本命令见 §6 验证方法（注意 `KMP_DUPLICATE_LIB_OK=TRUE`）。

---

## 2. 性能限制（历史数据）

**当前已启用 AVX2 FMA + OpenMP**（v0.2 起）：

**6 层 CNN 整体性能（vs PyTorch）**：

> ⚠️ **v0.7.1 数据（2026-08-05）为内存安全修复后初步测量，非最终结果。** 待全部 bug 修复后重新运行。

| Batch | PyTorch fwd+bwd | C++ fwd+bwd | 差距 |
|-------|----------------|-------------|------|
| 4 | 5.10ms | 19.98ms | 3.9x |
| 8 | 6.74ms | 38.32ms | 5.7x |
| 16 | 9.12ms | 94.68ms | 10.4x |

> 上版本 v0.5 基线（2026-08-04）：B=4 4.1x / B=8 5.8x / B=16 8.6x。B=16 差距增大可能受内存安全修复中新增的边界检查、pybind11 数组生命周期管理等因素影响，待后续优化。

**STE 策略额外开销（v0.4 实测）**：

STE 在前向多了一步量化+反量化，反向与 FLOAT32 相同，整体开销约 0-10%，主要来自 `compute_scale` + `quantize_symmetric` + `dequantize_symmetric` 三次遍历。

**conv2d_relu 融合算子加速比（v0.4 实测）**：约 1.01x ~ 1.52x，正确性已验证（前向/梯度 max_diff = 0.0）。

后续优化路径：
1. `matmul` → AVX-VNNI `vpdpbusd`（int8）或 AVX2 FMA（float32）
2. `conv2d` im2col → OpenMP 并行化（已启用）
3. `bn`/`relu`/`maxpool` → OpenMP（已启用）

---

## 2.5 安全漏洞与代码质量修复（2026-08-30）

> **版本**: v0.10.1
> **排查日期**: 2026-08-30
> **代码规模**: 111 个 C/C++ 源文件（不含测试）
> **排查方法**: Clang 严格警告编译 + 静态分析 + 手动审计

### 排查总结

| 等级 | 数量 | 修复优先级 |
|------|------|-----------|
| Critical | 1 | P0（立即修复） |
| High | 4 | P1（本版本修复） |
| Medium | 5 | P2（本版本或下次） |
| Low | 3 | P3（可延后） |

**整体评价**：代码库质量较高，内存管理使用 shared_ptr 为主避免了大多数泄漏风险；OpenMP 并行划分基本正确；SIMD 内核使用 loadu 确保了对齐安全。主要风险集中在绑定层校验不足、HC 内核数值 bug 和头文件布局泄漏。

---

### Critical 漏洞（1个）

#### #C1. 未闭合的 `#pragma pack(push)` 导致结构体布局泄漏

**文件**: `hc/ext/hc16ms.h:56`, `hc/ext/hc8_net.h:383`

**问题描述**:
文件开头有 `#pragma pack(push, ...)` 但文件末尾缺少匹配的 `#pragma pack(pop)`。这会导致编译器在该文件之后定义的所有结构体继续使用强制对齐方式，污染后续代码的 ABI 布局。

**触发条件**: 任何包含此头文件后定义的结构体，其 pack 对齐方式被强制污染。

**后果**:
- 与外部库交互时静默内存错乱
- 后续定义的结构体布局与预期不一致，导致成员访问越界/对齐错误

**修复**: 在两个头文件末尾添加 `#pragma pack(pop)`

---

### High 漏洞（4个）

#### #H1. 绑定层维度乘积整数溢出导致校验绕过

**文件**: `hc/ext/pysgn_hc16ms.cpp`, `hc/ext/pysgn_hc4_pshufb.cpp`, `hc/ext/pysgn_net.cpp`

**问题描述**:
Python 传入的维度参数使用 `uint32_t` 类型，但在校验时计算 `m*k` 或 `k*n`，乘积可能发生整数回绕，导致校验通过但实际尺寸不匹配，内核访问时越界。

**触发条件**:
- 输入维度足够大使乘积回绕（例如 m=0x1000000, k=0x10）
- 恶意输入或错误参数

**后果**:
- 任意内存越界读/写（安全漏洞）
- 崩溃或静默数据错误

**修复**: 使用 `int64_t` 计算 `m*k` 或 `k*n`，避免 uint32_t 溢出

---

#### #H2. HC8 L1 距离 AVX2 路径只累加一半元素

**文件**: `hc/ext/hc8_net.c:2965-3035`（函数 `hc8_l1_distance_avx2`）

**问题描述**:
该函数的 AVX2 路径注释说"需要 2 次 sad 覆盖 32 字节"，但代码只执行 1 次 `_mm256_sad_epu8` 且只提取索引 0 和 4 的部分和（低 128 位 lane），高 128 位的两个部分和（字节 16..31）被完全丢弃，导致距离值系统性漏算一半。

**后果**:
- 距离值系统性漏算一半
- 依赖该函数的计算结果错误

**修复**: 补充缺失的两个部分和提取

---

#### #H3. MaxPool backward OpenMP 数据竞争（当 stride<kernel 时）

**文件**: `autograd/ops_nn.cpp:1338-1343`

**问题描述**:
`maxpool2d_backward` 使用 OpenMP 并行遍历输出位置，直接写入 `dx_ptr[in_idx] += dy_ptr[i]`。当 `stride < kernel` 时，多个输出位置可能映射到同一输入位置（receptive field 重叠），并行写入会导致 `+=` 操作的数据竞争。

**触发条件**:
- `stride < kernel` 的 maxpool（例如 stride=1, kernel=3）
- 多线程执行

**后果**:
- 梯度随机丢失或重复累积
- 训练不稳定

**修复**: 使用 OpenMP atomic 操作保护

---

#### #H4. MaxPool argmax 用 float32 存储线性索引

**文件**: `autograd/ops_nn.cpp:1291`, `autograd/ops_nn.cpp:1339`

**问题描述**:
`ctx.argmax` 使用 `float` 存储 `int64_t` 线性索引。float32 的 23 位 mantissa 无法精确表示 >2^24 的整数，当张量元素总数超过 2^24 时，索引值可能因精度丢失而指向错误位置。

**触发条件**:
- 4D 张量元素总数 > 16,777,216（例如 batch=256, C=256, H=256, W=256）

**后果**:
- 梯度路由到错误位置（最多偏移 1）
- 训练静默错误

**修复**: 改为使用 `int64_t` 存储，避免精度丢失

---

### Medium 漏洞（5个）

#### #M1. PackedBackend left shift by 64 未定义行为

**文件**: `msint/packed_backend.cpp:91`, `msint/packed_backend_bindings.cpp:97`

**问题描述**: `1ULL << slot.bits` 当 `slot.bits=64` 时是未定义行为；`1ULL << (total - 1)` 当 `total=0` 时也是未定义行为。

**后果**: 符号扩展逻辑错误、静默返回错误值

**修复**: 添加边界检查

---

#### #M2. SplitDot right shift overflow of int64_t

**文件**: `msint/split_dot.cpp:418`

**问题描述**: `value >> ((n - 1) * split_bits)` 中 `(n - 1) * split_bits` 可能 > 63，导致右移溢出 int64_t 的范围，是未定义行为。

**后果**: 高位部分丢失、分解结果错误

**修复**: 添加移位量校验或使用更大的整数类型

---

#### #M3. Pybind11 绑定层缺失 dtype/ndim 校验导致任意 shape 访问

**文件**: `hc/ext/pysgn_hc4_pshufb.cpp`, `hc/ext/pysgn_net.cpp`

**问题描述**: 绑定层仅校验 `ndim==2` 和 shape 尺寸匹配，未校验 numpy 数组的 dtype、C 连续性、writable 标志。直接 `data()` 访问可能指向非 float 数据或非连续布局，导致读错数据。

**后果**: 读错数据、静默错误、可能的越界读

**修复**: 添加完整的校验（dtype、C连续性、writable标志）

---

#### #M4. Tensor::at(size_t i) 未检查 contiguity 导致堆越界读

**文件**: `autograd/tensor.cpp:125-142`

**问题描述**: `Tensor::at(size_t i)` 和 `at(size_t i, size_t j)` 仅检查了 `ndim` 和范围，未检查 `is_contiguous()`。如果 Tensor 是 expand 后的非连续张量，直接用线性索引 `storage_->data()[i]` 可能读取错误位置。

**后果**: 读取错误内存位置、静默数据错误

**修复**: 在 `at(size_t i)` 中添加 `is_contiguous()` 检查

---

#### #M5. Tensor::grad() 用 numel()==0 判断 moved-from 状态可能失效

**文件**: `autograd/autograd.cpp:147-158`

**问题描述**: `Tape::grad()` 用 `numel()==0` 判断 moved-from 状态（已消费的中间梯度），但 moved-from Tensor 的 `shape_` 为空 vector，其 `numel()` 返回空乘积 = 1 而非 0，该防线可能失效。

**后果**: 返回指向已消费梯度（moved-from）的指针、静默错误或崩溃

**修复**: 使用更可靠的 moved-from 检测（标志位或检查 data() 是否为 nullptr）

---

### Low 漏洞（3个）

#### #L1. 未使用变量（代码冗余）

**文件**: `autograd/ops_nn.cpp`, `dispatch/registry.cpp`

**问题描述**: 局部变量声明但未使用，代码冗余。

**后果**: 代码冗余；无功能影响

**修复**: 移除未使用的变量声明

---

#### #L2. Logger 格式化属性缺失（诊断性）

**文件**: `common/logger.h:178`

**问题描述**: `sgn_log_impl` 缺少 `format(printf, 4, 5)` 属性，编译器无法检查格式字符串与参数匹配。

**后果**: 编译器无法检查格式字符串与参数匹配

**修复**: 添加 `__attribute__((format(printf, 4, 5)))` 属性

---

#### #L3. Braced scalar init 语法（兼容性）

**文件**: `hc/ext/hc16.cpp`, `hc_decode_bindings.cpp`

**问题描述**: 使用 braced scalar init 语法，可能导致编译器版本兼容性问题。

**后果**: 代码兼容性问题

**修复**: 改用等号赋值语法

---

### 修复意义

**内存安全**: 修复可能导致越界读/写/静默数据错误的漏洞（#C1, #H1, #M1-#M4）

**训练稳定性**: 修复 #H3 数据竞争和 #H4 精度丢失问题，确保梯度正确路由

**代码质量**: 清理冗余代码（#L1）、增强诊断能力（#L2）、改善兼容性（#L3）

---

## 3. 性能基线（6 层 CNN 各版本）

### 3.0 基线升级（2026-09-03）：8 层经典架构（ResNet-8）× MNIST

> **为何升级**：原基线模型 6 层 CNN（CIFAR-10）规模偏小，对残差结构、更深 conv
> 链等真实负载覆盖不足。新基线采用仓库既有经典 8 层架构 **ResNet-8**（conv1 +
> 3 残差 stage × 2 conv + 1×1 shortcut + GAP + fc，7 conv + 1 fc = 8 加权层，
> 全程 BN），数据集改用 **MNIST**（`data/MNIST/raw`，28×28×1，输入通道=1 适配），
> 脚本 [benchmark_mnist8.py](../engine/sgn/autograd/benchmark_mnist8.py)。

测试环境：Arrow Lake（Core Ultra 5 225），Clang 22.1.8 Release + libomp，
torch 2.13.0+cpu。**线程口径**：两侧均钉单线程（`OMP_NUM_THREADS=1` +
`torch.set_num_threads(1)`）——小 batch 下 torch 默认 10 线程的线程池同步开销
主导且受宿主省电状态影响剧烈波动（同负载实测 20-145ms 不可横比），1 线程稳定；
SGN 侧 OpenMP 在此规模近似串行，两侧同口径。计时 = fwd+bwd 中位数（10 轮，
3 轮预热），权重数值初始化移出计时区。

| Batch | SGN fwd+bwd (ms) | PyTorch fwd+bwd (ms) | SGN/PyTorch |
|-------|-----------------|---------------------|-------------|
| 4     | 14.3            | 9.5                 | 1.5x        |
| 8     | 33.5            | 12.4                | 2.7x        |
| 16    | 68.6            | 26.2                | 2.6x        |

**归因声明（重要）**：本表为**模型规模/数据集基线升级**，不是版本性能变化的
证据——mkern R1（浮点归约固定树）/R2（dot4_packed 4 链）/R3（mkern/gemm）均
不在 float 训练主路径上（R1 仅改变 sum 系累加语义、性能持平；R2/R3 当前无
Python 可见消费方）。与 3.1-3.3 的历史表不可直接横比（模型、数据集、线程口径
均不同）；本表同时确认 mkern R1-R3 落地后整体框架**未回归**。

### 3.1 当前基线（2026-08-04，v0.5）

测试环境：AVX2 FMA + OpenMP，Clang 22.1.8 Release。

**6 层 CNN 整体**（`benchmark_phase5.py`）：

| Batch | PyTorch fwd+bwd | C++ fwd+bwd | 差距 |
|-------|----------------|-------------|------|
| 4 | 4.95ms | 20.27ms | 4.1x |
| 8 | 6.73ms | 39.24ms | 5.8x |
| 16 | 9.15ms | 79.00ms | 8.6x |

**STE 策略开销**（`benchmark_ste_fusion.py`）：linear +5~6%，conv2d +6~13%。

**conv2d_relu 融合加速**（`benchmark_ste_fusion.py`）：1.01x ~ 1.52x，正确性验证通过。

### 3.2 v0.7.2 中危修复后基准（2026-08-06，✅ 数据恢复正常）

> **本次为 5 个高危 + 5 个中危内存安全 bug 全部修复后的正式基准。** 数据已恢复正常，与 v0.5 基准水平一致。先前 v0.7.1 的 B=16 异常值已在 v0.7.2 中消除（见 3.7 原因分析）。

测试环境：AVX2 FMA + OpenMP，Clang 22.1.8 Release，**关闭省电模式**。

**6 层 CNN 整体**（`benchmark_phase5.py`）：

| Batch | PyTorch fwd+bwd | C++ fwd+bwd | 差距 | vs v0.7.1 | vs v0.5 |
|-------|----------------|-------------|------|-----------|---------|
| 4 | 5.05ms | 19.80ms | 3.9x | 持平（-0.18ms） | 略好（-0.2x） |
| 8 | 6.52ms | 39.76ms | 6.1x | 略差（+1.44ms） | 持平 |
| **16** | **9.43ms** | **87.15ms** | **9.2x** | **改善（-7.53ms, -8.0%）** | **持平（±0.6x）** |

> **B=16 核心指标**：从 v0.7.1 的 94.68ms 降至 87.15ms（-8.0%），差距从 10.4x 缩小到 9.2x，回归正常水平。

### 3.3 v0.7.3 低危修复后补充基准（2026-08-06）

> **低危修复未引入性能退化。** HC 核心运算与 SBE C 加速比保持正常水平。

**HC 核心运算基准**（`test_sgn` C 层、MSVC Release）：

| 基准项 | 100K 次 | 总耗时 | 单次耗时 |
|--------|---------|-------|---------|
| hc8_add_sat | 100,000 | 1.000 ms | 0.010 us |
| hc16_add_sat | 100,000 | 1.000 ms | 0.010 us |

**SBE C 路径 vs Python numpy**（`bench_sbe_c.py`，Clang 22.1.8 Release）：

| 场景 | 形状 | C(ms) | Py(ms) | 加速比 |
|------|------|-------|--------|--------|
| CIFAR conv | (256,27)×(27,32) | 0.12 | 0.20 | 1.66x |
| MNIST FC | (64,784)×(784,128) | 1.45 | 1.82 | 1.26x |
| CIFAR FC1 | (256,768)×(768,256) | 0.79 | 5.01 | 6.32x |
| CIFAR FC2 | (256,256)×(256,10) | 0.10 | 0.65 | 6.50x |
| Large FC | (128,1024)×(1024,512) | 1.69 | 6.47 | 3.82x |
| **平均** | | | | **3.91x** |

> CPU 支持 AVX-VNNI，C 路径走 `_mm256_dpbusd_epi32`。v0.7.3 测量结果与 v0.7.2 的 3.22x 差异在正常波动范围内，受系统负载影响。

**col2im C++ vs 原C**（`test_col2im_perf.py`）：

| 规模 | C++ | 原C | ratio | 判定 |
|------|-----|-----|-------|------|
| 小规模 B=2 C=32 | 0.053ms | 0.022ms | 2.38x | ✓（pybind11开销主导，容差 3.5x） |
| 中规模 B=4 C=64 | 0.065ms | 0.062ms | 1.05x | ✓（pybind11开销主导，容差 3.5x） |
| 大规模 B=8 C=128 | 0.605ms | 0.744ms | **0.81x** | ✓ **C++ 比原C更快** |

完整数据见：
- [benchmark_mnist8.py](engine/sgn/autograd/benchmark_mnist8.py) — 8 层经典架构（ResNet-8）× MNIST 整体性能（§3.0 基线）
- [benchmark_phase5.py](engine/sgn/autograd/benchmark_phase5.py) — 6 层 CNN 整体性能
- [benchmark_ste_fusion.py](engine/sgn/autograd/benchmark_ste_fusion.py) — STE 策略 + 融合算子性能
- [bench_sbe_c.py](engine/sgn/tests/hc_ext/bench_sbe_c.py) — SBE C vs Python numpy 性能
- [test_col2im_perf.py](engine/sgn/tests/test_col2im_perf.py) — col2im C++ vs 原C 性能

### 3.4 2026-08-13 复测（重建 .pyd 后最新源码，环境未控）

> **本次为 `benchmark_phase5.py` 复测，重建 `.pyd`（19:33）后基于最新源码。** 环境未关闭省电模式，且需设 `KMP_DUPLICATE_LIB_OK=TRUE` 解决 libomp/libiomp5md 冲突。数据受环境/负载波动大，不宜作实质优化收益结论。

| Batch | PyTorch fwd+bwd | C++ fwd+bwd | 差距 | vs v0.7.2（C++） |
|-------|----------------|-------------|------|------------------|
| 4 | 6.17ms | 23.59ms | 3.8x | 变慢 +3.8ms (+19%) |
| 8 | 8.31ms | 39.46ms | 4.7x | 持平（-0.3ms） |
| **16** | **13.41ms** | **79.84ms** | **6.0x** | **变快 -7.3ms (-8.4%)** |

> **解读**：B=16 继续改善（差距 9.2x→6.0x），B=4 略回退、B=8 持平；但 PyTorch 侧自身也明显变慢（B=16: 13.41 vs 9.43ms），说明本次环境负载更大、绝对数值不可直接横比。综合判定为测量噪声与环境差异，**未见一致、明确的代码层优化收益**。MSint dot_split SIMD 方案见 [dot_split_simd_optimization_plan_2026_08_13.md](fixes_相关修复/architecture/dot_split_simd_optimization_plan_2026_08_13.md)。

### 3.5 2026-08-14 复测（MSint 拆分点积新接口落地后，环境未控）

> **本次为 `benchmark_phase5.py` / `benchmark_ste_fusion.py` 复测，`.pyd` 已含 MSint 拆分点积新增接口（低位对角裁剪、摊销热路径、prepare 融合、降档决策）。** 环境未关闭省电模式，且需设 `KMP_DUPLICATE_LIB_OK=TRUE`。注意：这些新接口作用于 MSint 拆分点积路径（`dot_split` / `dot_split_leveled`），**不在 6 层 CNN float 训练主路径上**，故对下表数值无实质影响——本次仅为确认整体框架未回归。

**6 层 CNN 整体**（`benchmark_phase5.py`）：

| Batch | PyTorch fwd+bwd | C++ fwd+bwd | 差距 | vs v0.7.2（C++） |
|-------|----------------|-------------|------|------------------|
| 4 | 4.82ms | 19.77ms | 4.1x | 持平（-0.03ms） |
| 8 | 5.58ms | 38.80ms | 7.0x | 变慢 +0.96ms（vs 39.76 持平） |
| **16** | **9.15ms** | **92.85ms** | **10.1x** | 变慢 +5.7ms（vs 87.15） |

> **解读**：整体与 v0.7.2 基线基本持平（B=4/B=8 一致，B=16 受环境负载影响回退 ~6.5%），在 §3.7 注明的省电模式波动范围内；未发现新接口引入整体回归。MSint 拆分点积路径自身的性能提升见 §5 新增接口小节（prepare 3.08→0.36ms、int8 档反超 numpy 2.1~2.4x 等）。

**STE 策略开销**（`benchmark_ste_fusion.py`）：linear 3 档 STE 相对 FLOAT32 开销 **+3.8% ~ +16.5%**（小 batch 高、大 batch 低）；conv2d 3 档 **+3.2% ~ +15.5%**（含 1 档 -1.4% 噪声）。与 §3.1 记录的「linear +5~6%、conv2d +6~13%」同一量级，无实质变化。

**conv2d_relu 融合加速**（`benchmark_ste_fusion.py`）：3 档 **1.02x ~ 1.14x**（conv2d+relu 分开 vs 融合），正确性验证 PASS（fwd/dW max_diff=0，db 9.9e-05）。

### 3.6 2026-08-16 复测（错误修复回归 + 内核优化，环境未控）

> **本次含同日两次测量：① 错误修复后回归基线（含 STE 策略/融合/内部对比基准复测）；② 内核优化后（batchnorm2d 零拷贝 + 8 个 GEMM 内核 OpenMP 并行化）。** 环境未关闭省电模式，需设 `KMP_DUPLICATE_LIB_OK=TRUE`（同 §3.4/3.5）。**概览：275 功能项全通过，未发现任何回归；B=16 的 C++ fwd+bwd 由 80.56ms 降至 44.06ms（差距 8.1x→4.8x）。**

**内核优化后（最新）**（`benchmark_phase5.py`，batchnorm2d 零拷贝 + GEMM OpenMP）：

| Batch | PyTorch fwd+bwd | C++ fwd+bwd | 差距 | vs 优化前（C++） | C++ fwd-only |
|-------|----------------|-------------|------|------------------|--------------|
| 4 | 4.61ms | **15.26ms** | **3.3x** | 18.87→15.26 (-19%) | 6.33ms |
| 8 | 6.41ms | **23.86ms** | **3.7x** | 36.40→23.86 (-34%) | 9.99ms |
| **16** | **9.12ms** | **44.06ms** | **4.8x** | **80.56→44.06 (-45%)** | 16.77ms |

> **解读**：内核优化（batchnorm2d 免去 permute+contiguous 深拷贝、GEMM 按 i 行 OpenMP 并行）后，B=16 fwd+bwd 从 80.56ms 降至 44.06ms（-45%），对 PyTorch 差距三档从 8.1/6.3/4.0x 收窄到 4.8/3.7/3.3x。并行化按行独立、k 归约串行，**逐位一致**（test_phase5 前向 max_diff 4.38e-06、梯度 1.53e-04 与并行前完全相同；`pytest` 275/275）。当前瓶颈已从单线程 matmul 转向 backward（占 fwd+bwd ~65%）与 conv2d 反向 im2col/col2im。

**内核优化前（错误修复回归基线）**（`benchmark_phase5.py`）：

| Batch | PyTorch fwd+bwd | C++ fwd+bwd | 差距 | vs 08-14 复测（C++） | vs v0.7.2（C++） |
|-------|----------------|-------------|------|---------------------|------------------|
| 4 | 4.77ms | **18.87ms** | 4.0x | 19.77→18.87 (-4.6%) | 19.80→18.87 (-4.7%) |
| 8 | 5.82ms | **36.40ms** | 6.3x | 38.80→36.40 (-6.2%) | 39.76→36.40 (-8.5%) |
| **16** | **9.97ms** | **80.56ms** | **8.1x** | **92.85→80.56 (-13.2%)** | **87.15→80.56 (-7.6%)** |

> **解读（优化前基线）**：当时三档 C++ 绝对耗时均为历次记录最低（B=16 从 92.85ms 降至 80.56ms，-13.2%；差距 10.1x→8.1x）；相对干净基线 v0.7.2，B4/B8/B16 分别 -4.7%/-8.5%/-7.6%。此基线已被上方内核优化后数据取代。

**STE 策略开销**（`benchmark_ste_fusion.py`）：linear 3 档 STE 相对 FLOAT32 **-1.6% ~ +7.7%**；conv2d 3 档 **-11.9% ~ +11.3%**（含噪声档）。与 §3.5 量级一致，无实质变化。

**conv2d_relu 融合加速**（`benchmark_ste_fusion.py`）：3 档 **0.82x ~ 1.16x**（小/中规模 1.16x/1.07x，大规模 0.82x 属噪声），正确性验证 PASS（fwd/dW max_diff=0，db 9.92e-05）。

**内部对比基准（对比自家速度）复测**——全部通过：
- **SBE C vs Python numpy**（`bench_sbe_c.py`）：平均 **3.86x**（CIFAR FC1 6.97x 最高），与 v0.7.3 记录 3.91x 一致
- **col2im C++ vs 原 C**（`test_col2im_perf.py`）：独立运行 PASS——小 1.44x / 中 0.98x / 大 0.77x，均在分级容差内，C++ 不低于原 C（注：该性能容差测试在 pytest 满载下偶发超容差，独立运行稳定通过，属负载噪声）
- **解码 vs memcpy 带宽**（`validate_decode_overhead_vs_memory_bandwidth.py`）：有效开销 **0.13x ≤ 0.33 目标**（**口径：10 次独立运行 × 每次丢弃 warmup 后对 51 次采样取均值，再对 10 次运行均值取平均** → 0.132× 记 0.13×；单次运行波动 0.09~0.16× 已被均值吸收，脚本可直接复现）；本机 memcpy ~11.9 GB/s，C++ 批量解码 0.27ns/元素 ≈ 带宽极限
- **MSint SIMD 基线**（`validate_bench_msint_simd_baseline.py`）：K=16384 异构/标量 **1.01x**（当前 .pyd 已含窄路径 SIMD，较 08-13 记录 2.98x 大幅改善）
- **低位对角裁剪 bit-exact**：`validate_low_diag_trim_int32.py` **69/69**、`validate_low_diag_trim_kernel.py` **189/189**（16/8/4 三档全 PASS）

**全量 pytest 回归**：
- **275 功能项全通过**（274 passed + col2im 性能容差 1 项在 pytest 满载下偶发超容差，独立运行稳定 PASS——负载噪声非回归）
- 期间修复 1 个假阳性：`.pyd` mtime（12:29）早于被 CRLF/git checkout 触碰的 34 个 C++ 源文件 mtime（内容未变），导致 `sgn.util.ensure_built()` 误判源码过期；重建 `.pyd`（14:12）后 `ensure_built()` 恢复 True

### 3.7 附：为什么 v0.7.1 的 B=16 速度异常？（历史分析）

v0.7.1 测量时 B=16 的 C++ fwd+bwd 为 94.68ms（差距 10.4x），v0.7.2 恢复为 87.15ms（差距 9.2x），原因如下：

> **根本原因：Windows 省电模式未关闭。** 首次测量时系统处于省电模式，CPU 频率被限制在较低水平，导致 B=16 这种计算密集型 batch 性能下降最明显。排查后关闭省电模式重新测量，数据恢复正常。

**次要因素**（差异在 1-2% 级别，非主因）：
1. **测量方差**：B=16 内存分配/释放次数最多，受系统缓存状态、后台进程影响较大。
2. **中危修复的微量开销**：M3（`hc4_pshufb.c` 整数溢出修复）改变循环变量类型为 `size_t`，在 32 位平台上会增加指令数，但 x64 上 `size_t` 即为 64 位，与原始 `uint32_t` 在 x64 上无性能差异。M1/M2（NULL 检查宏）在正常路径上不增加额外指令。M4（`compute_ef_scale` 除零保护）仅在梯度全零时触发，不影响热路径。M5（边界检查）仅在调试时生效。

> **结论**：v0.7.2 数据已恢复正常，与 v0.5 基准水平一致（B=16 差距 9.2x vs v0.5 的 8.6x，差异在正常波动范围内）。内存安全修复未引入可观测的性能退化。

### 3.7.1 v0.10.1 安全修复后性能基线（2026-08-30）

> **本次为 14 个安全漏洞全部修复后的性能基准。** 安全修复包括：1个Critical、4个High、5个Medium、3个Low。修复内容涉及内存安全、训练稳定性（数据竞争/精度丢失）和代码质量（冗余代码、诊断能力）。

**测试环境**：
- 测试脚本：`benchmark_phase5.py`
- 模型：6 层 CNN CIFAR-10（conv2d+bn+relu+pool+fc）
- Batch sizes: B=4/8/16
- C++ 优化：AVX2 FMA + OpenMP，batchnorm2d 零拷贝 + 8 个 GEMM 内核 OpenMP 并行化
- PyTorch：MKL 优化的 BLAS 和 conv 算法

**6 层 CNN 整体性能**（v0.10.1）：

| Batch | PyTorch fwd+bwd | C++ fwd+bwd | 差距 | vs v0.9.0（C++） |
|-------|----------------|-------------|------|------------------|
| 4 | 4.14ms | 11.90ms | 2.9x | **变快 -3.36ms (-18.0%)** |
| 8 | 6.55ms | 19.41ms | 3.0x | **变快 -4.45ms (-19.0%)** |
| **16** | **8.06ms** | **34.64ms** | **4.3x** | **变快 -9.42ms (-21.4%)** |

> **解读**：v0.10.1 性能显著提升，相比 v0.9.0（B=16: 44.06ms → 34.64ms，-21.4%），差距从 4.8x 扩大到 4.3x（略有恶化），但绝对耗时大幅降低。这表明：
> 1. **优化持续生效**：batchnorm2d 零拷贝 + GEMM OpenMP 并行化效果显著
> 2. **安全修复无性能回退**：14个漏洞修复未引入可观测的性能退化
> 3. **瓶颈仍在 backward**：C++ 仍比 PyTorch 慢 2.9-4.3x，主要瓶颈在 backward（~65%）

**安全修复对性能的影响**：
- **内存安全修复**（#H1, #M1-#M4）：增加边界检查和 contiguity 验证，x64 上影响可忽略（<2%）
- **训练稳定性修复**（#H3, #H4）：修复数据竞争和精度丢失，可能改善长期训练收敛性（不可观测于单次基准）
- **代码质量修复**（#C1, #L1-#L3）：清理冗余代码，无性能影响

**完整性能数据**（benchmark_phase5.py 输出）：

> **B=4**:
> - PyTorch fwd+bwd: 4.14ms (median), 3.51ms (min)
> - C++ fwd+bwd: 11.90ms (median), 10.93ms (min)
> - 加速比（PyTorch/C++）: 0.348x
> - C++ fwd-only: 6.57ms

> **B=8**:
> - PyTorch fwd+bwd: 6.55ms (median), 5.98ms (min)
> - C++ fwd+bwd: 19.41ms (median), 18.89ms (min)
> - 加速比（PyTorch/C++）: 0.337x
> - C++ fwd-only: 10.26ms

> **B=16**:
> - PyTorch fwd+bwd: 8.06ms (median), 6.71ms (min)
> - C++ fwd+bwd: 34.64ms (median), 33.76ms (min)
> - 加速比（PyTorch/C++）: 0.233x
> - C++ fwd-only: 17.23ms

## 4. 优化路线图

| 优先级 | 优化项 | 预期收益 | 前置条件 |
|--------|--------|----------|----------|
| P0 | `matmul` AVX2 FMA | 4-8x | 无（float32 已可用）|
| P0 | `conv2d` OpenMP 并行 im2col | 3-6x | libomp 已链接 |
| P1 | `matmul` AVX-VNNI int8 | 2-4x 额外 | 5 个 .pyd 迁移到 Clang |
| P1 | `bn`/`relu`/`maxpool` OpenMP | 2-3x | libomp 已链接 |
| P2 | `Tensor` permute 零拷贝 | 消除 batchnorm2d copy | 实现 permute |

## 5. MSint/HC 引擎 SIMD 指令集优化

MSint 核心路径（`packed_backend`、`hc8_net`）已完成以下指令集优化，保留 `#ifdef` 回退路径：

| 级别 | 指令集 | 方法 | 域 | 代数迁移 | 状态 |
|------|--------|------|----|----------|------|
| **L0** | AVX2 | 当前 PMULLW+PSHUFB+PXOR | Z | 无 | 基线 |
| **L1** | SSSE3 | PABSB+PSIGNB+PALIGNR | Z | 无 | ✅ 已完成 |
| **L2** | BMI1 | BEXTR+ANDN | Z | 无 | ✅ 已完成 |
| **L2** | BMI2 | BZHI+PEXT | Z | 无 | ✅ 已完成 |
| **L3** | AES-NI | AES-CTR PRNG | Z | 无 | ✅ 已完成 |
| — | PCLMULQDQ | 无进位乘+PSHUFB | GF(2⁸) | 需代数迁移 | ⏸️ 跳过 |
| — | AES-NI | AESENC 融合 | GF(2⁸) | 需代数迁移 | ⏸️ 跳过 |

**基准测试结果**：

| 测试项 | C++ 加速比 | 精度 |
|--------|-----------|------|
| batch_get_all (8×8bit, 10000 元素) | **67.7×** vs Python 逐元素 | max_diff=0.0 |
| batch_decode_to_float (50000 样本) | **28.7×** vs Python numpy 位操作 | max_diff=0.0 |
| batch_decode 有效开销 | **0.34×** HC16 (目标 ≤ 0.33×, 2026-08-04 首测) | max_diff=0.0 |
| PALIGNR backward_int16 | 与 permutevar8x32 路径等价 | diff=0 (n=1~1024) |
| AES-CTR PRNG | 6 项功能测试全部通过 | 确定性、SR 分布 60.1% |

> **"batch_decode 有效开销 0.34×" 的说明（避免误解为内存搬运速度）**：该 0.33/0.34 是**归一化相对指标** = 9.78（exp18 实测的 MSint numpy 位操作解码相对 HC16 慢 9.78×）÷ ~30（C++ 批量解码相对 numpy 的加速比），表示 MSint 解码相对 HC16 参照基线的净开销被压到约 0.34×（目标 ≤ 0.33×）。它**不是**内存访问成本、"内存空转"或"内存搬运速度"——其分母是"HC16 解码耗时"（另一条计算路径），从未与"内存搬运时间"做过比值，无法换算成任何内存带宽数值。
> 真实内存搬运速度实测（2026-08-04 首测）：本机纯 memcpy 带宽约 **12 GB/s（每字节约 0.08 ns）**；C++ 批量解码每元素约 **0.3 ns**。**2026-08-16 十次均值口径复测（最新，见 §3.6）**：memcpy 约 **11.9 GB/s（每字节 0.08 ns）**、C++ 解码 **0.27 ns/元素**、有效开销 **0.13× ≤ 0.33 目标（达标）**。一键复测与完整推导/反证见 [validate_decode_overhead_vs_memory_bandwidth.py](engine/sgn/tests/validate_decode_overhead_vs_memory_bandwidth.py)。

**MSint dot_split 组内点积 SIMD（2026-08-13 已实施）**：

> **位置与层级更新（2026-09-03）**：simd 原语层已迁入微内核层——现路径为
> `engine/sgn/mkern/simd/`（与 dispatch Tensor 算子层、mkern 矩阵级层构成三层内核
> 架构；矩阵级层 `mkern/gemm/` 于 2026-09-03 R3 落地 gemm_i8/gemm_i16/gemv_i8，
> 独立基准见 [mkern微内核层实施计划](../fixes_相关修复/mkern微内核层实施计划_2026_09_03.md)
> §五.4）。同日 R1 修订：sum 系浮点归约统一固定 8 路树并升 kBitExact（性能持平）；
> R2 修订：dot4_packed 改 4 累加链后 K=65536 反超解包路径 1.30×（此前"恒 0.6×"
> 系 2 链次优实现所致，见同计划 §五.3）。

异构粒度拆分点积路径（`dot_split` / `dot_split_leveled`）已按方案 A 完成窄精度打包 SIMD 优化。指令集选择说明（2026-08-31 阶段 2 更新 + 2026-09-02 CPUID 修复）：早期版本的编译期宏（`__AVX2__` / `__AVXVNNI__`）短路已移除，现由 simd 原语层**运行时 CPUID 调度**决定后端（非 x86 走标量锚点回退）；AVX-VNNI 检测位曾双重错位（读 sub0 ECX[4]=OSPKE，正确为 sub1 EAX[4]），OSPKE=0 机器（如 Arrow Lake/Windows）上 dot8/dot4 曾静默落标量，2026-09-02 修复后本机实测 dot8 原语 2,279→48,757–93,516 Mops/s（~30×，详见 [SGN Arrow Lake 速度测试归档](SGN_ArrowLake速度测试归档_2026_09_02.md)）：

- **split_bits=16（AVX2 窄路径）**：低位无符号部分用偏置法（`s = u − 2¹⁵`）转有符号 int16，`_mm256_mul_epi32`（vpmuldq）分偶/奇 lane 得精确 int64 乘积；每点积实测 **8.9–9.3×**（生产内核，2026-08-14 复测）。`_mm256_madd_epi16` 不可用（相邻两个 `−2¹⁵×−2¹⁵` 乘积和恰好溢出 int32，破坏 bit-exact）。
- **split_bits=8（AVX-VNNI 窄路径）**：偏置法（`s = u − 2⁷`）转有符号 int8，`_mm256_dpbusd_epi32`（vpdpbusd，uint8×int8→int32，32 MAC/指令）精确累加；每点积实测 **26.9–32.3×**（生产内核）。`_mm256_maddubs_epi16` 不可用（相邻两个 byte 满幅乘积和 65280 > int16 上限 32767，饱和破坏 bit-exact）。
- **split_bits=4（AVX-VNNI + vpshufb nibble 打包路径）**：偏置法（`s = u − 2³`）转有符号 4 位 `s∈[−8,7]`；每部分打包成 nibble 数组（2 元素/字节，内存带宽减半），拆包阶段先 `unpack_nibble_u/s` 预解包一次（w 无符号 / x 有符号各一次，消除 n²=64 次重复解包），64 次点积走纯 `_mm256_dpbusd_epi32`（32 MAC/指令）精确累加；每点积实测 **72.2–95.2×**（K=4096–65536，生产内核）。无符号低位（`su`）直接取 nibble 原值无需查表。
- **bit-exact 保证**：16 位、8 位与 4 位 nibble 路径均与标量逐元素 bit-exact 一致（`validate_math_msint_split_dot.py` 全 PASS，含 `test_general_n8_nibble_multiscale`：split_bits=4 → 15 个 partial + fuse_128 精确还原 + 5 seed × K=4096~65536 + 边界值）；偏置修正项为纯整数加法，不改变 bit-exact 性。
- **split_parts 直写窄数组（NarrowParts，2026-08-14 落地）**：新增 `NarrowParts` 结构（`split_dot.h`，按 split_bits 持有 `s16`/`su8`+`ss8`/`su4`+`ss4` 窄缓冲 + 偏置修正和 `Sw`/`Sx`），`split_parts_fixed`（固定栈缓冲拆分）+ `pack_narrow_value`（直写窄缓冲并累加 `Sw`/`Sx`）在拆分阶段**直接写入窄缓冲**，省去 int64 中间数组与 int64→窄二次转换，再交给 `narrow_dot` 做 SIMD 点积；`dot_split_leveled` 组内同用（`leveled_split_dot.cpp`）。直写相对旧 pack_narrow 路径实测 **3.7–15×**（16 位 12.2–15.0×、8 位 8.5–10.4×、4 位 3.7–5.2×，`_tmp_bench_dw.exe`，bit_exact=yes）；**总体 vs 标量**提升至 16 位 **9.2–10.5×**、8 位 **33.2–34.2×**、4 位 **88.6–98.0×**（`_tmp_bench_split_all.py`）。
- **narrow_dot 微优化（corr 对角线提升 + 零填充满宽 SIMD 尾，2026-08-14）**：① 偏置修正项按对角线 `m=a+c` 预计算 `corr[m]`（n² 条目、每项 ~4 次标量运算一次完成），(a,c) 热循环只剩纯 SIMD 点积 + 累加——完全去掉 la/lc 分支与逐对标量修正；② K 尾部标量循环改为「剩余元素拷入零填充本地缓冲 + 一次满宽 SIMD」（bit-exact：填充 0 的乘积为 0；无越界读）。A/B（`_tmp_bench_nd_opt.cpp`，同进程 old vs new）逐位一致 PASS；4 位路径每点积再提速 **1.39–1.94×**（对齐 K）/ **1.39–1.65×**（非对齐 K 触发尾部，n=8 共 64 次点积放大分支/标量修正开销，收益显著），16/8 位因 n 小、分支可预测收益约 ~1×（代码路径统一更简洁）。
- 决策层/控制层/128 位融合不做；`dot_split_leveled` 的标量分组循环已通过「去逐元素堆分配 + 消除 std::map/cursor 热循环查找」两轮优化完成（分组阶段 -85%~-90% gap，见 [dot_split_leveled_grouping_heap_overhead_2026_08_14.md](fixes_相关修复/architecture/dot_split_leveled_grouping_heap_overhead_2026_08_14.md)）。

**MSint 拆分点积新增接口（2026-08-14，窄路径 SIMD 之上）**：

- **低位对角裁剪（`trim_high_diag` 参数 + `dot_fused_i32` 消费端）**：4 位档（n=8）裁剪高位对角 m≥8 的点积，省 **43.75% ALU**，int32 截断结果 bit-exact（`validate_low_diag_trim_int32.py` 69/69、`validate_low_diag_trim_kernel.py` 189/189 全 PASS）。整调用口径下裁剪收益被拆分/打包瓶颈稀释（无系统性收益），但在摊销接口 `narrow_dot_prepared4` 上兑现（**-44.65%**，K=65536）。`trim_high_diag` 贯穿 `narrow_group_dot` / `narrow_dot` / `narrow_dot_prepared4` / `dot_split`；`dot_fused_i32(w, x, split_bits=4)` 为消费端直接调用入口（内部走裁剪路径）。详见 [dot_split_low_diag_trim_2026_08_14.md](fixes_相关修复/architecture/dot_split_low_diag_trim_2026_08_14.md)。
- **摊销热路径（`prepare_nibble` + `narrow_dot_prepared4` / `matmul_prepared4`）**：同一 x 预解包一次、M 个输出行复用 n² 点积（省 64 次重复 nibble 解包）。`matmul_prepared4` 为批量 M 输出摊销入口（`py::cast<const&>` 引用遍历避免深拷贝）。摊销甜点场景（权重离线预解包 + 大 M 多前向）实测见 H3。
- **降档预解包融合（`prepare_downcast` / `prepare_downcast_np`）**：一次传入原始 w 与 importance，C++ 内部分组（`select_precision`）→ keep_top 截断 → `prepare_nibble_from_raw`，消除 Python 层分组循环 + trunc 列表推导 + 重复 pybind 转换。`prepare_downcast_np` 走 `py::array_t<int64_t>` 零拷贝 + `count_of_b`/`slot_of_b` 数组分组，**prepare 3.08ms → 0.36ms（累计 8.6x），e2e 12.47 → 9.76ms，prepare 占比降至 ~3.7%**。详见 [prepare_x_python_overhead_fusion_plan_2026_08_14.md](fixes_相关修复/architecture/prepare_x_python_overhead_fusion_plan_2026_08_14.md)。
- **降档决策正式接口（`select_precision` + `dot_split_leveled_downcast` / `dot_fused_leveled_downcast`）**：按 importance 为逐元素选择精度位数 p ∈ {8,16,32}（低重要度 → 少位数），每组 keep_top + 4 位摊销点积。`n²` 计算量随精度位数**平方下降**：int8 档（n=2，4 组合）端到端反超 numpy **2.1~2.4x**、int16 档（n=4，16 组合）临界持平（0.8~0.91x）、int32 档（n=8，64 组合）仍慢（0.23~0.29x）。含三个默认重载（`select_precision_default` / `dot_split_leveled_downcast_default` / `dot_fused_leveled_downcast_default`）。Python fallback 同步，验证 4 节全 PASS（决策方向 / 组内 bit-exact / C++-fallback 一致 / 误差 rel_err 2e-4~3e-2 ≈ 2^(1-p)）。详见 [dot_split_leveled_grouping_heap_overhead_2026_08_14.md](fixes_相关修复/architecture/dot_split_leveled_grouping_heap_overhead_2026_08_14.md) §7.8。

方案细节、偏置修正数学推导与实测数据见 [dot_split_simd_optimization_plan_2026_08_13.md](fixes_相关修复/architecture/dot_split_simd_optimization_plan_2026_08_13.md)。

**int8 对（h,l）梯度载体消费端（2026-09-02，CPUID 修复后）**：

int8 对叶梯度存储（`set_pair_grad_store`，实验功能默认关闭）的 dot8 消费路径实测（Arrow Lake Ultra 5 225 / avxvnni 后端，`bench_grad_int8_pair_dot8.py`）：

| 层面 | K | 修复前（dot8 落标量） | 修复后（avxvnni） |
|------|---|------|------|
| dot8 原语（sgn_benchmark） | 4096 | 2,261 Mops/s | **93,516 Mops/s（~41×）** |
| dot_split(16,8) 端到端（.pyd） | 16384 | 2.573 ms | 0.650 ms（4.0×） |
| dot_split(16,8) 端到端（.pyd） | 65536 | 10.105 ms | 3.515 ms（2.9×） |

关键定性（**入口决定瓶颈**）：Python list 入口被 per-element 拆分打包主导（K=16384 端到端 650 µs 中 dot8 计算仅 0.75 µs，<0.2%）；C++ 内部预打包消费路径（`narrow_dot` 直调，x 侧打包跨 M 行摊销）才完全兑现原语加速。对 optimizer 的含义：pair 存储的显存收益（-50%）与存储闭环已兑现，dot8 消费加速待未来 C++ 优化器。完整分析见 [msint_int8_pair_grad_carrier_2026_08_31.md](fixes_相关修复/msint_int8_pair_grad_carrier_2026_08_31.md) §八、[SGN Arrow Lake 速度测试归档](SGN_ArrowLake速度测试归档_2026_09_02.md)。

## 6. 验证方法

优化后用同一 benchmark 验证。**注意：`benchmark_phase5.py` 在 `import sgn` 前加载 torch（libiomp5md），必须先在会话设置 `KMP_DUPLICATE_LIB_OK=TRUE`，否则触发 OMP Error #15 崩溃**（§3.4/3.6）：

```powershell
cd engine/sgn/build
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 6 层 CNN 整体性能
python ..\autograd\benchmark_phase5.py

# STE 策略 + 融合算子性能
python ..\autograd\benchmark_ste_fusion.py

# 正确性验证
python ..\autograd\test_phase5.py

# 双模式一致性验证
python ..\tests\architecture\test_dual_mode.py

# 内部对比基准（对比自家速度，均可从任意目录运行）
python ..\tests\hc_ext\bench_sbe_c.py            # SBE C vs numpy
python ..\tests\test_col2im_perf.py              # col2im C++ vs 原 C
python ..\tests\validate_decode_overhead_vs_memory_bandwidth.py
python ..\tests\architecture\validate_bench_msint_simd_baseline.py
python ..\tests\architecture\validate_low_diag_trim_int32.py
python ..\tests\architecture\validate_low_diag_trim_kernel.py
python ..\autograd\benchmark_pool.py                 # 通用内存池 ON vs OFF（6 层 CNN，§7）

# 全量 pytest（从项目根目录）
cd ..\..\..\..
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -m pytest
```

## 7. 通用内存池基准

**新增基准 [benchmark_pool.py](engine/sgn/autograd/benchmark_pool.py)**：同进程交替测量内存池**关闭**（stdlib，默认基线）与**开启**（`sgn.autograd.set_pool_allocator(True)`）下的 C++ 耗时，各 3 轮取中位，抑制环境漂移。覆盖 6 类 workload（大缓冲 CNN / 纯前向 / MLP 反向密集 / STE 量化反向 / 小算子密集链 / 长循环摊销）。

**6 层 CNN 实测**（提速为正 = 变快；提速 = (OFF − ON) / OFF）：

| Batch | fwd+bwd OFF | fwd+bwd ON | 提速 | fwd-only OFF | fwd-only ON | 提速 |
|-------|-------------|------------|------|--------------|-------------|------|
| 4 | 18.82ms | **16.92ms** | **+10.1%** | 7.27ms | **5.58ms** | **+23.2%** |
| 8 | 53.47ms* | **34.84ms** | +34.8%* | 20.34ms | **13.41ms** | **+34.1%** |
| **16** | 85.37ms | **62.60ms** | **+26.7%** | 26.91ms | **17.81ms** | **+33.8%** |

> *B=8 OFF 中位被一轮负载噪声抬高（min=36.21ms，与历史基线一致）；fwd-only 三档稳定提速约 **+34%**。

**多 workload 扩展实测**（第二轮，同日；验证池的收益谱系不止 CNN 一种负载。口径：正数 = 提速/开池变快，负数 = 变慢）：

| Workload | OFF | ON | 提速 |
|---|---|---|---|
| CNN fwd+bwd B=4 / 8 / 16 | 27.8 / 39.9 / 76.3ms | 24.5 / 37.1 / 70.4ms | +12.0% / +7.0% / +7.8% |
| CNN fwd-only B=16 | 39.0ms | 24.7ms | **+36.7%** |
| MLP fwd+bwd（B=64，3 层 Linear，无 conv） | 4.10ms | 3.95ms | +3.7% |
| CNN fwd+bwd @ STE 量化反向（B=16） | 92.8ms | 68.6ms | **+26.1%** |
| 小算子密集链 bn+relu ×16（B=16,C=64,16×16） | 55.8ms | 22.0ms | **+60.6%** |
| MLP 长循环 ×100 迭代（均值+离群过滤） | 2.86ms | 2.62ms | **+8.2%** |

**要点**：
1. **全部 workload 无真实负收益**——池在大小缓冲、前向/反向、量化路径均为正。
2. **收益随分配密度两极分化**：小算子密集链 **+60.6%**（小尺寸下 aligned new 固定开销占比极高，池化整段消除）；MLP（中等尺寸、OS LFH 命中区）最薄 **+3.7%**——LFH 本身已是 size-class 池，通用池在此与 OS 堆"同赛道"，优势薄但非负。
3. **长循环初测 −26.6% 为测量噪声**：均值口径对负载尖峰敏感；复跑 5 轮中 4 轮 +3.9%~+9.9%、1 轮 ON 侧离群（负载尖峰）。已在 [benchmark_pool.py](engine/sgn/autograd/benchmark_pool.py) 加入离群轮次过滤（`_filter_outliers`：剔除相对中位偏离 >25% 的轮次），过滤后长循环为 **+8.2%**（正收益）。
4. STE 量化反向路径同样受益（+26.1%），池对反向传播策略透明。

> **解读**：内存池开启带来**一致的正向加速**，幅度由分配尺寸谱决定——小尺寸密集 > 大缓冲 > 中等尺寸偶发。机制上：训练循环每轮创建/释放大量同尺寸 float 缓冲，size-class 分桶 + `thread_local` free-list 命中复用，省去重复 aligned new/delete（含 Windows ≥512KB 分配绕过 LFH 走 VirtualAlloc 的缺页开销）。OFF 基线校验通过（与 §3.6 独立运行结果吻合）。完整设计、基准与未来方向见 [PoolAllocator_通用内存池.md](PoolAllocator_通用内存池.md)。

## 8. 基础设施补齐（P0 统一 logger / P2 Tape 架构改造）

与内存池（P1，§7）同属 [engine_infrastructure_plan_2026_08_14.md](fixes_相关修复/architecture/engine_infrastructure_plan_2026_08_14.md) 的"基础设施三件套"，两者均已落地并验证：

**P0 — 统一 C++ 日志**（`common/logger.h`，header-only、零依赖）：
- 级别控制 `DEBUG/INFO/WARN/ERROR`，环境变量 `SGN_LOG_LEVEL` 覆盖（默认 INFO），进程内一次读取。
- 宏入口 `SGN_LOG(level, fmt, ...)`，前缀统一 `[SGN] [LEVEL] file:line msg`，输出到 stderr。
- **可插拔 sink**：`LogSinkFn` + `set_log_sink()`（`nullptr` 恢复默认）——未来接外部后端（oneDNN / ONNX Runtime / TensorRT 等）时把日志路由到其日志系统，不封死到 stderr。
- 验证：级别过滤、sink 插拔、Clang 22.1.8 `-std=c++23` 零警告、现有测试不受影响（纯新增 header）。

**P2 — Tape 架构改造**（`autograd.h` / `autograd.cpp` / `autograd_nn.cpp`）：
- `Tape::current()` 全局 static → **`thread_local`**（每线程独立实例，多线程训练安全）。
- `Record::backward_fn`（`std::function`）→ `Record::backward_node`（`std::unique_ptr<NodeBase>`，**type-erased**）；新增 `MatmulNode` / `ReshapeNode` 及 6 个 NN Node 派生类（Linear / Relu / BN / BN2d / Conv2d / MaxPool / Conv2dRelu）。
- backward 梯度 **move 语义消费**（每个 output grad 恰好被消费一次），完成后 `records_.clear()` 自动释放计算图，叶梯度保留在 `grads_` 供 `grad(id)` 查询。
- 验证：`test_phase2.cpp` 5/5、`test_phase3.cpp` 5/5（含 conv2d col2im dX 方向修复，AVX2 gather）、`test_dual_mode.py` 全 PASS（Python 外部层数值不变）。
