# SGN C++ 库架构文档

> **版本**: Stage 3.0 完成版 (2026-08-02)
> **位置**: `engine/sgn/`
> **编译输出**: `sgn.cp314-win_amd64.pyd` (Clang 22.1.8 + Python 3.14.5)
> **语言**: C++23 + pybind11 + AVX2/AVX-VNNI + OpenMP (libomp)
> （安全审计 2026-08-16 决策项 7：原写 C++20，工程实际用 C++23——`CMAKE_CXX_STANDARD 23`）

---

## 一、这个库是什么

SGN C++ 库是一个**混合精度量化神经网络训练框架的底层加速核心**。它解决的核心问题是：当神经网络的前向传播和反向传播使用不同位宽的整数运算时，如何高效地存储、读取、调度和分配这些不同精度的数值。

整个库被编译成一个 Python 扩展模块（`.pyd` 文件），Python 代码通过 `import sgn` 来调用它的所有功能。但它不是简单地把 Python 代码翻译成 C++，而是**将 49 个数学实验的结论固化成了 C++ 数据结构和算法**，让数学理论变成了可以直接调用的工程组件。

这个库不训练神经网络本身——训练循环、数据加载、模型定义仍然在 Python 层完成。这个库负责的是训练过程中**最频繁调用的底层操作**：位操作、整数打包/解包、精度分配、方差统计、卡尔曼滤波等。这些操作每秒被调用数万次，必须用 C++ 实现。

---

## 二、整体结构——七层架构

整个库从底到顶分为七层，每一层只依赖它下面的层，不存在循环依赖：

```
┌─────────────────────────────────────────────────────────────────┐
│  第七层：Python 编排层                                          │
│  __init__.py (monkey-patch) / cpp_backend.py (fallback)        │
├─────────────────────────────────────────────────────────────────┤
│  第六层：Level 调度层 (C++ 实现)                                │
│  engine/sgn/level/ (LevelScheduler + LevelContext + BitsAllocator)│
├─────────────────────────────────────────────────────────────────┤
│  第五层：HC 算子层                                              │
│  col2im (反向传播 scatter-add) / HC8KalmanFilter (卡尔曼滤波)  │
├─────────────────────────────────────────────────────────────────┤
│  第四层：存储层                                                  │
│  PackedBackend (多槽位打包) / HC8Coproduct (6字节余积)         │
│  MSIntView (bitsplit/concat 视角)                               │
├─────────────────────────────────────────────────────────────────┤
│  第三层：预算分配层                                              │
│  PrecisionBudget (贪心边际分配, exp34 M-凸性)                   │
├─────────────────────────────────────────────────────────────────┤
│  第二层：单位值层                                                │
│  UnitValue (int64 + ValueSpec, 量化/反量化)                     │
├─────────────────────────────────────────────────────────────────┤
│  第一层：原语层                                                  │
│  ValueSpec (bits + scale, 最底层的精度描述)                     │
└─────────────────────────────────────────────────────────────────┘
```

下面从最底层开始，逐层讲解每一层"有什么东西"和"这是什么东西"。

---

## 三、第一层：原语层——ValueSpec

**文件**: `common/value_spec.h`

### 3.1 ValueSpec 是什么

ValueSpec 是整个库最底层的"原子"——它描述了**一个数值用多少个比特来表示，以及用什么方式计算缩放因子**。它不存储实际的数值，只存储"这个数值的精度规格"。

打个比方：ValueSpec 就像是一把尺子上的刻度信息——它告诉你这把尺子有多少个刻度（bits），以及刻度的间距怎么算（ScaleFn），但它本身不是被测量的物体。

### 3.2 ValueSpec 包含什么

**两个字段**：

1. **bits（位宽）**：8、12、16、20 或 24。这是"主杠杆"——在整个库的设计哲学中，bits 是控制精度和存储开销的最核心参数。bits 越大，能表示的数值范围越大，量化误差越小，但存储和计算开销也越大。

2. **ScaleFn（缩放函数）**：一个枚举，有四个选项：
   - **MAX**（默认）：缩放因子等于数据中绝对值最大的那个数。这是所有位宽下最优的选择（exp25 实验确认）。
   - **RMS**：缩放因子等于数据的均方根。保留接口但不是主推方案。
   - **L2**：缩放因子等于数据的 L2 范数。保留接口。
   - **P95**：缩放因子等于数据绝对值的第 95 百分位。保留接口。

### 3.3 ValueSpec 能做什么

- **从旧系统转换**：旧的 Level 系统用 `max_range`（最大范围值，如 255 或 65535）来描述精度。ValueSpec 可以从 max_range 反推出 bits（`bits = log2(max_range + 1)`），也可以把 bits 转换回 max_range（`max_range = 2^bits - 1`）。这个双向桥接确保旧数据可以无缝加载到新系统。

- **与旧系统兼容**：如果旧的 max_range 不是 2^n-1（比如 300），ValueSpec 会向上取整到最近的 bits（8 bits → max_range=255），并发出警告。

---

## 四、第二层：单位值层——UnitValue

**文件**: `common/unit_value.h`

### 4.1 UnitValue 是什么

UnitValue 是"带精度规格的整数值"。它把一个实际的浮点数（比如 0.123）量化成一个整数（比如 4035），同时记住这个整数是用什么精度规格（ValueSpec）存储的。

继续用尺子的比喻：如果 ValueSpec 是尺子的刻度信息，那 UnitValue 就是"用这把尺子测量后读出的读数"。读数本身是个整数，但配上尺子的刻度信息，就能还原出原始的浮点值。

### 4.2 UnitValue 包含什么

**两个字段**：

1. **raw（原始整数）**：一个 64 位有符号整数（`int64_t`）。之所以用 64 位，是因为它需要容纳从 8 位到 24 位的所有整数——8 位时范围是 -128~127，24 位时范围是 -8388608~8388607，64 位足以容纳所有情况。

2. **spec（精度规格）**：一个 ValueSpec，记录这个整数是用多少 bits 存储的。

### 4.3 UnitValue 能做什么

- **量化（from_float）**：把一个浮点数变成整数。过程是：先除以缩放因子，四舍五入到最近的整数，然后裁剪到 bits 允许的范围内。裁剪是对称的——8 bits 时范围是 [-127, 127]（不是 [-128, 127]），这样正负对称，避免零点偏移。

- **反量化（to_float）**：把整数变回浮点数。过程很简单：`整数 × 缩放因子`。

- **批量操作**：支持 per-tensor（整个张量用一个缩放因子）和 per-layer（每层有自己的缩放因子）两种模式。这覆盖了 HC8（per-tensor）和 HC16（per-channel/per-layer）两种使用场景。

- **numpy 互操作**：通过 pybind11 绑定，可以和 numpy 数组互相转换，在 Python 和 C++ 之间高效传递数据。

### 4.4 UnitValue 在整个库中的角色

UnitValue 是连接"数学世界"（浮点数、梯度）和"存储世界"（整数、位操作）的桥梁。上面三层的所有组件——PackedBackend、HC8Coproduct、Level 调度——最终都需要通过 UnitValue 来完成"浮点↔整数"的转换。

---

## 五、第三层：预算分配层——PrecisionBudget

**文件**: `common/precision_budget.h`

### 5.1 PrecisionBudget 是什么

PrecisionBudget 解决的问题是：**当总比特预算有限时（比如 124 bits 分给 10 层），如何分配这些 bits 才能让整体量化误差最小？**

这是一个整数规划问题。数学上已经证明（exp34），当误差函数满足 M-凸性时，贪心算法可以给出精确的全局最优解——不需要暴力搜索，不需要拉格朗日松弛，只需要一个简单的贪心循环。

### 5.2 PrecisionBudget 包含什么

**LayerCost（单层成本）**：每一层的"量化代价"描述。包含：
- **c（成本）**：一个浮点数，等于 `梯度L2范数的平方 × 输入维度`。这个公式来自 exp26 实验——它发现如果只用品方差而不乘以输入维度，会导致 29 倍的误差。输入维度大的层（如全连接层）需要更多 bits 来控制误差。
- **b_min（最小 bits）**：通常 4 或 8。低于这个值，量化完全失去意义。
- **b_max（最大 bits）**：通常 20 或 24。高于这个值，收益递减不明显。

**PrecisionBudget（预算容器）**：包含：
- **total_bits（总预算）**：所有层分配的 bits 总和不能超过这个值。
- **layers（层成本列表）**：每层的 LayerCost。
- **level_f / level_b（预留字段）**：前向和反向的精度规格。这是 Stage 3.0.4 的设计——前向传播和反向传播可以使用不同的精度。当这两个字段为 None 时，行为退化为旧的单一精度模式。

### 5.3 PrecisionBudget 的核心算法——贪心边际分配

**数学原理**：每层的量化误差函数是 `f(b) = c / (2^b - 1)`——bits 越大，误差越小，但减小的速度是递减的（因为分母增长是指数的）。

**边际收益**：给某层从 b bits 增加到 b+1 bits，误差减少量是 `Δ(b) = f(b) - f(b+1) = c × 2^b / ((2^b-1)(2^(b+1)-1))`。

**M-凸性**（exp34 证明）：`Δ(b) > Δ(b+1)` 对所有 b ≥ 1 成立。也就是说，**给同一层连续加 bits，每次加的收益都比上一次少**。这保证了贪心策略是正确的——总是把下一个 bit 分给当前边际收益最大的层。

**算法步骤**：
1. 所有层初始化到 b_min（最低精度）
2. 计算每层的边际收益 Δ_i(b_min)
3. 把 1 bit 分给边际收益最大的那层
4. 更新那层的边际收益（它会减小）
5. 重复步骤 3-4，直到预算用完或所有层达到 b_max

**复杂度**：O((B - n×b_min) × log n)，其中 B 是总预算，n 是层数。对于 ResNet-18 的 10 层、124 bits 预算，这个算法在微秒级完成。

**精度验证**（exp34 复现）：与拉格朗日数值解（连续松弛后取整）对比，gap = 0.00e+00——精确等价，不是近似。这意味着贪心算法给出的不是"差不多最优"，而是**数学上严格的全局最优**。

### 5.4 负向测试——什么不能做

exp26 的 B 方案实验发现，如果用 `log₂(方差)` 作为成本信号（而不是 `方差 × 输入维度`），会得到 6.5 倍差的解。这个负向测试被固化在单元测试中，确保未来不会有人误改成本信号公式。

---

## 六、第四层：存储层

存储层有三个组件，它们各自解决不同的存储问题。

### 6.1 PackedBackend——多槽位打包存储

**文件**: `msint/packed_backend.h`、`msint/packed_backend.cpp`

#### PackedBackend 是什么

PackedBackend 解决的问题是：**把多个不同位宽的整数打包进一个 64 位整数中，然后高效地读取出来**。

这就像把多个不同大小的物品塞进一个固定大小的箱子里——你需要知道每个物品在箱子里的精确位置（偏移量），才能正确地取出来。

#### PackedBackend 包含什么

**SlotSpec（槽位规格）**：描述一个槽位的属性——用多少 bits、是否有符号、在 64 位整数中的起始位置（偏移量）。第一个槽位放在高位。

**PackedBackend（打包后端）**：
- 构造时传入一组 SlotSpec，自动计算每个槽位的偏移量和掩码。
- 约束：所有槽位的 bits 总和不超过 64（因为打包进一个 uint64_t）。
- 支持的位宽：8、12、16、20、24 bits，可混合使用。

#### PackedBackend 能做什么

- **写入（set）**：把一个值写入指定槽位。过程是先清除该槽位的旧值（用掩码），再把新值移位后写入。负数自动用补码表示。
- **读取（get）**：从指定槽位读取值。过程是把 64 位整数右移到槽位位置，用掩码提取，如果有符号则做符号扩展。
- **批量读取（get_all / batch_get_all）**：一次性读取所有槽位的值。这是性能关键路径——在训练循环中，每个 batch 都要对 50000 个样本做批量读取。
- **AVX2 向量化（get_all_simd）**：当所有槽位都是 8 bits 且无符号时，用 `_mm_shuffle_epi8` 指令做字节级打乱，一次处理 16 个字节。这比逐元素读取快几十倍。

#### PackedBackend 的性能

- **批量读取加速**：68 倍（10000 元素 × 8 槽位，Python 16.31ms → C++ batch 0.24ms）
- **完整解码流水线**：34.23 倍加速（50000 样本，packed → float32）
- **有效开销**：0.2857 倍 HC16（目标 ≤ 0.33 倍，达成）

### 6.2 HC8Coproduct——6 字节余积存储

**文件**: `hc/hc8_coproduct.h`

#### HC8Coproduct 是什么

HC8Coproduct 是这个库中最复杂的组件。它解决的问题是：**在反向传播时，除了存储梯度本身，还要存储梯度的方差（用于驱动精度调度）和卡尔曼滤波状态（用于时间换精度），如何把这三种信息压缩到最少的字节中？**

答案是一个 6 字节的结构——梯度用 2 字节，方差用 2 字节，卡尔曼状态用 2 字节。这比分别存储节省 61.4% 的空间（exp40 验证），而且三种信息的协同效应可以让量化误差降低 383 倍。

#### 6 字节内存布局

```
┌──────────┬──────────┬──────────┐
│ Byte 0-1 │ Byte 2-3 │ Byte 4-5 │
│  梯度    │   方差   │  卡尔曼  │
│ int16    │  uint16  │  int16   │
│ (有符号)  │ (无符号) │ (Q15定点) │
└──────────┴──────────┴──────────┘
```

- **梯度（Byte 0-1）**：16 位有符号整数。可以进一步 bitsplit 为高字节（int8 前向路径）和低字节（int16 反向 concat），支持多视角读取。
- **方差（Byte 2-3）**：16 位无符号整数，存储滑动窗口内的梯度方差。通过 `variance_log2()` 转换为 [0, 16] 范围的整数，作为 Level_b 调度的反馈信号（方差大 → 需要更多 bits）。
- **卡尔曼（Byte 4-5）**：16 位 Q15 定点数，存储卡尔曼滤波器的状态估计。通过累积 N 步观测，可以等效提升 Δb ≈ ½log₂(N) bits 的精度（N=256 时 Δb≈4）。

#### HC8Coproduct 的三种使用模式

1. **单元素访问**：直接读写某个位置的梯度、方差或卡尔曼状态。
2. **批量访问**：通过 HC8CoproductArray 一次性读取所有元素的梯度（返回 numpy 数组），用于反向传播的批量计算。
3. **bitsplit 视角**：把 16 位梯度拆成高字节和低字节，分别用于前向（int8 快速路径）和反向（int16 concat），实现一份数据两种用途。

#### HC8CoproductArray——批量存储

支持两种存储模式：
- **per-element 方差**：每个元素都有自己的方差（6 bytes/elem）。精细模式，适合需要逐元素调度的场景。
- **per-layer 方差**：同一层共享一个方差（4 bytes/elem + 2 bytes/layer）。节省空间，适合层级别调度的场景（exp40 原始模式）。

#### HC8KalmanFilter——整数卡尔曼滤波器

**Q15 定点格式**：所有运算在整数域完成，避免浮点运算。Q15 表示 15 位小数位，1.0 对应 32768。

**更新公式**：
- 卡尔曼增益：`K = round(P × 32768 / (P + R))`，裁剪到 [0, 32767]
- 状态更新：`x̂ = x̂ + (K × (z - x̂)) / 32768`，裁剪到 int16 范围
- 协方差更新：`P = P - (K × P) / 32768`，裁剪到 [0, 32767]

**等效 bits 提升**：`Δb = ½log₂(R/P)`。当 R/P = 256 时（N=256 步累积），Δb = 4 bits。exp40 实测 Δb = 3.914，与理论值 3.903 完美匹配。

### 6.3 MSIntView——bitsplit / concat 视角

**文件**: `msint/msint_view.h`、`msint/msint_view.cpp`

#### MSIntView 是什么

MSIntView 提供两种位操作视角，让同一个整数可以从不同的"角度"被解读：

- **bitsplit（位拆分）**：把一个多位整数拆成几个较窄的片段。比如把 16 位整数拆成两个 8 位片段——高字节和低字节。
- **concat（拼接）**：把几个较窄的整数拼成一个较宽的整数。比如把两个 8 位整数拼成一个 16 位整数。

#### MSIntView 能做什么

- **bitsplit**：输入一个 64 位整数和它的总位宽，以及目标分片位宽，返回分片列表。负数自动转换为补码表示。分片数 = 向上取整(总位宽 / 分片位宽)。
- **concat**：输入一组值和它们的位宽，返回拼接后的 64 位整数。第一个值在高位，最后一个在低位。总位数不能超过 64。
- **concat_signed**：同 concat 但返回有符号整数。

#### MSIntView 的设计约束

根据 exp37 的负面经验，bitsplit 视角**仅用于推理读取**（PackedBackend 的批量解码），不用于训练调度。这是一个明确的设计决策——bitsplit 用于训练调度会导致性能退化，因此被主动规避。

### 6.4 cpp_backend.py——C++/Python 双后端适配

**文件**: `msint/cpp_backend.py`

#### 双后端是什么

当 C++ 扩展（`sgn.pyd`）不可用时（比如编译失败、平台不兼容），系统会自动回退到纯 Python 实现。这个回退是透明的——调用方不需要知道当前用的是 C++ 还是 Python 后端。

#### 工作方式

1. 尝试 `import sgn`（C++ 扩展）
2. 如果成功，`USING_CPP = True`，所有调用走 C++ 路径
3. 如果失败，`USING_CPP = False`，打印一条 warning 日志（不是崩溃），所有调用走 Python 路径
4. Python 路径包装原有的 `engine.ms_int.backends.PackedBackend`，接口与 C++ 版本完全对齐
   （安全审计 2026-08-16 决策项 7：`engine.ms_int` 已迁 legacy/；活跃入口为
   `engine.sgn.PackedBackend`，由 C++ `_native.PackedBackend` 提供）

---

## 七、第五层：HC 算子层

### 7.1 Col2im——反向传播 scatter-add 算子

**文件**: `hc/col2im.h`、`hc/col2im.cpp`

#### Col2im 是什么

Col2im 是卷积神经网络反向传播中的关键操作。当梯度通过 im2col 展开后，需要用 col2im 把它"还原"回图像空间——这个过程需要把重叠位置的梯度累加起来（scatter-add）。

数学等价：`x_padded[b, c, i + ho×stride, j + wo×stride] += x_col[b, c, i, j, ho, wo]`

#### Col2im 的双路径设计

这是这个库中一个精妙的工程设计——根据计算规模选择不同的执行路径：

1. **串行路径**（BC < 128）：当 batch×channel 总数小于 128 时，直接执行嵌套循环，不启动 OpenMP 线程池。原因是 libomp 线程池的启动开销（20-50 微秒）在小规模场景下比计算本身还大。

2. **并行路径**（BC ≥ 128）：当规模足够大时，调用原始 C 实现（`col2im_add_c`），它使用 OpenMP `#pragma omp parallel for` 并行化 batch×channel 维度。

#### Col2im 的迁移模板

Col2im 是 HC 扩展从 C 迁移到 C++ 的"模板"——后续的 hc8_net、hc16_net、hc4_pshufb、hc16ms 都可以按照同样的模式迁移：
1. 保留 `extern "C"` 接口（向后兼容）
2. 用 C++ 类封装核心逻辑
3. 底层复用原始 C 代码（C 源文件独立编译，保留 C 语义）
4. 头文件自包含（不依赖原 C 头文件路径，避免中文路径传染）

---

## 八、第六层：Level 调度层（C++ 实现）

**文件**: `engine/sgn/level/`（level_scheduler.cpp / level_context.h / bits_allocator.h 等，经 level_bindings.cpp 注册为 `sgn.level` 子模块）

> 安全审计 2026-08-16 决策项 7：原文件 `level/upgraded_context.py` 不存在——
> Level 调度层已于 2026-08-16 批次 2.5 在 C++ 侧完成（UpgradedLevelContext +
> 方案 D 梯度方差，注册于 `engine/sgn/level/level_bindings.cpp`），本文档结构描述
> 仍适用，实现载体已从 Python 迁移到 C++。

### 8.1 UpgradedLevelContext 是什么

UpgradedLevelContext 是 Level 调度器的 3.0.4 升级版。它在原有 LevelContext 的基础上增加了三个关键字段：bits、level_f、level_b。

### 8.2 新增的三个字段

1. **bits**：直接用比特数描述精度（8/12/16/20/24），而不是旧的 max_range（255/65535/...）。bits 和 max_range 之间自动双向同步——设置 bits 时自动更新 max_range，反之亦然。

2. **level_f（前向精度规格）**：一个可选的 ValueSpec，描述前向传播使用的精度。当不为 None 时，前向传播使用 level_f 的 bits 和 scale。

3. **level_b（反向精度规格）**：一个可选的 ValueSpec，描述反向传播使用的精度。当不为 None 时，反向传播使用 level_b 的 bits 和 scale。

**向后兼容**：当 level_f 和 level_b 都为 None 时，行为与旧版完全一致——使用单一的 level 精度。这确保旧数据可以无缝加载。

### 8.3 BitsAllocator——bits 分配适配器

BitsAllocator 是 Python 层的"胶水"组件，它把 C++ 的 PrecisionBudget 和 Level 调度器连接起来：

1. **收集成本信号**：从每层的梯度统计中提取 `(grad_l2, in_dim)`，计算 `c = grad_l2² × in_dim`
2. **构建预算**：用 LayerCost 列表构建 C++ PrecisionBudget 对象
3. **调用 C++ 分配**：调用 `PrecisionBudget.allocate()` 获得贪心最优 bits 分配
4. **应用滞后机制**（exp23 设计）：限制 bits 变化幅度为 ±2，防止精度剧烈波动。过程是先做贪心分配，然后 clamp 到 [prev-2, prev+2]，最后把 clamp 导致的预算差额重新分配。

### 8.4 为什么这一层是纯 Python

Level 调度层是纯 Python 实现，原因是：
- 调度逻辑不频繁（每 N 步才执行一次，不是每步都执行）
- 灵活性比性能更重要（需要支持多种调度策略、序列化、兼容旧数据）
- C++ PrecisionBudget 已经处理了性能关键的分配算法

---

## 九、第七层：Python 编排层

### 9.1 模块加载机制

**文件**: `__init__.py`

#### 递归导入问题的解决

由于 Python 包名和 .pyd 文件名都是 `sgn`，直接 `import sgn` 会递归导入 Python 包而不是 .pyd。解决方案是一个巧妙的 monkey-patch：

1. 临时把当前包从 `sys.modules` 移除
2. 把 `build/` 目录加入 `sys.path`
3. `import sgn` 此时加载的是 .pyd 文件（因为 `build/` 在 path 中，且 .pyd 的 `PyInit_sgn` 函数要求模块名是 `sgn`）
4. 恢复 `sys.modules` 和 `sys.path`
5. 把 native 模块注册到别名 `_sgn_native`

### 9.2 Monkey-patch 增强

C++ 绑定只暴露核心方法（构造、读写、计算），Python 端补充"胶水"功能：
- `__hash__`、`__eq__`、`__repr__` 魔术方法（让 C++ 对象在 Python 中表现自然）
- `to_dict()` / `from_dict()` 序列化（支持 JSON 兼容的字典格式）
- `to_json()` / `from_json()` JSON 序列化
- ScaleFn 枚举与字符串的双向转换

### 9.3 pybind11 模块注册

**文件**: `placeholder.cpp`

采用多文件注册模式——每个子目录提供一个 `register_xxx(py::module_&)` 函数，由 placeholder.cpp 统一调用。这样每个组件的绑定代码独立维护，不会全部堆在一个文件里。

注册顺序：
1. 基础工具函数（version、add、test_avx_vnni）
2. col2im
3. ValueSpec
4. UnitValue
5. PrecisionBudget
6. PackedBackend
7. MSIntView
8. HC8Coproduct

---

## 十、构建系统

### 10.1 编译器与选项

**文件**: `CMakeLists.txt`

- **编译器**：Clang 22.1.8（Windows MSVC 目标，`x86_64-pc-windows-msvc`）
- **C++ 标准**：C++23（强制要求，关闭编译器扩展）
  （安全审计 2026-08-16 决策项 7：原写 C++20，实际 CMakeLists 用 `CMAKE_CXX_STANDARD 23`）
- **优化选项**：`-O3`（最高优化）
- **指令集**：全局**无 AVX 宏**（2026-08-31 阶段 1-4 起，见 [全局AVX编译参数移除调查](engine/sgn/内部档案)）。AVX2/AVX-VNNI/FMA 下沉到 **per-file `COMPILE_OPTIONS`**（`mkern/simd/x86/*.cpp`、`mkern/gemm/x86/*.cpp`、`hc/`、`autograd/ops(.nn).cpp`；simd 原语层 2026-09-03 迁入 mkern 微内核层，浮点归约文件另加 `-ffp-contract=off` 防 FMA 收缩破坏 bit-exact）；AVX-512 为 `ENABLE_AVX512` 条件编译（`#ifdef __AVX512F__`）；运行时由 CPUID/SEH 自动门控 → 同一二进制按 CPU 能力自动回退，不再硬性要求 AVX-VNNI（int8×uint8→int32 点积 `vpdpbusd` 仅在支持 AVX-VNNI 的 CPU 上执行）
- **OpenMP**：`-fopenmp=libomp`（使用 LLVM 的 libomp 运行时，不用 MSVC 的 VCOMP140——两者符号不兼容）
- **LTO**：全局关闭（`-fno-lto` + `CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF`）。原因：pybind11 未定义该变量时会自动注入 `-flto`，与 `ops` 层 target 属性/omp 场景在 clang/lld 22.1.8 后端冲突（编译/链接崩溃），2026-08-31 阶段 4 起显式关闭（见调查 §9）
- **Sanitizer**：Debug 模式自动开启 AddressSanitizer（`-fsanitize=address`），可选 MemorySanitizer（`-DENABLE_MSAN=ON`）

### 10.2 为什么选择 Clang 而非 MSVC

MSVC 有一个已知的 AVX-VNNI 代码生成 bug——它无法正确生成 `vpdpbusd` 指令。Clang 可以正确生成 1 条 `vpdpbusd`（VEX 编码，内存操作数折叠），GCC 生成 2 条（函数体 + 内联），两者的 opcode 一致。因此选择 Clang 作为开发编译器，GCC 作为生产验证编译器。

### 10.3 OpenMP 运行时配置

- Clang 在 Windows 上只能使用 libomp（`-fopenmp=libomp` 生成 `__kmpc_*` 符号，VCOMP140 不兼容）
- `libomp.dll` 必须与 `.pyd` 同目录（Python 3.8+ 不从 PATH 加载 DLL）
- CMake 的 `add_custom_command` 在构建后自动复制 `libomp.dll`
- 许可（2026-09-05 注记）：libomp 为 LLVM Project 的 openmp runtime，**Apache 2.0
  with LLVM Exceptions**，随本项目分发免开源、附声明即可（见 THIRD_PARTY_NOTICES.md）

### 10.4 跨编译器性能对比

由于 Clang+libomp 和 MSVC+VCOMP140 的线程池开销不同，性能对比采用**交错基准测试**——每轮交替运行两个实现，保证它们经历相同的系统状态。分级容差策略：小于 0.1ms 场景容差 3 倍（OpenMP 开销主导），小于 1ms 容差 20%（中等噪声），大于等于 1ms 容差 5%（计算主导）。

### 10.5 未来统一 Clang（技术债）

当前项目 6 个 `.pyd` 中只有 `sgn` 用 Clang 编译，其余 5 个仍用 MSVC。这是 Stage 3.0 的工程妥协——当前代码未用 C++20 新特性，MSVC 的缺陷暂未触发。但未来必然统一到 Clang，触发条件包括：需要 AVX-VNNI 优化、引入 C++20 Concepts/Ranges、HC 库完整迁移、消除分级容差妥协。

> 安全审计 2026-08-16 决策项 7：**本段已过时**——2026-08-06 起全部 `.pyd` 已合并为
> Clang 编译的单一 `sgn` 模块（`sgn.cp314-win_amd64.pyd`），MSVC/Clang 双 .pyd 共存
> 场景不复存在；上述"技术债"章节保留为历史记录，不再适用。

详细路线图见 [COMPILER_TOOLCHAIN.md §10](COMPILER_TOOLCHAIN.md#10-未来统一-clang-路线图技术债记录)。

---

## 十一、数学实验与工程决策的对应关系

这个库的每一个设计决策都有对应的数学实验支撑。以下是关键对应关系：

| 实验编号 | 实验结论 | 工程决策 |
|---------|---------|---------|
| exp10-recheck | log(var) 与量化误差 Spearman=0.986（单调非线性） | 方差统计用 variance_log2() 作为 Level_b 反馈信号 |
| exp19 | bits 是独立于 scale 的第二杠杆 | ValueSpec.bits 作为主控制字段 |
| exp23 | bits 变化需要滞后机制 | BitsAllocator 限制 ±2 bits 变化 |
| exp25 | MAX scale 在所有 bits 下最优 | ScaleFn 默认 MAX |
| exp26 | 成本信号必须含 in_dim 因子 | LayerCost.c = grad_l2² × in_dim |
| exp34 | 贪心边际分配 = 全局最优（M-凸性） | PrecisionBudget 使用贪心算法 |
| exp37 | bitsplit 用于训练调度会退化 | bitsplit 仅用于推理读取 |
| exp39/40 | HC8 余积 + 卡尔曼 + Level_b 协同 383 倍 | HC8Coproduct 6 字节布局 |
| exp10-recheck | Pearson 0.575 < 0.7，Spearman 0.986 > 0.7 | 采用 Spearman（单调关系）而非 Pearson（线性关系） |

---

## 十二、负面经验规避

这个库不仅实现了有效的方案，还**主动规避了 10 条被实验证明无效的路径**：

1. **不实现 GEF 补偿路径**（exp21/exp29 失败）——梯度误差反馈补偿在实践中不如直接 bits 调度有效
2. **不实现 log 量化器**（exp22 失败）——对数量化在整数域实现复杂且收益不明显
3. **不实现 L2 scale 作为主杠杆**（exp24 失败）——L2 scale 只作为 ScaleFn 的一个选项，不是 bits 的替代品
4. **不用 log₂(var) 作成本信号**（exp26 B 失败）——缺少 in_dim 因子会导致 6.5 倍误差
5. **不用 Boltzmann T=1 固定调度**（exp30 失败）——概率化调度在 T=1 时不稳定
6. **不用 GEF 残差作反馈信号**（exp29 失败）——用 log(var) 代替
7. **不用 Lyapunov 硬滤除**（exp31 失败）——用确定性滞后机制代替
8. **不实现 Fisher det(g)=0 检测**（exp35 失败）——Fisher 几何在整数域不可行
9. **不实现图域 bits 分配**（exp36 失败）——基于 per-layer 标量成本更简单有效
10. **bitsplit 不用于训练调度**（exp37 失败）——仅用于推理读取

---

## 十三、测试体系

**目录**: `engine/sgn/tests/`

共 11 个测试文件，覆盖所有组件：

| 测试文件 | 测试内容 | 测试数量 |
|---------|---------|---------|
| test_unit_value.py | UnitValue 量化/反量化、HC8/HC16/HC12/HC20/HC24、per-tensor/per-layer | 15 |
| test_precision_budget.py | 贪心分配、拉格朗日对比（gap=0）、M-凸性、负向测试 | 11 |
| test_serialization.py | ValueSpec/UnitValue/PrecisionBudget 序列化往返一致性 | 20 |
| test_unified_import.py | 统一导入路径 sgn.hc.*/sgn.msint.*/sgn.level.* | 6 |
| test_level_upgrade.py | bits↔max_range 同步、level_f/level_b 接口 | 18 |
| test_packed_backend_cpp.py | PackedBackend get/set/get_all、batch_get_all、AVX2 | 15 |
| test_col2im_perf.py | col2im 性能（交错 benchmark + 分级容差） | 3 场景 |
| test_hc8_coproduct.py | HC8Coproduct 6 字节布局、bitsplit、卡尔曼 | 27 |
| test_exp40_integration_cpp.py | exp40 三组件整合 C++ 复现 | 10 |
| test_task_4_3_5_integration.py | Task 4.3.5 ResNet-18 bits 分配集成 | 8 |
| test_task_5_5_multiview_overhead.py | 多视角训练开销（50000 样本） | 3 |

**Stage 7 最终验证结果**：
- 50/50 Stage 2.7 实验脚本无导入错误
- 88/100 Stage 1.x-2.6 训练脚本通过（12 失败为预先存在问题）
- exp34 复现：贪心 vs 拉格朗日 gap = 0.00e+00
- exp10-recheck 复现：Spearman = 0.986 > 0.7
- exp40 复现：协同效应 440.68 倍（基准 383 倍）

---

## 十四、总结——这个库的价值

SGN C++ 库不是一个通用的量化库——它是一个**将 49 个数学实验的结论固化成工程组件**的领域专用库。它的价值在于：

1. **理论落地**：每个数据结构和算法都有数学证明支撑（M-凸性、power law、协同效应），不是经验调参
2. **负面经验规避**：10 条被实验证明无效的路径被主动排除，避免未来重蹈覆辙
3. **性能可控**：C++ 实现 + AVX2/AVX-VNNI 向量化 + OpenMP 并行，性能关键路径全部在 C++ 层
4. **兼容渐进**：C++/Python 双后端、旧数据兼容、re-export 策略，确保迁移过程零中断
5. **可扩展**：col2im 迁移模板可复用于其他 HC 扩展，ValueSpec/UnitValue/PrecisionBudget 三层抽象可支撑未来的 Transformer 迁移
