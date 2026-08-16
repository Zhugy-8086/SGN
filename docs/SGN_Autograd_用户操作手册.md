# SGN C++ Autograd 框架 用户操作手册

> **版本**: v0.9.0（SGN Lite 废弃代码剥离完成，独立项目里程碑）
> **更新日期**: 2026-08-16
> **上版本**: v0.8.1（MSint 多精度拆分组合落地 + H3 带宽/SIMD 基线，2026-08-13）
> **适用对象**: 协作者 / 团队成员
> **代码位置**: [engine/sgn/autograd/](engine/sgn/autograd/)、[engine/sgn/level/](engine/sgn/level/)、[engine/sgn/hc/](engine/sgn/hc/)（2026-08-16 legacy 独立：`engine/hc/` 已剥离，HC 头文件迁 `engine/sgn/include/hc/`）
> **研发链条**: 本文件属于 **第一批（数学验证 + 工程实现）**，详见 [docs/ 文档优先级链](DOCS_PRIORITY_CHAIN.md)
> **数学基础**: 本框架的数学理论（HC量化/SR/GEF/EF/Level-AMP/MSint）详见 [SGN 统一数学框架](SGN_统一数学框架.md)
> **普通用户**: 如你是普通使用者（非开发者），请参阅 [SGN 用户使用手册](SGN_用户使用手册.md)

> **路径约定**: 本文档中所有路径均相对于项目根目录（`SGN/`）。例如 `cd engine/sgn` 表示进入项目根目录下的 `engine/sgn/` 子目录。
> 
> **获取本机绝对路径**: 在项目根目录（`SGN/`）执行以下命令即可获取当前机器的完整路径：
> - Linux / Mac: `pwd`
> - Windows PowerShell: `cd`
> 
> 绝对路径主要用于以下场景：设置环境变量（如 `PYTHONPATH`、`PATH`）、配置 IDE/编辑器的工作目录、编写外部脚本的 `sys.path` 等。本文档中的命令行示例均使用相对路径，直接复制即可执行。

---

## 1. 概述

SGN C++ Autograd 框架是一个轻量级的自动微分引擎，用于替代 NumPy 单线程前向/反向计算。它采用 tape-based 设计（类似 PyTorch v0.4 之后），前向时录制操作，反向时逆序遍历计算梯度。

**能做什么**：
- 前向计算：matmul / linear / conv2d / relu / bn / maxpool / reshape
- 自动反向：调用 `y.backward(grad)` 自动计算所有参数梯度
- 与 PyTorch 互转：通过 numpy 拷贝（安全，无悬垂指针风险）
- **双模式编程**：稳定模式（`sgn.nn.Module` 封装）和调试模式（自由函数），可混合使用

**当前状态（v0.9.0 - SGN Lite 废弃代码剥离完成，独立项目里程碑）**：
- ✅ **独立项目确立**：2026-08-15 完成 SGN Lite 废弃代码剥离，`engine/sgn/` 为唯一活跃项目，版本号从继承的 5.x 体系切换为独立 v0.x 体系
- ✅ 功能完整：6 层 CNN 前向+反向已验证（与 PyTorch 数值一致）
- ✅ **双模式系统**：Module 基类（参数管理、序列化、模式切换）+ Parameter/Buffer 包装
- ✅ 双模式一致性测试全 PASS：调试模式、稳定模式、混合模式前向输出和梯度完全一致
- ✅ Tensor 视图操作已实现：`view`/`transpose`/`permute`/`squeeze`/`unsqueeze`/`expand` 全部可用
- ✅ **反向传播策略框架**：支持 FLOAT32 / STE / GEF / SR 四种策略，HC16/EF_SGD/MSINT/LEVEL_AMP 为桩
- ✅ **算子融合**：`conv2d_relu` 将 conv2d+relu 合并为一个 tape 记录，减少内存遍历
- ✅ **SIMD 指令集优化**：SSSE3（PABSB/PSIGNB/PALIGNR）、AES-NI（AES-CTR PRNG）、BMI1/2（BEXTR/ANDN/BZHI/PEXT）全部完成，保留 `#ifdef` 回退路径
- ✅ **Level 调度器 C++ 迁移**：`sgn.level` 子模块（LevelContext、LevelScheduler、BitsAllocator、LevelStrategy/AdaptiveStrategy 等）全部迁移到 C++（原 Python 包已随剥离冻结至 legacy/）
- ✅ **内存安全修复**：5 个高危 + 5 个中危 + 11 个低危全部已修复或确认为非安全问题（VNNI 缓冲区溢出、NULL 解引用、AVX2 越界写、use-after-free ×2、整数溢出、除零、边界检查、输出缓冲区大小参数、Merkle 边界检查、RS 线程安全初始化、trie 深度检查、malloc 失败清零、原子初始化等），详见 [memory-safety-audit.md
- ✅ **测试验证**：全量 pytest **275 功能项全通过**（2026-08-16 legacy 独立后回归 + 2026-08-16 重建 .pyd 后复测；详见 [SGN 性能白皮书](SGN_性能白皮书.md) §3.6）；原 93 个 HC C 测试（test_hpdc / test_global_bug_check / test_sgn）已随 `engine/hc/` 剥离至 `legacy/` 归档
- ⚠️ 性能已优化：AVX2 FMA + OpenMP 已启用（含 2026-08-16 batchnorm2d 零拷贝 + 8 个 GEMM 内核 OpenMP 并行化），比 PyTorch（MKL）慢 3.3-4.8x（fwd+bwd，B=4/8/16），待 AVX-VNNI 进一步优化（性能数据详见 [SGN 性能白皮书](SGN_性能白皮书.md)）
- ✅ **内存分配**：可插拔分配器抽象 + **通用内存池已实现**（size-class 分桶 + `thread_local` 无锁 free-list，`common/allocator.h` / `common/pool_allocator.h`）。Storage 经 `sgn_allocate_floats()` 走全局分配器（默认 64B 对齐 `aligned new`），并**在分配时捕获 deallocator** 保证 alloc/dealloc 严格配对。池化默认关闭（行为与未池化一致、数值不变），训练前调用 `sgn.set_pool_allocator(True)` 启用、`clear_pool()` 清池（实测见 [SGN 性能白皮书](SGN_性能白皮书.md) §7；设计见 [engine_infrastructure_plan_2026_08_14.md](fixes_相关修复/architecture/engine_infrastructure_plan_2026_08_14.md)）
- ✅ **统一 C++ 日志**：`common/logger.h`（header-only、零依赖），级别控制（DEBUG/INFO/WARN/ERROR，`SGN_LOG_LEVEL` 环境变量覆盖）+ 可插拔 sink 接口（`set_log_sink()` 接外部后端日志系统）。生产中无散落打印点，属纯基础设施补齐（P0）
- ✅ **Tape 架构改造**：`Tape::current()` 改为 `thread_local`（每线程独立实例，多线程训练安全）；`Record::backward_fn`（`std::function`）改为 `Record::backward_node`（`std::unique_ptr<NodeBase>`，type-erased）；backward 中梯度用 **move 语义消费**（每个 output grad 恰好被消费一次），完成后 `records_.clear()` 自动释放 graph，叶梯度保留在 `grads_` 供查询（P2）
- ✅ 标准层 Module 子类（Conv2d/Linear/ReLU/MaxPool2d/BatchNorm2d/Sequential）：已实现（`nn_layers.py`）
- ✅ `record_scope` 上下文管理器：已实现（`_RecordScope` 类，注入 `sgn.autograd`）
- ✅ `auto_build` 自动编译：`.pyd` 缺失或过期时自动触发 cmake 编译
- ✅ `pip install` 支持：`pyproject.toml` 已更新，支持 `pip install -e .`
- ✅ 训练辅助工具：`count_parameters()` / `summary()` 已实现
- ✅ 全局 verbosity 控制：`set_verbosity()` / `get_verbosity()` 已实现
- ✅ 内置损失函数：`sgn.loss` 模块（MSELoss / CrossEntropyLoss / WeightedSumLoss + LossDiagnoser 诊断器）
- ✅ **MSint 多精度拆分组合落地（v0.8.1）**：`sgn.SplitDot` / `sgn.MultiScaleView` / `sgn.PrecisionSelector` / `sgn.LeveledSplitDot` 已实现——1:N 多精度解释（int32→{16,8,4} 每层可逆）+ Level 逐元素精度选择（重要性越高拆分越细）组合为异构粒度逐元素拆分点积，融合仍 bit-exact 等价原始点积；数学验证 #22–#28 全通过（详见 [numpy_math_verification_plan](fixes_相关修复/architecture/numpy_math_verification_plan_2026_08_13.md)）
- ✅ **H3 带宽基准 + SIMD 优化基线（v0.8.1）**：H3 带宽加速已证实（按需位宽 + 1:N 多输出复用两个正交来源）；`validate_bench_msint_simd_baseline.py` 固化未优化标量 baseline 与 SIMD 适用性分析——**唯一优化焦点为 dot_split 组内点积**（已实施：16 位 AVX2 `mul_epi32` + 8 位 AVX-VNNI `dpbusd` + 4 位 nibble `vpshufb`+`dpbusd`，见 [SGN 性能白皮书](SGN_性能白皮书.md) §5），决策层/控制层/128 位融合不做；硬约束：SIMD 必须编译时宏保留非 x86 标量回退

**设计文档**：完整设计决策见 [framework_dual_mode_design.md](engine/sgn/fixes_相关修复/framework_dual_mode_design.md)

---

## 2. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.14.5 | 唯一支持版本（在终端执行 `python --version` 确认）|
| Clang | 22.1.8 | MSVC ABI target（`x86_64-pc-windows-msvc`）|
| CMake | ≥ 3.20 | 构建系统 |
| pybind11 | 随 Clang | Python 绑定 |

**编译器统一要求**：未来所有 6 个 .pyd 都会迁移到 Clang 22.1.8。详见 [COMPILER_TOOLCHAIN.md §10](COMPILER_TOOLCHAIN.md)。

---

## 3. 构建与安装

### 3.1 首次构建

```powershell
cd engine/sgn

# 配置（Release 模式，禁用 ASan）
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 编译 sgn 目标（生成 sgn.cp314-win_amd64.pyd）
cmake --build build --target sgn --config Release
```

### 3.2 增量编译

修改 C++ 源码后：

```powershell
cd engine/sgn
cmake --build build --target sgn --config Release
```

### 3.3 pip install（可编辑安装，推荐）

```powershell
cd SGN/
pip install -e .
```

安装后可从任意目录 `import sgn`，无需手动设置 `sys.path`。

### 3.4 auto_build 自动编译

当 `.pyd` 缺失或过期时，`import sgn` 会自动触发 `cmake --build build` 编译：

```python
import sgn  # 若 .pyd 不存在或源码更新，自动编译
```

也可手动调用：

```python
import sgn.util
sgn.util.auto_build()  # 检查并自动编译（如需要）
sgn.util.ensure_built()  # 仅检查，不自动编译
```

### 3.5 验证安装

```powershell
cd engine/sgn/build
python -c "import sgn; print(sgn.version()); print(sgn.test_avx_vnni())"
```

预期输出：
```
Stage 3.0 placeholder
2
```

> 若 `test_avx_vnni()` 返回 `2`，说明 AVX-VNNI intrinsics 可用（`_mm256_dpbusd_epi32`）。

### 3.6 运行 C++ 测试目标

C++ 层独立测试目标（`test_phase1/2/3`）仅在 **Debug** 构建下生成，验证 Autograd 各层正确性：

```powershell
cd engine/sgn
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug --target test_phase1 test_phase2 test_phase3
```

| 目标 | 验证内容 |
|------|----------|
| `test_phase1` | Storage + Tensor + matmul forward/backward（vs PyTorch） |
| `test_phase2` | tape-based Autograd 引擎（链式法则 + 累加） |
| `test_phase3` | 神经网络算子（linear/relu/bn/conv2d/maxpool forward+backward） |

> **注意**（2026-08-16 legacy 独立）：原 `engine/hc/` 的 `test_hpdc` / `test_global_bug_check` / `test_sgn` 三个 HC 测试目标已随 `engine/hc/` 剥离至 `legacy/`（含 `legacy.7z` 归档），不再存在于活跃树。

### 3.7 常见构建问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `ImportError: DLL load failed` | 缺 `libomp.dll` | CMake 已自动复制到 build 目录，确认文件存在 |
| `undefined symbol: __kmpc_*` | 链接了 MSVC `vcomp.lib` | 必须用 Clang + libomp，不要混用 |
| ASan 相关错误 | Debug 模式启用 ASan | 用 Release 模式，或加 `-fno-sanitize=address` |

---

## 4. 五分钟快速上手

### 4.1 调试模式（自由函数 → 低封装、高透明）

```python
import sys, os
sys.path.insert(0, os.path.join("engine", "sgn", "build"))

import numpy as np
import sgn

ag = sgn.autograd

# 1. 创建 Tensor（从 numpy 拷贝）
a = ag.Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
b = ag.Tensor.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
a.requires_grad = True
b.requires_grad = True

# 2. 录制前向操作到 tape
sgn.autograd.start_recording()
c = sgn.autograd.matmul(a, b)   # c = a @ b
sgn.autograd.stop_recording()

print(c)                          # Tensor(shape=[2,2], requires_grad=True)
print(c.to_numpy())               # 取前向输出

# 3. 反向传播
c.backward(np.ones((2, 2), dtype=np.float32))

# 4. 取梯度
print("a.grad =", a.grad)         # numpy array
print("b.grad =", b.grad)
```

**关键点**：
- `from_numpy` / `to_numpy` 都是**拷贝**（安全，但有一次开销）
- 输入 Tensor 必须设 `requires_grad = True`，否则梯度链中断
- `backward(grad)` 的 `grad` 是 numpy array，形状与输出 Tensor 一致
- `.grad` 返回 numpy array 或 `None`

### 4.2 稳定模式（Module → 高封装、少样板）

```python
import sys, os
sys.path.insert(0, os.path.join("engine", "sgn", "build"))

import numpy as np
import sgn

ag = sgn.autograd
nn = sgn.nn

# 1. 定义网络（继承 Module）
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        w = nn.Parameter([4, 2])      # 可学习参数
        nn.fill_(w.tensor(), 0.01)
        self.register_parameter("weight", w)
        b = nn.Parameter([4])
        nn.fill_(b.tensor(), 0.0)
        self.register_parameter("bias", b)

    def forward(self, inputs):
        x = inputs[0]
        return ag.linear(x, self.weight.tensor(), self.bias.tensor())

model = MyNet()
model.train()

# 2. record_scope 统一管理 tape
with ag.record_scope(clear=True):
    y_pred = model(x_np)

# 3. 反向
y_pred.backward(dY_np)

# 4. 序列化
state = model.state_dict()
model.load_state_dict(state)
```

**关键点**：
- Module 不管理 tape 生命周期，tape 由 `record_scope()` 统一管理
- 参数通过 `register_parameter()` 注册，自动被 `parameters()` 收集
- `state_dict()` / `load_state_dict()` 提供完整序列化（含 shape 校验）

---

## 5. 双模式系统

### 5.1 设计理念

框架提供两种编程模式，共享同一底层引擎，**模式由你使用的 API 决定，而非全局开关**：

| 维度 | 稳定模式 | 调试模式 |
|------|----------|----------|
| 入口 | `sgn.nn.Module` 子类 | `sgn.autograd.*` 自由函数 |
| 参数管理 | `register_parameter()` 自动注册 | 手动创建 Tensor |
| 状态管理 | `register_buffer()` 内聚 | 外部维护 |
| 模式切换 | `train()` / `eval()` | 手动控制 |
| 序列化 | `state_dict()` / `load_state_dict()` | 手动 numpy 保存 |
| tape 管理 | `record_scope()` 上下文管理器 | `record_scope()` 或 `start/stop_recording()` |
| 内省 | `named_parameters()` + `param.grad` | 手动检查 `tensor.grad` |
| 兼容性 | 可暴露 Tensor 给自由函数 | 可用 Module 管理参数 |

### 5.2 稳定模式（sgn.nn 子模块）

#### 5.2.1 Module 基类

```python
class MyModule(sgn.nn.Module):
    def __init__(self):
        super().__init__()

        # 注册可学习参数
        w = sgn.nn.Parameter([32, 3, 3, 3])      # shape: [out, in, k, k]
        sgn.nn.fill_(w.tensor(), 0.01)
        self.register_parameter('conv1_w', w)

        # 注册缓冲区（非学习状态）
        buf = sgn.nn.Buffer([32])                  # 如 BN running_mean
        self.register_buffer('running_mean', buf)

        # 注册子模块（自动递归收集参数）
        self.register_module('sub', SomeModule())

    def forward(self, inputs):
        x = inputs[0]  # Module 接收 list[Tensor]
        # 通过 .tensor() 获取内部 Tensor 传给算子
        y = ag.conv2d(x, self.conv1_w.tensor(), self.conv1_b.tensor(), 1, 1)
        return y
```

**Module 提供的核心功能**：

| 方法 | 说明 |
|------|------|
| `register_parameter(name, param)` | 注册可学习参数（param=None 取消注册） |
| `register_buffer(name, buf)` | 注册非学习状态缓冲区 |
| `register_module(name, child)` | 注册子模块（自动收集参数） |
| `parameters()` | 递归收集所有可学习参数 |
| `named_parameters(prefix="")` | 带路径名的参数列表 |
| `buffers()` | 递归收集所有缓冲区 |
| `named_buffers(prefix="")` | 带路径名的缓冲区列表 |
| `children()` | 直接子模块列表 |
| `train(mode=True)` | 递归设置训练模式 |
| `eval()` | 递归设置推理模式 |
| `training` | bool 属性，查询当前模式 |
| `forward(inputs)` | 子类实现（接收 `list[Tensor]`） |
| `__call__(inputs)` | → `forward(inputs)`（不管理 tape） |
| `state_dict()` | → `dict[str, numpy.ndarray]` |
| `load_state_dict(state)` | 加载（含完整 shape 校验） |
| `zero_grad()` | 清零参数梯度 |

**`named_parameters` 路径命名规则**：
- 本层参数：直接用注册名（如 `conv1_w`）
- 子模块参数：`子模块名.参数名`（如 `features.0.weight`）
- 嵌套路径用 `.` 分隔

#### 5.2.2 Parameter / Buffer

```python
# Parameter: requires_grad = True
p = sgn.nn.Parameter([32, 3, 3, 3])        # 创建 + 自动 requires_grad=True
p.tensor()                                  # → Tensor&（传给算子前需调用）
p.shape / p.ndim / p.numel                 # 属性
p.grad                                      # 梯度（numpy 或 None）
p.to_numpy()                                # 转为 numpy 数组

# Buffer: requires_grad = False
b = sgn.nn.Buffer([32])
b.tensor()                                  # → Tensor&
```

#### 5.2.3 训练循环

```python
model = MyNet()
model.train()

for epoch in range(num_epochs):
    for x_np, y_np in dataloader:
        # record_scope 统一管理 tape 生命周期（异常安全）
        with ag.record_scope(clear=True):
            y_pred = model(x_np)

        # 反向（梯度由外部预计算传入）
        dY = compute_loss_gradient(y_pred, y_np)
        y_pred.backward(dY)

        # 更新参数（用户提供优化器）
        optimizer.step(model.parameters())
        model.zero_grad()
```

#### 5.2.4 推理

```python
model.eval()
# 推理不需要 tape
y_pred = model(x_test)
```

> ⚠️ **注意（BN 推理限制）**：`model.eval()` 已支持训练/推理模式切换，但**当前 BatchNorm 的 eval（推理）前向尚未实现**（见 11.1 功能限制）——标准层 `BatchNorm2d` 的前向仍按训练模式计算并更新 running 统计，`model.eval()` 不会自动改用 running 统计，推理时需手动处理 BN。模型不含 BatchNorm 时不受影响。

#### 5.2.5 序列化

```python
# 保存
state = model.state_dict()
np.savez("model.npz", **state)

# 加载
loaded = dict(np.load("model.npz"))
model.load_state_dict(loaded)
# load_state_dict 包含完整 shape 校验：
# - ndim 必须一致
# - 每个维度大小必须一致
# - 非 contiguous 张量会抛出异常
```

#### 5.2.6 标准层（v0.8.0 新增）

为简化模型构建，`sgn.nn` 提供了标准层 Module 子类（位于 `nn_layers.py`），自动管理参数创建和注册：

```python
import sgn
nn = sgn.nn

# 全连接层
fc = nn.Linear(784, 256)          # 自动创建 weight [256,784] + bias [256]

# 2D 卷积层
conv = nn.Conv2d(3, 32, 3, stride=1, padding=1)  # 自动创建 weight [32,3,3,3] + bias [32]

# 激活层（无参数）
relu = nn.ReLU()
pool = nn.MaxPool2d(2, 2)         # kernel=2, stride=2

# BatchNorm2d（含 running_mean/var 缓冲区）
bn = nn.BatchNorm2d(64)

# 顺序容器
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
```

**可用标准层**：

| 类 | 参数 | 说明 |
|----|------|------|
| `nn.Linear(in_features, out_features)` | weight, bias | 全连接层，Kaiming 初始化 |
| `nn.Conv2d(in_c, out_c, kernel, stride, padding)` | weight, bias | 2D 卷积，Kaiming 初始化 |
| `nn.ReLU()` | 无 | ReLU 激活 |
| `nn.MaxPool2d(kernel, stride)` | 无 | 2D 最大池化 |
| `nn.BatchNorm2d(num_features)` | gamma, beta, running_mean, running_var | BN（含缓冲区） |
| `nn.Sequential(*layers)` | 子模块 | 顺序容器，自动注册子模块 |

**与手动注册的对比**：

```python
# 旧方式（手动注册）
w = nn.Parameter([128, 784])
nn.kaiming_uniform_(w.tensor(), 784, math.sqrt(5.0))
self.register_parameter("fc1_w", w)
b = nn.Parameter([128])
nn.fill_(b.tensor(), 0.0)
self.register_parameter("fc1_b", b)

# 新方式（标准层）
self.fc1 = nn.Linear(784, 128)  # 一行搞定，自动创建+注册+初始化
```

> **注意**：标准层在 `__init__` 中创建和注册参数，因此必须在 `super().__init__()` 之后调用。

### 5.3 调试模式（自由函数）

#### 5.3.1 基本用法

```python
ag = sgn.autograd

# 手动创建权重
w = ag.Tensor.from_numpy(np.full((32, 3, 3, 3), 0.01, dtype=np.float32))
w.requires_grad = True
b = ag.Tensor.from_numpy(np.zeros(32, dtype=np.float32))
b.requires_grad = True

x = ag.Tensor.from_numpy(x_np.copy())
x.requires_grad = True  # 必须设置！

# 方式1：上下文管理器（推荐）
with ag.record_scope(clear=True):
    y = ag.conv2d(x, w, b, 1, 1)
    y = ag.relu(y)
    y = ag.maxpool2d(y, 2, 2)
    y = ag.linear(y, fc_w, fc_b)

# 方式2：手动控制（等同方式1）
# ag.clear()
# ag.start_recording()
# y = ag.conv2d(x, w, b, 1, 1)
# ...
# ag.stop_recording()

# 反向
y.backward(dY_np.copy())

# 检查梯度
print(f"w.grad norm: {np.linalg.norm(w.grad):.6f}")
```

#### 5.3.2 注意事项

- **输入 Tensor 必须设 `requires_grad = True`**，否则梯度链在输入处中断
- **reshape 使用 `t.reshape(...)` 或 `ag.reshape(t, ...)`** 均可，Tensor 方法已绑定到 autograd-aware 版本
- 手动管理 tape 时，确保 `clear()` → `start_recording()` → 前向 → `stop_recording()` → `backward()` 顺序正确

### 5.4 混合模式

两种模式完全兼容，可自由混用：

```python
# 用 Module 管理参数，自由函数做前向
class ParamContainer(sgn.nn.Module):
    def __init__(self):
        super().__init__()
        w = sgn.nn.Parameter([32, 3, 3, 3])
        sgn.nn.fill_(w.tensor(), 0.01)
        self.register_parameter('conv1_w', w)
        # ...
    def forward(self, inputs):
        return inputs[0]  # 占位

pc = ParamContainer()

# 自由函数前向 + Module 参数
with ag.record_scope(clear=True):
    y = ag.conv2d(x, pc.conv1_w.tensor(), pc.conv1_b.tensor(), 1, 1)
    y = ag.linear(y, pc.fc_w.tensor(), pc.fc_b.tensor())

y.backward(dY)

# 通过 Module 访问梯度
for name, t in pc.named_parameters():
    print(f"{name}: grad_norm={np.linalg.norm(t.grad):.6f}")
```

---

## 6. 核心 API 速查

### 6.1 `sgn.nn` 子模块

| API | 说明 |
|-----|------|
| `Module` | 基类（参数注册、序列化、模式切换） |
| `Parameter(shape)` | 可学习参数（requires_grad=True） |
| `Buffer(shape)` | 状态缓冲区（requires_grad=False） |
| `Linear(in_features, out_features)` | 全连接层（v0.8.0） |
| `Conv2d(in_c, out_c, k, stride, pad)` | 2D 卷积层（v0.8.0） |
| `ReLU()` | ReLU 激活（v0.8.0） |
| `MaxPool2d(kernel, stride)` | 2D 最大池化（v0.8.0） |
| `BatchNorm2d(num_features)` | 2D 批归一化（v0.8.0） |
| `Sequential(*layers)` | 顺序容器（v0.8.0） |
| `kaiming_uniform_(t, fan_in, a)` | Kaiming 初始化 |
| `uniform_(t, low, high)` | 均匀分布初始化 |
| `fill_(t, value)` | 常量填充 |

### 6.2 `sgn.autograd` 子模块

#### Tensor

```python
t = sgn.autograd.Tensor.from_numpy(np_array)   # 从 numpy（拷贝）
t = sgn.autograd.Tensor([2, 3])                # 按形状分配（零初始化）

t.shape          # list[int]
t.ndim           # int
t.numel          # int
t.requires_grad  # bool（可读可写）
t.to_numpy()     # Tensor → numpy（拷贝）
t.reshape([6])   # reshape（autograd-aware，支持 -1）
t.contiguous()   # 返回 contiguous 拷贝

t.backward(grad_numpy)  # 反向传播
t.backward()            # 默认 grad=ones
t.grad                  # numpy array 或 None
```

#### Tape 管理

```python
sgn.autograd.record_scope(clear=True)   # 上下文管理器（推荐）
sgn.autograd.start_recording()          # 开始录制
sgn.autograd.stop_recording()           # 停止录制
sgn.autograd.is_recording()             # 是否正在录制
sgn.autograd.clear()                    # 清空 tape
```

#### 算子（autograd-aware）

| 算子 | 说明 |
|------|------|
| `matmul(a, b)` | 矩阵乘法 |
| `linear(x, w, b)` | 全连接层 |
| `relu(x)` | ReLU 激活 |
| `conv2d(x, w, b, stride, padding)` | 2D 卷积 |
| `maxpool2d(x, kernel, stride)` | 2D 最大池化 |
| `bn_train(x, gamma, beta, rm, rv, momentum, eps, dim)` | BatchNorm 训练模式 |
| `batchnorm2d(x, gamma, beta, rm, rv, momentum, eps)` | BatchNorm2d（4D 输入） |
| `reshape(x, shape)` | 改变形状（支持 -1） |
| `conv2d_relu(x, w, b, stride, padding)` | Conv2d+ReLU 融合算子 |

**前向-only 算子**（不记录 tape，用于推理或测试）：

```python
sgn.autograd.matmul_forward(a, b)
sgn.autograd.linear_forward(x, w, b)
sgn.autograd.relu_forward(x)
sgn.autograd.conv2d_forward(x, w, b, stride, padding)
sgn.autograd.maxpool2d_forward(x, kernel, stride)
sgn.autograd.linear_forward_ste(x, w, b, bits=8, clip_sigma=4.0)
sgn.autograd.conv2d_forward_ste(x, w, b, stride, padding, bits=8, clip_sigma=4.0)
```

### 6.3 反向传播策略

```python
ag = sgn.autograd

# 查看当前策略
print(ag.strategy_name(ag.get_backward_strategy()))  # "FLOAT32"

# 切换到 STE（前向量化 + 反向 float32 直通）
ag.set_backward_strategy(ag.BackwardStrategy.STE)
ag.set_ste_quant_config(bits=8, clip_sigma=4.0)

# 恢复默认
ag.set_backward_strategy(ag.BackwardStrategy.FLOAT32)
```

**可用策略**：

| 策略 | 枚举值 | 状态 | 说明 |
|------|--------|------|------|
| FLOAT32 | `BackwardStrategy.FLOAT32` | ✅ 已实现 | 纯 float32 反向（默认） |
| STE | `BackwardStrategy.STE` | ✅ 已实现 | 前向 HC8/16 量化，反向 float32 直通 |
| HC16 | `BackwardStrategy.HC16` | ⬜ 桩 | HC16 整数反向 |
| GEF | `BackwardStrategy.GEF` | ✅ 已实现 | HC16 + GEF 梯度误差补偿 |
| SR | `BackwardStrategy.SR` | ✅ 已实现 | 随机量化 |
| EF_SGD | `BackwardStrategy.EF_SGD` | ⬜ 桩 | 误差反馈跨 step 累积 |
| MSINT | `BackwardStrategy.MSINT` | ⬜ 桩 | MSint 多视角异构精度 |
| LEVEL_AMP | `BackwardStrategy.LEVEL_AMP` | ⬜ 桩（待实现） | Level 驱动逐层异构精度 |

> ⚠️ 未实现策略调用时会抛出 `RuntimeError`。

---

## 7. sgn.level 子模块（Level 调度器）

`sgn.level` 是精度分配决策系统，已从 Python 迁移到 C++ 引擎，通过 pybind11 暴露到 Python。

### 7.1 核心数学函数

```python
import sgn

sgn.level.bits_to_max_range(8)     # 255
sgn.level.bits_to_max_range(-1)    # 255（未设置返回默认值）
sgn.level.max_range_to_bits(255)   # 8
sgn.level.max_range_to_bits(0)     # -1（未设置）
```

### 7.2 LevelContext 数据结构

```python
ctx = sgn.level.LevelContext()                              # 默认：level=0, range=255
ctx2 = sgn.level.LevelContext(5, 1023, 10, sgn.level.LevelOperation.ADD, "test")
d = ctx2.to_dict()                                           # 序列化
ctx3 = sgn.level.LevelContext.from_dict(d)                   # 反序列化
```

### 7.3 BitsAllocator（精度分配）

```python
ba = sgn.level.BitsAllocator(total_bits=40, b_min=4, b_max=20)
costs = [(0.1, 4096), (0.2, 2048)]           # (grad_l2, in_dim)
result = ba.allocate(costs)                   # 贪心边际分配
result2 = ba.allocate_with_hysteresis(costs, [10, 10])  # 带滞后机制
```

### 7.4 LevelScheduler（自适应策略管理）

```python
sched = sgn.level.LevelScheduler(adapt_interval=50)
sched.register_strategy("high", sgn.level.StandardStrategy(level=2))
sched.register_strategy("adaptive", sgn.level.AdaptiveStrategy(
    base_level=0, variance_threshold=100.0, history_window=50))
sched.bind_neuron(0, "high")
sched.bind_neuron(1, "adaptive")
sched.get_level(0)                           # 2
sched.update_stats(1, 85, True)              # (match, verified)

d = sched.to_dict()
restored = sgn.level.LevelScheduler.from_dict(d)
restored.register_strategy("high", sgn.level.StandardStrategy(level=2))
```

### 7.5 常量

```python
sgn.level.LevelConstants.DEFAULT_BITS  # 8
sgn.level.LevelConstants.LEVEL_MIN     # -4
sgn.level.LevelConstants.LEVEL_MAX     # 2
sgn.level.LevelConstants.LEVEL_DEFAULT # 0
```

### 7.6 迁移状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 核心数学函数 + LevelContext 数据结构 | ✅ 已完成 |
| Phase 2 | BitsAllocator + 滞后机制 | ✅ 已完成 |
| Phase 3 | LevelScheduler 核心（策略、自适应、序列化） | ✅ 已完成 |
| Python 回退 | 原 Python `level_scheduler` 包已随 2026-08-15/16 剥离冻结至 `legacy/`；活跃层统一使用 C++ `sgn.level`（无 Python 回退） | ✅ 已完成（C++ 化） |
| Bug Fix | LevelConstants 范围对齐、suggest_demotion 拆分、peak_level 约束 | ✅ 已修复 |

> **保留在 Python 端**（不迁移）：`CompositeEncoding`、`ScaleGate`、`FeatureReader`、`SplitTree`、`MetaLearner`、`HookRegistry` 等辅助/实验性功能。

---

## 8. 完整示例：6 层 CNN 前向+反向

### 8.1 调试模式版本

```python
import sys, os
sys.path.insert(0, os.path.join("engine", "sgn", "build"))

import numpy as np
import sgn

ag = sgn.autograd

def make_tensor(np_arr, requires_grad=False):
    t = ag.Tensor.from_numpy(np.ascontiguousarray(np_arr, dtype=np.float32))
    t.requires_grad = requires_grad
    return t

def cnn6_forward_backward(x_np, weights, bn_params, dY_np):
    """6 层 CNN 前向+反向（调试模式：自由函数 + 手动参数）"""
    ag.clear()
    B = x_np.shape[0]
    momentum, eps = 0.1, 1e-5

    # 创建权重 Tensor
    w_conv1 = make_tensor(weights['conv1_w'], requires_grad=True)
    b_conv1 = make_tensor(weights['conv1_b'], requires_grad=True)
    # ... 其他权重同理 ...
    w_fc3 = make_tensor(weights['fc3_w'], requires_grad=True)
    b_fc3 = make_tensor(weights['fc3_b'], requires_grad=True)

    # BN 参数
    bn_g, bn_b, bn_rm, bn_rv = {}, {}, {}, {}
    for i in range(1, 6):
        bn_g[i] = make_tensor(bn_params[f'bn{i}_gamma'], requires_grad=True)
        bn_b[i] = make_tensor(bn_params[f'bn{i}_beta'], requires_grad=True)
        bn_rm[i] = make_tensor(bn_params[f'bn{i}_running_mean'], requires_grad=False)
        bn_rv[i] = make_tensor(bn_params[f'bn{i}_running_var'], requires_grad=False)

    x = make_tensor(x_np)

    # 前向（录制到 tape）
    ag.start_recording()
    # Layer 1: Conv → BN2d → ReLU → MaxPool  (B,3,32,32) → (B,32,16,16)
    y = ag.conv2d(x, w_conv1, b_conv1, stride=1, padding=1)
    y = ag.batchnorm2d(y, bn_g[1], bn_b[1], bn_rm[1], bn_rv[1], momentum, eps)
    y = ag.relu(y)
    y = ag.maxpool2d(y, kernel=2, stride=2)
    # Layer 2: (B,32,16,16) → (B,64,8,8)
    y = ag.conv2d(y, w_conv2, b_conv2, stride=1, padding=1)
    y = ag.batchnorm2d(y, bn_g[2], bn_b[2], bn_rm[2], bn_rv[2], momentum, eps)
    y = ag.relu(y)
    y = ag.maxpool2d(y, kernel=2, stride=2)
    # Layer 3: (B,64,8,8) → (B,128,4,4)
    y = ag.conv2d(y, w_conv3, b_conv3, stride=1, padding=1)
    y = ag.batchnorm2d(y, bn_g[3], bn_b[3], bn_rm[3], bn_rv[3], momentum, eps)
    y = ag.relu(y)
    y = ag.maxpool2d(y, kernel=2, stride=2)
    # Flatten: (B,128,4,4) → (B,2048)
    y = ag.reshape(y, [B, -1])
    # Layer 4-6: Linear → BN1d → ReLU → ...
    y = ag.linear(y, w_fc1, b_fc1)
    y = ag.bn_train(y, bn_g[4], bn_b[4], bn_rm[4], bn_rv[4], momentum, eps, dim=0)
    y = ag.relu(y)
    y = ag.linear(y, w_fc2, b_fc2)
    y = ag.bn_train(y, bn_g[5], bn_b[5], bn_rm[5], bn_rv[5], momentum, eps, dim=0)
    y = ag.relu(y)
    y = ag.linear(y, w_fc3, b_fc3)
    ag.stop_recording()

    output_np = y.to_numpy()
    y.backward(dY_np)

    grads = {
        'conv1_w': w_conv1.grad, 'conv1_b': b_conv1.grad,
        # ... 其他梯度 ...
        'fc3_w': w_fc3.grad, 'fc3_b': b_fc3.grad,
    }
    for i in range(1, 6):
        grads[f'bn{i}_gamma'] = bn_g[i].grad
        grads[f'bn{i}_beta'] = bn_b[i].grad

    return output_np, grads
```

**完整可运行版本**：见 [test_phase5.py](engine/sgn/autograd/test_phase5.py)（含与 PyTorch 的数值对比验证）。

### 8.2 稳定模式版本（标准层 + Module 封装）

```python
class CNN6(sgn.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 使用标准层（v0.8.0 新增）— 自动创建和注册参数
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2048, 256)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, inputs):
        x = inputs[0]
        B = x.shape[0]
        momentum, eps = 0.1, 1e-5
        ag = sgn.autograd

        x = self.conv1.forward([x])
        x = ag.batchnorm2d(x, self.bn1.gamma.tensor(), self.bn1.beta.tensor(),
                           self.bn1.running_mean.tensor(), self.bn1.running_var.tensor(), momentum, eps)
        x = self.relu.forward([x])
        x = self.pool.forward([x])
        # ... 其他层类似 ...
        return x

# 训练
model = CNN6()
model.train()
with ag.record_scope(clear=True):
    y_pred = model(x_np)
y_pred.backward(dY_np)
```

### 8.3 运行验证

```powershell
cd engine/sgn/build
python ..\autograd\test_phase5.py
```

预期结果：
```
[OK] output (B,10): max_diff=4.5e-06
[OK] conv1_w: max_diff=1.45e-04
...
PASS: C++ Autograd 6 层 CNN 前向+反向与 PyTorch 一致
```

---

## 9. 双模式一致性测试

### 9.1 测试目的

验证调试模式、稳定模式、混合模式三种编程方式下，**相同网络结构、相同输入、相同权重**的前向输出和反向梯度完全一致。

### 9.2 运行方式

```powershell
cd engine/sgn
python tests/architecture/test_dual_mode.py
```

### 9.3 预期结果

```
============================================================
双模式系统一致性测试
============================================================
输入: B=4, C=3, H=8, W=8
网络: Conv(3→32)→ReLU→MP(2)→Conv(32→64)→ReLU→MP(2)→Linear(256→10)

[1/3] 运行调试模式（自由函数 + 手动参数）...
[2/3] 运行稳定模式（Module 封装）...
[3/3] 运行混合模式（Module 参数 + 自由函数）...

--- 前向输出对比 ---
  PASS: 调试 vs 稳定
  PASS: 调试 vs 混合
  PASS: 稳定 vs 混合

--- 反向梯度对比 ---
  [conv1_w]  PASS: 调试 vs 稳定 / 调试 vs 混合 / 稳定 vs 混合
  [conv1_b]  PASS: 调试 vs 稳定 / 调试 vs 混合 / 稳定 vs 混合
  [conv2_w]  PASS: 调试 vs 稳定 / 调试 vs 混合 / 稳定 vs 混合
  [conv2_b]  PASS: 调试 vs 稳定 / 调试 vs 混合 / 稳定 vs 混合
  [fc_w]     PASS: 调试 vs 稳定 / 调试 vs 混合 / 稳定 vs 混合
  [fc_b]     PASS: 调试 vs 稳定 / 调试 vs 混合 / 稳定 vs 混合

--- 序列化不影响计算 ---
  PASS: load_state_dict 恢复参数

============================================================
结论: 全部 PASS — 双模式系统结果完全一致
============================================================
```

---

## 10. 与 PyTorch 互操作

C++ Autograd 通过 numpy 与 PyTorch 互转。**推荐流程**：

```python
import torch
import sgn

# PyTorch → C++（.detach().numpy() 拷贝）
x_torch = torch.randn(4, 3, 32, 32)
x_np = x_torch.detach().contiguous().numpy()
x_cpp = sgn.autograd.Tensor.from_numpy(x_np)

# C++ 前向+反向
# ...

# C++ → PyTorch（from_numpy 拷贝，无别名）
grad_torch = torch.from_numpy(grad_numpy)
```

**边界约定**：
- DataLoader、Optimizer、Loss 仍用 PyTorch
- C++ 只负责前向+反向的数值计算
- 每次 PyTorch ↔ C++ 转换都有一次拷贝开销（安全优先）

### 10.1 与 HC8 量化层集成

> **2026-08-16 legacy 独立**：原 HC8 量化层（`hc8_layers.py`，位于 `traditional/stage_2_9_cnn6_bn/`）已随剥离冻结至 `legacy/`（含 `legacy.7z` 归档）。HC8 活跃实现现为 `engine/sgn/hc/` 的 C 扩展（`sgn.hc8_net`）与 `tests/refs/` 的参考实现。本节集成建议为历史参考，待需要时基于活跃 C 扩展重新评估。

历史建议（保留作参考）：HC8 量化层曾用 numpy 单线程 matmul，未来可用 C++ Autograd 的 `matmul` 提速：

```python
# 原：numpy a_q @ b_q
# 新：sgn.autograd.matmul_forward(a_q_tensor, b_q_tensor).to_numpy()
```

> ⚠️ 该集成尚未实现，需要先完成 AVX-VNNI 优化（int8 路径）才能获得正向收益。

---

## 11. 已知限制

### 11.1 功能限制

| 限制 | 影响 | 临时方案 | 计划 |
|------|------|----------|------|
| 最多 4D Tensor | 无法直接处理 5D+ | reshape 到 4D | 扩展 `Storage` 支持任意维 |
| 无 `sum`/`mean` 等归约算子 | loss 需在 PyTorch 端算 | backward 时传入 dY | 按需添加 |
| 反向策略 FLOAT32/STE/GEF/SR | HC16/EF_SGD/MSINT/LEVEL_AMP 为桩 | 用 Python 层 GEF 替代 | 策略收敛后逐步实现 |
| 内存池默认关闭 | 池化需训练前显式 `sgn.set_pool_allocator(True)` 启用 | 不启用则与 stdlib 行为一致 | ✅ 已实现（`common/pool_allocator.h`，size-class + thread_local 无锁） |
| ✅ 标准层 Module 子类 | — | — | v0.8.0 已实现 |
| BN eval 模式前向未实现 | 推理时 BN 行为不正确 | 手动计算推理 BN | 按需添加 |
| 无内置优化器/损失函数 | 需用户自行实现 | 用 PyTorch 优化器 | 未来考虑 |
| ✅ 内置损失函数 | — | — | v0.8.0 已实现 `sgn.loss`（MSELoss/CrossEntropyLoss/WeightedSumLoss + LossDiagnoser） |

### 11.2 性能限制

> 本节的性能基准与限制数据已整体迁移至 [SGN 性能白皮书](SGN_性能白皮书.md)（2026-08-16）。这里仅保留结论性要点：
> - 已启用 AVX2 FMA + OpenMP（v0.2 起）；6 层 CNN 与 PyTorch 的差距、STE 额外开销、conv2d_relu 融合加速比等**全部历史实测数据见性能白皮书 §2**。
> - 后续优化路径（matmul AVX-VNNI、conv2d im2col OpenMP、bn/relu/maxpool OpenMP）见性能白皮书 §4 优化路线图。

---

## 12. 故障排查

### 12.1 `grad` 返回 `None`

**原因**：该 Tensor 在 `start_recording()` 之前创建，或 `requires_grad=False`，或 reshape 中断了梯度链。

**解决**：
```python
t.requires_grad = True  # 确认 requires_grad
# 使用 t.reshape() 或 ag.reshape(t, shape)（都是 autograd-aware）
sgn.autograd.start_recording()
y = ag.matmul(t, other)
sgn.autograd.stop_recording()
y.backward(grad)
print(t.grad)  # 现在不应为 None
```

### 12.2 `backward` 报 shape 不匹配

**原因**：传入的 `grad` 形状与输出 Tensor 不一致。

**解决**：`grad` 必须与 `y.shape` 完全一致：
```python
y.backward(np.ones(y.shape, dtype=np.float32))
```

### 12.3 梯度数值异常

**排查步骤**：
1. 用 [test_phase5.py](engine/sgn/autograd/test_phase5.py) 的 `compare` 函数对比 PyTorch
2. 检查 BN 的 `running_mean`/`running_var` 是否正确初始化
3. 确认 `momentum=0.1`、`eps=1e-5` 与 PyTorch 一致

### 12.4 内存持续增长

**原因**：tape 未清空。

**解决**：每次迭代前调用 `sgn.autograd.clear()`，或使用 `record_scope(clear=True)`：
```python
for x, y in train_loader:
    with ag.record_scope(clear=True):  # 自动 clear + start + stop
        y_pred = model(x)
```

### 12.5 `ImportError: DLL load failed`

见 [3.5 常见构建问题](#35-常见构建问题)。

### 12.6 OMP 冲突与路径解析调试日志

当遇到 OpenMP 运行时冲突（`Error #15: Initializing libiomp5md.dll, but found libomp.dll already initialized`）或 `import engine.sgn` 失败时，可通过 `SGN_DEBUG` 环境变量启用调试日志：

```powershell
# 启用调试日志
$env:SGN_DEBUG = "1"
python test_pysgn_net.py

# 正常模式（无日志输出，默认）
python test_pysgn_net.py
```

**OMP 冲突背景**（安全审计 2026-08-16 A2-6 更新）：
- 原描述的 MSVC/Clang 双 `.pyd` 共存场景自 2026-08-06 起不复存在——全部 `.pyd` 已合并为 Clang 编译的单一 `sgn` 模块
- `KMP_DUPLICATE_LIB_OK` 保留为防御性设置：进程内如加载了携带其它 OpenMP 运行时的第三方扩展（如 numpy/torch 分发的 libiomp），仍可避免 `Error #15` 崩溃

**路径解析背景**：
- `import engine.sgn` 依赖 `_PROJECT_ROOT` 正确指向项目根目录
- 若 `engine/` 在 `sys.path` 中且排在 `build/` 之前，Python 会优先加载 `engine/sgn/__init__.py`（包）而非 `sgn.cp*.pyd`（扩展），导致递归加载失败
- 修复：仅添加 `build/` 到 `sys.path`，移除 `engine/` 和 `engine/sgn/` 避免 shadow `.pyd`

**调试日志覆盖的文件**（共 11 个）：

| 文件 | 日志内容 |
|------|----------|
| `bench_sbe_c.py` | OMP 设置前后状态 + `.pyd` 路径 + 版本 |
| `benchmark_ste_fusion.py` | OMP 设置前状态 + `sys.path` |
| `test_pysgn_net.py` | OMP + 路径解析 + 导入异常详情 |
| `test_sbe_c.py` | OMP + 路径解析 + 导入异常详情 |
| `test_hc16_net.py` | OMP + 路径解析 + 加载后版本 |
| `test_hc4_pshufb.py` | OMP + 路径解析 + 加载后版本 |
| `_hc4_smoke_test.py` | 导入异常详情 + `sys.path` |
| `test_level_migration.py` | 路径解析（`_TEST_DIR`→`_BUILD_DIR`） |
| `test_grad_variance_signal.py` | 路径修复前后 + 属性探查 |
| `test_level_hc8_integration.py` | 导入异常详情 + `sys.path` |
| `test_level_upgrade.py` | 导入异常详情 + 可用属性列表 |

**日志格式**：
```
[DEBUG] <文件名>: <消息>
[DEBUG]   <键>=<值>
```

所有日志统一通过 `SGN_DEBUG in os.environ` 控制，不设置时不产生任何输出。

**快速启用命令**（可直接复制粘贴到终端）：

```powershell
# 启用调试日志
$env:SGN_DEBUG = "1"
python engine\sgn\tests\hc_ext\test_pysgn_net.py

# 关闭调试日志（恢复正常模式）
Remove-Item Env:SGN_DEBUG
```

> 设置一次 `$env:SGN_DEBUG = "1"` 后，当前终端会话中所有测试文件都会输出调试日志。用 `Remove-Item Env:SGN_DEBUG` 关闭。

---

## 13. 性能基线与优化（已迁移至性能白皮书）

> **2026-08-16：本节全部性能数据已整体迁移至独立文档 [SGN 性能白皮书](SGN_性能白皮书.md)（独立白皮书，按逻辑重编号为 §1~§8，原 §13/§11.2 映射见白皮书 §1）。** 目录速览：
> - 各版本基线与复测（v0.5 ~ 2026-08-16）：白皮书 §3.1~§3.7
> - 优化路线图：白皮书 §4
> - MSint/HC 引擎 SIMD 指令集优化：白皮书 §5
> - 验证方法（含 OMP 环境变量、benchmark 命令）：白皮书 §6
> - 通用内存池基准：白皮书 §7
> - 基础设施补齐（P0 logger / P2 Tape）：白皮书 §8

---

## 14. 已知 Bug 修复记录

### Bug 1: `register_module` 丢弃 name 参数

**问题**：`register_module` 内部未保存注册名，`named_parameters()` 使用索引生成路径名（如 `0.p` 而非 `inner.p`）。

**修复**：在 Module 中新增 `children_names_` 向量，与 `children_` 平行存储注册名。

**涉及文件**：[autograd/module.h](engine/sgn/autograd/module.h), [autograd/module.cpp](engine/sgn/autograd/module.cpp)

### Bug 2: `kaiming_uniform_` / `uniform_` 使用固定种子

**问题**：固定种子导致相同形状的参数初始化为相同值。

**修复**：改用 `std::mt19937` + `std::random_device` 生成随机种子。

**涉及文件**：[autograd/tensor.h](engine/sgn/autograd/tensor.h)

### Bug 3: `load_state_dict` 只检查 numel 不检查 shape

**问题**：形状不同但元素总数相同的张量可以互相加载（如 `[4, 4]` 和 `[2, 8]`）。

**修复**：添加完整的 shape 校验，比较 ndim 和各维度大小。

**涉及文件**：[autograd/module_bindings.cpp](engine/sgn/autograd/module_bindings.cpp)

### Bug 4: `load_state_dict` 非 contiguous 张量静默失败

**问题**：向非 contiguous 张量写入数据时，数据写入到临时拷贝，不修改原参数。

**修复**：非 contiguous 张量直接抛出异常。

**涉及文件**：[autograd/module_bindings.cpp](engine/sgn/autograd/module_bindings.cpp)

### Bug 5: `Tensor::reshape` 未传递 requires_grad

**问题**：`Tensor::reshape()` 方法不记录 tape，导致 `requires_grad` 丢失，梯度链在 reshape 处中断。

**修复**：将 Tensor 方法 `reshape` 的绑定改为调用 autograd-aware 的自由函数 `reshape(t, shape)`。

**涉及文件**：[autograd/autograd_bindings.cpp](engine/sgn/autograd/autograd_bindings.cpp)

---

## 15. 诊断与工具

为方便日常开发和调试，`sgn` 模块内置了轻量级诊断和工具函数。

### 15.1 `sgn.diagnose()` — 一键诊断

打印版本、构建时间、编译器、CPU 特性、子模块状态。

```python
>>> import sgn
>>> print(sgn.diagnose())
SGN version: Stage 3.0 placeholder
Build: 2026-08-06 10:32
Compiler: Clang 22.1.8
Python: 3.14.5
CPU: AVX2 ✓  AVX-VNNI ✓
Submodules: col2im_c ✓  hc8_net ✓  hc16 ✓  hc16ms ✓  hc4 ✓  autograd ✓  nn ✓
```

**用途**：排查构建问题、确认编译器和 CPU 特性、快速了解模块状态。

### 15.2 `sgn.test()` — 快速自检

验证 10 项核心功能是否正常，返回 `True`/`False`。

```python
>>> sgn.test()
[PASS] version
[PASS] col2im_c submodule
[PASS] hc8_net submodule
[PASS] hc16 submodule
[PASS] hc16ms submodule
[PASS] hc4 submodule
[PASS] autograd submodule
[PASS] nn submodule
[PASS] AVX-VNNI
[PASS] add(1, 2) = 3
---
All 10 tests passed.
```

**检查项**：版本号、7 个子模块、AVX-VNNI 指令、基础算子。

### 15.3 `sgn.util` 工具箱

#### `ensure_built()` — 检查 .pyd 是否最新

```python
>>> sgn.util.ensure_built()
sgn.pyd: C:\...\engine\sgn\build\sgn.cp314-win_amd64.pyd
  Build time: 2026-08-06 10:32:46
  Size: 2034.0 KB
  All source files are up-to-date.
```

如果源文件比 .pyd 新，会列出需要重新编译的文件。

#### `Timer` — 简单计时器

支持上下文管理器和手动启停两种模式：

```python
# 上下文管理器
with sgn.util.Timer("forward"):
    y = model.forward(x)
# [forward] 1.23 ms

# 手动启停
t = sgn.util.Timer("training")
t.start()
# ... work ...
t.stop()
print(f"{t.elapsed_ms:.2f} ms")
```

#### `describe()` — 打印参数信息

```python
>>> model = sgn.models.MLP()
>>> sgn.util.describe(model.fc1_w)
Parameter(shape=(128, 784), numel=100352, dtype=float32, requires_grad=True)
```

支持 `Parameter`、`Tensor`、`Buffer` 等类型。

#### `auto_build()` — 自动编译（v0.8.0 新增）

```python
>>> sgn.util.auto_build()
sgn.pyd: C:\...\engine\sgn\build\sgn.cp314-win_amd64.pyd
  Source is newer than .pyd, rebuilding...
  cmake --build build --config Release
  Build succeeded.
```

当 `.pyd` 缺失或源码过期时，自动运行 `cmake --build`。若已是最新，不执行任何操作。

#### `count_parameters()` — 模型参数量统计（v0.8.0 新增）

```python
>>> model = sgn.models.MLP()
>>> sgn.util.count_parameters(model)
109386
```

递归遍历所有子模块的 `parameters()`，返回总参数量。

#### `summary()` — 模型结构摘要（v0.8.0 新增）

```python
>>> print(sgn.util.summary(model))
Layer                           Output Shape         Param #
============================================================
fc1.weight                      [128, 784]              100352
fc1.bias                        [128]                      128
fc2.weight                      [64, 128]                  8192
fc2.bias                        [64]                        64
fc3.weight                      [10, 64]                   640
fc3.bias                        [10]                        10
============================================================
Total params: 109386
```

按层打印参数名、形状和参数量，底部汇总。

### 15.4 全局 verbosity 控制（v0.8.0 新增）

```python
import sgn

# 查询当前级别
sgn.get_verbosity()  # 1 (normal)

# 静默模式（0=quiet）
sgn.set_verbosity(0)

# 调试模式（2=debug）
sgn.set_verbosity(2)
```

| 级别 | 值 | 说明 |
|------|----|------|
| quiet | 0 | 不输出任何日志 |
| normal | 1 | 默认，输出关键信息 |
| debug | 2 | 输出详细调试信息 |

### 15.5 `Module.__repr__` — 模型结构树形打印

`sgn.nn.Module` 子类会自动显示参数和子模块的树形结构：

```python
>>> model = sgn.models.MLP(input_size=784, hidden1=128, hidden2=64, num_classes=10)
>>> print(model)
MLP(
  (fc1_w): Parameter([128, 784])
  (fc1_b): Parameter([128])
  (fc2_w): Parameter([64, 128])
  (fc2_b): Parameter([64])
  (fc3_w): Parameter([10, 64])
  (fc3_b): Parameter([10])
)

>>> model = sgn.models.CNN4(in_channels=3, img_size=32, num_classes=10)
>>> print(model)
CNN4(
  (conv1_w): Parameter([32, 3, 3, 3])
  (conv1_b): Parameter([32])
  (conv2_w): Parameter([64, 32, 3, 3])
  (conv2_b): Parameter([64])
  (fc1_w): Parameter([256, 4096])
  (fc1_b): Parameter([256])
  (fc2_w): Parameter([10, 256])
  (fc2_b): Parameter([10])
)
```

### 15.6 测试脚本

完整测试脚本位于 [engine/sgn/tests/test_util_diagnose.py](engine/sgn/tests/test_util_diagnose.py)，覆盖 19 项测试：

```bash
cd SGN/
python engine/sgn/tests/test_util_diagnose.py
```

预期输出 `Tests: 19/19 passed, All tests passed!`。

---

## 16. Loss 模块（v0.8.0 新增）

`sgn.loss` 提供可插拔的损失函数，支持**计算模式**和**诊断模式**双轨设计。

**设计动机**：此前 SGN 没有内置 Loss 函数，用户需手动用 numpy 计算损失和梯度，样板代码重复且容易出错（如残差反向传播 bug）。`sgn.loss` 将损失计算封装为统一接口，同时内置诊断器用于开发期验证。

**文件位置**：
- [engine/sgn/loss.py](engine/sgn/loss.py) — 损失函数 + 诊断器
- [engine/sgn/logger.py](engine/sgn/logger.py) — 梯度日志与诊断统一入口
- [engine/sgn/tests/test_loss.py](engine/sgn/tests/test_loss.py) — 63 项测试

### 16.1 计算模式（轨一）

#### 内置损失函数

| 类 | 公式 | 标签格式 |
|----|------|----------|
| `MSELoss` | `L = 0.5 * mean((y_pred - y_true)^2)` | `(batch, *)` 任意形状 |
| `CrossEntropyLoss` | softmax + NLL | `(batch,)` 类别索引 或 `(batch, C)` one-hot |
| `WeightedSumLoss` | `L = sum(w_i * loss_i)` | 取决于子损失 |

#### 基本用法

```python
import sgn
ag = sgn.autograd

criterion = sgn.loss.CrossEntropyLoss()

# 前向：录制 tape
with ag.record_scope(clear=True):
    logits = model.forward([x])

# 损失计算（numpy 域）
out_np = logits.to_numpy()
loss, dY = criterion(out_np, y_np)

# 反向：启动自动微分
logits.backward(dY.astype(np.float32))
optimizer.step()
```

**关键设计**：
- `forward()` 在 **numpy** 中计算损失值，不进入 C++ tape
- `backward()` 返回 **numpy array**，形状与 `y_pred` 一致
- 损失模块 **不管理 tape 生命周期**，tape 由 `record_scope()` 统一管理

#### 多任务加权组合

```python
criterion = sgn.loss.WeightedSumLoss(
    [("mse", sgn.loss.MSELoss()),
     ("ce", sgn.loss.CrossEntropyLoss())],
    weights=[0.3, 0.7],
)

# y_true 可以是 list，每个损失函数用自己的 target
loss, dY = criterion(y_pred, [y_true_mse, y_true_ce])
```

#### 梯度日志输出

损失函数支持内置日志，通过 `verbose` 和 `log_every` 控制：

```python
criterion = sgn.loss.CrossEntropyLoss(verbose=True, log_every=10)
# 每 10 步输出一次：
# [LossLog] CrossEntropyLoss | loss=2.302585e+00 | grad shape=(4,10) | L2=...
```

#### 自定义损失函数

继承 `BaseLoss`，实现 `forward()` 和 `backward()`：

```python
class MyLoss(sgn.loss.BaseLoss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._cached_y_pred = y_pred
        self._cached_y_true = y_true
        diff = y_pred - y_true
        return float(np.mean(np.abs(diff)))  # MAE

    def backward(self) -> np.ndarray:
        diff = self._cached_y_pred - self._cached_y_true
        return np.sign(diff) / diff.size
```

### 16.2 诊断模式（轨二）

`LossDiagnoser` 是独立的第三方裁判，用数值梯度检验用户的解析梯度，**只报告不修复**。

#### 一键诊断

```python
import sgn

model = sgn.nn.Sequential(
    sgn.nn.Linear(4, 8),
    sgn.nn.ReLU(),
    sgn.nn.Linear(8, 2),
)
criterion = sgn.loss.MSELoss()

diagnoser = sgn.loss.LossDiagnoser(model, criterion, sample_x, sample_y)
report = diagnoser.diagnose()
print(report.summary())

if report.is_ready_for_training():
    print("可以开始训练！")
```

**输出示例**：

```
============================================================
  SGN 损失函数诊断报告
============================================================
  总计: 10 项检查
  通过: 9  |  警告: 1  |  错误: 0
============================================================
  [✓ PASS] static/batch_size_match: batch_size: 3
  [✓ PASS] static/parameter_completeness: 所有参数参与前向计算
  [✓ PASS] forward/nan_inf_check: y_pred 无 NaN/Inf
  [✓ PASS] forward/output_range: mean=0.0123, std=0.8567, ...
  [✓ PASS] forward/loss_value: loss=0.234567
  [✓ PASS] backward/loss_grad_nan_inf: ∂L/∂y_pred 无 NaN/Inf
  [✓ PASS] backward/gradient_vanishing: max_norm=1.2345e-01
  [✓ PASS] backward/gradient_explosion: max_norm=1.2345e-01 (正常范围)
  [✓ PASS] backward/gradient_flow: 所有 4 个参数梯度非零
  [✓ PASS] gradient/numerical_gradient: median_diff=1.23e-02, ...
============================================================
  结论: 有警告但无错误，建议检查警告项后开始训练
```

#### 诊断模式详解

| 模式 | 方法 | 检查内容 | 计算成本 |
|------|------|----------|----------|
| `static` | `run_static_checks()` | batch_size 匹配、参数完整性（孤立参数检测） | 零 |
| `forward` | `run_forward_checks()` | NaN/Inf、输出范围、loss 值 | 低 |
| `backward` | `run_backward_checks()` | 梯度 NaN/Inf、消失/爆炸/流通 | 低 |
| `gradient` | `run_gradient_correctness()` | 数值梯度 vs 解析梯度（抽样对比） | 高 |
| `sgn` | `run_sgn_specific_checks()` | Δ 适配性（EF 临界间隙） | 低 |

**按需运行**：

```python
# 仅静态 + 前向（快速，适合每次改代码后运行）
report = diagnoser.diagnose(modes=['static', 'forward'])

# 训练前完整检查（含数值梯度，较慢）
report = diagnoser.diagnose()  # 默认全部模式

# 仅数值梯度（训练中周期性抽样）
results = diagnoser.run_gradient_correctness(eps=1e-2, n_samples=20)
```

**数值梯度说明**：`eps=1e-2`（默认）适配 SGN Tensor 的 float32 精度，过小的 eps（如 1e-5）会被 float32 精度淹没。用中位数判断偏差，对 ReLU kink 点等 outlier 鲁棒。

#### 诊断结果级别

| 级别 | 含义 | 行动 |
|------|------|------|
| `PASS` | 通过 | 无需操作 |
| `WARN` | 警告 | 建议检查，但不阻止训练 |
| `ERROR` | 错误 | 必须修复后再训练 |

**训练中快速抽检**：`examples/mnist_mlp.py` 和 `examples/cifar10_cnn.py` 内置了 `_run_gradient_spot_check()` 函数，封装 `diagnose(modes=['gradient'])` 返回 `(median_diff, max_diff, max_param)`，用于训练中周期性监控梯度健康度，检测退化（>5× 基线时告警）。

### 16.3 GradLogger 统一入口

`GradLogger` 同时提供运行时日志和开发期诊断，通过 `sgn.logger.GradLogger` 访问。

```python
from sgn.logger import GradLogger

logger = GradLogger(tag="TrainLog", enabled=True, log_every=10)

# 开发期：训练前诊断
report = logger.diagnose(model, criterion, sample_x, sample_y)
if not report.is_ready_for_training():
    raise RuntimeError("诊断未通过，请修复梯度问题")

# 仅检查梯度
grad_result = logger.check_gradient(model, criterion, sample_x, sample_y)
print(f"梯度检查: {grad_result.level.value}")

# 仅前向检查
fwd_report = logger.check_forward(model, criterion, sample_x, sample_y)

# 运行时：梯度日志（在损失函数 __call__ 中自动调用）
criterion = sgn.loss.CrossEntropyLoss(verbose=True, log_every=10)
```

**动态调整**：

```python
# 运行时开关
logger.enabled = False   # 关闭日志
logger.enabled = True    # 开启日志

# 调整采样频率
logger.log_every = 50    # 每 50 步输出一次
logger._call_count = 0   # 重置计数器
```

### 16.4 与现有代码的兼容性

- 现有手动计算模式**完全保留**，`sgn.loss` 是可选的
- `Tensor.backward()` 接口不变，loss 模块通过它启动反向传播
- `mnist_mlp.py` 和 `cifar10_cnn.py` 可逐步迁移到 `sgn.loss`，不强制

---

## 附录：文件结构

```
engine/sgn/autograd/
├── tensor.h / tensor.cpp           # Tensor + Storage（基础数据结构）
├── ops.h / ops.cpp                 # matmul forward/backward（AVX2 FMA）
├── ops_nn.h / ops_nn.cpp           # linear/relu/conv2d/bn/maxpool forward/backward + STE + 融合
├── backward_strategy.h             # 反向传播策略枚举 + 量化工具函数
├── autograd.h / autograd.cpp       # Tape + autograd-aware 算子（matmul/reshape）
├── autograd_nn.h / autograd_nn.cpp # autograd-aware nn 算子（策略感知） + conv2d_relu 融合
├── autograd_bindings.cpp           # pybind11 绑定（注册到 sgn.autograd 子模块）
├── module.h / module.cpp           # Module 基类 + Parameter/Buffer（稳定模式）
├── module_bindings.cpp             # Module pybind11 绑定（含 trampoline，支持 Python 子类化）
├── integer_ef.h / integer_ef.cpp   # AES-CTR PRNG 整数引擎
├── test_phase1.cpp ~ test_phase5.py # 各阶段测试
├── benchmark_phase5.py             # 性能基线
└── benchmark_ste_fusion.py         # STE + 融合算子性能

engine/sgn/tests/architecture/
└── test_dual_mode.py               # 双模式一致性测试

engine/sgn/
├── loss.py                          # 损失函数模块（计算 + 诊断）
├── logger.py                        # 梯度日志与诊断工具
└── tests/
    └── test_loss.py                 # 损失函数正确性测试（63 项）

examples/
├── mnist_mlp.py                     # MNIST MLP 训练示例（端到端验证）
├── cifar10_cnn.py                   # CIFAR-10 CNN 训练示例（端到端验证）
├── test_multi_strategy_mnist.py     # 多策略交叉测试（FLOAT32/GEF/SR）
└── test_diagnoser_catch_bug.py      # 诊断器抓 Bug 验证
```
