# SGN 用户使用手册

> **适用对象**: 普通用户（非开发者）—— 你不需要懂 C++、CMake 或编译器，只需会 pip install 和写 Python。
> **版本**: v0.10.1（2026-08-30）
> **版本策略**: 始终对应最新版本，不保留旧版历史
> **开发者**: 如你是协作者/团队成员，请参阅 [SGN C++ Autograd 用户操作手册](SGN_Autograd_用户操作手册.md)

---

## 📑 目录

1. [简介](#1-简介)
2. [快速开始](#2-快速开始)
   - [2.1 安装](#21-安装)
   - [2.2 第一个例子（最小训练闭环）](#22-第一个例子最小训练闭环)
3. [核心概念](#3-核心概念)
4. [构建模型](#4-构建模型)
5. [训练循环](#5-训练循环)
   - [5.1 推理模式（训练 / 评估切换）](#51-推理模式训练--评估切换)
6. [反向传播策略](#6-反向传播策略)
7. [序列化](#7-序列化)
8. [诊断与工具](#8-诊断与工具)
9. [Level 调度器](#9-level-调度器)
10. [数据桥接：与 numpy 互操作](#10-数据桥接与-numpy-互操作)
11. [常见问题](#11-常见问题)
12. [完整示例](#12-完整示例)
- [附录：API 速查](#附录api-速查)
- [附录：BatchNorm 推理临时处理](#附录batchnorm-推理临时处理)

---

## 1. 简介

SGN（Structured Gradient Network）是一个以**整数/量化路径为特色**的神经网络框架——它提供了独特的前向整数编码（HC/MSint）与多种量化反向传播策略，用于探索低精度、整数域的深度学习。

> **关于"整数框架"的理解**：SGN 的**核心特色**是整数/量化路径（MSint 编码、STE/GEF/SR 量化反向），但这**不代表 SGN 只能用整数**。为了**用户迁移与生态兼容**，SGN **默认采用 FLOAT32 反向**——这是最高精度、最稳妥的入手路径；其长期目标是在保持自身方向的同时，尽量兼容多平台、多计算路径，而非强制"纯整数"。因此，请不要把 SGN 理解成一个"只能用整数、不支持浮点"的框架——FLOAT32 是默认且开箱即用的。

SGN **完全独立运行，不需要 PyTorch 或其他深度学习框架**。

SGN 提供：
- 类 PyTorch 的自动微分引擎（tape-based）
- 标准神经网络层（Linear、Conv2d、ReLU 等），一行代码构建模型
- 内置损失函数（MSELoss、CrossEntropyLoss、WeightedSumLoss）
- 内置 SGD 优化器，开箱即用
- 多种整数反向传播策略（STE、GEF、SR）
- 与 numpy 的数据桥接

---

## 2. 快速开始

### 2.1 安装

**方式一：pip 安装（推荐，无需 C++ 编译器）**

在项目根目录（`SGN/`，即包含 `pyproject.toml` 或 `setup.py` 的目录）执行。**为避免当前目录不对导致 `pip install -e .` 报错，建议使用绝对路径**：

```bash
# 用绝对路径，无论当前在哪个目录都能正确安装
pip install -e /path/to/SGN

# 如果你已 cd 到 SGN/ 目录内，也可用相对写法
pip install -e .
```

安装后可从任意目录 `import sgn`，无需手动设置路径。

> 如果你的环境没有安装 Visual C++ Redistributable，`pip install` 可能失败。请先安装 [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)（64位版本）。

> 如需从源码编译安装，请参阅 [SGN C++ Autograd 用户操作手册](SGN_Autograd_用户操作手册.md)。

**验证安装：**

```python
import sgn
print(sgn.diagnose())  # 打印版本、编译器、CPU 特性、子模块状态
```

### 2.2 第一个例子（最小训练闭环）

```python
import numpy as np
import sgn

ag = sgn.autograd
nn = sgn.nn

# 1. 定义网络
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# 2. 前向传播
x = np.random.randn(4, 784).astype(np.float32)
with ag.record_scope(clear=True):
    y = model([x])

print(y.to_numpy().shape)  # (4, 10)

# 3. 反向传播
y.backward(np.ones_like(y.to_numpy()))  # grad 可以是 numpy 数组或 Tensor
grads = {name: p.grad for name, p in model.named_parameters()}  # 从 tape 收集梯度

# 4. 参数更新（真正的"训练"一步，优化器基础设施 B1，2026-08-29）
optimizer = sgn.SGD(model, lr=0.01, momentum=0.9)   # 支持 nn.Module 或 dict[str,ndarray]
optimizer.step(grads)   # 显式梯度更新（非 PyTorch 无参 step）
```
> 优化器：`sgn.SGD(params, lr, momentum, weight_decay, classical)` / `sgn.Adam(...)`；
> `step(grads)` 显式传入 {name: ndarray}（本项目 tape 每步重建 Tensor，梯度由调用方收集）。
> 检查点：`sgn.save_checkpoint / load_checkpoint` 保存完整训练状态（参数+优化器+step+随机源），
> 断点续训逐位可复现。

> **注意**：`model([x])` 使用方括号将输入包装为列表。这是因为 SGN 的 Module 支持多输入（如 Siamese 网络），即使单输入也需包装成列表。这与 PyTorch 的 `model(x)` 不同，请留意。
>
> 上面的示例只展示了**一轮前向 → 反向 → 更新**的最小闭环；一个完整的**训练循环**（数据分批、损失函数、多轮迭代、梯度抽检）见 [第 5 节 训练循环](#5-训练循环)。

---

## 3. 核心概念

### 3.1 Tensor

Tensor 是 SGN 的基本数据结构，类似 numpy 的 ndarray 但带有自动微分支持。

```python
# 创建 Tensor
t = ag.Tensor.from_numpy(np_array)   # 从 numpy 拷贝（推荐）
t = ag.Tensor([2, 3])                # 按形状创建（零初始化）

# 属性
t.shape          # 形状（list[int]）
t.ndim           # 维度数
t.numel          # 元素总数
t.requires_grad  # 是否需要梯度（可读可写）

# 操作
t.to_numpy()     # 转为 numpy 数组（拷贝）
t.reshape([6])   # 改变形状
```

### 3.2 Tape 录制

SGN 使用 tape 记录前向计算图，反向时自动遍历。

```python
# 推荐方式：上下文管理器（自动 clear + start + stop）
with ag.record_scope(clear=True):
    c = ag.matmul(a, b)
    d = ag.relu(c)

# 等效的手动方式
ag.start_recording()
c = ag.matmul(a, b)
ag.stop_recording()
```

### 3.3 反向传播

```python
# 方式一：传入 numpy 数组
c.backward(grad_numpy)    # grad_numpy 是 numpy 数组，形状与 c 一致

# 方式二：传入 Tensor
c.backward(grad_tensor)   # grad_tensor 是 ag.Tensor，形状与 c 一致

# 方式三：不传参（默认 grad = 全1，形状与 c 一致）
c.backward()

# 读取梯度
print(a.grad)             # numpy 数组或 None（如果未参与反向）
```

**关键规则**：
- 输入 Tensor 必须设 `requires_grad = True`，否则梯度链中断
- 不在 tape 录制期间创建/使用的 Tensor 不会参与反向
- `grad` 可接受 `numpy.ndarray` 或 `ag.Tensor`，框架自动识别

### 3.4 损失函数

SGN 内置 `sgn.loss` 模块，提供常用的损失函数，无需手动计算梯度。

**使用内置损失函数（推荐）**：

```python
import sgn

criterion = sgn.loss.CrossEntropyLoss()  # 或 MSELoss()

with ag.record_scope(clear=True):
    logits = model([x])

out_np = logits.to_numpy()
loss, dY = criterion(out_np, y_np)         # 一行计算 loss + 梯度
logits.backward(dY.astype(np.float32))
```

`__call__` 返回 `(loss, grad)` 元组，loss 用于监控，grad 传入 `backward()`。

**内置损失函数**：

| 类 | 公式 | 适用场景 |
|----|------|----------|
| `MSELoss` | `0.5 * mean((y_pred - y_true)^2)` | 回归任务 |
| `CrossEntropyLoss` | softmax + 交叉熵 | 分类任务（支持类别索引和 one-hot 标签） |
| `WeightedSumLoss` | `sum(w_i * loss_i)` | 多任务学习 |

**多任务加权组合**：

```python
criterion = sgn.loss.WeightedSumLoss([
    ("mse", sgn.loss.MSELoss()),
    ("ce", sgn.loss.CrossEntropyLoss()),
], weights=[0.3, 0.7])
loss, dY = criterion(y_pred, [y_true_mse, y_true_ce])
```

**仍然支持手动计算**（兼容旧代码）：

```python
# 手动计算 MSE 梯度
loss_grad = (y_pred.to_numpy() - y_label) / batch_size
y_pred.backward(loss_grad)

# 手动计算交叉熵梯度
out_np = logits.to_numpy()
exp = np.exp(out_np - out_np.max(axis=1, keepdims=True))
softmax = exp / exp.sum(axis=1, keepdims=True)
dL = softmax.copy()
dL[np.arange(batch_size), class_indices] -= 1.0
dL /= batch_size
logits.backward(dL.astype(np.float32))
```

> 手动计算模式完全保留，`sgn.loss` 是可选的。
> 完整可运行示例见 [examples/mnist_mlp.py](examples/mnist_mlp.py)、[examples/cifar10_cnn.py](examples/cifar10_cnn.py)。
> 验证脚本：[examples/test_multi_strategy_mnist.py](examples/test_multi_strategy_mnist.py)（多策略交叉测试）、[examples/test_diagnoser_catch_bug.py](examples/test_diagnoser_catch_bug.py)（诊断器抓 Bug 验证）。

---

## 4. 构建模型

### 4.1 使用标准层（推荐）

`nn.Linear`、`nn.Conv2d` 等标准层自动创建和注册参数，无需手动管理：

```python
import sgn
nn = sgn.nn

# 全连接层
fc = nn.Linear(784, 256)          # 自动创建 weight [256,784] + bias [256]

# 卷积层
conv = nn.Conv2d(3, 32, 3, stride=1, padding=1)  # weight [32,3,3,3] + bias [32]

# 激活层（无参数）
relu = nn.ReLU()
pool = nn.MaxPool2d(2, 2)

# BatchNorm
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

| 层 | 用法 | 参数 |
|----|------|------|
| Linear | `nn.Linear(in, out)` | weight, bias |
| Conv2d | `nn.Conv2d(in_c, out_c, k, stride, pad)` | weight, bias |
| ReLU | `nn.ReLU()` | 无 |
| MaxPool2d | `nn.MaxPool2d(kernel, stride)` | 无 |
| BatchNorm2d | `nn.BatchNorm2d(num_features)` | gamma, beta, running_mean, running_var |
| Sequential | `nn.Sequential(*layers)` | 子模块 |

### 4.2 自定义 Module（高级用法）

```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = inputs[0]
        x = self.fc1.forward([x])
        x = self.relu.forward([x])
        x = self.fc2.forward([x])
        x = self.relu.forward([x])
        x = self.fc3.forward([x])
        return x

model = MyNet()
model.train()
```

> **提示**：标准层在 `__init__` 中自动创建和注册参数，因此必须在 `super().__init__()` 之后调用。

### 4.3 调试模式（自由函数）

适合快速验证，不需要定义 Module：

```python
x = ag.Tensor.from_numpy(x_np)
x.requires_grad = True

with ag.record_scope(clear=True):
    h = ag.linear(x, w_tensor, b_tensor)
    h = ag.relu(h)
    h = ag.linear(h, w2_tensor, b2_tensor)

h.backward()
```

---

## 5. 训练循环

SGN 自带 `sgn.SGD` 优化器（基础设施 B1，2026-08-29），无需依赖 PyTorch 或其他框架：

```python
import sgn
ag = sgn.autograd
nn = sgn.nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
model.train()

# 创建优化器（支持 nn.Module 或 dict[str,ndarray]）
optimizer = sgn.SGD(model, lr=0.01, momentum=0.9)

for step in range(num_steps):
    # 1. 准备数据
    x_batch = np.random.randn(32, 784).astype(np.float32)
    y_batch = np.random.randn(32, 10).astype(np.float32)

    # 2. 前向（record_scope 自动管理 tape）
    with ag.record_scope(clear=True):
        y_pred = model([x_batch])

    # 3. 计算损失梯度（MSE：dL/dy_pred = (y_pred - y_label) / N）
    loss_grad = (y_pred.to_numpy() - y_batch) / y_batch.shape[0]

    # 4. 反向传播
    y_pred.backward(loss_grad)

    # 5. 从 tape 收集梯度并更新参数（一行完成）
    grads = {name: p.grad for name, p in model.named_parameters()}
    optimizer.step(grads)
```

`optimizer.step(grads)` 用 `{name: ndarray}` 显式更新对应参数（SGD 支持经典速度式
`v=m·v-lr·g; p+=v`（`classical=True`，复现训练脚本手写循环）与默认 PyTorch 式 `p-=lr·v`；
Adam 见 `sgn.Adam`）。tape 生命周期由 `record_scope` 管理（自动 clear+start+stop），
**无需手动 zero_grad**。

### 5.1 推理模式（训练 / 评估切换）

SGN 的 Module 有**训练 / 推理**两种模式，可切换如下：

```python
model.train()   # 训练模式（默认）
model.eval()    # 推理模式
```

> **⚠️ 当前实现的重要限制（关于 BatchNorm）**：
> - SGN 已具备 `train()`/`eval()` 模式切换，以及 BatchNorm 的 `running_mean`/`running_var` 缓冲区；
> - 但**当前 BatchNorm 的 eval（推理）前向尚未实现**：标准层 `BatchNorm2d` 的前向仍按训练模式计算（用当前 batch 统计量，并更新 running 统计），`model.eval()` 目前**不会**自动改用 running 统计。
> - 因此**现阶段含 BatchNorm 的网络做推理时，需手动处理 BN**（例如自行用保存的 running 统计计算归一化），或关注后续版本对 BN eval 前向的实现。不含 BatchNorm/Dropout 的模型不受影响。
>
> 一个**临时用法示例**（推理时手动用 running 统计归一化，三步）：
> ① 取该层保存的 `running_mean` / `running_var`（shape `(C,)`）；
> ② 对每个通道 `x_hat = (x - mu) / sqrt(var + eps)`，其中 `eps` 取**训练时相同的值**（通常为 `1e-5`）；
> ③ `y = gamma * x_hat + beta`。
> 完整可运行示例见文末「附录：BatchNorm 推理临时处理」。
>
> 待后续版本实现 BN eval 前向后，可直接用 `model.eval()` 自动切换，不再需要上述手写。
> 训练/推理模式切换在框架层面已就绪；此限制的详细说明见开发者手册的"功能限制"一节。

---

## 6. 反向传播策略

SGN 支持多种反向传播策略，可在训练前切换：

```python
ag = sgn.autograd

# 查看当前策略
print(ag.strategy_name(ag.get_backward_strategy()))

# 切换策略
ag.set_backward_strategy(ag.BackwardStrategy.FLOAT32)  # 纯 float32 反向（默认）
ag.set_backward_strategy(ag.BackwardStrategy.STE)       # 前向量化 + 反向 float32 直通
ag.set_backward_strategy(ag.BackwardStrategy.GEF)       # HC16 + GEF 梯度误差补偿
ag.set_backward_strategy(ag.BackwardStrategy.SR)        # 随机量化
```

**策略说明**：

| 策略 | 适用场景 | 说明 |
|------|---------|------|
| FLOAT32 | 验证/调试 | 纯 float32 反向，精度最高但无量化 |
| STE | 量化训练入门 | 前向量化量化，反向直通，简单高效 |
| GEF | 高精度量化训练 | 梯度误差补偿，精度接近 FLOAT32 |
| SR | 随机量化 | 随机量化，无偏但方差较大 |

---

## 7. 序列化

### 保存模型

```python
state = model.state_dict()
# state 是 dict[str, np.ndarray]
np.savez("model_weights.npz", **state)
```

### 加载模型

```python
data = np.load("model_weights.npz")
state = {k: data[k] for k in data.files}
model.load_state_dict(state)
# load_state_dict 包含完整 shape 校验

# 加载后做推理前，务必切换到推理模式（详见 5.1）
model.eval()   # （若模型含 BatchNorm：当前 eval 前向尚未实现，还需按 5.1 的临时用法手动处理 running 统计）
```

> **完整训练检查点（基础设施 B2，2026-08-29）**：模型权重保存用 `state_dict/np.savez` 即可；
> 但**训练中断续跑**（长验证实验）需同时保存优化器状态与随机源，用
> `sgn.save_checkpoint(path, params=..., optimizer=opt, step=t, seed=s, rngs={...})` /
> `sgn.load_checkpoint(path)`——断点续训与不间断训练逐位一致（含 rng 恢复），
> 替代手写 np.savez。

---

## 8. 诊断与工具

```python
import sgn

# 一键诊断：打印环境信息、版本、构建状态
sgn.diagnose()

# 快速自检：运行内置测试
sgn.test()

# 打印模型结构
print(model)

# 查看参数信息
sgn.util.describe(model)

# 统计参数量
print(f"总参数: {sgn.util.count_parameters(model)}")

# 打印模型结构摘要
print(sgn.util.summary(model))

# 梯度诊断：在训练前验证损失函数的梯度是否正确
from sgn.logger import GradLogger

logger = GradLogger(tag="TrainLog", enabled=True)
report = logger.diagnose(model, criterion, sample_x, sample_y)
if report.is_ready_for_training():
    print("梯度检查通过，可以开始训练！")
```

> `LossDiagnoser` 会自动检查：静态参数完整性、前向 NaN/Inf、反向梯度消失/爆炸、数值梯度 vs 解析梯度对比。

> **C++ 内核日志**（可选）：可用环境变量 `SGN_LOG_LEVEL=DEBUG|INFO|WARN|ERROR` 控制 C++ 侧日志（默认 `INFO`，显示常规/警告/错误；`DEBUG` 输出详细诊断）。例如 PowerShell：`$env:SGN_LOG_LEVEL = "DEBUG"` 后再运行脚本。
>
> C++ 内核日志行带**毫秒级时间戳**与规则前缀，格式如 `[SGN] [HH:MM:SS.mmm] [LEVEL] file:line msg`。内置两条进程级 `INFO` 打点（log 输出到标准错误 stderr）：
> - `sgn.autograd module init` —— 首次 `import sgn` 时记录模块初始化（可作进程"开始时间"）；
> - `kernel backend selected: <name>` —— 内核能力检测完成后记录所选后端（如 `x86_avx2`）。
>
> 这两条默认在 `SGN_LOG_LEVEL=INFO` 下即可见；若不想看到，设 `SGN_LOG_LEVEL=WARN` 或 `ERROR` 仅保留告警/错误。Python 侧的统计型日志见上方 `sgn.logger.GradLogger`，两套相互独立。

**训练中快速梯度抽检**（周期性监控梯度健康度）：

```python
from sgn.loss import LossDiagnoser

def spot_check(model, criterion, x, y, eps=1e-2, n_samples=10):
    """快速数值梯度抽检：返回 (median_diff, max_diff, max_param) 或 None"""
    diagnoser = LossDiagnoser(model, criterion, x, y, sgn_module=sgn)
    report = diagnoser.diagnose(modes=['gradient'], eps=eps, n_samples=n_samples)
    for c in report.checks:
        if c.name == 'numerical_gradient' and 'median_diff=' in c.message:
            # 从报告消息中提取指标
            parts = c.message.split(', ')
            median_diff = float([p for p in parts if 'median_diff=' in p][0].split('=')[1])
            max_diff = float([p for p in parts if 'max_diff=' in p][0].split('=')[1].split()[0])
            max_param = [p for p in parts if 'at ' in p][0].split('at ')[1].split(' ')[0]
            return median_diff, max_diff, max_param
    return None

# 使用示例：训练中每 N 步抽检
baseline = spot_check(model, criterion, x_sample, y_sample)
for step in range(steps):
    # ... 训练 ...
    if (step + 1) % 25 == 0:
        result = spot_check(model, criterion, x_batch, y_batch)
        if result and baseline and result[0] > baseline[0] * 5:
            print(f"[WARN] 梯度退化! step {step+1}")
```

> 此函数已在 [examples/mnist_mlp.py](examples/mnist_mlp.py) 和 [examples/cifar10_cnn.py](examples/cifar10_cnn.py) 中作为 `_run_gradient_spot_check()` 内置使用。

### 控制日志输出

```python
sgn.set_verbosity(0)  # 静默模式
sgn.set_verbosity(1)  # 默认
sgn.set_verbosity(2)  # 调试模式
```

---

## 9. Level 调度器

SGN 的 Level 调度器用于**动态精度分配**——根据数据分布特征，为不同层分配不同的 bit 宽度。以下是一个**端到端示例**：

> **⚠️ 规划中的功能演示（重要）**：本节展示的是 Level 调度器的**目标形态与调用方式**，`allocate()` 目前可用并能返回每层位宽建议，但**把这些建议落到具体层（Linear/Conv2d）的逐层量化配置仍在开发中，尚不能直接驱动实际训练**。因此下方案例是"示意"，alloc 结果**暂不直接生效到网络训练**——请勿据此期待某层真的变成 8/6/4 位。若你需要实际控制位宽，当前可靠做法仍是切换全局 [反向传播策略](#6-反向传播策略)（如 STE/SR）。

```python
import sgn
level = sgn.level

# 1. 创建调度器
scheduler = level.LevelScheduler()

# 2. 准备输入数据
#    activations: list of numpy 数组，每个数组对应一层的激活值
#    layer_configs: list of dict，每层一个配置
activations = [
    np.random.randn(32, 784).astype(np.float32),  # 第0层输入
    np.random.randn(32, 128).astype(np.float32),  # 第1层激活
    np.random.randn(32, 10).astype(np.float32),   # 第2层激活
]

layer_configs = [
    {"bits_per_neuron": 8, "max_range": 127},   # 第0层：8-bit
    {"bits_per_neuron": 6, "max_range": 31},    # 第1层：6-bit
    {"bits_per_neuron": 4, "max_range": 7},     # 第2层：4-bit
]

# 3. 分配精度
allocations = scheduler.allocate(activations, layer_configs)
# allocations 是 dict，包含每层建议的 bit 宽度

# 4. 在实际训练中应用分配的精度
#    ⚠️ 以下不会真正把位宽应用到层上，仅示意映射逻辑；
#    逐层位宽尚未直接生效（见本节开头 ⚠️ 说明）
ag = sgn.autograd
if allocations[0]["bits"] >= 8:
    ag.set_backward_strategy(ag.BackwardStrategy.FLOAT32)
elif allocations[0]["bits"] >= 4:
    ag.set_backward_strategy(ag.BackwardStrategy.STE)
else:
    ag.set_backward_strategy(ag.BackwardStrategy.SR)
```

> Level 调度器是一个高级功能。初学者可以先从默认的 FLOAT32 策略开始，等熟悉 SGN 的基本用法后再探索精度调度。
>
> **关于"逐层 bits 如何应用到层上"**：`allocate()` 返回的 `allocations` 给出的是**每层建议的位宽**，而真正把某一位宽落到 `Linear`/`Conv2d` 上，需要更底层的量化配置，并不在本手册的全局策略示例范围内。相关背景可参阅 [MSint 多精度拆分计算范式](msint_multisplit_paradigm/MSint多精度拆分计算范式.md) 与开发者手册的 Level 调度器章节；**当前把逐层位宽应用到具体层的量化配置仍在开发中**，尚未展开。

---

## 10. 数据桥接：与 numpy 互操作

SGN **完全独立运行**，无需 PyTorch 或其他深度学习框架。如果项目中有 PyTorch 代码，可以通过 numpy 桥接数据：

```python
# SGN ←→ numpy（推荐，安全）
sgn_tensor = ag.Tensor.from_numpy(numpy_array)  # numpy → SGN
numpy_array = sgn_tensor.to_numpy()              # SGN → numpy

# 如果你有 PyTorch 代码，可以通过 numpy 中转：
# PyTorch → numpy → SGN
sgn_tensor = ag.Tensor.from_numpy(pytorch_tensor.detach().numpy())

# SGN → numpy → PyTorch
pytorch_tensor = torch.from_numpy(sgn_tensor.to_numpy())
```

**数据流向**：

```
SGN 前向/反向  ◀──▶  numpy 数组  ◀──▶  PyTorch（可选）
```

> 注意：SGN 是独立框架，不需要 PyTorch 即可运行。PyTorch 桥接仅用于数据迁移或混合实验。

---

## 11. 常见问题

### 梯度为 None

- 检查输入 Tensor 是否设置了 `requires_grad = True`
- 检查操作是否在 `record_scope` 或 `start_recording` / `stop_recording` 之间
- 检查 Tensor 是否在录制期间创建的

### backward 报 shape 不匹配

- `backward(grad)` 的 `grad` 形状必须与输出 Tensor 一致
- 常见错误：batch 维度不匹配
- `grad` 可以是 `numpy.ndarray` 或 `ag.Tensor`

### 导入失败

**情况一：未安装 Visual C++ Redistributable**

SGN 的 C++ 扩展依赖于系统的 VC++ 运行时。如果 `import sgn` 报错 `DLL load failed`，请安装：
- [Visual C++ Redistributable for Visual Studio 2022](https://aka.ms/vs/17/release/vc_redist.x64.exe)（64位版本）

**情况二：安装后仍无法导入**

- 确保已在项目根目录执行 `pip install -e .`
- 如果仍有问题，尝试重新安装：`pip install -e . --force-reinstall`
- 如仍无法解决，请参阅 [SGN C++ Autograd 用户操作手册](SGN_Autograd_用户操作手册.md) 的故障排查章节

### 内存持续增长

- 使用 `record_scope(clear=True)` 自动管理 tape 生命周期
- 每次迭代前清空 tape

### 性能建议

- 在前向/反向循环外创建 Tensor，避免重复分配
- 使用 `record_scope(clear=True)` 自动管理 tape 生命周期
- 大数据量时考虑批量处理

### 指令集加速（SIMD）

SGN 已在支持相应指令集的 CPU 上自动启用多种底层加速，**无需任何额外配置**；在不支持的平台上会自动回退到标准实现，保证结果一致。

支持的加速项：批量解码（8×8bit / 转浮点）、矩阵乘法（AVX2 FMA + 多线程）、MSint 拆分点积（16/8/4 位窄路径 SIMD，含低位对角裁剪与摊销热路径）；"解码有效开销"为归一化相对指标，2026-08-16 十次均值口径实测约 **0.13×**（目标 ≤ 0.33×，达标）。

> 各加速项的**实测倍率、推导口径与一键复现方法**已集中收录于 [SGN 性能白皮书](SGN_性能白皮书.md)：SIMD 优化与各倍率见 §5；解码 vs memcpy 带宽与"解码有效开销 0.13×"的推导/复现口径见 §3.6。

#### MSint 拆分点积接口（当前可用）

MSint 拆分点积（`dot_split` / `dot_split_leveled`）在窄路径 SIMD 之上，当前提供以下可直接调用的接口（C++ 绑定 + Python fallback 双路径，未启用 C++ 扩展时自动回退、逐位一致）：

| 接口 | 用途 |
|------|------|
| `trim_high_diag`（`dot_split` 参数） | 低位对角裁剪：4 位档（n=8）裁剪高位对角 m≥8 的点积，省 **43.75% ALU**，int32 截断结果 bit-exact |
| `dot_fused_i32` | 单融合 int32 截断模式，内部走裁剪路径，消费端可直接调用 |
| `prepare_downcast` / `prepare_downcast_np` | 降档预解包融合：一次传入原始 w 与 importance，C++ 内部分组（`select_precision`）→ keep_top 截断 → `prepare_nibble_from_raw`，消除 Python 层分组/截断/转换开销（prepare 3.08ms → 0.36ms，累计 8.6x） |
| `select_precision` | 降档决策：按 importance 为逐元素选择精度位数 p ∈ {8,16,32}（低重要度 → 少位数），配合 `dot_split_leveled_downcast` / `dot_fused_leveled_downcast` 使用 |
| `dot_split_leveled_downcast` / `dot_fused_leveled_downcast` | 多输出 / 单融合的降档摊销点积：每组 keep_top + 4 位摊销点积，`n²` 计算量随精度位数平方下降（int8 档端到端反超 numpy 2.1~2.4x） |
| `prepare_nibble` + `narrow_dot_prepared4` / `matmul_prepared4` | 摊销热路径：同一 x 预解包一次、M 个输出行复用 n² 点积（省 64 次重复 nibble 解包） |

> 正确性：全部接口与标量路径 bit-exact 一致（`validate_math_msint_split_dot.py` / `validate_math_msint_leveled_dot.py` / `_tmp_verify_downcast_interface.py` 全 PASS）。接口语义与实测数据详见 [dot_split_simd_summary_report_2026_08_14.md](fixes_相关修复/architecture/dot_split_simd_summary_report_2026_08_14.md) 与 [prepare_x_python_overhead_fusion_plan_2026_08_14.md](fixes_相关修复/architecture/prepare_x_python_overhead_fusion_plan_2026_08_14.md)。

### 内存池加速（可选）

SGN 内置通用内存池（size-class 分桶 + 线程本地无锁复用），**默认关闭**，关闭时行为与标准分配完全一致。对反复创建/释放同尺寸张量的训练循环，可在训练开始前开启：

```python
import sgn
sgn.autograd.set_pool_allocator(True)   # 开启内存池（建议训练前调用）
# ... 训练 ...
sgn.autograd.set_pool_allocator(False)  # 关闭并清池（可选）
```

> 多 workload 实测均为**正向提速**，分配越密集收益越大；详细收益数据与基准口径见 [SGN 性能白皮书](SGN_性能白皮书.md) §7（2026-08-16 多 workload 基准）。

### 与 PyTorch 的速度对比（注意：不同架构）

SGN 提供与 PyTorch 的对比基准，最新实测（2026-08-16，6 层 CNN fwd+bwd，内核优化后）：B=16 下 SGN 约 44.06ms、慢约 **4.8×**（三档 3.3×/3.7×/4.8×，均为历次记录最低）。

> **⚠️ 重要：这是不同架构、不同数据处理方式的对比，不是同一份数据在同一路径上的对比。**
> - PyTorch 走的是**浮点**计算——业界最成熟、最常规的路径，有高度优化的 MKL/BLAS 加速。
> - SGN 走的是**自定义整数（MSint）表示与量化调度**——这是为探索低精度/整数网络而做的独特设计，而非与 PyTorch 在同一套数据上竞争。
>
> 因此"慢多少倍"反映的是**架构与数据处理路径的差异**，并不代表 SGN 处理同样的任务就一定慢这么多。SGN 的价值在于其自定义的整数表示、量化与 Level 调度能力，而非在浮点路径上追赶 PyTorch 的绝对速度。请不要据此理解为"同一任务 SGN 比 PyTorch 慢 10 倍"。完整对比表与历次复测见 [SGN 性能白皮书](SGN_性能白皮书.md) §3。
>
> **为什么会慢（具体可感知的开销）**：SGN 走的整数路径在多处比 PyTorch 的浮点路径多做了额外的数据变换——
> - **MSint 位布局的解包/解码**：MSint 为了多精度拆分而采用的特殊位布局，不是内存里直接的连续整数数组；每步前向之前需把它解码/转浮点，PyTorch 则直接把浮点喂给高度优化的 MKL/BLAS，少这一步解码；
> - **量化调度与视图解码**：每步按位宽视图（如 16/8/4 位漫游）解码、重量化后再参与计算，是一个额外的中间变换阶段；
> - **Python ↔ 引擎桥接**：当前训练循环用 `to_numpy()`、以 numpy 数组回传梯度，存在逐轮的 numpy 桥接开销（数据来回进出引擎内存），尚未全程留在引擎内。
>
> **未来优化方向**：窄路径 SIMD（16/8/4 位）已经随规模增大越来越快（位平面解码从 L1 到主存的速度比提升）；把位宽调度做成**编译期静态规划**（预先算好分配、避免运行期逐层解码）、减少 Python 桥接、以及层间算子融合，是目前最被看好的缓解路径。相关调研见 [SGN 性能白皮书](SGN_性能白皮书.md)。

### 如何选择反向传播策略？

- **刚入门**：使用默认的 `FLOAT32`，先跑通训练流程
- **想尝试量化**：切换到 `STE`，简单高效
- **追求高精度量化**：使用 `GEF`，精度接近浮点
- **探索随机量化**：使用 `SR`

### MSint 是整数，可以直接用外部整数库吗？

一个常见的联想是：「MSint 内部用的也是 int，所以可以直接调用外部的整数推理库 / 整数矩阵乘法库（无论 Google 的还是其他开源的）来加速，甚至替换 SGN 的计算」。这个方向**可以理解，但技术上不建议这样做**，原因有两点：

1. **外部整数库没有「整数反向传播」**。市面上主流的整数库（量化推理引擎、int8 矩阵库等）做的都是**前向推理**：给定权重和输入，算出输出。它们背后没有与 SGN 兼容的**整数反向传播链路**（梯度如何按整数语义回流、如何更新整数权重）。SGN 的价值是**完整训练闭环（前向 + 反向 + 更新）**，接一个只有前向的库进来，只是一块拼不上的碎片，凑不成训练。

2. **MSint 的「int」不是普通 int 数组**。MSint 内部是按**多精度拆分 + 偏置/符号约定**打包的特殊位布局（如 int32 拆成多个 4/8/16 位部分，配合偏置修正），不是外部库预期的那种直接可算的连续整数数组。外部库把它当普通 int 直接读，**语义对不上**——轻则算不出你要的结果，重则**乱码或直接报错**。

> 需要加速 MSint 计算时，请走 SGN 自己针对该位布局定制的窄路径 SIMD 内核（16/8/4 位，均已与标量 bit-exact 一致，见「指令集加速（SIMD）」小节），而不是尝试接外部整数库。

---

## 12. 完整示例

```python
import numpy as np
import sgn

ag = sgn.autograd
nn = sgn.nn

# 使用 GEF 策略
ag.set_backward_strategy(ag.BackwardStrategy.GEF)

# 定义网络（使用标准层）
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
model.train()
print(model)

# 创建损失函数和优化器
criterion = sgn.loss.MSELoss()
optimizer = sgn.SGD(model, lr=0.01, momentum=0.9)

# 训练循环
for step in range(100):
    x = np.random.randn(32, 784).astype(np.float32)
    y = np.random.randn(32, 10).astype(np.float32)

    with ag.record_scope(clear=True):
        y_pred = model([x])

    # 使用内置损失函数，一行完成 loss 和梯度计算
    loss, dY = criterion(y_pred.to_numpy(), y)
    y_pred.backward(dY.astype(np.float32))
    grads = {name: p.grad for name, p in model.named_parameters()}
    optimizer.step(grads)      # 一行完成参数更新（tape 由 record_scope 管理，无需 zero_grad）
```

> 更多示例见 [examples/mnist_mlp.py](examples/mnist_mlp.py) 和 [examples/cifar10_cnn.py](examples/cifar10_cnn.py)。
> 验证脚本：[examples/test_multi_strategy_mnist.py](examples/test_multi_strategy_mnist.py)（多策略交叉测试）、[examples/test_diagnoser_catch_bug.py](examples/test_diagnoser_catch_bug.py)（诊断器抓 Bug 验证）。

---

## 附录：API 速查

> **函数式 vs 有状态层（避免混淆）**：SGN 有两套"看似相同"的 API——
> - `sgn.autograd.relu(x)`、`sgn.autograd.linear(x, w, b)` 等是**函数式 API（无状态）**：直接对张量做一次计算，不保存任何可学习参数，适合在自定义 `Module.forward` 内部自由调用。
> - `sgn.nn.ReLU()`、`sgn.nn.Linear(...)` 等是**有状态层（层 API）**：是持有参数/缓冲区的 `Module`，需实例化后放入 `Sequential` 或作为子模块。
>
> 一般建模请用 `sgn.nn` 层；仅当你手写 forward 且需要临时做一次运算时，才用 `sgn.autograd` 函数式 API。

### `sgn.autograd`

| API | 说明 |
|-----|------|
| `Tensor.from_numpy(arr)` | 从 numpy 创建 Tensor |
| `Tensor(shape)` | 按形状创建 Tensor |
| `tensor.to_numpy()` | Tensor → numpy 拷贝 |
| `tensor.backward(grad)` | 反向传播（grad 接受 numpy 数组或 Tensor，不传参默认全1） |
| `tensor.reshape(shape)` | 改变形状 |
| `record_scope(clear=True)` | tape 录制上下文管理器 |
| `start_recording()` | 开始录制 |
| `stop_recording()` | 停止录制 |
| `matmul(a, b)` | 矩阵乘法 |
| `linear(x, w, b)` | 全连接层 |
| `relu(x)` | ReLU 激活 |
| `conv2d(x, w, b, stride, padding)` | 2D 卷积 |
| `maxpool2d(x, kernel, stride)` | 2D 最大池化 |
| `bn_train(x, gamma, beta, rm, rv, ...)` | BatchNorm **训练**模式（推理时需手动用 running 统计归一化，见 [5.1](#51-推理模式训练--评估切换)） |
| `reshape(x, shape)` | 改变形状 |
| `conv2d_relu(x, w, b, stride, padding)` | Conv2d+ReLU 融合 |

### `sgn.nn`

| API | 说明 |
|-----|------|
| `Module` | 模型基类 |
| `Parameter(shape)` | 可学习参数 |
| `Buffer(shape)` | 状态缓冲区 |
| `Linear(in_features, out_features)` | 全连接层 |
| `Conv2d(in_c, out_c, k, stride, pad)` | 2D 卷积层 |
| `ReLU()` | ReLU 激活 |
| `MaxPool2d(kernel, stride)` | 2D 最大池化 |
| `BatchNorm2d(num_features)` | 2D 批归一化 |
| `Sequential(*layers)` | 顺序容器 |
| `kaiming_uniform_(t, fan_in, a)` | Kaiming 初始化 |
| `uniform_(t, low, high)` | 均匀初始化 |
| `fill_(t, value)` | 常量填充 |

### `sgn` 优化器（基础设施 B1，2026-08-29）

| API | 说明 |
|-----|------|
| `SGD(params, lr, momentum, classical)` | 随机梯度下降优化器（params=nn.Module 或 dict[str,ndarray]） |
| `Adam(params, lr, betas, eps)` | Adam 优化器 |
| `optimizer.step(grads)` | 用 {name: ndarray} 显式更新参数（梯度从 tape 的 p.grad 收集） |
| `save_checkpoint / load_checkpoint` | 完整训练状态保存/恢复（参数+优化器+step+随机源） |

### `sgn.util`

| API | 说明 |
|-----|------|
| `count_parameters(model)` | 统计模型参数量 |
| `summary(model)` | 打印模型结构摘要 |
| `describe(param)` | 打印单个参数信息 |
| `Timer(name)` | 上下文管理器计时器 |

### `sgn` 全局

| API | 说明 |
|-----|------|
| `diagnose()` | 一键诊断（环境、版本、模块状态） |
| `test()` | 快速自检 |
| `set_verbosity(level)` | 设置日志级别（0=quiet, 1=normal, 2=debug） |
| `get_verbosity()` | 获取当前日志级别 |

### `sgn.loss`（v0.8.0 新增）

| API | 说明 |
|-----|------|
| `BaseLoss` | 损失函数基类，继承后实现 `forward()` / `backward()` |
| `MSELoss` | 均方误差损失：`0.5 * mean((y_pred - y_true)^2)` |
| `CrossEntropyLoss` | 交叉熵损失（含 softmax），支持类别索引和 one-hot 标签 |
| `WeightedSumLoss` | 多任务加权组合：`sum(w_i * loss_i)` |
| `LossDiagnoser` | 开发期诊断器：数值梯度验证、前向/反向健康检查 |
| `DiagnoseReport` | 诊断报告，含 `summary()` 和 `is_ready_for_training()` |

### `sgn.logger`（v0.8.0 新增）

| API | 说明 |
|-----|------|
| `GradLogger(tag, enabled, log_every)` | 梯度日志器，支持运行时开关和采样频率控制 |
| `logger.diagnose(model, criterion, x, y)` | 一键诊断，返回 `DiagnoseReport` |
| `logger.check_gradient(model, criterion, x, y)` | 快速数值梯度检查 |
| `logger.check_forward(model, criterion, x, y)` | 快速前向健康检查 |

### `sgn.level`（高级功能）

| API | 说明 |
|-----|------|
| `LevelScheduler` | 自适应精度调度器 |
| `BitsAllocator` | 精度分配器 |
| `LevelContext` | 精度上下文 |
| `bits_to_max_range(bits)` | bits → 最大范围 |
| `max_range_to_bits(max_range)` | 最大范围 → bits |

### 附录：BatchNorm 推理临时处理

> 背景：当前 `BatchNorm2d` 的 **eval（推理）前向尚未实现**，`model.eval()` 不会自动改用 running 统计（详见 [5.1](#51-推理模式训练--评估切换)）。推理时如需归一化，可用保存的 running 统计手动计算，代码如下：

```python
import numpy as np

# bn: 你已经训练好的 BatchNorm2d 层（参数与统计量在训练后已冻结）
# x : 输入，numpy 数组，shape (B, C, H, W)
mu  = bn.running_mean            # (C,) 训练累积均值
var = bn.running_var             # (C,) 训练累积方差
g   = bn.weight                  # (C,) gamma
btd = bn.bias                    # (C,) beta
eps = 1e-5

mu_c  = mu.reshape(1, C, 1, 1)   # 对齐 (B,C,H,W) 的通道维
var_c = var.reshape(1, C, 1, 1)
g_c   = g.reshape(1, C, 1, 1)
b_c   = btd.reshape(1, C, 1, 1)

x_hat = (x - mu_c) / np.sqrt(var_c + eps)   # 归一化
y_inf = g_c * x_hat + b_c                    # 仿射 → 推理输出
```

> 说明：若你的 `bn.weight` / `bn.bias` / `running_mean` / `running_var` 是引擎 Tensor，先 `.to_numpy()` 再参与 numpy 运算。待后续版本实现 BN eval 前向后，这段可替换为一行 `model.eval()`。