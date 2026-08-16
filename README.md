# SGN — Structured Gradient Network

SGN（Structured Gradient Network）是一个以**整数 / 量化路径为特色**的神经网络框架，提供独特的前向整数编码（HC / MSint）与多种量化反向传播策略（STE / GEF / SR），用于探索低精度、整数域的深度学习。

**SGN 完全独立运行，不需要 PyTorch 或其他深度学习框架**。默认采用 FLOAT32 反向（最高精度、开箱即用），同时保留完整的整数 / 量化路径——因此它既不是一个"只能用整数"的框架，也不是 PyTorch 的复刻。

---

## 核心特性

- **类 PyTorch 的自动微分引擎**（tape-based，多线程安全）
- **标准神经网络层**：Linear、Conv2d、ReLU、MaxPool2d、BatchNorm2d、Sequential，一行代码构建模型
- **内置损失函数**：MSELoss、CrossEntropyLoss、WeightedSumLoss
- **内置优化器**：SGD，开箱即用
- **多种反向传播策略**：FLOAT32（默认）/ STE / GEF / SR，可训练中切换
- **整数路径**：MSint 多精度拆分编码、HC 引擎、Level 调度器（动态精度分配）
- **CPU 指令集加速**：AVX2 FMA、AVX-VNNI、SSSE3、BMI1/BMI2、AES-NI，自动启用、自动回退
- **与 numpy 零成本桥接**，可经 numpy 中转接入 PyTorch 数据

---

## 快速开始

```bash
pip install -e /path/to/SGN
```

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

# 2. 前向
x = np.random.randn(4, 784).astype(np.float32)
with ag.record_scope(clear=True):
    y = model([x])          # 注意：Module 输入需用 [] 包装为列表

# 3. 反向
y.backward(np.ones_like(y.to_numpy()))

# 4. 训练
optimizer = sgn.optim.SGD(model, lr=0.01)
optimizer.step()
optimizer.zero_grad()
```

> 完整示例见 `examples/mnist_mlp.py`、`examples/cifar10_cnn.py`。

---

## 性能亮点（实测，口径见文档）

以下数据均来自本机实测，完整推导口径、复现脚本见 [SGN 性能白皮书](docs/SGN_性能白皮书.md)。

### 底层 SIMD 加速（bit-exact，非近似）

| 路径 | 实测加速 |
|------|----------|
| MSint 拆分点积 16 位（AVX2 窄路径） | 每点积 **8.9–9.3×** |
| MSint 拆分点积 8 位（AVX-VNNI） | 每点积 **26.9–32.3×** |
| MSint 拆分点积 4 位（nibble 打包） | 每点积 **72.2–95.2×** |
| batch_get_all（8×8bit，10000 元素） | **67.7×** vs Python 逐元素 |
| batch_decode_to_float（50000 样本） | **28.7×** vs numpy 位操作 |
| 解码有效开销（归一化相对指标） | **0.13×**（目标 ≤ 0.33×，达标） |

所有 SIMD 路径均与标量逐位一致（bit-exact），并保留非 x86（ARM/GPU）标量回退。

> **内存搬运速度参考**（复现口径见白皮书 §3.6）：本机纯 memcpy 带宽约 **11.9 GB/s**（每字节约 0.08 ns），C++ 批量解码约 **0.27 ns/元素**，已贴近带宽极限。上表中"解码有效开销 0.13×"是**归一化相对指标**（= MSint numpy 位操作解码相对 HC16 慢的倍数 ÷ C++ 批量解码相对 numpy 的加速比），**不是**内存搬运/带宽速度，两者不可混读。

### 通用内存池（可选，默认关闭）

size-class 分桶 + 线程本地无锁复用，对反复分配同尺寸张量的训练循环显著提速，**全部 workload 实测无负收益**：

| Workload | 提速 |
|----------|------|
| 小算子密集链（bn+relu ×16） | **+60.6%** |
| CNN fwd-only B=16 | **+36.7%** |
| CNN fwd+bwd @ STE 量化反向 | **+26.1%** |
| CNN fwd+bwd（B=4/8/16） | +12.0% / +7.0% / +7.8% |
| MLP fwd+bwd（B=64，3 层 Linear） | +3.7% |

### 与 PyTorch 的对比（注意：不同架构）

最新实测（6 层 CNN fwd+bwd，内核优化后）：B=16 下 SGN 约 **44.06ms**，三档差距 **3.3× / 3.7× / 4.8×**，均为历次记录最低。

> **⚠️ 重要**：这是**不同架构、不同数据处理路径**的对比——PyTorch 走高度优化的浮点路径（MKL/BLAS），SGN 走自定义整数（MSint）表示与量化调度。该倍数反映的是架构差异，并非"同一任务 SGN 必然慢"；SGN 的价值在于整数表示、量化与 Level 调度能力，而非在浮点路径上追赶绝对速度。完整对比与历次复测见白皮书 §3。

### 正确性

全量回归 **275 功能项通过**，SIMD 内核与标量逐位一致，多轮复测无回归。

---

## 文档

- [SGN 用户使用手册](docs/SGN_用户使用手册.md) — 面向普通用户（无需 C++ 基础）
- [SGN 性能白皮书](docs/SGN_性能白皮书.md) — 性能基线、SIMD 优化与复现方法
- [SGN 开发者操作手册](docs/SGN_Autograd_用户操作手册.md) — 面向开发者 / 协作者

---

## License

Apache-2.0
