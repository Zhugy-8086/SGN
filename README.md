# SGN — Structured Gradient Network

SGN（Structured Gradient Network）是一个以**整数 / 量化路径为特色**的神经网络框架，提供独特的前向整数编码（HC / MSint）与多种量化反向传播策略（STE / GEF / SR），用于探索低精度、整数域的深度学习。

**SGN 完全独立运行，不需要 PyTorch 或其他深度学习框架**。默认采用 FLOAT32 反向（最高精度、开箱即用），同时保留完整的整数 / 量化路径——因此它既不是一个"只能用整数"的框架，也不是 PyTorch 的复刻。

---

## 核心特性

- **类 PyTorch 的自动微分引擎**（tape-based，多线程安全）
- **标准神经网络层**：Linear、Conv2d、ReLU、MaxPool2d、BatchNorm2d、Sequential，一行代码构建模型
- **内置损失函数**：MSELoss、CrossEntropyLoss、WeightedSumLoss
- **内置优化器**：SGD / Adam，开箱即用
- **多种反向传播策略**：FLOAT32（默认）/ STE / GEF / SR，可训练中切换
- **整数路径**：MSint 多精度拆分编码、HC 引擎、Level 调度器（动态精度分配）
- **嵌套量化与平面推理**（`sgn.mkern_nested` / `sgn.leveled_state`）：一次 SR 量化到最细格，任意精度档位视图即取（Q4/Q8/Q16/Q32 零重采样切换）；LeveledState 平面布局使 RNN/LTC 类"状态即记忆"单元的状态搬运带宽降至 1/4–1/8（逐神经元混合位宽）
- **CPU 指令集加速**：AVX-512 / AVX-512VNNI、AVX2 FMA、AVX-VNNI、SSSE3、BMI1/BMI2、AES-NI；运行时 CPUID 检测自动启用、自动回退（无对应指令集的 CPU 安全降级）
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
grads = {name: p.grad for name, p in model.named_parameters()}  # 从 tape 收集梯度

# 4. 训练（优化器见 sgn.SGD/Adam，基础设施 B1，2026-08-29）
optimizer = sgn.SGD(model, lr=0.01, momentum=0.9)   # 支持 nn.Module 或 dict[str,ndarray]
...
optimizer.step(grads)                                # 显式梯度更新（非 PyTorch 无参 step）
```
> 优化器用法：`sgn.SGD(params, lr, momentum, weight_decay, classical)` / `sgn.Adam(...)`；
> `params` 可为 nn.Module（经 state_dict 读写）或 `dict[str, np.ndarray]`（训练脚本现状）；
> `step(grads)` 显式传入 {name: ndarray}（本项目 tape 每步重建 Tensor，梯度收集在训练脚本侧）；
> `classical=True` 用经典速度式 `v=m·v-lr·g; p+=v`（复现 resnet8 手写更新循环，默认 False 为 PyTorch 式）。
> 检查点：`sgn.save_checkpoint/load_checkpoint`（params+Optimizer 状态+rng，断点续训逐位可复现）；
> 实验框架：`sgn.run_seeds/aggregate/export_*`（多 seed 聚合 + CSV/JSON）。

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

### 服务器 AVX-512 加速（EPYC 远程实测，2026-08-31）

远程云主机（AMD EPYC 9Y24 / Zen 4，2 vCPU）实测 AVX-512 路径（K=65536，单核）：

| 原语 | Mops/s |
|------|--------|
| dot16（512 位 madd 快速路径） | 23,121 |
| dot8（512 位 VNNI） | 57,570 |

正确性 **238/238 全过**（512 位路径与标量 bit-exact，含满幅极值）；运行时后端自动选中
`avx512vnni`。完整方法/数据见 [SGN EPYC 速度测试归档](docs/SGN_EPYC速度测试归档_2026_08_31.md)。

### 本机 AVX-VNNI 实测 + CPUID 检测修复（Arrow Lake，2026-09-02）

原语基准 `sgn_benchmark` 收编进 simd 主构建（portable 版，CMake target）后，本机
（Core Ultra 5 225，无 AVX-512）首轮运行即暴露潜伏 bug：**AVX-VNNI CPUID 检测读错位**
（旧读 leaf7 sub0 ECX[4]=OSPKE，正确为 sub1 EAX[4]）——OSPKE=0 机器上 dot8/dot4 静默
落标量，EPYC/Linux 因 OSPKE=1 凑巧掩盖。修复后：

| 原语 | K | 修复前（标量回退） | 修复后（avxvnni） |
|------|---|------|------|
| dot8 | 4096 | 2,261 Mops/s | **93,516 Mops/s（~41×）** |
| dot8 | 65536 | 2,279 Mops/s | **48,757 Mops/s（~21×）** |

正确性 238/238 + UBSan 边界测试 + 强制标量回归全过。完整分析（含 EPYC 归档勘误、
消费端"入口决定瓶颈"定性）见 [SGN Arrow Lake 速度测试归档](docs/SGN_ArrowLake速度测试归档_2026_09_02.md)。

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

### 公平对比（同任务三方口径，2026-09-07 重构）

MLP 784-256-128-10 fwd+bwd、B=256、f32、同硬件（三方都有该功能，同口径中位）：

| 实现 | ms/iter | vs numpy |
|---|---|---|
| numpy 手写（无框架基线） | 2.342 | 1.00× |
| **SGN C++ autograd** | 4.774 | 2.04× |
| PyTorch（MKL，工业参照） | 1.561 | 0.67× |

> **诚实声明**：SGN **不主张**在 float 训练速度上超越 PyTorch——vendor BLAS
> 在该维度更快是架构差异。SGN 的价值在**整数/量化路径**（SIMD 整数点积、
> 嵌套量化一次量化多档取用、Level 调度、LeveledState 平面推理状态带宽
> 4.00–4.57×）与框架设施（autograd/checkpoint/实验框架）。复现脚本
> `engine/sgn/tests/bench_fair_compare.py`；口径详见白皮书 §3.7
> （旧口径 §3.1–3.6 引用须附口径说明）。

### 正确性

全量回归 **369 项（pytest）+ C++ 边界测试**通过，SIMD 内核与标量逐位一致（kBitExact，含大 K 分块域与满幅对抗），多轮复测无回归。

---

## 文档

- [SGN 用户使用手册](docs/SGN_用户使用手册.md) — 面向普通用户（无需 C++ 基础）
- [SGN 性能白皮书](docs/SGN_性能白皮书.md) — 性能基线、SIMD 优化与复现方法
- [SGN 开发者操作手册](docs/SGN_Autograd_用户操作手册.md) — 面向开发者 / 协作者
- [SGN EPYC 速度测试归档](docs/SGN_EPYC速度测试归档_2026_08_31.md) — AVX-512 服务器路径远程验证（正确性 238 项 + 性能基线）
- [MSint 多精度拆分计算范式](docs/msint_multisplit_paradigm/MSint多精度拆分计算范式.md) — 整数路径设计

---

## License

Apache-2.0

**第三方组件**：本项目分发的构建产物捆绑 LLVM Project 的 OpenMP 运行时 `libomp.dll`（Apache 2.0 with LLVM Exceptions）。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
