# SGN 统一数学框架

> **创建日期**: 2026-08-07
> **目的**: 整合分散的数学结论（EF/GEF/SR/MSint/Level-AMP/HC量化）为统一框架，阐明组件间关系、适用条件与工程优先级
> **状态**: 第一批（数学验证）— 综合集成
> **关联文档**:
>   - [SGN_Autograd_用户操作手册.md](SGN_Autograd_用户操作手册.md)
>   - [SGN_用户使用手册.md](SGN_用户使用手册.md)
>   - [DOCS_PRIORITY_CHAIN.md](DOCS_PRIORITY_CHAIN.md)
>   - 源文档: `内部档案` 下各数学报告

---

## 目录

- [SGN 统一数学框架](#sgn-统一数学框架)
  - [目录](#目录)
  - [一、概述](#一概述)
    - [1.1 为什么需要统一框架](#11-为什么需要统一框架)
    - [1.2 组件全景图](#12-组件全景图)
    - [1.3 符号系统](#13-符号系统)
  - [二、HC量化数学基础](#二hc量化数学基础)
    - [2.1 对称线性量化](#21-对称线性量化)
    - [2.2 步长与精度层级](#22-步长与精度层级)
    - [2.3 量化噪声](#23-量化噪声)
    - [2.4 确定性最近邻量化（round）](#24-确定性最近邻量化round)
    - [2.5 关键结论](#25-关键结论)
  - [三、SR随机量化](#三sr随机量化)
    - [3.1 定义](#31-定义)
    - [3.2 无偏性定理](#32-无偏性定理)
    - [3.3 方差分析](#33-方差分析)
    - [3.4 跨步平均化](#34-跨步平均化)
    - [3.5 clip敏感性](#35-clip敏感性)
    - [3.6 关键结论](#36-关键结论)
  - [四、GEF梯度误差补偿](#四gef梯度误差补偿)
    - [4.1 数学定义](#41-数学定义)
    - [4.2 误差分析](#42-误差分析)
    - [4.3 单步 vs 跨步澄清](#43-单步-vs-跨步澄清)
    - [4.4 GEF的限制](#44-gef的限制)
    - [4.5 关键结论](#45-关键结论)
  - [五、EF噪声整形理论](#五ef噪声整形理论)
    - [5.1 EF与delta-sigma调制同构](#51-ef与delta-sigma调制同构)
    - [5.2 N阶EF推广](#52-n阶ef推广)
    - [5.3 权重噪声传递函数](#53-权重噪声传递函数)
    - [5.4 分离残差EF](#54-分离残差ef)
    - [5.5 深层放大公式](#55-深层放大公式)
    - [5.6 临界间隙](#56-临界间隙)
    - [5.7 阻尼2阶EF](#57-阻尼2阶ef)
    - [5.8 实验验证状态](#58-实验验证状态)
    - [5.9 EF定位修正](#59-ef定位修正)
    - [5.10 关键结论](#510-关键结论)
  - [六、Level-AMP与M-凸性](#六level-amp与m-凸性)
    - [6.1 问题形式化](#61-问题形式化)
    - [6.2 M-凸性定理](#62-m-凸性定理)
    - [6.3 贪心边际分配算法](#63-贪心边际分配算法)
    - [6.4 Level-AMP精度分配策略](#64-level-amp精度分配策略)
    - [6.5 关键结论](#65-关键结论)
  - [七、MSint多精度拆分范式](#七msint多精度拆分范式)
    - [7.1 核心洞察](#71-核心洞察)
    - [7.2 多精度拆分点积](#72-多精度拆分点积)
    - [7.3 累加器溢出分析](#73-累加器溢出分析)
    - [7.4 两种融合模式](#74-两种融合模式)
    - [7.5 运行时精度变形](#75-运行时精度变形)
    - [7.6 关键结论](#76-关键结论)
  - [八、组件关系与工程优先级](#八组件关系与工程优先级)
    - [8.1 组件依赖关系图](#81-组件依赖关系图)
    - [8.2 策略推荐排序](#82-策略推荐排序)
    - [8.3 工程实现路线图](#83-工程实现路线图)
    - [8.4 未解决问题](#84-未解决问题)
  - [附录A：实验验证汇总](#附录a实验验证汇总)
  - [附录B：关键公式速查](#附录b关键公式速查)

---

## 一、概述

### 1.1 为什么需要统一框架

SGN 的反向传播方法经过多轮探索，形成了分散在不同文档中的数学结论。这些结论涉及：

- **HC量化**：基础整数存储格式
- **SR**：随机量化的无偏性
- **GEF**：梯度误差的单步补偿
- **EF**：跨步噪声整形的 delta-sigma 同构
- **Level-AMP**：逐层异构精度的 M-凸性最优性
- **MSint**：多精度拆分计算范式

各组件之间存在复杂的依赖关系和交互影响。本框架将这些分散结论整合为统一体系，明确：

1. 每个组件的数学定义与严格定理
2. 组件间的依赖关系与交互机制
3. 适用条件与已验证边界
4. 工程实现的优先级排序

### 1.2 组件全景图

```
                    ┌─────────────────────────────────────────────┐
                    │              SGN 统一数学框架                    │
                    └─────────────────────────────────────────────┘
                                      │
          ┌───────────────┬───────────┴───────────┬───────────────┐
          ▼               ▼                       ▼               ▼
    ┌──────────┐   ┌──────────┐           ┌──────────────┐   ┌──────────┐
    │ 基础层    │   │ 量化层   │           │ 补偿层        │   │ 调度层   │
    │ HC格式   │   │ SR/round │           │ GEF/EF       │   │ Level   │
    │ MSint   │   │          │           │              │   │ AMP     │
    └──────────┘   └──────────┘           └──────────────┘   └──────────┘
         │              │                       │                │
         │              ├── SR 无偏性 ──────────┤                │
         │              ├── SR 方差 ─────────── EF 噪声整形       │
         │              │                       │                │
         ├── MSint 拆分 ──┤                    GEF 误差补偿        │
         │              │                       │                │
         └── HC 步长 ────┘                      └──── M-凸性 ────┘
```

### 1.3 符号系统

| 符号 | 含义 | 定义位置 |
|------|------|---------|
| $\Delta_b$ | HC量化步长，$\Delta_b = m_f / (2^{b-1} - 1)$ | §2.1 |
| $Q_b^{\text{det}}$ | 确定性最近邻量化（round） | §2.4 |
| $Q_b^{\text{SR}}$ | 随机量化（Bernoulli SR） | §3.1 |
| $\eta_x$ | 量化噪声 $\eta_x = Q_b(x) - x$ | §2.3 |
| $u(x)$ | 分数部分 $u(x) = \text{frac}(x/\Delta_b)$ | §3.1 |
| $\varepsilon_{\text{GEF}}$ | GEF 误差 $\varepsilon_{\text{GEF}} = g_{dq} \cdot \varepsilon_w$ | §4.2 |
| $\text{NTF}$ | 噪声传递函数 $\text{NTF} = (1 - z^{-1})^N$ | §5.2 |
| $e_t$ | EF 残差 $e_t = \hat{g}_t - q_t$ | §5.1 |
| $\Delta_i(b)$ | 边际增益 $\Delta_i(b) = f_i(b) - f_i(b+1)$ | §6.2 |

---

## 二、HC量化数学基础

### 2.1 对称线性量化

**定义 2.1（对称线性量化步长）**. 设 $f \in \mathbb{R}^d$，记 $m_f := \max_i |f_i|$。对位宽 $b \in \{4, 8, 16\}$，定义对称量化步长

$$
\Delta_b(f) := \frac{m_f}{2^{b-1} - 1}.
$$

SGN 使用**对称**整数范围 $[-(2^{b-1} - 1), 2^{b-1} - 1]$，即丢弃 $-2^{b-1}$ 这一档以保持对称性。这导致 $\Delta_8 = m_f / 127$（非 $m_f / 128$），是后文 66568 vs 65536 数值差异的根源。

### 2.2 步长与精度层级

| 格式 | 位宽 $b$ | 步长 $\Delta_b$ | 整数范围 | 级数 | 对称性 |
|------|---------|----------------|---------|------|--------|
| HC4 | 4 | $m_f / 7$ | $[-7, 7]$ | 15 | 对称 |
| HC8 | 8 | $m_f / 127$ | $[-127, 127]$ | 255 | 对称 |
| HC16 | 16 | $m_f / 32767$ | $[-32767, 32767]$ | 65535 | 对称 |

**步长比**（关键常数）：

$$
\frac{\Delta_8}{\Delta_{16}} = \frac{32767}{127} \approx 258.0079, \qquad
\frac{\Delta_4}{\Delta_8} = \frac{127}{7} \approx 18.14, \qquad
\frac{\Delta_4}{\Delta_{16}} = \frac{32767}{7} \approx 4681.
$$

**方差比**（同分布、同 $m_f$、同 SR 形式下）：

$$
\frac{\text{Var}(\text{HC8})}{\text{Var}(\text{HC16})} = \left(\frac{\Delta_8}{\Delta_{16}}\right)^2
= \left(\frac{32767}{127}\right)^2 \approx 66568.06.
$$

### 2.3 量化噪声

**定义 2.2（量化噪声）**. 对量化函数 $Q_b$（可为 round 或 SR），量化噪声为

$$
\eta_x := Q_b(x) - x.
$$

对于确定性 round，$\eta_x$ 是确定性偏差（有偏）。
对于 SR，$\eta_x$ 是随机变量（无偏，见第三章）。

### 2.4 确定性最近邻量化（round）

GEF 所用的确定性量化：

$$
Q_b^{\text{det}}(x; f) := \text{clip}\!\left( \text{round}\!\left( \frac{x}{\Delta_b(f)} \right),\ -(2^{b-1} - 1),\ 2^{b-1} - 1 \right) \cdot \Delta_b(f).
$$

### 2.5 关键结论

| 结论 | 状态 |
|------|------|
| HC8/HC16是对称线性量化，范围对称 | ✅ 已确认 |
| 步长比 $\Delta_8/\Delta_{16} \approx 258$，方差比 $\approx 66568$ | ✅ 已确认 |
| 确定性 round 有偏，SR 无偏 | ✅ 已确认 |

---

## 三、SR随机量化

### 3.1 定义

**定义 3.1（Stochastic Rounding 量化）**. 令 $u(x) := \text{frac}(x/\Delta_b) = x/\Delta_b - \lfloor x/\Delta_b \rfloor \in [0, 1)$。定义

$$
Q_b^{\text{SR}}(x; f) := \text{clip}\!\left( \big(\lfloor x/\Delta_b \rfloor + B_x\big) \cdot \Delta_b,\ -(2^{b-1} - 1)\Delta_b,\ (2^{b-1} - 1)\Delta_b \right),
$$

其中 $B_x \sim \text{Bernoulli}(u(x))$ 为**对每个 $x$ 独立采样**的 Bernoulli 随机变量。

量化噪声（clip 不触发时）：

$$
\eta_x = (B_x - u(x)) \cdot \Delta_b.
$$

### 3.2 无偏性定理

**定理 3.1（SR 乘积估计量的严格无偏性）**. 设 $g, w \in \mathbb{R}^d$ 为任意确定性向量，$\tilde g_i := Q_b^{\text{SR}}(g_i)$，$\tilde w_i := Q_b^{\text{SR}}(w_i)$，且 $\{B_{g,i}\}$ 与 $\{B_{w,i}\}$ 条件独立。则

$$
\mathbb{E}\!\left[ \sum_{i=1}^d \tilde g_i \tilde w_i \,\Big|\, g, w \right] = \sum_{i=1}^d g_i w_i = g \cdot w.
$$

**前提条件**：
1. **条件独立性**：$\{B_{g,i}\}$ 与 $\{B_{w,i}\}$ 在给定 $(g, w)$ 下相互独立
2. **clip 不触发**：$|g_i| \leq m_g$，$|w_i| \leq m_w$（per-tensor per-step scale 保证）

**反例 3.1（共享 Bernoulli draw 破坏无偏性）**. 若使用同一个 Bernoulli 随机变量驱动 $g_i$ 与 $w_i$，则存在系统性正偏差 $\Delta_g \Delta_w \cdot u(1-u)$。

### 3.3 方差分析

**定理 3.2（SR 单步方差）**. 给定 $x$，clip 不触发时：

$$
\text{Var}[\eta_x \mid x] = u(x)(1 - u(x)) \cdot \Delta_b^2.
$$

- 最坏情形（$u = 1/2$）：$\text{Var}[\eta_x \mid x] = \Delta_b^2 / 4$
- 若 $u \sim \text{Uniform}[0, 1)$：$\mathbb{E}_u[\text{Var}[\eta_x \mid x]] = \Delta_b^2 / 6$

**重要澄清**：$\Delta^2/12$ 对应**加性均匀噪声 + 确定性 round**，非 Bernoulli SR 机制。Bernoulli SR 的方差是 $\Delta^2/6$，是前者的 **2 倍**。

### 3.4 跨步平均化

**定理 3.3（SR 跨步平均化）**. 在独立/相关/固定梯度场景下，SR 累积误差随步数 $T$ 以 $O(1/\sqrt{T})$ 衰减：

$$
\frac{1}{T} \sum_{t=1}^T \eta_t \xrightarrow{T \to \infty} 0, \qquad \text{Var}\!\left[ \frac{1}{T} \sum_{t=1}^T \eta_t \right] = O\!\left( \frac{\Delta^2}{6T} \right).
$$

### 3.5 clip敏感性

**实验结论**：clip 倍数 $\geq 4\sigma$ 时 clip 率 $< 0.01\%$，方差比 $\approx 1.00$。

| clip 倍数 | clip 率 | 方差影响 |
|-----------|---------|---------|
| $\geq 4\sigma$ | $< 0.01\%$ | 无影响 |
| $3\sigma$ | $\sim 0.1\%$ | 轻微 |
| $2\sigma$ | $\sim 4.5\%$ | 显著 |
| $1\sigma$ | $\sim 32\%$ | 严重 |

### 3.6 关键结论

| 结论 | 状态 | 来源 |
|------|------|------|
| SR 乘积估计量严格无偏（条件独立下） | ✅ 定理 1.1 证明 | 严格数学验证 |
| Bernoulli SR 方差 $= \Delta^2/6$ | ✅ 实验一 | 实验验证 |
| clip 倍数 $\geq 4\sigma$ 安全 | ✅ 实验三 | 实验验证 |
| SR 跨步平均化 $O(1/\sqrt{T})$ | ✅ 实验五 | 实验验证 |
| 共享 RNG 破坏无偏性 | ✅ 反例 1.1 | 严格数学验证 |
| $\Delta^2/6$（SR）与 $\Delta^2/12$（均匀噪声+round）口径区分 | ✅ 复核 2026-08-13（#4）| 严格数学验证 |

### 3.7 scale-outlier 敏感性（2026-09-06 审查补记）

per-tensor **max scale** 是对 outlier 最敏感的 scale 选择：单个元素决定全局步长
$\Delta = \max|g| / \text{max\_val}$。若 $\max|g|$ 是典型 $|g|$ 的 $R$ 倍，典型元素
只用到 $\sim\text{max\_val}/R$ 个网格档，量化噪声方差 $\propto \Delta^2$ 即放大
$R^2$（SNR 损失 $20\log_{10}R$ dB）。

且在现网实现下 `clip_bound = clip_sigma·scale·max_val = 4·max|g| > max|g|`，
**outlier 永远不会被 clip 抑制**——`clip_sigma` 在现网所有 SR/GEF 路径上是
惰性参数。§3.5 的 clip 敏感性表仅在改用非 per-tensor-max 的 scale（运行估计、
百分位 scale）后才开始适用。文档化依据：[SR算法审查与dot8溢出_2026_09_05.md]
(../内部档案) §1.3.4 与
[修补清单与预计改法_2026_09_06.md] P2-6。

---

## 四、GEF梯度误差补偿

### 4.1 数学定义

GEF（Gradient Error Feedback）在 Q16 网格整数反向路径上叠加补偿项（Q16 为自包含对称线性网格 ±32767，与 HC 库无关；历史名为 HC16），消除梯度量化误差。

**GEF 数学**：

$$
\begin{aligned}
g_{dq} &= \text{deq}(Q_{16}^{\text{det}}(g)) \quad &&\text{(梯度量化 + 反量化)} \\
w_{dq} &= \text{deq}(Q_{16}^{\text{det}}(w)) \quad &&\text{(权重量化 + 反量化)} \\
\text{grad}\_x^{\text{quant}} &= g_{dq} \cdot w_{dq} \quad &&\text{(量化路径 matmul)} \\
\varepsilon_g &= g - g_{dq} \quad &&\text{(梯度量化残差，float32)} \\
\text{grad}\_x^{\text{comp}} &= \varepsilon_g \cdot w_{\text{float}} \quad &&\text{(GEF 补偿项，float matmul)} \\
\text{grad}\_x &= g_{dq} \cdot w_{dq} + \varepsilon_g \cdot w_{\text{float}} \quad &&\text{(合并)}
\end{aligned}
$$

### 4.2 误差分析

**定理 4.1（GEF 误差）**. 相对于真值 $g \cdot w$，GEF 梯度估计的误差为：

$$
\varepsilon_{\text{GEF}} := \text{grad}_x - g \cdot w = g_{dq} \cdot (w_{dq} - w) = -g_{dq} \cdot \varepsilon_w,
$$

其中 $\varepsilon_w := w - w_{dq}$ 为 Q16 权重量化误差。

**相比无GEF的误差缩减**：

| 方案 | 误差项 | 3ep gap (ResNet-18) |
|------|--------|-------------------|
| 无 GEF（纯 Q16） | $g \times \varepsilon_w + \varepsilon_g \times w + \varepsilon_g \times \varepsilon_w$ | $\sim -1.60\%$ |
| 有 GEF | $g_{dq} \times \varepsilon_w$（仅权重量化残差） | $\sim -0.27\%$ |

**改善 6 倍**：GEF 将 $\varepsilon_g \cdot w$ 和 $\varepsilon_g \cdot \varepsilon_w$ 项消除，只保留 $g_{dq} \cdot \varepsilon_w$。

### 4.3 单步 vs 跨步澄清

| 特性 | SGN 实际 GEF | 假设性跨步 GEF |
|------|-------------|---------------|
| 残差使用 | 单步，不跨步累积 | 跨步累加 $\Sigma e_t$ |
| 残差管理 | 每次独立计算 $\varepsilon_g$ | 残差持久化 |
| 稳定性 | 稳定（单步） | 发散（已验证） |

**关键区分**：GEF 的单步补偿与 EF 的跨步闭环是完全不同的机制。GEF 不跨步，不存在累积发散风险。

### 4.4 GEF的限制

GEF 补偿项 $\varepsilon_g \cdot w_{\text{float}}$ 需要 **float32 权重与 float matmul**，与"全整数训练"目标冲突。这是 SR 方案（第三章）的动机——用无偏随机量化替代 GEF 的有偏确定性 round + 补偿。

### 4.5 关键结论

| 结论 | 状态 |
|------|------|
| GEF 误差降为 $g_{dq} \cdot \varepsilon_w$（仅权重残差） | ✅ 定理 4.1 |
| 3ep gap -0.27%（vs 无GEF -1.60%） | ✅ 实验验证 |
| GEF 是单步补偿，无跨步累积 | ✅ 已澄清 |
| GEF 需要 float 路径，与全整数目标冲突 | ✅ 已确认 |

---

## 五、EF噪声整形理论

### 5.1 EF与delta-sigma调制同构

**定理 5.1（EF = delta-sigma 调制器）**. 标准 EF 闭环更新方程与一阶 delta-sigma 调制器数学同构：

$$
\begin{aligned}
\hat{g}_t &= g_t + e_{t-1} \quad &&\text{(残差反馈)} \\
q_t &= Q(\hat{g}_t) = \hat{g}_t + \eta_t \quad &&\text{(量化)} \\
e_t &= \hat{g}_t - q_t = -\eta_t \quad &&\text{(新残差)}
\end{aligned}
$$

代入后得到：

$$
q_t = g_t + (\eta_t - \eta_{t-1}).
$$

**z 域传递函数**：

$$
\begin{aligned}
Q(z) &= G(z) + \eta(z) \cdot (1 - z^{-1}) \\
\text{STF} &= 1 \quad \text{(梯度信号无损通过)} \\
\text{NTF} &= (1 - z^{-1}) \quad \text{(一阶高通噪声整形)}
\end{aligned}
$$

$$
|\text{NTF}(e^{j\omega})|^2 = 4 \cdot \sin^2(\omega/2).
$$

- 低频 $(\omega \to 0)$：噪声被抑制到 0
- 高频 $(\omega = \pi)$：噪声被放大 4 倍

**梯度下降本身就是低通滤波器**——对多步梯度做加权平均。EF 把量化噪声推到高频，梯度下降把高频噪声滤掉。这是 telescoping 性质的频域解释。

### 5.2 N阶EF推广

N 阶 EF 使用 N 个残差缓冲区，反馈系数为二项式系数 $a_k = (-1)^{k+1} \cdot C(N, k)$：

$$
\text{NTF}_N = (1 - z^{-1})^N.
$$

| 阶数 | NTF | 低频抑制 | 高频增益 |
|------|-----|---------|---------|
| 1 阶 | $(1 - z^{-1})$ | $\omega^2 \to 12\text{ dB/oct}$ | $4\times$ |
| 2 阶 | $(1 - z^{-1})^2$ | $\omega^4 \to 24\text{ dB/oct}$ | $16\times$ |
| 3 阶 | $(1 - z^{-1})^3$ | $\omega^6 \to 36\text{ dB/oct}$ | $64\times$ |

### 5.3 权重噪声传递函数

SGD 更新 $w_{t+1} = w_t - \text{lr} \cdot q_t$ 的累积作用是 $1/(1 - z^{-1})$，因此：

$$
W_{\text{noise}}(z) = -\text{lr} \cdot \eta(z) \cdot \frac{\text{NTF}(z)}{1 - z^{-1}} = -\text{lr} \cdot \eta(z) \cdot (1 - z^{-1})^{N-1}.
$$

| EF 阶数 | 权重噪声 TF | DC 分量 |
|---------|------------|--------|
| 1 阶 | $(1 - z^{-1})^0 = 1$（白噪声） | 有限（$\sigma^2 / H^2$） |
| 2 阶 | $(1 - z^{-1})^1$（高通） | **零**（无系统性漂移） |
| 3 阶 | $(1 - z^{-1})^2$（更强高通） | **零**（更强抑制） |

**核心洞察**：2 阶 EF 消除权重的系统性漂移（DC 噪声），1 阶 EF 只能做到有界。

DC 功率实测：1 阶 $\to$ 2 阶 **DC 功率比 = 657740$\times$**。

### 5.4 分离残差EF

**NTF 的隐含前提**：量化噪声 $\eta$ 必须是白噪声（零均值、不相关）。

| 噪声来源 | $\eta$ 性质 | NTF 整形效果 | EF 收益 |
|---------|--------|------------|--------|
| **精度截断**（truncate） | 白噪声，零均值 | ✅ 有效（低频抑制） | ✅ 正收益 |
| **范围 clip** | 有偏，与信号相关 | ❌ 失效（低频放大） | ❌ 有害 |

**标准 EF 对 clip 噪声失效**：低频功率不降反增 **$\approx 9.0 \times 10^4 \times$（实测 90327$\times$）**（实验 4）。

**分离残差方案**：

$$
\begin{aligned}
\text{标准 EF：} &\quad e_t = \hat{g}_t - \text{clip}(\text{truncate}(\hat{g}_t)) \quad \text{— 包含 clip 误差（有害）} \\
\text{分离 EF：} &\quad e_t = \hat{g}_t - \text{truncate}(\hat{g}_t) \quad \text{— 只含 truncate 误差（白噪声）}
\end{aligned}
$$

**验证结果**：分离 EF 几乎完全消除 clip 的有害影响。标准 EF 在 clip 率 7-55% 区间严重有害（clamp=0.2 时 loss 暴增 40 倍），分离 EF 保持稳定。

### 5.5 深层放大公式

深层放大因子 = NTF 级联增益 $\times$ Jacobian$^L$：

$$
\text{NTF}_{\text{cascade}} = (1 - z^{-1})^L \quad (L \text{ 层级联}).
$$

| $J$ | $L=2$ | $L=3$ |
|-----|-------|-------|
| 0.5 | 0.59$\times$ | 0.53$\times$ |
| 1.0 | 2.34$\times$ | 4.22$\times$ |
| 1.5 | 5.27$\times$ | 14.25$\times$ |
| 2.0 | 9.36$\times$ | 33.78$\times$ |

**86$\times$ 放大的归因**（实验十五：L1 $\to$ L3 残差放大 86$\times$）：
- NTF 级联贡献：$\sim 4\times$（$L=2$）
- Jacobian 贡献：$\sim 22\times$（$J \approx 4.7$，轻度梯度爆炸）
- **结论：主要来自 Jacobian 范数 $> 1$，不是 NTF 级联**

### 5.6 临界间隙

EF 有临界间隙：$\Delta < 0.2$ 时 EF 有害，$\Delta > 0.2$ 时 EF 有益。

| $\Delta$ | benefit | 结论 |
|----------|---------|------|
| 0.001 | -0.000001 | 中性偏负 |
| 0.1 | -0.000025 | 负 |
| **0.2** | **+0.000039** | **临界点** |
| 0.5 | +0.000059 | 正 |
| 1.0 | +0.000215 | 正 |

### 5.7 阻尼2阶EF

标准 2 阶 NTF $= (1 - z^{-1})^2$ 在高频增益 $16\times$ 太大。引入阻尼参数 $\alpha$：

$$
\text{NTF}(z) = (1 - z^{-1})(1 - \alpha z^{-1}), \quad \alpha \in (0, 1).
$$

**训练验证**：$\alpha = 0$（等价于 1 阶 EF）是最优，$\alpha$ 增大单调退化。阻尼不起作用。

### 5.8 实验验证状态

| 方向 | 内容 | 状态 |
|------|------|------|
| **C** | NTF 频域验证（1/2/3 阶 ratio $\approx 1.0$） | ✅ 完成 |
| **C** | DC 漂移验证（1 阶 $\to$ 2 阶 657740$\times$ 抑制） | ✅ 完成 |
| **C** | 非凸训练收敛（1 阶 EF 超越 Full） | ✅ 完成 |
| **C** | 残差稳定性（1/2/3 阶全部有界） | ✅ 完成 |
| **A** | 精度间隙结构（截断 vs clip 区分） | ✅ 完成 |
| **A+** | 分离残差 EF 修复 clip 问题 | ✅ 完成 |
| **B** | 深层放大公式（NTF $\times$ Jacobian） | ✅ 完成 |
| **D** | 阻尼 2 阶 EF（$\alpha = 0$ 最优） | ✅ 完成 |
| **E** | 1 阶 EF 超越 Full 的机制（三重机制） | ✅ 完成 |
| **F** | int8 整数域 EF 验证 | ✅ 完成 |
| **G** | Full+整形噪声伪现象验证 | ✅ 完成（证伪） |
| **I** | EF+SR 在更难设置下重新验证 | ✅ 完成 |
| **H** | EF 在不同网络结构中的表现 | ✅ 完成 |

### 5.9 EF定位修正

经过方向 G、I 和 H 的验证，EF 的定位从"训练增强器"修正为"量化补偿器"：

| 原结论 | 修正后结论 | 依据 |
|--------|-----------|------|
| EF+SR 超越 Full precision | EF+SR 在更难设置下不再超越 Full（0.89$\times$） | 方向 I |
| Full+整形噪声 8000$\times$ 优于 Full | 伪现象，简单 teacher-student 产物 | 方向 G |
| EF 提供训练增强 | EF 在量化约束下可能减轻量化损失 | 综合 |
| NTF 噪声整形理论 | 数学结构仍然成立，训练收益不如预期 | 综合 |
| EF 在不同结构下表现一致 | 宽度极端（W=16/128）可超越 Full（1.09-1.14$\times$），中等宽度（W=32/64）不及 Full（0.88-0.91$\times$）；残差连接加剧退化（0.88$\times$$\to$0.79$\times$） | 方向 H |

### 5.10 关键结论

| 结论 | 状态 |
|------|------|
| NTF $= (1 - z^{-1})^N$ 噪声整形频域严格成立 | ✅ 已确认 |
| 2 阶 EF 消除 DC 漂移（657740$\times$ 抑制） | ✅ 已确认 |
| 分离残差 EF 修复 clip 发散问题 | ✅ 已确认 |
| 深层放大主要来自 Jacobian，非 NTF 级联 | ✅ 已确认 |
| 1 阶 EF 是最优选择（阻尼 $\alpha = 0$） | ✅ 已确认 |
| EF 有临界间隙 $\Delta > 0.2$ | ✅ 已确认 |
| EF 定位：量化补偿器，非训练增强器 | ✅ 修正后 |
| 分离残差是 int8 实现的前提条件 | ✅ 已确认 |
| **EF 在 int8 域残差发散（1037$\times$），分离 EF 保持稳定** | ✅ F-4 确认 |
| **EF 在不同网络结构下表现不一致（宽度 U 型、残差有害）** | ✅ H 实验确认 |
| **NTF 理论代数层（telescoping 有界性）无条件成立，统计层假设被破坏（见 §5.11）** | ✅ 2026-08-17 审计 |

### 5.11 EF 理论完备性审计（2026-08-17）

NTF 理论分两层：**代数层**（q-g = NTF(η)、Σ 有界，EF 定义恒等，无条件成立）与
**统计层**（η 白噪声 → 谱整形 + EF 有益，依赖隐含假设）。"数学成立但实践不成立"
全部来自统计层假设破坏。审计（[ef_theory_completeness_audit_2026_08_17.md]，
脚本 [validate_ef_theory_audit.py](../engine/sgn/tests/architecture/validate_ef_theory_audit.py)）：

| 假设 | 破坏证据 | 后果 |
|------|---------|------|
| H1 η 白噪声 | 确定性量化（round/truncate，HC 类）η 的 lag1 自相关 **+0.91**；SR 保持白 | NTF 谱整形精确性只对 SR 成立 |
| H3 g-η 不相关 | EF 瞬时 SNR 在 g→0 退化 **20~5.7e11x**（AUD-3b/7） | 有效梯度被噪声稀释；但权重上被 telescoping 抵消（凸问题 EF1 损失 = SR，+0.03%） |
| H4 纯 SGD | 动量使 EF 权重噪声放大 **2.6e10x**（AUD-5） | 精确 telescope 退化为温和有效 |
| H2 结构无关 | 非凸 MLP：无残差 EF 最优（超越 Full），**加残差后 EF 从最优变最差**（AUD-8，复现方向 H-3） | 收益被残差结构逆转 |

**结论**：EF 的 telescoping 有界性不可破（代数层），但"EF 有益"的论证不完备——它
未覆盖 H1-H4 的破坏条件。HC 确定性量化路径下 EF 只应被期待为"有界补偿器"（代数层
保证），不应期待谱整形精确性；残差网络中使用 EF 需谨慎（AUD-8 复现 H-3 逆转）。

**层级化修复方向（2026-08-19）**：残差破坏 EF 的根因是恒等路径主导 per-tensor scale
（主路径量化过粗）。层级化 EF（分组独立 scale + 残差，见 [EF层级化与空间编码改法储备.md](msint_multisplit_paradigm/EF层级化与空间编码改法储备.md) §八）
数学钉死（[validate_ef_hierarchical.py](../engine/sgn/tests/architecture/validate_ef_hierarchical.py)）：
① 分组后整体 telescope 有界性保持 O(1)（开放问题 1 解决）；② 跨组解耦——主路径组
不被恒等组主导（R=100 下整体 scale 放大 10.9x，分组 scale 免疫）；③ 残差网络训练
修复逆转 +14.1%。**残差网络的 EF 使用有修复路径（分组 scale，即 R2 方案），不需放弃 EF**。

---

## 六、Level-AMP与M-凸性

### 6.1 问题形式化

将 bits 分配形式化为可分离凸整数规划：

$$
\begin{aligned}
\min_{b_i} \quad & \sum_i f_i(b_i) = \sum_i \frac{c_i}{2^{b_i} - 1} \\
\text{s.t.} \quad & \sum_i b_i \leq B_{\text{total}} \quad \text{(存储预算)} \\
& b_{\min} \leq b_i \leq b_{\max}, \quad b_i \in \mathbb{Z}
\end{aligned}
$$

其中 $c_i$ 为第 $i$ 层的成本系数，$b_i$ 为位宽。

### 6.2 M-凸性定理

**定理 6.1（M-凸性）**. 成本函数 $f_i(b) = c_i / (2^b - 1)$ 满足 M-凸性，即边际增益 $\Delta_i(b) = f_i(b) - f_i(b+1)$ 严格递减。

**证明**：

$$
\Delta_i(b) = c_i \cdot \left[ \frac{1}{2^b - 1} - \frac{1}{2^{b+1} - 1} \right] = c_i \cdot \frac{2^b}{(2^b - 1)(2^{b+1} - 1)}.
$$

$$
\frac{\Delta_i(b)}{\Delta_i(b+1)} = \frac{4 \cdot 2^b - 1}{2 \cdot 2^b - 2} > 1 \quad \forall b \geq 1.
$$

极限 $b \to \infty$ 时，比率 $\to 2.0$。

**Ibaraki-Katoh 定理**：若 $f_i$ 满足 M-凸性，则贪心边际分配给出精确 IP 最优。

### 6.3 贪心边际分配算法

```
初始化: b_i = b_min, R = B - n · b_min
while R > 0:
    i* = argmax_i [f_i(b_i) - f_i(b_i+1)]  # 最大边际增益
    b_{i*} += 1; R -= 1
```

**复杂度**：$O((B - n \cdot b_{\min}) \cdot \log n)$

**验证结果**（exp34）：

| 方案 | 策略 | max gap |
|------|------|---------|
| 贪心 vs 穷举 | 5 层 $\times$ bits 4-16 $\times$ budget 40 | **0%**（完全匹配） |
| 贪心 vs 拉格朗日 | 10 层 ResNet-18 | **1.10%** |
| 贪心速度优势 | vs 穷举 | **475$\times$** |

### 6.4 Level-AMP精度分配策略

Level-AMP 将 M-凸性最优分配应用于逐层反向精度：

$$
\text{bits}_l = \begin{cases}
32\ (\text{float32}) & l \leq N/3 \quad \text{(深层，保证梯度流)} \\
16\ (\text{int16}) & N/3 < l \leq 2N/3 \quad \text{(中层)} \\
8\ (\text{int8}) & l > 2N/3 \quad \text{(浅层)}
\end{cases}
$$

**与 NVIDIA AMP 的对比**：

| 维度 | 传统 AMP | Level-AMP |
|------|---------|-----------|
| 切换依据 | tensor 范数（动态） | 层深度 + M-凸性（结构化） |
| 精度档位 | fp16/fp32（2 档） | int8/int16/fp32（多档） |
| 切换粒度 | 逐 tensor | 逐层 + 逐时刻 |
| 学习率联动 | ✗ | ✓（B1 sqrt lr） |
| 防抖 | ✗ | ✓（LEVEL_CHANGE_COOLDOWN） |

### 6.5 关键结论

| 结论 | 状态 |
|------|------|
| M-凸性严格成立（边际增益递减） | ✅ 定理 6.1 + exp34 |
| 贪心 = IP 最优（穷举一致，gap = 0%） | ✅ exp34 |
| 贪心 $\approx$ 拉格朗日（gap = 1.10%） | ✅ exp34 |
| 贪心速度优势 475$\times$ | ✅ exp34 |
| Level-AMP 已实现但未启用 | ⬜ HierarchicalScheduler 未激活 |

---

## 七、MSint多精度拆分范式

### 7.1 核心洞察

现有神经网络在**数据搬运层面都是 1:1 的**：搬运 1 个 weight $\to$ 参与 1 次乘法 $\to$ 产出 1 个 partial sum。

MSint 的整数拆分能力使得**一次搬运，多精度解释**成为可能：

```
搬运 1 个 int32 → 可拆分为：
  ├── int16_h（高位）→ 参与粗粒度计算
  ├── int16_l（低位）→ 参与细粒度修正
  ├── int8_hh/int8_hl/int8_lh/int8_ll → 更细粒度层次
```

**为什么只有 MSint 能做**：

| 特性 | 浮点网络 | 二值/INT8 量化 | MSint |
|------|---------|--------------|-------|
| 拆分有数学语义 | ❌ float32 $\to$ 2$\times$float16 无独立意义 | ❌ 只有一个粒度 | ✅ int32 $\to$ 2$\times$int16 严格可逆 |
| 拆分后可独立计算 | ❌ | ❌ | ✅ 各层点积独立 |
| 动态精度选择 | ❌ 全局固定 | ❌ 量化后固定 | ✅ 逐元素可拆分 |
| 一次搬运多精度产出 | ❌ | ❌ | ✅ |

**数学基础**：整数的位拆分是严格的 `value = high \* 2^p + low`，浮点数无此性质。

### 7.2 多精度拆分点积

一个 N 维点积，拆为 int16 高低位：

$$
\begin{aligned}
w_i &= w_{h,i} \cdot 2^{16} + w_{l,i} \\
x_i &= x_{h,i} \cdot 2^{16} + x_{l,i} \\
y &= \sum (w_{h,i} \cdot 2^{16} + w_{l,i}) \cdot (x_{h,i} \cdot 2^{16} + x_{l,i}) \\
  &= \underbrace{(\sum w_{h,i} \cdot x_{h,i})}_{\text{粗粒度}} \cdot 2^{32} + \underbrace{(\sum (w_{h,i} \cdot x_{l,i} + w_{l,i} \cdot x_{h,i}))}_{\text{交叉修正}} \cdot 2^{16} + \underbrace{\sum w_{l,i} \cdot x_{l,i}}_{\text{细粒度}}
\end{aligned}
$$

**一次数据搬运，产出 3 层不同粒度的计算结果**。

**工程实现（SIMD 窄路径，2026-08-14）**：`dot_split` / `dot_split_leveled` 组内点积已按方案 A 落地
三档窄精度打包 SIMD（16 位 AVX2 `mul_epi32`、8 位 AVX-VNNI `dpbusd`、4 位 nibble 预解包 + `dpbusd`），
`split_parts` 直写窄数组（NarrowParts）+ narrow_dot 微优化（偏置修正项对角线提升、K 尾部零填充满宽
SIMD）。每点积实测 16/8/4 位约 **8.9–9.3× / 26.9–32.3× / 72.2–95.2×**，总体 vs 标量约
**9.2–10.5× / 33.2–34.2× / 88.6–98.0×**，全部 bit-exact 与标量一致（详见
[dot_split_simd_optimization_plan_2026_08_13.md]
与 [dot_split_simd_summary_report_2026_08_14.md]）。

**新增能力（2026-08-14，窄路径 SIMD 之上）**：
- **低位对角裁剪**：`trim_high_diag` 参数裁剪高位对角点积（4 位档 n=8 省 43.75% ALU，int32 截断 bit-exact），消费端入口 `dot_fused_i32`；摊销接口 `narrow_dot_prepared4` 上兑现 -44.65%。
- **摊销热路径**：`prepare_nibble` + `narrow_dot_prepared4` / `matmul_prepared4`——同一 x 预解包一次、M 行复用 n² 点积。
- **降档预解包融合**：`prepare_downcast` / `prepare_downcast_np`（C++ 内部分组 + keep_top 截断 + `prepare_nibble_from_raw`，numpy 零拷贝 + 数组分组），prepare 3.08ms → 0.36ms（累计 8.6x）。
- **降档决策**：`select_precision`（按 importance 逐元素选 p ∈ {8,16,32}）+ `dot_split_leveled_downcast` / `dot_fused_leveled_downcast`（keep_top + 4 位摊销点积）。n² 计算量随精度位数平方下降——int8 档端到端反超 numpy 2.1~2.4x、int16 档临界持平、int32 档仍慢（设计根因，见 [dot_split_leveled_grouping_heap_overhead_2026_08_14.md] §7.7/§7.8）。

### 7.3 累加器溢出分析

| 子点积 | 操作数范围 | 单次乘积范围 | K 维累加需求 | 累加器类型 |
|--------|-----------|------------|-------------|-----------|
| 粗粒度 $\sum w_h \cdot x_h$ | int16 $\times$ int16 | $[-2^{31}, 2^{31})$ | $K \times 2^{31}$ | **int64** |
| 交叉项 | int16 $\times$ int16 | $[-2^{31}, 2^{31})$ | $2K \times 2^{31}$ | **int64** |
| 细粒度 $\sum w_l \cdot x_l$ | int16 $\times$ int16 | $[-2^{31}, 2^{31})$ | $K \times 2^{31}$ | **int64** |

**关键约束**：$K > 2$ 时 int32 累加器溢出。最终融合结果需 96 位以上精度（int128 或大整数）。

### 7.4 两种融合模式

**单融合**（最终合并为一个结果）：

$$
y = \text{coarse} \cdot 2^{32} + \text{cross} \cdot 2^{16} + \text{fine}.
$$

**多输出模式**（各子点积独立输出，不融合）：
- 每个输出只需 int64 累加器
- 不需要 96 位融合
- 后处理可选择不同粒度组合

### 7.5 运行时精度变形

**定义 7.1（运行精度变形 - Runtime Precision Morphing）**. MSint 的 $v[0] \sim v[5]$ 是同一寄存器的不同解释视角，精度切换是零内存开销的视角旋转。

**增量计算结构**：

$$
\begin{aligned}
\text{只用 } v[0] &: y_0 = Q(x) \cdot (v[0] \cdot \text{scale}[0]) \\
\text{展开 } v[0]+v[1] &: y_1 = y_0 + Q(x) \cdot (v[1] \cdot \text{scale}[1]) \\
\text{展开 } v[0..D] &: y_D = y_{D-1} + Q(x) \cdot (v[D] \cdot \text{scale}[D])
\end{aligned}
$$

**与 EF 的协同**：MSint 多输出的精度选择可视为空间域的 Level-AMP，而 EF 是时间域的噪声整形。两者在数学上正交，可在工程上独立实现。

### 7.6 关键结论

| 结论 | 状态 |
|------|------|
| MSint 拆分严格可逆 $value = \text{high} \cdot 2^p + \text{low}$ | ✅ 数学恒等式 |
| 多精度拆分点积数学等价于原始点积 | ✅ 展开验证 |
| 累加器需 int64（$K > 2$ 时 int32 溢出） | ✅ 溢出分析 |
| 运行时精度变形零内存开销 | ✅ 已验证 |
| **int8 对（h,l）梯度载体数学可行，消费端 dot8 零粘合损耗（fine 2 次/coarse 1 次 dot8，bit-exact）** | ✅ #30/#31 验证（2026-08-31/09-02） |
| 前向 int8 + 反向 int16 异构精度 | ⬜ 待实现 |
| MSint 多精度范式落地 | ⬜ 待反向传播收敛后启动 |

---

## 八、组件关系与工程优先级

### 8.1 组件依赖关系图

```
HC量化（基础存储格式）
  ├── SR（随机量化，替换确定性 round）
  │     ├── 需要 HC 步长定义
  │     └── 需要独立 RNG 流
  ├── GEF（误差补偿）
  │     ├── 需要 HC16 量化路径
  │     └── 需要 float 补偿路径（与全整数目标冲突）
  ├── EF（跨步噪声整形）
  │     ├── 需要量化器（2026-08-17 审计后首选 SR；round 仅作"有界补偿器"）
  │     ├── 需要残差缓冲区
  │     └── 分离残差需要 truncate 分离
  ├── Level-AMP（逐层精度分配）
  │     ├── 需要 M-凸性定理（已证明）
  │     └── 需要 HierarchicalScheduler（已实现未启用）
  └── MSint（多精度拆分）
        ├── 需要 MSint 存储格式（已实现）
        └── 需要反向视角注册（未实现）
```

### 8.2 策略推荐排序（2026-08-17 更新）

基于 [EF 理论完备性审计] 重新评估，HC 确定性量化
（round/truncate）作为反向梯度量化器的可行性整体下调（噪声非白 +0.91 + 有偏 -Δ/2），
**SR（无偏白噪声）被确认为反向路径最优先选**。

| 排名 | 方案 | 收益 | 复杂度 | 推荐度 | 定位 |
|------|------|------|--------|-------|------|
| 1 | **A1 (HC8+SR 去 GEF)** | 去 float 路径，改动最小 | 低 | ★★★★★ | **主推方向**——SR 白+无偏，被审计强化 |
| 2 | **H (整数累积+SR)** | $\sqrt{N}$ 方差缩减，精确累加 | 低 | ★★★★★ | 前向/反向均 SR，无确定性量化 |
| 3 | **F (EF-SGD + SR)** | 累积误差有界 | 低 | ★★★★★ | EF 量化器首选 SR（白噪声，谱整形有效） |
| 4 | **C1 (前向 HC8+反向 Q16+SR)** | 1.2-1.33$\times$ 速度 | 中 | ★★★★★ | 反向 Q16 用 SR 量化器，非确定性 round |
| 5 | **C3 (GEF $\to$ MSint 视角)** | 去 float 路径 | 中 | ★★★★ | 依赖 MSint 反向视角落地 |
| 6 | **STE (标准)** | 基准对比 | 低 | ★★★ | 无变动 |
| 7 | **GEF (当前实现)** | 单步误差补偿 | 低 | ★★★ | ⬇️ 下调，仅作对比基线（round 偏置+非白） |
| 8 | **EF+确定性量化** | 有界补偿器（无谱整形） | 低 | ★★★ | 新条目：telescoping 有界性成立，但无谱整形收益 |
| 9 | **Level-AMP** | 逐层异构精度 | 高 | ★★★ | 无变动 |
| 10 | **Q16 无 GEF**（历史名 HC16） | 完全整数路径 | 低 | ★★ | ⬇️ 最低，非训练主路径 |
| 11 | **float32 baseline** | 精度参考 | 低 | ★★★ | 无变动 |

**关键变化**：
- **A1 (HC8+SR)** 保持排名 1 并标记为"主推方向"，审计补齐了理论依据（此前只有实验六经验排序）
- **GEF** 从 ★★★★ 下调至 ★★★，定位从"当前实现"降为"对比基线"
- **Q16 无 GEF**（历史名 HC16）从 ★★★ 下调至 ★★，明确标注非训练主路径
- 新增 **EF+确定性量化** 条目，定位收紧为"有界补偿器"（无谱整形收益）
- 所有涉及反向量化器的方案，SR 标记为 **首选量化器**（C1 的反向 Q16 需换 SR 而非 round）

### 8.3 工程实现路线图（2026-08-17 更新：明确 A1+SR 主推）

> **推荐路径**：**A1（HC8+SR 去 GEF）为主推**——前向/存储用 HC 量化，反向梯度用 SR
> （无偏白噪声），去 float 路径、改动最小。EF（若启用）量化器首选 SR。GEF 保留为
> 对比基线，不主推。

**阶段 1 — 短期（C++ Autograd 算子实现）**：
1. FLOAT32 反向传播 ✅ 已实现
2. STE 反向传播 ✅ 已实现
3. conv2d+relu 算子融合 ✅ 已实现
4. Q16 整数反向 + GEF（C++ 层，历史名 HC16）—— ⬇️ 已实现，**降为对比基线**（round 偏置+非白，审计下调）
5. **SR 随机量化（C++ 层，RNG 实现）**—— ✅ 已实现，**A1 主推核心量化器**

**阶段 2 — 中期（跨步补偿与融合，量化器一律选 SR）**：
6. EF-SGD 跨步残差闭环（量化器首选 **SR**；确定性 round 仅作"有界补偿器"定位）
7. 分离残差实现（int8 前提条件，truncate 分离保留）
8. MSint backward_int16 视角注册
9. Level-AMP 启用（HierarchicalScheduler）

**阶段 3 — 长期（彻底 SGN 化）**：
10. MSint 多精度拆分点积实现
11. 运行时精度变形
12. ✅ 内存池分配（2026-08-16 已实现：size-class 分桶 + thread_local 无锁 free-list，`engine/sgn/common/pool_allocator.h`）

**落地顺序建议**：A1（HC8+SR）→ 6 层 CNN 对比验证（vs FLOAT32/STE/GEF）→ EF-SR（若需跨步）→
ResNet-18 对比。残差网络若用 EF 需谨慎（AUD-8 复现 H-3 逆转），建议直接评估 A1/SR。

### 8.4 未解决问题

| 问题 | 影响 | 优先级 |
|------|------|--------|
| SR 在 C++ 层的 RNG 实现（性能 vs 质量权衡） | 阻碍 SR 落地 | 高 |
| GEF + Conv2d 的算子融合 | 阻碍 GEF 落地 | 高 |
| 各方案在 6 层 CNN 上的实际精度对比 | 验证阶段 | 高 |
| 各方案在 ResNet-18 上的实际精度对比 | 验证阶段 | 高 |
| MSint backward_int16 视角在 C++ 层的注册 | 阻碍 MSint 落地 | 中 |
| EF-SGD 跨步残差缓冲区生命周期管理 | 阻碍 EF 落地 | 中 |
| Level-AMP 的 HierarchicalScheduler 启用 | 阻碍 Level-AMP 落地 | 中 |
| 内存池分配 | ✅ 已完成（2026-08-16，`common/pool_allocator.h`，默认关闭、`set_pool_allocator(True)` 启用） | 低 |
| int8 对梯度载体引擎侧落地（反向末端"写 pair 而非写回 float32"可选路径 + `StrategyContext` 开关） | 数学层 + 引擎层双就绪（#30/#31/#32），工程未启动；落地约束：C++ 内部闭环（Python list 入口被拆分打包主导，见 #32） | 中 |
| ~~int8 对载体引擎侧吞吐基准（需构建 .pyd 后实测 dot8 消费路径）~~ | ✅ 已完成（2026-09-02，#32）：CPUID 潜伏 bug 修复后 dot8 原语 ~30×；端到端"入口决定瓶颈"定性 | 中 |

---

## 附录A：实验验证汇总

### A.1 数学验证实验

| 实验 | 脚本 | 验证内容 | 结论 |
|------|------|---------|------|
| 实验一 | validate_sr_variance.py | SR 单步方差 | Bernoulli SR 方差 $= \Delta^2/6$ 严格成立 |
| 实验二 | validate_gef_accumulation.py | GEF 跨步累积 | Proper EF 有界，Naive 累加发散 |
| 实验三 | validate_clip_sensitivity.py | clip 阈值 | clip $\geq 4\sigma$ 安全 |
| 实验四 | validate_ef_nonconvex.py | EF 非凸 | EF telescoping 非凸成立 |
| 实验五 | validate_sr_averaging.py | SR 跨步平均化 | SR 平均化 $O(1/\sqrt{T})$ 成立 |
| 实验六 | validate_sr_vs_gef_training.py | SR vs GEF 训练 | Full $>$ SR $\approx$ EF+SR $>$ GEF |

### A.2 EF 方向验证

| 方向 | 脚本 | 验证内容 | 结论 |
|------|------|---------|------|
| C | validate_ef_noise_shaping.py | NTF 频域、DC 漂移、非凸训练 | ✅ 全部成立 |
| A | validate_ef_precision_gap.py | 精度间隙结构、临界间隙 | ✅ 成立 |
| A+ | validate_ef_separated_and_cascade.py | 分离残差、深层放大 | ✅ 成立 |
| B | validate_ef_separated_and_cascade.py | 级联 NTF + Jacobian 调制 | ✅ 成立 |
| D | validate_ef_damped_and_noise_benefit.py | 阻尼 2 阶 EF | ✅ $\alpha=0$ 最优 |
| E | validate_ef_damped_and_noise_benefit.py | 三重机制分析 | ✅ 成立 |
| F | validate_f_int8_ef.py | int8 整数域 EF | ✅ 分离残差修复 |
| G | validate_g1_shaped_noise_verification.py | 整形噪声伪现象 | ❌ 证伪 |
| I | validate_i_ef_sr_hard_settings.py | EF+SR 更难设置 | ⚠️ 不再超越 Full |
| **H** | validate_h_ef_network_structures.py | EF 在不同网络结构中的表现 | ⚠️ 结构依赖，宽度极端可超越 Full |
| 严格验证 #1–#21 | validate_math_stage1~6_*.py | 六阶段数学复核（SR 统计/位精确/精度分配/NTF/梯度恒等式/网络性质） | ✅ 全通过（详见表 A.4） |
| 严格验证 #22–#28 | validate_math_msint_*.py / validate_bench_msint_simd_baseline.py | MSint 多精度拆分/解释/逐元素选择/异构组合/H3 带宽/SIMD 优化基线复核（多精度点积等价/1:N 解释可逆/信息分离 H2/逐元素精度选择/LeveledSplitDot 组合落地/H3 带宽性能基准/SIMD 优化前 baseline+适用性分析） | ✅ 全通过（详见表 A.4） |
| 严格验证 #29 | validate_decode_overhead_vs_memory_bandwidth.py | 解码有效开销真实含义复核（实测内存搬运带宽 vs 解码计算速度，9.78÷30≈0.33 推导链与反证，消除"0.34× 是内存搬运速度"误解） | ✅ 已建立（实测，详见表 A.4） |

**H 方向详细结果**（2026-08-07）：

| 子实验 | 条件 | EF/Full | EF/SR | 结论 |
|--------|------|---------|-------|------|
| H-1 深度扫描 | D=2 | 0.94× | 0.91× | EF 在各深度均未超越 Full |
| | D=3 | 0.91× | 0.92× | |
| | D=4 | 0.88× | 0.92× | |
| | D=5 | 0.91× | 1.01× | |
| | D=6 | 0.93× | 1.00× | |
| | **趋势** | 0.88-0.94× | | 非单调（无观测到的深层放大） |
| H-2 激活函数 | ReLU | 0.98× | 1.02× | 差异不显著 |
| | Tanh | 0.99× | 0.96× | |
| H-3 残差连接 | 无残差 | 0.88× | 0.92× | 残差连接加剧 EF 退化 |
| | 有残差 | 0.79× | 0.93× | -10.3% |
| H-4 宽度对比 | W=16 | 1.14× | 1.06× | U 型模式：极端宽度可超越 Full |
| | W=32 | 0.91× | 1.02× | |
| | W=64 | 0.88× | 0.92× | |
| | W=128 | 1.09× | 1.08× | |

**H 方向关键结论**：
- EF/Full ratio 范围 0.79-1.14×，平均 0.95×，维持"量化补偿器"定位
- 宽度极端（W=16/128）EF 可超越 Full，中等宽度（W=32/64）不及 Full —— U 型模式
- 残差连接加剧 EF 退化（-10.3%），与"残差缓解深层放大"的直觉相反
- 深层放大在 2-6 层范围内非单调，与 Jacobian 主导的深层放大理论（§5.5）不矛盾（该理论在更深/更宽网络才显著）
- 激活函数（ReLU vs Tanh）对 EF 行为影响不显著

### A.3 Level-AMP 验证

| 实验 | 验证内容 | 结论 |
|------|---------|------|
| exp34 | 贪心 = IP 最优（M-凸性） | ✅ 成立（gap = 0%） |
| exp34 | 贪心 $\approx$ 拉格朗日 | ✅ 成立（gap = 1.10%） |
| exp34 | 速度优势 475$\times$ | ✅ 成立 |
| exp35 | Fisher 度量本质缺陷 | ⚠️ 部分成立 |
| exp36 | 图正则化平滑 bits 分配 | ✅ 成立 |

### A.4 严格数学验证（2026-08-13）

> 一次完整的纯数学复核（#1–#28），过程记录见 `内部档案`。方法：T/S/E 分类 + 正推/倒推/更大规模三通道 + 多 seed + C++ 交叉。

| 阶段 | 验证项 | 结论 | 备注 |
|------|--------|------|------|
| 1 | SR 方差 $\Delta^2/6$、无偏性、clip 含 DC、Var∝Δ² | ✅ 证实 | #1–#3 + 补充 |
| 1 | $\Delta^2/12$ vs $\Delta^2/6$ 区分 | ⚠️ 修正 | **#4 抓错**：均匀抖动+round 实为 Δ²/6（非 Δ²/12），纠正 `validate_sr_variance.py` 结论#4；框架 §3 正文本就正确 |
| 2 | int12/16/32 位拆分可逆、backward_int16≡int8×2、scale 公式 | ✅ 证实（bit-exact）| #5–#8，含 C++ 交叉 |
| 3 | 贪心=全局最优（gap=0）、成本含 in_dim、误差≈1/(2^b−1) | ✅ 证实 | #9–#12，C++ 在 1 ulp 内 |
| 4 | EF≡delta-sigma、低频抑制、clip 噪声放大、分离残差修复 | ✅ 证实 | #13–#16 |
| 5 | 残差 skip 恒等梯度、÷size、CE 梯度 | ✅ 证实（机器精度）| #17–#19 |
| 6 | 超度量性违反率 100% | ✅ 复现证伪 | **#20**：定理 7.4 复核通过，旧版"梯度超度量性"假设被最终钉死；框架正文未引用该假设 |
| 6 | 深层残差放大 | ✅ 证实（结构性）| **#21**：相对残差随深度指数放大（每层 8.8×、R²=0.75）+ Jacobian 梯度放大>1；86× 为配置相关数值 |
| 7 | 多精度拆分点积数值等价 | ✅ 证实（bit-exact）| **#22**：fused==Σw·x（5 seed×K）+ 边界值；多输出→单融合可逆（fuse_128 精确还原，128 位符号扩展重建）；C++ SplitDot 与 Python 参考一致 |
| 7 | 1:N 多精度解释每层可逆 | ✅ 证实（bit-exact）| **#23**：int32 同时产出 {16,8,4} 三层次，5009 值（含边界/负数）每层精确可逆；批量 200× 亦可逆（MultiScaleView） |
| 7 | 多输出层信息分离（H2）| ✅ 证实 | **#24**：位带依赖——low-only→fine 不变且 cross=coarse=0、high-only→coarse 不变且 fine=cross=0（bit-exact，无跨带泄漏）；尺度分离——coarse·2³²/fine 贡献中位 6.29e7、/cross·2¹⁶ 中位 2.16e4 → 各层占据分离幅度带 |
| 7 | Level 逐元素精度选择（H4）| ✅ 证实（bit-exact）| **#25**：select 阈值语义（<100→16、100~999→8、>=1000→4）；300 元素按重要性逐元素选粒度并精确重建（PrecisionSelector + MultiScaleView 组合） |
| 7 | 异构粒度逐元素拆分点积（组合落地 H1+H4）| ✅ 证实（bit-exact）| **#26**：LeveledSplitDot——每元素按重要性异构选粒度（低=2/中=4/高=8 parts/元素），`dot_fused_leveled==原始点积` bit-exact（150 组）；分组 1:N 多输出（各组 partials 长 2n-1，fuse_128 累加==原始）；全粗/全细/异构三策略均精确等价（按需分配不损失精度）；C++/Python 200 组一致 |
| 7 | H3 带宽加速（性能基准）| ✅ 证实（带边界）| **#27**：两个正交加速来源——①按需位宽（H4×H3）：字节节省 2.0-3.3×、实测耗时加速 2.69×（但有效带宽≈0.94×，省时不省信息密度，以按需降次要精度为代价）；②1:N 多输出复用（H3 核心）：一次搬运产出 3/7/15 层（粒度 16/8/4）、实测 3.55×（vs 传统逐层搬运，省重复搬运） |
| 7 | SIMD 优化前 baseline + 适用性分析（工程基线）| ✅ 建立基准 | **#28**（validate_bench_msint_simd_baseline.py）：固化未优化标量基线——K=1024 异构 fused=0.285ms、异构多输出=0.294ms；异构/标量参照 K=256 1.86×→16384 3.04×。SIMD 适用性：**dot_split 组内点积=唯一优化焦点**（✅ 最适用，**已实施 2026-08-14 方案 A 三档窄路径**：16 位 AVX2 `mul_epi32` 每点积 8.9–9.3× + 8 位 AVX-VNNI `dpbusd` 26.9–32.3× + 4 位 nibble 预解包 `unpack_nibble_u/s`+`dpbusd` 72.2–95.2×，配合 `split_parts` 直写窄数组 NarrowParts 与 narrow_dot 微优化（corr 对角线提升 + K 尾部零填充满宽 SIMD，4 位再提速 1.39–1.94×）后总体 vs 标量 9.2–10.5×/33.2–34.2×/88.6–98.0×，bit-exact 均与标量一致），split_parts=⚠️ 视分组、fused 128 位=⚠️ 收益小、select_levels(决策层)/分组(控制层)=❌ 不做；硬约束：SIMD 必须编译时宏保留非 x86 标量回退 |
| 7 | 解码有效开销含义（工程基线，消除误解）| ✅ 建立基准 | **#29**（validate_decode_overhead_vs_memory_bandwidth.py）：实测**真实内存搬运带宽**约 12 GB/s（每字节 0.08 ns，纳秒级）；C++ 批量解码每元素约 0.3 ns（已优化到接近带宽极限）；有效开销 = 9.78（exp18 numpy 位操作相对 HC16 慢 9.78×）÷ ~30（C++ 加速比）≈ 0.31×（目标 ≤ 0.33×）。**反证**：0.34× 是无量纲相对比值，其分母是"HC16 解码耗时"（另一条计算路径），从未与"内存搬运时间"做过比值 → 它绝非内存搬运速度/空转成本，无法换算成任何内存带宽数值 |
| 7 | int8 对（h,l）作梯度载体数学探索（MSint×Level）| ✅ 证实（双库 6/6 ×3）| **#30**（validate_math_grad_int8_pair{,_alt,_dual}.py，2026-08-31）：Q16 网格梯度以 int8 对存储（u=256h+l）——T1 鸽笼：对称肢 255²=65025<65535，S 方案半宽须取 32639=127·257（scale 损失 0.39%）；T2 往返全域穷举 bit-exact；T3 Q8 嵌套（电平 257k↔对角肢 (k,k)，双解释零额外误差）；S1 粗解码必须配 SR（round 小梯度段相对误差 99.6% 清零，SR 严格无偏）；S2 方差标度 65536=256²；E1 粗解码 SNR≈Q8-SR（差<0.1 dB）。U 方案（无符号低字节 l′=u mod 256）消解鸽笼：满格 ±32767 零缺口 + vpdpbusd(u8×i8) 原生匹配；取舍=视图同一性（S）vs 满格精度/字节对齐（U），两方案并存。报告 [msint_int8_pair_grad_carrier_2026_08_31.md] |
| 7 | int8 对消费端 dot8（VNNI）路径点积验证 | ✅ 证实（双库 7/7）| **#31**（validate_math_grad_int8_pair_dot8.py，2026-09-02）：载体经 simd::dot8（avxvnni.cpp vpdpbusd）消费零粘合损耗——V1 通道模拟（4 累加器×8 int32 lane+零填充尾）== int64 标量参考全档 bit-exact（168 例）；V2/V3 fine/coarse 偏置法分解恒等式 bit-exact（fine=2 次 dot8+O(1) 修正、coarse=1 次，Sw 摊销）；V4 Q8 对角嵌套点积；V5 对抗溢出界 K≈530463（K=65536 余量 ≥8×）；V6 点积级 SR 无偏+方差比 65536=256²；V7 pair-fine≡Q16-SR（dev=0）、pair-coarse≈Q8-SR（0.32 dB<3σ 0.45）。见上报告 §七 |
| 7 | int8 对载体引擎侧吞吐基准（.pyd 实测 + CPUID 潜伏 bug 修复）| ✅ 完成（修复前后双轮）| **#32**（bench_grad_int8_pair_dot8.py + simd/sgn_benchmark.cpp 收编，2026-09-02，Arrow Lake Ultra 5 225 / Clang 22.1.8 / avxvnni）：①基准期间暴露 simd_dispatch AVX-VNNI CPUID 双重错位（旧读 sub0 ECX[4]=OSPKE 位，应为 sub1 EAX[4]）——OSPKE=0 机器 dot8/dot4 静默落标量，EPYC/Linux 因 OSPKE=1 侥幸误判掩盖（单平台遮蔽，与 batch_get_all SSSE3 同型）；修复后 dot8 原语 2930→93516 Mops/s（~30×），238/238 bit-exact + UBSan 边界全过。②端到端定性**入口决定瓶颈**：Python list 入口被 per-element 拆分打包主导（K=16384 端到端 650µs 中 dot8 仅 0.75µs），pair-fine 省一半 dot8 收益仅在该入口 <0.1%；C++ 内部预打包消费路径（narrow_dot 直调，x 侧跨 M 行摊销）才完全兑现 §七 成本模型。落地约束：pair 路径须 C++ 内部闭环。见上报告 §八 |

**方法论注记**：
- **超大 N 统计显著性伪影**（#1/#2）：N≥10⁶ 时 z/t 统计量显著（z=-7.73、t=-2.66）但实际偏差 <0.1% 且随 N 收敛——判据应看相对量级而非 p 值。
- **#21 的 86× 未精确复现**（793× 几何均）：经验标度类结论的精确数值对网络结构/尺度敏感，86× 应视为结构性结论（深层残差放大 + Jacobian>1）而非普适常数。

---

## 附录B：关键公式速查

### B.1 HC量化

| 公式 | 表达式 |
|------|--------|
| 步长 | $\Delta_b = m_f / (2^{b-1} - 1)$ |
| 量化 | $Q_b^{\text{det}}(x) = \text{clip}(\text{round}(x/\Delta_b), -2^{b-1}+1, 2^{b-1}-1) \cdot \Delta_b$ |
| 步长比 $8/16$ | $\Delta_8 / \Delta_{16} = 32767/127 \approx 258$ |
| 方差比 $8/16$ | $\text{Var}_8 / \text{Var}_{16} = (32767/127)^2 \approx 66568$ |

### B.2 SR

| 公式 | 表达式 |
|------|--------|
| SR 量化 | $Q_b^{\text{SR}}(x) = (\lfloor x/\Delta_b \rfloor + B_x) \cdot \Delta_b$，$B_x \sim \text{Bernoulli}(u(x))$ |
| 噪声 | $\eta_x = (B_x - u(x)) \cdot \Delta_b$ |
| 条件期望 | $\mathbb{E}[\eta_x \mid x] = 0$ |
| 条件方差 | $\text{Var}[\eta_x \mid x] = u(1-u) \cdot \Delta_b^2$ |
| 平均方差 | $\mathbb{E}_u[\text{Var}] = \Delta_b^2 / 6$ |
| 无偏性 | $\mathbb{E}[Q_b^{\text{SR}}(g) \cdot Q_b^{\text{SR}}(w)] = g \cdot w$（条件独立下） |

### B.3 GEF

| 公式 | 表达式 |
|------|--------|
| GEF 梯度 | $\text{grad}_x = g_{dq} \cdot w_{dq} + (g - g_{dq}) \cdot w_{\text{float}}$ |
| GEF 误差 | $\varepsilon_{\text{GEF}} = g_{dq} \cdot \varepsilon_w$ |
| 无GEF 误差 | $\varepsilon_{\text{noGEF}} = g \cdot \varepsilon_w + \varepsilon_g \cdot w + \varepsilon_g \cdot \varepsilon_w$ |

### B.4 EF

| 公式 | 表达式 |
|------|--------|
| 闭环更新 | $\hat{g}_t = g_t + e_{t-1}$, $q_t = Q(\hat{g}_t)$, $e_t = \hat{g}_t - q_t$ |
| 等价形式 | $q_t = g_t + (\eta_t - \eta_{t-1})$ |
| NTF | $\text{NTF}_N = (1 - z^{-1})^N$ |
| 权重噪声 TF | $W_{\text{noise}}(z) = -\text{lr} \cdot \eta(z) \cdot (1 - z^{-1})^{N-1}$ |
| 分离残差 | $e_t = \hat{g}_t - \text{truncate}(\hat{g}_t)$ |
| 深层放大 | 放大因子 $= \text{NTF 级联增益}(2^L) \times \text{Jacobian}^L$ |

### B.5 Level-AMP（M-凸性）

| 公式 | 表达式 |
|------|--------|
| 成本函数 | $f_i(b) = c_i / (2^b - 1)$ |
| 边际增益 | $\Delta_i(b) = c_i \cdot 2^b / [(2^b - 1)(2^{b+1} - 1)]$ |
| M-凸性比率 | $\Delta_i(b) / \Delta_i(b+1) = (4 \cdot 2^b - 1) / (2 \cdot 2^b - 2) > 1$ |

### B.6 MSint 多精度

| 公式 | 表达式 |
|------|--------|
| 位拆分 | $w_i = w_{h,i} \cdot 2^{16} + w_{l,i}$ |
| 多精度点积 | $y = \text{coarse} \cdot 2^{32} + \text{cross} \cdot 2^{16} + \text{fine}$ |
| 增量计算 | $y_D = y_{D-1} + Q(x) \cdot (v[D] \cdot \text{scale}[D])$ |