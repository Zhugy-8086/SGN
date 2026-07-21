# SGN-Lite v5.1.8

Sparse Graph Network Lite — 纯整数化竞争/赫布学习识别引擎

## 简介

SGN-Lite 是一个**核心识别路径无浮点、无反向传播**的轻量级识别引擎。核心机制为整数化竞争学习 + 模板匹配，适用于 PC 平台的算法验证。

**核心特性**:

- **核心引擎全程整数运算** (DiscreteCoordinate 离散坐标体系)
- 竞争学习 + 赫布增强/削弱
- **自适应随机静默机制** — 类 Dropout 正则化
- 图模式层级记忆
- 多层神经元架构 (Layer 0 基础特征 / Layer 1 概念判断)
- **L1 决策层** — 72 维图特征匹配 + Top-2 跨类平均分类
- **Level 调度器** — 运算精度语境管理
- 模块化引擎架构 (engine 拆分为 5 个模块)
- 策略插件化

> **注意**: 输入层 (矢量渲染、噪声生成、对比度拉伸) 涉及外部模拟信号，保留浮点运算。核心识别层 (神经元匹配、赫布学习、竞争排序) 严格整数化。

## 安装

### 环境要求

- Python 3.10+
- Windows / Linux / macOS

### 依赖

核心功能无需额外安装，Python 标准库即可运行。

```bash
# 可选依赖
pip install matplotlib    # 图表导出
pip install pygame-ce      # 函数工厂 GUI
```

## 快速开始

```bash
# 交互式启动
python run.py

# 内置预设
python run.py --preset quick      # 小网络 64 神经元, 500 步
python run.py --preset standard   # 中等网络 256 神经元, 2000 步
python run.py --preset precise    # 大网络 1024 神经元, 10000 步

# 批量模式
python run.py --batch

# 自动模式 (50ms/步)
python run.py --auto 50
```

### 函数工厂 GUI

基于 pygame 的图形化绘图与训练界面，支持手绘像素图形、实时变换、一键训练。

```bash
python sgn/gui/sgn_gui_factory.py
```

## 项目结构

```
.
├── sgn/
│   ├── engine/                # 核心引擎层
│   │   ├── core.py            # SGNCore 主引擎骨架
│   │   ├── config.py          # 配置管理 (ConfigRegistry, DiscreteCoordinate)
│   │   ├── level.py           # Level 调度器
│   │   ├── graph_train.py     # 图模式训练
│   │   ├── multi_layer_train.py  # 多层+分批训练
│   │   ├── l0_peer_compare.py # L0 同级比较与合并
│   │   ├── l1_decision.py     # L1 桶化/决策层
│   │   ├── hooks.py           # 事件总线
│   │   ├── strategies.py      # 策略抽象
│   │   ├── layers.py          # 层提取
│   │   ├── input.py           # 输入管道
│   │   └── utils.py
│   ├── graph/                 # 图模式层 (graph/stack/merge/graph_match)
│   ├── app/                   # 应用层 (main/interactive/panel/training/test/persist)
│   ├── gui/                   # GUI 层 (factory/canvas/charlib/transform/bridge/ui)
│   ├── tests/                 # 测试
│   ├── docs/                  # 架构文档
│   └── config/                # 默认配置
└── run.py                     # 启动入口
```

## 编程接口

```python
from engine import SGNCore
from engine.input import create_mixed_vector_source
from engine.layers import classify_sample

# 1. 创建引擎
core = SGNCore(seed=42)

# 2. 生成样本
source = create_mixed_vector_source(grid_size=8, samples_per_formula=100)
samples = source.generate_batch(2000)

# 3. 训练
for intensity, label in samples:
    core.train(intensity, label)

# 4. 推理
test_intensity, test_label = samples[0]
pred, score = classify_sample(core, test_intensity)
print(f"True: {test_label}, Pred: {pred}, Score: {score}%")
```

## 运行模式

| 模式 | 说明 |
|------|------|
| `full` | 全记载，每步输出 (默认) |
| `compact` | 精简，仅检查点摘要 |
| `blackbox` | 黑箱，训练全程零输出 |

## 输入源

| 类型 | 说明 |
|------|------|
| `pattern` | 内置 8×8 字符 (0-9 A-Z) |
| `vector` | 矢量图形 (line/circle/sine/arch/leaf/mixed)，支持 8/16/32/64 网格 |
| `file` | 从 CSV/JSON 文件加载 |
| `custom` | 函数工厂 GUI 生成的自定义训练集 |

## 关键配置

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `MAX_NEURONS` | 256 | 神经元数量上限 (与 L0+L1 双向同步) |
| `NEURON_LAYER_0_COUNT` | 128 | Layer 0 神经元数 (基础特征专家) |
| `NEURON_LAYER_1_COUNT` | 64 | Layer 1 神经元数 (概念判断专家) |
| `MAX_TEMPLATES` | 500 | 模板库容量上限 |
| `MAX_ITERATIONS` | 100000 | 训练总步数上限 |
| `SEED` | 42 | 随机种子 |
| `TOP_K` | 6 | 竞争 Top-K |
| `ENABLE_ADAPTIVE_SILENCE` | True | 自适应随机静默 |
| `SILENCE_MIN_RATIO` | 0.30 | 静默比例下限 |
| `SILENCE_MAX_RATIO` | 0.80 | 静默比例上限 |
| `ENABLE_GRAPH_MODE` | False | 图模式 |
| `ENABLE_MULTI_LAYER_NEURON` | False | 多层神经元架构 |
| `L1_DECISION_LAYER` | True | L1 决策层 |

完整配置见 [`sgn/config/sgn_config.json`](sgn/config/sgn_config.json)。

## 评估指标

训练完成后可用：

- `[t]` 批量测试: 识别率
- `[c]` 混淆矩阵: 各类别准确率
- `[n]` 噪声测试: 复合/高斯/椒盐/块遮挡鲁棒性
- `[s]` 统计信息: 神经元/模板/校验通过率
- `[g]` 仪表盘: 综合状态

## 概念声明

SGN 不是传统意义上的 SNN (脉冲神经网络) 或 DNN (深度神经网络)，拥有独立的概念体系和术语定义。

## 依赖

- Python 3.10+
- 可选: matplotlib (图表导出)
- 可选: pygame-ce (函数工厂 GUI)
- 可选: sqlite3 (SQLite 存储后端)

## 许可证

[Apache License 2.0](LICENSE)
