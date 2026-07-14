# SGN-Lite v5.1.8

Sparse Graph Network Lite -- 纯整数化竞争/赫布学习识别系统

## 简介

SGN-Lite 是一个**核心识别路径无浮点、无反向传播**的轻量级识别引擎。核心机制为整数化竞争学习 + 模板匹配，适用于嵌入式 MCU 环境的算法验证。

**核心特性**:
- **核心引擎全程整数运算** (DiscreteCoordinate 离散坐标体系)
- 竞争学习 + 赫布增强/削弱
- **自适应随机静默机制** (v5.1.6) -- 取代旧门控，类 Dropout 正则化
- 图模式层级记忆 (v5.0)
- 多层神经元架构 (v5.1.3)
- **L1 决策层与自组织分类** (v5.1.6) -- 分类由神经元输出涌现，非外部分类器
- **Level 调度器** (v5.1.5) -- 运算精度语境管理
- **模块化引擎架构** (v5.1.7) -- core.py 拆分为 5 个模块，技术债清理
- **数据污染与错位系统修复** (v5.1.8) -- 标签守卫、L0 纯特征化、图特征去共享化、推理一致性
- **CLI 增强与主页简化** (v5.1.8-ux) -- `--set`/`--preset`/`--status` + 模式选择快速启动
- 策略插件化架构

> **注意**: 输入层 (矢量渲染、噪声生成、对比度拉伸) 涉及外部模拟信号，保留浮点运算。核心识别层 (神经元匹配、赫布学习、竞争排序) 严格整数化。

## 安装

### 环境要求
- Python 3.10+
- Windows / Linux / macOS

### 安装依赖

核心功能无需额外安装，Python 标准库即可运行。

```bash
# 可选依赖（图表导出功能）
pip install matplotlib
```

> 未安装 matplotlib 时，脚本会自动降级为 ASCII 曲线显示。

## 快速开始

```bash
# 方式 1：使用启动入口
python run.py

# 方式 2：直接运行 app/main.py
python app/main.py
```

启动后进入**主页**，选择训练模式：
- `[1]` 快速验证 — 小网络，500步，快速跑通
- `[2]` 标准训练 — 中等网络，2000步
- `[3]` 精确训练 — 大网络，10000步
- `[4]` 自定义 — 进入完整参数面板

选择模式后自动配参，按 Enter 直接开始训练。也可使用 CLI 预设：

```bash
python run.py --preset quick --batch    # 快速验证
python run.py --preset standard --auto 50  # 标准训练
```

### 函数工厂 GUI

基于 pygame 的图形化绘图与训练界面，支持手绘像素图形、实时变换、一键训练。

```bash
# 直接启动 GUI（无参数，使用默认配置）
python gui/sgn_gui_factory.py

# 通过命令行控制启动配置
python gui/sgn_gui_factory.py --grid 16 --char A --label A
python gui/sgn_gui_factory.py --angle 30 --offset-x 2 --scale 0.8
python gui/sgn_gui_factory.py --variants 200 --export dataset.csv
python gui/sgn_gui_factory.py --dataset my_data.json --auto-train --train-steps 1000
python gui/sgn_gui_factory.py --noise gaussian --noise-prob 0.2

# 或从菜单进入
python run.py
# 按 f 启动函数工厂
```

**GUI 功能**：
- 网格绘制（8/16/32/64）+ 橡皮擦工具
- 内置字符库（0-9 A-Z 共 36 字符，8×8 插件化 CharRegistry，支持缩放到更大网格）
- 实时旋转变换预览（旋转/偏移/缩放）
- 一键训练 + 实时测试
- 撤销/重做（Ctrl+Z / Ctrl+Y，50步历史）
- 图案保存/加载（JSON 文件）
- 画布导出为 PNG 图片
- 自定义训练集管理（多标签共存）
- 导出 CSV/JSON 训练集
- 模型保存/加载（通过桥接层，与 SGN 核心解耦）

**GUI 命令行参数**（v5.1.5 新增）：

| 参数 | 说明 |
|------|------|
| `--grid N` | 初始网格大小 (8/16/32/64) |
| `--char C` | 启动时加载到画布的字符 |
| `--label L` | 初始训练标签 |
| `--angle DEG` | 初始旋转角度 |
| `--offset-x N` / `--offset-y N` | 初始偏移量 |
| `--scale F` | 初始缩放因子 |
| `--variants N` | 启动时生成 N 个变体 |
| `--export FILE` | 导出变体为 CSV/JSON |
| `--variant-angle DEG` | 变体随机旋转范围 (默认 15) |
| `--variant-offset N` | 变体随机偏移范围 (默认 6) |
| `--variant-scale F` | 变体随机缩放范围 (默认 0.2) |
| `--auto-train` | 启动后自动开始训练 |
| `--train-steps N` | 训练目标步数 |
| `--train-speed N` | 每帧训练步数 |
| `--noise TYPE` | 噪声类型 (composite/gaussian/salt_pepper/block/none) |
| `--noise-prob P` | 噪声翻转概率 |
| `--dataset FILE` | 启动时加载训练集 JSON 文件 |

详见 [`gui/sgn_gui_factory_documentation.md`](../gui/sgn_gui_factory_documentation.md)。

### 输出示例

```
============================================================
  SGN-Lite v5.1.8
============================================================
  当前网络: 256神经元 / 0模板 / 0步

  选择训练模式:
  [1] 快速验证  小网络(64), 500步, 快速跑通
  [2] 标准训练  中等网络(256), 2000步
  [3] 精确训练  大网络(1024), 10000步
  [4] 自定义    进入完整参数面板

  [i] 输入源  [o] 保存配置  [l] 加载配置
  [Enter] 开始训练  [q] 退出
============================================================
选择: 
```

选择 `[1]` 后自动应用预设，按 Enter 开始训练。选择 `[4]` 进入完整参数面板：

```
============================================================
  控制面板
============================================================
  当前网络: 256神经元 / 81模板 / 3200步
  [1] 网络架构    [2] 学习参数    [3] 资源限制
  [4] 噪声参数    [5] 界面偏好    [6] 高级选项
  [e] 扩展功能    [s] 保存配置    [l] 加载配置
  [r] 恢复默认    [Enter] 开始训练  [q] 退出
```

训练完成后可使用菜单中的 `[t]` 批量测试、`[c]` 混淆矩阵、`[n]` 噪声测试等功能。

### 命令行参数

```bash
python run.py                          # 交互式训练
python run.py --batch                  # 批量模式
python run.py --auto 50                # 自动模式(50ms/步)
python run.py --mode compact           # 精简模式
python run.py --mode blackbox          # 黑箱模式
python run.py --config config.json     # 加载配置
python run.py --no-color               # 禁用颜色
python run.py --custom-dataset file.json  # 从自定义训练集导入
python run.py --version                # 显示版本号
python run.py --set KEY=VALUE          # 设置配置项(可多次指定)
python run.py --preset quick           # 使用内置预设(quick/standard/precise)
python run.py --status                 # 输出网络状态后退出(非交互)
python run.py --status --json          # 输出 JSON 格式状态
```

**--set 示例**:
```bash
python run.py --set MAX_NEURONS=512 --set SEED=123 --batch
python run.py --preset standard --set FLIP_PROB=0.05 --auto 50
```

**--preset 预设**:
| 预设 | 神经元 | 步数 | 噪声 | 场景 |
|------|--------|------|------|------|
| quick | 64 | 500 | 0.15 | 快速验证 |
| standard | 256 | 2000 | 0.10 | 日常实验 |
| precise | 1024 | 10000 | 0.05 | 精确训练 |

## 项目结构

```
SGN5.1.3/
├── engine/                    # 核心引擎层
│   ├── core.py               # SGNCore 主引擎（骨架，v5.1.7 拆分后约 1100 行）
│   ├── graph_train.py        # 图模式训练 (v5.1.7 拆分)
│   ├── multi_layer_train.py  # 多层+分批训练 (v5.1.7 拆分)
│   ├── l0_peer_compare.py    # L0 同级比较与合并 (v5.1.7 拆分)
│   ├── l1_decision.py        # L1 桶化/决策层/P2 纠错 (v5.1.7 拆分)
│   ├── config.py             # 配置管理 (ConfigRegistry, DiscreteCoordinate)
│   ├── level.py              # Level 调度器 (v5.1.5)
│   ├── level_utils.py        # Level 辅助类
│   ├── hooks.py              # 事件总线 (HookRegistry, v5.1.7 修复 _WeakCallback)
│   ├── strategies.py         # 策略抽象
│   ├── layers.py             # 层提取
│   ├── input.py              # 输入管道
│   └── utils.py              # 工具函数
│
├── graph/                    # 图模式层
│   ├── graph.py              # DynamicGraph, GraphNode
│   ├── stack.py              # 图构建与投影
│   ├── merge.py              # 跨图合并
│   └── graph_match.py        # 图匹配与推理
│
├── app/                      # 应用层
│   ├── main.py               # 入口文件
│   ├── interactive.py        # 交互菜单
│   ├── panel.py              # 控制面板 + 主页 (homepage)
│   ├── presets.py            # 内置预设配置 (quick/standard/precise)
│   ├── training.py           # 训练循环
│   ├── test.py               # 推理/测试
│   ├── classify_utils.py     # 分类统一入口+模式感知阈值 (v5.1.7-patch2)
│   ├── visual.py             # 可视化
│   ├── report.py             # 图表导出
│   ├── persist.py            # 模型持久化
│   └── ...
│
├── gui/                      # GUI 层
│   ├── factory.py            # 函数工厂主窗口
│   ├── canvas.py             # 绘制画布
│   ├── charlib.py            # 字符库
│   ├── transform.py          # 变换引擎
│   ├── bridge.py             # SGN 桥接层
│   └── ui.py                 # UI 组件
│
├── tests/                    # 测试
│   ├── test_level_scheduler.py
│   └── test_level_complete.py
│
├── docs/                     # 文档
│   ├── README.md
│   ├── ARCHITECTUREv5.1.3.md
│   ├── level-scheduler-v5.1.5.md
│   ├── neuron-silence-mechanism-v5.1.6.md
│   └── l1-decision-layer-v5.1.6.md
│
├── config/                   # 配置
│   └── sgn_config.json
│
└── run.py                    # 启动入口
```

## 编程接口

除 CLI 外，SGN-Lite 可作为 Python 库直接调用：

```python
from engine import SGNCore
from engine.input import create_vector_source, create_mixed_vector_source
from engine.layers import classify_sample

# 1. 创建引擎
core = SGNCore(seed=42)

# 2. 创建输入源（矢量混合模式）
source = create_mixed_vector_source(grid_size=8, samples_per_formula=100)
samples = source.generate_batch(2000)

# 3. 训练
for intensity, label in samples:
    info = core.train(intensity, label)

# 4. 推理
test_intensity, test_label = samples[0]
pred, score = classify_sample(core, test_intensity)
print(f"True: {test_label}, Pred: {pred}, Score: {score}%")

# 5. 保存/加载模型
from app.persist import save_model, load_model
save_model(core, "my_model.json")
```

```python
# 使用内置字符模式（8×8 插件化）
from engine import SGNCore
from engine.input import create_default_source

core = SGNCore(seed=42)
source = create_default_source()
samples = source.generate_batch(1000)

for intensity, label in samples:
    core.train(intensity, label)

print(f"Templates: {len(core.templates)}")
```

### Level 调度器 API (v5.1.5)

```python
from engine import LevelScheduler, OperationType

# 获取调度器
scheduler = core.level_scheduler

# 获取运算上下文
ctx = scheduler.get_context(OperationType.ADD, neuron_id=0)
print(f"Target level: {ctx.target_level}")

# 更新神经元统计
scheduler.update_stats(neuron_id=0, match=85, verified=True)

# 获取缓存统计
cache_stats = scheduler.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
```

## 运行模式

| 模式 | 说明 |
|------|------|
| full | 全记载，每步输出 (默认) |
| compact | 精简，仅检查点摘要 |
| blackbox | 黑箱，训练全程零输出 |

## 输入源

| 类型 | 说明 |
|------|------|
| pattern | 内置 8×8 字符 (0-9 A-Z, CharRegistry 插件化) |
| vector | 矢量图形 (line/circle/sine/arch/leaf/mixed) |
| file | 从 CSV/JSON 文件加载 |
| custom | 函数工厂 GUI 生成的自定义训练集 |

矢量图形支持网格大小: 8/16/32/64。通过 `--vector-grid` 或控制面板 `[g]` 设置。

## 关键配置项

| 配置 | 默认值 | 可调范围 | 说明 |
|------|--------|---------|------|
| MAX_NEURONS | 256 | 1 ~ 4096 | 神经元数量上限（与 L0+L1 双向同步） |
| NEURON_LAYER_0_COUNT | 128 | 16 ~ 512 | Layer 0 神经元数（基础特征专家） |
| NEURON_LAYER_1_COUNT | 64 | 8 ~ 256 | Layer 1 神经元数（概念判断专家） |
| MAX_TEMPLATES | 500 | 1 ~ 10000 | 模板库容量上限 |
| MAX_ITERATIONS | 100000 | 1 ~ 1000000 | 训练总步数上限 |
| SEED | 42 | 0 ~ 99999 | 随机种子 |
| TOP_K | 6 | 1 ~ 128 | 竞争 Top-K |
| MAX_LOCKOUT | 120 | 1 ~ 1000 | 锁定阈值 |
| ENABLE_ADAPTIVE_SILENCE | True | -- | 自适应随机静默 (v5.1.6) |
| SILENCE_MIN_RATIO | 0.30 | 0.0 ~ 0.95 | 静默比例下限 |
| SILENCE_MAX_RATIO | 0.80 | 0.05 ~ 1.0 | 静默比例上限 |
| ENABLE_GRAPH_MODE | False | -- | 图模式 (v5.0) |
| ENABLE_MULTI_LAYER_NEURON | False | -- | 多层神经元架构 (v5.1.3) |
| L1_DECISION_LAYER | True | -- | L1 决策层 (v5.1.6, v5.1.9 重写) |
| L1_BUCKET_ENABLED | True | -- | L1 图桶化防串位 (v5.1.6) |
| L1_FEATURE_LR | 0.3 | 0.05 ~ 0.8 | L1 特征模板赫布学习率 (v5.1.7-patch) |
| ACTIVE_SET_DECAY | 0.1 | 0.01 ~ 0.5 | 激活集频次衰减率，通用任意层 (v5.1.7-patch) |
| ACTIVE_SET_KEEP_THRESH | 0.3 | 0.05 ~ 1.0 | 激活集保留阈值，低于此淘汰 (v5.1.7-patch) |
| FEATURE_VARIANCE_THRESH | 100.0 | 1.0 ~ 10000.0 | 特征方差阈值，低于此降权 (v5.1.7-patch) |

> **说明**: 上表"默认值"为系统初始值，"可调范围"为允许的最小/最大值。实际稳态模板数通常在 80 个左右，无需达到上限。
>
> **神经元同步**: 修改 `MAX_NEURONS` 时自动按 67:33 比例更新 Layer 0/1 数量；修改 Layer 0 或 Layer 1 时自动更新 `MAX_NEURONS = L0 + L1`。
>
> **v5.1.6 旧门控已删除**: `ENABLE_GATE_MATCHING` / `ENABLE_SOFT_GATE` / `GATE_*` 系列配置已彻底删除，由自适应随机静默机制取代。详见 [`docs/neuron-silence-mechanism-v5.1.6.md`](neuron-silence-mechanism-v5.1.6.md)。

## Level 调度器 (v5.1.5)

Level 调度器将 level 从"数值的属性"变成"运算的语境"，实现存储和运算分离。

**核心设计**：
- level 管"怎么算"，不管"存什么"
- 调度器管"什么时候用什么算"，不管"算的是什么"

**内置策略**：
| 策略 | 说明 |
|------|------|
| StandardStrategy | 固定 level，不自适应 |
| AdaptiveStrategy | 根据匹配值方差动态调整 |
| LayerAwareStrategy | L0/L1 自动选择不同 level |

**自适应机制**：
- 匹配值方差 < 阈值 → 细粒度 level（更精确）
- 匹配值方差 > 阈值 → 粗粒度 level（更稳定）
- 验证率下降 → 回退原策略

详见 [`docs/level-scheduler-v5.1.5.md`](docs/level-scheduler-v5.1.5.md)。

## 自适应随机静默机制 (v5.1.6)

v5.1.6 将旧门控机制（`ENABLE_GATE_MATCHING` + `ENABLE_SOFT_GATE` + `GATE_*` 系列）重构为**自适应随机静默机制**：

**核心思想**：每步训练中，30%～80% 的神经元被随机选中进入"静默"状态，不参与任何判断、不输出值、不参与竞争。被静默的神经元下一步自动复活（标志每步重置）。

**与旧门控的区别**：
- 旧门控：gate 值连续增减，低于阈值退出竞争，难以复活
- 新静默：每步随机抽样，被选中即静默，下一步自动复活，类 Dropout 正则化

**补丁 P1（静默豁免）**：专精神经元被选入静默集的概率降低到普通神经元的 30%，避免专精状态因频繁静默而无法收敛。

核心配置：
- `ENABLE_ADAPTIVE_SILENCE` / `SILENCE_MIN_RATIO` / `SILENCE_MAX_RATIO`: 静默总开关与比例范围
- `SILENCE_TRIGGER_PROB`: 每步触发静默的概率（默认 0.5）
- `SILENCE_SPECIALIZE_THRESHOLD` / `SILENCE_SPECIALIZED_WEIGHT`: 专精触发阈值与豁免权重

> **v5.1.7-patch 专精机制更新**：`SILENCE_SPECIALIZE_THRESHOLD` 语义从"连续 N 次"改为"累计 N 次"+ 标签频次投票，shuffle 数据下专精神经元数量提升 8 倍。详见 [`docs/ARCHITECTURE_v5.1.7.md`](ARCHITECTURE_v5.1.7.md) §3.10。

> **旧门控已删除**：`ENABLE_GATE_MATCHING` / `ENABLE_SOFT_GATE` / `GATE_HIGH_THRESH` / `GATE_LOW_THRESH` / `GATE_DECAY_RATE` / `GATE_SPECIALIZE_THRESHOLD` 已从代码和 UI 中彻底删除。

详见 [`docs/neuron-silence-mechanism-v5.1.6.md`](neuron-silence-mechanism-v5.1.6.md)。

## 图模式 (v5.0)

图模式在神经元之上挂接层级化记忆系统，解决模板识别的三个固有缺陷:
1. 层级缺失 -> 多层图结构
2. 空间结构丢失 -> 连通域 + 位置归一化
3. 过拟合倾向 -> 一致性过滤 + 反馈循环

核心机制:
- 神经元竞争 -> 投影为图节点
- 多视图合并 -> 一致性过滤 (高层节点 >=2 投影)
- 反馈迭代 -> 误差图重新扫描
- 层级下压 -> 遗忘机制

## 多层神经元架构 (v5.1.3)

多层神经元架构在神经元之上新增 Layer 1 概念判断层，通过图中间层汇总器连接两层神经元，实现从基础特征到概念判断的层级抽象。

数据流:
- Layer 0 神经元竞争 (基础特征专家: 边缘/局部模式)
- 获胜者投影为图节点 -> 图赫布融合
- 提取图结构特征 (10维固定向量) + L0 激活 ID 列表
- Layer 1 神经元竞争 (图特征向量相似度匹配)
- 两层各自独立赫布学习

核心配置:
- `NEURON_LAYER_0_COUNT` / `NEURON_LAYER_1_COUNT`: 两层神经元数量（与 `MAX_NEURONS` 双向同步，比例约 2:1）
- `LAYER1_LEARNING_RATE_SCALE`: Layer 1 学习率缩放
- `TOP_K_L1`: Layer 1 竞争 Top-K
- `SILENCE_SPECIALIZE_THRESHOLD`: 专精化触发阈值（v5.1.6 由静默机制驱动；v5.1.7-patch 语义改为累计计数+标签频次投票）
- `L1_FEATURE_LR`: L1 特征模板赫布学习率 (v5.1.7-patch)
- `ACTIVE_SET_DECAY` / `ACTIVE_SET_KEEP_THRESH`: 激活集频次衰减参数（v5.1.7-patch，通用任意层）
- `FEATURE_VARIANCE_THRESH`: 特征方差降权阈值（v5.1.7-patch，通用任意层）

> **v5.1.7-patch 多层信号链修复**：修复了 L1 训练信号链的冷启动死锁和 4 个特征质量病根。所有修复为通用机制（相对阈值/激活集衰减/专精累计计数/方差降权），对未来 L2/L3 扩展友好。详见 [`docs/ARCHITECTURE_v5.1.7.md`](ARCHITECTURE_v5.1.7.md) §3.10。

> **v5.1.7-patch2 测试与分类一致性修复**：修复 4 个测试/分类一致性问题——统一 `classify_multi_layer` 分类逻辑、删除 `_graph_features` 隐式注入、`compute_l1_predicted_label` 返回 `(label, score)`、新建 `classify_utils.py` 按模式感知阈值（single/graph/multi = 80/80/30）替换 6 处硬编码 `>= 80`。基线验证确认 80 阈值导致 27%→0% 是测量误差而非算法缺陷。详见 [`docs/ARCHITECTURE_v5.1.7.md`](ARCHITECTURE_v5.1.7.md) §3.11。

> **v5.1.7-patch3 桶化数据污染修复**：L0 神经元回归纯特征检测器——删除 L0 的 specialization 设置逻辑，桶化一律按样本标签分桶。切断"5 的特征被分到 1 的桶"的污染链。L1 保留 specialization（分类层职责）。详见 [`fixes_相关修复/v5.1.7_桶化数据污染诊断与修复方案.md`](../fixes_相关修复/v5.1.7_桶化数据污染诊断与修复方案.md)。
>
> **v5.1.8 图特征去共享化**：删除原 18 维特征中所有图共享的 8 维 L0 掩码特征（`T[0] % 10000`），特征向量回到 10 维。新增 `hebbian_merge` 标签守卫，L0 specialization 彻底移除，推理统一应用静默机制。详见 [`fixes_相关修复/v5.1.8_综合修复方案.md`](../fixes_相关修复/v5.1.8_综合修复方案.md)。

> **群组织架构方向**：长期架构方向简述。归一化贯穿全程（二值化→磁化对比→神经节压缩→神经组归一化），群组织是归一化的延续。统一图存储物理特征标签，与任务无关。详见 [`docs/群组织架构方向.md`](群组织架构方向.md)。

> **同步机制**: 修改 `MAX_NEURONS` 时自动按 67:33 比例分配到 Layer 0/1；修改任一 Layer 数量时自动更新 `MAX_NEURONS = L0 + L1`。详见 [`fixes_相关修复/v5.1.5_config_sync_display_fixes.md`](../fixes_相关修复/v5.1.5_config_sync_display_fixes.md)。

## L1 决策层 (v5.1.6, v5.1.9 重写)

v5.1.9 将 L1 决策层从"输出模式涌现聚类"重写为"72 维图特征匹配 + Top-2 跨类平均分类"，解决了旧版 10 维手工特征表达力不足和多类分类坍缩问题。

**核心思想**：L1 神经元记忆 72 维图特征模板（`T_features`），推理时将当前图特征与所有 L1 神经元的模板匹配，取按类别 Top-2 平均分最高的标签作为预测结果。

**三项机制**：
1. **72 维分布直方图特征（v5.1.9）**：激活强度 16 维 + 边强度 16 维 + 层级深度占比 8 维 + 空间位置 16 维 + 连通域 8 维 + 度分布 8 维，全 Level 0 整数，替代旧版 10 维手工统计量
2. **图特征匹配（v5.1.9）**：`match_layer1` 用 `<= 1` 整数最小分辨率容差匹配 72 维特征 + L0 激活重叠度 Jaccard 相似度，权重动态调整
3. **赢家衰减 + 输家复活（v5.1.9）**：连胜 L1 神经元学习量自动衰减（`win_streak`），长期未获胜神经元 `base.index` 每 100 步提升 1 点，防止单类坍缩
4. **Top-2 跨类平均（v5.1.9）**：分类时按预测标签分组，每组取 Top-2 平均分跨类比较，弱势类别也能被表达

> **v5.1.9 已删除的旧机制**：输出模式向量（`output_pattern`）、同级比较聚类（`l1_peer_compare_and_cluster`）、回馈图模式（`l1_feedback_to_graph`）、纠错机制（`cluster_reorganization` / `negative_feedback_uncluster`）。详见 [`fixes_相关修复/v5.1.9.md`](../fixes_相关修复/v5.1.9.md)。

核心配置：
- `L1_BUCKET_ENABLED` / `L1_BUCKET_BY` / `L1_CHAIN_ENABLED` / `L1_CHAIN_SIMILARITY`: 桶化与链条化
- `L1_DECISION_LAYER`: 决策层开关
- `L1_FEATURE_LR`: L1 特征模板赫布学习率
- `ACTIVE_SET_DECAY` / `ACTIVE_SET_KEEP_THRESH`: L0 激活集频次衰减
- `FEATURE_VARIANCE_THRESH`: 特征方差阈值（低于此降权特征匹配）

## 模块化引擎架构 (v5.1.7)

v5.1.7 将 2065 行的 `core.py` 拆分为 5 个模块，并清理历史技术债：

**模块拆分**：
| 模块 | 职责 | 函数数 |
|------|------|--------|
| `engine/core.py` | 骨架：`__init__`、`create_neuron`、核心方法、`train()`、`get_state` | 23 个保留方法 |
| `engine/graph_train.py` | 图模式训练（反馈循环/重建/误差/下压） | 8 个函数 |
| `engine/multi_layer_train.py` | 多层+分批训练（L0竞争/L1阶段/batch） | 7 个函数 |
| `engine/l0_peer_compare.py` | L0 同级比较与合并（模板相似度/投票合并） | 4 个函数 |
| `engine/l1_decision.py` | L1 桶化/决策层/聚类/回馈/P2纠错 | 11 个函数 |

`core.py` 通过委托方法调用各模块函数，外部接口完全不变。

**技术债清理**：
1. **删除双写死代码**：`base_index` / `enc_b_index` 从未被读取，直接删除（`_hebbian_learn` 中 6 处赋值 + `create_neuron` 中 2 个字段定义）
2. **修复 _WeakCallback 边界情况**：
   - lambda 注册静默丢失 → 检测即时 GC，自动转强引用兜底
   - staticmethod 对象直接传入崩溃 → 提取 `__func__` 走普通函数路径
   - partial 注册的回调无法 unregister → 补充 `_orig_callback` 匹配分支

> **向后兼容**：模块拆分通过委托方法实现，所有外部调用（`layers.py`、`storage.py`、`app/*`）无需修改。

## 评估指标

训练完成后可用的测试:
- `[t]` 批量测试: 识别率
- `[c]` 混淆矩阵: 各类别准确率
- `[n]` 噪声测试: 复合/高斯/椒盐/块遮挡鲁棒性
- `[s]` 统计信息: 神经元/模板/校验通过率
- `[g]` 仪表盘: 综合状态

## 版本历史

- v4.3: 长周期参数重构、Bug修复、8x8标准字符库
- v4.4: 双图叠加门控识别、边缘提取、分块编码
- v5.0: 图模式层级记忆 (graph/stack/merge/graph_match)
- v5.0-fix: 矢量圆距离计算Bug修复、渲染渐变方向修复
- v5.0-vis: 可视化增强 (10级色阶/8级热力图/模板强度模式/形状标注)
- v5.0-grid: VECTOR_GRID 扩展支持 64x64
- v5.1: catear 语义修正、核心引擎性能优化
- v5.1.1: 函数工厂 GUI (pygame)
- v5.1.2: GUI 桥接层解耦
- v5.1.3: 多层神经元架构 -- Layer 0/1 双层竞争、图中间层汇总器、软门控
- v5.1.5: 4×4 硬编码移除 → 插件化 8×8 (CharRegistry)、GUI 命令行参数支持 (argparse)、D 默认值 16→64
- **v5.1.5**: Level 调度器 -- 运算精度语境管理、策略可热替换、自适应 level 调整、性能缓存优化
- v5.1.5-fix: 配置文件路径改绝对路径；神经元数量三方双向同步；架构参数重建检测改精确追踪；二值化显示改分层+合并签名
- **v5.1.6** (内部版本，未公开): 多层模式 Bug#1-#8 修复；旧门控重构为自适应随机静默机制（含 P1 静默豁免）；分批次训练（3.1）；L0 同级比较与合并（3.2 含 P4 累积缓冲）；L1 图模式桶化与链条化（3.3）；L1 决策层与自组织分类（3.4 含 P2 纠错机制）
- **v5.1.7** : core.py 拆分为 5 个模块（graph_train / multi_layer_train / l0_peer_compare / l1_decision / core 骨架）；删除 base_index/enc_b_index 双写死代码；修复 _WeakCallback 边界情况（lambda 静默丢失、staticmethod 崩溃、partial 无法 unregister）
- **v5.1.7-patch** (信号链与特征质量修复): 两阶段修复多层训练信号链——阶段一打破 `l1_verified` 冷启动死锁（新增 `compute_l1_predicted_label`）；阶段二修复 4 个特征质量病根（相对阈值/激活集频次衰减/专精累计计数+标签投票/方差降权）。所有修复为通用机制，对未来 L2/L3 扩展友好。详见 [`fixes_相关修复/v5.1.7_信号链死锁修复补丁.md`](../fixes_相关修复/v5.1.7_信号链死锁修复补丁.md) 和 [`fixes_相关修复/v5.1.7_特征质量与专精瓶颈修复方案.md`](../fixes_相关修复/v5.1.7_特征质量与专精瓶颈修复方案.md)
- **v5.1.7-patch2** (多层测试与分类一致性修复): 修复 4 个测试/分类一致性问题——统一 `classify_multi_layer` 三路重复逻辑为 `compute_l1_predicted_label`、删除 `_graph_features` 隐式注入改显式参数、`compute_l1_predicted_label` 返回 `(label, score)` 避免重复匹配、新建 `classify_utils.py` 按模式感知阈值（single/graph/multi = 80/80/30）替换 6 处硬编码 `>= 80`。基线验证确认 80 阈值导致 27%→0% 是测量误差而非算法缺陷。详见 [`fixes_相关修复/v5.1.7_多层测试与分类一致性修复方案.md`](../fixes_相关修复/v5.1.7_多层测试与分类一致性修复方案.md)
- **v5.1.7-patch3** (桶化数据污染修复): L0 神经元回归纯特征检测器——删除 L0 的 specialization 设置逻辑，桶化一律按样本标签分桶。切断"5 的特征被分到 1 的桶"的污染链。L1 保留 specialization（分类层职责）。详见 [`fixes_相关修复/v5.1.7_桶化数据污染诊断与修复方案.md`](../fixes_相关修复/v5.1.7_桶化数据污染诊断与修复方案.md)
- **v5.1.8** (数据污染与错位综合修复): 综合修复 10 个独立诊断问题，分两阶段实施——阶段一补上已规划但未实施的缺口（`hebbian_merge` 标签守卫、`primary_graph_features` 回退删除、L0 specialization 永禁）；阶段二修复新发现的系统性污染路径（删除所有图共享的 L0 掩码特征 18→10 维、推理 L0 竞争统一静默机制、L1 specialization 退出机制、维度变更保护警告）。详见 [`fixes_相关修复/v5.1.8_综合修复方案.md`](../fixes_相关修复/v5.1.8_综合修复方案.md)
- **v5.1.8-ux** (CLI与控制面板用户体验改进): CLI 增强（`--set KEY=VALUE` 批量配置、`--version`、`--status`/`--json` 非交互状态输出、`--preset` 内置预设）；主页简化（模式选择快速启动，选 quick/standard/precise 自动配参，选自定义进完整面板）。详见 [`fixes_相关修复/v5.1.8_CLI控制面板用户体验改进方案.md`](../fixes_相关修复/v5.1.8_CLI控制面板用户体验改进方案.md)

> **版本公开说明**: v5.1.6 为内部开发版本，未对外公开发布。v5.0 是首个公开版本，在 v5.1.6 全部功能基础上完成技术债清理和模块化重构。v5.1.8 是当前最新版本，完成数据污染系统性修复。

## 当前版本声明 (先验机状态)

> **SGN-Lite v5.1.8 目前处于"先验机"状态——即正在完善中的原型机，尚未达到完整原型机阶段。**
>
> **版本号说明**: 当前版本号为先验机阶段版本号。一旦完成先验、正式定型为完整原型机，版本号将清零并作为独立项目重新发布。
>
> "先验"在此项目中指"原型机完成前的完善与验证过程"。先验机的含义是：核心框架已搭建、主要机制已实现，但仍处于持续完善阶段，关键基础设施尚未全部到位。
>
> **当前已实现**: 记忆层面的优化代码 (图模式层级记忆、模板合并、竞争学习机制、Level 调度器、自适应随机静默机制、L1 决策层与自组织分类、模块化引擎架构、数据污染系统修复)
>
> **尚未完善**:
> - 时间变量当前仅为计步器，未实现真正的时间维度处理
> - 神经元基础设施 (如动态拓扑、跨层信号传递) 尚未扩展
> - 图模式与神经元的深度衔接尚未完成
>
> **平台定位说明**: SGN 当前为 PC 平台的验证框架代码，**并非 MCU 部署版本**。未来发展方向包含 MCU 适配，但当前代码无法在嵌入式设备上直接运行。
>
> **概念声明**: SGN 不是传统意义上的 SNN (脉冲神经网络) 或 DNN (深度神经网络)，拥有独立的概念体系和术语定义。

## 常见问题

### 环境问题

**看到乱码/方块**
使用 PowerShell/Windows Terminal，或执行 `chcp 65001`，或添加 `--no-color` 禁用颜色输出。

**matplotlib 报错**
安装命令: `pip install matplotlib`。脚本会自动降级为 ASCII 曲线显示。

### 训练问题

**配置加载失败 "文件不存在"**
配置文件位于 `config/sgn_config.json`。v5.1.5-fix 已改用绝对路径，不依赖运行时工作目录。如仍有问题，确认 `config/` 目录存在且包含 `sgn_config.json`。

**训练速度太慢**
使用自动模式: `python run.py --auto 10`，或调大跳步显示。

**想继续之前训练**
启动时使用: `python run.py --resume`，或手动 `[d]` 加载模型文件。

## 依赖

- Python 3.10+
- 可选: matplotlib (图表导出)
- 可选: sqlite3 (SQLite存储后端)
- 可选: pygame-ce (函数工厂 GUI)

## 许可证

Apache License 2.0
