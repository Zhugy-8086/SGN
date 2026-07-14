# SGN-Lite v5.1.8 架构设计文档

> **版本说明**：本文档涵盖 v5.1.5 Level 调度器、v5.1.6 自适应静默机制/Bug修复/L0 同级比较/L1 决策层、v5.1.7 模块化引擎拆分、v5.1.8 数据污染综合修复全部架构变更。
>
> **旧文档**：`ARCHITECTUREv5.1.3.md`（已归档）。

---

## 1. 项目概述

SGN-Lite (Sparse Graph Network Lite) 是一个**核心识别路径无浮点、无反向传播**的轻量级识别引擎。核心机制为整数化竞争学习 + 模板匹配，适用于嵌入式 MCU 环境的算法验证。

### 1.1 核心特性

- **核心引擎全程整数运算** (DiscreteCoordinate 离散坐标体系)
- 竞争学习 + 赫布增强/削弱
- **自适应随机静默机制** (v5.1.6) — 取代旧门控，类 Dropout 正则化
- 图模式层级记忆 (v5.0)
- 多层神经元架构 (v5.1.3)
- **Level 调度器** (v5.1.5) — 运算精度语境管理，策略可热替换
- **L0 同级比较与合并** (v5.1.6) — 神经元模板投票合并，类集成学习
- **L1 决策层与自组织分类** (v5.1.6) — 分类由神经元输出涌现，非外部分类器
- **分批次训练** (v5.1.6) — 缓冲多步数据后批量更新
- **模块化引擎架构** (v5.1.7) — core.py 拆分为 5 个模块，技术债清理
- **数据污染与错位系统修复** (v5.1.8) — 标签守卫、L0 纯特征化、图特征去共享化、推理一致性
- 策略插件化架构

### 1.2 设计哲学

```
浮点连续空间 ──一次性投影──→ 整数格点空间 ──永不还原──→ 全程整数运算
```

与彩色识别管线的同构：
- 彩色识别：RGB 连续强度 → 布尔空间（0/1），运算为集合运算
- DiscreteCoordinate：浮点连续值 → 整数格点空间，运算为整数算术

### 1.3 旧门控已删除

v5.1.6 将以下旧门控配置**彻底删除**：
- `ENABLE_GATE_MATCHING` / `ENABLE_SOFT_GATE`
- `GATE_HIGH_THRESH` / `GATE_LOW_THRESH` / `GATE_DECAY_RATE`
- `GATE_SPECIALIZE_THRESHOLD`

由自适应随机静默机制取代。详见 [neuron-silence-mechanism-v5.1.6.md](neuron-silence-mechanism-v5.1.6.md)。

---

## 2. 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     应用层 (app/)                                │
│  main.py / training.py / test.py / interactive.py / panel.py    │
│  visual.py / report.py / metrics.py / draw.py / blackbox.py     │
│  persist.py / storage.py / backends.py / help.py                │
│  commands.py / cmd_registry.py / noise_equivalent.py            │
├─────────────────────────────────────────────────────────────────┤
│                     策略层 (engine/ + graph/)                     │
│  input.py / layers.py / strategies.py / level.py / level_utils  │
│  graph.py / stack.py / graph_match.py / merge.py                │
├─────────────────────────────────────────────────────────────────┤
│                     扩展层 (engine/)                              │
│  hooks.py / config.py / commands.py                              │
├─────────────────────────────────────────────────────────────────┤
│                     核心层 (engine/)                              │
│  core.py + graph_train.py + multi_layer_train.py                │
│  + l0_peer_compare.py + l1_decision.py                         │
│  (v5.1.7 拆分为 5 个模块，通过委托方法整合)                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 核心层 (Core Layer)

**v5.1.7 架构变更**：原 2065 行的 `core.py` 拆分为 5 个模块：

| 模块 | 行数 | 职责 |
|------|------|------|
| `engine/core.py` | ~1100 | 骨架：`__init__`、`create_neuron`、核心方法（匹配/校验/赫布学习/静默）、`train()`、`get_state` |
| `engine/graph_train.py` | ~270 | 图模式训练（反馈循环/重建/误差/下压/平行视图） |
| `engine/multi_layer_train.py` | ~395 | 多层+分批训练（L0竞争/L1阶段/batch/build_sample_pool） |
| `engine/l0_peer_compare.py` | ~195 | L0 同级比较与合并（模板相似度/投票合并/累积缓冲） |
| `engine/l1_decision.py` | ~440 | L1 桶化/链条化/输出模式/聚类/回馈/P2纠错/匹配 (v5.1.8: 移除共享掩码特征, -14行) |

**拆分方式**：`core.py` 通过委托方法调用各模块函数，外部接口完全不变：

```python
# core.py 内的委托模式
def _train_multi_layer(self, intensity, label):
    from .multi_layer_train import train_multi_layer
    return train_multi_layer(self, intensity, label)
```

**SGNCore 核心类**：

```python
class SGNCore:
    def __init__(self, seed=None, layer_strategy=None, verify_strategy=None):
        self.N = [create_neuron(i, self.D) for i in range(CONFIG["MAX_NEURONS"])]
        self.templates = []  # (label, mask, success_count, hit_counter)
        self.graphs: Dict[str, DynamicGraph] = {}
        self.neuron_layers = {0: self.N, 1: [...]}  # v5.1 多层
        self.level_scheduler = LevelScheduler()      # v5.1.5
        self.history = []                             # 训练记录
        self._step_counter = 0                        # 步数计
```

**神经元结构**（`create_neuron` 返回的字典）：

```python
{
    "nid": nid,                    # 神经元 ID
    "T": [...],                    # 模板向量（每层一个整数掩码）
    "base": DiscreteCoordinate,    # 基线响应速度（离散坐标）
    "lock": 0,                     # 锁定计数器
    "enc_r": 0,                    # 鼓励触发余数
    "enc_b": DiscreteCoordinate,   # 鼓励加成
    "L": False,                    # 是否锁定
    # v5.1.6 自适应随机静默字段
    "silenced": False,             # 本步是否被静默（每步重置）
    "specialization": None,        # 专精标签
    "consecutive_verified": 0,     # 累计验证通过计数（v5.1.7-patch: 不再要求连续）
    "label_freq": {},              # v5.1.7-patch: 验证通过时的标签频次统计（专精投票）
    "layer": 0,                    # 所属层 (0/1)
    # Layer 1 专用字段
    "T_features": [],              # 图特征模板
    "T_l0_active": {},             # L0 激活频次表（v5.1.7-patch: dict 存频次+衰减，防无限增长）
    "T_features_initialized": False,
    # 补丁 P4：合并累积缓冲
    "T_pending": None,             # 待写回的合并模板
    "merge_count": 0,              # 连续合并批次数
    # v5.1.9 L1 决策层字段
    "win_streak": 0,               # 赢家衰减计数器（连胜越多学习量越小）
}
```

**核心数据流**：

```
输入强度图 → extract_layers → 掩码层
    ↓
自适应随机静默 → 30%-80% 神经元被标记 silenced（每步随机抽取）
    ↓
未静默的 Layer 0 神经元竞争 (基础特征专家: 边缘/局部模式)
    ↓
获胜者投影为图节点 → 图赫布融合
    ↓
提取图结构特征 (10维固定向量) + L0 激活 ID 列表
    ↓
Layer 1 神经元竞争 (图特征向量相似度匹配)
    ↓
Layer 1 输出判断模式 → 桶化+链条化 → 同级比较聚类 → 涌现分类
    ↓
两层各自独立赫布学习 (+ 正确时 L1 回馈图模式)
```

### 2.2 扩展层 (Extension Layer)

#### 2.2.1 事件总线 - engine/hooks.py

**HookRegistry**: 弱引用事件系统，支持插件化扩展

```python
class HookRegistry:
    _callbacks: Dict[str, List[_WeakCallback]] = {}

    @classmethod
    def register(cls, event: str, callback: Callable, weak: bool = True): ...
    @classmethod
    def emit(cls, event: str, *args, **kwargs): ...
```

**v5.1.7 修复**：`_WeakCallback` 处理了三个边界情况：
- lambda 注册即时 GC → 自动转强引用兜底
- staticmethod 对象直接传入崩溃 → 提取 `__func__`
- partial 注册的回调无法 unregister → 补充 `_orig_callback` 匹配分支

**事件命名约定**:
| 事件名 | 触发时机 |
|--------|----------|
| `sgn:before_step` | 训练步开始前 |
| `sgn:after_step` | 训练步完成后 |
| `sgn:on_template_added` | 新模板被添加时 |
| `sgn:on_config_changed` | 配置项变更时 |
| `sgn:on_neuron_locked` | 神经元被锁定时 (预留) |

#### 2.2.2 配置注册表 - engine/config.py

**ConfigRegistry**: 类型安全的配置管理，支持运行时注册

```python
@dataclass
class ConfigItem:
    key: str
    default: Any
    type: Type
    range_hint: Tuple[Any, Any]
    requires_rebuild: bool = False
    description: str = ""

class ConfigRegistry:
    _registry: Dict[str, ConfigItem] = {}
    _values: Dict[str, Any] = {}
```

**v5.1.5 改进**：
- 架构参数修改精确追踪（`_arch_modified_keys`），避免"改了又改回"误判
- 神经元数量三方双向同步（MAX_NEURONS ↔ L0_COUNT ↔ L1_COUNT）
- 配置文件绝对路径（不依赖 cwd）

**CharRegistry**（v5.1.5 新增）：插件化 8×8 字符库，取代 4×4 硬编码

- 内置 0-9 A-Z 共 36 字符
- 支持缩放到更大网格
- `STANDARD_CHARS_8x8` 统一字符定义

**DiscreteCoordinate**: 离散坐标系统

```python
class DiscreteCoordinate:
    """浮点连续空间一次性投影到整数格点，进入后永不还原"""
    def __init__(self, index: int, level: int = 2):
        self.index = int(index)  # 该层级下的整数坐标
        self.level = int(level)  # 层级编号

    @classmethod
    def from_float(cls, f: float) -> "DiscreteCoordinate":
        # 0.02 → level=2, index=2
        # 2.0  → level=0, index=2
```

**层级定义**:
| Level | Scale | 格点间距 | 用途 |
|-------|-------|----------|------|
| 0 | 1 | 1.0 | 整数空间 |
| 1 | 10 | 0.1 | 0.1 空间 |
| 2 | 100 | 0.01 | 0.01 空间 |
| 3 | 1000 | 0.001 | 0.001 空间 |

#### 2.2.3 命令注册表 - app/commands.py + app/cmd_registry.py

命令模式封装，支持动态扩展菜单命令。v5.1.5 新增 `register_all_commands()` 聚合入口。
包含 21 个命令回调，覆盖训练/测试/可视化/系统全类别。

---

### 2.3 策略层 (Strategy Layer)

#### 2.3.1 输入管道 - engine/input.py

**抽象接口**:
```python
class NoiseModel(ABC):
    @abstractmethod
    def apply(self, base_pattern: List[int]) -> List[int]: ...

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, intensity: List[int]) -> Tuple[List[int], int]: ...

class InputSource(ABC):
    @abstractmethod
    def generate_batch(self, n: int) -> List[Tuple[List[int], str]]: ...
```

**内置噪声模型**:
| 模型 | 类 | 说明 |
|------|-----|------|
| 复合噪声 | `DefaultCompositeNoise` | 40% 电平翻转 + 30% 强抖动 + 30% 弱抖动 |
| 高斯噪声 | `GaussianNoise` | 连续随机噪声 |
| 椒盐噪声 | `SaltPepperNoise` | 离散故障 |
| 块遮挡 | `BlockNoise` | 大面积缺失 |

#### 2.3.2 策略抽象 - engine/strategies.py

**v5.1.5 重构**：纯抽象化，所有硬编码阈值解耦

| 策略 | 适用范围 | 特点 |
|------|----------|------|
| `DefaultLayerStrategy` | 8×8 ~ 16×16 | 基线策略 |
| `SparseLineStrategy` | 16×16+ | 细线条/矢量图形，降低标记阈值 |
| `GlobalMatchStrategy` | 全局 | 匹配策略（可热插拔） |

此外还有 `VerifyStrategy`（校验策略）系列，可选注入。

#### 2.3.3 Level 调度器 - engine/level.py + engine/level_utils.py (v5.1.5 新增)

**设计原则**：
- level 管"怎么算"，不管"存什么"
- 调度器管"什么时候用什么算"，不管"算的是什么"
- 存储和运算分离，策略和数值分离，上层完全无感

**核心组件**：

```python
class OperationType(Enum):
    ADD = "add"          # 加法（赫布学习增强）
    SUB = "sub"          # 减法（赫布学习削弱）
    MUL = "mul"          # 乘法（响应速度计算）
    COMPARE = "compare"  # 比较（排序/匹配）
    ASSIGN = "assign"    # 赋值（初始化/重置）

@dataclass
class LevelContext:
    target_level: int
    operation: OperationType
    source: str = ""

class LevelScheduler:
    def get_context(self, operation, neuron_id) -> LevelContext: ...
    def resolve_binary_op(self, op, left_level, right_level) -> int: ...
    def update_stats(self, neuron_id, match, verified): ...
    def register_strategy(self, strategy) -> bool: ...
    def serialize(self) -> Dict: ...
    def deserialize(self, data: Dict) -> None: ...

class LevelStrategy(ABC):
    @property
    def name(self) -> str: ...
    @property
    def default_level(self) -> int: ...
    def get_level_for_operation(self, operation, neuron_id, stats) -> int: ...
    def suggest_adaptation(self, stats) -> Optional[int]: ...
```

**内置策略**：
| 策略 | 类 | 说明 |
|------|-----|------|
| 标准策略 | `StandardStrategy` | 固定 level，不自适应 |
| 自适应策略 | `AdaptiveStrategy` | 根据匹配值方差动态调整 |
| 层级感知策略 | `LayerAwareStrategy` | L0/L1 自动选择不同 level |

**缓存优化**：调度器内置 LRU 缓存，同一 (neuron_id, operation) 组合的上下文查询结果会缓存，命中率可达 90%+，减少重复计算。

**LayerAwareStrategy 示例**：
```python
# L0 神经元 → 细粒度 (level=1)，因为 L0 处理基础特征
# L1 神经元 → 粗粒度 (level=2)，因为 L1 处理概念判断
```

#### 2.3.4 图数据结构 - graph/graph.py

```python
@dataclass
class GraphNode:
    node_id: int
    layer: int  # 0=L0底层, 1=L1, 2=L2, ...
    feature_vector: List[DiscreteCoordinate]
    position_norm: Tuple[int, int]  # 归一化位置 (0~1000)
    neighbors: Dict[int, DiscreteCoordinate]  # 邻居连接强度
    activation: int  # 激活计数（赫布学习累积）

@dataclass
class DynamicGraph:
    nodes: Dict[int, GraphNode]
    _hash_index: Dict[int, Set[int]]  # MinHash 索引
```

**图操作**:
- `hebbian_merge()`: 赫布融合新节点
- `demote_all()`: 层级下压遗忘
- `find_best_match()`: MinHash 快速召回 + 精确匹配
- `get_total_nodes()`: 图节点总数

#### 2.3.5 图构建与投影 - graph/stack.py

```python
def project_neurons_to_graph(winners, intensity, d, grid_size, step, label):
    """神经元投影为图节点"""
    # 1. 掩码 → 连通域 (BFS 8-邻域)
    # 2. 连通域 → L0 GraphNode (质心/面积/宽高比)
    # 3. 返回 DynamicGraph
```

#### 2.3.6 跨图合并 - graph/merge.py

```python
def merge_winner_projections(projections, label, step):
    """合并多个神经元投影图"""
    # 1. 一致性过滤：高层节点必须出现在≥2个投影中
    # 2. MinHash 聚类
    # 3. 簇内合并（特征向量平均）
```

#### 2.3.7 图匹配与推理 - graph/graph_match.py

```python
def graph_similarity(graph1, graph2) -> int:
    """图相似度计算 (0-100)"""
    # MinHash 快速召回 + 节点级特征比较

def classify_with_graph(core, intensity, d, grid_size):
    """图模式分类"""
```

---

### 2.4 应用层 (app/)

| 文件 | 职责 |
|------|------|
| `main.py` | 入口文件，命令行解析，主循环 |
| `interactive.py` | 交互菜单调度（转发到 panel/help/training/blackbox） |
| `panel.py` | 控制面板/扩展菜单/高级选项 |
| `training.py` | 训练循环（含分批次训练） |
| `test.py` | 推理/批量/混淆/噪声测试 |
| `visual.py` | 可视化/统计/仪表盘 |
| `report.py` | 图表导出 (matplotlib/ASCII) |
| `draw.py` | 二值网格/索引网格绘制 |
| `metrics.py` | Metric 抽象体系 + 准确率/混淆矩阵/噪声鲁棒性 |
| `backends.py` | 图表后端（matplotlib/ASCII/CSV） |
| `storage.py` | 持久化策略（JSON/SQLite + AutosaveStrategy） |
| `persist.py` | 模型保存/加载/自动保存检测 |
| `blackbox.py` | 黑箱验证模式 |
| `help.py` | 帮助手册 |
| `commands.py` | 命令模式核心 |
| `cmd_registry.py` | 21 个命令回调注册聚合 |
| `noise_equivalent.py` | 噪声等效性验证 |
| `__init__.py` | 包标识 |

### 2.5 GUI 层 (gui/)

基于 pygame 的图形化绘图与训练界面。v5.1.5 新增命令行参数支持（argparse）。

| 文件 | 职责 |
|------|------|
| `sgn_gui_factory.py` | GUI 启动入口/命令行参数解析 |
| `factory.py` | 函数工厂主窗口 (GUIFactory) |
| `canvas.py` | 绘制画布 (网格/橡皮擦/Bresenham) |
| `charlib.py` | 字符库 (全ASCII + pygame字体渲染) |
| `transform.py` | 变换引擎 (旋转/偏移/缩放) |
| `bridge.py` | SGN 桥接层 (封装 history/save/load) |
| `ui.py` | UI 组件 (Button/Slider/Label/TextBox) |
| `theme.py` | GUI 主题配置 |
| `utils.py` | GUI 工具函数 |
| `dataset_store.py` | 自定义训练集存储与加载 |
| `custom_input_source.py` | 自定义训练集 InputSource 适配器 |

---

## 3. 核心算法

### 3.1 竞争学习

```python
def train(self, intensity, label):
    # 0. 自适应随机静默（v5.1.6）
    self._apply_adaptive_silence()

    # 1. 提取掩码层
    layers, layer_count = extract_layers(intensity, d=self.D)

    # 2. 竞争：计算每个未静默神经元的响应速度
    matches = []
    for i, n in enumerate(self.N):
        if n.get("silenced"):  # 静默神经元跳过
            continue
        match = self._match(n, layers, layer_count)
        speed = self._response_speed(n, match)
        matches.append((speed, match, i))

    # 3. Top-K 选择
    matches.sort(reverse=True)
    winners = [i for _, _, i in matches[:top_k]]

    # 4. 校验
    verified = self._verify(intensity, layers, layer_count)

    # 5. 赫布学习
    self._hebbian_learn(winners, verified)

    # 6. 模板处理
    if verified:
        self._add_template(label, layers, layer_count)
```

### 3.2 响应速度计算

```python
def _response_speed(self, n, match):
    """响应速度 = base.index + match * gamma.index（全程整数）"""
    gamma = CONFIG["GAMMA"]     # DiscreteCoordinate
    base = n["base"]             # DiscreteCoordinate

    if base.level != gamma.level:
        base = base.to_level(gamma.level)

    rsp = base.index + match * gamma.index

    if n["enc_r"] > 0:
        enc = n["enc_b"]
        if enc.level != gamma.level:
            enc = enc.to_level(gamma.level)
        rsp += enc.index

    return rsp
```

### 3.3 赫布学习

```python
def _hebbian_learn(self, winners, verified):
    for nid in winners:
        n = self.N[nid]
        ctx = self.level_scheduler.get_context(
            OperationType.ADD if verified else OperationType.SUB,
            neuron_id=nid
        )
        target_level = ctx.target_level

        if verified:
            n["base"] = DiscreteCoordinate(base.index + delta, target_level)
            n["enc_r"] = enc_cnt
            n["enc_b"] = enc_bonus
        else:
            n["base"] = DiscreteCoordinate(base.index - wr.index, target_level)
            n["lock"] += 1
            if n["lock"] >= CONFIG["MAX_LOCKOUT"]:
                n["L"] = True  # 锁定
```

### 3.4 自适应随机静默机制 (v5.1.6)

**核心思想**：每步训练前，30%～80% 的神经元被随机选中进入"静默"状态，不参与任何判断、不输出值、不参与竞争。被静默的神经元下一步自动复活（标志每步重置）。

```
旧门控: gate 值驱动 → 验证通过 gate+1, 失败 gate-1 → 低于阈值退出竞争
新静默: 随机抽样驱动 → 每步重新抽 → 被选中即静默 → 下一步自动复活
```

**与 Dropout 的关系**：自适应随机静默等价于神经元级的 Dropout，但比标准 Dropout 更简洁——不涉及概率缩放。静默比例在 `[SILENCE_MIN_RATIO, SILENCE_MAX_RATIO]` 范围内自适应调整：

- 正确率高 → 静默比例趋近 `MAX_RATIO`（更多正则化）
- 正确率低 → 静默比例趋近 `MIN_RATIO`（更多学习）

**补丁 P1（静默豁免）**：专精神经元被选入静默集的概率降低到普通神经元的 30%，避免专精状态因频繁静默而无法收敛。

**配置项**：
| 配置 | 默认 | 说明 |
|------|------|------|
| `ENABLE_ADAPTIVE_SILENCE` | True | 启用 |
| `SILENCE_MIN_RATIO` | 0.30 | 静默比例下限 |
| `SILENCE_MAX_RATIO` | 0.80 | 静默比例上限 |
| `SILENCE_TRIGGER_PROB` | 0.5 | 每步触发静默的概率 |
| `SILENCE_SPECIALIZE_THRESHOLD` | 10 | 专精触发阈值 |
| `SILENCE_SPECIALIZED_WEIGHT` | 0.3 | 专精神经元静默权重 |

### 3.5 多层神经元架构 (v5.1.3)

**Layer 0 → 图汇总 → Layer 1 → 涌现分类 (v5.1.6)**

```
输入强度图
    ↓
extract_layers → 掩码
    ↓
自适应随机静默 → 标记 silenced
    ↓
Layer 0 神经元竞争（基础特征专家：边缘/局部模式）
    ↓
project_neurons_to_graph → 图 L0 节点
    ↓
DynamicGraph.hebbian_merge → 图 L0→L1 组合
    ↓
extract_graph_features → 10维图结构特征 + L0 激活 ID 列表
    ↓
Layer 1 竞争（图特征向量相似度匹配）
    ↓
(v5.1.7) L1 桶化+链条化 → L1 输出判断模式 → 同级比较 → 涌现分类 → 回馈
    ↓
两层各自独立赫布学习
```

**图特征向量 (10维)**:
| 维度 | 特征 | 说明 |
|------|------|------|
| 0 | total_nodes | 总节点数 |
| 1 | max_layer | 最高层级 |
| 2 | avg_activation | 平均激活 |
| 3 | spread | 位置分散度 |
| 4 | layer_counts[0] | L0 节点数 |
| 5 | layer_counts[1] | L1 节点数 |
| 6 | layer_counts[2] | L2 节点数 |
| 7 | avg_edges | 平均边数 |
| 8 | feat_dim | 特征维度 |
| 9 | len(l0_active) | 本次激活的 L0 神经元数 |

**Layer 1 匹配策略** (v5.1.7-patch 更新):
```python
def _match_layer1(self, n, graph_features, l0_active_list):
    # 图特征相似度 - 相对阈值匹配（v5.1.7-patch: 不依赖特征值数量级）
    feat_sim = compare_features(n["T_features"], graph_features)

    # L0 激活重叠度 - Jaccard 相似度（兼容 dict 频次表和 list 旧格式）
    l0_sim = jaccard(n["T_l0_active"], l0_active_list)

    # v5.1.7-patch: 权重动态调整（基于特征方差）
    # 特征方差 < FEATURE_VARIANCE_THRESH 时，特征无区分度，降低 feat_sim 权重
    if feature_variance(graph_features) < FEATURE_VARIANCE_THRESH:
        return (feat_sim * 3 + l0_sim * 7) // 10  # 特征不可靠时依赖激活集
    else:
        return (feat_sim * 6 + l0_sim * 4) // 10  # 默认 60/40
```

**v5.1.7-patch 匹配机制改进**（面向多层扩展，对任意层通用）：
- **相对阈值**：`|t - g| / max(|t|, |g|, 1) <= 0.2`，不依赖特征值数量级，L2/L3 无需调阈值
- **激活集频次衰减**：`T_l0_active` 用 dict `{id: count}` 存储，旧计数衰减 + 低频淘汰，防止集合无限增长。L2 的 `T_l1_active` 可复用同一机制
- **方差降权**：用特征方差评估可靠性，方差小时自动降权特征、升权激活集。对任意特征向量通用

### 3.6 L0 同级比较与合并机制 (v5.1.6)

**核心思想**：多个 L0 神经元可能学到极其相似的模板，与其各自竞争，不如合并为一。流程：

1. **模板相似度计算**：批次结束后，对 L0 pool 中的每对神经元计算 `template_similarity(a, b)`
2. **投票合并**：相似度超过 `L0_MERGE_SIMILARITY`（默认 0.75）时，对两个神经元的模板逐位投票（1 的位越多，保留 1）
3. **弱信号过滤**：`merge_count` 小于 `L0_MERGE_BUFFER_THRESH`（默认 2）的神经元跳过合并

**补丁 P4（合并累积缓冲）**：多批次合并不就地覆盖，而是写入 `T_pending` 缓冲字段，达到阈值后才统一写回 `T`，避免覆盖冲突。

**配置项**：
| 配置 | 默认 | 说明 |
|------|------|------|
| `L0_PEER_COMPARE_ENABLED` | True | 启用 L0 同级比较 |
| `L0_MERGE_SIMILARITY` | 0.75 | 合并相似度阈值 |
| `L0_WEAK_SIGNAL_RATIO` | 0.15 | 弱信号过滤比例 |
| `L0_BINARIZE_THRESH` | 50 | 二值化阈值 |
| `L0_MERGE_BUFFER_THRESH` | 2 | 累积缓冲阈值 |

### 3.7 L1 决策层 (v5.1.6, v5.1.9 重写)

> **v5.1.9 重大变更**：L1 决策层从"输出模式涌现聚类"重写为"72 维图特征匹配 + Top-2 跨类平均分类"。旧的输出模式向量、同级比较聚类、回馈图模式、纠错机制已全部删除。详见 [`fixes_相关修复/v5.1.9.md`](../fixes_相关修复/v5.1.9.md)。

**核心思想**：L1 神经元记忆 72 维图特征模板（`T_features`），推理时将当前图特征与所有 L1 神经元的模板匹配，取按类别 Top-2 平均分最高的标签作为预测结果。

**三项机制**：

#### 3.7.1 L1 桶化与链条化 (3.3)

```
L1 获胜神经元
    ↓
按标签/专精分桶 → 每个桶独立投影 → 防数据串位
    ↓
桶内按模板相似度串链 → 保留特征演化路径
```

- `bucket_winners_by_label(core, winners, pool, sample)`：按 `L1_BUCKET_BY`（"label" 或 "specialization"）分桶
- `chain_winners(core, bucket, pool)`：桶内按相似度排序链接

#### 3.7.2 72 维分布直方图特征 (v5.1.9)

替代旧版 10 维手工统计量。全部 Level 0 整数，消除 `to_level` 转换开销和精度丢失。

| 特征组 | 维度 | 内容 |
|--------|------|------|
| 激活强度分布直方图 | 16 | 节点 activation 分 16 桶 |
| 边强度分布直方图 | 16 | 邻居连接强度分 16 桶 |
| 层级深度占比 | 8 | L0~L7 节点数占比（百分比整数） |
| 空间位置分布 | 16 | L0 节点 X 8 桶 + Y 8 桶 |
| 连通域大小分布 | 8 | BFS 连通域大小分 8 桶 |
| 度分布 | 8 | 节点度数分 8 桶 |

`match_layer1` 用 `<= 1` 整数最小分辨率容差匹配 72 维特征 + L0 激活重叠度 Jaccard 相似度，权重动态调整（特征方差 < 阈值时降权）。

#### 3.7.3 赢家衰减 + 输家复活 (v5.1.9)

在学习更新阶段做负抑制，防止单类坍缩：

- **赢家衰减**：连胜 L1 神经元 `win_streak` 递增，学习量 `eff_delta = max(1, delta // (1 + max(0, win_streak - 20) // 20))`
- **输家复活**：长期未获胜神经元 `win_streak` 递减，每 100 步 `base.index += 1`（受 `SPEED_SAT` 上限约束）

#### 3.7.4 Top-2 跨类平均分类 (v5.1.9)

替代旧版全局 argmax。按预测标签分组，每组取 Top-2 平均分跨类比较，弱势类别也能被表达。

### 3.8 分批次训练 (v5.1.6)

**核心思想**：不每步立即更新，而是缓冲多步数据后批量处理。

```python
def build_sample_pool(core, samples, step):
    """构建批次池"""
    # BATCH_SIZE = 32（默认）
    # 缓冲 BATCH_SIZE 条样本后统一处理
```

```python
def train_batch(core, samples, max_step):
    """分批训练循环"""
    while step < max_step:
        pool = build_sample_pool(core, samples, step)
        for intensity, label in pool:
            core.train(intensity, label)
        # 批次结束后触发 L0 同级比较
        if L0_PEER_COMPARE_ENABLED:
            l0_peer_compare_and_merge(core)
```

### 3.9 模块化引擎拆分 (v5.1.7)

| 模块 | 函数数 | 关键函数 |
|------|--------|----------|
| `engine/core.py` | 23 个保留方法 | `train()`, `_match()`, `_verify()`, `_hebbian_learn()`, `_apply_adaptive_silence()`, `get_state()` |
| `engine/graph_train.py` | 8 个函数 | `train_graph_mode()`, `get_parallel_views()`, `feedback_loop()`, `reconstruct_intensity()` |
| `engine/multi_layer_train.py` | 7 个函数 | `train_multi_layer()`, `l0_compete()`, `multi_layer_l1_phase()`, `build_sample_pool()`, `train_batch()` |
| `engine/l0_peer_compare.py` | 4 个函数 | `l0_peer_compare_and_merge()`, `template_similarity()`, `merge_templates_by_voting()` |
| `engine/l1_decision.py` | 5 个函数 | `bucket_winners_by_label()`, `chain_winners()`, `extract_graph_features()` (v5.1.9), `match_layer1()` (v5.1.9), `compute_l1_predicted_label()` (v5.1.7-patch) |

### 3.10 信号链与特征质量修复 (v5.1.7-patch)

v5.1.7-patch 分两阶段修复了多层训练信号链的死锁和特征质量问题。**设计原则：修复通用机制，不绑定特定数据格式，对未来 L2/L3 扩展友好。**

#### 阶段一：信号链死锁修复

**问题**：原 `l1_verified` 依赖 `specialization`，而 `specialization` 又依赖 `verified=True` 才能设置——冷启动死锁，L1 永远无法被验证。

**解法**：新增 `compute_l1_predicted_label()`（[`engine/l1_decision.py`](../engine/l1_decision.py)），将 L1 神经元的 `T_features` 与所有已积累的图匹配，取最佳匹配图的标签作为预测标签。完全不依赖 `specialization`，打破死锁。

```python
# multi_layer_train.py 中的核心逻辑
pred_label = core._compute_l1_predicted_label(best_l1)
l1_verified = (pred_label == label)
```

**死锁打破链**：
- 新生 L1 的 `T_features` 初始化为当前样本图特征 → 匹配自己出生的图得分最高 → `pred_label = 出生标签` → `l1_verified = True`
- `verified=True` → T_features 赫布更新分支可达 → `consecutive_verified` 递增 → 最终获得 `specialization`

#### 阶段二：特征质量与专精瓶颈修复

死锁打破后，准确率仍卡在 25%（随机基线）。诊断出 4 个独立病根，全部用**通用机制**修复：

| 修复 | 病根 | 通用机制 | L2 复用方式 |
|------|------|---------|------------|
| 相对阈值 | 绝对阈值对大值过松 | `|t-g| / max(|t|,|g|,1) <= 0.2` | 任意特征值范围无需调阈值 |
| 激活集衰减 | `T_l0_active` 只增不减 | dict 存频次 + 衰减 + 低频淘汰 | L2 的 `T_l1_active` 同机制维护 |
| 专精累计计数 | 连续 10 次在 shuffle 下概率≈0 | 累计 N 次 + 标签频次投票 | L2/L3 专精判定同机制 |
| 方差降权 | 60/40 固定权重放大无效信号 | 特征方差 < 阈值时降权特征 | 任意特征向量可靠性评估 |

**专精机制改进**（`_hebbian_multi_layer` 的 L1 分支）：
```python
# v5.1.7-patch: 累计计数 + 标签频次投票（不要求连续）
n["consecutive_verified"] += 1  # 失败时不重置（累计计数）
n["label_freq"][label] += 1
if n["consecutive_verified"] >= threshold:
    n["specialization"] = max(n["label_freq"], key=n["label_freq"].get)
```

**激活集频次衰减**（`_hebbian_multi_layer` 的 L1 分支）：
```python
# v5.1.7-patch: dict 存储 + 衰减 + 淘汰（防无限增长）
for nid in list(n["T_l0_active"].keys()):
    n["T_l0_active"][nid] *= (1 - ACTIVE_SET_DECAY)
    if n["T_l0_active"][nid] < ACTIVE_SET_KEEP_THRESH:
        del n["T_l0_active"][nid]
for nid in current_active:
    n["T_l0_active"][nid] += 1.0
```

#### 验证结果（400 步训练）

| 指标 | 死锁修复后 | 特征质量修复后 |
|------|-----------|---------------|
| 准确率 | 25.0%（随机基线） | 27-30% |
| 专精神经元 | 1/64 | 8/64 |
| L1 总 verified 次数 | 86 | 114 |
| T_l0_active 大小 | 无限增长 | 12-20（稳定） |

> **遗留**：准确率提升有限（27-30%），根因是特征提取层（`extract_graph_features`）的前 10 维拓扑统计特征无区分度。这是特征提取的独立问题，不在本 patch 处理——未来可为每层单独优化特征提取，不影响已修复的匹配/学习机制。

---

### 3.11 多层测试与分类一致性修复 (v5.1.7-patch2)

v5.1.7-patch2 修复了 patch 后暴露的 4 个测试/分类一致性问题。**设计原则：修复机制而非数据格式，所有改动对未来 L2/L3 扩展友好。**

#### 问题 1：`classify_multi_layer` 三路重复逻辑

**问题**：[`engine/layers.py`](../engine/layers.py) 的 `classify_multi_layer` 有多条重叠分类路径——A（`cluster_id` 涌现）、B（`specialization` 兜底）、C（遍历标签匹配）。三路径与训练侧 `_compute_l1_predicted_label` 并行演化，导致训练/推理不一致。

**解法**：删除路径 A/B 约 60 行，统一为复用 `compute_l1_predicted_label`（v5.1.9 进一步改为 Top-2 跨类平均）：

```python
# layers.py 中 v5.1.9 统一后的分类逻辑
label_scores = {}
for n in layer1_pool:
    if n["L"] or not n.get("T_features_initialized"):
        continue
    pred, pred_score = core._compute_l1_predicted_label(
        n, l0_active_nids, graph_features_cache
    )
    label_scores.setdefault(pred, []).append(pred_score)

for label, scores in label_scores.items():
    scores.sort(reverse=True)
    top2_avg = sum(scores[:2]) // min(2, len(scores))
    if top2_avg > best_score:
        best_score = top2_avg
        best_label = label
```

#### 问题 2：`_graph_features` 隐式数据注入

**问题**：训练侧通过 `w["_graph_features"] = primary_graph_features` 把图特征塞进 winners dict，`_hebbian_multi_layer` 再用 `w.get("_graph_features")` 读取——隐藏依赖，签名看不出来，失败时静默丢弃。

**解法**：改为显式参数：

```python
# core.py 新签名
def _hebbian_multi_layer(self, winners, verified, layer=0, label=None,
                         graph_features=None, l0_active_list=None):

# multi_layer_train.py 显式传参
core._hebbian_multi_layer(
    winners_l1, l1_verified, layer=1, label=label,
    graph_features=primary_graph_features,
    l0_active_list=primary_l0_active_list,
)
```

#### 问题 3：`compute_l1_predicted_label` 返回签名优化

**问题**：原函数只返回 `label`，调用方想知道分数必须再调一次 `_match_layer1`，重复计算。

**解法**：返回 `(label, score)` 元组，所有调用方一次调用即可同时获得标签和分数：

```python
# l1_decision.py
def compute_l1_predicted_label(...) -> Tuple[Optional[str], int]:
    ...
    return best_label, best_score
```

3 处调用方（`layers.py`、`multi_layer_train.py` 训练侧 2 处）全部更新为元组解包。

#### 问题 4：硬编码 `>= 80` 阈值不适用多层模式

**问题**：`app/test.py` 和 `app/metrics.py` 中 6 处 `>= 80` 判定正确性，但多层模式下 L1 匹配分数自然分布在 45-69（平均 51.9），80 阈值导致准确率从 27% 被低估为 0%。这是**测量误差**，不是算法缺陷（基线验证确认：无阈值 27%，80 阈值 0%，30 阈值 27%）。

**解法**：新建 [`app/classify_utils.py`](../app/classify_utils.py) 统一入口，按模式感知阈值：

```python
CLASSIFY_THRESHOLDS = {"single": 80, "graph": 80, "multi": 30}

def classify_with_mode(core, intensity, d=None) -> Tuple[str, int, str]:
    """返回 (预测标签, 最佳分数, 模式)"""

def classify_pass(core, intensity, label, d=None) -> Tuple[bool, str, int, str]:
    """返回 (是否正确, 预测标签, 最佳分数, 模式)"""
```

**替换范围**：
| 文件 | 函数 | 改动 |
|------|------|------|
| `app/test.py` | `_classify` | 委托到 `classify_with_mode` |
| `app/test.py` | `do_batch_test` / `do_noise_test` | `>= 80` → `classify_pass` |
| `app/metrics.py` | `_classify_single` | 委托到 `classify_with_mode` |
| `app/metrics.py` | `AccuracyMetric` / `NoiseRobustnessMetric` / `GeneralizationMetric` | `>= 80` → `classify_pass`（4 处） |

**未修改**（显示用阈值，不影响正确性）：
- `app/test.py` L130 `conf_str`（高/中/低标签）
- `app/test.py` L403 `pct_col`（颜色）
- `app/blackbox.py` L153/L251（置信度标签与颜色）
- `app/test.py` `do_confusion` / `app/metrics.py` `ConfusionMetric`（本就无阈值，按预测标签统计）

#### 验证结果

| 测试项 | 结果 |
|--------|------|
| 导入检查 | 7/8 模块通过（blackbox 函数名拼写无关） |
| `test_v517_supplement.py` | 8/8 通过，准确率 30.0% |
| `test_training_fixes.py` | 4/4 通过 |
| 基线对比 | 80 阈值 0% → 30 阈值 27%，确认测量误差已修复 |

#### 扩展性保证

所有修复均针对**机制**而非数据格式：
- `classify_with_mode` 通过 mode 字符串扩展新层（无需修改函数签名）
- `compute_l1_predicted_label` 对任意层通用（基于图匹配，不依赖层级硬编码）
- 显式参数传递消除了 winners dict 的隐式依赖，新增层数只需增加参数

> **v5.1.8 已处理遗留**：`T[0] % 10000` 模丢失问题已在 v5.1.8 通过删除共享 L0 掩码特征（18维→10维）解决——这 8 维特征对所有图标签完全相同，删除后反而提升了区分度。

详见 [`fixes_相关修复/v5.1.7_多层测试与分类一致性修复方案.md`](../fixes_相关修复/v5.1.7_多层测试与分类一致性修复方案.md)。

---

### 3.12 数据污染与错位综合修复 (v5.1.8)

v5.1.8 在吸收前序全部修复教训的基础上，对诊断出的 10 个独立问题分两阶段实施系统修复。**核心原则：修复机制而非数据格式，对未来 L2/L3 扩展友好。**

#### 阶段一：补上已知缺口（~20行）

实施前序文档已设计但未落到代码的 3 项修复：

**1. `hebbian_merge` 标签守卫**（[`graph/graph.py`](../graph/graph.py#L256-L264)）：
source_node 和 target 标签不一致时拒绝特征融合，强制新建独立节点。这是防止跨标签图特征污染的最终防线。

**2. `primary_graph_features` 回退删除**（[`engine/multi_layer_train.py`](../engine/multi_layer_train.py#L155-L159)）：
删除 `or not primary_graph_features` 的回退逻辑。配合已实施的桶化按真实标签（patch3 方向 A），`bucket_label == label` 几乎总会有匹配桶。边缘场景下 L1 跳过初始化优于错误初始化。

**3. L0 specialization 永禁**（[`engine/core.py`](../engine/core.py#L1173)）：
将专精判定条件从 `layer == 1` 改为 `layer >= 1`，删除 L0 的 `consecutive_verified` 累计逻辑。L0 回归纯特征检测器——不持任何分类认知。同时将专精保护条件从 `layer == 1` 改为 `layer >= 1`，为未来 L2/L3 扩展做好准备。

#### 阶段二：修复新发现的系统性污染（~50行）

**4. 图特征向量去共享化**（[`engine/l1_decision.py`](../engine/l1_decision.py#L368-L383)）**— 核心修复**：
删除 `extract_graph_features` 末尾追加的 8 维 L0 掩码特征（`T[0] % 10000`）。这 8 维数据来自当前样本的 L0 winners，对所有图标签完全相同——只有前 10 维拓扑特征能区分不同图。特征维度从 18 回到 10 维，干净且每维都有真实区分度。

**5. 推理 L0 竞争统一静默机制**（[`engine/layers.py`](../engine/layers.py#L177-L198)）：
将 `classify_multi_layer` 中的裸 L0 竞争循环替换为统一的 `l0_compete()` 调用（与训练时一致）。消除了训练/推理的 distribution shift——训练时 L1 看到"30-80% 静默后的 L0 激活模式"，推理时也看到同样的静默场景。

**6. L1 specialization 退出机制**（[`engine/core.py`](../engine/core.py#L1189-L1192)）：
专精神经元在 `spec_mismatch` 时不再永久跳过削弱。新增 `spec_mismatch_count` 计数器，连续 10 次 mismatch 后重置 `specialization = None`，给予纠正错误专精的机会。

**7. 维度变更保护警告**（[`engine/core.py`](../engine/core.py#L771-L777)）：
`_rebuild_for_dimension` 清空所有 graph/history 之前输出明确警告（包含图数量和节点数），避免静默毁灭训练数据。

#### 修改文件清单（6 文件，净 ~25 行）

| 文件 | 修复项 | 行变化 |
|------|-------|--------|
| [graph/graph.py](../graph/graph.py) | 标签守卫 | +8, -6 |
| [engine/multi_layer_train.py](../engine/multi_layer_train.py) | 回退删除 + 初始化空判断 | +3, -2 |
| [engine/core.py](../engine/core.py) | L0 专精永禁 + specialization 退出 + 维度警告 | +12, -7 |
| [engine/l1_decision.py](../engine/l1_decision.py) | 删除共享 L0 掩码特征 | -14 |
| [engine/layers.py](../engine/layers.py) | 推理统一静默 | +5, -12 |

#### 与前期修复的关系

```
v5.1.7 重构 → 死锁 patch → 特征质量 patch → patch2(阈值) → patch3(桶化A)
                                                                ↓
                                                          v5.1.8 本方案
                                                      ┌────────┼────────┐
                                                   阶段一    阶段二    (阶段三)
                                                 (补缺口)  (系统性修复) (统一图架构预留)
```

详见 [`fixes_相关修复/v5.1.8_综合修复方案.md`](../fixes_相关修复/v5.1.8_综合修复方案.md) 和 [`fixes_相关修复/v5.1.7_数据污染与错位问题综合诊断.md`](../fixes_相关修复/v5.1.7_数据污染与错位问题综合诊断.md)。

---

## 4. 数据流

### 4.1 训练流程 (v5.1.7)

```
┌─────────────────────┐
│ InputSource         │ 生成训练样本 (intensity, label)
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ NoiseModel          │ 施加噪声增强
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ SGNCore.train()     │
└──────┬──────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│  0. _apply_adaptive_silence()  ← 随机静默 30%-80% 神经元   │
│  1. extract_layers() → 掩码层                                │
│  2. 竞争: _match() + _response_speed() → Top-K winners      │
│  3. 校验: _verify() → verified                               │
│  4. 赫布: _hebbian_learn(winners, verified)   ← Level 调度器 │
│  5. 模板: _add_template(label, layers, layer_count)          │
│                                                              │
│  [图模式]: _train_graph_mode()                             │
│    → project_neurons_to_graph + merge + feedback loop       │
│                                                              │
│  [多层]: _train_multi_layer()                               │
│    → L0 竞争 → 图特征提取 → L1 竞争                         │
│    → (v5.1.7) L1 桶化/链条化/输出模式/聚类/回馈             │
│                                                              │
│  [分批次]: 缓冲 BATCH_SIZE 条后统一触发 L0 合并 + L1 聚类   │
└──────────────────────────────────────────────────────────────┘
       ↓
┌─────────────┐
│  history    │ 记录训练信息
└─────────────┘
```

### 4.2 推理流程 (v5.1.7)

```
输入强度图
    ↓
extract_layers → 掩码层
    ↓
神经元竞争 → Top-K winners
    ↓
[模板模式]                       [图模式]
classify_sample()               classify_with_graph()
  → 遍历 core.templates           → graph_similarity()
  → popcount 匹配度               → 与模板图比较
  → 取最高分标签                  → 取最高分标签

[多层模式]                       [L1 决策层模式]
classify_multi_layer()          图特征匹配分类（v5.1.9）
  → L0 竞争 → 图特征              → L1 匹配 72 维图特征
  → L1 匹配 → 遍历 graphs         → Top-2 跨类平均 → 预测标签
  → Top-2 跨类平均取最高分标签
```

---

## 5. 配置系统

### 5.1 关键配置项 (v5.1.7)

| 配置 | 默认值 | 可调范围 | 说明 |
|------|--------|---------|------|
| MAX_NEURONS | 256 | 1 ~ 4096 | 神经元数量上限（与 L0+L1 双向同步） |
| NEURON_LAYER_0_COUNT | 128 | 16 ~ 512 | Layer 0 神经元数（67% 比例） |
| NEURON_LAYER_1_COUNT | 64 | 8 ~ 256 | Layer 1 神经元数（33% 比例） |
| MAX_TEMPLATES | 500 | 1 ~ 10000 | 模板库容量上限 |
| MAX_ITERATIONS | 100000 | 1 ~ 1000000 | 训练总步数上限 |
| SEED | 42 | 0 ~ 99999 | 随机种子 |
| TOP_K | 6 | 1 ~ 128 | 竞争 Top-K |
| MAX_LOCKOUT | 120 | 1 ~ 1000 | 锁定阈值 |
| D | 64 | 16 ~ 4096 | 窗口像素总数（= grid_size²） |

### 5.2 静默机制配置 (v5.1.6)

| 配置 | 默认值 | 说明 |
|------|--------|------|
| ENABLE_ADAPTIVE_SILENCE | True | 启用自适应随机静默 |
| SILENCE_MIN_RATIO | 0.30 | 静默比例下限 |
| SILENCE_MAX_RATIO | 0.80 | 静默比例上限 |
| SILENCE_TRIGGER_PROB | 0.5 | 每步触发静默的概率 |
| SILENCE_SPECIALIZE_THRESHOLD | 10 | 专精触发阈值 |
| SILENCE_SPECIALIZED_WEIGHT | 0.3 | 专精神经元静默权重 |

### 5.3 分批次训练配置 (v5.1.6)

| 配置 | 默认值 | 说明 |
|------|--------|------|
| BATCH_TRAIN_ENABLED | True | 启用分批次训练 |
| BATCH_SIZE | 32 | 批次大小 |
| BATCH_SHUFFLE | True | 批次内是否打乱 |

### 5.4 L0 同级比较配置 (v5.1.6)

| 配置 | 默认值 | 说明 |
|------|--------|------|
| L0_PEER_COMPARE_ENABLED | True | 启用 L0 同级比较 |
| L0_MERGE_SIMILARITY | 0.75 | 合并相似度阈值 |
| L0_WEAK_SIGNAL_RATIO | 0.15 | 弱信号过滤比例 |
| L0_BINARIZE_THRESH | 50 | 二值化阈值 |
| L0_MERGE_BUFFER_THRESH | 2 | 累积缓冲阈值 |

### 5.5 L1 决策层配置 (v5.1.7)

| 配置 | 默认值 | 说明 |
|------|--------|------|
| L1_BUCKET_ENABLED | True | L1 图桶化 |
| L1_BUCKET_BY | "label" | 桶化依据 (label/specialization) |
| L1_CHAIN_ENABLED | True | L1 链条化 |
| L1_CHAIN_SIMILARITY | 0.60 | 链条相似度阈值 |
| L1_DECISION_LAYER | True | L1 决策层总开关 (v5.1.9 重写) |
| L1_FEATURE_LR | 0.3 | L1 特征模板赫布学习率 (v5.1.7-patch) |
| ACTIVE_SET_DECAY | 0.1 | 激活集频次衰减率（通用，适用任意层）(v5.1.7-patch) |
| ACTIVE_SET_KEEP_THRESH | 0.3 | 激活集保留阈值（低于此淘汰）(v5.1.7-patch) |
| FEATURE_VARIANCE_THRESH | 100.0 | 特征方差阈值（低于此降权，通用）(v5.1.7-patch) |

### 5.6 图模式配置

| 配置 | 默认值 | 说明 |
|------|--------|------|
| ENABLE_GRAPH_MODE | False | 图模式总开关 |
| PARALLEL_VIEWS | 6 | 平行视图数 |
| MAX_FEEDBACK_LOOPS | 3 | 最大反馈循环次数 |
| FEEDBACK_THRESHOLD | 85 | 反馈触发阈值 |
| GRAPH_SIMILARITY_THRESHOLD | 80 | 图相似度阈值 |

---

## 6. 扩展机制

### 6.1 事件钩子

```python
# 注册训练事件回调
from engine.hooks import HookRegistry

def on_step_complete(step, info, core):
    print(f"Step {step}: match={info['match']}")

HookRegistry.register("sgn:after_step", on_step_complete)
```

### 6.2 自定义噪声模型

```python
from engine.input import NoiseModel

class MyNoise(NoiseModel):
    def apply(self, base_pattern: List[int]) -> List[int]:
        return [self._clamp(v + random.randint(-10, 10)) for v in base_pattern]
```

### 6.3 自定义策略

```python
from engine.strategies import LayerStrategy

class MyLayerStrategy(LayerStrategy):
    @property
    def applicable_range(self):
        return (64, 0)  # 64×64 以上

    def get_mark_count(self, d, active_pixels):
        return active_pixels * 3 // 4
```

### 6.4 自定义 Level 策略

```python
from engine.level import LevelStrategy, NeuronLevelStats, OperationType

class MyLevelStrategy(LevelStrategy):
    @property
    def name(self): return "my_strategy"
    @property
    def default_level(self): return 1

    def get_level_for_operation(self, operation, neuron_id, stats):
        if operation == OperationType.ADD:
            return 1  # 加法用细粒度
        return 2      # 其他用粗粒度

# 注入调度器
core.level_scheduler.register_strategy(MyLevelStrategy())
```

---

## 7. 文件依赖图

```
run.py
  └── app/main.py
        ├── engine/config.py (CONFIG, DiscreteCoordinate, ConfigRegistry, CharRegistry)
        ├── engine/core.py (SGNCore)
        │     ├── engine/graph_train.py   (图模式训练)
        │     ├── engine/multi_layer_train.py (多层+分批训练)
        │     ├── engine/l0_peer_compare.py (L0 同级比较合并)
        │     ├── engine/l1_decision.py  (L1 桶化/决策层/聚类/回馈)
        │     ├── engine/level.py        (LevelScheduler, LevelStrategy)
        │     ├── engine/level_utils.py  (Level 辅助函数)
        │     ├── graph/graph.py         (DynamicGraph, GraphNode)
        │     ├── graph/stack.py         (图构建与投影)
        │     ├── graph/merge.py         (跨图合并)
        │     └── graph/graph_match.py   (图匹配与推理)
        ├── engine/hooks.py (HookRegistry)
        ├── engine/input.py (InputSource, NoiseModel, FeatureExtractor)
        ├── engine/layers.py (extract_layers, classify_sample)
        ├── engine/strategies.py (LayerStrategy, VerifyStrategy)
        ├── engine/utils.py (C, box, hr, popcount)
        ├── app/interactive.py (menu)
        │     ├── app/panel.py     (控制面板)
        │     ├── app/help.py      (帮助手册)
        │     └── app/blackbox.py  (黑箱验证)
        ├── app/training.py (run_training_loop)
        ├── app/test.py (do_batch_test, do_confusion, do_noise_test)
        ├── app/visual.py (do_stats, do_gauge, do_visualize)
        ├── app/report.py (do_plot)
        │     └── app/backends.py (ChartBackend 多后端)
        ├── app/metrics.py (Metric 抽象体系)
        ├── app/draw.py (网格绘制)
        ├── app/storage.py (StorageBackend: JSON/SQLite)
        │     └── app/persist.py (save_model, load_model)
        ├── app/commands.py (CommandRegistry)
        └── app/cmd_registry.py (21 个命令回调)
```

---

## 8. 版本演进

| 版本 | 特性 |
|------|------|
| v4.3 | 长周期参数重构、Bug修复、8x8标准字符库 |
| v4.4 | 双图叠加门控识别、边缘提取、分块编码 |
| v5.0 | 图模式层级记忆 (graph/stack/merge/graph_match) |
| v5.1 | catear→arch 语义修正、核心引擎性能优化 |
| v5.1.1 | 函数工厂 GUI (pygame) |
| v5.1.2 | GUI 桥接层解耦 |
| v5.1.3 | 多层神经元架构 — Layer 0/1 双层竞争、图中间层汇总器、软门控 |
| **v5.1.5** | **Level 调度器** — 运算精度语境管理、策略可热替换、自适应 level 调整、性能缓存 |
| v5.1.5-fix | 配置绝对路径、神经元三方同步、架构重建精确追踪、二值化显示修复 |
| **v5.1.6** (内部) | **8 个多层 Bug 修复** + **旧门控→自适应随机静默** + **分批次训练** + **L0 同级比较与合并** |
| **v5.1.7** | **core.py 拆分为 5 模块** + **L1 桶化/链条化/决策层/涌现分类/P2纠错** + **base_index 死代码清理** + **_WeakCallback 边界修复** |
| **v5.1.7-patch** | **信号链死锁修复**（`compute_l1_predicted_label` 打破冷启动死锁）+ **特征质量修复**（相对阈值/激活集衰减/专精累计计数/方差降权） |
| **v5.1.7-patch2** | **多层测试与分类一致性修复**：统一 `classify_multi_layer` 分类逻辑、删除 `_graph_features` 隐式注入、`compute_l1_predicted_label` 返回 `(label, score)`、新建 `classify_utils.py` 按模式感知阈值（single/graph/multi = 80/80/30），替换 6 处硬编码 `>= 80` |
| **v5.1.7-patch3** | **桶化数据污染修复**：L0 回归纯特征检测器，桶化按样本标签分桶，切断"5 的特征被分到 1 的桶"污染链 |
| **v5.1.8** | **数据污染与错位综合修复**：`hebbian_merge` 标签守卫、`primary_graph_features` 回退删除、L0 specialization 永禁；删除共享 L0 掩码特征（18→10维）、推理统一静默、L1 specialization 退出机制、维度变更警告 |

> **版本公开说明**：v5.1.6 为内部开发版本。v5.1.7 是首个公开版本，在 v5.1.6 全部功能基础上完成技术债清理和模块化重构。v5.1.8 是当前最新版本，在前序全部修复基础上完成数据污染系统性修复。

---

## 9. 运行模式

| 模式 | 说明 |
|------|------|
| full | 全记载，每步输出 (默认) |
| compact | 精简，仅检查点摘要 |
| blackbox | 黑箱，训练全程零输出 |

## 10. 输入源

v5.1.5 将输入源从 4×4 硬编码升级为插件化 8×8：
- 4×4 模式（历史遗留，已移除）
- **8×8 标准字符** (CharRegistry: 0-9 A-Z 共 36 字符)
- 矢量图形 (line/circle/sine/arch/leaf/mixed 支持 8/16/32/64 网格)
- 文件加载 (CSV/JSON)
- **自定义训练集** (函数工厂 GUI 生成的 JSON)

---

## 11. 模块化引擎 (v5.1.7 拆分参照)

```
engine/
├── core.py              ← 骨架 (~1100行)：__init__ / create_neuron / 核心方法 / train()
├── graph_train.py       ← 图模式训练 (~270行)：反馈循环 / 重建 / 误差 / 下压 / 视图
├── multi_layer_train.py ← 多层+分批训练 (~395行)：L0竞争 / L1阶段 / batch / pool
├── l0_peer_compare.py   ← L0 同级比较 (~195行)：模板相似度 / 投票合并 / 累积缓冲
├── l1_decision.py       ← L1 决策层 (~455行)：桶化 / 链条 / 输出模式 / 聚类 / 回馈 / P2 / 匹配
├── config.py            ← 配置管理 (~1198行)：ConfigRegistry / DiscreteCoordinate / CharRegistry
├── level.py             ← Level 调度器 (~707行)：LevelScheduler / 3 种策略 / 缓存优化
├── level_utils.py       ← Level 辅助 (~126行)：LevelHelper / 便捷函数
├── hooks.py             ← 事件总线 (~275行)：HookRegistry / _WeakCallback
├── strategies.py        ← 策略抽象 (~261行)：LayerStrategy / VerifyStrategy / MatchStrategy
├── layers.py            ← 层处理 (~423行)：popcount / match_bits / extract_layers / classify
├── input.py             ← 输入管道 (~883行)：噪声模型 / 输入源 / 特征提取
└── utils.py             ← 工具函数 (~500行)：颜色输出 / 日志 / 交叉验证
```

---

## 12. 已删除/废弃的配置 (迁移指南)

| 旧配置 (v5.1.3) | 状态 (v5.1.7) | 替代 |
|------------------|---------------|------|
| `ENABLE_GATE_MATCHING` | ❌ 已删除 | `ENABLE_ADAPTIVE_SILENCE` |
| `ENABLE_SOFT_GATE` | ❌ 已删除 | 同上 |
| `GATE_HIGH_THRESH` | ❌ 已删除 | 无（随机机制不需要阈值） |
| `GATE_LOW_THRESH` | ❌ 已删除 | 无 |
| `GATE_DECAY_RATE` | ❌ 已删除 | 无 |
| `GATE_SPECIALIZE_THRESHOLD` | ❌ 已删除 | `SILENCE_SPECIALIZE_THRESHOLD` |
| 4×4 硬编码字符 | ❌ 已移除 | CharRegistry 8×8 插件化 |

---

## 13. 术语表

| 术语 | 定义 |
|------|------|
| 神经元 (Neuron) | 竞争单元，持有模板向量和基线响应速度 |
| 模板 (Template) | 校验通过后的特征签名，用于后续匹配 |
| 离散坐标 (DiscreteCoordinate) | 浮点空间一次性投影到整数格点的坐标系统 |
| 层级 (Level) | 离散坐标精度层级，level=0 为整数空间；也指运算精度语境 (v5.1.5) |
| 竞争 (Competition) | 多神经元对同一输入计算响应速度，取 Top-K |
| 赫布学习 (Hebbian Learning) | 验证通过增强，验证失败削弱 |
| 锁定 (Lockout) | 累积失败达阈值后神经元停止参与竞争 |
| 图模式 (Graph Mode) | 在神经元之上挂接层级化记忆系统 |
| 多层神经元 (Multi-Layer Neuron) | Layer 0 基础特征 + Layer 1 概念判断的双层架构 |
| 自适应随机静默 (Adaptive Silence) | 每步随机静默一定比例神经元，类 Dropout 正则化 (v5.1.6) |
| 涌现分类 (Emergent Classification) | 分类由 L1 神经元输出模式自组织聚类产生 (v5.1.7) |
| 同级比较 (Peer Compare) | 同层神经元之间比较模板/输出模式，合并或聚类 (v5.1.6) |
| Level 调度器 (Level Scheduler) | 管理运算精度语境的调度系统 (v5.1.5) |
| 先验机 (A Priori Machine) | 项目当前阶段——正在完善中的原型机，尚未达到完整原型机阶段 |

---

## 附录: DiscreteCoordinate level 详解

level 是 SGN 离散坐标系统的核心概念，代表精度层级。

**层级含义**:
| Level | Scale | 格点间距 |
|-------|-------|----------|
| 0 | 1 | 1.0 |
| 1 | 10 | 0.1 |
| 2 | 100 | 0.01 |
| 3 | 1000 | 0.001 |

**从浮点数投影**:
```python
@classmethod
def from_float(cls, f: float) -> "DiscreteCoordinate":
    # 0.02  → level=2, index=2  (2位小数)
    # 0.002 → level=3, index=2  (3位小数)
    # 2.0   → level=0, index=2  (0位小数)
```

**层级映射**:
```python
dc = DiscreteCoordinate(35, 2)    # level=2, index=35 (表示0.35)
dc.coarse_to(3)                   # → level=3, index=350 (表示0.350)
dc.fine_to(2)                     # → level=2, index=35 (截断低位)
dc.to_level(target_level)         # 自动映射到目标层级
```

**核心约束**：同一层级才能运算：
```python
def __add__(self, other):
    a, b, level = self._ensure_same_level(other)  # 必须同层
    return DiscreteCoordinate(a + b, level)
```

**为什么需要 level**？
| 场景 | level 作用 |
|------|-----------|
| 学习率 (LEARNING_RATE) | 控制参数调整精度 |
| 基线响应 (base) | 神经元响应速度的精度层级 |
| 坐标 (position_norm) | 归一化位置的精度 |

**设计哲学**：浮点值进入系统时只做一次性投影，之后永不还原为 float，所有运算都是同一层级下的整数算术。
