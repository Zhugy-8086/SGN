# Level 调度器 - SGN v5.1.5 设计文档

## 核心思路

把 level 从"数值的属性"变成"运算的语境"。

**当前设计**：每个 `DiscreteCoordinate` 存 `(index, level)`，运算前必须先对齐。
**新方案**：数值只存纯整数 index，level 由外部调度器提供。

---

## 一、核心组件

### 1.1 运算类型枚举

```python
class OperationType(Enum):
    ADD = "add"          # 赫布学习增强
    SUB = "sub"          # 赫布学习削弱
    MUL = "mul"          # 响应速度计算
    COMPARE = "compare"  # 排序/匹配
    ASSIGN = "assign"    # 初始化/重置
```

### 1.2 运算上下文

```python
@dataclass
class LevelContext:
    target_level: int              # 目标层级
    operation: OperationType       # 运算类型
    source: str = ""               # 来源标识（调试用）
```

### 1.3 策略接口

```python
class LevelStrategy(ABC):
    """level 策略接口 - 可热替换"""

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    @property
    @abstractmethod
    def default_level(self) -> int:
        """默认 level"""
        pass

    @abstractmethod
    def get_level_for_operation(
        self,
        operation: OperationType,
        neuron_id: Optional[int] = None,
        stats: Optional[NeuronLevelStats] = None
    ) -> int:
        """获取指定运算的 level"""
        pass

    def suggest_adaptation(self, stats: NeuronLevelStats) -> Optional[int]:
        """根据统计建议新的 level（可选实现）"""
        return None
```

### 1.4 神经元统计

```python
@dataclass
class NeuronLevelStats:
    neuron_id: int
    current_level: int = 2
    match_history: List[int] = field(default_factory=list)
    verified_count: int = 0
    total_count: int = 0
    level_change_count: int = 0

    @property
    def match_variance(self) -> float:
        """匹配值方差（用于自适应判断）"""
        if len(self.match_history) < 2:
            return 0.0
        mean = sum(self.match_history) / len(self.match_history)
        return sum((x - mean) ** 2 for x in self.match_history) / len(self.match_history)

    @property
    def verification_rate(self) -> float:
        """验证通过率"""
        return self.verified_count / max(1, self.total_count)

    def update(self, match: int, verified: bool) -> None:
        self.last_match = match
        self.total_count += 1
        if verified:
            self.verified_count += 1
        self.match_history.append(match)
        if len(self.match_history) > 100:
            self.match_history.pop(0)
```

### 1.5 调度器核心

```python
class LevelScheduler:
    """level 调度器 - 管理运算精度语境"""

    def __init__(self):
        self._strategies: Dict[str, LevelStrategy] = {}
        self._neuron_strategy: Dict[int, str] = {}
        self._neuron_stats: Dict[int, NeuronLevelStats] = {}
        self._default_strategy: Optional[LevelStrategy] = None
        self._adapt_interval: int = 100
        self._step_counter: int = 0

    # ---- 核心接口 ----

    def get_context(
        self,
        operation: OperationType,
        neuron_id: Optional[int] = None,
        source: str = ""
    ) -> LevelContext:
        """获取运算上下文"""
        strategy = self._get_strategy_for_neuron(neuron_id)
        stats = self._neuron_stats.get(neuron_id)
        level = strategy.get_level_for_operation(operation, neuron_id, stats)
        return LevelContext(target_level=level, operation=operation, source=source)

    def resolve_binary_op(
        self,
        op: OperationType,
        left_level: int,
        right_level: int,
        neuron_id: Optional[int] = None
    ) -> int:
        """二元运算的 level 决策"""
        strategy = self._get_strategy_for_neuron(neuron_id)
        stats = self._neuron_stats.get(neuron_id)
        target = strategy.get_level_for_operation(op, neuron_id, stats)
        # 取更精细的 level（数值更小 = 更精细）
        return min(left_level, right_level, target)

    def update_stats(self, neuron_id: int, match: int, verified: bool) -> None:
        """更新神经元统计"""
        if neuron_id not in self._neuron_stats:
            self._neuron_stats[neuron_id] = NeuronLevelStats(neuron_id=neuron_id)
        stats = self._neuron_stats[neuron_id]
        stats.update(match, verified)
        # 定期检查自适应
        self._step_counter += 1
        if self._step_counter % self._adapt_interval == 0:
            self._check_adaptation(neuron_id)

    def _check_adaptation(self, neuron_id: int) -> None:
        """检查并执行自适应调整"""
        strategy = self._get_strategy_for_neuron(neuron_id)
        stats = self._neuron_stats.get(neuron_id)
        if stats is None:
            return
        suggested = strategy.suggest_adaptation(stats)
        if suggested is not None and suggested != stats.current_level:
            old_level = stats.current_level
            stats.current_level = suggested
            stats.level_change_count += 1
            # 发射事件通知上层
            from sgn_hooks import HookRegistry
            HookRegistry.emit(
                "sgn:level_adapted",
                neuron_id=neuron_id,
                old_level=old_level,
                new_level=suggested,
                variance=stats.match_variance
            )
```

---

## 二、内置策略实现

### 2.1 标准策略

```python
class StandardStrategy(LevelStrategy):
    """标准策略 - 固定 level，不自适应"""

    def __init__(self, level: int = 2):
        self._level = level

    @property
    def name(self) -> str:
        return f"standard(L{self._level})"

    @property
    def default_level(self) -> int:
        return self._level

    def get_level_for_operation(self, operation, neuron_id=None, stats=None) -> int:
        return self._level
```

### 2.2 自适应策略

```python
class AdaptiveStrategy(LevelStrategy):
    """自适应策略 - 根据神经元统计动态调整 level

    规则：
      - 匹配值方差 < 阈值 → 细粒度 level（更精确）
      - 匹配值方差 > 阈值 → 粗粒度 level（更稳定）
    """

    def __init__(self, base_level: int = 2, variance_threshold: float = 100.0):
        self._base_level = base_level
        self._variance_threshold = variance_threshold

    def get_level_for_operation(self, operation, neuron_id=None, stats=None) -> int:
        if stats is None:
            return self._base_level
        return stats.current_level

    def suggest_adaptation(self, stats: NeuronLevelStats) -> Optional[int]:
        if len(stats.match_history) < 50:
            return None

        variance = stats.match_variance
        current = stats.current_level

        # 方差小 → 细粒度（level 增大）
        if variance < self._variance_threshold / 4:
            return min(current + 1, 4)

        # 方差大 → 粗粒度（level 减小）
        if variance > self._variance_threshold * 4:
            return max(current - 1, 0)

        return None
```

### 2.3 层级感知策略

```python
class LayerAwareStrategy(LevelStrategy):
    """层级感知策略 - 根据神经元所在层自动选择 level

    L0 神经元：标准精度（level=2）
    L1 神经元：粗精度（level=1），因为图特征已经过抽象
    """

    def __init__(self):
        self._layer_levels = {0: 2, 1: 1}

    def get_level_for_operation(self, operation, neuron_id=None, stats=None) -> int:
        if stats and hasattr(stats, 'layer'):
            return self._layer_levels.get(stats.layer, 2)
        return 2
```

---

## 三、已修改的文件 [v5.1.5 Phase 1-3]

### 3.1 sgn_level.py (Phase 1 新增)

- LevelScheduler 核心调度器
- LevelStrategy 策略接口
- StandardStrategy / AdaptiveStrategy / LayerAwareStrategy
- NeuronLevelStats 神经元统计
- 便捷函数：get_level_for_add(), update_neuron_stats() 等

### 3.2 sgn_core.py (Phase 2 修改)

| 修改点 | 原实现 | 新实现 |
|--------|--------|--------|
| 导入 | 无 | 导入 sgn_level 模块 |
| create_neuron() | 无 nid 字段 | 添加 nid、base_index、enc_b_index、gate_index |
| __init__() | 无调度器 | 初始化 LevelScheduler，绑定神经元策略 |
| _response_speed() | 硬编码 `base.to_level(gamma.level)` | 委托 `scheduler.get_context(OperationType.MUL, ...)` |
| _hebbian_learn() | 硬编码 `max(lr.level, wr.level, ...)` | 委托 `scheduler.get_context(OperationType.ADD/SUB, ...)` |
| train() | 无统计更新 | 添加 `scheduler.update_stats(i, match, verified)` |
| get_state() | 无 level 信息 | 返回 level_distribution |
| serialize/deserialize | 无 | 新增方法 |

### 3.3 sgn_graph.py (Phase 3 修改)

| 修改点 | 原实现 | 新实现 |
|--------|--------|--------|
| feature_similarity() | 硬编码 `max(all_levels)` | 委托 `scheduler.get_context(OperationType.COMPARE, ...)` |
| hebbian_merge() | 硬编码 `max(target.feature_vector[i].level, ...)` | 委托 `scheduler.get_context(OperationType.ADD, ...)` |

### 3.4 sgn_merge.py (Phase 3 修改)

| 修改点 | 原实现 | 新实现 |
|--------|--------|--------|
| _merge_cluster_nodes() | 硬编码 `max(v.level for v in values)` | 委托 `scheduler.get_context(OperationType.ASSIGN, ...)` |

### 3.5 sgn_level_utils.py (Phase 3 新增)

- LevelHelper 辅助类
- 便捷函数：get_level_for_hebbian(), get_level_for_weaken() 等

---

## 四、测试结果

```
=== 单层模式 ===
Train result: A V: False Match: 65
Level scheduler: True
Level info: {'total_neurons': 256, 'level_distribution': {2: 256}, 'adapted_count': 0}

=== Level Helper ===
Helper info: {'total_neurons': 256, 'level_distribution': {2: 256}, 'adapted_count': 0}

=== 多层模式 ===
Multi-layer: True
L0 active: 128
L1 active: 64
Level distribution: {2: 128, 1: 64}

=== 序列化 ===
Serialized strategies: 192

=== 图模式 ===
Graph mode: True
Graph nodes: 83
```

---

## 五、文件依赖图 [v5.1.5]

```
sgn_core.py
  ├── sgn_level.py (LevelScheduler, OperationType)
  ├── sgn_level_utils.py (LevelHelper)
  ├── sgn_graph.py (feature_similarity, hebbian_merge)
  │     └── sgn_level.py
  ├── sgn_merge.py (_merge_cluster_nodes)
  │     └── sgn_level.py
  └── sgn_config.py (DiscreteCoordinate)

sgn_level.py
  ├── sgn_hooks.py (HookRegistry)
  └── 无其他依赖
```

---

## 六、下一步：Phase 4（可选）

1. 清理旧字段（完全迁移到纯整数存储）
2. HC 存储接入
3. 性能优化（缓存调度器查找结果）
4. 单元测试覆盖

---

## 七、Phase 4 完成情况 [v5.1.5]

### 7.1 性能优化

```python
class LevelScheduler:
    def __init__(self, cache_size: int = 1024):
        self._cache_size = cache_size
        self._context_cache: Dict[Tuple[OperationType, Optional[int]], LevelContext] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def get_context(self, operation, neuron_id, source):
        """带缓存的版本"""
        cache_key = (operation, neuron_id)
        if cache_key in self._context_cache:
            self._cache_hits += 1
            return cached
        # ... 正常逻辑
```

### 7.2 单元测试覆盖

| 测试 | 描述 | 状态 |
|------|------|------|
| Test 1 | 调度器基本功能 | ✅ |
| Test 2 | 自适应策略 | ✅ |
| Test 3 | 二元运算 level 决策 | ✅ |
| Test 4 | 层级感知策略 | ✅ |
| Test 5 | 便捷函数 | ✅ |
| Test 6 | 序列化/反序列化 | ✅ |
| Test 7 | 统计功能 | ✅ |
| Test 8 | 缓存性能 | ✅ |
| Test 9 | 与 SGNCore 集成 | ✅ |
| Test 10 | 多层神经元集成 | ✅ |

### 7.3 测试结果

```
Test Results: 10 passed, 0 failed
```

---

## 八、完成的 Phase 总结

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | 创建 sgn_level.py，调度器接口和策略 | ✅ 完成 |
| Phase 2 | sgn_core.py 委托调度器，神经元结构双写 | ✅ 完成 |
| Phase 3 | 图模块委托调度器，便捷函数模块 | ✅ 完成 |
| Phase 4 | 性能优化（缓存），单元测试覆盖 | ✅ 完成 |

---

## 九、文件依赖图 [v5.1.5 最终版]

```
sgn_core.py
  ├── sgn_level.py (LevelScheduler, OperationType)
  ├── sgn_level_utils.py (LevelHelper)
  ├── sgn_graph.py (feature_similarity, hebbian_merge)
  │     └── sgn_level.py
  ├── sgn_merge.py (_merge_cluster_nodes)
  │     └── sgn_level.py
  └── sgn_config.py (DiscreteCoordinate)

sgn_level.py
  ├── sgn_hooks.py (HookRegistry)
  └── 无其他依赖

测试文件
  ├── test_level_scheduler.py (Phase 1 基础测试)
  └── test_level_complete.py (Phase 4 完整测试套件)
```

---

## 十、核心原则

**level 管"怎么算"，不管"存什么"**
**调度器管"什么时候用什么算"，不管"算的是什么"**

存储和运算分离，策略和数值分离，上层完全无感。

---

## 六、核心原则

**level 管"怎么算"，不管"存什么"**
**调度器管"什么时候用什么算"，不管"算的是什么"**

存储和运算分离，策略和数值分离，上层完全无感。
