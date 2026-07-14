# 神经元自适应随机静默机制 - SGN v5.1.6 设计文档

## 核心思路

把"哪些神经元参与本步竞争"从**固定规则**变成**随机抽样**。

**旧机制（门控）**：每个神经元持有 `gate` 值（0~100），低于阈值的神经元被"软锁"不参与竞争，门控值随验证结果增减。
**新机制（静默）**：每步训练前，从候选池中随机选中 30%~80% 的神经元标记为 `silenced`，被静默的神经元本步不参与竞争、不输出值、不参与判断。

```
旧: gate 值驱动 → 验证通过 gate+1, 验证失败 gate-1 → 低于阈值退出竞争
新: 随机抽样驱动 → 每步重新抽 → 被选中即静默 → 下一步自动复活
```

---

## 一、设计动机

### 1.1 旧门控的根本问题

旧门控（`ENABLE_GATE_MATCHING` + `ENABLE_SOFT_GATE`）存在三个问题：

| 问题 | 表现 |
|------|------|
| **状态固化** | gate 值一旦降到低位，神经元长期退出竞争，难以"复活" |
| **参数耦合** | `GATE_HIGH_THRESH` / `GATE_LOW_THRESH` / `GATE_DECAY_RATE` 三个参数互相影响，调参困难 |
| **与静默语义重叠** | 软门控本质是"部分静默"，但用连续值表达离散行为，名实不符 |

### 1.2 新机制的优势

| 优势 | 说明 |
|------|------|
| **无状态固化** | `silenced` 标志每步重置，被静默的神经元下一步自动复活 |
| **参数极简** | 只有 3 个核心参数（min/max 比例 + 触发概率），语义独立 |
| **与专精分片正交** | 静默是"本步谁休息"，专精是"谁擅长什么"，两者独立 |
| **天然正则化** | 随机静默等价于 Dropout，防止神经元共适应 |

---

## 二、核心配置

### 2.1 配置项

```python
ConfigItem("ENABLE_ADAPTIVE_SILENCE", True, bool, None,
           "启用自适应随机静默（取代旧门控）", "高级选项", True),
ConfigItem("SILENCE_MIN_RATIO", 0.30, float, (0.0, 0.95),
           "静默比例下限", "学习参数", False),
ConfigItem("SILENCE_MAX_RATIO", 0.80, float, (0.05, 1.0),
           "静默比例上限", "学习参数", False),
ConfigItem("SILENCE_TRIGGER_PROB", 0.5, float, (0.0, 1.0),
           "每步触发静默的概率", "学习参数", False),
ConfigItem("SILENCE_SPECIALIZE_THRESHOLD", 10, int, (3, 50),
           "静默专精触发阈值（融合自旧 GATE_SPECIALIZE_THRESHOLD）", "学习参数", False),
ConfigItem("SILENCE_SPECIALIZED_WEIGHT", 0.3, float, (0.0, 1.0),
           "补丁P1:专精神经元静默权重(降低专精被静默概率)", "学习参数", False),
```

### 2.2 参数语义

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `SILENCE_MIN_RATIO` | 0.30 | 每步最少静默 30% 的候选神经元 |
| `SILENCE_MAX_RATIO` | 0.80 | 每步最多静默 80% 的候选神经元 |
| `SILENCE_TRIGGER_PROB` | 0.5 | 50% 的训练步会应用静默，其余 50% 全员参与 |
| `SILENCE_SPECIALIZE_THRESHOLD` | 10 | 连续验证通过 10 次后标记为专精 |
| `SILENCE_SPECIALIZED_WEIGHT` | 0.3 | 专精神经元被选入静默集的概率只有普通的 30% |

### 2.3 已删除的旧配置

以下旧门控配置已从 `engine/config.py` 和 `config/sgn_config.json` 中彻底删除：

- `ENABLE_GATE_MATCHING`（双图叠加门控）
- `GATE_HIGH_THRESH` / `GATE_LOW_THRESH`（门控阈值）
- `GATE_DECAY_RATE`（门控衰减率）
- `ENABLE_SOFT_GATE`（软门控）
- `GATE_SPECIALIZE_THRESHOLD`（门控专精阈值，语义迁移到 `SILENCE_SPECIALIZE_THRESHOLD`）

> **控制面板说明**：菜单从 `ConfigRegistry._schema` 动态生成，schema 中无的配置项不会出现。旧门控选项已从 UI 中彻底消失，不会变成"高级选项"。

---

## 三、神经元字段重构

### 3.1 旧字段（已删除）

```python
"gate": DiscreteCoordinate(100, 2),   # 门控强度 (0~100) — 已删除
"gate_index": 100,                    # 门控索引 — 已删除
```

### 3.2 新字段

```python
"silenced": False,                    # 本步是否被静默（每步重置）
"specialization": None,               # 专精标签（保留，融合自旧字段）
"consecutive_verified": 0,            # 连续验证通过计数（保留）
```

### 3.3 字段语义

| 字段 | 类型 | 生命周期 | 说明 |
|------|------|---------|------|
| `silenced` | bool | 每步重置 | True=本步静默，不参与任何判断 |
| `specialization` | str/None | 永久 | 验证通过累积后赋值，代表神经元专精的标签 |
| `consecutive_verified` | int | 可重置 | 连续验证通过次数，达阈值后触发专精 |

---

## 四、核心实现

### 4.1 静默机制主方法

```python
def _apply_adaptive_silence(self, pool: List[Dict]) -> List[int]:
    """v5.1.6 自适应随机静默

    每步训练前调用，从 pool 中随机选中 30%~80% 的神经元标记为 silenced。
    被静默的神经元本步不参与竞争、不输出值、不参与判断。

    Returns:
        active_indices: 未被静默的神经元索引列表
    """
    # 1. 总开关
    if not CONFIG.get("ENABLE_ADAPTIVE_SILENCE", True):
        return [i for i, n in enumerate(pool) if not n["L"]]

    # 2. 概率触发：不是每步都静默
    if random.random() > CONFIG.get("SILENCE_TRIGGER_PROB", 0.5):
        for n in pool:
            n["silenced"] = False
        return [i for i, n in enumerate(pool) if not n["L"]]

    # 3. 计算本步静默比例（在 min~max 区间内随机）
    min_r = CONFIG.get("SILENCE_MIN_RATIO", 0.30)
    max_r = CONFIG.get("SILENCE_MAX_RATIO", 0.80)
    ratio = min_r + random.random() * (max_r - min_r)

    # 4. 候选池：未锁定的神经元
    candidates = [i for i, n in enumerate(pool) if not n["L"]]
    if not candidates:
        return []

    # 5. 补丁 P1：专精神经元静默权重降低
    specialized_weight = CONFIG.get("SILENCE_SPECIALIZED_WEIGHT", 0.3)
    weighted_candidates = []
    for i in candidates:
        n = pool[i]
        if n.get("specialization") is not None:
            # 专精神经元：以降低后的权重参与静默抽样
            if random.random() < specialized_weight:
                weighted_candidates.append(i)
        else:
            weighted_candidates.append(i)

    # 6. 随机选中静默集
    silence_count = int(len(candidates) * ratio)
    silenced_set = set(random.sample(
        weighted_candidates,
        min(silence_count, len(weighted_candidates))
    )) if weighted_candidates else set()

    # 7. 标记并返回活跃集
    active = []
    for i in candidates:
        pool[i]["silenced"] = i in silenced_set
        if i not in silenced_set:
            active.append(i)

    return active
```

### 4.2 竞争循环改造

```python
# 旧代码：所有未锁定神经元参与竞争
for i, n in enumerate(self.N):
    if n["L"]:
        continue
    match = self._match(n, layers, layer_count, intensity)
    ...

# 新代码：只让未被静默的神经元参与竞争
active_indices = self._apply_adaptive_silence(self.N)
for i in active_indices:
    n = self.N[i]
    if n["L"] or n.get("silenced", False):
        continue
    match = self._match(n, layers, layer_count, intensity)
    ...
```

---

## 五、补丁 P1：静默豁免

### 5.1 问题

专精神经元的 `specialization` 由 `consecutive_verified` 累积到阈值（默认 10）触发。如果专精神经元被频繁静默，它就难以连续验证通过，`consecutive_verified` 永远达不到阈值，专精状态永远无法收敛。

### 5.2 方案

专精神经元被选入静默候选集的概率降低到普通神经元的 30%（`SILENCE_SPECIALIZED_WEIGHT=0.3`）：

```python
if n.get("specialization") is not None:
    # 专精神经元：只有 30% 概率进入静默候选集
    if random.random() < specialized_weight:
        weighted_candidates.append(i)
else:
    # 普通神经元：100% 进入候选集
    weighted_candidates.append(i)
```

### 5.3 效果

- 普通神经元：静默概率 ≈ ratio（30%~80%）
- 专精神经元：静默概率 ≈ ratio × 0.3（9%~24%）
- 专精状态可以稳定累积，不会被静默打断

---

## 六、专精分片融合

### 6.1 旧软门控的专精逻辑

```python
# 旧代码：gate 值增减驱动专精
if CONFIG.get("ENABLE_SOFT_GATE", False):
    if n["specialization"] is None:
        n["consecutive_verified"] += 1
        if n["consecutive_verified"] >= CONFIG.get("GATE_SPECIALIZE_THRESHOLD", 10):
            n["specialization"] = verified
    elif n.get("specialization") == verified:
        new_gate = min(100, n["gate"].index + CONFIG.get("GATE_DECAY_RATE", 1))
        n["gate"] = DiscreteCoordinate(new_gate, n["gate"].level)
```

### 6.2 新静默机制的专精逻辑

```python
# 新代码：专精状态由静默随机性自然维护
if CONFIG.get("ENABLE_ADAPTIVE_SILENCE", True):
    if n["specialization"] is None:
        n["consecutive_verified"] += 1
        if n["consecutive_verified"] >= CONFIG.get("SILENCE_SPECIALIZE_THRESHOLD", 10):
            n["specialization"] = verified if isinstance(verified, str) else "unknown"
    # 旧 gate 值增减逻辑删除
```

### 6.3 关键差异

| 维度 | 旧软门控 | 新静默机制 |
|------|---------|-----------|
| 专精触发 | `consecutive_verified` ≥ 阈值 | 相同（阈值参数改名） |
| 专精维护 | gate 值增减，专精错则 gate 衰减 | 静默豁免（P1），专精不易被静默 |
| 退出专精 | gate 降到 0 | 无显式退出（由静默随机性自然淘汰） |

---

## 七、多层模式同步

静默机制同时应用于 Layer 0 和 Layer 1：

```python
# Layer 0 竞争应用静默
active_l0 = self._apply_adaptive_silence(layer0_pool)
for i in active_l0:
    n = layer0_pool[i]
    if n["L"] or n.get("silenced", False):
        continue
    match = self._match(n, layers, layer_count, intensity)
    ...

# Layer 1 竞争应用静默
active_l1 = self._apply_adaptive_silence(layer1_pool)
for i in active_l1:
    n = layer1_pool[i]
    if n["L"] or n.get("silenced", False):
        continue
    match_l1 = self._match_layer1(n, graph_features, l0_active_list)
    ...
```

每步训练结束时，所有神经元的 `silenced` 标志会被重置（下一步重新抽样）。

---

## 八、静默 vs 旧门控对比

| 维度 | 旧门控 | 新静默 |
|------|--------|--------|
| **决策方式** | gate 值连续增减，低于阈值退出 | 每步随机抽样，被选中即静默 |
| **状态生命周期** | 永久累积（gate 值不重置） | 每步重置（silenced 标志） |
| **复活机制** | 无（gate 降到底基本永久退出） | 天然复活（下一步重新抽样） |
| **参数数量** | 5 个（互相耦合） | 3 个核心 + 2 个专精（语义独立） |
| **正则化效果** | 无 | 类 Dropout，防共适应 |
| **专精维护** | gate 值增减 | 静默豁免（P1） |
| **可解释性** | gate 值含义模糊 | "本步 X% 神经元休息"，直观 |

---

## 九、配置调优指南

### 9.1 默认值适用场景

默认配置（min=0.3, max=0.8, trigger=0.5）适用于：
- 中等规模网络（128~512 神经元）
- 标准训练步数（5000~10000）

### 9.2 调优建议

| 场景 | 建议调整 | 原因 |
|------|---------|------|
| 网络过小（<64 神经元） | 降低 max_ratio 到 0.5 | 静默太多导致竞争不足 |
| 网络过大（>1024 神经元） | 提高 min_ratio 到 0.5 | 大网络需要更强正则化 |
| 训练不稳定 | 降低 trigger_prob 到 0.3 | 减少静默频率，稳定学习 |
| 过拟合 | 提高 max_ratio 到 0.9 | 增强正则化 |
| 专精无法收敛 | 降低 specialized_weight 到 0.1 | 进一步保护专精神经元 |

### 9.3 关闭静默

```python
# 完全关闭静默，回到"全员参与"模式
CONFIG["ENABLE_ADAPTIVE_SILENCE"] = False
```

关闭后所有神经元每步都参与竞争，等价于 v4.4 的无门控行为。

---

## 十、关联文件

| 文件 | 改动 |
|------|------|
| `engine/config.py` | 删除 6 个旧 gate 配置，新增 6 个 silence 配置 |
| `engine/core.py` | `create_neuron` 删除 gate 字段新增 silenced；新增 `_apply_adaptive_silence`；改造 `train` / `_train_multi_layer` 竞争循环 |
| `engine/strategies.py` | 删除 `GateMatchStrategy` |
| `app/storage.py` | 序列化删除 gate/gate_index，新增 silenced/specialization |
| `app/panel.py` | 帮助文档从门控说明改为静默说明 |

---

## 十一、术语表

| 术语 | 定义 |
|------|------|
| 静默 (Silence) | 神经元本步不参与竞争、不输出值、不参与判断 |
| 静默比例 (Silence Ratio) | 本步被静默的神经元占候选池的比例 |
| 触发概率 (Trigger Prob) | 每步应用静默的概率（其余步全员参与） |
| 专精豁免 (Specialized Exemption) | 专精神经元被选入静默集的概率降低 |
| 候选池 (Candidate Pool) | 未锁定（`L=False`）的神经元集合，静默抽样从中抽取 |
