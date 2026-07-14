# L1 决策层与自组织分类 - SGN v5.1.6 设计文档

> **⚠️ 已废弃 (v5.1.9)**：本文档描述的"输出模式向量 + 同级比较聚类 + 回馈图模式 + 纠错机制"已在 v5.1.9 全部删除。L1 决策层已重写为"72 维分布直方图特征匹配 + 赢家衰减/输家复活 + Top-2 跨类平均分类"。
>
> **当前实现**：详见 [`fixes_相关修复/v5.1.9.md`](../fixes_相关修复/v5.1.9.md) 和 [`docs/ARCHITECTURE_v5.1.7.md`](ARCHITECTURE_v5.1.7.md) §3.7。
>
> 以下内容仅作历史参考。

## 核心思路

把"分类"从**外部分类器代码遍历标签**变成**神经元输出结果自组织涌现**。

**旧机制（外部分类）**：`classify_multi_layer` 遍历 `core.graphs`，逐标签计算 L1 匹配度，取最高分作为预测标签。神经元只是被动产生匹配度数字，分类逻辑全部在外部代码里。

**新机制（涌现分类）**：L1 神经元输出"判断模式向量"，输出相似的 L1 神经元自然聚成一类，这类对应的标签就是预测结果——不是被代码分配的，而是涌现的。

```
旧: 外部代码遍历标签 → 逐个算匹配度 → 取最高分 = 预测标签
新: L1 输出判断模式 → 同级比较聚类 → 簇标签 = 涌现分类
```

---

## 一、设计动机

### 1.1 旧机制的根本矛盾

当前 SGN 的"分类"本质上是**物理分类**——靠外层代码遍历 `core.templates` / `core.graphs`，逐个计算匹配度，取最高分作为预测标签。神经元本身**不参与分类决策**，它们只是产生匹配度数字，分类逻辑全部在外部代码里。

这带来一个根本矛盾：**神经元学了东西，但分类不是神经元做的**。神经元是被动特征库，分类器是主动判断者——这与"神经元网络自组织学习"的初衷相悖。

### 1.2 新机制的本质

> **分类不依靠其他代码进行物理分类，而是依靠神经元的输出结果对这个结果进行分类。输出结果与什么答案的回馈相似度一样，就归为那一类。**

翻译成机制语言：

1. **L1 神经元不输出标签，输出"判断模式"**：L1 的输出是一个向量，代表它对当前输入的判断倾向
2. **L1 同级比较**：与 L0 同级比较对称——L0 比较的是**输入端模板 T**，L1 比较的是**输出端判断模式**
3. **输出相似的 L1 神经元自然聚成一类**：这类对应的就是某个标签，涌现而非分配
4. **正确判断时 L1 回馈调整图模式**：图模式是 L1 的输入中间层，L1 的回馈塑造图模式的演化方向

### 1.3 L0 与 L1 同级比较的对称关系

| 维度 | L0 同级比较 | L1 同级比较 |
|------|-------------------|-------------------|
| 比较对象 | 输入端模板 `T` | 输出端判断模式 |
| 比较时机 | 批次内 L0 竞争后 | 批次内 L1 输出后 |
| 比较目的 | 合并冗余模板，过滤噪点 | 聚类输出模式，涌现分类 |
| 合并/聚类结果 | 相似模板合并为一个 | 相似输出归为同一标签类 |
| 谁驱动 | 模板相似度 | 输出模式相似度 + 正确回馈 |

**关键洞察**：L0 的同级比较让"输入特征"自组织合并；L1 的同级比较让"输出判断"自组织分类。两者**同时进行**，从输入端和输出端双向收敛。

---

## 二、L1 的角色定位：决策层

L1 是当前架构的**决策层/最后一层**。它的职责：

1. 读取图模式（图模式是 L1 的输入，也是 L0→L1 之间的中间层）
2. 输出判断模式（不是标签，是判断倾向向量）
3. 判断被验证为正确时，**回馈调整图模式**
4. 参与 L1 同级比较，输出相似的神经元聚成一类

```
输入 → L0(同级比较合并) → 图模式(中间层) → L1(决策层)
                                            ↓ 输出判断模式
                                       L1 同级比较
                                            ↓ 相似输出聚类
                                       涌现分类 = 标签
                                            ↓ 判断正确
                                       L1 回馈 → 调整图模式
```

### 2.1 图模式的定位：中间层节点

**当前**：图模式是 L0 → L1 之间的唯一桥梁。
**未来**：图模式只是多层叠加结构中的一个中间节点。

```
当前（双层）:
  L0 → 图 → L1(决策)

未来（多层叠加）:
  L0 → 图1 → L1 → 图2 → L2 → 图3 → L3(决策)
        ↑                    ↑
     中间节点              中间节点
```

设计原则：
- 图模式不绑定于"L0→L1"这一层，而是**可复用的中间节点**
- 每一层图模式都可以被下一层读取、被上一层回馈调整
- L1 回馈图模式的机制，未来 L2 回馈图2、L3 回馈图3 都用同一套接口
- `DynamicGraph` 作为独立的中间层抽象，不硬编码层数

---

## 三、L1 图模式桶化与链条化（3.3）

### 3.1 问题

当前 `project_neurons_to_graph` 把所有 winner 神经元一股脑装进同一个图，导致：
- 不同标签的图特征互相污染（数据串位）
- L1 神经元无法区分特征来源

### 3.2 桶化（Bucketing）

按标签/特征类别把 winner 分到不同的桶（子图），每个桶独立维护特征：

```
批次内 winners_l0（已合并）
      ↓ 按标签分桶
桶 A (label="圆")  桶 B (label="线")  桶 C (label="?")
  ├─ winner 1         ├─ winner 3        ├─ winner 5
  ├─ winner 2         ├─ winner 4        └─ winner 6
  ↓ 独立投影          ↓ 独立投影         ↓ 独立投影
  graph["圆"]         graph["线"]        graph["?"]
  (只含圆特征)        (只含线特征)       (无标签特征)
```

### 3.3 链条化（Chaining）

桶内 winner 按模板相似度串成链条，保留特征演化路径：

1. 以第一个 winner 为链头
2. 依次找与当前链尾相似度 >= `L1_CHAIN_SIMILARITY` 的 winner 接上
3. 不相似的 winner 另起一条链（追加到末尾）

```
桶 A 内: winner1 → winner2 (相似度0.85) → winner5 (相似度0.72)
                                          ↑ 链条保留特征演化路径
```

### 3.4 配置项

```python
ConfigItem("L1_BUCKET_ENABLED", True, bool, None,
           "启用L1图桶化(防数据串位)", "学习参数", False),
ConfigItem("L1_BUCKET_BY", "label", str, None,
           "分桶依据(label/specialization)", "学习参数", False),
ConfigItem("L1_CHAIN_ENABLED", True, bool, None,
           "启用桶内链条化", "学习参数", False),
ConfigItem("L1_CHAIN_SIMILARITY", 0.60, float, (0.3, 0.95),
           "链条连接相似度阈值", "学习参数", False),
```

### 3.5 分桶实现

```python
def _bucket_winners_by_label(self, winners, pool, sample) -> Dict[str, List[Dict]]:
    """按标签/专精分桶

    - 有专精标签的神经元 → 对应标签桶
    - 无专精标签的神经元 → 当前样本标签桶
    - 无法归类 → "?" 通用桶
    """
    bucket_by = CONFIG.get("L1_BUCKET_BY", "label")
    sample_label = sample[1] if sample else "?"
    buckets = {}

    for w in winners:
        n = pool[w["nid"]]
        if bucket_by == "specialization" and n.get("specialization"):
            key = n["specialization"]
        elif bucket_by == "label":
            key = n.get("specialization") or sample_label
        else:
            key = "?"
        buckets.setdefault(key, []).append(w)

    return buckets
```

### 3.6 链条化实现

```python
def _chain_winners(self, winners, pool) -> List[Dict]:
    """桶内链条化

    以第一个 winner 为链头，依次找与当前链尾相似度 >= 阈值的 winner 接上。
    """
    if len(winners) <= 1:
        return winners

    chain_sim = CONFIG.get("L1_CHAIN_SIMILARITY", 0.60)
    chained = [winners[0]]
    remaining = list(winners[1:])

    while remaining:
        current = chained[-1]
        best_idx, best_sim = -1, -1.0
        for i, w in enumerate(remaining):
            sim = self._template_similarity(
                pool[current["nid"]], pool[w["nid"]]
            )
            if sim >= chain_sim and sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0:
            chained.append(remaining.pop(best_idx))
        else:
            chained.append(remaining.pop(0))

    return chained
```

---

## 四、L1 输出模式与决策层（3.4）

### 4.1 L1 神经元新增字段

```python
# v5.1.6 L1 决策层字段
"output_pattern": [],          # 输出判断模式向量
"output_history": [],          # 近 N 次输出模式历史
"verified_count": 0,           # 该输出模式被验证正确的次数
"cluster_id": None,            # 所属聚类 ID（涌现分类）
# 补丁 P2（纠错机制）：簇置信度计数器
"cluster_verified_count": 0,   # 所属簇判断正确次数
"cluster_total_count": 0,      # 所属簇参与判断总次数
```

### 4.2 输出模式生成

L1 匹配图特征后，不只是返回一个 score，而是生成一个**判断模式向量**，代表"L1 认为这个输入长什么样"：

```python
def _l1_generate_output_pattern(self, n, graph_features, l0_active_list) -> List[float]:
    """生成输出判断模式

    输出模式 = 图特征与 L1 记忆模板的逐位相似度（归一化到 0~1）。
    """
    dim = CONFIG.get("L1_OUTPUT_PATTERN_DIM", 64)
    if not n.get("T_features_initialized"):
        return [0.0] * dim
    pattern = []
    t_feat = n.get("T_features", [])
    for i in range(dim):
        if i < len(t_feat) and i < len(graph_features):
            t_val = t_feat[i].index if hasattr(t_feat[i], 'index') else t_feat[i]
            g_val = graph_features[i].index if hasattr(graph_features[i], 'index') else graph_features[i]
            diff = abs(t_val - g_val)
            pattern.append(1.0 - min(1.0, diff / 100.0))
        else:
            pattern.append(0.0)
    return pattern
```

### 4.3 输出模式相似度

使用**余弦相似度**衡量判断方向的一致性（与 L0 用的汉明相似度不同，不可混用）：

```python
def _pattern_similarity(self, p1, p2) -> float:
    """计算两个输出模式的余弦相似度 (0.0~1.0)"""
    if not p1 or not p2 or len(p1) != len(p2):
        return 0.0
    dot = sum(a * b for a, b in zip(p1, p2))
    norm1 = sum(a * a for a in p1) ** 0.5
    norm2 = sum(b * b for b in p2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
```

### 4.4 配置项

```python
ConfigItem("L1_DECISION_LAYER", True, bool, None,
           "启用L1决策层(输出模式自组织分类)", "学习参数", True),
ConfigItem("L1_OUTPUT_PATTERN_DIM", 64, int, (8, 512),
           "L1输出模式向量维度", "学习参数", False),
ConfigItem("L1_PEER_COMPARE_ENABLED", True, bool, None,
           "启用L1同级比较(输出相似度聚类)", "学习参数", False),
ConfigItem("L1_CLUSTER_SIMILARITY", 0.70, float, (0.4, 0.99),
           "L1输出聚类相似度阈值", "学习参数", False),
ConfigItem("L1_FEEDBACK_TO_GRAPH", True, bool, None,
           "判断正确时L1回馈调整图模式", "学习参数", False),
ConfigItem("L1_FEEDBACK_STRENGTH", 0.1, float, (0.01, 1.0),
           "L1回馈图模式的强度", "学习参数", False),
```

---

## 五、L1 同级比较与自组织聚类

### 5.1 聚类算法

对批次内所有 L1 神经元的输出模式做横向比较：

1. 输出模式相似度 >= `L1_CLUSTER_SIMILARITY` 的归为同一聚类
2. 聚类标签由该簇内被验证正确的次数最多的神经元主导
3. 聚类结果 = 涌现的分类标签（不依赖外部代码分配）

```python
def _l1_peer_compare_and_cluster(self, l1_outputs, pool) -> Dict[int, str]:
    """L1 同级比较与自组织聚类

    Args:
        l1_outputs: [(nid, output_pattern, verified, label), ...]
    Returns:
        nid_to_cluster: {nid: cluster_label} 神经元到涌现分类的映射
    """
    cluster_sim = CONFIG.get("L1_CLUSTER_SIMILARITY", 0.70)
    # 1. 计算输出模式两两相似度，构建聚类
    # 2. 为每个聚类确定标签（涌现分类）
    # 3. 更新簇置信度计数器（补丁 P2）
```

### 5.2 聚类标签的涌现规则

| 簇内状态 | 聚类标签 | 说明 |
|---------|---------|------|
| 有验证正确的神经元 | 出现最多的正确标签 | 涌现的真实分类 |
| 无验证正确的神经元 | `"?"` | 未分类，等待后续训练 |

> **重要**：涌现分类需要足够训练量。随机样本+少量训练时 `verified` 全 False 是正常的，`cluster_id` 全为 `"?"` 是预期行为，真实标签会随训练积累从 `cluster_id` 涌现出来。

### 5.3 训练流程整合

```python
# 批次内收集所有 L1 输出
all_l1_outputs = []
for idx, (layers, layer_count, intensity, label) in enumerate(sample_ctx):
    winners_l0 = all_winners[idx]
    info = self._multi_layer_l1_phase(winners_l0, layers, layer_count, intensity, label)
    all_l1_outputs.extend(info.get("_l1_outputs", []))

# 批次级 L1 同级比较与自组织聚类
if CONFIG.get("L1_PEER_COMPARE_ENABLED", True) and all_l1_outputs:
    layer1_pool = self.neuron_layers[1]
    self._l1_peer_compare_and_cluster(all_l1_outputs, layer1_pool)
    # 补丁 P2：周期性重组
    self._batch_counter = getattr(self, '_batch_counter', 0) + 1
    if self._batch_counter % CONFIG.get("L1_CLUSTER_REORG_INTERVAL", 10) == 0:
        self._cluster_reorganization(layer1_pool)
```

---

## 六、L1 回馈图模式

### 6.1 机制

图模式是 L1 的输入中间层。L1 的回馈塑造图模式的演化方向：

- **判断正确** → 增强图模式中与 L1 模板一致的特征（正反馈）
- **判断错误** → 衰减图模式中导致误判的特征（负反馈）

这是简化的反向传播：不计算梯度，而是用相似度差异调整图节点权重。

```python
def _l1_feedback_to_graph(self, n, graph, graph_features, verified):
    """L1 判断正确时回馈调整图模式"""
    strength = CONFIG.get("L1_FEEDBACK_STRENGTH", 0.1)
    t_feat = n.get("T_features", [])

    if verified:
        # 正反馈：图特征向 L1 记忆模板靠拢
        for node in graph.nodes.values():
            for i in range(min(len(node.features), len(t_feat))):
                node.features[i] = int(
                    node.features[i] * (1 - strength) +
                    t_feat[i] * strength
                )
    else:
        # 负反馈：图特征远离导致误判的模式
        for node in graph.nodes.values():
            for i in range(min(len(node.features), len(t_feat))):
                node.features[i] = int(
                    node.features[i] * (1 + strength * 0.5) -
                    t_feat[i] * strength * 0.5
                )
                node.features[i] = max(0, node.features[i])
```

---

## 七、补丁 P2：纠错机制

### 7.1 问题

3.4 节的 L1 涌现分类存在一个设计漏洞：如果两个 L1 神经元把 LINE 和 SINE 的输出模式聚成了同一类（判断方向相似），系统会产生错误分类簇。簇内没有正确验证标签时文档写的是"标记为 `?`"，但没有描述"如何拆散已形成的错误聚类"。

### 7.2 三层纠错方案

| 机制 | 触发时机 | 作用范围 | 目的 |
|------|---------|---------|------|
| **簇置信度** | 每次聚类后更新 | 全簇 | 量化簇的质量 |
| **周期性重组** | 每 N 批次 | 批量 | 拆散持续低质量的簇 |
| **负反馈拆簇** | 每次误判时 | 单神经元 | 即时移出明显错误的归类 |

三层纠错从"量化→批量纠正→即时纠正"递进，避免错误聚类固化。

### 7.3 簇置信度

每个簇维护两个计数器：
- `cluster_verified_count`：该簇判断正确的次数
- `cluster_total_count`：该簇参与判断的总次数

置信度 = `cluster_verified_count / cluster_total_count`

在 `_l1_peer_compare_and_cluster` 的聚类完成后更新：

```python
for cluster in clusters:
    cluster_verified = sum(1 for idx in cluster if l1_outputs[idx][2])
    cluster_total = len(cluster)
    for idx in cluster:
        nid = l1_outputs[idx][0]
        pool[nid]["cluster_verified_count"] += cluster_verified
        pool[nid]["cluster_total_count"] += cluster_total
```

### 7.4 周期性重组

```python
def _cluster_reorganization(self, pool):
    """每 L1_CLUSTER_REORG_INTERVAL 批次调用一次

    1. 计算每个神经元的簇置信度
    2. 置信度 < 阈值的神经元，重置 cluster_id = None
    3. 重置计数器，给重组后的神经元一个干净的起点
    """
    thresh = CONFIG.get("L1_CLUSTER_CONFIDENCE_THRESH", 0.5)
    for n in pool:
        total = n.get("cluster_total_count", 0)
        if total < 5:  # 样本不足，不重组
            continue
        confidence = n.get("cluster_verified_count", 0) / total
        if confidence < thresh:
            n["cluster_id"] = None
            n["cluster_verified_count"] = 0
            n["cluster_total_count"] = 0
```

### 7.5 负反馈即时拆簇

```python
def _negative_feedback_uncluster(self, n, verified):
    """L1 误判时，若所属簇置信度已降到阈值以下，强制移出簇

    与 _cluster_reorganization 的区别：
    - 周期性重组是批量的（每 N 批次）
    - 负反馈拆簇是即时的（每次误判时）
    """
    if verified:
        return
    thresh = CONFIG.get("L1_CLUSTER_CONFIDENCE_THRESH", 0.5)
    total = n.get("cluster_total_count", 0)
    if total < 5:
        return
    confidence = n.get("cluster_verified_count", 0) / total
    if confidence < thresh:
        n["cluster_id"] = None
```

### 7.6 纠错配置

```python
ConfigItem("L1_CLUSTER_CONFIDENCE_THRESH", 0.5, float, (0.1, 0.9),
           "补丁P2:簇置信度阈值(低于此触发重组)", "学习参数", False),
ConfigItem("L1_CLUSTER_REORG_INTERVAL", 10, int, (1, 100),
           "补丁P2:周期性重组间隔(每N批次)", "学习参数", False),
```

> **调优提示**：`L1_CLUSTER_REORG_INTERVAL` 过小（如 2）会导致 `cluster_id` 被过早重置——因为训练初期 `verified` 全 False → 置信度=0 < 阈值 → 重组清空 `cluster_id`。这是 P2 纠错的正确行为，但需足够训练量让 `verified=True` 出现后才能给 `cluster_id` 赋予真实标签。

---

## 八、推理路径改造

### 8.1 涌现分类（优先）

当 `L1_DECISION_LAYER` 开启且有 `cluster_id` 时，改用基于 `cluster_id` 的涌现分类：

```python
# 检查是否有已初始化 cluster_id 的 L1 神经元
has_cluster = any(
    n.get("cluster_id") is not None and n.get("T_features_initialized")
    for n in layer1_pool if not n["L"]
)
if has_cluster:
    # 涌现分类——L1 输出模式 → cluster_id = 预测标签
    for graph_label, graph in core.graphs.items():
        graph_features, l0_active_list = core._extract_graph_features(...)
        for n in layer1_pool:
            cluster = n.get("cluster_id")
            if cluster is None:
                continue
            score = core._match_layer1(n, graph_features, l0_active_list)
            if score > best_score:
                best_score = score
                best_label = cluster
    return best_label, best_score
```

### 8.2 向后兼容回退

`L1_DECISION_LAYER` 关闭或无 `cluster_id` 时回退到遍历标签匹配：

```python
# 遍历所有标签的图做 L1 匹配（3.3 按桶匹配）
for graph_label, graph in core.graphs.items():
    graph_features, l0_active_list = core._extract_graph_features(...)
    for n in layer1_pool:
        score = core._match_layer1(n, graph_features, l0_active_list)
        spec = n.get("specialization") or n.get("cluster_id")
        pred_label = spec or graph_label
        if score > best_score:
            best_score = score
            best_label = pred_label
```

---

## 九、机制要点总结

| 概念 | 含义 |
|------|------|
| **L1 决策层** | L1 是当前架构的最后一层，负责输出判断模式（不是标签） |
| **输出模式** | L1 输出的是判断倾向向量，不是直接的标签 |
| **L1 同级比较** | 与 L0 同级比较对称——L0 比较输入模板，L1 比较输出模式 |
| **自组织分类** | 输出相似的 L1 神经元自然聚成一类，这类=标签，涌现而非分配 |
| **不依赖外部代码分类** | 分类由神经元输出结果驱动，不是外部分类器遍历标签 |
| **L1 回馈图模式** | 判断正确时 L1 调整图模式（中间层），简化的反向传播 |
| **桶化** | 按标签把 winner 分到不同子图，防数据串位 |
| **链条化** | 桶内 winner 按相似度串链，保留特征演化路径 |
| **图模式=中间节点** | 当前是 L0→L1 桥梁，未来多层叠加时只是其中一个节点 |
| **面向未来多层** | L1 回馈机制可复用于 L2→图2、L3→图3，接口不绑层数 |

---

## 十、面向未来的扩展性设计

当前是双层（L0 → 图 → L1），但设计必须考虑未来多层叠加：

```python
# 未来扩展接口（当前预留，暂不实现）
# 每一层都有：竞争 → 输出模式 → 同级比较 → 回馈下层图
# 图作为层间中间节点，可被任意相邻层复用

# 伪代码框架：
for layer_idx in range(max_layers):
    pool = self.neuron_layers[layer_idx]
    graph = self.layer_graphs[layer_idx]  # 该层输入图

    # 1. 竞争 + 生成输出模式
    # 2. 同级比较聚类
    # 3. 回馈调整 graph（图是该层的输入中间层）
    # 4. 输出模式 → 投影为下一层的输入图 layer_graphs[layer_idx+1]
```

设计约束（YAGNI 原则——推迟通用化）：
- `neuron_layers` 用 dict 存储，key 是层号——**数据结构层已通用**，零成本，保留
- `layer_graphs`（未来）用 dict 存储——**数据结构层预留**，零成本，保留
- `_l1_peer_compare_and_cluster` 等方法名——**保持 `_l1_` 前缀，推迟去前缀化**。当前只有两层，名实相符读代码更清晰；真正需要三层时再统一重构（去前缀+参数化层号），那时已知真实需求
- **原则**：数据结构层可零成本通用化（dict 已支持任意层数），方法名层保持具体（YAGNI），避免为不存在的 L2/L3 设计抽象

---

## 十一、配置汇总

### 11.1 桶化链条化（3.3）

| 配置 | 默认值 | 可调范围 | 说明 |
|------|--------|---------|------|
| `L1_BUCKET_ENABLED` | True | -- | 启用桶化 |
| `L1_BUCKET_BY` | "label" | label/specialization | 分桶依据 |
| `L1_CHAIN_ENABLED` | True | -- | 启用链条化 |
| `L1_CHAIN_SIMILARITY` | 0.60 | 0.3 ~ 0.95 | 链条连接相似度阈值 |

### 11.2 决策层（3.4）

| 配置 | 默认值 | 可调范围 | 说明 |
|------|--------|---------|------|
| `L1_DECISION_LAYER` | True | -- | 启用决策层 |
| `L1_OUTPUT_PATTERN_DIM` | 64 | 8 ~ 512 | 输出模式向量维度 |
| `L1_PEER_COMPARE_ENABLED` | True | -- | 启用同级比较 |
| `L1_CLUSTER_SIMILARITY` | 0.70 | 0.4 ~ 0.99 | 聚类相似度阈值 |
| `L1_FEEDBACK_TO_GRAPH` | True | -- | 启用回馈图模式 |
| `L1_FEEDBACK_STRENGTH` | 0.1 | 0.01 ~ 1.0 | 回馈强度 |

### 11.3 纠错机制（P2）

| 配置 | 默认值 | 可调范围 | 说明 |
|------|--------|---------|------|
| `L1_CLUSTER_CONFIDENCE_THRESH` | 0.5 | 0.1 ~ 0.9 | 簇置信度阈值 |
| `L1_CLUSTER_REORG_INTERVAL` | 10 | 1 ~ 100 | 周期性重组间隔（批次） |

---

## 十二、关联文件

| 文件 | 改动 |
|------|------|
| `engine/config.py` | 新增 12 个配置项（4 桶化 + 6 决策层 + 2 纠错） |
| `engine/core.py` | `create_neuron` 新增 6 个字段；新增 9 个方法；重写 `_multi_layer_l1_phase`；改造 `_train_batch_multi_layer` |
| `engine/layers.py` | `classify_multi_layer` 重写：优先 cluster_id 涌现分类 + 向后兼容回退 |
| `app/storage.py` | 序列化/反序列化新增 5 个 L1 决策层字段 |
| `config/sgn_config.json` | 同步 12 个新配置项 |

---

## 十三、术语表

| 术语 | 定义 |
|------|------|
| L1 决策层 | L1 作为架构最后一层，输出判断模式而非标签 |
| 输出模式 (Output Pattern) | L1 输出的判断倾向向量，代表对输入的判断 |
| L1 同级比较 | 批次内 L1 神经元输出模式的横向比较，与 L0 同级比较对称 |
| 自组织分类 | 输出相似的 L1 神经元自然聚成一类，聚类标签=涌现分类 |
| 涌现 (Emergence) | 分类不是代码分配的，而是神经元输出结果自组织产生的 |
| 桶化 (Bucketing) | 按标签把 winner 分到不同子图，防数据串位 |
| 链条化 (Chaining) | 桶内 winner 按模板相似度串链，保留特征演化路径 |
| 簇置信度 | 簇判断正确次数 / 簇参与判断总次数，量化簇质量 |
| L1 回馈 | 判断正确时 L1 调整图模式（中间层），简化的反向传播 |
| 图模式=中间节点 | 图模式是 L0→L1 桥梁，未来多层叠加时是可复用的中间节点 |
