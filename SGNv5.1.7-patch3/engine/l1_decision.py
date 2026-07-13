#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 [zhugy-8086]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGN-Lite v5.1.5 L1 决策层模块 - 桶化/链条化/输出模式/同级聚类/纠错/匹配"""

from typing import List, Dict, Tuple

from .config import CONFIG
from graph.graph import DynamicGraph


def bucket_winners_by_label(
    core,
    winners: List[Dict],
    pool: List[Dict],
    sample: Tuple
) -> Dict[str, List[Dict]]:
    """v5.1.7: 按样本标签分桶（v5.1.7-patch3: L0 回归纯特征检测器）

    v5.1.7-patch3: L0 神经元不再持有 specialization（违背原始设计：
    《SGN神经元与神经节·行为定义》§2.1.1 明确 L0 模板不与任何标签绑定）。
    桶化一律按当前样本标签分桶，切断"5 的特征被分到 1 的桶"的污染链。

    每个桶独立投影，避免不同标签特征互相串位。
    """
    sample_label = sample[1] if sample else "?"
    buckets: Dict[str, List[Dict]] = {}

    for w in winners:
        key = sample_label
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(w)

    return buckets


def chain_winners(
    core,
    winners: List[Dict],
    pool: List[Dict]
) -> List[Dict]:
    """v5.1.7: 桶内链条化

    将桶内 winner 按模板相似度串成链条：
    1. 以第一个 winner 为链头
    2. 依次找与当前链尾相似度 >= L1_CHAIN_SIMILARITY 的 winner 接上
    3. 不相似的 winner 另起一条链（追加到末尾）

    链条保留特征演化路径，L1 读取时按链顺序匹配。
    """
    if len(winners) <= 1:
        return winners

    chain_sim = CONFIG.get("L1_CHAIN_SIMILARITY", 0.60)
    chained = [winners[0]]
    remaining = list(winners[1:])

    while remaining:
        current = chained[-1]
        best_idx = -1
        best_sim = -1.0
        for i, w in enumerate(remaining):
            sim = core._template_similarity(
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


def get_bucket(core, nid: int, pool: List[Dict]) -> str:
    """v5.1.7-patch3: 返回神经元所属桶标签（推理时用）

    L0 神经元无 specialization，一律返回 "?"。
    保留接口兼容性，调用方可通过其他方式（如图匹配）确定标签。
    """
    return "?"


# ============================================================
# v5.1.7 3.4: L1 决策层与自组织分类
# ============================================================

def l1_generate_output_pattern(
    core, n: Dict, graph_features: List, l0_active_list: List
) -> List[float]:
    """v5.1.7: L1 神经元生成输出判断模式

    不是返回单个 score，而是生成一个向量，代表 L1 对当前输入的判断倾向。
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


def pattern_similarity(
    core, p1: List[float], p2: List[float]
) -> float:
    """v5.1.7: 计算两个输出模式的余弦相似度 (0.0~1.0)

    衡量判断方向的一致性。
    """
    if not p1 or not p2 or len(p1) != len(p2):
        return 0.0
    dot = sum(a * b for a, b in zip(p1, p2))
    norm1 = sum(a * a for a in p1) ** 0.5
    norm2 = sum(b * b for b in p2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def l1_peer_compare_and_cluster(
    core,
    l1_outputs: List[Tuple[int, List[float], bool, str]],
    pool: List[Dict]
) -> Dict[int, str]:
    """v5.1.7: L1 同级比较与自组织聚类

    对批次内所有 L1 神经元的输出模式做横向比较：
    - 输出模式相似度 >= L1_CLUSTER_SIMILARITY 的归为同一聚类
    - 聚类标签由该簇内被验证正确的次数最多的神经元主导
    - 聚类结果 = 涌现的分类标签（不依赖外部代码分配）

    Args:
        l1_outputs: [(nid, output_pattern, verified, label), ...]
    Returns:
        nid_to_cluster: {nid: cluster_label} 神经元到涌现分类的映射
    """
    cluster_sim = CONFIG.get("L1_CLUSTER_SIMILARITY", 0.70)
    n = len(l1_outputs)
    assigned = [False] * n
    clusters: List[List[int]] = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            sim = core._pattern_similarity(
                l1_outputs[i][1], l1_outputs[j][1]
            )
            if sim >= cluster_sim:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # 为每个聚类确定标签（涌现分类）+ 补丁 P2 簇置信度
    nid_to_cluster: Dict[int, str] = {}
    for cluster in clusters:
        verified_labels: Dict[str, int] = {}
        cluster_verified = 0
        cluster_total = 0
        for idx in cluster:
            nid, _, verified, label = l1_outputs[idx]
            cluster_total += 1
            if verified and label:
                verified_labels[label] = verified_labels.get(label, 0) + 1
                cluster_verified += 1

        if verified_labels:
            cluster_label = max(verified_labels, key=verified_labels.get)
        else:
            cluster_label = "?"

        # 写回神经元的 cluster_id + 补丁 P2 簇计数器
        for idx in cluster:
            nid = l1_outputs[idx][0]
            pool[nid]["cluster_id"] = cluster_label
            pool[nid]["cluster_verified_count"] += cluster_verified
            pool[nid]["cluster_total_count"] += cluster_total
            nid_to_cluster[nid] = cluster_label

    return nid_to_cluster


def l1_feedback_to_graph(
    core,
    n: Dict,
    graph: DynamicGraph,
    graph_features: List,
    verified: bool
):
    """v5.1.7: L1 判断正确时回馈调整图模式

    图模式是 L1 的输入中间层。L1 的回馈塑造图模式的演化方向：
    - 判断正确 → 增强图模式中与 L1 模板一致的特征（正反馈）
    - 判断错误 → 衰减图模式中导致误判的特征（负反馈）
    """
    if not CONFIG.get("L1_FEEDBACK_TO_GRAPH", True):
        return

    strength = CONFIG.get("L1_FEEDBACK_STRENGTH", 0.1)
    t_feat = n.get("T_features", [])

    if verified:
        for node in graph.nodes.values():
            if hasattr(node, 'features') and node.features:
                for i in range(min(len(node.features), len(t_feat))):
                    t_val = t_feat[i].index if hasattr(t_feat[i], 'index') else t_feat[i]
                    node.features[i] = int(
                        node.features[i] * (1 - strength) +
                        t_val * strength
                    )
    else:
        for node in graph.nodes.values():
            if hasattr(node, 'features') and node.features:
                for i in range(min(len(node.features), len(t_feat))):
                    t_val = t_feat[i].index if hasattr(t_feat[i], 'index') else t_feat[i]
                    node.features[i] = int(
                        node.features[i] * (1 + strength * 0.5) -
                        t_val * strength * 0.5
                    )
                    node.features[i] = max(0, node.features[i])


# ============================================================
# v5.1.7 补丁 P2：纠错机制
# ============================================================

def cluster_reorganization(core, pool: List[Dict]):
    """补丁 P2: 周期性重组低置信度簇

    每 L1_CLUSTER_REORG_INTERVAL 批次调用一次：
    1. 计算每个神经元的簇置信度 = verified / total
    2. 置信度 < L1_CLUSTER_CONFIDENCE_THRESH 的神经元，
       重置 cluster_id = None，强制下次同级比较重新归类
    3. 重置计数器，给重组后的神经元一个干净的起点
    """
    thresh = CONFIG.get("L1_CLUSTER_CONFIDENCE_THRESH", 0.5)
    for n in pool:
        total = n.get("cluster_total_count", 0)
        if total < 5:
            continue
        confidence = n.get("cluster_verified_count", 0) / total
        if confidence < thresh:
            n["cluster_id"] = None
            n["cluster_verified_count"] = 0
            n["cluster_total_count"] = 0


def negative_feedback_uncluster(
    core, n: Dict, verified: bool
):
    """补丁 P2: 负反馈即时拆簇

    L1 回馈图模式时，如果 verified=False 且该神经元所属簇的
    置信度已降到阈值以下，强制将该神经元移出簇。
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


def extract_graph_features(
    core, graph: DynamicGraph, d: int, grid_size: int,
    l0_active_nids: List[int]
) -> Tuple[List, List[int]]:
    """
    从 DynamicGraph 提取特征向量 + Layer 0 激活 ID 列表

    特征设计原则：
      - 每个特征都是 DiscreteCoordinate，与现有整数化体系一致
      - 特征维度固定（不随图大小变化），便于 Layer 1 模板比较
      - 包含结构信息（连通域、层级分布）和统计信息（激活、位置）

    Args:
        graph: 目标标签的动态图
        d: 输入维度
        grid_size: 网格边长
        l0_active_nids: Layer 0 获胜神经元 ID 列表（不用位掩码 Bug 4）

    Returns:
        graph_features: 图特征向量（固定 10 维）
        l0_active_list: Layer 0 激活 ID 列表（直接返回，不用位掩码）
    """
    from .config import DiscreteCoordinate

    total_nodes = len(graph.nodes)
    if total_nodes == 0:
        return [DiscreteCoordinate(0, 0)] * 10, list(l0_active_nids)

    max_layer = graph.get_max_layer()

    # 层级分布（固定 4 层：L0/L1/L2/L3）
    layer_counts = [0, 0, 0, 0]
    for l in range(min(max_layer + 1, 4)):
        layer_counts[l] = len(graph.get_layer_nodes(l))

    # 平均激活
    avg_activation = sum(n.activation for n in graph.nodes.values()) // total_nodes

    # L0 节点位置分散度
    l0_nodes = graph.get_layer_nodes(0)
    if l0_nodes:
        avg_r = sum(n.position_norm[0] for n in l0_nodes) // len(l0_nodes)
        avg_c = sum(n.position_norm[1] for n in l0_nodes) // len(l0_nodes)
        spread = sum(
            abs(n.position_norm[0] - avg_r) + abs(n.position_norm[1] - avg_c)
            for n in l0_nodes
        ) // len(l0_nodes)
    else:
        spread = 0

    # 连通域数（L0 节点的邻居连接数）
    edge_count = 0
    for n in graph.nodes.values():
        edge_count += len(n.neighbors)
    avg_edges = edge_count // max(1, total_nodes)

    # 特征向量维度分布（取 L0 第一个节点的特征维度）
    feat_dim = len(l0_nodes[0].feature_vector) if l0_nodes else 0

    graph_features = [
        DiscreteCoordinate(total_nodes, 0),           # 0: 总节点数
        DiscreteCoordinate(max_layer, 0),             # 1: 最高层级
        DiscreteCoordinate(avg_activation, 0),        # 2: 平均激活
        DiscreteCoordinate(spread, 2),                # 3: 位置分散度
        DiscreteCoordinate(layer_counts[0], 0),       # 4: L0 节点数
        DiscreteCoordinate(layer_counts[1], 0),       # 5: L1 节点数
        DiscreteCoordinate(layer_counts[2], 0),       # 6: L2 节点数
        DiscreteCoordinate(avg_edges, 0),             # 7: 平均边数
        DiscreteCoordinate(feat_dim, 0),              # 8: 特征维度
        DiscreteCoordinate(len(l0_active_nids), 0),   # 9: 本次激活的 L0 神经元数
    ]

    # v5.1.7-fix: 增加 L0 掩码位级特征（视觉模式信息）
    # 将 L0 获胜神经元的掩码 T[0] 聚合为位置化特征，
    # 解决拓扑统计量无法区分相似结构但不同视觉模式的问题
    l0_mask_features = []
    layer0_pool = core.neuron_layers.get(0, [])
    for nid in l0_active_nids[:8]:
        if nid < len(layer0_pool):
            n = layer0_pool[nid]
            l0_mask_features.append(DiscreteCoordinate(n["T"][0] % 10000, 0))
    while len(l0_mask_features) < 8:
        l0_mask_features.append(DiscreteCoordinate(0, 0))

    graph_features = graph_features + l0_mask_features  # 10 + 8 = 18 维

    # L0 激活列表直接返回（不用位掩码 Bug 4）
    return graph_features, list(l0_active_nids)


def match_layer1(
    core, n: Dict,
    graph_features: List,
    l0_active_list: List[int]
) -> int:
    """
    Layer 1 神经元匹配：图特征向量相似度 + L0 激活重叠度

    匹配策略：
      - 图特征相似度（60%权重）：用 DiscreteCoordinate 特征向量比较
      - L0 激活重叠度（40%权重）：用 Jaccard 相似度比较 ID 列表

    注意：Layer 1 的 T 是随机像素位掩码（不使用），
          实际匹配用 T_features 和 T_l0_active 字段。

    Args:
        n: Layer 1 神经元
        graph_features: 当前输入的图特征向量
        l0_active_list: 当前输入的 L0 激活 ID 列表

    Returns:
        匹配度 0-100
    """
    from .config import DiscreteCoordinate

    # 如果模板未初始化，返回 0（让神经元自然竞争）
    if not n.get("T_features_initialized", False):
        return 0

    t_features = n.get("T_features", [])
    t_l0_active = n.get("T_l0_active", [])

    # --- 图特征相似度 ---
    if not t_features or not graph_features:
        feat_sim = 0
    else:
        min_len = min(len(t_features), len(graph_features))
        if min_len == 0:
            feat_sim = 0
        else:
            # 统一到最高层级后比较 index
            all_levels = []
            for i in range(min_len):
                all_levels.append(t_features[i].level)
                all_levels.append(graph_features[i].level)
            target_lvl = max(all_levels) if all_levels else 0

            matches = 0
            for i in range(min_len):
                t_val = t_features[i].to_level(target_lvl)
                g_val = graph_features[i].to_level(target_lvl)
                # v5.1.7-patch: 相对阈值（不依赖特征值数量级，对任意层通用）
                # 值差在两者最大值的 20% 以内算匹配
                max_val = max(abs(t_val.index), abs(g_val.index), 1)
                if abs(t_val.index - g_val.index) / max_val <= 0.2:
                    matches += 1
            feat_sim = (matches * 100) // min_len

    # --- L0 激活重叠度（Jaccard 相似度）---
    if not t_l0_active or not l0_active_list:
        l0_sim = 50  # 无模板时给中间值
    else:
        # v5.1.7-patch: 兼容 dict（频次衰减）和 list（旧格式）两种存储
        if isinstance(t_l0_active, dict):
            set_t = set(t_l0_active.keys())
        else:
            set_t = set(t_l0_active)
        set_g = set(l0_active_list)
        intersection = len(set_t & set_g)
        union = len(set_t | set_g)
        l0_sim = (intersection * 100) // max(1, union)

    # v5.1.7-patch: 权重动态调整（基于特征可靠性）
    # 当特征值方差小时（特征无区分度），降低 feat_sim 权重，提高 l0_sim 权重
    # 该机制对任意特征向量通用——未来 L2 的特征也能用方差评估可靠性
    feat_weight = 6
    l0_weight = 4
    if t_features and graph_features:
        min_len_w = min(len(t_features), len(graph_features))
        if min_len_w > 0:
            # 计算当前图特征的方差（反映区分度）
            g_vals = [g.index if hasattr(g, 'index') else g for g in graph_features[:min_len_w]]
            try:
                g_mean = sum(g_vals) / len(g_vals)
                g_var = sum((v - g_mean) ** 2 for v in g_vals) / len(g_vals)
            except (TypeError, ZeroDivisionError):
                g_var = 0
            # 方差小 → 特征无区分度 → 降低 feat_sim 权重
            var_thresh = CONFIG.get("FEATURE_VARIANCE_THRESH", 100.0)
            if g_var < var_thresh:
                feat_weight = 3
                l0_weight = 7
    return (feat_sim * feat_weight + l0_sim * l0_weight) // (feat_weight + l0_weight)


def compute_l1_predicted_label(
    core, n: Dict, d: int, grid_size: int,
    l0_active_nids: List[int],
    graph_features_cache: Dict[str, Tuple] = None
):
    """v5.1.7-patch: 计算 L1 神经元当前最匹配的图标签

    通过将神经元的 T_features 与所有图的图特征匹配，
    找到最佳匹配的图标签作为预测标签。

    与 classify_multi_layer 的逻辑一致，但不依赖 specialization/cluster_id。
    用于打破原修复方案中 specialization/cluster_id 的冷启动死锁。

    Args:
        n: Layer 1 神经元
        d: 输入维度
        grid_size: 网格边长
        l0_active_nids: 当前样本的 L0 激活神经元 ID 列表
        graph_features_cache: 预计算的图特征缓存 {label: (features, l0_active)}
                              None 时实时计算

    Returns:
        v5.1.7-patch2: (预测标签, 最佳匹配分数)
        标签为 None 表示神经元未初始化无法预测
    """
    if not n.get("T_features_initialized"):
        return None, -1

    best_score = -1
    best_label = None

    if graph_features_cache is not None:
        for graph_label, (gf, l0a) in graph_features_cache.items():
            score = core._match_layer1(n, gf, l0a)
            if score > best_score:
                best_score = score
                best_label = graph_label
    else:
        for graph_label, graph in core.graphs.items():
            gf, l0a = core._extract_graph_features(
                graph, d, grid_size, l0_active_nids
            )
            score = core._match_layer1(n, gf, l0a)
            if score > best_score:
                best_score = score
                best_label = graph_label

    return best_label, best_score
