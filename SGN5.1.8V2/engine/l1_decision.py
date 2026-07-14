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
"""SGN-Lite v5.1.9 L1 决策层模块 - 桶化/链条化/图特征提取/匹配/预测"""

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
# v5.1.7 3.4: L1 决策层
# ============================================================


def extract_graph_features(
    core, graph: DynamicGraph, d: int, grid_size: int
) -> List:
    """v5.1.9: 从 DynamicGraph 提取 72 维分布直方图特征向量（全 Level 0 整数）

    替代旧版 10 维手工统计量。核心改进：
      - 用分布直方图代替平均值/总数，保留完整分布信息
      - 全部 Level 0 整数，消除 to_level 转换开销和精度丢失
      - 不再接收 l0_active_nids（它是当前样本的瞬时状态，不是图的固有属性）

    特征组（72 维）：
      1. 激活强度分布直方图 (16 维)：节点 activation 分 16 桶
      2. 边强度分布直方图 (16 维)：邻居连接强度分 16 桶
      3. 层级深度占比 (8 维)：L0~L7 节点数占比（百分比整数）
      4. 空间位置分布 (16 维)：L0 节点 X 8 桶 + Y 8 桶
      5. 连通域大小分布 (8 维)：BFS 连通域大小分 8 桶
      6. 度分布 (8 维)：节点度数分 8 桶

    Args:
        graph: 目标标签的动态图
        d: 输入维度
        grid_size: 网格边长

    Returns:
        graph_features: 图特征向量（固定 72 维，全 Level 0）
    """
    from .config import DiscreteCoordinate

    total_nodes = len(graph.nodes)
    if total_nodes == 0:
        return [DiscreteCoordinate(0, 0)] * 72

    features = []

    # --- 1. 激活强度分布直方图 (16 维) ---
    act_hist = [0] * 16
    for node in graph.nodes.values():
        bucket = min(node.activation // 16, 15)
        act_hist[bucket] += 1
    max_c = max(max(act_hist), 1)
    features.extend(
        DiscreteCoordinate((c * 255) // max_c, 0) for c in act_hist
    )

    # --- 2. 边强度分布直方图 (16 维) ---
    edge_hist = [0] * 16
    for node in graph.nodes.values():
        for strength in node.neighbors.values():
            s_val = strength.index if hasattr(strength, 'index') else strength
            bucket = min(s_val // 16, 15)
            edge_hist[bucket] += 1
    max_c = max(max(edge_hist), 1)
    features.extend(
        DiscreteCoordinate((c * 255) // max_c, 0) for c in edge_hist
    )

    # --- 3. 层级深度占比 (8 维) ---
    max_layer = graph.get_max_layer()
    for l in range(8):
        count = len(graph.get_layer_nodes(l)) if l <= max_layer else 0
        features.append(DiscreteCoordinate((count * 100) // total_nodes, 0))

    # --- 4. 空间位置分布 (16 维) ---
    l0_nodes = graph.get_layer_nodes(0)
    x_hist = [0] * 8
    y_hist = [0] * 8
    for node in l0_nodes:
        x_val = node.position_norm[1]
        y_val = node.position_norm[0]
        x_hist[min(x_val * 8 // 1001, 7)] += 1
        y_hist[min(y_val * 8 // 1001, 7)] += 1
    max_x = max(max(x_hist), 1)
    max_y = max(max(y_hist), 1)
    features.extend(
        DiscreteCoordinate((c * 255) // max_x, 0) for c in x_hist
    )
    features.extend(
        DiscreteCoordinate((c * 255) // max_y, 0) for c in y_hist
    )

    # --- 5. 连通域大小分布 (8 维) ---
    cc_hist = [0] * 8
    visited = set()
    for start_id in graph.nodes:
        if start_id in visited:
            continue
        queue = [start_id]
        visited.add(start_id)
        comp_size = 0
        while queue:
            nid = queue.pop(0)
            comp_size += 1
            node = graph.nodes[nid]
            for neighbor_id in node.neighbors:
                if neighbor_id not in visited and neighbor_id in graph.nodes:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)
        if comp_size <= 4:
            bucket = comp_size - 1
        elif comp_size <= 6:
            bucket = 4
        elif comp_size <= 8:
            bucket = 5
        elif comp_size <= 12:
            bucket = 6
        else:
            bucket = 7
        cc_hist[bucket] += 1
    max_c = max(max(cc_hist), 1)
    features.extend(
        DiscreteCoordinate((c * 255) // max_c, 0) for c in cc_hist
    )

    # --- 6. 度分布 (8 维) ---
    deg_hist = [0] * 8
    for node in graph.nodes.values():
        degree = len(node.neighbors)
        if degree <= 4:
            bucket = degree
        elif degree <= 6:
            bucket = 5
        elif degree <= 8:
            bucket = 6
        else:
            bucket = 7
        deg_hist[bucket] += 1
    max_c = max(max(deg_hist), 1)
    features.extend(
        DiscreteCoordinate((c * 255) // max_c, 0) for c in deg_hist
    )

    return features


def match_layer1(
    core, n: Dict,
    graph_features: List,
    l0_active_list: List[int]
) -> int:
    """v5.1.9: Layer 1 神经元匹配：图特征向量相似度 + L0 激活重叠度

    匹配策略：
      - 图特征相似度（60%权重）：72 维 Level 0 整数比较，容差 <= 1
      - L0 激活重叠度（40%权重）：用 Jaccard 相似度比较 ID 列表

    v5.1.9 改进：
      - 容差从 20% 相对阈值改为 <= 1（整数最小分辨率），零调参
      - 消除 to_level 转换，直接比较 .index（全 Level 0）

    Args:
        n: Layer 1 神经元
        graph_features: 当前输入的图特征向量（72 维 Level 0）
        l0_active_list: 当前输入的 L0 激活 ID 列表

    Returns:
        匹配度 0-100
    """
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
            matches = 0
            for i in range(min_len):
                t_val = t_features[i].index if hasattr(t_features[i], 'index') else t_features[i]
                g_val = graph_features[i].index if hasattr(graph_features[i], 'index') else graph_features[i]
                # v5.1.9: 整数最小分辨率容差，零调参
                if abs(t_val - g_val) <= 1:
                    matches += 1
            feat_sim = (matches * 100) // min_len

    # --- L0 激活重叠度（Jaccard 相似度）---
    if not t_l0_active or not l0_active_list:
        l0_sim = 50  # 无模板时给中间值
    else:
        # 兼容 dict（频次衰减）和 list（旧格式）两种存储
        if isinstance(t_l0_active, dict):
            set_t = set(t_l0_active.keys())
        else:
            set_t = set(t_l0_active)
        set_g = set(l0_active_list)
        intersection = len(set_t & set_g)
        union = len(set_t | set_g)
        l0_sim = (intersection * 100) // max(1, union)

    # 权重动态调整（基于特征可靠性）
    # 当特征值方差小时（特征无区分度），降低 feat_sim 权重，提高 l0_sim 权重
    feat_weight = 6
    l0_weight = 4
    if t_features and graph_features:
        min_len_w = min(len(t_features), len(graph_features))
        if min_len_w > 0:
            g_vals = [g.index if hasattr(g, 'index') else g for g in graph_features[:min_len_w]]
            try:
                g_mean = sum(g_vals) / len(g_vals)
                g_var = sum((v - g_mean) ** 2 for v in g_vals) / len(g_vals)
            except (TypeError, ZeroDivisionError):
                g_var = 0
            var_thresh = CONFIG.get("FEATURE_VARIANCE_THRESH", 100.0)
            if g_var < var_thresh:
                feat_weight = 3
                l0_weight = 7
    return (feat_sim * feat_weight + l0_sim * l0_weight) // (feat_weight + l0_weight)


def compute_l1_predicted_label(
    core, n: Dict, d: int, grid_size: int,
    l0_active_nids: List[int],
    graph_features_cache: Dict = None
):
    """v5.1.9: 计算 L1 神经元当前最匹配的图标签

    通过将神经元的 T_features 与所有图的图特征匹配，
    找到最佳匹配的图标签作为预测标签。

    v5.1.9 改进：图特征缓存只存储 gf（不含 l0_active），
    l0_active_nids 作为当前样本的瞬时状态直接传给 match_layer1，
    彻底切断"所有图共用同一份 l0_active"的污染源。

    Args:
        n: Layer 1 神经元
        d: 输入维度
        grid_size: 网格边长
        l0_active_nids: 当前样本的 L0 激活神经元 ID 列表
        graph_features_cache: 预计算的图特征缓存 {label: features}
                              None 时实时计算

    Returns:
        (预测标签, 最佳匹配分数)
        标签为 None 表示神经元未初始化无法预测
    """
    if not n.get("T_features_initialized"):
        return None, -1

    best_score = -1
    best_label = None

    if graph_features_cache is not None:
        for graph_label, gf in graph_features_cache.items():
            score = core._match_layer1(n, gf, l0_active_nids)
            if score > best_score:
                best_score = score
                best_label = graph_label
    else:
        for graph_label, graph in core.graphs.items():
            gf = core._extract_graph_features(graph, d, grid_size)
            score = core._match_layer1(n, gf, l0_active_nids)
            if score > best_score:
                best_score = score
                best_label = graph_label

    return best_label, best_score
