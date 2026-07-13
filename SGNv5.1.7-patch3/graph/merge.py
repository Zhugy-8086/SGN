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
# sgn_merge.py - SGN-Lite v5.0 跨图合并
# 完整文件，依赖 sgn_config, sgn_graph

from typing import List, Dict, Set, Tuple
from collections import defaultdict

from engine.config import CONFIG, DiscreteCoordinate
from graph.graph import GraphNode, DynamicGraph


def merge_winner_projections(
    projections: List[DynamicGraph],
    label: str,
    step: int
) -> DynamicGraph:
    """
    合并多个神经元投影图（来自不同变体视图）
    输入: List[DynamicGraph] - 每个由 project_neurons_to_graph 生成
    输出: 合并后的模板图

    关键：高层节点必须出现在至少2个投影中（一致性过滤）
    """
    if not projections:
        return DynamicGraph()

    merged = DynamicGraph()

    # 1. 收集所有节点，按层级分组，同时记录来源投影索引
    all_nodes_by_layer = defaultdict(list)  # layer -> [(GraphNode, g_idx)]
    for g_idx, g in enumerate(projections):
        for nid, node in g.nodes.items():
            new_node = GraphNode(
                node_id=-1,
                layer=node.layer,
                source_image=-1,
                feature_vector=list(node.feature_vector),
                position_norm=node.position_norm,
                activation=node.activation,
                last_updated=step,
                label=label,
                layer_history=list(node.layer_history),
                source_neurons=list(node.source_neurons)
            )
            all_nodes_by_layer[node.layer].append((new_node, g_idx))

    # 2. 一致性过滤（高层节点去噪）
    # 统计每个 (layer, feature_hash) 出现在多少个投影中
    consistency = defaultdict(set)  # (layer, hash_key) -> set(投影索引)

    for g_idx, g in enumerate(projections):
        for nid, node in g.nodes.items():
            if node.layer >= 2:  # 只对高层做一致性过滤
                key = tuple((dc.level, dc.index) for dc in node.feature_vector[:4])
                consistency[(node.layer, key)].add(g_idx)

    # 3. 过滤节点：高层节点必须出现在至少2个不同投影中
    filtered_nodes = []
    for layer, node_pairs in all_nodes_by_layer.items():
        if layer < 2:
            # 低层节点全部保留
            filtered_nodes.extend([n for n, _ in node_pairs])
        else:
            # 按特征key分组，统计每个key出现在多少个不同投影中
            key_proj_map = defaultdict(set)  # key -> set(投影索引)
            key_node_map = defaultdict(list)  # key -> [GraphNode]
            for node, g_idx in node_pairs:
                key = tuple((dc.level, dc.index) for dc in node.feature_vector[:4])
                key_proj_map[key].add(g_idx)
                key_node_map[key].append(node)

            # 只保留在≥2个投影中出现的节点
            for key, proj_set in key_proj_map.items():
                if len(proj_set) >= 2:
                    filtered_nodes.extend(key_node_map[key])

    if not filtered_nodes:
        return merged

    # 4. 按层级聚类合并
    layer_groups = defaultdict(list)
    for node in filtered_nodes:
        layer_groups[node.layer].append(node)

    for layer, nodes in layer_groups.items():
        if not nodes:
            continue
        threshold = CONFIG.get("GRAPH_SIMILARITY_THRESHOLD", 80)
        clusters = _minhash_cluster(nodes, threshold)

        for cluster in clusters:
            if len(cluster) == 1:
                merged.add_node(cluster[0])
            else:
                merged_node = _merge_cluster_nodes(cluster, layer, label, step)
                merged.add_node(merged_node)

    return merged


def _minhash_cluster(nodes: List[GraphNode], threshold: int = 80) -> List[List[GraphNode]]:
    """
    基于MinHash的O(n)聚类
    """
    if not nodes:
        return []

    # 构建临时MinHash索引
    hash_index = defaultdict(set)
    for idx, node in enumerate(nodes):
        for h in node.get_minhash():
            hash_index[h].add(idx)

    unassigned = set(range(len(nodes)))
    clusters = []

    while unassigned:
        seed_idx = unassigned.pop()
        seed = nodes[seed_idx]
        cluster = [seed]
        cluster_indices = {seed_idx}

        # 召回候选
        candidates = set()
        for h in seed.get_minhash():
            candidates.update(hash_index.get(h, set()))

        # 精确验证
        for cand_idx in candidates:
            if cand_idx in unassigned and cand_idx != seed_idx:
                cand = nodes[cand_idx]
                sim = DynamicGraph.feature_similarity(
                    seed.feature_vector,
                    cand.feature_vector
                )
                if sim >= threshold:
                    cluster.append(cand)
                    cluster_indices.add(cand_idx)

        unassigned -= cluster_indices
        clusters.append(cluster)

    return clusters


def _merge_cluster_nodes(
    cluster: List[GraphNode],
    layer: int,
    label: str,
    step: int
) -> GraphNode:
    """
    合并聚类中的多个节点为一个代表性节点
    特征取中位数，位置取平均

    【v5.1.5】委托 Level 调度器决定运算精度语境
    """
    if len(cluster) == 1:
        return cluster[0]

    # v5.1.5: 委托调度器决定 level
    try:
        from engine.level import get_global_scheduler, OperationType
        scheduler = get_global_scheduler()
        ctx = scheduler.get_context(OperationType.ASSIGN, source="merge_cluster")
        target_level = ctx.target_level
    except ImportError:
        # fallback: 使用最高层级
        all_levels = []
        for n in cluster:
            for f in n.feature_vector:
                all_levels.append(f.level)
        target_level = max(all_levels) if all_levels else 2

    # 特征中位数（使用调度器决定的 level）
    max_len = max(len(n.feature_vector) for n in cluster)
    merged_feat = []

    for dim_idx in range(max_len):
        values = []
        for n in cluster:
            if dim_idx < len(n.feature_vector):
                values.append(n.feature_vector[dim_idx])
        if not values:
            merged_feat.append(DiscreteCoordinate(0, 0))
            continue

        indices = [v.to_level(target_level).index for v in values]
        indices.sort()
        median_idx = indices[len(indices) // 2]
        merged_feat.append(DiscreteCoordinate(median_idx, target_level))

    # 位置平均
    avg_r = sum(n.position_norm[0] for n in cluster) // len(cluster)
    avg_c = sum(n.position_norm[1] for n in cluster) // len(cluster)

    # 确保前两位是空间坐标
    if len(merged_feat) >= 2:
        merged_feat[0] = DiscreteCoordinate(avg_r, target_level)
        merged_feat[1] = DiscreteCoordinate(avg_c, target_level)

    # 合并源神经元
    source_neurons = []
    for n in cluster:
        source_neurons.extend(n.source_neurons)
    source_neurons = list(set(source_neurons))  # 去重

    # 合并层级历史
    layer_history = []
    for n in cluster:
        layer_history.extend(n.layer_history)
    layer_history = sorted(set(layer_history))

    # 合并激活
    total_activation = sum(n.activation for n in cluster)

    return GraphNode(
        node_id=-1,
        layer=layer,
        source_image=-1,
        feature_vector=merged_feat,
        position_norm=(avg_r, avg_c),
        activation=total_activation,
        last_updated=step,
        label=label,
        layer_history=layer_history,
        source_neurons=source_neurons
    )
