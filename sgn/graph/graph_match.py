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
# sgn_graph_match.py - SGN-Lite v5.0 图匹配与推理
# 完整文件，依赖 sgn_config, sgn_graph, sgn_stack

from typing import Tuple, List, Optional

from engine.config import CONFIG
from graph.graph import DynamicGraph
from graph.stack import build_graph_from_intensity


def graph_similarity(
    query_graph: DynamicGraph,
    template_graph: DynamicGraph
) -> Tuple[int, str]:
    """
    计算查询图与模板图的相似度 (0-100)
    强制最高层参与判决，防止局部误判
    """
    if not query_graph.nodes or not template_graph.nodes:
        return 0, '?'

    # 提取标签
    labels = [n.label for n in template_graph.nodes.values() if n.label]
    pred_label = labels[0] if labels else '?'

    # 确定最高可用层
    max_q = max(n.layer for n in query_graph.nodes.values())
    max_t = max(n.layer for n in template_graph.nodes.values())
    max_layer = min(max_q, max_t)

    # 如果最高层 < 1，直接判负（防止L0单层误判）
    if max_layer < 1:
        return 0, pred_label

    layer_scores = []

    for layer_idx in range(max_layer + 1):
        q_nodes = [n for n in query_graph.nodes.values() if n.layer == layer_idx]
        t_nodes = [n for n in template_graph.nodes.values() if n.layer == layer_idx]

        if not q_nodes or not t_nodes:
            continue

        score = _greedy_layer_match(q_nodes, t_nodes)
        layer_scores.append((layer_idx, score))

    if not layer_scores:
        return 0, pred_label

    # 加权：高层权重指数增长 (L0=1, L1=2, L2=4, L3=8, ...)
    total_w = 0
    weighted = 0
    for layer_idx, score in layer_scores:
        w = 1 << layer_idx
        weighted += score * w
        total_w += w

    final = weighted // max(1, total_w)
    return min(100, final), pred_label


def _greedy_layer_match(
    q_nodes: List,
    t_nodes: List
) -> int:
    """
    贪心节点匹配：按空间位置排序，优先匹配位置相近的节点
    """
    if not q_nodes or not t_nodes:
        return 0

    used_t = set()
    total_sim = 0
    matches = 0

    # 按位置排序（使匹配更稳定）
    q_sorted = sorted(q_nodes, key=lambda n: (n.position_norm[0], n.position_norm[1]))

    for q in q_sorted:
        best_sim = -1
        best_t = None

        for t in t_nodes:
            if t.node_id in used_t:
                continue

            # 空间距离惩罚
            dr = abs(q.position_norm[0] - t.position_norm[0])
            dc = abs(q.position_norm[1] - t.position_norm[1])
            spatial_penalty = (dr + dc) // 20
            spatial_penalty = min(50, spatial_penalty)

            feat_sim = DynamicGraph.feature_similarity(
                q.feature_vector,
                t.feature_vector
            )
            combined = max(0, feat_sim - spatial_penalty)

            if combined > best_sim:
                best_sim = combined
                best_t = t

        if best_t is not None:
            total_sim += best_sim
            matches += 1
            used_t.add(best_t.node_id)

    return total_sim // max(1, matches)


def classify_with_graph(
    core,
    intensity: List[int],
    d: Optional[int] = None
) -> Tuple[str, int]:
    """
    使用图模式进行分类
    与每标签的图匹配，取最高分
    """
    if d is None:
        d = getattr(core, 'D', CONFIG.get("D", 64))
    grid_size = int(d ** 0.5)

    # 推理模式：只构建到配置指定的层级
    inference_layer = CONFIG.get("INFERENCE_LAYER", 2)

    query_graph = build_graph_from_intensity(
        intensity, d, grid_size,
        source_image=-1, step=0,
        max_layer_limit=inference_layer
    )

    if not query_graph.nodes:
        return '?', 0

    best_score = -1
    best_label = '?'

    for label, template_graph in core.graphs.items():
        score, _ = graph_similarity(query_graph, template_graph)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score
