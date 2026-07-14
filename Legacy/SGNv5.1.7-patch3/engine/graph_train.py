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
"""SGN-Lite v5.1.5 图模式训练模块 - 从 core.py 提取的图模式训练方法"""

from typing import List, Dict, Tuple

from engine.config import CONFIG, SGNConstants
from engine.utils import extract_layers
from graph.graph import DynamicGraph
from graph.stack import project_neurons_to_graph, rebuild_with_feedback
from graph.merge import merge_winner_projections


def train_graph_mode(core, intensity: List[int], label: str) -> Dict:
    """
    图模式训练 - 基于神经元判断 + 反馈循环 + 层级下压
    """
    d = core.D
    grid_size = core.grid_size
    core._step_counter += 1

    # 1. 获取变体视图（数据增强）
    views = get_parallel_views(core, label, intensity)

    # 2. 对每个视图执行神经元竞争 -> 投影为图
    projections = []
    for view in views:
        layers, lc = extract_layers(view, d=d, strategy=core.layer_strategy)
        if lc == 0:
            continue

        # 神经元竞争
        matches = []
        for i, n in enumerate(core.N):
            if n["L"]:
                continue
            match = core._match(n, layers, lc, view)
            speed = core._response_speed(n, match)
            matches.append((speed, match, i))

        matches.sort(reverse=True)
        top_k = CONFIG["TOP_K"]
        winners = []
        for _, match_score, idx in matches[:top_k]:
            n = core.N[idx].copy()
            n["match"] = match_score
            n["nid"] = idx
            winners.append(n)

        if not winners:
            continue

        # 投影为图
        proj = project_neurons_to_graph(
            winners, view, d, grid_size, core._step_counter, label
        )
        projections.append(proj)

    if not projections:
        return make_graph_info(core, label, False, 0, [])

    # 3. 合并投影 -> 模板图
    if label not in core.graphs:
        core.graphs[label] = DynamicGraph()

    merged = merge_winner_projections(projections, label, core._step_counter)

    # 4. 赫布融合到现有图
    for nid, node in merged.nodes.items():
        core.graphs[label].hebbian_merge(
            node, core._step_counter,
            lr_alpha_pct=int(CONFIG.get("GRAPH_LEARNING_RATE", 0.3) * 100)
        )

    # 5. 【核心】反馈迭代循环
    final_score, final_label = feedback_loop(
        core, intensity, label, core.graphs[label], d, grid_size
    )

    # 6. 层级下压遗忘（每50步）
    if core._step_counter % 50 == 0:
        demote_and_cover(core, label)

    verified = (final_score >= 80)
    return make_graph_info(core, label, verified, final_score, [merged])


def get_parallel_views(core, label: str, base_intensity: List[int]) -> List[List[int]]:
    """生成同一标签的多张变体图"""
    n_views = CONFIG.get("PARALLEL_VIEWS", 3)
    if n_views <= 1:
        return [base_intensity]

    from engine.input import DefaultCompositeNoise
    noise = DefaultCompositeNoise(CONFIG.get("FLIP_PROB", 0.15))

    views = [base_intensity]
    for _ in range(n_views - 1):
        views.append(noise.apply(base_intensity.copy()))
    return views


def feedback_loop(
    core,
    intensity: List[int],
    label: str,
    graph: DynamicGraph,
    d: int,
    grid_size: int
) -> Tuple[int, str]:
    """
    反馈迭代循环：全局匹配失败 -> 生成误差图 -> 重新局部扫描
    返回: (最终匹配度, 预测标签)
    """
    max_loops = CONFIG.get("MAX_FEEDBACK_LOOPS", 3)
    threshold = CONFIG.get("FEEDBACK_THRESHOLD", 85)

    best_score = 0
    best_label = label
    current_graph = graph

    for loop in range(max_loops):
        # 从当前图重建强度图，与原始输入比较
        reconstructed = reconstruct_intensity(core, current_graph, d, grid_size)
        # 计算重建误差作为匹配度（误差越小匹配度越高）
        match_score = compute_reconstruction_score(core, intensity, reconstructed, d)

        if match_score >= threshold:
            return match_score, label
        if match_score > best_score:
            best_score = match_score
            best_label = label

        # 生成误差图
        error_intensity = generate_error_map(core, intensity, current_graph, d, grid_size)

        # 用误差图反馈重建
        current_graph = rebuild_with_feedback(
            current_graph, error_intensity, d, grid_size,
            core._step_counter, label
        )

    return best_score, best_label


def reconstruct_intensity(
    core,
    graph: DynamicGraph,
    d: int,
    grid_size: int
) -> List[int]:
    """
    从图结构重建强度图（与 _generate_error_map 共用逻辑）
    """
    reconstructed = [0] * d
    for node in graph.get_layer_nodes(0):
        r = (node.position_norm[0] * max(1, grid_size - 1)) // SGNConstants.POSITION_NORM
        c = (node.position_norm[1] * max(1, grid_size - 1)) // SGNConstants.POSITION_NORM
        idx = r * grid_size + c
        if 0 <= idx < d:
            reconstructed[idx] = 255
    return reconstructed


def compute_reconstruction_score(
    core,
    original: List[int],
    reconstructed: List[int],
    d: int
) -> int:
    """
    计算重建匹配度 (0-100)
    匹配度 = 100 - 平均像素误差百分比
    """
    if d == 0:
        return 0
    total_diff = sum(abs(original[i] - reconstructed[i]) for i in range(d))
    max_possible = 255 * d
    if max_possible == 0:
        return 0
    error_pct = (total_diff * SGNConstants.PERCENT_BASE) // max_possible
    return max(0, SGNConstants.PERCENT_BASE - error_pct)


def generate_error_map(
    core,
    original: List[int],
    graph: DynamicGraph,
    d: int,
    grid_size: int
) -> List[int]:
    """
    从图重建强度图，计算与原始的差异
    """
    reconstructed = [0] * d

    # 取所有L0节点的位置
    for node in graph.get_layer_nodes(0):
        r = (node.position_norm[0] * max(1, grid_size - 1)) // SGNConstants.POSITION_NORM
        c = (node.position_norm[1] * max(1, grid_size - 1)) // SGNConstants.POSITION_NORM
        idx = r * grid_size + c
        if 0 <= idx < d:
            reconstructed[idx] = 255

    # 误差 = |original - reconstructed|，放大差异
    error = [abs(original[i] - reconstructed[i]) for i in range(d)]
    return error


def demote_and_cover(core, label: str) -> None:
    """
    层级下压 + L0覆盖遗忘
    """
    if label not in core.graphs:
        return

    graph = core.graphs[label]
    threshold = CONFIG.get("DEMOTION_THRESHOLD", 5)
    cover_age = CONFIG.get("LAYER_COVER_THRESHOLD", 1000)

    # 逐级下压（从高层到低层）
    total_demoted = graph.demote_all(threshold)

    # L0覆盖：删除长期未激活的L0节点
    to_cover = graph.get_cover_candidates(core._step_counter, cover_age)
    for nid in to_cover:
        node = graph.get_node(nid)
        if node and node.activation < 2:
            graph.remove_node(nid)


def make_graph_info(
    core,
    label: str,
    verified: bool,
    score: int,
    graphs: List[DynamicGraph]
) -> Dict:
    """构造info字典"""
    total_nodes = sum(g.get_total_nodes() for g in graphs) if graphs else 0
    return {
        "label": label,
        "layer_count": 0,
        "V": verified,
        "match": score,
        "base": 0,
        "winners": [],
        "active": total_nodes,
        "locked": 0,
        "templates": len(core.graphs),
        "_D": core.D,
        "graph_mode": True,
        "graph_nodes": total_nodes,
        "step": core._step_counter,
    }
