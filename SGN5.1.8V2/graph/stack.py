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
# sgn_stack.py - SGN-Lite v5.0 图构建与投影
# 完整文件，依赖 sgn_config, sgn_layers, sgn_graph

import math
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

from engine.config import CONFIG, DiscreteCoordinate
from engine.layers import extract_layers
from graph.graph import GraphNode, DynamicGraph


# ============================================================
# 辅助工具：掩码 ↔ 连通域
# ============================================================

# 预计算邻域偏移缓存：{grid_size: [列表: 位置→邻居位掩码]}
_NEIGHBOR_OFFSETS_CACHE: Dict[int, List[int]] = {}


def _build_neighbor_masks(gs: int) -> List[int]:
    """为给定网格边长预计算每个位置的8-邻域掩码"""
    d = gs * gs
    result = [0] * d
    for pos in range(d):
        r, c = pos // gs, pos % gs
        nm = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs:
                    nm |= (1 << (nr * gs + nc))
        result[pos] = nm
    return result


def _get_neighbor_masks(gs: int) -> List[int]:
    if gs not in _NEIGHBOR_OFFSETS_CACHE:
        _NEIGHBOR_OFFSETS_CACHE[gs] = _build_neighbor_masks(gs)
    return _NEIGHBOR_OFFSETS_CACHE[gs]


def mask_to_connected_components(mask: int, grid_size: int) -> List[List[Tuple[int, int]]]:
    """
    将掩码转为8-邻域连通域列表
    返回: [[(r1,c1), (r2,c2), ...], [...]]
    """
    if mask == 0 or grid_size <= 0:
        return []

    d = grid_size * grid_size
    nm_tbl = _get_neighbor_masks(grid_size)
    visited = 0
    remaining = mask
    components = []

    while remaining:
        start = (remaining & -remaining).bit_length() - 1
        start_bit = 1 << start
        component_mask = start_bit
        visited |= start_bit
        frontier = start_bit

        while frontier:
            nxt = 0
            f = frontier
            while f:
                lsb = f & -f
                nxt |= nm_tbl[lsb.bit_length() - 1]
                f &= f - 1
            nxt &= (mask & ~visited)
            if not nxt:
                break
            component_mask |= nxt
            visited |= nxt
            frontier = nxt

        remaining = mask & ~visited

        comp = []
        cm = component_mask
        while cm:
            lsb = cm & -cm
            pos = lsb.bit_length() - 1
            comp.append((pos // grid_size, pos % grid_size))
            cm &= cm - 1
        if comp:
            components.append(comp)

    return components


def components_to_l0_nodes(
    components: List[List[Tuple[int, int]]],
    grid_size: int,
    source_image: int,
    layer_idx: int,
    extra_features: Optional[List[DiscreteCoordinate]] = None
) -> List[GraphNode]:
    """
    将连通域列表转为 L0 GraphNode 列表
    """
    nodes = []
    for comp in components:
        if not comp:
            continue

        # 计算质心
        sum_r = sum(r for r, c in comp)
        sum_c = sum(c for r, c in comp)
        area = len(comp)
        center_r = sum_r // area
        center_c = sum_c // area

        # 边界框
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)
        bbox_h = max_r - min_r + 1
        bbox_w = max_c - min_c + 1
        aspect = (bbox_h * 100) // max(1, bbox_w)

        # 归一化坐标 (0~1000)
        row_norm = (center_r * 1000) // max(1, grid_size - 1)
        col_norm = (center_c * 1000) // max(1, grid_size - 1)

        # 特征向量：[行, 列, 面积, 宽高比, 层索引] + 额外特征
        feat = [
            DiscreteCoordinate(row_norm, 2),
            DiscreteCoordinate(col_norm, 2),
            DiscreteCoordinate(area, 0),
            DiscreteCoordinate(aspect, 0),
            DiscreteCoordinate(layer_idx, 0),
        ]
        if extra_features:
            feat.extend(extra_features)

        node = GraphNode(
            node_id=-1,
            layer=0,
            source_image=source_image,
            feature_vector=feat,
            position_norm=(row_norm, col_norm),
            activation=1
        )
        nodes.append(node)

    return nodes


# ============================================================
# 从强度图构建初始图
# ============================================================

def build_graph_from_intensity(
    intensity: List[int],
    d: int,
    grid_size: int,
    source_image: int,
    step: int,
    strategy=None,
    max_layer_limit: Optional[int] = None
) -> DynamicGraph:
    """
    从原始强度图构建初始图结构
    底层复用 extract_layers，L0基于连通域
    """
    graph = DynamicGraph()

    # 1. 提取掩码层
    layers, layer_count = extract_layers(intensity, d=d, strategy=strategy)
    if layer_count == 0:
        return graph

    # 2. 计算层数
    if max_layer_limit is not None:
        stack_depth = max_layer_limit
    else:
        stack_depth = max(1, math.ceil(math.log2(grid_size)))
        cfg_depth = CONFIG.get("STACK_DEPTH", 0)
        if cfg_depth > 0:
            stack_depth = min(stack_depth, cfg_depth)

    # 3. L0层：从掩码连通域建节点
    l0_nodes = []
    for layer_idx in range(min(layer_count, stack_depth)):
        mask = layers[layer_idx]
        comps = mask_to_connected_components(mask, grid_size)
        extra = [DiscreteCoordinate(1, 0)]  # 标记来自掩码层
        nodes = components_to_l0_nodes(comps, grid_size, source_image, layer_idx, extra)
        l0_nodes.extend(nodes)

    l0_ids = []
    for node in l0_nodes:
        nid = graph.add_node(node)
        l0_ids.append(nid)

    if not l0_ids or stack_depth <= 1:
        return graph

    # 4. 逐层组合 (L1 ~ L_{depth-1})
    prev_ids = l0_ids
    for layer_idx in range(1, stack_depth):
        receptive_size = min(2 ** (layer_idx + 1), grid_size)
        new_nodes, parent_map = _compose_layer(
            prev_ids, graph, layer_idx, receptive_size,
            grid_size, source_image, step
        )
        new_ids = []
        for node, p_list in zip(new_nodes, parent_map):
            nid = graph.add_node(node)
            new_ids.append(nid)
            for pid in p_list:
                if pid in graph.nodes:
                    # 连接强度 0.10 (L2:I10)
                    graph.nodes[pid].neighbors[nid] = DiscreteCoordinate(10, 2)
        if not new_ids:
            break
        prev_ids = new_ids

    return graph


# ============================================================
# 神经元 → 图投影
# ============================================================

def project_neurons_to_graph(
    winner_neurons: List[Dict],
    intensity: List[int],
    d: int,
    grid_size: int,
    step: int,
    label: str
) -> DynamicGraph:
    """
    将神经元竞争结果投影为图节点
    这是图模式与神经元的唯一连接点
    """
    graph = DynamicGraph()
    if not winner_neurons:
        return graph

    # 收集所有获胜神经元的T掩码
    all_masks = []
    for n in winner_neurons:
        T = n.get("T", [])
        match_score = n.get("match", 50)
        base_val = n.get("base", 0)
        base_score = base_val.index if hasattr(base_val, 'index') else int(base_val)
        for layer_idx, mask in enumerate(T):
            if mask != 0:
                all_masks.append((mask, layer_idx, match_score, base_score))

    if not all_masks:
        return graph

    # 对每个掩码提取连通域作为L0节点
    for mask, layer_idx, match_score, base_score in all_masks:
        comps = mask_to_connected_components(mask, grid_size)
        extra = [
            DiscreteCoordinate(match_score, 0),   # 匹配度
            DiscreteCoordinate(base_score, 0),    # 基础速度
        ]
        nodes = components_to_l0_nodes(
            comps, grid_size, -1, layer_idx, extra
        )
        for node in nodes:
            node.label = label
            node.activation = match_score // 10
            node.last_updated = step
            node.source_neurons = [n.get("nid", -1) for n in winner_neurons]
            # 添加一层组合到L1（简化：每个L0节点直接提升为L1）
            # 这样保证图至少有两层
            graph.add_node(node)

    # 如果有节点，构建一层L1组合
    if graph.nodes:
        l0_ids = list(graph.nodes.keys())
        l1_nodes, parent_map = _compose_layer(
            l0_ids, graph, 1, 4, grid_size, -1, step
        )
        for node, p_list in zip(l1_nodes, parent_map):
            nid = graph.add_node(node)
            for pid in p_list:
                if pid in graph.nodes:
                    graph.nodes[pid].neighbors[nid] = DiscreteCoordinate(10, 2)

    # [v5.1] Establish real neighbor connections
    _connect_nearby_nodes(graph)

    return graph


def _connect_nearby_nodes(graph, max_distance: int = 2) -> None:
    """[v5.1] Establish neighbor connections for spatially close nodes"""
    from engine.config import SGNConstants
    nodes = list(graph.nodes.values())
    grid_factor = SGNConstants.POSITION_NORM // 10  # Coarse grid

    for i, n1 in enumerate(nodes):
        r1 = n1.position_norm[0] // grid_factor
        c1 = n1.position_norm[1] // grid_factor
        for n2 in nodes[i+1:]:
            r2 = n2.position_norm[0] // grid_factor
            c2 = n2.position_norm[1] // grid_factor
            if abs(r1 - r2) <= max_distance and abs(c1 - c2) <= max_distance:
                sim = graph.feature_similarity(n1.feature_vector, n2.feature_vector)
                if sim > 50:
                    n1.neighbors[n2.node_id] = DiscreteCoordinate(sim, 2)
                    n2.neighbors[n1.node_id] = DiscreteCoordinate(sim, 2)


# ============================================================
# 反馈重建
# ============================================================

def rebuild_with_feedback(
    base_graph: DynamicGraph,
    error_intensity: List[int],
    d: int,
    grid_size: int,
    step: int,
    label: str
) -> DynamicGraph:
    """
    基于误差图反馈重建图结构
    将误差图提取的新节点与base_graph融合
    """
    if not base_graph.nodes:
        return build_graph_from_intensity(
            error_intensity, d, grid_size, -1, step
        )

    # 从误差图提取L0节点
    error_graph = build_graph_from_intensity(
        error_intensity, d, grid_size, -1, step, max_layer_limit=1
    )
    error_l0 = error_graph.get_layer_nodes(0)

    if not error_l0:
        return base_graph

    base_l0 = base_graph.get_layer_nodes(0)

    # 如果没有base节点，直接添加误差节点
    if not base_l0:
        for node in error_l0:
            node.label = label
            base_graph.add_node(node)
        return base_graph

    # 差异化融合：对每个误差节点，找最近的base节点进行特征微调
    alpha_float = CONFIG.get("GRAPH_LEARNING_RATE", 0.3)
    alpha_int = int(alpha_float * 100)

    for e_node in error_l0:
        best_id, best_sim = base_graph.find_best_match(e_node, threshold=20)
        if best_id is not None:
            target = base_graph.nodes[best_id]
            # 微调特征
            min_len = min(len(target.feature_vector), len(e_node.feature_vector))
            for i in range(min_len):
                t_lvl = max(target.feature_vector[i].level, e_node.feature_vector[i].level)
                t_idx = target.feature_vector[i].to_level(t_lvl).index
                e_idx = e_node.feature_vector[i].to_level(t_lvl).index
                new_idx = (t_idx * (100 - alpha_int) + e_idx * alpha_int) // 100
                target.feature_vector[i] = DiscreteCoordinate(new_idx, t_lvl)
            target.activation += 1
            target.last_updated = step
        else:
            # 差异太大，作为新节点添加
            e_node.label = label
            e_node.last_updated = step
            base_graph.add_node(e_node)

    return base_graph


# ============================================================
# 内部辅助：层级组合
# ============================================================

def _compose_layer(
    parent_ids: List[int],
    graph: DynamicGraph,
    layer_idx: int,
    receptive_size: int,
    grid_size: int,
    source_image: int,
    step: int
) -> Tuple[List[GraphNode], List[List[int]]]:
    """
    组合上一层节点形成当前层节点
    返回: (新节点列表, 每个新节点对应的父节点ID列表)
    """
    if not parent_ids:
        return [], []

    parent_nodes = [graph.nodes[pid] for pid in parent_ids if pid in graph.nodes]
    if not parent_nodes:
        return [], []

    # 空间分桶
    bucket_size = 1000 // max(1, grid_size // receptive_size)
    buckets = defaultdict(list)

    for n in parent_nodes:
        br = n.position_norm[0] // max(1, bucket_size)
        bc = n.position_norm[1] // max(1, bucket_size)
        buckets[(br, bc)].append(n.node_id)

    new_nodes = []
    parent_map = []

    for (br, bc), pids in buckets.items():
        if not pids:
            continue

        p_nodes = [graph.nodes[pid] for pid in pids if pid in graph.nodes]
        if not p_nodes:
            continue

        # 组合特征：按维度取中位数
        max_len = max(len(n.feature_vector) for n in p_nodes)
        combined_feat = []

        for dim_idx in range(max_len):
            values = []
            for n in p_nodes:
                if dim_idx < len(n.feature_vector):
                    values.append(n.feature_vector[dim_idx])
            if not values:
                combined_feat.append(DiscreteCoordinate(0, 0))
                continue

            target_lvl = max(v.level for v in values)
            indices = [v.to_level(target_lvl).index for v in values]
            indices.sort()
            median_idx = indices[len(indices) // 2]
            combined_feat.append(DiscreteCoordinate(median_idx, target_lvl))

        # 位置：桶中心
        center_r = (br * bucket_size + bucket_size // 2)
        center_c = (bc * bucket_size + bucket_size // 2)
        center_r = min(1000, center_r)
        center_c = min(1000, center_c)

        # 确保前两位是空间坐标
        if len(combined_feat) >= 2:
            combined_feat[0] = DiscreteCoordinate(center_r, 2)
            combined_feat[1] = DiscreteCoordinate(center_c, 2)
        else:
            combined_feat = [
                DiscreteCoordinate(center_r, 2),
                DiscreteCoordinate(center_c, 2)
            ] + combined_feat

        # 计算平均激活
        avg_act = sum(n.activation for n in p_nodes) // max(1, len(p_nodes))
        avg_label = p_nodes[0].label if p_nodes else None

        node = GraphNode(
            node_id=-1,
            layer=layer_idx,
            source_image=source_image,
            feature_vector=combined_feat,
            position_norm=(center_r, center_c),
            activation=avg_act,
            last_updated=step,
            label=avg_label
        )
        new_nodes.append(node)
        parent_map.append(pids)

    # 限制每层节点数
    max_nodes_per_layer = CONFIG.get("MAX_NODES_PER_LAYER", 50)
    if len(new_nodes) > max_nodes_per_layer:
        scored = [(n.activation, i) for i, n in enumerate(new_nodes)]
        scored.sort(key=lambda x: -x[0])
        keep_indices = {i for _, i in scored[:max_nodes_per_layer]}

        filtered_nodes = []
        filtered_parent_map = []
        for i, node in enumerate(new_nodes):
            if i in keep_indices:
                filtered_nodes.append(node)
                filtered_parent_map.append(parent_map[i])
        return filtered_nodes, filtered_parent_map

    return new_nodes, parent_map
