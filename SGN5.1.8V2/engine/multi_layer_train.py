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
"""SGN-Lite v5.1.5 多层+分批训练模块（从 core.py 抽取）"""

from typing import List, Dict

from .config import CONFIG, DiscreteCoordinate
from .utils import extract_layers
from graph.graph import DynamicGraph
from graph.stack import project_neurons_to_graph


# ============================================================
# v5.1 多层神经元训练
# ============================================================

def train_multi_layer(core, intensity: List[int], label: str) -> Dict:
    """多层神经元训练 - 单样本入口（向后兼容）

    v5.1.6: 拆分为 L0 阶段（竞争）+ L1 阶段（图投影/竞争/学习），
    L1 阶段由 multi_layer_l1_phase 共用，供批次模式复用。
    """
    d = core.D
    core._step_counter += 1

    # 1. 提取掩码层
    layers, layer_count = extract_layers(intensity, d=d, strategy=core.layer_strategy)
    if layer_count == 0:
        return make_multi_layer_info(core, label, False, 0, 0, 0)

    # 2. Layer 0 神经元竞争（v5.1.6：应用自适应静默）
    winners_l0 = l0_compete(core, intensity, layers, layer_count)
    if not winners_l0:
        return make_multi_layer_info(core, label, False, 0, 0, 0)

    # 3-11. L1 阶段（图投影 → L1 竞争 → 学习）
    return multi_layer_l1_phase(core, winners_l0, layers, layer_count, intensity, label)


def l0_compete(core, intensity, layers, layer_count) -> List[Dict]:
    """v5.1.6: Layer 0 竞争（抽取自 train_multi_layer，供单样本/批次共用）

    Returns:
        winners_l0: List[{"nid","match","speed"}]，空列表表示无可用神经元
    """
    layer0_pool = core.neuron_layers[0]
    active_l0 = core._apply_adaptive_silence(layer0_pool)
    matches_l0 = []
    for i in active_l0:
        n = layer0_pool[i]
        if n["L"] or n.get("silenced", False):
            continue
        match = core._match(n, layers, layer_count, intensity)
        speed = core._response_speed(n, match)
        matches_l0.append((speed, match, i))

    if not matches_l0:
        return []

    matches_l0.sort(reverse=True)
    top_k_l0 = CONFIG["TOP_K"]
    # 统一为 Dict 格式（避免 tuple/dict 混用 Bug 7）
    return [
        {"nid": idx, "match": m, "speed": s}
        for s, m, idx in matches_l0[:top_k_l0]
    ]


def multi_layer_l1_phase(
    core, winners_l0: List[Dict], layers, layer_count: int,
    intensity: List[int], label: str
) -> Dict:
    """v5.1.7: 多层训练 L1 阶段（桶化投影 → L1 竞争+输出模式 → 学习 → 回馈 → 返回 info）

    v5.1.6: 从 train_multi_layer 抽取，供单样本与批次模式共用。
    v5.1.7 3.3: 图投影改为桶化+链条化，按标签分桶防串位。
    v5.1.7 3.4: L1 竞争增加输出模式生成，学习后增加回馈图模式。
    数据流：winners_l0 → 分桶 → 链条化 → 每桶独立投影 → 图融合
          → extract_graph_features → Layer 1 竞争+输出模式 → 赫布学习 → 回馈
    """
    d = core.D
    grid_size = core.grid_size
    layer0_pool = core.neuron_layers[0]
    layer1_pool = core.neuron_layers[1]

    # ---- 3. v5.1.7: 桶化+链条化分桶投影 ----
    sample = (intensity, label)
    if CONFIG.get("L1_BUCKET_ENABLED", True):
        buckets = core._bucket_winners_by_label(winners_l0, layer0_pool, sample)
    else:
        buckets = {"?" if not label else label: winners_l0}

    total_proj_nodes = 0
    primary_graph_features = []
    primary_l0_active_list = []
    primary_bucket_label = list(buckets.keys())[0] if buckets else "?"

    for bucket_label, bucket_winners in buckets.items():
        if not bucket_winners:
            continue
        # v5.1.7: 桶内链条化
        if CONFIG.get("L1_CHAIN_ENABLED", True):
            bucket_winners = core._chain_winners(bucket_winners, layer0_pool)

        # 投影到该桶的子图
        proj_input = []
        for w in bucket_winners:
            n = layer0_pool[w["nid"]]
            proj_input.append({
                "T": n["T"],
                "base": n["base"],
                "match": w["match"],
                "nid": w["nid"],
            })
        proj = project_neurons_to_graph(
            proj_input, intensity, d, grid_size,
            core._step_counter, bucket_label
        )

        if not proj.nodes:
            continue

        total_proj_nodes += len(proj.nodes)

        # 图赫布融合到对应桶的图
        graph_key = bucket_label if bucket_label else "?"
        if graph_key not in core.graphs:
            core.graphs[graph_key] = DynamicGraph()
        for nid, node in proj.nodes.items():
            core.graphs[graph_key].hebbian_merge(
                node, core._step_counter,
                lr_alpha_pct=int(CONFIG.get("GRAPH_LEARNING_RATE", 0.3) * 100)
            )

        # 提取该桶图特征（v5.1.9: 不再传 l0_active_nids）
        graph_features = core._extract_graph_features(
            core.graphs[graph_key], d, grid_size
        )
        l0_active_list = [w["nid"] for w in bucket_winners]

        # 记录主桶特征（用于 L1 竞争）
        if bucket_label == label:
            primary_graph_features = graph_features
            primary_l0_active_list = l0_active_list
            primary_bucket_label = bucket_label

    if total_proj_nodes == 0:
        verified_l0 = core._verify(intensity, layers, layer_count)
        core._hebbian_multi_layer(winners_l0, verified_l0, layer=0, label=label)
        info = make_multi_layer_info(core, label, verified_l0, 0, 0, 0)
        info["_l1_outputs"] = []
        return info

    # ---- 6. Layer 1 神经元竞争 ----
    active_l1 = core._apply_adaptive_silence(layer1_pool)
    matches_l1 = []
    for i in active_l1:
        n = layer1_pool[i]
        if n["L"] or n.get("silenced", False):
            continue
        match_l1 = core._match_layer1(n, primary_graph_features, primary_l0_active_list)
        speed_l1 = core._response_speed(n, match_l1)
        matches_l1.append({
            "nid": i,
            "match": match_l1,
            "speed": speed_l1,
        })

    if not matches_l1:
        verified_l0 = core._verify(intensity, layers, layer_count)
        core._hebbian_multi_layer(winners_l0, verified_l0, layer=0, label=label)
        info = make_multi_layer_info(core, label, verified_l0, 0, total_proj_nodes, 0)
        info["_l1_outputs"] = []
        return info

    matches_l1.sort(key=lambda x: x["speed"], reverse=True)
    top_k_l1 = CONFIG.get("TOP_K_L1", 4)
    winners_l1 = matches_l1[:top_k_l1]

    # 7. Layer 1 神经元首次学习：写入 T_features 和 T_l0_active
    for w in winners_l1:
        n = layer1_pool[w["nid"]]
        if not n["T_features_initialized"]:
            if not primary_graph_features:
                continue
            n["T_features"] = list(primary_graph_features)
            n["T_l0_active"] = list(primary_l0_active_list)
            n["T_features_initialized"] = True
        n["source_layer0"] = [w["nid"] for w in winners_l0]
        # v5.1.7-patch2: 删除隐式注入，改为 _hebbian_multi_layer 的显式参数

    # 8. 最终分类验证（基于图匹配的 L1 预测标签）
    # 将 L1 神经元的 T_features 与所有图匹配，最佳匹配图标签作为预测标签。
    l0_active_nids = [w["nid"] for w in winners_l0]
    l0_verified = core._verify(intensity, layers, layer_count)

    # 预计算所有图的图特征（v5.1.9: 缓存只存 gf，l0_active_nids 实时传入 match_layer1）
    graph_features_cache = {}
    for graph_label, graph in core.graphs.items():
        gf = core._extract_graph_features(graph, d, grid_size)
        graph_features_cache[graph_label] = gf

    l1_verified = False
    if winners_l1:
        best_l1 = layer1_pool[winners_l1[0]["nid"]]
        # v5.1.7-patch2: compute_l1_predicted_label 返回 (label, score)
        pred_label, _ = core._compute_l1_predicted_label(
            best_l1, l0_active_nids, graph_features_cache
        )
        if pred_label is None:
            # 冷启动：L1 模板未初始化，用 L0 校验作为弱信号
            l1_verified = l0_verified
        else:
            l1_verified = (pred_label == label)

    # 9. 赫布学习（两层各自独立，用各自的校验信号）
    core._hebbian_multi_layer(winners_l0, l0_verified, layer=0, label=label)
    # v5.1.7-patch2: 显式传递 graph_features/l0_active_list，不再隐式注入 winners dict
    core._hebbian_multi_layer(
        winners_l1, l1_verified, layer=1, label=label,
        graph_features=primary_graph_features,
        l0_active_list=primary_l0_active_list,
    )

    # 10. enc_r 鼓励衰减（全局 O(N)）
    for pool in core.neuron_layers.values():
        for n in pool:
            if n["enc_r"] > 0:
                n["enc_r"] -= 1
                if n["enc_r"] == 0:
                    n["enc_b"] = DiscreteCoordinate(0, n["enc_b"].level)

    # 11. 返回 info
    info = make_multi_layer_info(
        core, label, l1_verified, matches_l1[0]["match"],
        total_proj_nodes, len(winners_l1)
    )

    return info


def make_multi_layer_info(
    core, label: str, verified: bool, match: int,
    graph_nodes: int, layer1_active: int
) -> Dict:
    """构造多层训练 info 字典"""
    l0_active = sum(1 for n in core.neuron_layers.get(0, []) if not n["L"])
    l1_active = sum(1 for n in core.neuron_layers.get(1, []) if not n["L"])
    return {
        "label": label,
        "layer_count": 0,
        "V": verified,
        "match": match,
        "base": 0,
        "winners": [],
        "active": l0_active + l1_active,
        "locked": 0,
        "templates": len(core.graphs),
        "_D": core.D,
        "multi_layer": True,
        "layer0_active": l0_active,
        "layer1_active": l1_active,
        "graph_nodes": graph_nodes,
        "step": core._step_counter,
    }


# ============================================================
# v5.1.6 分批次训练
# ============================================================

def build_sample_pool(samples=None) -> List:
    """v5.1.6: 构建样本池（按标签交错排列，保证批次内标签多样）

    Args:
        samples: 外部生成的 (intensity, label) 样本列表；
                 None 时返回空列表（由外层自行提供样本）
    Returns:
        重排后的样本列表：同标签样本被分散，使每个批次包含多种标签
    """
    if not samples:
        return []
    # 按标签分组
    by_label = {}
    for s in samples:
        label = s[1]
        by_label.setdefault(label, []).append(s)
    # 交错排列：轮流从各标签取一个，使相邻样本标签不同
    pool = []
    labels = sorted(by_label.keys())
    while any(by_label.values()):
        for label in labels:
            if by_label[label]:
                pool.append(by_label[label].pop(0))
    return pool


def train_batch(core, batch: List) -> List[Dict]:
    """v5.1.6: 批次训练入口

    与 train() 的区别：
    1. 批次内所有样本先各自跑 L0 竞争（多层模式）
    2. 收集批次内所有 winner（3.2 将在此插入同级比较与合并）
    3. 合并后再统一进入 L1 / 图模式
    4. 返回每个样本的 info 列表（history 由外层统一 append）

    Args:
        batch: [(intensity, label), ...] 样本列表
    """
    if not batch:
        return []
    # 窗口大小识别（按首个样本校准，同批维度应一致）
    first_intensity = batch[0][0]
    if len(first_intensity) != core.D:
        core._rebuild_for_dimension(len(first_intensity))
    # 登记标签
    for sample in batch:
        core.label_set.add(sample[1])
    # 分派
    if core.multi_layer_enabled:
        return train_batch_multi_layer(core, batch)
    elif core.graph_mode:
        return core._train_batch_graph(batch)
    else:
        return core._train_batch_full(batch)


def train_batch_multi_layer(core, batch: List) -> List[Dict]:
    """v5.1.6: 批次多层训练

    Phase 1: 批次内所有样本各自 L0 竞争，收集 all_winners
    (Phase 1.5: 同级比较与合并 —— 3.2 实现，此处直通)
    Phase 2: 每个样本用 winners_l0 进入 L1 阶段
    """
    d = core.D
    top_k_l0 = CONFIG["TOP_K"]

    # Phase 1: 批次内 L0 竞争
    all_winners = []   # List[List[Dict]]
    sample_ctx = []    # List[(layers, layer_count, intensity, label)]
    for sample in batch:
        intensity, label = sample  # 适配外部 (intensity, label) 格式
        layers, layer_count = extract_layers(intensity, d=d, strategy=core.layer_strategy)
        if layer_count == 0:
            all_winners.append([])
            sample_ctx.append((layers, layer_count, intensity, label))
            continue
        winners = l0_compete(core, intensity, layers, layer_count)
        all_winners.append(winners)
        sample_ctx.append((layers, layer_count, intensity, label))

    # Phase 1.5: 同级比较与合并（v5.1.6 3.2）
    if CONFIG.get("L0_PEER_COMPARE_ENABLED", True):
        core._l0_peer_compare_and_merge(
            all_winners, core.neuron_layers[0], batch
        )

    # Phase 2: 每个样本进入 L1 阶段
    batch_info = []
    for idx, (layers, layer_count, intensity, label) in enumerate(sample_ctx):
        # Bug #1 修复: _step_counter 必须在每个样本处理前递增，
        # 否则所有图操作（投影/赫布融合/节点老化）都使用 step=0
        core._step_counter += 1
        winners_l0 = all_winners[idx]
        if not winners_l0:
            info = make_multi_layer_info(core, label, False, 0, 0, 0)
            batch_info.append(info)
            continue
        info = multi_layer_l1_phase(core, winners_l0, layers, layer_count, intensity, label)
        batch_info.append(info)

    return batch_info
