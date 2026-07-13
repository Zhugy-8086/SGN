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
"""SGN-Lite v5.0 层处理核心算法模块 —— 从 sgn_utils 拆分

包含：popcount、match_bits、extract_layers、combine_layers、classify_sample
这些函数被核心引擎和评估层共享，独立为模块避免应用层污染。
"""

from __future__ import annotations

from typing import List, Tuple


# ============================================================
# 位运算核心
# ============================================================

def popcount(x: int) -> int:
    """通用位计数（支持任意位宽，随窗口大小自动扩展）"""
    return x.bit_count()


def match_bits(a: int, b: int, d: int = None) -> int:
    """匹配度 0-100（自动适应窗口大小）

    Args:
        a, b: 待比较的掩码整数
        d: 窗口像素总数（默认从 CONFIG['D'] 读取，运行时推导）
    """
    if d is None:
        from engine.config import CONFIG
        d = CONFIG.get("D", 64)
    mask = (1 << d) - 1
    same = popcount((~(a ^ b)) & mask)
    return (same * 100) // d


# ============================================================
# 层提取
# ============================================================

def extract_layers(intensity: List[int], layer_max: int = 4, d: int = 16, strategy=None) -> Tuple[List[int], int]:
    """二值化提取最多 layer_max 层掩膜（窗口大小自适应）

    v4.3 行为: 内部转发到 DefaultLayerExtractor（如果可用），
    否则回退到本机旧实现。对核心引擎透明。
    """
    # 优先使用 v4.2 的 DefaultLayerExtractor
    try:
        from engine.input import DefaultLayerExtractor
        extractor = DefaultLayerExtractor(layer_max=layer_max, d=d)
        return extractor.extract(intensity)
    except (ImportError, Exception):
        pass

    # 降级旧实现
    if d is None:
        from engine.config import CONFIG
        d = CONFIG.get("D", 64)

    layer_masks = []
    remaining = list(intensity[:d])

    if not remaining:
        return [], 0

    # 全亮画布特判：只有当剩余像素中 >128 的比例超过 90% 时才返回全掩码
    # 【v4.3-fix】避免稀疏矢量图形的背景噪声触发全亮误判
    if remaining and sum(1 for v in remaining if v > 128) / len(remaining) > 0.9:
        return [(1 << d) - 1], 1

    for _ in range(layer_max):
        active_pixels = sum(1 for v in remaining if v > 0)
        if strategy is not None and hasattr(strategy, 'get_mark_count'):
            mark_count = strategy.get_mark_count(d, active_pixels)
        else:
            # 【v4.3-fix】大窗口细线条图形使用更高标记比例，避免特征丢失
            if d >= 64:
                mark_count = max(max(3, d // 16), active_pixels * 4 // 5)
            else:
                mark_count = max(max(3, d // 8), active_pixels // 2)
        active = sorted(((v, i) for i, v in enumerate(remaining) if v > 0),
                        key=lambda x: (-x[0], x[1]))
        if not active:
            break
        mask = 0
        for _, idx in active[:mark_count]:
            mask |= (1 << idx)
            remaining[idx] = 0
        layer_masks.append(mask)
    return layer_masks, len(layer_masks)


def combine_layers(layers: List[int]) -> int:
    """将所有层的掩膜进行 OR 组合，生成更丰富的特征签名"""
    combined = 0
    for mask in layers:
        combined |= mask
    return combined


# ============================================================
# 样本分类（v4.3 从 sgn_utils / sgn_metrics 提取到此处，消除重复）
# ============================================================

def classify_sample(core, intensity: List[int], d: int = None) -> Tuple[str, int]:
    """对单个样本进行分类，返回 (预测标签, 最佳匹配度 0-100)

    提取自 sgn_visual.py / sgn_metrics.py / sgn_utils.py 中重复的模板匹配逻辑。
    """
    if d is None:
        d = getattr(core, 'D', 16)
        if d is None:
            from engine.config import CONFIG
            d = CONFIG.get("D", 64)

    layers, lc = extract_layers(intensity, d=d)
    if lc == 0:
        return '?', 0

    signature = combine_layers(layers)
    best_s, pred_lb = -1, '?'
    for tlb, tm, ts, thc in core.templates:
        s = match_bits(tm, signature, d=d)
        if s > best_s:
            best_s, pred_lb = s, tlb
    return pred_lb, best_s


def classify_multi_layer(core, intensity: List[int], d: int = None) -> Tuple[str, int]:
    """多层神经元模式推理（Bug #1 修复新增，v5.1.7 改造）

    v5.1.7 3.3: 推理时按桶匹配，每个标签的图是独立桶。
    v5.1.7 3.4: 当 L1_DECISION_LAYER 开启且有 cluster_id 时，
                 改用基于 cluster_id 的涌现分类（不依赖外部分类器遍历）。
    向后兼容：L1_DECISION_LAYER 关闭或无 cluster_id 时回退到遍历标签匹配。

    数据流与 _train_multi_layer 一致，但不做赫布学习/不更新图。
    """
    if d is None:
        from engine.config import CONFIG
        d = getattr(core, 'D', CONFIG.get("D", 64))

    grid_size = getattr(core, 'grid_size', int(d ** 0.5))

    # 1. 提取掩码层
    layers, layer_count = extract_layers(intensity, d=d)
    if layer_count == 0:
        return '?', 0

    # 2. Layer 0 神经元竞争
    layer0_pool = core.neuron_layers.get(0, [])
    matches_l0 = []
    for i, n in enumerate(layer0_pool):
        if n["L"]:
            continue
        match = core._match(n, layers, layer_count, intensity)
        speed = core._response_speed(n, match)
        matches_l0.append((speed, match, i))

    if not matches_l0:
        return '?', 0

    matches_l0.sort(reverse=True)
    top_k_l0 = core._top_k if hasattr(core, '_top_k') else None
    if top_k_l0 is None:
        from engine.config import CONFIG
        top_k_l0 = CONFIG.get("TOP_K", 10)
    winners_l0 = matches_l0[:top_k_l0]

    # 3. 无图时降级：L0 无 specialization（v5.1.7-patch3），返回 '?'
    if not hasattr(core, 'graphs') or not core.graphs:
        return '?', 0

    # 4. Layer 0 获胜者投影为图
    from graph.stack import project_neurons_to_graph
    proj_input = []
    for speed, match_score, idx in winners_l0:
        n = layer0_pool[idx]
        proj_input.append({
            "T": n["T"],
            "base": n["base"],
            "match": match_score,
            "nid": idx,
        })
    proj = project_neurons_to_graph(
        proj_input, intensity, d, grid_size, 0, "?"
    )
    if not proj.nodes:
        # v5.1.7-patch3: L0 无 specialization，降级返回 '?'
        return '?', 0

    # 5. v5.1.7-patch2: 统一 L1 分类逻辑（复用 compute_l1_predicted_label）
    # 消除 has_cluster 切换导致的训练/推理不一致，删除路径 A/B 约 60 行重复代码
    layer1_pool = core.neuron_layers.get(1, [])
    if not layer1_pool:
        return '?', 0

    l0_active_nids = [idx for _, _, idx in winners_l0]

    # 预计算图特征缓存（避免重复提取，与训练时一致）
    graph_features_cache = {}
    for graph_label, graph in core.graphs.items():
        gf, l0a = core._extract_graph_features(
            graph, d, grid_size, l0_active_nids
        )
        graph_features_cache[graph_label] = (gf, l0a)

    # 对所有已初始化的 L1 神经元做图匹配，取最佳
    best_score = -1
    best_label = '?'
    for n in layer1_pool:
        if n["L"] or not n.get("T_features_initialized"):
            continue
        # compute_l1_predicted_label 返回 (label, score)，避免重复调用 _match_layer1
        pred, pred_score = core._compute_l1_predicted_label(
            n, l0_active_nids, graph_features_cache
        )
        if pred is None:
            continue
        if pred_score > best_score:
            best_score = pred_score
            best_label = pred

    if best_score < 0:
        return '?', 0
    return best_label, best_score


# ============================================================
# 双图叠加门控识别架构（jnn.md v4.0）
# ============================================================

def extract_edge_map(intensity: List[int], d: int = None, threshold: int = 80) -> int:
    """算法 A：整数化边缘强度提取（高层特征）

    4-邻域梯度幅值近似：
    Grad(x,y) = |I(x,y) - I(x+1,y)| + |I(x,y) - I(x,y+1)|

    Args:
        intensity: 强度图 [0..255]
        d: 网格像素总数（自动推导网格边长）
        threshold: 边缘阈值（默认 80）

    Returns:
        边缘掩码 E（D位整数）
    """
    if d is None:
        from engine.config import CONFIG
        d = CONFIG.get("D", 64)
    G = int(d ** 0.5)
    if G * G != d:
        G = int(d ** 0.5)  # 非完全平方数时取整

    E = 0
    for r in range(G):
        for c in range(G):
            idx = r * G + c
            v_cur = intensity[idx] if idx < len(intensity) else 0
            # 右邻梯度
            diff_r = 0
            if c + 1 < G:
                idx_right = r * G + (c + 1)
                diff_r = abs(v_cur - (intensity[idx_right] if idx_right < len(intensity) else 0))
            # 下邻梯度
            diff_d = 0
            if r + 1 < G:
                idx_down = (r + 1) * G + c
                diff_d = abs(v_cur - (intensity[idx_down] if idx_down < len(intensity) else 0))
            grad = diff_r + diff_d
            if grad > threshold:
                E |= (1 << idx)
    return E


def extract_patch_vector(mask: int, d: int = None, patch_size: int = 2) -> List[int]:
    """算法 C：分块局部模式编码

    将全局掩码拆解为若干 P×P 小块，每个小块编码为一个整数 atom。

    Args:
        mask: 全局掩码（D位整数）
        d: 网格像素总数
        patch_size: 块大小 P（默认 2）

    Returns:
        局部模式向量 V（长度 = (G/P)^2 的整数列表）
    """
    if d is None:
        from engine.config import CONFIG
        d = CONFIG.get("D", 64)
    G = int(d ** 0.5)
    P = patch_size
    unit = G // P
    if unit <= 0:
        return []

    V = []
    for br in range(unit):
        for bc in range(unit):
            atom = 0
            for dr in range(P):
                for dc in range(P):
                    idx = (br * P + dr) * G + (bc * P + dc)
                    bit = (mask >> idx) & 1
                    atom = (atom << 1) | bit
            V.append(atom)
    return V


def patch_score(V_in: List[int], V_tpl: List[int]) -> int:
    """算法 E：拼接匹配度计算

    逐个部件匹配，精确匹配或模糊匹配（汉明距离 <= 1）。

    Args:
        V_in: 输入的原子向量
        V_tpl: 模板的原子向量

    Returns:
        匹配度 (0~100)
    """
    if not V_in or not V_tpl:
        return 0
    N = min(len(V_in), len(V_tpl))
    match_count = 0
    for i in range(N):
        # 精确匹配
        if V_in[i] == V_tpl[i]:
            match_count += 1
        # 模糊匹配：汉明距离 <= 1
        elif popcount(V_in[i] ^ V_tpl[i]) <= 1:
            match_count += 1
    return (match_count * 100) // N


def scale_mask(mask: int, src_size: int, dst_size: int) -> int:
    """整数缩放掩码（最近邻，使用 DiscreteCoordinate level/index）

    使用 level=3（千分之一精度）表示缩放比例，全程纯整数。

    Args:
        mask: 源掩码
        src_size: 源网格边长
        dst_size: 目标网格边长

    Returns:
        缩放后的掩码
    """
    if src_size == dst_size:
        return mask

    # 缩放比例：scale_index = src_size * 1000 // dst_size
    scale_level = 3
    scale_index = (src_size * 1000) // dst_size
    scale_divisor = 10 ** scale_level  # = 1000

    M_scaled = 0
    for r in range(dst_size):
        for c in range(dst_size):
            # 坐标映射：目标坐标 → 源坐标（纯整数）
            src_r = (r * scale_index) // scale_divisor
            src_c = (c * scale_index) // scale_divisor
            idx_src = src_r * src_size + src_c
            idx_dst = r * dst_size + c
            if idx_src < src_size * src_size and (mask >> idx_src) & 1:
                M_scaled |= (1 << idx_dst)
    return M_scaled
