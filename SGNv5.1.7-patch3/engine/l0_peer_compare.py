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
"""SGN-Lite v5.1.5 L0 同级比较与合并模块 - 从 core.py 提取

包含 L0 神经元模板的横向比较、贪心聚类、位级投票合并与跨层一致性过滤。
"""

from typing import List, Dict

from .config import CONFIG
from .utils import popcount


# ============================================================
# v5.1.6 3.2: L0 同级比较与合并（含补丁 P4 累积缓冲）
# ============================================================

def l0_peer_compare_and_merge(
    core,
    all_winners: List[List[Dict]],
    pool: List[Dict],
    batch: List
) -> List[List[Dict]]:
    """v5.1.6: L0 同级比较与合并（含补丁 P4 累积缓冲）

    对批次内所有 winner 神经元做横向比较：
    1. 收集所有被激活的神经元 ID（去重）
    2. 对每对计算模板相似度（位级 XOR + popcount）
    3. 相似度 ≥ L0_MERGE_SIMILARITY 的归为同一簇（贪心聚类）
    4. 簇内合并模板：位级投票保留强信号，丢弃孤立弱信号
    5. 补丁 P4：合并模板写入 T_pending，连续 N 批才写回 T

    Args:
        all_winners: List[List[{"nid","match","speed"}]]，每个样本一组
        pool: Layer 0 神经元池（core.neuron_layers[0]）
        batch: 原始批次（保留接口，当前未使用）

    Returns:
        all_winners（结构不变，pool 中 T 可能被 P4 缓冲写回更新）
    """
    merge_thresh = CONFIG.get("L0_MERGE_SIMILARITY", 0.75)
    weak_ratio = CONFIG.get("L0_WEAK_SIGNAL_RATIO", 0.15)
    bin_thresh = CONFIG.get("L0_BINARIZE_THRESH", 50)
    merge_buffer_thresh = CONFIG.get("L0_MERGE_BUFFER_THRESH", 2)

    # 1. 收集批次内所有被激活的神经元 ID
    active_ids = set()
    for winners in all_winners:
        for w in winners:
            active_ids.add(w["nid"])

    if len(active_ids) < 2:
        # 不足两个无法比较，但仍需重置参与者的 merge_count（P4 连续性）
        for nid in active_ids:
            pool[nid]["merge_count"] = 0
            pool[nid]["T_pending"] = None
        return all_winners

    # 2. 计算模板相似度，构建相似簇（贪心聚类）
    active_list = sorted(active_ids)
    clusters: List[set] = []
    assigned: set = set()
    for i, id_i in enumerate(active_list):
        if id_i in assigned:
            continue
        cluster = {id_i}
        assigned.add(id_i)
        for id_j in active_list[i + 1:]:
            if id_j in assigned:
                continue
            sim = template_similarity(core, pool[id_i], pool[id_j])
            if sim >= merge_thresh:
                cluster.add(id_j)
                assigned.add(id_j)
        clusters.append(cluster)

    # 3. 簇内合并模板 + 补丁 P4 累积缓冲
    merged_nids_this_batch: set = set()
    for cluster in clusters:
        if len(cluster) == 1:
            continue  # 单神经元簇无需合并
        cluster_Ts = [pool[nid]["T"] for nid in cluster]
        merged_T = merge_templates_by_voting(core, cluster_Ts, weak_ratio)
        merged_T = binarize_template(core, merged_T, bin_thresh)
        for nid in cluster:
            pool[nid]["T_pending"] = merged_T
            pool[nid]["merge_count"] = pool[nid].get("merge_count", 0) + 1
            merged_nids_this_batch.add(nid)
            # 连续 merge_buffer_thresh 批次都被合并 → 统一写回 T
            if pool[nid]["merge_count"] >= merge_buffer_thresh:
                pool[nid]["T"] = pool[nid]["T_pending"]
                # v5.1.7-patch3: L0 不设置 specialization（回归纯特征检测器）
                pool[nid]["merge_count"] = 0
                pool[nid]["T_pending"] = None

    # 补丁 P4：未参与合并的神经元重置 merge_count（保持"连续"语义）
    for nid in active_ids:
        if nid not in merged_nids_this_batch:
            pool[nid]["merge_count"] = 0
            pool[nid]["T_pending"] = None

    return all_winners


def template_similarity(core, n1: Dict, n2: Dict) -> float:
    """v5.1.6: 计算两个神经元模板的位级相似度 (0.0~1.0)

    T 是整数位掩码列表（List[int]），通过 XOR + popcount
    计算每层匹配位数，取所有层的平均匹配率。
    """
    t1, t2 = n1["T"], n2["T"]
    if not t1 or len(t1) != len(t2):
        return 0.0
    total_bits = len(t1) * core.D
    if total_bits == 0:
        return 0.0
    match_bits_total = sum(
        core.D - popcount(t1[ll] ^ t2[ll])
        for ll in range(len(t1))
    )
    return match_bits_total / total_bits


def merge_templates_by_voting(
    core, templates: List[List[int]], weak_ratio: float
) -> List[int]:
    """v5.1.6: 多模板位级投票合并

    T 是整数位掩码列表。对每层每个位：
    - 该位在 ≥ (1 - weak_ratio) 比例的模板中为 1 → 保留（强信号）
    - 否则 → 清除（孤立弱信号视为噪点）

    本质：2+ 个神经元共有的位被保留，仅 1 个神经元有的位被丢弃。
    """
    if not templates:
        return []
    n = len(templates)
    length = len(templates[0])
    # 共识阈值：至少 ceil(n * (1 - weak_ratio)) 个模板有该位才保留
    consensus_thresh = max(1, int(n * (1 - weak_ratio) + 0.999999))

    merged = []
    for ll in range(length):
        mask = 0
        for bit in range(core.D):
            cnt = 0
            for t in templates:
                if ll < len(t) and (t[ll] >> bit) & 1:
                    cnt += 1
            if cnt >= consensus_thresh:
                mask |= (1 << bit)
        merged.append(mask)
    return merged


def binarize_template(core, T: List[int], thresh: int) -> List[int]:
    """v5.1.6: 模板跨层一致性过滤（位掩码版"二值化"）

    T 是整数位掩码列表。对每个位位置，统计跨层被设置的次数：
    - 被设置次数 ≥ thresh% 的层 → 保留该位（跨层一致强信号）
    - 否则 → 清除该位（仅个别层出现的噪点）

    实现方式：构建跨层共识掩码，与每层位掩码做 AND。
    """
    n_layers = len(T)
    if n_layers == 0:
        return T
    # 至少 ceil(n_layers * thresh / 100) 层有该位才保留
    min_layers = max(1, (n_layers * thresh + 99) // 100)
    consensus = 0
    for bit in range(core.D):
        cnt = 0
        for ll in range(n_layers):
            if (T[ll] >> bit) & 1:
                cnt += 1
        if cnt >= min_layers:
            consensus |= (1 << bit)
    # 每层与共识掩码做 AND，仅保留跨层一致的位
    return [T[ll] & consensus for ll in range(n_layers)]
