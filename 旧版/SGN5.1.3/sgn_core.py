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
"""SGN-Lite v5.0 核心引擎模块 - 整数化竞争/校验/学习/模板合并"""

import random
from typing import List, Dict, Optional, Tuple

from sgn_config import CONFIG, D, DiscreteCoordinate, SGNConstants
from sgn_utils import popcount, match_bits, extract_layers, combine_layers, C
from sgn_graph import DynamicGraph, GraphNode
from sgn_stack import project_neurons_to_graph, rebuild_with_feedback
from sgn_merge import merge_winner_projections
from sgn_graph_match import graph_similarity, classify_with_graph


def create_neuron(nid, d=16):
    """创建神经元 - 随机整数模板初始化（窗口大小自适应 + 离散坐标）

    Args:
        nid: 神经元编号
        d: 窗口像素总数（4×4=16, 8×8=64, 16×16=256）
    """
    from sgn_config import DiscreteCoordinate
    rd = random.Random(CONFIG["SEED"] + nid)
    mask_max = (1 << d) - 1  # 动态位宽：16→0xFFFF, 64→2^64-1
    base_dc = CONFIG["BASE_INIT"]  # DiscreteCoordinate 对象
    return {
        "T": [rd.randint(0, mask_max) for _ in range(CONFIG["LAYER_MAX"])],
        "base": base_dc,  # 离散坐标，全程不还原
        "lock": 0,
        "enc_r": 0,
        "enc_b": DiscreteCoordinate(0, base_dc.level),  # 初始为 0，同层级
        "L": False,
        # v5.1 多层神经元新增字段
        "gate": DiscreteCoordinate(100, 2),        # 门控强度 (0~100)
        "specialization": None,                      # 专精标签 (None=通用)
        "consecutive_verified": 0,                   # 连续验证通过计数
        "layer": 0,                                  # 所属神经元层 (0=Layer0, 1=Layer1)
        # Layer 1 专用字段（Layer 0 神经元忽略这些）
        "T_features": [],                            # 图特征模板 (List[DiscreteCoordinate])
        "T_l0_active": [],                           # Layer 0 激活 ID 列表
        "T_features_initialized": False,             # 特征模板是否已学习
    }


class SGNCore:
    """SGN核心引擎 (v5.0 策略插件化 + 图模式层级记忆)"""

    def __init__(self, seed=None, layer_strategy=None, verify_strategy=None):
        if seed is not None:
            random.seed(seed)
            CONFIG["SEED"] = seed
        # D 初始为 16（向后兼容），首次 train() 时根据 len(intensity) 自动校准
        self.D = CONFIG.get("D", 16)
        self.grid_size = int(self.D ** 0.5)  # 4, 8, 16...

        # 【v4.3 策略注入】解耦 4x4 硬编码阈值
        self.layer_strategy = layer_strategy
        self.verify_strategy = verify_strategy
        self._sync_strategies()  # 根据当前 D 自动匹配策略

        # 大窗口自动扩大神经元/模板库
        self._auto_scale_resources()

        # v5.1 多层神经元架构
        self.multi_layer_enabled = CONFIG.get("ENABLE_MULTI_LAYER_NEURON", False)
        if self.multi_layer_enabled:
            l0_count = CONFIG.get("NEURON_LAYER_0_COUNT", 128)
            l1_count = CONFIG.get("NEURON_LAYER_1_COUNT", 64)
            self.neuron_layers = {
                0: [create_neuron(i, self.D) for i in range(l0_count)],
                1: [create_neuron(i + l0_count, self.D) for i in range(l1_count)],
            }
            # 向后兼容：self.N 指向 Layer 0
            self.N = self.neuron_layers[0]
            # 标记 Layer 1 神经元的 layer 字段
            for n in self.neuron_layers[1]:
                n["layer"] = 1
            # Bug 11 修复：多层模式也需要 self.graphs（图中间层汇总器）
            if not hasattr(self, 'graphs') or not self.graphs:
                self.graphs = {}
        else:
            self.N = [create_neuron(i, self.D) for i in range(CONFIG["MAX_NEURONS"])]
            self.neuron_layers = {0: self.N}
        self.templates = []       # (label, mask, success_count, hit_counter)
        self._template_index: Dict[str, list] = {}  # [v5.1] label -> template index list
        self.history = []
        self.blackbox_log = []    # 黑箱验证记录
        self.label_set = set()    # 训练中出现过的所有标签

        # 【v4.4 双图叠加门控识别】新增成员
        self.hist_buffer = []     # 历史样本缓冲区 [(intensity, label), ...]
        self.atom_dict = {}       # label → {atom → count}，按类别原子字典
        self.high_templates = {}  # label → 边缘掩码（高层特征模板）
        self.low_templates = {}   # label → 像素掩码（低层特征模板）
        self.use_gate_matching = CONFIG.get("ENABLE_GATE_MATCHING", False)
        self.patch_size = CONFIG.get("PATCH_SIZE", 2)

        # 【v5.0 图模式】新增成员
        self.graph_mode = CONFIG.get("ENABLE_GRAPH_MODE", False)
        self.graphs: Dict[str, DynamicGraph] = {}
        self._step_counter = 0

        # 如果图模式开启，冻结模板系统
        if self.graph_mode:
            self._template_backup = list(self.templates)
            self.templates = []

        # 匹配策略（可插拔）
        from sgn_strategies import GlobalMatchStrategy
        self.match_strategy = GlobalMatchStrategy()  # 默认传统匹配

    def _sync_strategies(self):
        """根据当前 D 同步策略（维度变更时调用）"""
        from sgn_strategies import StrategyRegistry, DefaultLayerStrategy, DefaultVerifyStrategy
        if self.layer_strategy is None or not self.layer_strategy.is_applicable(self.D):
            self.layer_strategy = StrategyRegistry.auto_select_layer(self.D)
            if self.layer_strategy is None:
                self.layer_strategy = DefaultLayerStrategy()
        if self.verify_strategy is None:
            self.verify_strategy = StrategyRegistry.auto_select_verify(self.grid_size)
            if self.verify_strategy is None:
                self.verify_strategy = DefaultVerifyStrategy()

    def _auto_scale_resources(self):
        """大窗口自动扩大神经元/模板库上限"""
        from sgn_config import ConfigRegistry
        neurons = CONFIG["MAX_NEURONS"]
        templates = CONFIG.get("MAX_TEMPLATES", 500)
        if self.D >= 4096:   # 64×64+
            neurons = max(neurons, 512)
            templates = max(templates, 1200)
        elif self.D >= 1024:   # 32×32+
            neurons = max(neurons, 384)
            templates = max(templates, 800)
        elif self.D >= 256:  # 16×16+
            neurons = max(neurons, 256)
            templates = max(templates, 500)
        if neurons != CONFIG["MAX_NEURONS"]:
            ConfigRegistry._values["MAX_NEURONS"] = neurons
        if templates != CONFIG.get("MAX_TEMPLATES", 500):
            ConfigRegistry._values["MAX_TEMPLATES"] = templates

    # ---- 响应速度计算 [v4.3 全程整数化] ----
    def _response_speed(self, n, match):
        """响应速度 = base.index + match * gamma.index（全程整数）

        【v4.3-fix】删除错误的 GAMMA_DENOM=100 除法。
        交接文档原版的 GAMMA_DENOM 设计是错误的：
        gamma 已经是 DiscreteCoordinate，其 index 已经包含了 scale 信息。
        再除以 100 会导致增益被压缩 100 倍，响应速度排序失效，学习停滞。

        正确做法：统一层级后，base.index 和 gamma.index 直接相加。
        match 是 0~100 的整数百分比，gamma.index 是同一 scale 下的坐标，
        乘积自然落在合理的整数范围内，无需额外归一化。
        """
        from sgn_config import DiscreteCoordinate
        gamma = CONFIG["GAMMA"]  # DiscreteCoordinate
        base = n["base"]         # DiscreteCoordinate

        # 统一到 gamma 的层级（或任何共同层级，只要一致）
        if base.level != gamma.level:
            base = base.to_level(gamma.level)

        # 直接相加，无需额外的除法或缩放
        # base.index 和 gamma.index 是同一 scale 下的整数
        rsp = base.index + match * gamma.index

        if n["enc_r"] > 0:
            enc = n["enc_b"]
            if enc.level != gamma.level:
                enc = enc.to_level(gamma.level)
            rsp += enc.index

        return rsp

    # ---- 匹配计算 [v4.2 动态维度] ----
    def _match(self, n, layers, layer_count, intensity=None):
        """匹配计算（支持门控匹配和传统匹配）

        【v4.4】如果启用门控匹配，使用高层（边缘）AND 低层（像素）双重匹配。
        否则使用传统全局 XOR 匹配。

        Args:
            n: 神经元
            layers: 掩码列表
            layer_count: 层数
            intensity: 原始强度图（门控匹配时需要）
        """
        if self.use_gate_matching:
            return self._gate_match(n, layers, layer_count, intensity)
        else:
            # 传统匹配：全局 XOR
            t = sum(
                self.D - popcount(n["T"][ll] ^ layers[ll])
                for ll in range(min(layer_count, CONFIG["LAYER_MAX"]))
            )
            denom = min(layer_count, CONFIG["LAYER_MAX"]) * self.D
            return (t * 100) // denom if denom > 0 else 0

    def _gate_match(self, n, layers, layer_count, intensity=None):
        """门控匹配：高层（边缘）AND 低层（像素）

        使用边缘特征（高层）和像素特征（低层）进行门控匹配。
        只有高层匹配且低层匹配时，才返回有效匹配度。

        Args:
            n: 神经元
            layers: 掩码列表
            layer_count: 层数
            intensity: 原始强度图（用于边缘提取）
        """
        if layer_count == 0:
            return 0

        high_in, low_in = self._extract_gate_features(layers, layer_count, intensity)

        if not (hasattr(self, 'high_templates') and self.high_templates):
            return self._compute_traditional_match(n, layers, layer_count)

        best_score = 0
        for label, indices in self._template_index.items():
            high_tpl = self.high_templates.get(label)
            low_tpl = self.low_templates.get(label)
            if high_tpl is None or low_tpl is None:
                continue

            for idx in indices:
                tlb, tm, sc, thc = self.templates[idx]
                score = self._compute_gate_score(high_in, low_in, high_tpl, low_tpl)
                if score > best_score:
                    best_score = score

        return best_score if best_score > 0 else self._compute_traditional_match(n, layers, layer_count)

    def _extract_gate_features(self, layers, layer_count, intensity):
        """Extract high-level/low-level features needed for gate matching (compute once)"""
        from sgn_layers import extract_edge_map
        input_mask = layers[0]
        if intensity is not None:
            high_in = extract_edge_map(intensity, self.D)
        else:
            intensity_from_mask = [int((input_mask >> i) & 1) * 255 for i in range(self.D)]
            high_in = extract_edge_map(intensity_from_mask, self.D)
        return high_in, input_mask

    def _compute_gate_score(self, high_in, low_in, high_tpl, low_tpl):
        """Compute gate score for single template (integerized)"""
        from sgn_layers import popcount as _popcount

        low_union = _popcount(low_in | low_tpl)
        low_match = (_popcount(low_in & low_tpl) * SGNConstants.PERCENT_BASE) // max(1, low_union)
        low_thresh = CONFIG.get("GATE_LOW_THRESH", 30)
        if low_match < low_thresh:
            return 0

        high_union = _popcount(high_in | high_tpl)
        high_match = (_popcount(high_in & high_tpl) * SGNConstants.PERCENT_BASE) // max(1, high_union)
        high_thresh = CONFIG.get("GATE_HIGH_THRESH", 40)
        if high_match < high_thresh:
            return 0

        return (high_match + low_match) // 2

    def _compute_traditional_match(self, n, layers, layer_count):
        """Pre-compute traditional match score (avoid repetition in loop)"""
        from sgn_layers import popcount as _popcount
        t = 0
        max_layer = min(layer_count, CONFIG["LAYER_MAX"])
        for ll in range(max_layer):
            t += self.D - _popcount(n["T"][ll] ^ layers[ll])
        denom = max_layer * self.D
        return (t * SGNConstants.PERCENT_BASE) // denom if denom > 0 else 0

    # ---- 赫布学习 [v4.3 全程整数化 + 鼓励触发] ----
    def _hebbian_learn(self, winners, verified):
        """赫布学习 + 鼓励触发（全程整数坐标位移）

        验证通过时：
          1. 永久增强 base.index（赫布学习，慢变量）
          2. 触发鼓励 enc_r = ENCOURAGE_CNT（短期脉冲，快变量）
          3. 同步 enc_b（锁定到神经元，离散坐标）

        验证失败时：
          1. 削弱 base.index
          2. 增加 lock 计数，可能触发锁定
          3. 不触发鼓励

        【v4.3 离散坐标】
        n["base"] 和 n["enc_b"] 是 DiscreteCoordinate。
        所有运算都是同一层级下的整数坐标位移。
        """
        from sgn_config import DiscreteCoordinate
        participants = len(winners)
        if participants == 0:
            return

        # 读取离散坐标参数
        lr = CONFIG["LEARNING_RATE"]       # DiscreteCoordinate
        wr = CONFIG["WEAKEN_RATE"]         # DiscreteCoordinate
        sat = CONFIG["SPEED_SAT"]          # DiscreteCoordinate
        min_base = CONFIG["MIN_BASE"]      # DiscreteCoordinate
        enc_bonus = CONFIG["ENCOURAGE_BONUS"]  # DiscreteCoordinate
        enc_cnt = CONFIG["ENCOURAGE_CNT"]

        # 【防御】确保所有参数都是 DiscreteCoordinate
        required_dc = [("LEARNING_RATE", lr), ("WEAKEN_RATE", wr), ("SPEED_SAT", sat),
                       ("MIN_BASE", min_base), ("ENCOURAGE_BONUS", enc_bonus)]
        for name, obj in required_dc:
            if not hasattr(obj, 'level'):
                raise TypeError(f"CONFIG['{name}'] 不是 DiscreteCoordinate，类型={type(obj).__name__}，"
                                f"请检查配置文件或恢复默认配置")

        # 统一层级（通常所有参数已在同一层级初始化）
        target_level = max(lr.level, wr.level, sat.level, min_base.level, enc_bonus.level)
        lr = lr.to_level(target_level)
        wr = wr.to_level(target_level)
        sat = sat.to_level(target_level)
        min_base = min_base.to_level(target_level)
        enc_bonus = enc_bonus.to_level(target_level)

        # 学习步长 = max(1, lr.index // participants)
        delta = max(1, lr.index // participants)

        for nid in winners:
            n = self.N[nid]
            if n["L"]:
                continue

            # 确保神经元 base 在同一层级
            base = n["base"]
            if base.level != target_level:
                base = base.to_level(target_level)

            if verified:
                # 1. 赫布学习（永久）：base.index += delta
                new_index = base.index + delta
                if new_index <= sat.index:
                    n["base"] = DiscreteCoordinate(new_index, target_level)
                else:
                    n["base"] = sat  # 饱和
                # 2. 鼓励触发（短期脉冲）
                n["enc_r"] = enc_cnt
                n["enc_b"] = enc_bonus
            else:
                # 削弱：base.index -= wr.index
                new_index = base.index - wr.index
                if new_index > min_base.index:
                    n["base"] = DiscreteCoordinate(new_index, target_level)
                else:
                    n["base"] = min_base  # 不低于最小值
                n["lock"] += 1
                if n["lock"] >= CONFIG["MAX_LOCKOUT"]:
                    n["L"] = True

    # ---- 校验 [v4.2 交叉相乘+稀疏保护+动态网格] ----
    def _verify(self, intensity, layers, layer_count):
        if layer_count == 0:
            return False

        m0 = layers[0]
        marked = [i for i in range(self.D) if (m0 >> i) & 1]
        mc = len(marked)
        if mc == 0:
            return False

        # 【v4.3 策略化】稀疏保护阈值由 LayerStrategy 提供
        min_marked = self.layer_strategy.get_min_marked(self.D)
        if mc < min_marked:
            return False

        total_raw = sum(intensity)

        # 交叉相乘避除法
        # 正确条件: marked_avg - unmarked_avg > THRESH
        if mc < self.D:
            marked_sum = sum(intensity[i] for i in marked)
            unmarked_sum = total_raw - marked_sum
            uc = self.D - mc
            lhs = marked_sum * uc
            rhs = unmarked_sum * mc
            # 【v4.3 策略化】强度差异阈值由 VerifyStrategy 提供
            delta = self.verify_strategy.get_intensity_diff_thresh() * mc * uc
            if lhs <= rhs + delta:
                return False

        # 空间校验（动态网格）
        rows = [i // self.grid_size for i in marked]
        cols = [i % self.grid_size for i in marked]
        # 【v4.3 策略化】max_span 由 VerifyStrategy 提供
        max_span = self.verify_strategy.get_max_span(self.grid_size)
        span = (max(rows) - min(rows))**2 + (max(cols) - min(cols))**2
        if span >= max_span:
            return False

        return True

    # ---- 分级模板合并 [v4.2 OR/AND + 动态维度] ----
    def _add_template(self, label, layers, layer_count, intensity=None):
        OR_T = CONFIG["OR_THRESH"]
        AND_T = CONFIG["AND_THRESH"]
        # 使用所有层的 OR 组合作为模板特征，区分度远高于单层
        signature = combine_layers(layers[:layer_count]) if layer_count > 0 else 0

        merged = False
        for i, (tlb, tm, sc, hc) in enumerate(self.templates):
            if tlb != label:
                continue
            # 传入 self.D 确保不同维度下匹配度可比
            sim = match_bits(tm, signature, d=self.D)
            if sim >= OR_T:
                # OR合并，扩大覆盖范围（并集），防全满
                new_mask = tm | signature
                if new_mask != ((1 << self.D) - 1):
                    self.templates[i] = (tlb, new_mask, sc + 1, min(SGNConstants.MAX_HIT_COUNTER, hc + SGNConstants.HIT_COUNTER_INC_OR))
                else:
                    # 全满时退化为交集，避免无差别匹配
                    self.templates[i] = (tlb, tm & signature, sc + 1, min(SGNConstants.MAX_HIT_COUNTER, hc + SGNConstants.HIT_COUNTER_INC_AND))
                merged = True
                break
            elif sim >= AND_T:
                # 【v4.3 策略化】AND收敛最小位数由 LayerStrategy 提供
                merged_mask = tm & signature
                bits = bin(merged_mask).count("1")
                min_bits = self.layer_strategy.get_min_bits(self.D)
                if merged_mask and bits >= min_bits:
                    self.templates[i] = (tlb, merged_mask, sc + 1, min(SGNConstants.MAX_HIT_COUNTER, hc + SGNConstants.HIT_COUNTER_INC_AND))
                    merged = True
                    break
                # 位太少时不合并，继续检查其他模板或添加新模板

        if not merged:
            # 未合并，添加新模板
            if len(self.templates) < CONFIG["MAX_TEMPLATES"]:
                idx = len(self.templates)
                self.templates.append((label, signature, 1, CONFIG["HIT_COUNTER_INIT"]))
                # [v5.1] Maintain index
                self._template_index.setdefault(label, []).append(idx)

                # 【v4.4 双图叠加】更新高层/低层模板
                if self.use_gate_matching and layer_count > 0:
                    from sgn_layers import extract_edge_map
                    # 高层模板：使用原始强度图提取边缘（如果有）
                    if intensity is not None:
                        self.high_templates[label] = extract_edge_map(intensity, self.D)
                    else:
                        # Fallback：从掩码反推强度图
                        intensity_from_mask = [int((signature >> i) & 1) * 255 for i in range(self.D)]
                        self.high_templates[label] = extract_edge_map(intensity_from_mask, self.D)
                    # 低层模板：像素特征
                    self.low_templates[label] = signature

                # 【v4.4 双图叠加】更新原子字典
                if self.use_gate_matching:
                    self._update_atoms(label, signature)
            else:
                # 模板库已满，静默丢弃（每100步提示一次避免刷屏）
                if len(self.history) % 100 == 0:
                    print(f"\n  {C.YEL}⚠ 模板库已满 ({CONFIG['MAX_TEMPLATES']}/{CONFIG['MAX_TEMPLATES']})，新模板被丢弃{C.RST}")
                    print(f"  {C.DIM}建议：调大 MAX_TEMPLATES 或检查模板合并阈值{C.RST}")

    # ---- 原子字典更新 [v4.4 双图叠加] ----
    def _update_atoms(self, label, mask):
        """学习成功时更新按类别原子字典"""
        from sgn_layers import extract_patch_vector
        V = extract_patch_vector(mask, self.D, self.patch_size)
        if label not in self.atom_dict:
            self.atom_dict[label] = {}
        d = self.atom_dict[label]
        for atom in V:
            d[atom] = d.get(atom, 0) + 1
        # 每类独立限制大小
        max_atoms = CONFIG.get("MAX_ATOMS", 200)
        max_per_class = max_atoms // max(1, len(self.atom_dict))
        if len(d) > max_per_class:
            self.atom_dict[label] = dict(sorted(d.items(), key=lambda x: -x[1])[:max_per_class])

    def _query_atom_score(self, mask):
        """查询原子字典，返回 (最佳类别, 匹配度)"""
        from sgn_layers import extract_patch_vector
        V = extract_patch_vector(mask, self.D, self.patch_size)
        if not V or not self.atom_dict:
            return "?", 0
        best_label = "?"
        best_score = 0
        for label, d in self.atom_dict.items():
            hit = sum(1 for atom in V if atom in d)
            score = (hit * 100) // len(V)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    # ---- hit_counter全局衰减 ----
    def _decay_hit_counters(self):
        for i in range(len(self.templates)):
            tlb, tm, sc, hc = self.templates[i]
            if hc > 1:
                hc >>= 1
            self.templates[i] = (tlb, tm, sc, hc)

    # ---- 单步训练 ----
    def train(self, intensity, label):
        # 【窗口大小识别】首样本自动校准维度
        detected_d = len(intensity)
        if detected_d != self.D:
            self._rebuild_for_dimension(detected_d)

        self.label_set.add(label)

        # v5.1 多层神经元模式
        if self.multi_layer_enabled:
            return self._train_multi_layer(intensity, label)

        # 【v5.0 图模式】分发到图模式或模板模式
        if self.graph_mode:
            return self._train_graph_mode(intensity, label)

        # 【v4.4 双图叠加】保存到历史缓冲区
        if self.use_gate_matching:
            self.hist_buffer.append((intensity[:], label))
            max_hist = CONFIG.get("HIST_BUFFER_SIZE", 10)
            if len(self.hist_buffer) > max_hist:
                self.hist_buffer.pop(0)

        layers, layer_count = extract_layers(intensity, d=self.D, strategy=self.layer_strategy)

        # 1. 竞争
        matches = []
        for i, n in enumerate(self.N):
            if n["L"]:
                continue
            match = self._match(n, layers, layer_count, intensity)
            speed = self._response_speed(n, match)
            matches.append((speed, match, i))

        if not matches:
            # 所有神经元被锁定，无法竞争
            return {
                "label": label,
                "layer_count": layer_count,
                "V": False,
                "match": 0,
                "base": 0,
                "winners": [],
                "active": 0,
                "locked": len(self.N),
                "templates": len(self.templates),
                "_m0": layers[0] if layer_count > 0 else 0,
                "_D": self.D,
                "_grid": self.grid_size,
            }
        matches.sort(reverse=True)
        top_k = CONFIG["TOP_K"]
        winners = [i for _, _, i in matches[:top_k]]

        # 2. 校验
        verified = self._verify(intensity, layers, layer_count)

        # 3. 赫布学习
        self._hebbian_learn(winners, verified)

        # 4. 鼓励衰减（全局 O(N)，N=256 时约 256 次整数递减/步，可忽略）
        # 【步内衰减说明】同一 train() 中先设置 enc_r=ENCOURAGE_CNT，
        # 末尾再衰减。所以用户看到"设置 3 但显示 2"是正确行为——
        # 实际持续 ENCOURAGE_CNT 个完整步。
        for n in self.N:
            if n["enc_r"] > 0:
                n["enc_r"] -= 1
                if n["enc_r"] == 0:
                    n["enc_b"] = DiscreteCoordinate(0, n["enc_b"].level)  # 同层级零值清零

        # 5. 模板处理
        if verified:
            self._add_template(label, layers, layer_count, intensity)

        # 6. hit_counter衰减
        self._decay_hit_counters()

        # 统计
        avg_base = sum(n["base"].index for n in self.N) // CONFIG["MAX_NEURONS"]
        active = sum(1 for n in self.N if not n["L"])
        locked = sum(1 for n in self.N if n["L"])
        winner_match = matches[0][1] if matches else 0

        # 神经元全锁警告
        if active == 0 and len(self.N) > 0:
            print(f"\n  {C.RED}⚠ 所有神经元已被锁定！网络失去学习能力。{C.RST}")
            print(f"  {C.DIM}建议：按 [r] 重置网络或调大 MAX_LOCKOUT{C.RST}")

        return {
            "label": label,
            "layer_count": layer_count,
            "V": verified,
            "match": winner_match,
            "base": avg_base,
            "winners": winners,
            "active": active,
            "locked": locked,
            "templates": len(self.templates),
            "_m0": layers[0] if layer_count > 0 else 0,
            "_D": self.D,          # 调试：当前窗口维度
            "_grid": self.grid_size, # 调试：网格边长
            # 【v4.4 双图叠加】门控匹配信息
            "gate_enabled": self.use_gate_matching,
            "hist_buffer_size": len(self.hist_buffer) if self.use_gate_matching else 0,
            "atom_dict_size": sum(len(d) for d in self.atom_dict.values()) if self.use_gate_matching else 0,
        }

    # ============================================================
    # v5.0 图模式方法
    # ============================================================

    def _rebuild_for_dimension(self, new_d: int) -> None:
        """维度变更时重建"""
        old_d = self.D
        self.D = new_d
        self.grid_size = int(self.D ** 0.5)
        if self.grid_size * self.grid_size != self.D:
            raise ValueError(f"输入维度 D={self.D} 不是完全平方数")

        from sgn_config import ConfigRegistry
        ConfigRegistry._values["D"] = self.D

        self._auto_scale_resources()

        # v5.1 多层模式下重建分层池
        if self.multi_layer_enabled:
            l0_count = CONFIG.get("NEURON_LAYER_0_COUNT", 128)
            l1_count = CONFIG.get("NEURON_LAYER_1_COUNT", 64)
            self.neuron_layers = {
                0: [create_neuron(i, self.D) for i in range(l0_count)],
                1: [create_neuron(i + l0_count, self.D) for i in range(l1_count)],
            }
            self.N = self.neuron_layers[0]
            for n in self.neuron_layers[1]:
                n["layer"] = 1
        else:
            self.N = [create_neuron(i, self.D) for i in range(CONFIG["MAX_NEURONS"])]
            self.neuron_layers = {0: self.N}

        self._sync_strategies()

        if self.graph_mode or self.multi_layer_enabled:
            self.graphs = {}
            print(f"  [窗口识别] D={self.D} ({self.grid_size}×{self.grid_size})，网络已重建")
        else:
            self.templates = []
            self._template_index = {}  # [v5.1] Clear index
            self.history = []
            self.label_set = set()
            print(f"  [窗口识别] D={self.D} ({self.grid_size}×{self.grid_size})，网络已重建")

    def _train_graph_mode(self, intensity: List[int], label: str) -> Dict:
        """
        图模式训练 - 基于神经元判断 + 反馈循环 + 层级下压
        """
        d = self.D
        grid_size = self.grid_size
        self._step_counter += 1

        # 1. 获取变体视图（数据增强）
        views = self._get_parallel_views(label, intensity)

        # 2. 对每个视图执行神经元竞争 -> 投影为图
        projections = []
        for view in views:
            layers, lc = extract_layers(view, d=d, strategy=self.layer_strategy)
            if lc == 0:
                continue

            # 神经元竞争
            matches = []
            for i, n in enumerate(self.N):
                if n["L"]:
                    continue
                match = self._match(n, layers, lc, view)
                speed = self._response_speed(n, match)
                matches.append((speed, match, i))

            matches.sort(reverse=True)
            top_k = CONFIG["TOP_K"]
            winners = []
            for _, match_score, idx in matches[:top_k]:
                n = self.N[idx].copy()
                n["match"] = match_score
                n["nid"] = idx
                winners.append(n)

            if not winners:
                continue

            # 投影为图
            proj = project_neurons_to_graph(
                winners, view, d, grid_size, self._step_counter, label
            )
            projections.append(proj)

        if not projections:
            return self._make_graph_info(label, False, 0, [])

        # 3. 合并投影 -> 模板图
        if label not in self.graphs:
            self.graphs[label] = DynamicGraph()

        merged = merge_winner_projections(projections, label, self._step_counter)

        # 4. 赫布融合到现有图
        for nid, node in merged.nodes.items():
            self.graphs[label].hebbian_merge(
                node, self._step_counter,
                lr_alpha_pct=int(CONFIG.get("GRAPH_LEARNING_RATE", 0.3) * 100)
            )

        # 5. 【核心】反馈迭代循环
        final_score, final_label = self._feedback_loop(
            intensity, label, self.graphs[label], d, grid_size
        )

        # 6. 层级下压遗忘（每50步）
        if self._step_counter % 50 == 0:
            self._demote_and_cover(label)

        verified = (final_score >= 80)
        return self._make_graph_info(label, verified, final_score, [merged])

    def _get_parallel_views(self, label: str, base_intensity: List[int]) -> List[List[int]]:
        """生成同一标签的多张变体图"""
        n_views = CONFIG.get("PARALLEL_VIEWS", 3)
        if n_views <= 1:
            return [base_intensity]

        from sgn_input import DefaultCompositeNoise
        noise = DefaultCompositeNoise(CONFIG.get("FLIP_PROB", 0.15))

        views = [base_intensity]
        for _ in range(n_views - 1):
            views.append(noise.apply(base_intensity.copy()))
        return views

    def _feedback_loop(
        self,
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
            reconstructed = self._reconstruct_intensity(current_graph, d, grid_size)
            # 计算重建误差作为匹配度（误差越小匹配度越高）
            match_score = self._compute_reconstruction_score(intensity, reconstructed, d)

            if match_score >= threshold:
                return match_score, label
            if match_score > best_score:
                best_score = match_score
                best_label = label

            # 生成误差图
            error_intensity = self._generate_error_map(intensity, current_graph, d, grid_size)

            # 用误差图反馈重建
            current_graph = rebuild_with_feedback(
                current_graph, error_intensity, d, grid_size,
                self._step_counter, label
            )

        return best_score, best_label

    def _reconstruct_intensity(
        self,
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

    def _compute_reconstruction_score(
        self,
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

    def _generate_error_map(
        self,
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

    def _demote_and_cover(self, label: str) -> None:
        """
        层级下压 + L0覆盖遗忘
        """
        if label not in self.graphs:
            return

        graph = self.graphs[label]
        threshold = CONFIG.get("DEMOTION_THRESHOLD", 5)
        cover_age = CONFIG.get("LAYER_COVER_THRESHOLD", 1000)

        # 逐级下压（从高层到低层）
        total_demoted = graph.demote_all(threshold)

        # L0覆盖：删除长期未激活的L0节点
        to_cover = graph.get_cover_candidates(self._step_counter, cover_age)
        for nid in to_cover:
            node = graph.get_node(nid)
            if node and node.activation < 2:
                graph.remove_node(nid)

    def _make_graph_info(
        self,
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
            "templates": len(self.graphs),
            "_m0": 0,
            "_D": self.D,
            "_grid": self.grid_size,
            "graph_mode": True,
            "graph_nodes": total_nodes,
            "step": self._step_counter,
        }

    # ============================================================
    # v5.1 多层神经元训练
    # ============================================================

    def _train_multi_layer(self, intensity: List[int], label: str) -> Dict:
        """
        多层神经元训练 - Layer 0 → 图汇总 → Layer 1 → 分类

        数据流：
          输入强度图 → extract_layers → 掩码
            ↓
          Layer 0 神经元竞争（基础特征专家：边缘/局部模式）
            ↓
          project_neurons_to_graph → 图 L0
            ↓
          DynamicGraph.hebbian_merge → 图 L0→L1 组合
            ↓
          extract_graph_features → 图结构特征 + L0 激活 ID 列表
            ↓
          Layer 1 神经元竞争（图特征向量相似度匹配）
            ↓
          赫布学习（两层各自独立，label 作为参数传递）
        """
        from sgn_config import DiscreteCoordinate

        d = self.D
        grid_size = self.grid_size
        self._step_counter += 1

        # 1. 提取掩码层
        layers, layer_count = extract_layers(intensity, d=d, strategy=self.layer_strategy)
        if layer_count == 0:
            return self._make_multi_layer_info(label, False, 0, 0, 0)

        # 2. Layer 0 神经元竞争
        layer0_pool = self.neuron_layers[0]
        matches_l0 = []
        for i, n in enumerate(layer0_pool):
            if n["L"]:
                continue
            if CONFIG.get("ENABLE_SOFT_GATE", False):
                if n["gate"].index <= 0:
                    continue
            match = self._match(n, layers, layer_count, intensity)
            speed = self._response_speed(n, match)
            matches_l0.append((speed, match, i))

        if not matches_l0:
            return self._make_multi_layer_info(label, False, 0, 0, 0)

        matches_l0.sort(reverse=True)
        top_k_l0 = CONFIG["TOP_K"]
        # 统一为 Dict 格式（避免 tuple/dict 混用 Bug 7）
        winners_l0 = []
        for speed, match_score, idx in matches_l0[:top_k_l0]:
            winners_l0.append({
                "nid": idx,
                "match": match_score,
                "speed": speed,
            })

        # 3. Layer 0 获胜者投影为图
        proj_input = []
        for w in winners_l0:
            n = layer0_pool[w["nid"]]
            proj_input.append({
                "T": n["T"],
                "base": n["base"],
                "match": w["match"],
                "nid": w["nid"],
            })
        proj = project_neurons_to_graph(
            proj_input, intensity, d, grid_size, self._step_counter, label
        )

        if not proj.nodes:
            # 无图节点，仅 Layer 0 学习
            verified_l0 = self._verify(intensity, layers, layer_count)
            self._hebbian_multi_layer(winners_l0, verified_l0, layer=0, label=label)
            return self._make_multi_layer_info(label, verified_l0, 0, 0, 0)

        # 4. 图赫布融合到现有图
        if label not in self.graphs:
            self.graphs[label] = DynamicGraph()

        for nid, node in proj.nodes.items():
            self.graphs[label].hebbian_merge(
                node, self._step_counter,
                lr_alpha_pct=int(CONFIG.get("GRAPH_LEARNING_RATE", 0.3) * 100)
            )

        # 5. 提取图特征作为 Layer 1 输入
        graph_features, l0_active_list = self._extract_graph_features(
            self.graphs[label], d, grid_size, [w["nid"] for w in winners_l0]
        )

        # 6. Layer 1 神经元竞争
        layer1_pool = self.neuron_layers[1]
        matches_l1 = []
        for i, n in enumerate(layer1_pool):
            if n["L"]:
                continue
            if CONFIG.get("ENABLE_SOFT_GATE", False):
                if n["gate"].index <= 0:
                    continue
            match_l1 = self._match_layer1(n, graph_features, l0_active_list)
            speed_l1 = self._response_speed(n, match_l1)
            matches_l1.append({
                "nid": i,
                "match": match_l1,
                "speed": speed_l1,
            })

        if not matches_l1:
            verified_l0 = self._verify(intensity, layers, layer_count)
            self._hebbian_multi_layer(winners_l0, verified_l0, layer=0, label=label)
            return self._make_multi_layer_info(label, verified_l0, 0, len(proj.nodes), 0)

        matches_l1.sort(key=lambda x: x["speed"], reverse=True)
        top_k_l1 = CONFIG.get("TOP_K_L1", 4)
        winners_l1 = matches_l1[:top_k_l1]

        # 7. Layer 1 神经元首次学习：写入 T_features 和 T_l0_active
        for w in winners_l1:
            n = layer1_pool[w["nid"]]
            if not n["T_features_initialized"]:
                # 一次性学习：将当前图特征和 L0 激活写入模板
                n["T_features"] = list(graph_features)  # 深拷贝
                n["T_l0_active"] = list(l0_active_list)  # 深拷贝
                n["T_features_initialized"] = True
            # 记录溯源关系
            n["source_layer0"] = [w["nid"] for w in winners_l0]

        # 8. 最终分类验证（用像素级 _verify，不跨标签图比较 Bug 6）
        verified_final = self._verify(intensity, layers, layer_count)

        # 9. 赫布学习（两层各自独立，label 作为参数传递 Bug 3）
        self._hebbian_multi_layer(winners_l0, verified_final, layer=0, label=label)
        self._hebbian_multi_layer(winners_l1, verified_final, layer=1, label=label)

        # 10. Bug 12 修复：enc_r 鼓励衰减（全局 O(N)）
        for pool in self.neuron_layers.values():
            for n in pool:
                if n["enc_r"] > 0:
                    n["enc_r"] -= 1
                    if n["enc_r"] == 0:
                        n["enc_b"] = DiscreteCoordinate(0, n["enc_b"].level)

        # 11. Bug 13 修复：更新 history（供进度追踪使用）
        self.history.append({
            "label": label,
            "V": verified_final,
            "match": matches_l1[0]["match"],
            "layer0_active": len(winners_l0),
            "layer1_active": len(winners_l1),
        })

        return self._make_multi_layer_info(
            label, verified_final, matches_l1[0]["match"],
            len(proj.nodes), len(winners_l1)
        )

    def _extract_graph_features(
        self, graph: DynamicGraph, d: int, grid_size: int,
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
        from sgn_config import DiscreteCoordinate

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

        # L0 激活列表直接返回（不用位掩码 Bug 4）
        return graph_features, list(l0_active_nids)

    def _match_layer1(
        self, n: Dict,
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
        from sgn_config import DiscreteCoordinate

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
                    # 值差在 20% 以内算匹配
                    threshold = max(1, t_val.index // 5)
                    if abs(t_val.index - g_val.index) <= threshold:
                        matches += 1
                feat_sim = (matches * 100) // min_len

        # --- L0 激活重叠度（Jaccard 相似度）---
        if not t_l0_active or not l0_active_list:
            l0_sim = 50  # 无模板时给中间值
        else:
            set_t = set(t_l0_active)
            set_g = set(l0_active_list)
            intersection = len(set_t & set_g)
            union = len(set_t | set_g)
            l0_sim = (intersection * 100) // max(1, union)

        # 组合匹配度（图特征 60% + L0 激活 40%）
        return (feat_sim * 6 + l0_sim * 4) // 10

    def _hebbian_multi_layer(self, winners: List[Dict], verified: bool, layer: int = 0, label: str = None):
        """
        多层赫布学习（按层独立）

        Layer 0：基础学习率
        Layer 1：学习率 × LAYER1_LEARNING_RATE_SCALE

        Args:
            winners: 统一格式 List[Dict]，每个 dict 包含 {"nid": int, "match": int}
            verified: 校验是否通过
            layer: 神经元层 (0 或 1)
            label: 当前训练标签（作为参数传递，不用 self Bug 3）
        """
        from sgn_config import DiscreteCoordinate

        if not winners:
            return

        participants = len(winners)
        lr = CONFIG["LEARNING_RATE"]
        wr = CONFIG["WEAKEN_RATE"]
        sat = CONFIG["SPEED_SAT"]
        min_base = CONFIG["MIN_BASE"]
        enc_bonus = CONFIG["ENCOURAGE_BONUS"]
        enc_cnt = CONFIG["ENCOURAGE_CNT"]

        # 确保所有参数是 DiscreteCoordinate
        required_dc = [("LEARNING_RATE", lr), ("WEAKEN_RATE", wr), ("SPEED_SAT", sat),
                       ("MIN_BASE", min_base), ("ENCOURAGE_BONUS", enc_bonus)]
        for name, obj in required_dc:
            if not hasattr(obj, 'level'):
                raise TypeError(f"CONFIG['{name}'] 不是 DiscreteCoordinate")

        target_level = max(lr.level, wr.level, sat.level, min_base.level, enc_bonus.level)
        lr = lr.to_level(target_level)
        wr = wr.to_level(target_level)
        sat = sat.to_level(target_level)
        min_base = min_base.to_level(target_level)
        enc_bonus = enc_bonus.to_level(target_level)

        # Layer 1 学习率缩放
        lr_scale = CONFIG.get("LAYER1_LEARNING_RATE_SCALE", 0.5) if layer == 1 else 1.0
        delta = max(1, int(lr.index * lr_scale) // participants)

        pool = self.neuron_layers.get(layer, self.N)

        for w in winners:
            nid = w["nid"]
            if nid < 0 or nid >= len(pool):
                continue

            n = pool[nid]
            if n["L"]:
                continue

            base = n["base"]
            if base.level != target_level:
                base = base.to_level(target_level)

            # 专精神经元保护：标签不匹配时跳过削弱
            is_specialized = n.get("specialization") is not None
            spec_mismatch = is_specialized and label and n["specialization"] != label

            if verified:
                # 增强
                new_index = base.index + delta
                if new_index <= sat.index:
                    n["base"] = DiscreteCoordinate(new_index, target_level)
                else:
                    n["base"] = sat
                n["enc_r"] = enc_cnt
                n["enc_b"] = enc_bonus

                # 软门控更新
                if CONFIG.get("ENABLE_SOFT_GATE", False):
                    if n["specialization"] is None:
                        n["consecutive_verified"] += 1
                        if n["consecutive_verified"] >= CONFIG.get("GATE_SPECIALIZE_THRESHOLD", 10):
                            n["specialization"] = label
                    elif n["specialization"] == label:
                        new_gate = min(100, n["gate"].index + CONFIG.get("GATE_DECAY_RATE", 1))
                        n["gate"] = DiscreteCoordinate(new_gate, n["gate"].level)
                    else:
                        new_gate = max(0, n["gate"].index - CONFIG.get("GATE_DECAY_RATE", 1))
                        n["gate"] = DiscreteCoordinate(new_gate, n["gate"].level)
                        n["consecutive_verified"] = 0
            else:
                # 专精神经元保护
                if spec_mismatch:
                    continue
                # 削弱
                new_index = base.index - wr.index
                if new_index > min_base.index:
                    n["base"] = DiscreteCoordinate(new_index, target_level)
                else:
                    n["base"] = min_base
                n["lock"] += 1
                if n["lock"] >= CONFIG["MAX_LOCKOUT"]:
                    n["L"] = True

                if CONFIG.get("ENABLE_SOFT_GATE", False):
                    n["consecutive_verified"] = 0

    def _make_multi_layer_info(
        self, label: str, verified: bool, match: int,
        graph_nodes: int, layer1_active: int
    ) -> Dict:
        """构造多层训练 info 字典"""
        l0_active = sum(1 for n in self.neuron_layers.get(0, []) if not n["L"])
        l1_active = sum(1 for n in self.neuron_layers.get(1, []) if not n["L"])
        return {
            "label": label,
            "layer_count": 0,
            "V": verified,
            "match": match,
            "base": 0,
            "winners": [],
            "active": l0_active + l1_active,
            "locked": 0,
            "templates": len(self.graphs),
            "_m0": 0,
            "_D": self.D,
            "_grid": self.grid_size,
            "multi_layer": True,
            "layer0_active": l0_active,
            "layer1_active": l1_active,
            "graph_nodes": graph_nodes,
            "step": self._step_counter,
        }

    # ---- 状态摘要 ----
    def get_state(self):
        if self.graph_mode:
            total_nodes = sum(g.get_total_nodes() for g in self.graphs.values())
            return {
                "neurons": len(self.N),
                "active": total_nodes,
                "locked": 0,
                "encouraged": 0,
                "templates": len(self.graphs),
                "avg_base": 0,
                "graph_mode": True,
            }

        # 【v4.3】avg_base 改为整数索引平均值，不还原 float
        total_base_index = sum(n["base"].index for n in self.N) if self.N else 0
        avg_base_index = total_base_index // len(self.N) if self.N else 0
        return {
            "neurons": len(self.N),
            "active": sum(1 for n in self.N if not n["L"]),
            "locked": sum(1 for n in self.N if n["L"]),
            "encouraged": sum(1 for n in self.N if n["enc_r"] > 0),
            "templates": len(self.templates),
            "avg_base": avg_base_index,  # 整数索引，不还原 float
        }
