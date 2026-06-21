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

from sgn_config import CONFIG, D, DiscreteCoordinate
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

        self.N = [create_neuron(i, self.D) for i in range(CONFIG["MAX_NEURONS"])]
        self.templates = []       # (label, mask, success_count, hit_counter)
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
        from sgn_layers import extract_edge_map, popcount as _popcount

        if layer_count == 0:
            return 0

        # 提取输入的高层（边缘）和低层（像素）特征
        input_mask = layers[0]

        # 使用原始强度图提取边缘（如果有）
        if intensity is not None:
            high_in = extract_edge_map(intensity, self.D)
        else:
            # Fallback：从掩码反推强度图（信息丢失）
            intensity_from_mask = [int((input_mask >> i) & 1) * 255 for i in range(self.D)]
            high_in = extract_edge_map(intensity_from_mask, self.D)

        # 低层特征：直接用掩码
        low_in = input_mask

        # 与模板的高层/低层特征匹配
        # 如果模板有独立的高层/低层特征，使用门控匹配
        # 否则 fallback 到传统匹配
        if hasattr(self, 'high_templates') and hasattr(self, 'low_templates') and self.high_templates:
            # 遍历所有模板，找最佳门控匹配
            best_score = 0
            best_traditional_score = 0
            for tlb, tm, ts, thc in self.templates:
                # 获取模板的高层/低层特征
                high_tpl = self.high_templates.get(tlb, tm)
                low_tpl = self.low_templates.get(tlb, tm)

                # 门控匹配
                high_union = _popcount(high_in | high_tpl)
                high_match = (_popcount(high_in & high_tpl) * 100) // max(1, high_union)

                low_union = _popcount(low_in | low_tpl)
                low_match = (_popcount(low_in & low_tpl) * 100) // max(1, low_union)

                # 门控：高层 AND 低层
                high_thresh = CONFIG.get("GATE_HIGH_THRESH", 40)
                low_thresh = CONFIG.get("GATE_LOW_THRESH", 30)

                if high_match >= high_thresh and low_match >= low_thresh:
                    # 门控通过，计算综合得分
                    score = (high_match + low_match) // 2
                    if score > best_score:
                        best_score = score

                # 同时计算传统匹配得分（用于回退）
                trad_score = 0
                for ll in range(min(layer_count, CONFIG.get("LAYER_MAX", 4))):
                    trad_score += self.D - _popcount(n["T"][ll] ^ layers[ll])
                denom = min(layer_count, CONFIG.get("LAYER_MAX", 4)) * self.D
                trad_score = (trad_score * 100) // denom if denom > 0 else 0
                if trad_score > best_traditional_score:
                    best_traditional_score = trad_score

            # 如果门控匹配找到结果，返回门控得分
            if best_score > 0:
                return best_score
            # 否则回退到传统匹配
            return best_traditional_score

        # Fallback：传统匹配
        t = sum(
            self.D - _popcount(n["T"][ll] ^ layers[ll])
            for ll in range(min(layer_count, CONFIG["LAYER_MAX"]))
        )
        denom = min(layer_count, CONFIG["LAYER_MAX"]) * self.D
        return (t * 100) // denom if denom > 0 else 0

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
                    self.templates[i] = (tlb, new_mask, sc + 1, min(255, hc + 32))
                else:
                    # 全满时退化为交集，避免无差别匹配
                    self.templates[i] = (tlb, tm & signature, sc + 1, min(255, hc + 16))
                merged = True
                break
            elif sim >= AND_T:
                # 【v4.3 策略化】AND收敛最小位数由 LayerStrategy 提供
                merged_mask = tm & signature
                bits = bin(merged_mask).count("1")
                min_bits = self.layer_strategy.get_min_bits(self.D)
                if merged_mask and bits >= min_bits:
                    self.templates[i] = (tlb, merged_mask, sc + 1, min(255, hc + 16))
                    merged = True
                    break
                # 位太少时不合并，继续检查其他模板或添加新模板

        if not merged:
            # 未合并，添加新模板
            if len(self.templates) < CONFIG["MAX_TEMPLATES"]:
                self.templates.append((label, signature, 1, CONFIG["HIT_COUNTER_INIT"]))

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
        self.N = [create_neuron(i, self.D) for i in range(CONFIG["MAX_NEURONS"])]
        self._sync_strategies()

        if self.graph_mode:
            self.graphs = {}
            print(f"  [窗口识别] D={self.D} ({self.grid_size}×{self.grid_size})，图结构已重置")
        else:
            self.templates = []
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
                lr_alpha=CONFIG.get("GRAPH_LEARNING_RATE", 0.3)
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
            r = (node.position_norm[0] * max(1, grid_size - 1)) // 1000
            c = (node.position_norm[1] * max(1, grid_size - 1)) // 1000
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
        error_ratio = total_diff / max_possible if max_possible > 0 else 1.0
        return max(0, int(100 * (1.0 - error_ratio)))

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
            r = (node.position_norm[0] * max(1, grid_size - 1)) // 1000
            c = (node.position_norm[1] * max(1, grid_size - 1)) // 1000
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
