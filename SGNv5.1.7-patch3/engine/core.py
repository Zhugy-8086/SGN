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
"""SGN-Lite v5.1.5 核心引擎模块 - 整数化竞争/校验/学习/模板合并 + Level 调度器"""

import random
from typing import List, Dict, Optional, Tuple

from .config import CONFIG, D, DiscreteCoordinate, SGNConstants
from .utils import popcount, match_bits, extract_layers, combine_layers, C
from graph.graph import DynamicGraph, GraphNode
from graph.stack import project_neurons_to_graph, rebuild_with_feedback
from graph.merge import merge_winner_projections
from graph.graph_match import graph_similarity, classify_with_graph
from .level import (
    LevelScheduler, LevelContext, OperationType,
    StandardStrategy, AdaptiveStrategy, LayerAwareStrategy,
    get_global_scheduler, set_global_scheduler
)


def create_neuron(nid, d=16):
    """创建神经元 - 随机整数模板初始化（窗口大小自适应 + 离散坐标）

    Args:
        nid: 神经元编号
        d: 窗口像素总数（8×8=64, 16×16=256, 32×32=1024）
    """
    from .config import DiscreteCoordinate
    rd = random.Random(CONFIG["SEED"] + nid)
    mask_max = (1 << d) - 1  # 动态位宽：16→0xFFFF, 64→2^64-1
    base_dc = CONFIG["BASE_INIT"]  # DiscreteCoordinate 对象
    return {
        "nid": nid,                    # v5.1.5: 神经元 ID（用于调度器）
        "T": [rd.randint(0, mask_max) for _ in range(CONFIG["LAYER_MAX"])],
        "base": base_dc,               # 离散坐标，全程不还原
        "lock": 0,
        "enc_r": 0,
        "enc_b": DiscreteCoordinate(0, base_dc.level),  # 初始为 0，同层级
        "L": False,
        # v5.1.6 自适应随机静默（取代旧 gate 字段）
        "silenced": False,                           # 本步是否被静默（每步重置）
        "specialization": None,                      # 专精标签 (None=通用)
        "consecutive_verified": 0,                   # 累计验证通过计数（v5.1.7-patch: 不再要求连续）
        "label_freq": {},                            # v5.1.7-patch: 验证通过时的标签频次统计
        "layer": 0,                                  # 所属神经元层 (0=Layer0, 1=Layer1)
        # Layer 1 专用字段（Layer 0 神经元忽略这些）
        "T_features": [],                            # 图特征模板 (List[DiscreteCoordinate])
        "T_l0_active": [],                           # Layer 0 激活 ID 列表
        "T_features_initialized": False,             # 特征模板是否已学习
        # 补丁 P4（合并累积缓冲）：L0 同级比较合并的缓冲字段
        "T_pending": None,                           # 待写回的合并模板（累积用）
        "merge_count": 0,                            # 连续被合并的批次数
        # v5.1.7 L1 决策层：输出模式与自组织分类字段
        "output_pattern": [],                        # 输出判断模式向量
        "output_history": [],                        # 近 N 次输出模式历史
        "verified_count": 0,                         # 该输出模式被验证正确的次数
        "cluster_id": None,                          # 所属聚类 ID（涌现分类）
        # 补丁 P2（纠错机制）：簇置信度计数器
        "cluster_verified_count": 0,                 # 所属簇判断正确次数
        "cluster_total_count": 0,                    # 所属簇参与判断总次数
    }


class SGNCore:
    """SGN核心引擎 (v5.1.5 策略插件化 + 图模式层级记忆 + Level 调度器)"""

    def __init__(self, seed=None, layer_strategy=None, verify_strategy=None):
        if seed is not None:
            random.seed(seed)
            CONFIG["SEED"] = seed
        # D 初始为 64（8×8），首次 train() 时根据 len(intensity) 自动校准
        self.D = CONFIG.get("D", 64)
        self.grid_size = int(self.D ** 0.5)  # 8, 16, 32...

        # 【v4.3 策略注入】解耦硬编码阈值（v5.1.5：基线从 4×4 提升到 8×8）
        self.layer_strategy = layer_strategy
        self.verify_strategy = verify_strategy
        self._sync_strategies()  # 根据当前 D 自动匹配策略

        # 大窗口自动扩大神经元/模板库
        self._auto_scale_resources()

        # v5.1.5 Level 调度器集成
        self.level_scheduler = LevelScheduler()
        # 注册默认策略
        self.level_scheduler.register_strategy(StandardStrategy(2))
        self.level_scheduler.register_strategy(StandardStrategy(1))
        self.level_scheduler.register_strategy(AdaptiveStrategy(2))
        self.level_scheduler.register_strategy(LayerAwareStrategy())
        self.level_scheduler.set_default_strategy("standard(L2)")

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

            # v5.1.5: 为多层神经元绑定不同的 level 策略
            self._bind_level_strategies(l0_count, l1_count)
        else:
            self.N = [create_neuron(i, self.D) for i in range(CONFIG["MAX_NEURONS"])]
            self.neuron_layers = {0: self.N}
            # v5.1.5: 为单层神经元绑定标准策略
            self._bind_level_strategies_single()

        self.templates = []       # (label, mask, success_count, hit_counter)
        self._template_index: Dict[str, list] = {}  # [v5.1] label -> template index list
        self.history = []
        self.blackbox_log = []    # 黑箱验证记录
        self.label_set = set()    # 训练中出现过的所有标签

        # 【v5.0 图模式】新增成员
        self.graph_mode = CONFIG.get("ENABLE_GRAPH_MODE", False)
        self.graphs: Dict[str, DynamicGraph] = {}
        self._step_counter = 0

        # v5.1.7 补充: 参与率诊断计数器
        self._total_silence_calls = 0   # 累计调用 _apply_adaptive_silence 的次数
        self._total_active_count = 0    # 累计返回的 active_indices 长度

        # 如果图模式开启，冻结模板系统
        if self.graph_mode:
            self._template_backup = list(self.templates)
            self.templates = []

        # 匹配策略（可插拔）
        from .strategies import GlobalMatchStrategy
        self.match_strategy = GlobalMatchStrategy()  # 默认传统匹配

    def _bind_level_strategies(self, l0_count: int, l1_count: int) -> None:
        """v5.1.5: 为多层神经元绑定 level 策略"""
        # L0 神经元：标准精度
        for i in range(l0_count):
            self.level_scheduler.bind_neuron(i, "standard(L2)", initial_level=2)

        # L1 神经元：粗精度（图特征已经过抽象）
        for i in range(l1_count):
            self.level_scheduler.bind_neuron(i + l0_count, "standard(L1)", initial_level=1)

    def _bind_level_strategies_single(self) -> None:
        """v5.1.5: 为单层神经元绑定标准策略"""
        for i in range(len(self.N)):
            self.level_scheduler.bind_neuron(i, "standard(L2)", initial_level=2)

    def _sync_strategies(self):
        """根据当前 D 同步策略（维度变更时调用）"""
        from .strategies import StrategyRegistry, DefaultLayerStrategy, DefaultVerifyStrategy
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
        from .config import ConfigRegistry
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

    # ---- 响应速度计算 [v5.1.5 委托 Level 调度器] ----
    def _response_speed(self, n, match):
        """响应速度 = base.index + match * gamma.index（全程整数）

        【v5.1.5】委托 Level 调度器决定运算精度语境。
        调度器根据神经元 ID 和运算类型返回目标 level，
        上层建筑（竞争排序）完全不感知 level 细节。

        【v4.3-fix】删除错误的 GAMMA_DENOM=100 除法。
        """
        gamma = CONFIG["GAMMA"]  # DiscreteCoordinate
        base = n["base"]         # DiscreteCoordinate

        # v5.1.5: 委托调度器决定 level
        ctx = self.level_scheduler.get_context(
            OperationType.MUL,
            neuron_id=n.get("nid"),
            source="response_speed"
        )
        target_level = ctx.target_level

        # 统一到调度器决定的 level
        if base.level != target_level:
            base = base.to_level(target_level)
        if gamma.level != target_level:
            gamma = gamma.to_level(target_level)

        # 直接相加，无需额外的除法或缩放
        rsp = base.index + match * gamma.index

        if n["enc_r"] > 0:
            enc = n["enc_b"]
            if enc.level != target_level:
                enc = enc.to_level(target_level)
            rsp += enc.index

        return rsp

    # ---- 匹配计算 [v4.2 动态维度] ----
    def _match(self, n, layers, layer_count, intensity=None):
        """匹配计算（传统全局 XOR）

        v5.1.6: 旧门控匹配已删除，由自适应随机静默机制取代。
        silenced 神经元在竞争循环中被跳过，不进入此方法。

        Args:
            n: 神经元
            layers: 掩码列表
            layer_count: 层数
            intensity: 原始强度图（保留接口兼容，当前未使用）
        """
        # 传统匹配：全局 XOR
        t = sum(
            self.D - popcount(n["T"][ll] ^ layers[ll])
            for ll in range(min(layer_count, CONFIG["LAYER_MAX"]))
        )
        denom = min(layer_count, CONFIG["LAYER_MAX"]) * self.D
        return (t * 100) // denom if denom > 0 else 0

    def _compute_traditional_match(self, n, layers, layer_count):
        """Pre-compute traditional match score (avoid repetition in loop)"""
        from .layers import popcount as _popcount
        t = 0
        max_layer = min(layer_count, CONFIG["LAYER_MAX"])
        for ll in range(max_layer):
            t += self.D - _popcount(n["T"][ll] ^ layers[ll])
        denom = max_layer * self.D
        return (t * SGNConstants.PERCENT_BASE) // denom if denom > 0 else 0

    # ---- 赫布学习 [v5.1.5 委托 Level 调度器] ----
    def _hebbian_learn(self, winners, verified):
        """赫布学习 + 鼓励触发（全程整数坐标位移）

        【v5.1.5】委托 Level 调度器决定运算精度语境。
        调度器根据神经元 ID 和运算类型返回目标 level，
        上层建筑（赫布学习）完全不感知 level 细节。

        验证通过时：
          1. 永久增强 base.index（赫布学习，慢变量）
          2. 触发鼓励 enc_r = ENCOURAGE_CNT（短期脉冲，快变量）
          3. 同步 enc_b（锁定到神经元，离散坐标）

        验证失败时：
          1. 削弱 base.index
          2. 增加 lock 计数，可能触发锁定
          3. 不触发鼓励
        """
        from .config import DiscreteCoordinate
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

        # 学习步长 = max(1, lr.index // participants)
        delta = max(1, lr.index // participants)

        for nid in winners:
            n = self.N[nid]
            if n["L"]:
                continue

            # v5.1.5: 委托调度器决定该神经元的 level
            op = OperationType.ADD if verified else OperationType.SUB
            ctx = self.level_scheduler.get_context(op, neuron_id=nid, source="hebbian")
            target_level = ctx.target_level

            # 确保神经元 base 在同一层级
            base = n["base"]
            if base.level != target_level:
                base = base.to_level(target_level)

            # 将参数统一到调度器决定的 level
            lr_t = lr.to_level(target_level)
            wr_t = wr.to_level(target_level)
            sat_t = sat.to_level(target_level)
            min_t = min_base.to_level(target_level)
            enc_t = enc_bonus.to_level(target_level)

            if verified:
                # 1. 赫布学习（永久）：base.index += delta
                new_index = base.index + delta
                if new_index <= sat_t.index:
                    n["base"] = DiscreteCoordinate(new_index, target_level)
                else:
                    n["base"] = sat_t  # 饱和
                # 2. 鼓励触发（短期脉冲）
                n["enc_r"] = enc_cnt
                n["enc_b"] = enc_t

                # v5.1.5: 更新调度器统计
                self.level_scheduler.update_stats(nid, match=0, verified=True)

                # v5.1.7-patch3: L0 不再设置 specialization（回归原始设计：
                # L0 是纯特征检测器，模板不与任何标签绑定）
                # consecutive_verified 仍然累计（供静默机制参考），但不触发专精
            else:
                # 削弱：base.index -= wr.index
                new_index = base.index - wr_t.index
                if new_index > min_t.index:
                    n["base"] = DiscreteCoordinate(new_index, target_level)
                else:
                    n["base"] = min_t  # 不低于最小值
                n["lock"] += 1
                if n["lock"] >= CONFIG["MAX_LOCKOUT"]:
                    n["L"] = True

                # v5.1.5: 更新调度器统计
                self.level_scheduler.update_stats(nid, match=0, verified=False)

                if CONFIG.get("ENABLE_ADAPTIVE_SILENCE", True):
                    n["consecutive_verified"] = 0

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
            else:
                # 模板库已满，静默丢弃（每100步提示一次避免刷屏）
                if len(self.history) % 100 == 0:
                    print(f"\n  {C.YEL}⚠ 模板库已满 ({CONFIG['MAX_TEMPLATES']}/{CONFIG['MAX_TEMPLATES']})，新模板被丢弃{C.RST}")
                    print(f"  {C.DIM}建议：调大 MAX_TEMPLATES 或检查模板合并阈值{C.RST}")

    # ---- hit_counter全局衰减 ----
    def _decay_hit_counters(self):
        for i in range(len(self.templates)):
            tlb, tm, sc, hc = self.templates[i]
            if hc > 1:
                hc >>= 1
            self.templates[i] = (tlb, tm, sc, hc)

    # ---- v5.1.6 自适应随机静默 ----
    def _apply_adaptive_silence(self, pool: List[Dict]) -> List[int]:
        """v5.1.6 自适应随机静默

        每步训练前调用，从 pool 中随机选中 30%~80% 的神经元标记为 silenced。
        被静默的神经元本步不参与竞争、不输出值、不参与判断。

        补丁 P1（静默豁免）：专精神经元静默概率降低，
        避免 specialization 永远无法收敛。

        Returns:
            active_indices: 未被静默的神经元索引列表
        """
        if not CONFIG.get("ENABLE_ADAPTIVE_SILENCE", True):
            active = [i for i, n in enumerate(pool) if not n["L"]]
            self._total_silence_calls += 1
            self._total_active_count += len(active)
            return active

        # 概率触发：不是每步都静默
        if random.random() > CONFIG.get("SILENCE_TRIGGER_PROB", 0.5):
            for n in pool:
                n["silenced"] = False
            active = [i for i, n in enumerate(pool) if not n["L"]]
            self._total_silence_calls += 1
            self._total_active_count += len(active)
            return active

        # 计算本步静默比例（在 min~max 区间内随机）
        min_r = CONFIG.get("SILENCE_MIN_RATIO", 0.30)
        max_r = CONFIG.get("SILENCE_MAX_RATIO", 0.80)
        ratio = min_r + random.random() * (max_r - min_r)

        # 候选池：未锁定的神经元
        candidates = [i for i, n in enumerate(pool) if not n["L"]]
        if not candidates:
            self._total_silence_calls += 1
            return []

        # 补丁 P1：专精神经元静默权重降低
        specialized_weight = CONFIG.get("SILENCE_SPECIALIZED_WEIGHT", 0.3)
        weighted_candidates = []
        for i in candidates:
            n = pool[i]
            if n.get("specialization") is not None:
                if random.random() < specialized_weight:
                    weighted_candidates.append(i)
            else:
                weighted_candidates.append(i)

        # 随机选中静默集
        silence_count = int(len(candidates) * ratio)
        silenced_set = set(random.sample(
            weighted_candidates,
            min(silence_count, len(weighted_candidates))
        )) if weighted_candidates else set()

        active = []
        for i in candidates:
            pool[i]["silenced"] = i in silenced_set
            if i not in silenced_set:
                active.append(i)

        # v5.1.7 补充: 参与率统计
        self._total_silence_calls += 1
        self._total_active_count += len(active)

        return active

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

        layers, layer_count = extract_layers(intensity, d=self.D, strategy=self.layer_strategy)

        # 1. 竞争（v5.1.6：应用自适应静默）
        active_indices = self._apply_adaptive_silence(self.N)
        matches = []
        for i in active_indices:
            n = self.N[i]
            # silenced 神经元不参与判断、不输出值
            if n["L"] or n.get("silenced", False):
                continue
            match = self._match(n, layers, layer_count, intensity)
            speed = self._response_speed(n, match)
            matches.append((speed, match, i))
            # v5.1.5: 更新调度器匹配统计
            self.level_scheduler.update_stats(i, match=match, verified=False)

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
                "_D": self.D,
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
            "_D": self.D,          # 调试：当前窗口维度
            # v5.1.5: Level 调度器信息
            "level_scheduler_active": True,
            "neuron_levels": [self.level_scheduler.get_neuron_level(i) for i in winners[:3]],
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

        from .config import ConfigRegistry
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
            # Bug #4 修复：维度重建后必须重置 history/label_set/step_counter
            # 否则切换网格后旧 history 残留 → step 从非 0 起跳 → 训练步数不足
            self.history = []
            self.label_set = set()
            self._step_counter = 0
            print(f"  [窗口识别] D={self.D} ({self.grid_size}×{self.grid_size})，网络已重建")
        else:
            self.templates = []
            self._template_index = {}  # [v5.1] Clear index
            self.history = []
            self.label_set = set()
            self._step_counter = 0  # Bug #3 修复: 单层模式维度重建时也需重置计数器
            print(f"  [窗口识别] D={self.D} ({self.grid_size}×{self.grid_size})，网络已重建")

    def _train_graph_mode(self, intensity: List[int], label: str) -> Dict:
        """委托到 graph_train 模块"""
        from .graph_train import train_graph_mode
        return train_graph_mode(self, intensity, label)

    def _get_parallel_views(self, label: str, base_intensity: List[int]) -> List[List[int]]:
        """委托到 graph_train 模块"""
        from .graph_train import get_parallel_views
        return get_parallel_views(self, label, base_intensity)

    def _feedback_loop(
        self,
        intensity: List[int],
        label: str,
        graph: DynamicGraph,
        d: int,
        grid_size: int
    ) -> Tuple[int, str]:
        """委托到 graph_train 模块"""
        from .graph_train import feedback_loop
        return feedback_loop(self, intensity, label, graph, d, grid_size)

    def _reconstruct_intensity(
        self,
        graph: DynamicGraph,
        d: int,
        grid_size: int
    ) -> List[int]:
        """委托到 graph_train 模块"""
        from .graph_train import reconstruct_intensity
        return reconstruct_intensity(self, graph, d, grid_size)

    def _compute_reconstruction_score(
        self,
        original: List[int],
        reconstructed: List[int],
        d: int
    ) -> int:
        """委托到 graph_train 模块"""
        from .graph_train import compute_reconstruction_score
        return compute_reconstruction_score(self, original, reconstructed, d)

    def _generate_error_map(
        self,
        original: List[int],
        graph: DynamicGraph,
        d: int,
        grid_size: int
    ) -> List[int]:
        """委托到 graph_train 模块"""
        from .graph_train import generate_error_map
        return generate_error_map(self, original, graph, d, grid_size)

    def _demote_and_cover(self, label: str) -> None:
        """委托到 graph_train 模块"""
        from .graph_train import demote_and_cover
        return demote_and_cover(self, label)

    def _make_graph_info(
        self,
        label: str,
        verified: bool,
        score: int,
        graphs: List[DynamicGraph]
    ) -> Dict:
        """委托到 graph_train 模块"""
        from .graph_train import make_graph_info
        return make_graph_info(self, label, verified, score, graphs)

    # ============================================================
    # v5.1 多层神经元训练
    # ============================================================

    def _train_multi_layer(self, intensity: List[int], label: str) -> Dict:
        """委托到 multi_layer_train 模块"""
        from .multi_layer_train import train_multi_layer
        return train_multi_layer(self, intensity, label)

    def _l0_compete(self, intensity, layers, layer_count) -> List[Dict]:
        """委托到 multi_layer_train 模块"""
        from .multi_layer_train import l0_compete
        return l0_compete(self, intensity, layers, layer_count)

    def _multi_layer_l1_phase(
        self, winners_l0: List[Dict], layers, layer_count: int,
        intensity: List[int], label: str
    ) -> Dict:
        """委托到 multi_layer_train 模块"""
        from .multi_layer_train import multi_layer_l1_phase
        return multi_layer_l1_phase(self, winners_l0, layers, layer_count, intensity, label)

    # ============================================================
    # v5.1.6 分批次训练
    # ============================================================

    def build_sample_pool(self, samples=None) -> List:
        """委托到 multi_layer_train 模块"""
        from .multi_layer_train import build_sample_pool
        return build_sample_pool(samples)

    def train_batch(self, batch: List) -> List[Dict]:
        """委托到 multi_layer_train 模块"""
        from .multi_layer_train import train_batch
        return train_batch(self, batch)

    def _train_batch_multi_layer(self, batch: List) -> List[Dict]:
        """委托到 multi_layer_train 模块"""
        from .multi_layer_train import train_batch_multi_layer
        return train_batch_multi_layer(self, batch)

    def _train_batch_graph(self, batch: List) -> List[Dict]:
        """v5.1.6: 图模式批次训练（3.1: 逐样本，3.2 将接入同级比较）
        注：_train_graph_mode 内部已自增 _step_counter，此处不再重复
        """
        batch_info = []
        for sample in batch:
            intensity, label = sample
            info = self._train_graph_mode(intensity, label)
            batch_info.append(info)
        return batch_info

    def _train_batch_full(self, batch: List) -> List[Dict]:
        """v5.1.6: full 模式批次训练（3.1: 逐样本，3.2 将接入同级比较）"""
        batch_info = []
        for sample in batch:
            intensity, label = sample
            info = self.train(intensity, label)
            batch_info.append(info)
        return batch_info

    # ============================================================
    # v5.1.6 3.2: L0 同级比较与合并（含补丁 P4 累积缓冲）
    # ============================================================

    def _l0_peer_compare_and_merge(
        self,
        all_winners: List[List[Dict]],
        pool: List[Dict],
        batch: List
    ) -> List[List[Dict]]:
        """委托到 l0_peer_compare 模块"""
        from .l0_peer_compare import l0_peer_compare_and_merge
        return l0_peer_compare_and_merge(self, all_winners, pool, batch)

    def _template_similarity(self, n1: Dict, n2: Dict) -> float:
        """委托到 l0_peer_compare 模块"""
        from .l0_peer_compare import template_similarity
        return template_similarity(self, n1, n2)

    def _merge_templates_by_voting(
        self, templates: List[List[int]], weak_ratio: float
    ) -> List[int]:
        """委托到 l0_peer_compare 模块"""
        from .l0_peer_compare import merge_templates_by_voting
        return merge_templates_by_voting(self, templates, weak_ratio)

    def _binarize_template(self, T: List[int], thresh: int) -> List[int]:
        """委托到 l0_peer_compare 模块"""
        from .l0_peer_compare import binarize_template
        return binarize_template(self, T, thresh)

    # ============================================================
    # v5.1.7 3.3: L1 图模式桶化与链条化
    # ============================================================

    def _bucket_winners_by_label(
        self,
        winners: List[Dict],
        pool: List[Dict],
        sample: Tuple
    ) -> Dict[str, List[Dict]]:
        """委托到 l1_decision 模块"""
        from .l1_decision import bucket_winners_by_label
        return bucket_winners_by_label(self, winners, pool, sample)

    def _chain_winners(
        self,
        winners: List[Dict],
        pool: List[Dict]
    ) -> List[Dict]:
        """委托到 l1_decision 模块"""
        from .l1_decision import chain_winners
        return chain_winners(self, winners, pool)

    def _get_bucket(self, nid: int, pool: List[Dict]) -> str:
        """委托到 l1_decision 模块"""
        from .l1_decision import get_bucket
        return get_bucket(self, nid, pool)

    # ============================================================
    # v5.1.7 3.4: L1 决策层与自组织分类
    # ============================================================

    def _l1_generate_output_pattern(
        self, n: Dict, graph_features: List, l0_active_list: List
    ) -> List[float]:
        """委托到 l1_decision 模块"""
        from .l1_decision import l1_generate_output_pattern
        return l1_generate_output_pattern(self, n, graph_features, l0_active_list)

    def _pattern_similarity(
        self, p1: List[float], p2: List[float]
    ) -> float:
        """委托到 l1_decision 模块"""
        from .l1_decision import pattern_similarity
        return pattern_similarity(self, p1, p2)

    def _l1_peer_compare_and_cluster(
        self,
        l1_outputs: List[Tuple[int, List[float], bool, str]],
        pool: List[Dict]
    ) -> Dict[int, str]:
        """委托到 l1_decision 模块"""
        from .l1_decision import l1_peer_compare_and_cluster
        return l1_peer_compare_and_cluster(self, l1_outputs, pool)

    def _l1_feedback_to_graph(
        self,
        n: Dict,
        graph: "DynamicGraph",
        graph_features: List,
        verified: bool
    ):
        """委托到 l1_decision 模块"""
        from .l1_decision import l1_feedback_to_graph
        return l1_feedback_to_graph(self, n, graph, graph_features, verified)

    # ============================================================
    # v5.1.7 补丁 P2：纠错机制
    # ============================================================

    def _cluster_reorganization(self, pool: List[Dict]):
        """委托到 l1_decision 模块"""
        from .l1_decision import cluster_reorganization
        return cluster_reorganization(self, pool)

    def _negative_feedback_uncluster(
        self, n: Dict, verified: bool
    ):
        """委托到 l1_decision 模块"""
        from .l1_decision import negative_feedback_uncluster
        return negative_feedback_uncluster(self, n, verified)

    def _extract_graph_features(
        self, graph: DynamicGraph, d: int, grid_size: int,
        l0_active_nids: List[int]
    ) -> Tuple[List, List[int]]:
        """委托到 l1_decision 模块"""
        from .l1_decision import extract_graph_features
        return extract_graph_features(self, graph, d, grid_size, l0_active_nids)

    def _match_layer1(
        self, n: Dict,
        graph_features: List,
        l0_active_list: List[int]
    ) -> int:
        """委托到 l1_decision 模块"""
        from .l1_decision import match_layer1
        return match_layer1(self, n, graph_features, l0_active_list)

    def _compute_l1_predicted_label(
        self, n: Dict, l0_active_nids: List[int],
        graph_features_cache: Dict = None
    ):
        """v5.1.7-patch: 委托到 l1_decision 模块，计算 L1 神经元预测标签"""
        from .l1_decision import compute_l1_predicted_label
        return compute_l1_predicted_label(
            self, n, self.D, self.grid_size, l0_active_nids, graph_features_cache
        )

    def _hebbian_multi_layer(self, winners: List[Dict], verified: bool, layer: int = 0, label: str = None,
                             graph_features=None, l0_active_list=None):
        """
        多层赫布学习（按层独立）

        Layer 0：基础学习率
        Layer 1：学习率 × LAYER1_LEARNING_RATE_SCALE

        Args:
            winners: 统一格式 List[Dict]，每个 dict 包含 {"nid": int, "match": int}
            verified: 校验是否通过
            layer: 层级（0 或 1）
            label: 当前样本标签
            graph_features: v5.1.7-patch2 layer=1 时 L1 模板赫布学习所需的图特征向量
            l0_active_list: v5.1.7-patch2 layer=1 时 L1 激活集更新所需的 L0 激活 ID 列表
        """
        from .config import DiscreteCoordinate

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

            # v5.1.7-patch3: 专精神经元保护只对 L1 生效（L0 无 specialization）
            is_specialized = (layer == 1) and n.get("specialization") is not None
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

                # v5.1.7-fix: L1 特征模板赫布学习（向当前图特征移动）
                if layer == 1:
                    l1_feat_lr = CONFIG.get("L1_FEATURE_LR", 0.3)
                    # v5.1.7-patch2: 用显式参数，不再从 winners dict 隐式读取
                    gf = graph_features
                    la = l0_active_list
                    if gf and n.get("T_features"):
                        for i in range(min(len(n["T_features"]), len(gf))):
                            t = n["T_features"][i]
                            g = gf[i]
                            t_lvl = max(t.level if hasattr(t, 'level') else 0,
                                        g.level if hasattr(g, 'level') else 0)
                            t_idx = t.index if hasattr(t, 'index') else t
                            g_idx = g.index if hasattr(g, 'index') else g
                            n["T_features"][i] = DiscreteCoordinate(
                                int(t_idx * (1 - l1_feat_lr) + g_idx * l1_feat_lr),
                                t_lvl
                            )
                    if la and n.get("T_l0_active") is not None:
                        # v5.1.7-patch: T_l0_active 频次衰减（防止集合无限增长）
                        # 机制：用 dict {id: count} 存储频次，每次学习时：
                        #   1. 旧计数衰减（× (1 - decay)）
                        #   2. 当前激活的 ID 计数 +1
                        #   3. 低于阈值的 ID 被淘汰
                        # 该机制对任意层的"激活集"通用（L2 的 T_l1_active 同样适用）
                        decay = CONFIG.get("ACTIVE_SET_DECAY", 0.1)
                        keep_thresh = CONFIG.get("ACTIVE_SET_KEEP_THRESH", 0.3)
                        if not isinstance(n["T_l0_active"], dict):
                            # 兼容旧格式：list → dict（初始化频次为 1.0）
                            n["T_l0_active"] = {nid: 1.0 for nid in n["T_l0_active"]}
                        # 衰减旧计数
                        for nid in list(n["T_l0_active"].keys()):
                            n["T_l0_active"][nid] *= (1 - decay)
                            if n["T_l0_active"][nid] < keep_thresh:
                                del n["T_l0_active"][nid]
                        # 累加当前激活
                        for nid in la:
                            n["T_l0_active"][nid] = n["T_l0_active"].get(nid, 0.0) + 1.0

                # v5.1.7-patch3: specialization 只对 L1 生效（L0 是纯特征检测器）
                if CONFIG.get("ENABLE_ADAPTIVE_SILENCE", True):
                    if layer == 1 and n["specialization"] is None:
                        # v5.1.7-patch: 专精从"连续 N 次"改为"累计 N 次 + 标签频次投票"
                        # 原逻辑要求连续 10 次 verified=True，shuffle 数据下概率 ≈ 0.25^10 ≈ 0
                        # 新逻辑：累计 N 次验证通过，取出现次数最多的标签作为专精
                        # 该机制对任意层通用（L2/L3 的专精同样适用）
                        n["consecutive_verified"] += 1
                        n["label_freq"][label] = n["label_freq"].get(label, 0) + 1
                        # 达到累计阈值后，取频次最高的标签作为专精
                        threshold = CONFIG.get("SILENCE_SPECIALIZE_THRESHOLD", 10)
                        if n["consecutive_verified"] >= threshold:
                            n["specialization"] = max(
                                n["label_freq"], key=n["label_freq"].get
                            )
                    elif layer == 0:
                        # L0 仍累计 consecutive_verified（供静默机制参考），但不设置 specialization
                        n["consecutive_verified"] += 1
                    # 旧 gate 值增减逻辑已删除，专精状态由静默随机性自然维护
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

                if CONFIG.get("ENABLE_ADAPTIVE_SILENCE", True):
                    # v5.1.7-patch: 不再重置 consecutive_verified（改为累计计数）
                    # label_freq 保留所有历史标签频次，用于专精投票
                    pass

    def _make_multi_layer_info(
        self, label: str, verified: bool, match: int,
        graph_nodes: int, layer1_active: int
    ) -> Dict:
        """委托到 multi_layer_train 模块"""
        from .multi_layer_train import make_multi_layer_info
        return make_multi_layer_info(self, label, verified, match, graph_nodes, layer1_active)

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

        # Bug #5 修复：多层模式合并 L0+L1 统计，templates 改为 graphs 数
        if getattr(self, 'multi_layer_enabled', False):
            l0_pool = self.neuron_layers.get(0, [])
            l1_pool = self.neuron_layers.get(1, [])
            all_pool = l0_pool + l1_pool
            total_base_index = sum(n["base"].index for n in all_pool) if all_pool else 0
            avg_base_index = total_base_index // len(all_pool) if all_pool else 0
            # v5.1.7 补充: 平均参与率
            calls = getattr(self, '_total_silence_calls', 0)
            avg_participation_rate = (
                (self._total_active_count / calls / len(all_pool))
                if calls > 0 and all_pool else 0.0
            )
            return {
                "neurons": len(all_pool),
                "active": sum(1 for n in all_pool if not n["L"]),
                "locked": sum(1 for n in all_pool if n["L"]),
                "encouraged": sum(1 for n in all_pool if n["enc_r"] > 0),
                "templates": len(self.graphs),
                "avg_base": avg_base_index,
                "multi_layer": True,
                "layer0_count": len(l0_pool),
                "layer1_count": len(l1_pool),
                "avg_participation_rate": round(avg_participation_rate, 4),
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

    # ============================================================
    # v5.1.5 Level 调度器序列化/反序列化
    # ============================================================

    def serialize_level_scheduler(self) -> Dict:
        """序列化 Level 调度器状态"""
        return self.level_scheduler.serialize()

    def deserialize_level_scheduler(self, data: Dict) -> None:
        """反序列化 Level 调度器状态"""
        self.level_scheduler.deserialize(data)

    def get_level_info(self) -> Dict:
        """获取 Level 调度器摘要信息"""
        stats = self.level_scheduler.get_all_stats()
        level_counts = {}
        for nid, stat in stats.items():
            level = stat.current_level
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            "total_neurons": len(stats),
            "level_distribution": level_counts,
            "adapted_count": sum(1 for s in stats.values() if s.level_change_count > 0),
        }
