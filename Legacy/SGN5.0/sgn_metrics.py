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
"""SGN-Lite v5.0 评估指标模块 - Metric / MetricRegistry

阶段4重构：允许注册新评估指标，解耦 batch_test / confusion 逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


# ============================================================
# Metric - 评估指标抽象
# ============================================================

class Metric(ABC):
    """评估指标抽象基类"""

    name: str = ""

    @abstractmethod
    def compute(self, core, test_samples: List[Tuple], **kwargs) -> Dict[str, Any]:
        """计算指标

        Args:
            core: SGNCore 实例
            test_samples: [(intensity, label), ...] 测试样本

        Returns:
            字典形式的指标结果
        """
        pass



# ============================================================
# 共用模板匹配函数（v4.3 消除 6 处重复）
# ============================================================

def _classify_single(core, intensity, d=None) -> Tuple[str, int]:
    """对单个样本进行分类，返回 (预测标签, 最佳匹配度 0-100)

    自动适配模板模式和图模式。

    Args:
        core: SGNCore 实例
        intensity: 输入强度值列表
        d: 窗口像素总数（默认从 core.D 或 CONFIG['D'] 读取）

    Returns:
        (pred_label, best_score): 预测标签和匹配度 0-100
    """
    from sgn_config import CONFIG

    if d is None:
        d = getattr(core, 'D', CONFIG.get("D", 16))

    # 图模式：使用图匹配
    if getattr(core, 'graph_mode', False):
        from sgn_graph_match import classify_with_graph
        return classify_with_graph(core, intensity, d)

    # 模板模式：使用模板匹配
    from sgn_utils import extract_layers, combine_layers, match_bits

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


class AccuracyMetric(Metric):
    """准确率指标（原 do_batch_test 逻辑）"""

    name = "accuracy"

    def compute(self, core, test_samples: List[Tuple], **kwargs) -> Dict[str, Any]:
        from sgn_config import CONFIG

        # 动态收集标签，兼容非内置字符输入源
        if not test_samples:
            return {"error": "test_samples 为空，无法计算准确率", "accuracy": 0.0}
        all_labels = sorted(set(lb for _, lb in test_samples))

        correct = 0
        total = 0
        per_class = {lb: {"correct": 0, "total": 0} for lb in all_labels}

        for intensity, label in test_samples:
            pred_lb, best_s = _classify_single(core, intensity)
            match = (pred_lb == label and best_s >= 80)
            if match:
                correct += 1
                per_class[label]["correct"] += 1
            total += 1
            per_class[label]["total"] += 1

        overall = (correct / total * 100) if total else 0
        per_class_acc = {}
        for lb, stats in per_class.items():
            if stats["total"] > 0:
                per_class_acc[lb] = stats["correct"] / stats["total"] * 100
            else:
                per_class_acc[lb] = 0.0

        return {
            "accuracy": overall,
            "correct": correct,
            "total": total,
            "per_class": per_class_acc,
        }


class ConfusionMetric(Metric):
    """混淆矩阵指标（原 do_confusion 逻辑）"""

    name = "confusion"

    def compute(self, core, test_samples: List[Tuple], **kwargs) -> Dict[str, Any]:
        from sgn_config import CONFIG

        # 动态收集标签，兼容非内置字符输入源
        if not test_samples:
            return {"error": "test_samples 为空，无法生成混淆矩阵", "confusion_matrix": [], "labels": []}
        all_labels = sorted(set(lb for _, lb in test_samples))

        lb2i = {lb: i for i, lb in enumerate(all_labels)}
        size = len(all_labels)
        cm = [[0] * size for _ in range(size)]
        per_class_total = {lb: 0 for lb in all_labels}
        per_class_correct = {lb: 0 for lb in all_labels}

        for intensity, label in test_samples:
            pred, best_sim = _classify_single(core, intensity)
            if pred in lb2i:
                cm[lb2i[label]][lb2i[pred]] += 1
                per_class_total[label] += 1
                if pred == label:
                    per_class_correct[label] += 1

        # 计算每类准确率
        per_class_acc = {}
        for lb in all_labels:
            if per_class_total[lb] > 0:
                per_class_acc[lb] = per_class_correct[lb] / per_class_total[lb] * 100
            else:
                per_class_acc[lb] = 0.0

        return {
            "confusion_matrix": cm,
            "labels": all_labels,
            "per_class_accuracy": per_class_acc,
        }


class NoiseRobustnessMetric(Metric):
    """噪声鲁棒性指标（原 do_noise_test 逻辑）"""

    name = "noise_robustness"

    def __init__(self, flip_probs=None):
        from sgn_config import CONFIG
        noise_type = CONFIG.get("NOISE_TEST_TYPE", "composite")
        if noise_type == "composite":
            self.flip_probs = flip_probs or [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        else:
            # 其他噪声类型使用标准概率点
            self.flip_probs = flip_probs or [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        self.noise_type = noise_type

    def compute(self, core, test_samples: List[Tuple] = None, **kwargs) -> Dict[str, Any]:
        from sgn_config import CONFIG
        from sgn_input import DefaultCompositeNoise, GaussianNoise, SaltPepperNoise, BlockNoise

        source = kwargs.get('source')
        results = []

        # 【fix】优先使用传入的 test_samples 或 source，而非硬编码 gen_samples
        if test_samples is not None:
            # 对传入样本叠加噪声测试
            for flip in self.flip_probs:
                correct = 0
                total = 0
                for intensity, label in test_samples:
                    # 根据噪声类型施加噪声
                    if self.noise_type == "gaussian":
                        # 【v4.3-fix】矢量模式下降低高斯噪声概率，避免全亮画布误判
                        source = kwargs.get('source')
                        is_vector = (source is not None and hasattr(source, 'formula_type'))
                        noise_prob = 0.15 if is_vector else 1.0
                        nm = GaussianNoise(noise_prob=noise_prob, sigma=flip * 255 if flip <= 1.0 else flip)
                    elif self.noise_type == "salt_pepper":
                        nm = SaltPepperNoise(noise_prob=flip)
                    elif self.noise_type == "block":
                        gs = int(getattr(core, 'grid_size', CONFIG.get("D", 16) ** 0.5))
                        nm = BlockNoise(block_size=max(2, gs // 4), prob=flip, grid_size=gs)
                    else:
                        nm = DefaultCompositeNoise(noise_prob=flip)
                    noisy = nm.apply(intensity.copy())
                    pred_lb, best_s = _classify_single(core, noisy)
                    if pred_lb == label and best_s >= 80:
                        correct += 1
                    total += 1
                pct = (correct / total * 100) if total else 0
                results.append({"flip_prob": flip, "accuracy": pct})
        elif source is not None:
            # 从 source 生成样本
            for flip in self.flip_probs:
                correct = 0
                total = 0
                if self.noise_type == "gaussian":
                    # 【v4.3-fix】矢量模式下降低高斯噪声概率
                    is_vector = (source is not None and hasattr(source, 'formula_type'))
                    noise_prob = 0.15 if is_vector else 1.0
                    nm = GaussianNoise(noise_prob=noise_prob, sigma=flip * 255 if flip <= 1.0 else flip)
                elif self.noise_type == "salt_pepper":
                    nm = SaltPepperNoise(noise_prob=flip)
                elif self.noise_type == "block":
                    gs = int(getattr(core, 'grid_size', CONFIG.get("D", 16) ** 0.5))
                    nm = BlockNoise(block_size=max(2, gs // 4), prob=flip, grid_size=gs)
                else:
                    nm = DefaultCompositeNoise(noise_prob=flip)
                # 临时替换 source 的噪声模型（安全替换）
                old_noise = getattr(source, 'noise', None)
                if old_noise is not None:
                    source.noise = nm
                # 动态计算批次大小：从 source 或 core.templates 获取标签数
                if hasattr(source, 'patterns') and source.patterns:
                    n_labels = len(source.patterns)
                elif hasattr(source, '_samples') and source._samples:
                    n_labels = len(set(lb for _, lb in source._samples))
                else:
                    n_labels = max(1, len(set(t[0] for t in core.templates))) if core.templates else 1
                samples = source.generate_batch(40 * n_labels, split='all')
                if old_noise is not None:
                    source.noise = old_noise
                for intensity, label in samples:
                    pred_lb, best_s = _classify_single(core, intensity)
                    if pred_lb == label and best_s >= 80:
                        correct += 1
                    total += 1
                pct = (correct / total * 100) if total else 0
                results.append({"flip_prob": flip, "accuracy": pct})
        else:
            # 无 test_samples 且无 source，无法执行噪声测试
            return {
                "error": "噪声鲁棒性测试需要 test_samples 或 source 参数。请确保训练流程正确传递了测试样本或输入源。",
                "results": [],
                "max_accuracy": 0,
                "min_accuracy": 0,
            }

        return {
            "results": results,
            "max_accuracy": max(r["accuracy"] for r in results) if results else 0,
            "min_accuracy": min(r["accuracy"] for r in results) if results else 0,
        }


# ============================================================
# MetricRegistry - 指标注册表
# ============================================================

class MetricRegistry:
    """指标注册表 - 允许插件注册新指标"""

    _metrics: Dict[str, Metric] = {}

    @classmethod
    def register(cls, metric: Metric) -> None:
        if metric.name in cls._metrics:
            raise KeyError(f"指标 '{metric.name}' 已存在")
        cls._metrics[metric.name] = metric

    @classmethod
    def run(cls, name: str, core, test_samples: List[Tuple]) -> Dict[str, Any]:
        """运行指定指标"""
        metric = cls._metrics.get(name)
        if not metric:
            raise KeyError(f"未找到指标: {name}")
        return metric.compute(core, test_samples)

    @classmethod
    def run_all(cls, core, test_samples: List[Tuple], train_samples: List[Tuple] = None) -> Dict[str, Any]:
        """运行所有已注册的指标，返回聚合报告

        Args:
            train_samples: 可选的训练集样本，供 GeneralizationMetric 计算泛化差距
        """
        from sgn_config import CONFIG
        enabled = CONFIG.get("ENABLED_METRICS", list(cls._metrics.keys()))
        report = {}
        for name, metric in cls._metrics.items():
            if name not in enabled:
                continue
            try:
                if name == "generalization" and train_samples is not None:
                    report[name] = metric.compute(core, test_samples, train_samples=train_samples)
                else:
                    report[name] = metric.compute(core, test_samples)
            except Exception as e:
                report[name] = {"error": str(e)}
        return report

    @classmethod
    def list_metrics(cls) -> List[str]:
        return list(cls._metrics.keys())

    @classmethod
    def clear(cls) -> None:
        cls._metrics.clear()



class GeneralizationMetric(Metric):
    """泛化能力指标：计算训练集与留出测试集的准确率差异

    揭示 SGN 的"训练集记忆率"与"真实泛化能力"之间的 gap。
    """

    name = "generalization"

    def compute(self, core, test_samples: List[Tuple], **kwargs) -> Dict[str, Any]:
        """计算训练集与留出测试集的准确率差异

        Args:
            train_samples: 通过 kwargs 传入的训练集样本，用于计算 generalization_gap
        """
        train_samples = kwargs.get('train_samples', [])
        def _eval(samples):
            correct = 0
            total = 0
            for intensity, label in samples:
                pred_lb, best_s = _classify_single(core, intensity)
                if pred_lb == label and best_s >= 80:
                    correct += 1
                total += 1
            return (correct / total * 100) if total else 0

        train_acc = _eval(train_samples)
        test_acc = _eval(test_samples)
        gap = train_acc - test_acc

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "generalization_gap": gap,
            "status": "healthy" if gap < 15 else "overfit" if gap < 30 else "severe_overfit"
        }


# 自动注册默认指标
MetricRegistry.register(AccuracyMetric())
MetricRegistry.register(ConfusionMetric())
MetricRegistry.register(NoiseRobustnessMetric())
MetricRegistry.register(GeneralizationMetric())
