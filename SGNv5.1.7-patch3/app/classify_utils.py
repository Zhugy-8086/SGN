# -*- coding: utf-8 -*-
"""v5.1.7-patch2: 统一分类入口与模式感知阈值

消除 test.py._classify 和 metrics.py._classify_single 的重复，
并提供模式感知的置信度判定，替代散落各处的 `>= 80` 硬编码。

设计原则：不写死多层阈值为某个值，而是让分类函数返回 mode，
测试层根据 mode 从阈值表选择，未来 L2/L3 只需加一行。
"""
from typing import Tuple

# 默认置信度阈值（按模式区分）
CLASSIFY_THRESHOLDS = {
    "single": 80,   # 单层位匹配：80% 位相同
    "graph": 80,    # 图模式：保持原阈值（待独立验证）
    "multi": 30,    # 多层：L1 匹配分数天然偏低（45-65），30 为保守阈值
}


def classify_with_mode(core, intensity, d=None) -> Tuple[str, int, str]:
    """统一分类入口 - 自动适配模板/图/多层模式

    v5.1.7-patch2: 返回 (label, score, mode)，mode 用于阈值选择
    替代 test.py._classify 和 metrics.py._classify_single

    Args:
        core: SGNCore 实例
        intensity: 输入强度值列表
        d: 窗口像素总数（默认从 core.D 或 CONFIG['D'] 读取）

    Returns:
        (pred_label, best_score, mode): 预测标签、匹配度 0-100、模式名
    """
    if d is None:
        from engine.config import CONFIG
        d = getattr(core, 'D', CONFIG.get("D", 64))

    # 图模式：使用图匹配
    if getattr(core, 'graph_mode', False):
        from graph.graph_match import classify_with_graph
        pred, score = classify_with_graph(core, intensity, d)
        return pred, score, "graph"
    # 多层模式：使用 L1 决策层分类
    elif getattr(core, 'multi_layer_enabled', False):
        from engine.layers import classify_multi_layer
        pred, score = classify_multi_layer(core, intensity, d)
        return pred, score, "multi"
    # 单层模式：模板匹配
    else:
        from engine.layers import classify_sample
        pred, score = classify_sample(core, intensity, d)
        return pred, score, "single"


def classify_pass(core, intensity, label, d=None) -> Tuple[bool, str, int, str]:
    """统一判定：标签匹配 + 模式感知阈值

    v5.1.7-patch2: 替代原来的 `pred_lb == label and best_s >= 80`

    Args:
        core: SGNCore 实例
        intensity: 输入强度值列表
        label: 真实标签
        d: 窗口像素总数

    Returns:
        (passed, pred_label, best_score, mode):
            passed: 是否通过（标签匹配且分数达到阈值）
            pred_label: 预测标签
            best_score: 匹配度 0-100
            mode: 分类模式
    """
    pred_lb, best_s, mode = classify_with_mode(core, intensity, d)
    threshold = CLASSIFY_THRESHOLDS.get(mode, 80)
    passed = (pred_lb == label) and (best_s >= threshold)
    return passed, pred_lb, best_s, mode
