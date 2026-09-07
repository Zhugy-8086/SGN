#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""方案 D 单元测试：STE 梯度方差信号

验证 v1.4-rc20 方案 D 的三件事：
1. NeuronLevelStats 正确累积 grad_variance_history
2. AdaptiveStrategy 梯度方差触发条件正确（降级/升级）
3. grad_variance_threshold=0 时保持 v5.1.9 行为（向后兼容）
"""

import sys
import os
# ── 路径/导入调式日志 ──────────────────────────────────────────
# 背景：import sgn 时，如果 engine/ 在 sys.path 中且排在 build/ 之前，
# Python 会优先加载 engine/sgn/__init__.py（包）而非 sgn.cp*.pyd（扩展），
# 导致 sgn.__init__.py 在 _load_native_module() 中递归加载失败。
# 修复：仅添加 build/ 到 sys.path，移除 engine/ 和 engine/sgn/ 避免 shadow .pyd。
# 偶发条件：任何依赖 sgn C++ 扩展的测试如果同时添加了 engine/ 到 path 都可能触发。
_DEBUG = "SGN_DEBUG" in os.environ
if _DEBUG:
    print(f"[DEBUG] test_grad_variance_signal: 路径修复前 sys.path={sys.path}")

# 仅添加 build/ 目录（避免 engine/ 导致 import sgn 解析为 package 而非 .pyd）
_file_dir = os.path.dirname(os.path.abspath(__file__))
_build_dir = os.path.abspath(os.path.join(_file_dir, '..', '..', 'build'))
# 移除 engine/ 避免 shadow .pyd
_engine_dir = os.path.abspath(os.path.join(_file_dir, '..', '..', '..'))
for _p in [_engine_dir, os.path.join(_engine_dir, 'sgn')]:
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, _build_dir)

if _DEBUG:
    print(f"[DEBUG] test_grad_variance_signal: 修复后 sys.path(前3)={sys.path[:3]}")
    print(f"[DEBUG]   _build_dir={_build_dir}, _engine_dir={_engine_dir}")

from sgn.level import NeuronLevelStats, AdaptiveStrategy

# C++ NeuronLevelStats 尚未实现 grad_variance/grad_variance_history 等梯度方差字段
# C++ AdaptiveStrategy 尚未实现 grad_variance_threshold 参数
# 该测试需要等 C++ 端补全梯度方差信号功能后重新启用
_HAS_GRAD_VARIANCE = hasattr(NeuronLevelStats(0), 'grad_variance')
if _DEBUG and not _HAS_GRAD_VARIANCE:
    _nls_attrs = [x for x in dir(NeuronLevelStats(0)) if not x.startswith('_')]
    print(f"[DEBUG] test_grad_variance_signal: NeuronLevelStats 属性={_nls_attrs}")
    _as_attrs = [x for x in dir(AdaptiveStrategy()) if not x.startswith('_')]
    print(f"[DEBUG] test_grad_variance_signal: AdaptiveStrategy 属性={_as_attrs}")


def test_grad_variance_accumulation():
    """测试 1：梯度方差正确累积"""
    print("=== 测试 1：梯度方差累积 ===")
    stats = NeuronLevelStats(neuron_id=0)

    # 初始状态
    assert stats.grad_variance == 0.0, "初始 grad_variance 应为 0.0"
    assert len(stats.grad_variance_history) == 0, "初始历史应为空"

    # 累积 3 个样本
    stats.update(match=50, verified=True, grad_variance=0.5)
    stats.update(match=50, verified=True, grad_variance=1.5)
    stats.update(match=50, verified=True, grad_variance=2.0)

    assert len(stats.grad_variance_history) == 3
    assert stats.last_grad_variance == 2.0
    # 滑动平均 = (0.5 + 1.5 + 2.0) / 3 = 1.333...
    expected = (0.5 + 1.5 + 2.0) / 3
    assert abs(stats.grad_variance - expected) < 1e-6, f"期望 {expected}, 得到 {stats.grad_variance}"

    print(f"  ✓ grad_variance_history 长度: {len(stats.grad_variance_history)}")
    print(f"  ✓ last_grad_variance: {stats.last_grad_variance}")
    print(f"  ✓ grad_variance (滑动平均): {stats.grad_variance:.4f}")
    print()


def test_grad_variance_zero_no_update():
    """测试 2：grad_variance=0 时不更新历史（保持向后兼容）"""
    print("=== 测试 2：grad_variance=0 不更新 ===")
    stats = NeuronLevelStats(neuron_id=0)

    stats.update(match=50, verified=True)  # 不传 grad_variance
    stats.update(match=50, verified=True, grad_variance=0.0)

    assert len(stats.grad_variance_history) == 0, "grad_variance=0 不应更新历史"
    assert stats.grad_variance == 0.0
    assert stats.last_grad_variance == 0.0

    print(f"  ✓ grad_variance_history 长度: {len(stats.grad_variance_history)}")
    print(f"  ✓ grad_variance: {stats.grad_variance}")
    print()


def test_grad_variance_demotion():
    """测试 3：梯度方差大 → 降 level"""
    print("=== 测试 3：梯度方差触发降级 ===")
    strategy = AdaptiveStrategy(
        base_level=2,
        variance_threshold=100.0,
        history_window=10,
        grad_variance_threshold=1.0,  # 梯度方差 > 1.0 触发降级
    )
    stats = NeuronLevelStats(neuron_id=0, current_level=2, peak_level=2)

    # 填充 match_history 满足 history_window
    for _ in range(10):
        stats.update(match=50, verified=True, grad_variance=2.0)  # 梯度方差 2.0 > 1.0

    suggested = strategy.suggest_adaptation(stats)
    assert suggested == 1, f"梯度方差 2.0 > 1.0 应触发降级到 level 1, 得到 {suggested}"

    print(f"  ✓ 梯度方差 {stats.grad_variance} > 阈值 1.0 → 降级到 level {suggested}")
    print()


def test_grad_variance_promotion():
    """测试 4：梯度方差小 → 升 level"""
    print("=== 测试 4：梯度方差触发升级 ===")
    strategy = AdaptiveStrategy(
        base_level=0,
        variance_threshold=100.0,
        history_window=10,
        grad_variance_threshold=1.0,  # 梯度方差 < 0.25 触发升级
    )
    stats = NeuronLevelStats(neuron_id=0, current_level=0, peak_level=2)

    # 填充 match_history，梯度方差很小
    for _ in range(10):
        stats.update(match=50, verified=True, grad_variance=0.1)  # 0.1 < 0.25

    suggested = strategy.suggest_adaptation(stats)
    assert suggested == 1, f"梯度方差 0.1 < 0.25 应触发升级到 level 1, 得到 {suggested}"

    print(f"  ✓ 梯度方差 {stats.grad_variance} < 阈值/4 (0.25) → 升级到 level {suggested}")
    print()


def test_grad_variance_disabled_backward_compat():
    """测试 5：grad_variance_threshold=0 时保持 v5.1.9 行为"""
    print("=== 测试 5：禁用时向后兼容 ===")
    strategy = AdaptiveStrategy(
        base_level=2,
        variance_threshold=100.0,
        history_window=10,
        grad_variance_threshold=0.0,  # 禁用
    )
    stats = NeuronLevelStats(neuron_id=0, current_level=2, peak_level=2)

    # 即使有梯度方差历史，也不应触发
    for _ in range(10):
        stats.update(match=50, verified=True, grad_variance=100.0)  # 极大方差

    suggested = strategy.suggest_adaptation(stats)
    # match_variance=0（match 都是 50），不触发 match_variance 条件
    # grad_variance_threshold=0，不触发梯度方差条件
    assert suggested is None, f"禁用梯度方差时应不调整, 得到 {suggested}"

    print(f"  ✓ grad_variance_threshold=0 + match_variance=0 → 不调整 (suggested={suggested})")
    print()


def test_grad_variance_priority_over_match():
    """测试 6：梯度方差降级优先于 match_variance 升级"""
    print("=== 测试 6：梯度方差优先级 ===")
    strategy = AdaptiveStrategy(
        base_level=2,
        variance_threshold=100.0,
        history_window=10,
        grad_variance_threshold=1.0,
    )
    stats = NeuronLevelStats(neuron_id=0, current_level=1, peak_level=2)

    # match_variance 很小（会触发升级），但梯度方差很大（应优先降级）
    for _ in range(10):
        stats.update(match=50, verified=True, grad_variance=5.0)  # match 都相同=0 方差，grad=5.0

    suggested = strategy.suggest_adaptation(stats)
    # 梯度方差 5.0 > 1.0 应优先触发降级（而非 match_variance=0 触发升级）
    assert suggested == 0, f"梯度方差应优先触发降级到 level 0, 得到 {suggested}"

    print(f"  ✓ match_variance=0 (会升级) 但 grad_variance=5.0 > 1.0 → 优先降级到 level {suggested}")
    print()


def test_history_window_overflow():
    """测试 7：grad_variance_history 滑动窗口（最多 100）"""
    print("=== 测试 7：滑动窗口溢出 ===")
    stats = NeuronLevelStats(neuron_id=0)

    # 填充 120 个样本
    for i in range(120):
        stats.update(match=50, verified=True, grad_variance=float(i))

    assert len(stats.grad_variance_history) == 100, f"应限制在 100, 得到 {len(stats.grad_variance_history)}"
    # 最近的 100 个是 i=20..119，平均 = (20+119)/2 = 69.5
    expected_avg = sum(range(20, 120)) / 100
    assert abs(stats.grad_variance - expected_avg) < 1e-6

    print(f"  ✓ 历史 120 个 → 保留 100 个，平均 {stats.grad_variance:.4f} (期望 {expected_avg})")
    print()


def main():
    print("v1.4-rc20 方案 D 单元测试：STE 梯度方差信号\n")
    if not _HAS_GRAD_VARIANCE:
        print("[SKIP] C++ sgn.level.NeuronLevelStats 尚未实现 grad_variance 字段，测试待重新启用")
        return
    test_grad_variance_accumulation()
    test_grad_variance_zero_no_update()
    test_grad_variance_demotion()
    test_grad_variance_promotion()
    test_grad_variance_disabled_backward_compat()
    test_grad_variance_priority_over_match()
    test_history_window_overflow()
    print("=== 全部测试通过 ===")


if __name__ == "__main__":
    main()
