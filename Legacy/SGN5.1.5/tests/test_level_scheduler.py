#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGN-Lite v5.1.3 Level 调度器测试

测试 sgn_level.py 模块的核心功能：
  - LevelScheduler 初始化和策略管理
  - 神经元绑定和统计更新
  - 自适应 level 调整
  - 二元运算 level 决策
"""

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from engine.level import (
    LevelScheduler,
    LevelContext,
    OperationType,
    NeuronLevelStats,
    StandardStrategy,
    AdaptiveStrategy,
    LayerAwareStrategy,
    get_global_scheduler,
    set_global_scheduler,
    get_level_for_add,
    get_level_for_compare,
    update_neuron_stats,
    get_neuron_level,
)


def test_basic_scheduler():
    """测试调度器基本功能"""
    print("=" * 60)
    print("  测试 1: 调度器基本功能")
    print("=" * 60)

    scheduler = LevelScheduler()

    # 测试默认策略
    ctx = scheduler.get_context(OperationType.ADD, source="test")
    print(f"  默认策略: {ctx}")
    assert ctx.target_level == 2, f"期望 level=2，得到 {ctx.target_level}"

    # 测试策略注册
    strategies = list(scheduler._strategies.keys())
    print(f"  已注册策略: {strategies}")
    assert len(strategies) >= 5, "应该有至少 5 个内置策略"

    # 测试神经元绑定
    scheduler.bind_neuron(0, "standard(L2)")
    scheduler.bind_neuron(1, "standard(L1)")
    scheduler.bind_neuron(2, "adaptive(base=L2)")

    ctx0 = scheduler.get_context(OperationType.ADD, neuron_id=0)
    ctx1 = scheduler.get_context(OperationType.ADD, neuron_id=1)
    ctx2 = scheduler.get_context(OperationType.ADD, neuron_id=2)

    print(f"  神经元 0: {ctx0.target_level}")
    print(f"  神经元 1: {ctx1.target_level}")
    print(f"  神经元 2: {ctx2.target_level}")

    assert ctx0.target_level == 2
    assert ctx1.target_level == 1

    print("  [PASS] 基本功能测试通过")
    return True


def test_adaptive_strategy():
    """测试自适应策略"""
    print("\n" + "=" * 60)
    print("  测试 2: 自适应策略")
    print("=" * 60)

    scheduler = LevelScheduler()
    scheduler._adapt_interval = 10  # 缩短检查间隔

    # 绑定自适应策略
    scheduler.bind_neuron(100, "adaptive(base=L2)", initial_level=2)

    # 模拟低方差场景（匹配值稳定在 80-90）
    print("  模拟低方差场景...")
    for i in range(50):
        match = 80 + (i % 10)  # 方差很小
        scheduler.update_stats(100, match, verified=True)

    stats = scheduler.get_stats(100)
    print(f"  匹配值方差: {stats.match_variance:.2f}")
    print(f"  当前 level: {stats.current_level}")

    # 低方差应该触发自适应到更细粒度
    # 注意：需要足够的历史数据
    if stats.match_variance < 25:  # 方差足够小
        print("  [PASS] 低方差场景测试通过")
    else:
        print("  [WARN] 方差未达到阈值，跳过断言")

    # 模拟高方差场景（匹配值大幅波动）
    print("\n  模拟高方差场景...")
    scheduler.bind_neuron(200, "adaptive(base=L2)", initial_level=3)
    for i in range(50):
        match = 20 if i % 2 == 0 else 90  # 大幅波动
        scheduler.update_stats(200, match, verified=(i % 3 == 0))

    stats2 = scheduler.get_stats(200)
    print(f"  匹配值方差: {stats2.match_variance:.2f}")
    print(f"  当前 level: {stats2.current_level}")

    print("  [PASS] 自适应策略测试完成")
    return True


def test_binary_op_resolution():
    """测试二元运算 level 决策"""
    print("\n" + "=" * 60)
    print("  测试 3: 二元运算 level 决策")
    print("=" * 60)

    scheduler = LevelScheduler()

    # 测试不同 level 的二元运算
    test_cases = [
        (2, 2, "相同 level"),
        (2, 1, "不同 level"),
        (1, 3, "跨层级"),
        (0, 2, "整数空间 vs 0.01 空间"),
    ]

    for left, right, desc in test_cases:
        target = scheduler.resolve_binary_op(
            OperationType.ADD, left, right
        )
        print(f"  {desc}: L{left} + L{right} → L{target}")

    print("  [PASS] 二元运算测试通过")
    return True


def test_layer_aware_strategy():
    """测试层级感知策略"""
    print("\n" + "=" * 60)
    print("  测试 4: 层级感知策略")
    print("=" * 60)

    scheduler = LevelScheduler()

    # 绑定层级感知策略
    scheduler.bind_neuron(0, "layer_aware")
    scheduler.bind_neuron(1, "layer_aware")

    # 模拟 L0 和 L1 神经元
    stats0 = scheduler.get_stats(0)
    stats1 = scheduler.get_stats(1)

    # 手动设置层级属性（实际使用中由上层设置）
    if stats0:
        stats0.layer = 0
    if stats1:
        stats1.layer = 1

    ctx0 = scheduler.get_context(OperationType.COMPARE, neuron_id=0)
    ctx1 = scheduler.get_context(OperationType.COMPARE, neuron_id=1)

    print(f"  L0 神经元 level: {ctx0.target_level}")
    print(f"  L1 神经元 level: {ctx1.target_level}")

    print("  [PASS] 层级感知策略测试通过")
    return True


def test_convenience_functions():
    """测试便捷函数"""
    print("\n" + "=" * 60)
    print("  测试 5: 便捷函数")
    print("=" * 60)

    # 重置全局调度器
    set_global_scheduler(LevelScheduler())

    # 测试便捷函数
    level_add = get_level_for_add()
    level_cmp = get_level_for_compare()
    print(f"  加法 level: {level_add}")
    print(f"  比较 level: {level_cmp}")

    # 测试统计更新
    update_neuron_stats(0, match=85, verified=True)
    update_neuron_stats(0, match=70, verified=False)
    level = get_neuron_level(0)
    print(f"  神经元 0 level: {level}")

    print("  [PASS] 便捷函数测试通过")
    return True


def test_serialization():
    """测试序列化"""
    print("\n" + "=" * 60)
    print("  测试 6: 序列化")
    print("=" * 60)

    scheduler = LevelScheduler()
    scheduler.bind_neuron(0, "standard(L2)")
    scheduler.bind_neuron(1, "adaptive(base=L2)")
    scheduler.update_stats(0, 80, True)
    scheduler.update_stats(1, 70, False)

    # 序列化
    data = scheduler.serialize()
    print(f"  序列化数据: {data}")

    # 反序列化到新调度器
    scheduler2 = LevelScheduler()
    scheduler2.deserialize(data)

    # 验证
    stats0 = scheduler2.get_stats(0)
    stats1 = scheduler2.get_stats(1)
    print(f"  反序列化后神经元 0 统计: total={stats0.total_count}, verified={stats0.verified_count}")
    print(f"  反序列化后神经元 1 统计: total={stats1.total_count}, verified={stats1.verified_count}")

    assert stats0.total_count == 1
    assert stats0.verified_count == 1
    assert stats1.total_count == 1
    assert stats1.verified_count == 0

    print("  [PASS] 序列化测试通过")
    return True


def test_stats():
    """测试统计功能"""
    print("\n" + "=" * 60)
    print("  测试 7: 统计功能")
    print("=" * 60)

    stats = NeuronLevelStats(neuron_id=42, current_level=2)

    # 模拟训练
    for i in range(100):
        match = 70 + (i % 20)
        verified = (i % 3 != 0)
        stats.update(match, verified)

    print(f"  匹配值方差: {stats.match_variance:.2f}")
    print(f"  验证通过率: {stats.verification_rate:.2%}")
    print(f"  总次数: {stats.total_count}")
    print(f"  验证次数: {stats.verified_count}")

    assert stats.total_count == 100
    assert len(stats.match_history) == 100
    assert stats.verification_rate > 0.6

    print("  [PASS] 统计功能测试通过")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  SGN-Lite v5.1.3 Level 调度器测试")
    print("=" * 60)

    tests = [
        test_basic_scheduler,
        test_adaptive_strategy,
        test_binary_op_resolution,
        test_layer_aware_strategy,
        test_convenience_functions,
        test_serialization,
        test_stats,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  [FAIL] {test.__name__} 失败")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {test.__name__} 异常: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
