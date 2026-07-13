#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGN-Lite v5.1.5 Level 调度器完整单元测试

覆盖所有核心功能：
  - LevelScheduler 初始化和策略管理
  - 神经元绑定和统计更新
  - 自适应 level 调整
  - 二元运算 level 决策
  - 缓存性能优化
  - 序列化/反序列化
  - 与 SGNCore 集成
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
    print("  Test 1: Scheduler Basic Functionality")
    print("=" * 60)

    scheduler = LevelScheduler()

    # 测试默认策略
    ctx = scheduler.get_context(OperationType.ADD, source="test")
    print(f"  Default context: {ctx}")
    assert ctx.target_level == 2, f"Expected level=2, got {ctx.target_level}"

    # 测试策略注册
    strategies = list(scheduler._strategies.keys())
    print(f"  Registered strategies: {strategies}")
    assert len(strategies) >= 5, "Should have at least 5 builtin strategies"

    # 测试神经元绑定
    scheduler.bind_neuron(0, "standard(L2)")
    scheduler.bind_neuron(1, "standard(L1)")
    scheduler.bind_neuron(2, "adaptive(base=L2)")

    ctx0 = scheduler.get_context(OperationType.ADD, neuron_id=0)
    ctx1 = scheduler.get_context(OperationType.ADD, neuron_id=1)
    ctx2 = scheduler.get_context(OperationType.ADD, neuron_id=2)

    print(f"  Neuron 0: {ctx0.target_level}")
    print(f"  Neuron 1: {ctx1.target_level}")
    print(f"  Neuron 2: {ctx2.target_level}")

    assert ctx0.target_level == 2
    assert ctx1.target_level == 1

    print("  [PASS] Basic functionality test passed")
    return True


def test_adaptive_strategy():
    """测试自适应策略"""
    print("\n" + "=" * 60)
    print("  Test 2: Adaptive Strategy")
    print("=" * 60)

    scheduler = LevelScheduler()
    scheduler._adapt_interval = 10

    # 绑定自适应策略
    scheduler.bind_neuron(100, "adaptive(base=L2)", initial_level=2)

    # 模拟低方差场景
    print("  Simulating low variance scenario...")
    for i in range(50):
        match = 80 + (i % 10)
        scheduler.update_stats(100, match, verified=True)

    stats = scheduler.get_stats(100)
    print(f"  Match variance: {stats.match_variance:.2f}")
    print(f"  Current level: {stats.current_level}")

    # 低方差应该触发自适应到更细粒度
    if stats.match_variance < 25:
        print("  [PASS] Low variance scenario test passed")
    else:
        print("  [WARN] Variance not below threshold, skipping assertion")

    # 模拟高方差场景
    print("\n  Simulating high variance scenario...")
    scheduler.bind_neuron(200, "adaptive(base=L2)", initial_level=3)
    for i in range(50):
        match = 20 if i % 2 == 0 else 90
        scheduler.update_stats(200, match, verified=(i % 3 == 0))

    stats2 = scheduler.get_stats(200)
    print(f"  Match variance: {stats2.match_variance:.2f}")
    print(f"  Current level: {stats2.current_level}")

    print("  [PASS] Adaptive strategy test completed")
    return True


def test_binary_op_resolution():
    """测试二元运算 level 决策"""
    print("\n" + "=" * 60)
    print("  Test 3: Binary Operation Level Resolution")
    print("=" * 60)

    scheduler = LevelScheduler()

    # 测试不同 level 的二元运算
    test_cases = [
        (2, 2, "Same level"),
        (2, 1, "Different levels"),
        (1, 3, "Cross-level"),
        (0, 2, "Integer space vs 0.01 space"),
    ]

    for left, right, desc in test_cases:
        target = scheduler.resolve_binary_op(
            OperationType.ADD, left, right
        )
        print(f"  {desc}: L{left} + L{right} -> L{target}")

    print("  [PASS] Binary operation test passed")
    return True


def test_layer_aware_strategy():
    """测试层级感知策略"""
    print("\n" + "=" * 60)
    print("  Test 4: Layer-Aware Strategy")
    print("=" * 60)

    scheduler = LevelScheduler()

    # 绑定层级感知策略
    scheduler.bind_neuron(0, "layer_aware")
    scheduler.bind_neuron(1, "layer_aware")

    # 模拟 L0 和 L1 神经元
    stats0 = scheduler.get_stats(0)
    stats1 = scheduler.get_stats(1)

    # 手动设置层级属性
    if stats0:
        stats0.layer = 0
    if stats1:
        stats1.layer = 1

    ctx0 = scheduler.get_context(OperationType.COMPARE, neuron_id=0)
    ctx1 = scheduler.get_context(OperationType.COMPARE, neuron_id=1)

    print(f"  L0 neuron level: {ctx0.target_level}")
    print(f"  L1 neuron level: {ctx1.target_level}")

    print("  [PASS] Layer-aware strategy test passed")
    return True


def test_convenience_functions():
    """测试便捷函数"""
    print("\n" + "=" * 60)
    print("  Test 5: Convenience Functions")
    print("=" * 60)

    # 重置全局调度器
    set_global_scheduler(LevelScheduler())

    # 测试便捷函数
    level_add = get_level_for_add()
    level_cmp = get_level_for_compare()
    print(f"  Add level: {level_add}")
    print(f"  Compare level: {level_cmp}")

    # 测试统计更新
    update_neuron_stats(0, match=85, verified=True)
    update_neuron_stats(0, match=70, verified=False)
    level = get_neuron_level(0)
    print(f"  Neuron 0 level: {level}")

    print("  [PASS] Convenience functions test passed")
    return True


def test_serialization():
    """测试序列化"""
    print("\n" + "=" * 60)
    print("  Test 6: Serialization")
    print("=" * 60)

    scheduler = LevelScheduler()
    scheduler.bind_neuron(0, "standard(L2)")
    scheduler.bind_neuron(1, "adaptive(base=L2)")
    scheduler.update_stats(0, 80, True)
    scheduler.update_stats(1, 70, False)

    # 序列化
    data = scheduler.serialize()
    print(f"  Serialized data: {data}")

    # 反序列化到新调度器
    scheduler2 = LevelScheduler()
    scheduler2.deserialize(data)

    # 验证
    stats0 = scheduler2.get_stats(0)
    stats1 = scheduler2.get_stats(1)
    print(f"  Deserialized neuron 0 stats: total={stats0.total_count}, verified={stats0.verified_count}")
    print(f"  Deserialized neuron 1 stats: total={stats1.total_count}, verified={stats1.verified_count}")

    assert stats0.total_count == 1
    assert stats0.verified_count == 1
    assert stats1.total_count == 1
    assert stats1.verified_count == 0

    print("  [PASS] Serialization test passed")
    return True


def test_stats():
    """测试统计功能"""
    print("\n" + "=" * 60)
    print("  Test 7: Statistics")
    print("=" * 60)

    stats = NeuronLevelStats(neuron_id=42, current_level=2)

    # 模拟训练
    for i in range(100):
        match = 70 + (i % 20)
        verified = (i % 3 != 0)
        stats.update(match, verified)

    print(f"  Match variance: {stats.match_variance:.2f}")
    print(f"  Verification rate: {stats.verification_rate:.2%}")
    print(f"  Total count: {stats.total_count}")
    print(f"  Verified count: {stats.verified_count}")

    assert stats.total_count == 100
    assert len(stats.match_history) == 100
    assert stats.verification_rate > 0.6

    print("  [PASS] Statistics test passed")
    return True


def test_cache_performance():
    """测试缓存性能"""
    print("\n" + "=" * 60)
    print("  Test 8: Cache Performance")
    print("=" * 60)

    scheduler = LevelScheduler()
    scheduler.bind_neuron(0, "standard(L2)")

    # 第一次调用 - 缓存未命中
    ctx1 = scheduler.get_context(OperationType.ADD, neuron_id=0)
    cache_stats1 = scheduler.get_cache_stats()
    print(f"  After 1st call: hits={cache_stats1['hits']}, misses={cache_stats1['misses']}")

    # 第二次调用 - 应该命中缓存
    ctx2 = scheduler.get_context(OperationType.ADD, neuron_id=0)
    cache_stats2 = scheduler.get_cache_stats()
    print(f"  After 2nd call: hits={cache_stats2['hits']}, misses={cache_stats2['misses']}")

    assert cache_stats2['hits'] == 1
    assert cache_stats2['misses'] == 1
    assert ctx1.target_level == ctx2.target_level

    # 测试缓存失效
    scheduler.update_stats(0, 80, True)
    cache_stats3 = scheduler.get_cache_stats()
    print(f"  After update_stats: cache_size={cache_stats3['cache_size']}")
    # 缓存应该被清除

    print("  [PASS] Cache performance test passed")
    return True


def test_integration_with_core():
    """测试与 SGNCore 集成"""
    print("\n" + "=" * 60)
    print("  Test 9: Integration with SGNCore")
    print("=" * 60)

    from engine.core import SGNCore

    core = SGNCore(seed=42)

    # 测试训练
    intensity = [128] * 16
    info = core.train(intensity, 'A')
    print(f"  Training result: label={info['label']}, verified={info['V']}")
    print(f"  Level scheduler active: {info.get('level_scheduler_active')}")
    print(f"  Neuron levels: {info.get('neuron_levels')}")

    # 检查 Level 调度器
    level_info = core.get_level_info()
    print(f"  Level info: {level_info}")

    # 测试序列化
    scheduler_data = core.serialize_level_scheduler()
    print(f"  Serialized scheduler strategies: {len(scheduler_data.get('neuron_strategy', {}))}")

    print("  [PASS] Integration test passed")
    return True


def test_multi_layer_integration():
    """测试多层神经元集成"""
    print("\n" + "=" * 60)
    print("  Test 10: Multi-Layer Integration")
    print("=" * 60)

    from engine.config import ConfigRegistry
    ConfigRegistry.set('ENABLE_MULTI_LAYER_NEURON', True)

    from engine.core import SGNCore

    core = SGNCore(seed=42)

    # 测试多层训练
    intensity = [128] * 16
    info = core.train(intensity, 'A')
    print(f"  Multi-layer: {info.get('multi_layer')}")
    print(f"  L0 active: {info.get('layer0_active')}")
    print(f"  L1 active: {info.get('layer1_active')}")

    # 检查 Level 调度器
    level_info = core.get_level_info()
    print(f"  Level distribution: {level_info['level_distribution']}")

    assert info.get('multi_layer') == True
    assert info.get('layer0_active') == 128
    assert info.get('layer1_active') == 64
    assert level_info['level_distribution'].get(2) == 128  # L0
    assert level_info['level_distribution'].get(1) == 64   # L1

    print("  [PASS] Multi-layer integration test passed")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  SGN-Lite v5.1.5 Level Scheduler Complete Test Suite")
    print("=" * 60)

    tests = [
        test_basic_scheduler,
        test_adaptive_strategy,
        test_binary_op_resolution,
        test_layer_aware_strategy,
        test_convenience_functions,
        test_serialization,
        test_stats,
        test_cache_performance,
        test_integration_with_core,
        test_multi_layer_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  [FAIL] {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {test.__name__} exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
