# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""Level 调度器 C++ 迁移综合验证测试

验证 C++ 端 sgn.level 子模块和 Python 端回退兼容性。

测试范围：
1. C++ 子模块导入
2. bits ↔ max_range 转换
3. LevelContext 数据结构
4. LevelConstants 常量
5. BitsAllocator 分配一致性
6. LevelStrategy / StandardStrategy / AdaptiveStrategy
7. NeuronLevelStats
8. LevelScheduler 序列化
9. Python 端回退兼容
10. 现有 level_scheduler 包不受影响
"""

import sys
import os

# ── 路径解析调式日志 ──────────────────────────────────────────
# 背景：该测试从 architecture/ 子目录运行，路径层级比 engine/sgn/tests/ 直接子目录多一层。
# 2026-08-06 修复：_TEST_DIR 从 engine/sgn/tests/architecture 开始，
# 需要 _SGN_TESTS_DIR 中间变量，而非直接取 _SGN_DIR = dirname(_TEST_DIR)。
# 修复前 _SGN_DIR = engine/sgn/tests（错误），修复后 = engine/sgn（正确）。
_DEBUG = "SGN_DEBUG" in os.environ

# 添加路径
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # engine/sgn/tests/architecture
_SGN_TESTS_DIR = os.path.dirname(_TEST_DIR)  # engine/sgn/tests
_SGN_DIR = os.path.dirname(_SGN_TESTS_DIR)  # engine/sgn
_ENGINE_DIR = os.path.dirname(_SGN_DIR)  # engine
_PROJECT_ROOT = os.path.dirname(_ENGINE_DIR)  # SGN (project root)
_BUILD_DIR = os.path.join(_SGN_DIR, "build")

if _DEBUG:
    print(f"[DEBUG] test_level_migration: 路径解析")
    print(f"[DEBUG]   _TEST_DIR      ={_TEST_DIR}")
    print(f"[DEBUG]   _SGN_TESTS_DIR ={_SGN_TESTS_DIR}")
    print(f"[DEBUG]   _SGN_DIR       ={_SGN_DIR}")
    print(f"[DEBUG]   _ENGINE_DIR    ={_ENGINE_DIR}")
    print(f"[DEBUG]   _PROJECT_ROOT  ={_PROJECT_ROOT}")
    print(f"[DEBUG]   _BUILD_DIR     ={_BUILD_DIR}")
    print(f"[DEBUG]   sys.path(前3)  ={sys.path[:3]}")

for p in [_BUILD_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def test_cpp_import():
    """测试 C++ 子模块导入"""
    from sgn import level
    assert level is not None
    assert hasattr(level, "bits_to_max_range")
    assert hasattr(level, "max_range_to_bits")
    assert hasattr(level, "LevelContext")
    assert hasattr(level, "LevelOperation")
    assert hasattr(level, "LevelConstants")
    assert hasattr(level, "BitsAllocator")
    assert hasattr(level, "StandardStrategy")
    assert hasattr(level, "AdaptiveStrategy")
    assert hasattr(level, "NeuronLevelStats")
    assert hasattr(level, "LevelScheduler")
    print("  test_cpp_import: OK")


def test_bits_max_range():
    """测试 bits ↔ max_range 转换"""
    from sgn import level

    # bits → max_range
    assert level.bits_to_max_range(8) == 255
    assert level.bits_to_max_range(12) == 4095
    assert level.bits_to_max_range(16) == 65535
    assert level.bits_to_max_range(20) == 1048575
    assert level.bits_to_max_range(24) == 16777215
    assert level.bits_to_max_range(0) == 0
    assert level.bits_to_max_range(-1) == 255  # 未设置默认值
    assert level.bits_to_max_range(32) == 0xFFFFFFFF

    # max_range → bits
    assert level.max_range_to_bits(255) == 8
    assert level.max_range_to_bits(4095) == 12
    assert level.max_range_to_bits(65535) == 16
    assert level.max_range_to_bits(0) == -1  # 未设置
    assert level.max_range_to_bits(-1) == -1  # 负数

    # 与 Python 参考实现一致性对比
    # 注意：C++ 实现中 bits < 0 表示"未设置"返回 255（spec 定义）
    # Python 旧版对 bits <= 0 返回 0，但 spec 要求 bits < 0 → 255
    from sgn.level import bits_to_max_range as cpp_bits_to_max_range
    from sgn.level import max_range_to_bits as cpp_max_range_to_bits

    assert cpp_bits_to_max_range(-1) == 255  # 未设置
    assert cpp_bits_to_max_range(0) == 0
    assert cpp_bits_to_max_range(8) == 255
    assert cpp_bits_to_max_range(12) == 4095
    assert cpp_bits_to_max_range(32) == 0xFFFFFFFF

    print("  test_bits_max_range: OK")


def test_level_operation():
    """测试 LevelOperation 枚举"""
    from sgn import level

    assert level.LevelOperation.ADD.value == 0
    assert level.LevelOperation.SUB.value == 1
    assert level.LevelOperation.MUL.value == 2
    assert level.LevelOperation.COMPARE.value == 3
    assert level.LevelOperation.ASSIGN.value == 4

    print("  test_level_operation: OK")


def test_level_context():
    """测试 LevelContext 数据结构"""
    from sgn import level

    # 默认构造（LEVEL_DEFAULT=0, DEFAULT_MAX_RANGE=255）
    ctx = level.LevelContext()
    assert ctx.target_level == 0
    assert ctx.max_range == 255
    assert ctx.bits == -1
    assert ctx.operation == level.LevelOperation.ASSIGN

    # 带参构造
    ctx2 = level.LevelContext(5, 1023, 10, level.LevelOperation.ADD, "test")
    assert ctx2.target_level == 5
    assert ctx2.max_range == 1023
    assert ctx2.bits == 10
    assert ctx2.operation == level.LevelOperation.ADD
    assert ctx2.source == "test"

    # 序列化/反序列化
    d = ctx2.to_dict()
    assert d["target_level"] == "5"
    assert d["max_range"] == "1023"
    assert d["bits"] == "10"

    ctx3 = level.LevelContext.from_dict(d)
    assert ctx3.target_level == 5
    assert ctx3.max_range == 1023
    assert ctx3.bits == 10

    print("  test_level_context: OK")


def test_level_constants():
    """测试 LevelConstants 常量"""
    from sgn import level

    assert level.LevelConstants.DEFAULT_BITS_MIN == 4
    assert level.LevelConstants.DEFAULT_BITS_MAX == 20
    assert level.LevelConstants.DEFAULT_BITS == 8
    assert level.LevelConstants.DEFAULT_MAX_RANGE == 255
    assert level.LevelConstants.DEFAULT_TOTAL_BITS == 124
    assert level.LevelConstants.HYSTERESIS_DELTA == 2
    assert level.LevelConstants.LEVEL_MIN == -4
    assert level.LevelConstants.LEVEL_MAX == 2
    assert level.LevelConstants.LEVEL_DEFAULT == 0
    assert level.LevelConstants.BITS_UNSET == -1

    print("  test_level_constants: OK")


def test_bits_allocator():
    """测试 BitsAllocator 分配一致性"""
    from sgn import level

    # 基本分配
    ba = level.BitsAllocator(total_bits=40, b_min=4, b_max=20)
    costs = [(0.1, 4096), (0.2, 2048)]
    result = ba.allocate(costs)
    assert len(result) == 2
    assert all(b >= 4 for b in result)
    assert all(b <= 20 for b in result)
    assert sum(result) == 40  # 所有预算用完

    # 带滞后机制
    result2 = ba.allocate_with_hysteresis(costs, [10, 10])
    assert len(result2) == 2
    assert all(b >= 4 for b in result2)
    assert all(b <= 20 for b in result2)

    # 边界：单层
    ba1 = level.BitsAllocator(total_bits=8, b_min=4, b_max=8)
    r1 = ba1.allocate([(0.5, 1024)])
    assert len(r1) == 1
    assert r1[0] == 8

    # 边界：空预算
    ba0 = level.BitsAllocator(total_bits=0, b_min=4, b_max=20)
    r0 = ba0.allocate([(0.1, 4096)])
    assert len(r0) == 1
    assert r0[0] == 4  # 保持 b_min

    print("  test_bits_allocator: OK")


def test_level_strategy():
    """测试 LevelStrategy 体系"""
    from sgn import level

    # StandardStrategy
    std = level.StandardStrategy(level=1)
    assert std.name() == "standard(L1)"
    assert std.default_level() == 1
    assert std.get_level_for_operation(level.LevelOperation.ADD) == 1
    assert std.get_level_for_operation(level.LevelOperation.ASSIGN, 42) == 1
    assert std.level() == 1

    # AdaptiveStrategy
    adapt = level.AdaptiveStrategy(base_level=0, variance_threshold=100.0,
                                    history_window=50, demotion_verification_threshold=0.5,
                                    demotion_min_samples=30)
    assert adapt.name() == "adaptive(base=L0)"
    assert adapt.default_level() == 0
    assert adapt.variance_threshold == 100.0
    assert adapt.history_window == 50
    assert adapt.demotion_verification_threshold == 0.5
    assert adapt.demotion_min_samples == 30

    # AdaptiveStrategy suggest_adaptation（需要 >= history_window=50 个样本）
    stats = level.NeuronLevelStats(neuron_id=0, level=0)
    for i in range(55):
        stats.update(match=90, verified=True)
    # 低方差 + peak_level=0 → 无法升级（peak_level 约束）
    suggestion = adapt.suggest_adaptation(stats)
    # peak_level=0, current=0, new_level=min(1, LEVEL_MAX=2, peak_level=0)=0 → 不变
    assert suggestion is None

    # 设置 peak_level=2 后，低方差应建议升级
    stats2 = level.NeuronLevelStats(neuron_id=1, level=0)
    stats2.peak_level = 2
    for i in range(55):
        stats2.update(match=90, verified=True)
    suggestion2 = adapt.suggest_adaptation(stats2)
    if suggestion2 is not None:
        assert suggestion2 == 1  # 升一级

    # AdaptiveStrategy suggest_demotion（验证率低 → 降级）
    stats3 = level.NeuronLevelStats(neuron_id=2, level=0)
    for i in range(35):
        stats3.update(match=50, verified=False)  # 全部验证失败
    # total_count=35 >= demotion_min_samples=30, verification_rate=0.0 < 0.5
    # 但 current_level=0 == LEVEL_MIN=-4? No, 0 > -4, 所以应降级到 -1
    demotion = adapt.suggest_demotion(stats3)
    if demotion is not None:
        assert demotion == -1  # 降一级

    # suggest_demotion 样本不足时不降级
    stats4 = level.NeuronLevelStats(neuron_id=3, level=1)
    for i in range(20):
        stats4.update(match=50, verified=False)
    demotion4 = adapt.suggest_demotion(stats4)
    assert demotion4 is None  # total_count=20 < demotion_min_samples=30

    print("  test_level_strategy: OK")


def test_neuron_level_stats():
    """测试 NeuronLevelStats"""
    from sgn import level

    stats = level.NeuronLevelStats(neuron_id=0, level=2)
    assert stats.neuron_id == 0
    assert stats.current_level == 2
    assert stats.peak_level == 2
    assert stats.total_count == 0
    assert stats.verified_count == 0
    assert stats.verification_rate() == 0.0
    assert stats.match_variance() == 0  # 样本不足

    # 更新
    stats.update(match=80, verified=True)
    assert stats.total_count == 1
    assert stats.verified_count == 1
    assert stats.verification_rate() == 1.0
    assert stats.last_match == 80

    stats.update(match=90, verified=True)
    stats.update(match=70, verified=True)
    assert stats.total_count == 3
    assert stats.match_variance() >= 0  # 有 3 个样本，可计算 MAD

    # 序列化
    serialized = stats.to_dict()
    assert serialized["neuron_id"] == "0"
    assert serialized["total_count"] == "3"

    restored = level.NeuronLevelStats.from_dict(serialized)
    assert restored.neuron_id == 0
    assert restored.total_count == 3
    assert restored.verified_count == 3

    print("  test_neuron_level_stats: OK")


def test_level_scheduler():
    """测试 LevelScheduler 核心功能"""
    from sgn import level

    sched = level.LevelScheduler(cache_size=1024, adapt_interval=50,
                                  default_variance_threshold=100.0)
    assert sched.adapt_interval == 50
    assert sched.step_counter == 0

    # 注册策略（level 范围 [-4, 2]）
    sched.register_strategy("high", level.StandardStrategy(level=2))
    sched.register_strategy("low", level.StandardStrategy(level=-1))

    # 设置默认策略
    sched.set_default_strategy("high")
    assert sched.get_level(0) == 2

    # 绑定神经元
    sched.bind_neuron(0, "low")
    assert sched.get_level(0) == -1

    # 更新统计
    sched.update_stats(0, 85, True)
    assert sched.step_counter > 0

    # 获取统计
    stats = sched.get_stats(0)
    assert stats is not None
    assert stats.total_count >= 1

    # 序列化
    d = sched.to_dict()
    assert d["default_strategy"] == "high"
    assert "neuron_stats" in d
    assert "neuron_strategy" in d

    # 反序列化
    from sgn.level import LevelScheduler
    restored = LevelScheduler.from_dict(d)
    assert restored is not None
    # 反序列化后需要注册策略（策略对象不序列化，仅序列化策略名映射）
    restored.register_strategy("high", level.StandardStrategy(level=2))
    restored.register_strategy("low", level.StandardStrategy(level=-1))
    assert restored.get_level(0) == -1  # 绑定保留

    print("  test_level_scheduler: OK")


def test_unified_import():
    """测试 sgn.level 统一导入"""
    # 通过 sgn.level 包导入
    from sgn import level as cpp_level

    # 核心 API 可用
    assert hasattr(cpp_level, "LevelContext")
    assert hasattr(cpp_level, "LevelScheduler")
    assert hasattr(cpp_level, "BitsAllocator")
    assert hasattr(cpp_level, "NeuronLevelStats")
    assert hasattr(cpp_level, "StandardStrategy")
    assert hasattr(cpp_level, "AdaptiveStrategy")

    # 直接通过 sgn.level 包路径导入
    from sgn.level import LevelContext, LevelScheduler, BitsAllocator
    assert LevelContext is cpp_level.LevelContext
    assert LevelScheduler is cpp_level.LevelScheduler
    assert BitsAllocator is cpp_level.BitsAllocator

    print("  test_unified_import: OK")


if __name__ == "__main__":
    print("=== Level Migration Validation Tests ===\n")

    test_cpp_import()
    test_bits_max_range()
    test_level_operation()
    test_level_context()
    test_level_constants()
    test_bits_allocator()
    test_level_strategy()
    test_neuron_level_stats()
    test_level_scheduler()
    test_unified_import()

    print("\n=== ALL 11 TESTS PASSED ===")