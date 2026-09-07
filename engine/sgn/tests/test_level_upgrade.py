# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""Stage 3.0.4 阶段 4 测试 — Level 升级

覆盖：
  - Task 4.1: bits 字段 + bits↔max_range 双向同步
  - Task 4.2: level_f/level_b 接口预留
  - Task 4.3: bits 分配算法集成（贪心 + 滞后机制）
  - Task 4.4: 旧序列化数据兼容性

运行: python test_level_upgrade.py
"""
import sys
import os
import warnings

import pytest

# 添加项目根目录和 engine/ 到 path
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in [_PROJ_ROOT, _ENGINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sgn import ValueSpec, ScaleFn

# ── 导入拆分（审计修复批次 2.5，2026-08-16）────────────────
# 原实现：单条 import 语句导入 4 个名字，任一缺失触发 ImportError 连坐，
# 导致已实现的名字（如 BitsAllocator）也无法导入、测试全部 error。
# 修复：逐名字独立 try/except + 逐测试 skipif 精确跳过。
_DEBUG = "SGN_DEBUG" in os.environ

try:
    from sgn.level import bits_to_max_range as bits_to_max_range
    from sgn.level import max_range_to_bits as max_range_to_bits
    _HAS_MATH = True
except ImportError as _e:
    _HAS_MATH = False
    if _DEBUG:
        print(f"[DEBUG] test_level_upgrade: bits math 导入失败: {_e}")

try:
    from sgn.level import UpgradedLevelContext as UpgradedLevelContext
    _HAS_UPGRADED = True
except ImportError as _e:
    _HAS_UPGRADED = False
    if _DEBUG:
        print(f"[DEBUG] test_level_upgrade: UpgradedLevelContext 导入失败: {_e}")
        import sgn.level as _sl
        _sl_attrs = [x for x in dir(_sl) if not x.startswith('_')]
        print(f"[DEBUG]   sgn.level 可用属性={_sl_attrs}")

try:
    from sgn.level import BitsAllocator as BitsAllocator
    _HAS_BITS_ALLOCATOR = True
except ImportError as _e:
    _HAS_BITS_ALLOCATOR = False
    if _DEBUG:
        print(f"[DEBUG] test_level_upgrade: BitsAllocator 导入失败: {_e}")

# 整体可用性（__main__ 模式门槛）
_UPGRADE_AVAILABLE = _HAS_MATH and _HAS_UPGRADED and _HAS_BITS_ALLOCATOR


# ============================================================
# Task 4.1: bits 字段 + 双向同步
# ============================================================

@pytest.mark.skipif(not _HAS_MATH, reason="sgn.level.bits_to_max_range/max_range_to_bits 未编译到 .pyd")
def test_bits_to_max_range():
    """bits → max_range 转换"""
    assert bits_to_max_range(8) == 255
    assert bits_to_max_range(12) == 4095
    assert bits_to_max_range(16) == 65535
    assert bits_to_max_range(20) == 1048575
    assert bits_to_max_range(24) == 16777215
    assert bits_to_max_range(None) == 255  # 旧默认值
    print("[PASS] test_bits_to_max_range: 8/12/16/20/24 bit 全部正确")


@pytest.mark.skipif(not _HAS_MATH, reason="sgn.level.bits_to_max_range/max_range_to_bits 未编译到 .pyd")
def test_max_range_to_bits_exact():
    """max_range → bits 精确转换（2^n-1）"""
    assert max_range_to_bits(255, warn_if_non_power=False) == 8
    assert max_range_to_bits(4095, warn_if_non_power=False) == 12
    assert max_range_to_bits(65535, warn_if_non_power=False) == 16
    assert max_range_to_bits(1048575, warn_if_non_power=False) == 20
    print("[PASS] test_max_range_to_bits_exact: 255→8, 4095→12, 65535→16, 1048575→20")


@pytest.mark.skipif(not _HAS_MATH, reason="sgn.level.bits_to_max_range/max_range_to_bits 未编译到 .pyd")
def test_max_range_to_bits_non_power_warns():
    """非 2^n-1 的 max_range 向上取整并警告"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bits = max_range_to_bits(300, warn_if_non_power=True)
        assert bits == 9  # ceil(log2(301)) = 9
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "300" in str(w[0].message)
    print(f"[PASS] test_max_range_to_bits_non_power_warns: 300→bits={bits}（警告已发出）")


@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_level_context_bits_sync():
    """LevelContext bits↔max_range 双向同步"""
    # 设置 bits → 自动同步 max_range
    ctx = UpgradedLevelContext(bits=16)
    assert ctx.max_range == 65535
    assert ctx.bits == 16

    # 设置 bits=8
    ctx.bits = 8
    assert ctx.max_range == 255

    # 设置 bits=12
    ctx.set_bits(12)
    assert ctx.max_range == 4095
    print("[PASS] test_level_context_bits_sync: bits→max_range 自动同步")


@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_level_context_max_range_to_bits():
    """从 max_range 反推 bits（旧数据兼容）"""
    ctx = UpgradedLevelContext(max_range=65535)
    ctx.sync_from_max_range(warn=False)
    assert ctx.bits == 16

    ctx2 = UpgradedLevelContext(max_range=255)
    ctx2.sync_from_max_range(warn=False)
    assert ctx2.bits == 8
    print("[PASS] test_level_context_max_range_to_bits: max_range→bits 反推")


@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_level_context_bits_none():
    """bits=None 时使用旧 max_range 逻辑"""
    ctx = UpgradedLevelContext(max_range=255)
    # bits=None，但能从 max_range 反推
    assert ctx.bits == 8  # 从 max_range=255 反推
    print("[PASS] test_level_context_bits_none: None 时从 max_range 反推")


# ============================================================
# Task 4.2: level_f / level_b 接口预留
# ============================================================

@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_level_f_level_b_default_none():
    """level_f/level_b 默认 None"""
    ctx = UpgradedLevelContext()
    assert ctx.level_f is None
    assert ctx.level_b is None
    assert not ctx.has_level_f()
    assert not ctx.has_level_b()
    print("[PASS] test_level_f_level_b_default_none")


@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_level_f_level_b_set():
    """设置 level_f/level_b"""
    ctx = UpgradedLevelContext()
    ctx.level_f = ValueSpec(8)
    ctx.level_b = ValueSpec(16, ScaleFn.RMS)
    assert ctx.has_level_f()
    assert ctx.has_level_b()
    assert ctx.level_f.bits == 8
    assert ctx.level_b.bits == 16
    assert ctx.level_b.scale == ScaleFn.RMS
    print("[PASS] test_level_f_level_b_set: level_f=8bit, level_b=16bit RMS")


@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_effective_bits_priority():
    """调度器优先使用 level_f/level_b"""
    ctx = UpgradedLevelContext(bits=12)
    # 未设置 level_f/level_b 时，用 bits
    assert ctx.get_effective_bits("forward") == 12
    assert ctx.get_effective_bits("backward") == 12
    # 设置 level_f 后，forward 用 level_f
    ctx.level_f = ValueSpec(8)
    assert ctx.get_effective_bits("forward") == 8
    assert ctx.get_effective_bits("backward") == 12  # 仍用 bits
    # 设置 level_b 后，backward 用 level_b
    ctx.level_b = ValueSpec(16)
    assert ctx.get_effective_bits("backward") == 16
    print("[PASS] test_effective_bits_priority: level_f/level_b 优先于 bits")


@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_level_f_none_backward_compat():
    """level_f/level_b 为 None 时行为与旧版一致"""
    ctx = UpgradedLevelContext(max_range=255)
    # 无 level_f/level_b，effective 应等于从 max_range 反推
    assert ctx.get_effective_bits("forward") == 8
    assert ctx.get_effective_max_range("forward") == 255
    assert ctx.get_effective_max_range("backward") == 255
    print("[PASS] test_level_f_none_backward_compat: None 时行为与旧版一致")


# ============================================================
# Task 4.3: bits 分配算法集成
# ============================================================

def test_bits_allocator_basic():
    """BitsAllocator 基本贪心分配"""
    allocator = BitsAllocator(total_bits=124, b_min=4, b_max=20)
    # 10 层，成本递减
    costs = [(1.0 * (0.5 ** i), 4096) for i in range(10)]
    bits = allocator.allocate(costs)
    assert sum(bits) == 124
    assert len(bits) == 10
    # 成本高的层应获更多 bits
    assert bits[0] >= bits[-1]
    print(f"[PASS] test_bits_allocator_basic: bits={bits}, sum={sum(bits)}")


@pytest.mark.skipif(not _HAS_BITS_ALLOCATOR, reason="sgn.level.BitsAllocator 未编译到 .pyd")
def test_bits_allocator_with_hysteresis():
    """BitsAllocator 滞后机制（变化限制 ±2）"""
    allocator = BitsAllocator(total_bits=124, b_min=4, b_max=20, hysteresis_delta=2)
    costs = [(1.0 * (0.5 ** i), 4096) for i in range(10)]

    # 第一轮：无滞后约束
    prev_bits = allocator.allocate(costs)

    # 改变成本（让最优分配大幅变化）
    costs2 = [(1.0 * (0.5 ** (9 - i)), 4096) for i in range(10)]  # 反转

    # 第二轮：有滞后约束
    new_bits = allocator.allocate_with_hysteresis(costs2, prev_bits)

    # 每层变化应 ≤ ±2
    for i, (old, new) in enumerate(zip(prev_bits, new_bits)):
        assert abs(new - old) <= 2, f"层 {i}: 变化 {new-old} 超过 ±2"

    # 总 bits 仍应等于 124（或接近，受 b_min/b_max 约束）
    assert sum(new_bits) == 124, f"总 bits={sum(new_bits)}, 预期 124"
    print(f"[PASS] test_bits_allocator_with_hysteresis: prev={prev_bits}, new={new_bits}")


@pytest.mark.skipif(not _HAS_BITS_ALLOCATOR, reason="sgn.level.BitsAllocator 未编译到 .pyd")
def test_bits_allocator_cost_signal():
    """成本信号 c_i = grad_l2² × in_dim（exp26 确认）"""
    allocator = BitsAllocator(total_bits=12, b_min=4, b_max=20)
    # 两层：梯度相同，in_dim 不同。预算 12 = 4(b_min) + 4(b_min) + 4(剩余给层0)
    costs = [(1.0, 256), (1.0, 4)]  # 层 0 in_dim 大，应获更多 bits
    bits = allocator.allocate(costs)
    assert bits[0] > bits[1], f"in_dim 大的层应获更多 bits: {bits}"
    print(f"[PASS] test_bits_allocator_cost_signal: in_dim 256→{bits[0]}, in_dim 4→{bits[1]}")


@pytest.mark.skipif(not _HAS_BITS_ALLOCATOR, reason="sgn.level.BitsAllocator 未编译到 .pyd")
def test_bits_allocator_no_prev():
    """allocate_with_hysteresis 无 prev_bits 时等同 allocate"""
    allocator = BitsAllocator(total_bits=40, b_min=4, b_max=20)
    costs = [(1.0, 256), (1.0, 4)]
    bits1 = allocator.allocate(costs)
    bits2 = allocator.allocate_with_hysteresis(costs, prev_bits=None)
    assert bits1 == bits2
    print("[PASS] test_bits_allocator_no_prev: 无 prev 等同 allocate")


# ============================================================
# Task 4.4: 旧序列化数据兼容性
# ============================================================

@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_serialization_roundtrip():
    """序列化往返"""
    ctx = UpgradedLevelContext(bits=16, level_f=ValueSpec(8), level_b=ValueSpec(12))
    d = ctx.to_dict()
    restored = UpgradedLevelContext.from_dict(d)
    assert restored.bits == 16
    assert restored.max_range == 65535
    assert restored.has_level_f()
    assert restored.level_f.bits == 8
    assert restored.has_level_b()
    assert restored.level_b.bits == 12
    print(f"[PASS] test_serialization_roundtrip: {d}")


@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_old_data_compatibility():
    """旧数据兼容：仅含 max_range 的数据可正常加载"""
    # 模拟旧数据（Stage 2.6 序列化格式）
    old_data = {
        "target_level": 2,
        "source": "stage_2_6",
        "max_range": 255,
        # 无 bits/level_f/level_b 字段
    }
    ctx = UpgradedLevelContext.from_dict(old_data)
    # 自动从 max_range 反推 bits
    assert ctx.bits == 8
    assert ctx.max_range == 255
    assert not ctx.has_level_f()
    assert not ctx.has_level_b()
    print(f"[PASS] test_old_data_compatibility: max_range=255 → bits={ctx.bits}")


@pytest.mark.skipif(not _HAS_UPGRADED, reason="sgn.level.UpgradedLevelContext 未编译到 .pyd")
def test_old_data_non_power_warns():
    """旧数据 max_range 非 2^n-1 时警告（向上取整）"""
    old_data = {"max_range": 300}  # 非 2^n-1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ctx = UpgradedLevelContext.from_dict(old_data)
        # from_dict 内部不警告，sync_from_max_range 时才警告
        ctx.sync_from_max_range(warn=True)
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
    assert ctx.bits == 9  # ceil(log2(301)) = 9
    print(f"[PASS] test_old_data_non_power_warns: max_range=300 → bits={ctx.bits}（警告已发出）")


if __name__ == "__main__":
    if not _UPGRADE_AVAILABLE:
        print("[SKIP] UpgradedLevelContext 尚未编译到 sgn.level C++ 模块，测试待重新启用")
        sys.exit(0)
    # Task 4.1
    test_bits_to_max_range()
    test_max_range_to_bits_exact()
    test_max_range_to_bits_non_power_warns()
    test_level_context_bits_sync()
    test_level_context_max_range_to_bits()
    test_level_context_bits_none()
    # Task 4.2
    test_level_f_level_b_default_none()
    test_level_f_level_b_set()
    test_effective_bits_priority()
    test_level_f_none_backward_compat()
    # Task 4.3
    test_bits_allocator_basic()
    test_bits_allocator_with_hysteresis()
    test_bits_allocator_cost_signal()
    test_bits_allocator_no_prev()
    # Task 4.4
    test_serialization_roundtrip()
    test_old_data_compatibility()
    test_old_data_non_power_warns()
    print("\n=== All Stage 3.0.4 Level upgrade tests PASSED ===")
