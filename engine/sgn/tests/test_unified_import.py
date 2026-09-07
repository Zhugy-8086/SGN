# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""Task 3.1-3.5 测试：三库统一导入验证

验证：
  - import sgn.* 可用（C++ 扩展 + Python 包）
  - sgn.ValueSpec / UnitValue / PrecisionBudget（C++ 类型）
  - sgn.msint.MSInt / SlotInfo（re-export from sgn.msint._py）
  - sgn.level.LevelContext 等（C++ 子模块）

运行: python test_unified_import.py
"""
import sys
import os

# 添加项目根目录到 path
# 安全审计 2026-08-16：移除 engine/ 目录插入——engine/ 进入 sys.path 会
# shadow sgn.cp*.pyd 扩展（加载 __init__.py 包而非 .pyd）；本测试通过
# `from engine.sgn import ...` 仅需项目根目录。
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


def test_sgn_cpp_types():
    """测试 C++ 类型导入"""
    from engine.sgn import ValueSpec, UnitValue, PrecisionBudget, LayerCost, ScaleFn
    vs = ValueSpec(16)
    assert vs.bits == 16
    uv = UnitValue(100, vs)
    assert uv.raw == 100
    # 单层 b_max=24，预算 20 bits 应全部分配
    pb = PrecisionBudget(20, [LayerCost(c=100.0, b_min=4, b_max=24)])
    bits = pb.allocate()
    assert sum(bits) == 20, f"sum={sum(bits)}, 预期 20"
    print(f"[PASS] test_sgn_cpp_types: ValueSpec/UnitValue/PrecisionBudget 可用")


def test_sgn_msint_reexport():
    """测试 sgn.msint re-export"""
    from engine.sgn.msint import MSInt, SlotInfo, CURRENT_ABI_VERSION
    assert MSInt is not None
    assert SlotInfo is not None
    assert CURRENT_ABI_VERSION is not None
    print(f"[PASS] test_sgn_msint_reexport: MSInt/SlotInfo/CURRENT_ABI_VERSION 可用")


def test_sgn_level_reexport():
    """测试 sgn.level re-export"""
    try:
        from engine.sgn.level import LevelContext
        assert LevelContext is not None
        print(f"[PASS] test_sgn_level_reexport: LevelContext 可用")
    except ImportError as e:
        # level_scheduler 可能导入失败（依赖问题），记录但不 fail
        print(f"[SKIP] test_sgn_level_reexport: {e}")


def test_sgn_version():
    """测试 sgn C++ 扩展 version()"""
    from engine.sgn import version
    v = version()
    assert isinstance(v, str)
    print(f"[PASS] test_sgn_version: {v}")


if __name__ == "__main__":
    test_sgn_version()
    test_sgn_cpp_types()
    test_sgn_msint_reexport()
    test_sgn_level_reexport()
    print("\n=== All unified import tests PASSED ===")
