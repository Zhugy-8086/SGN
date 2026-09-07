# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""Task 2.4 测试：魔术方法 + 序列化/反序列化

覆盖：
  - __hash__: ValueSpec/UnitValue/PrecisionBudget 可哈希（set/dict 可用）
  - __eq__/__repr__: 魔术方法正确性
  - to_dict/from_dict: JSON 兼容序列化
  - to_json/from_json: JSON 字符串序列化
  - level_f/level_b 字段保留: PrecisionBudget 序列化含双向控制字段
  - 往返一致性: serialize → deserialize → 与原对象等价

运行: python test_serialization.py
"""
import sys
import os

# 添加 engine/sgn 目录到 path（用于 import sgn 包，即 engine/sgn/__init__.py）
_SGN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # engine/sgn/
_PARENT_DIR = os.path.dirname(_SGN_DIR)  # engine/ 添加 parent 以便 `from sgn import ...` 找到 engine/sgn/
sys.path.insert(0, _PARENT_DIR)

from sgn import ValueSpec, UnitValue, PrecisionBudget, LayerCost, ScaleFn
import json


# ============================================================
# ValueSpec 测试
# ============================================================

def test_value_spec_hash():
    """ValueSpec 可哈希"""
    s1 = ValueSpec(16)
    s2 = ValueSpec(16)
    s3 = ValueSpec(8)
    assert hash(s1) == hash(s2)
    assert hash(s1) != hash(s3)
    # 可用作 dict key
    d = {s1: "hello"}
    assert d[s2] == "hello"
    # 可加入 set
    s = {s1, s2, s3}
    assert len(s) == 2
    print("[PASS] test_value_spec_hash")


def test_value_spec_repr():
    """ValueSpec repr"""
    s = ValueSpec(16)
    r = repr(s)
    assert "16" in r
    assert "ValueSpec" in r
    print(f"[PASS] test_value_spec_repr: {r}")


def test_value_spec_to_dict():
    """ValueSpec to_dict"""
    s = ValueSpec(16, ScaleFn.MAX)
    d = s.to_dict()
    assert d == {"bits": 16, "scale": "MAX"}
    print(f"[PASS] test_value_spec_to_dict: {d}")


def test_value_spec_from_dict():
    """ValueSpec from_dict"""
    d = {"bits": 12, "scale": "RMS"}
    s = ValueSpec.from_dict(d)
    assert s.bits == 12
    assert s.scale == ScaleFn.RMS
    print(f"[PASS] test_value_spec_from_dict: bits={s.bits}")


def test_value_spec_json_roundtrip():
    """ValueSpec JSON 往返"""
    original = ValueSpec(20, ScaleFn.L2)
    json_str = original.to_json()
    restored = ValueSpec.from_json(json_str)
    assert restored == original
    # JSON 字符串可解析
    parsed = json.loads(json_str)
    assert parsed["bits"] == 20
    assert parsed["scale"] == "L2"
    print(f"[PASS] test_value_spec_json_roundtrip: {json_str}")


# ============================================================
# UnitValue 测试
# ============================================================

def test_unit_value_hash():
    """UnitValue 可哈希"""
    uv1 = UnitValue(100, ValueSpec(8))
    uv2 = UnitValue(100, ValueSpec(8))
    uv3 = UnitValue(200, ValueSpec(8))
    uv4 = UnitValue(100, ValueSpec(16))
    assert hash(uv1) == hash(uv2)
    assert hash(uv1) != hash(uv3)  # raw 不同
    assert hash(uv1) != hash(uv4)  # spec 不同
    s = {uv1, uv2, uv3, uv4}
    assert len(s) == 3
    print("[PASS] test_unit_value_hash")


def test_unit_value_eq():
    """UnitValue __eq__"""
    uv1 = UnitValue(100, ValueSpec(8))
    uv2 = UnitValue(100, ValueSpec(8))
    uv3 = UnitValue(200, ValueSpec(8))
    assert uv1 == uv2
    assert uv1 != uv3
    # 与非 UnitValue 比较
    assert uv1 != "not a unit value"
    print("[PASS] test_unit_value_eq")


def test_unit_value_repr():
    """UnitValue repr"""
    uv = UnitValue(100, ValueSpec(8))
    r = repr(uv)
    assert "100" in r
    assert "UnitValue" in r
    print(f"[PASS] test_unit_value_repr: {r}")


def test_unit_value_json_roundtrip():
    """UnitValue JSON 往返"""
    original = UnitValue(-500, ValueSpec(16, ScaleFn.L2))
    json_str = original.to_json()
    restored = UnitValue.from_json(json_str)
    assert restored == original
    print(f"[PASS] test_unit_value_json_roundtrip: {json_str}")


# ============================================================
# PrecisionBudget 测试
# ============================================================

def test_precision_budget_hash():
    """PrecisionBudget 可哈希"""
    layers = [LayerCost(c=100.0, b_min=4, b_max=20)]
    pb1 = PrecisionBudget(20, layers)
    pb2 = PrecisionBudget(20, [LayerCost(c=100.0, b_min=4, b_max=20)])
    pb3 = PrecisionBudget(30, layers)  # total_bits 不同
    assert hash(pb1) == hash(pb2)
    assert hash(pb1) != hash(pb3)
    print("[PASS] test_precision_budget_hash")


def test_precision_budget_repr():
    """PrecisionBudget repr"""
    pb = PrecisionBudget(124, [LayerCost(c=10.0)])
    r = repr(pb)
    assert "124" in r
    assert "layers=1" in r
    print(f"[PASS] test_precision_budget_repr: {r}")


def test_precision_budget_repr_with_level():
    """PrecisionBudget repr 含 level_f/level_b"""
    pb = PrecisionBudget(124, [LayerCost(c=10.0)])
    pb.set_level_f(ValueSpec(8))
    pb.set_level_b(ValueSpec(16))
    r = repr(pb)
    assert "level_f" in r
    assert "level_b" in r
    print(f"[PASS] test_precision_budget_repr_with_level: {r}")


def test_precision_budget_to_dict():
    """PrecisionBudget to_dict"""
    layers = [
        LayerCost(c=100.0, b_min=4, b_max=20),
        LayerCost(c=10.0, b_min=8, b_max=24),
    ]
    pb = PrecisionBudget(124, layers)
    d = pb.to_dict()
    assert d["total_bits"] == 124
    assert len(d["layers"]) == 2
    assert d["layers"][0] == {"c": 100.0, "b_min": 4, "b_max": 20}
    assert d["level_f"] is None
    assert d["level_b"] is None
    print(f"[PASS] test_precision_budget_to_dict: total_bits={d['total_bits']}")


def test_precision_budget_to_dict_with_level():
    """PrecisionBudget to_dict 含 level_f/level_b（序列化保留双向控制字段）"""
    layers = [LayerCost(c=100.0, b_min=4, b_max=20)]
    pb = PrecisionBudget(20, layers)
    pb.set_level_f(ValueSpec(8))
    pb.set_level_b(ValueSpec(16, ScaleFn.RMS))
    d = pb.to_dict()
    assert d["level_f"] == {"bits": 8, "scale": "MAX"}
    assert d["level_b"] == {"bits": 16, "scale": "RMS"}
    print(f"[PASS] test_precision_budget_to_dict_with_level: level_f={d['level_f']}, level_b={d['level_b']}")


def test_precision_budget_from_dict():
    """PrecisionBudget from_dict"""
    d = {
        "total_bits": 124,
        "layers": [
            {"c": 100.0, "b_min": 4, "b_max": 20},
            {"c": 10.0, "b_min": 8, "b_max": 24},
        ],
        "level_f": None,
        "level_b": None,
    }
    pb = PrecisionBudget.from_dict(d)
    assert pb.total_bits == 124
    assert len(pb.layers) == 2
    assert pb.layers[0].c == 100.0
    assert not pb.has_level_f
    assert not pb.has_level_b
    print("[PASS] test_precision_budget_from_dict")


def test_precision_budget_from_dict_with_level():
    """PrecisionBudget from_dict 含 level_f/level_b（反序列化恢复双向控制字段）"""
    d = {
        "total_bits": 20,
        "layers": [{"c": 100.0, "b_min": 4, "b_max": 20}],
        "level_f": {"bits": 8, "scale": "MAX"},
        "level_b": {"bits": 16, "scale": "RMS"},
    }
    pb = PrecisionBudget.from_dict(d)
    assert pb.has_level_f
    assert pb.has_level_b
    assert pb.level_f.bits == 8
    assert pb.level_b.bits == 16
    assert pb.level_b.scale == ScaleFn.RMS
    print(f"[PASS] test_precision_budget_from_dict_with_level: level_f.bits={pb.level_f.bits}, level_b.bits={pb.level_b.bits}")


def test_precision_budget_json_roundtrip():
    """PrecisionBudget JSON 往返（无 level_f/level_b）"""
    layers = [
        LayerCost(c=100.0, b_min=4, b_max=20),
        LayerCost(c=10.0, b_min=8, b_max=24),
    ]
    original = PrecisionBudget(124, layers)
    json_str = original.to_json()
    restored = PrecisionBudget.from_json(json_str)
    assert restored.total_bits == 124
    assert len(restored.layers) == 2
    assert restored.layers[0].c == 100.0
    assert not restored.has_level_f
    # allocate 结果一致
    assert original.allocate() == restored.allocate()
    print(f"[PASS] test_precision_budget_json_roundtrip")


def test_precision_budget_json_roundtrip_with_level():
    """PrecisionBudget JSON 往返（含 level_f/level_b）"""
    layers = [LayerCost(c=100.0, b_min=4, b_max=20)]
    original = PrecisionBudget(20, layers)
    original.set_level_f(ValueSpec(8))
    original.set_level_b(ValueSpec(16, ScaleFn.RMS))
    json_str = original.to_json()
    restored = PrecisionBudget.from_json(json_str)
    assert restored.has_level_f
    assert restored.has_level_b
    assert restored.level_f.bits == 8
    assert restored.level_b.bits == 16
    assert restored.level_b.scale == ScaleFn.RMS
    print(f"[PASS] test_precision_budget_json_roundtrip_with_level: {json_str}")


def test_precision_budget_json_file_compatible():
    """PrecisionBudget JSON 可写入文件再读取"""
    import tempfile
    layers = [LayerCost(c=100.0, b_min=4, b_max=20)]
    original = PrecisionBudget(124, layers)
    original.set_level_b(ValueSpec(16))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(original.to_json())
        path = f.name
    try:
        with open(path) as f:
            restored = PrecisionBudget.from_json(f.read())
        assert restored.total_bits == 124
        assert restored.has_level_b
        assert restored.level_b.bits == 16
    finally:
        os.remove(path)
    print("[PASS] test_precision_budget_json_file_compatible")


# ============================================================
# 综合测试
# ============================================================

def test_all_types_scale_fn_serialization():
    """所有 ScaleFn 枚举可序列化"""
    for fn in [ScaleFn.MAX, ScaleFn.RMS, ScaleFn.L2, ScaleFn.P95]:
        s = ValueSpec(16, fn)
        d = s.to_dict()
        restored = ValueSpec.from_dict(d)
        assert restored == s
        assert restored.scale == fn
    print("[PASS] test_all_types_scale_fn_serialization")


if __name__ == "__main__":
    test_value_spec_hash()
    test_value_spec_repr()
    test_value_spec_to_dict()
    test_value_spec_from_dict()
    test_value_spec_json_roundtrip()
    test_unit_value_hash()
    test_unit_value_eq()
    test_unit_value_repr()
    test_unit_value_json_roundtrip()
    test_precision_budget_hash()
    test_precision_budget_repr()
    test_precision_budget_repr_with_level()
    test_precision_budget_to_dict()
    test_precision_budget_to_dict_with_level()
    test_precision_budget_from_dict()
    test_precision_budget_from_dict_with_level()
    test_precision_budget_json_roundtrip()
    test_precision_budget_json_roundtrip_with_level()
    test_precision_budget_json_file_compatible()
    test_all_types_scale_fn_serialization()
    print("\n=== All serialization tests PASSED ===")
