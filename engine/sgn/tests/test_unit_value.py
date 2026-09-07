# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""UnitValue 单元测试 — 覆盖 HC/MSint/Level 单位值场景

运行: python test_unit_value.py
"""
import sys
import os

# 安全审计 2026-08-16 A2-7：模式 B（顶层 import sgn + build 目录 hack）
# → 模式 A（import engine.sgn as sgn，统一项目根目录），与其余测试一致。
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import engine.sgn as sgn
import numpy as np


def test_basic_api():
    """基本 API 验证"""
    uv = sgn.UnitValue(100, sgn.ValueSpec(8))
    assert uv.raw == 100
    assert uv.spec.bits == 8
    assert uv.to_float(0.1) == 10.0
    assert "raw=100" in repr(uv)
    print("[PASS] test_basic_api")


def test_hc8_per_tensor():
    """HC8 per-tensor 场景：scale=max(|w|)/127（HC8WeightSchema 模式）"""
    w = [1.0, -0.5, 0.3, -0.8, 0.0]
    max_abs = max(abs(x) for x in w)
    scale = max_abs / 127.0
    spec = sgn.ValueSpec(8)
    uvs = [sgn.UnitValue.from_float(x, scale, spec) for x in w]
    w_dq = [uv.to_float(scale) for uv in uvs]
    for orig, dq in zip(w, w_dq):
        assert abs(orig - dq) <= scale, f"往返误差 {abs(orig - dq)} > {scale}"
    max_err = max(abs(o - d) for o, d in zip(w, w_dq))
    print(f"[PASS] test_hc8_per_tensor: scale={scale:.6f}, max_err={max_err:.6f}")


def test_hc16_per_tensor():
    """HC16 per-tensor 场景：scale=max(|w|)/32767"""
    w = [100.0, -50.0, 30.0, -80.0]
    max_abs = max(abs(x) for x in w)
    scale = max_abs / 32767.0
    spec = sgn.ValueSpec(16)
    uvs = [sgn.UnitValue.from_float(x, scale, spec) for x in w]
    w_dq = [uv.to_float(scale) for uv in uvs]
    for orig, dq in zip(w, w_dq):
        assert abs(orig - dq) <= scale
    print(f"[PASS] test_hc16_per_tensor: scale={scale:.6f}")


def test_hc12_non_standard_bits():
    """HC12 非常规 bits：signed_max=2047"""
    assert sgn.UnitValue.signed_max_for_bits(12) == 2047
    spec = sgn.ValueSpec(12)
    uv = sgn.UnitValue.from_float(1000.0, 0.5, spec)
    assert uv.raw == 2000
    assert uv.to_float(0.5) == 1000.0
    # clamp 测试
    uv_clamped = sgn.UnitValue.from_float(100000.0, 0.5, spec)
    assert uv_clamped.raw == 2047, f"expected 2047, got {uv_clamped.raw}"
    print("[PASS] test_hc12_non_standard_bits")


def test_clamp():
    """clamp 验证：超出范围的值被正确 clamp"""
    spec8 = sgn.ValueSpec(8)
    uv = sgn.UnitValue.from_float(1000.0, 0.1, spec8)
    assert uv.raw == 127, f"expected 127, got {uv.raw}"
    uv = sgn.UnitValue.from_float(-1000.0, 0.1, spec8)
    assert uv.raw == -127, f"expected -127, got {uv.raw}"
    print("[PASS] test_clamp")


def test_scale_zero():
    """scale=0 边界：from_float 应返回 raw=0（避免除零）"""
    spec = sgn.ValueSpec(8)
    uv = sgn.UnitValue.from_float(5.0, 0.0, spec)
    assert uv.raw == 0
    assert uv.spec.bits == 8
    print("[PASS] test_scale_zero")


def test_per_layer_batch():
    """per-layer 场景：不同层不同 bits 和 scale"""
    specs = [sgn.ValueSpec(8), sgn.ValueSpec(16), sgn.ValueSpec(12)]
    scales = [0.01, 0.001, 0.005]
    values = [1.0, -2.0, 0.5]
    uvs = [sgn.UnitValue.from_float(x, s, sp) for x, s, sp in zip(values, scales, specs)]
    w_dq = [uv.to_float(s) for uv, s in zip(uvs, scales)]
    for orig, dq in zip(values, w_dq):
        assert abs(orig - dq) <= max(scales)
    print(f"[PASS] test_per_layer_batch: dq={[round(d, 4) for d in w_dq]}")


def test_batch_to_float_per_tensor():
    """batch_to_float per-tensor（单个 scale）"""
    spec = sgn.ValueSpec(8)
    uvs = [sgn.UnitValue(10, spec), sgn.UnitValue(20, spec), sgn.UnitValue(-30, spec)]
    result = sgn.batch_to_float(uvs, 0.1)
    assert isinstance(result, np.ndarray)
    expected = [1.0, 2.0, -3.0]
    for i, e in enumerate(expected):
        assert abs(result[i] - e) < 1e-6
    print(f"[PASS] test_batch_to_float_per_tensor: result={result.tolist()}")


def test_batch_to_float_per_layer():
    """batch_to_float per-layer（数组 scale）"""
    spec = sgn.ValueSpec(8)
    uvs = [sgn.UnitValue(10, spec), sgn.UnitValue(20, spec), sgn.UnitValue(30, spec)]
    scales = [0.1, 0.01, 0.001]
    result = sgn.batch_to_float(uvs, scales)
    expected = [1.0, 0.2, 0.03]
    for i, e in enumerate(expected):
        assert abs(result[i] - e) < 1e-6
    print(f"[PASS] test_batch_to_float_per_layer: result={result.tolist()}")


def test_batch_from_float_numpy():
    """batch_from_float numpy 互操作"""
    spec = sgn.ValueSpec(8)
    values = np.array([1.0, -0.5, 0.3, -0.8], dtype=np.float32)
    scale = 1.0 / 127.0
    uvs = sgn.batch_from_float(values, scale, spec)
    assert len(uvs) == 4
    result = sgn.batch_to_float(uvs, scale)
    for orig, dq in zip(values, result):
        assert abs(orig - dq) <= scale * 1.5
    print(f"[PASS] test_batch_from_float_numpy: dq={[round(d, 4) for d in result]}")


def test_value_spec_compat():
    """ValueSpec 兼容性：from_max_range/to_max_range（Level 桥接）"""
    spec = sgn.ValueSpec.from_max_range(255)
    assert spec.bits == 8
    assert spec.to_max_range() == 255
    uv = sgn.UnitValue(64, spec)
    assert uv.spec.bits == 8
    assert uv.to_float(1.0) == 64.0
    spec16 = sgn.ValueSpec.from_max_range(65535)
    assert spec16.bits == 16
    print("[PASS] test_value_spec_compat")


def test_level_bridge():
    """Level 场景模拟：max_range 桥接 + LinearMapping 等价

    Level 的 LinearMapping.map(raw, max_range) = float(raw)
    UnitValue 等价：to_float(scale=1.0) = raw * 1.0 = float(raw)
    """
    spec = sgn.ValueSpec.from_max_range(255)
    raw_val = 100
    uv = sgn.UnitValue(raw_val, spec)
    assert uv.to_float(1.0) == 100.0
    print("[PASS] test_level_bridge: Level LinearMapping 等价")


def test_hc20_hc24_large_bits():
    """HC20/HC24 大 bits 场景"""
    assert sgn.UnitValue.signed_max_for_bits(20) == 524287
    assert sgn.UnitValue.signed_max_for_bits(24) == 8388607
    spec20 = sgn.ValueSpec(20)
    uv = sgn.UnitValue.from_float(100000.0, 0.2, spec20)
    assert uv.raw == 500000
    assert uv.to_float(0.2) == 100000.0
    print("[PASS] test_hc20_hc24_large_bits")


def test_equality():
    """相等性验证"""
    spec = sgn.ValueSpec(8)
    uv1 = sgn.UnitValue(100, spec)
    uv2 = sgn.UnitValue(100, spec)
    uv3 = sgn.UnitValue(101, spec)
    assert uv1 == uv2
    assert uv1 != uv3
    print("[PASS] test_equality")


def test_msint_slot_precision():
    """MSint 场景模拟：UnitValue 仅描述 slot 精度（无 scale 运算）

    MSint PackedBackend 是纯位运算，不涉及 scale。
    UnitValue 可用于描述 slot 的精度规格。
    """
    # MSint slot: 8-bit value + 8-bit value_low（backward_int16 视角）
    value_spec = sgn.ValueSpec(8)
    value_low_spec = sgn.ValueSpec(8)
    # 存储 raw 值，scale 不使用（传 0 或 1）
    uv_val = sgn.UnitValue(100, value_spec)
    uv_low = sgn.UnitValue(50, value_low_spec)
    # 验证 raw 可读回（MSint get 操作等价）
    assert uv_val.raw == 100
    assert uv_low.raw == 50
    # concat 视角：value + value_low → 16-bit
    concat_raw = (uv_val.raw << 8) | (uv_low.raw & 0xFF)
    assert concat_raw == 25650  # 100*256 + 50
    print(f"[PASS] test_msint_slot_precision: concat_raw={concat_raw}")


if __name__ == "__main__":
    test_basic_api()
    test_hc8_per_tensor()
    test_hc16_per_tensor()
    test_hc12_non_standard_bits()
    test_hc20_hc24_large_bits()
    test_clamp()
    test_scale_zero()
    test_per_layer_batch()
    test_batch_to_float_per_tensor()
    test_batch_to_float_per_layer()
    test_batch_from_float_numpy()
    test_value_spec_compat()
    test_level_bridge()
    test_msint_slot_precision()
    test_equality()
    print("\n=== All UnitValue tests PASSED ===")
