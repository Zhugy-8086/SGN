# test_nested_view.py - NestedLevelView 薄层验收测试（W1-W4，Phase 1 接线）
#
# 预先登记验收钩子（nested_quant_Level接线设计_2026_09_05.md §六）：
#   W1 吸附映射档位正确性（D1 规则边界）
#   W2 跨层视图 = 码字的确定性函数（格点隶属 + 最近粗格点 + level 32 恒等）
#   W3 切换零重采样（任意档位序列两遍逐位一致 + 同 seed 码字确定）
#   W4 u 同源约束（u == float32(max|h|·2⁻³¹)）+ 零张量守卫 + 单元素恒等陷阱声明
#
# 运行：SGN 根目录 pytest engine/sgn/tests/test_nested_view.py

import numpy as np
import pytest

from engine.sgn.nested_view import NestedLevelView


def _ref_rtn(I, m):
    """规格 d 的整数域 RTN（half-to-even）——与 nested_api.h 冻结规格一致。"""
    q = I >> m
    if m > 0:
        r = I - (q << m)
        half = int(1) << (m - 1)
        if r > half or (r == half and (q & 1)):
            q += 1
    return q << m


def _random_h(rng, n=256, lo=-4, hi=6):
    mag = 10.0 ** rng.uniform(lo, hi)
    return (rng.standard_normal(n) * mag).astype(np.float32)


def test_w1_snap_rules():
    """W1：D1 吸附边界（b<6→4 / 6-11→8 / 12-23→16 / ≥24→32）。"""
    assert [NestedLevelView.snap(b) for b in (0, 4, 5)] == [4, 4, 4]
    assert [NestedLevelView.snap(b) for b in (6, 8, 11)] == [8, 8, 8]
    assert [NestedLevelView.snap(b) for b in (12, 16, 23)] == [16, 16, 16]
    assert [NestedLevelView.snap(b) for b in (24, 32, 100)] == [32, 32, 32]
    # 临界值各归其位
    assert NestedLevelView.snap(5) == 4 and NestedLevelView.snap(6) == 8
    assert NestedLevelView.snap(11) == 8 and NestedLevelView.snap(12) == 16
    assert NestedLevelView.snap(23) == 16 and NestedLevelView.snap(24) == 32


def test_w2_cross_level_views():
    """W2：跨层视图 = 码字的确定性函数（格点隶属 + 最近粗格点 + 32 恒等）。"""
    rng = np.random.default_rng(11)
    for _ in range(8):
        h = _random_h(rng)
        view = NestedLevelView(h, seed=20260905)
        assert view.u == float(np.float32(view.max_abs * 2.0 ** -31))
        d32 = view.dequant(32)
        # level 32 恒等：out == code·u（同一整数重建表达式）
        expect32 = (view.code.astype(np.float64) * np.float64(view.u)).astype(
            np.float32)
        assert np.array_equal(d32, expect32)
        for level in (4, 8, 16):
            m = 32 - level
            cell = int(1) << m
            half = int(1) << (m - 1)
            db = view.dequant(level)
            trunc = np.array([_ref_rtn(int(i), m) for i in view.code],
                             dtype=np.int64)
            # 格点隶属：out == (float)(trunc·u)
            expect_val = (trunc.astype(np.float64) * np.float64(view.u)).astype(
                np.float32)
            assert np.array_equal(db, expect_val)
            # 最近粗格点：|I − trunc| ≤ 2^(m−1)（tie 取等）
            dist = np.abs(view.code - trunc)
            assert int(dist.max()) <= half
            # 视图间关系：粗视图与细视图差 ≤ 一粗格（组合界，N2 同源）
            diff = np.abs(db.astype(np.float64) - d32.astype(np.float64))
            assert float(diff.max()) <= cell * view.u * (1.0 + 1e-12)


def test_w3_zero_resampling():
    """W3：切换零重采样——任意档位序列两遍逐位一致；同 seed 码字确定。"""
    rng = np.random.default_rng(22)
    h = _random_h(rng)
    seq = [int(v) for v in rng.choice([4, 8, 16, 32], size=50)]
    v1 = NestedLevelView(h, seed=42)
    v2 = NestedLevelView(h, seed=42)
    pass1 = [v1.dequant(level) for level in seq]
    pass2 = [v2.dequant(level) for level in seq]
    for a, b in zip(pass1, pass2):
        assert np.array_equal(a, b)
    # 同 seed 码字确定
    assert np.array_equal(v1.code, v2.code)
    # 档位序列来回切换 = 纯函数（无隐状态）
    back_forth = [v1.dequant(level) for level in (8, 4, 8, 16, 4, 8)]
    ref = [v1.dequant(level) for level in (8, 4, 8, 16, 4, 8)]
    for a, b in zip(back_forth, ref):
        assert np.array_equal(a, b)


def test_w3_seed_sensitivity():
    """W3 补：不同 seed 码字不同（seed 未进随机流则恒同）。"""
    rng = np.random.default_rng(33)
    h = _random_h(rng, n=500)
    c1 = NestedLevelView(h, seed=7).code
    c2 = NestedLevelView(h, seed=8).code
    assert not np.array_equal(c1, c2)


def test_w4_u_same_source_and_guards():
    """W4：u 同源 + 零张量守卫 + 单元素恒等陷阱声明。"""
    rng = np.random.default_rng(44)
    h = _random_h(rng)
    view = NestedLevelView(h, seed=1)
    # u 同源：恰为 float32(max|h|·2⁻³¹)，且与构造输入的 max 一致
    assert view.u == float(np.float32(view.max_abs * 2.0 ** -31))
    assert view.max_abs == float(np.max(np.abs(h)))
    # 零张量守卫：码字全 0，各档视图全 0
    z = NestedLevelView(np.zeros(16, dtype=np.float32), seed=1)
    assert np.all(z.code == 0)
    for level in NestedLevelView.LEVELS:
        assert np.all(z.dequant(level) == 0)
    # 单元素恒等陷阱（S0-A 嵌套形态）：所有视图 ≈ 原值——调用方须保证 n≥2
    one = NestedLevelView(np.array([0.7], dtype=np.float32), seed=1)
    for level in NestedLevelView.LEVELS:
        assert abs(float(one.dequant(level)[0]) - 0.7) < 1e-6
    # 非法 level：显式 ValueError（比 C++ 无操作更严格）
    with pytest.raises(ValueError):
        view.dequant(5)
    # storage 记账
    assert [view.storage_bits(level) for level in (4, 8, 16, 32)] == [5, 9, 17, 33]
    with pytest.raises(ValueError):
        view.storage_bits(5)


def test_w4_n0_and_large_magnitude():
    """W4 补：n=0 空视图；宽幅/饱和输入码字在 int32 域。"""
    empty = NestedLevelView(np.array([], dtype=np.float32), seed=1)
    assert empty.n == 0 and empty.u == 0.0
    assert empty.dequant(8).size == 0
    rng = np.random.default_rng(55)
    h = _random_h(rng, n=100, lo=2, hi=12)   # 幅值最高 1e12 ≫ 原语满幅
    view = NestedLevelView(h, seed=1)
    # int32 有符号域 [−2³¹, 2³¹−1]（|−2³¹| = 2147483648 合法，勿用 abs 断言）
    assert int(view.code.min()) >= -2147483648
    assert int(view.code.max()) <= 2147483647
    d = view.dequant(32)
    assert d.shape == h.shape


def test_phase2_view_codes():
    """Phase 2：view_codes 整数视图码——dequant 一致性 + 非对称域 + 32 恒等。"""
    rng = np.random.default_rng(66)
    h = _random_h(rng)
    view = NestedLevelView(h, seed=99)
    for level in (4, 8, 16, 32):
        m = 32 - level
        vc = view.view_codes(level)
        # 非对称域 [−2^(b−1), 2^(b−1)]（饱和顶码 RTN 上取整可达 +2^(b−1)）
        assert int(vc.min()) >= -(1 << (level - 1))
        assert int(vc.max()) <= (1 << (level - 1))
        # dequant 一致性：out == (float)(ldexp(vc, m)·u)（与 C++ 同式）
        expect = (vc.astype(np.float64) * float(2.0 ** m)
                  * np.float64(view.u)).astype(np.float32)
        assert np.array_equal(view.dequant(level), expect)
        if level == 32:
            assert np.array_equal(vc, view.code)
    # 零张量 → 全 0；非法 level 报错
    z = NestedLevelView(np.zeros(8, dtype=np.float32), seed=1)
    assert np.all(z.view_codes(8) == 0)
    with pytest.raises(ValueError):
        view.view_codes(5)
