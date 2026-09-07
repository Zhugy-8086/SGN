# test_msint_consistency.py - MSint Python/C++ 双实现一致性测试（基础设施 A4）
# 交叉校验：engine.sgn（C++ 扩展）vs engine.sgn.msint.cpp_backend 的 _Py* 参考实现，
# 随机输入下逐位一致。C++ 扩展不可用时 skip。
import sys
from pathlib import Path

import numpy as np
import pytest

_ENGINE = Path(__file__).resolve().parents[2]   # engine/
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from engine.sgn import (  # noqa: E402
    SplitDot, MSIntView, MultiScaleView, LeveledSplitDot,
)
from engine.sgn.msint.cpp_backend import (  # noqa: E402
    _PySplitDot, _PyMSIntView, _PyMultiScaleView, _PyLeveledSplitDot,
    _batch_get_all_py, _batch_get_all_cpp,
    _batch_decode_to_float_py, _batch_decode_to_float_cpp,
)

pytestmark = pytest.mark.skipif(
    SplitDot is None or _PySplitDot is None,
    reason="C++ MSint 扩展不可用（回退 Python 模式，无需交叉校验）")


@pytest.fixture
def rng():
    return np.random.default_rng(20260829)


def _rand_ints(rng, n, bits=32):
    return [int(x) for x in rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), size=n)]


# ---- SplitDot ----

@pytest.mark.parametrize("split_bits", [4, 8, 16])
def test_split_parts(rng, split_bits):
    for total_bits in (32, 64):
        if total_bits % split_bits:
            continue
        for _ in range(5):
            v = _rand_ints(rng, 1, bits=total_bits)[0]
            assert _PySplitDot.split_parts(v, total_bits, split_bits) == \
                SplitDot.split_parts(v, total_bits, split_bits)


@pytest.mark.parametrize("split_bits", [4, 8, 16])
@pytest.mark.parametrize("trim", [False, True])
def test_dot_split(rng, split_bits, trim):
    for _ in range(5):
        w = _rand_ints(rng, 12)
        x = _rand_ints(rng, 12)
        py = _PySplitDot.dot_split(w, x, 32, split_bits, trim)
        cpp = SplitDot.dot_split(w, x, 32, split_bits, trim)
        assert py == cpp


def test_dot_fused(rng):
    for split_bits in (8, 16):
        for _ in range(5):
            w = _rand_ints(rng, 12)
            x = _rand_ints(rng, 12)
            assert _PySplitDot.dot_fused(w, x, 32, split_bits) == \
                SplitDot.dot_fused(w, x, 32, split_bits)


def test_dot_fused_i32(rng):
    for split_bits in (4, 8, 16):
        for _ in range(5):
            w = _rand_ints(rng, 12)
            x = _rand_ints(rng, 12)
            assert _PySplitDot.dot_fused_i32(w, x, split_bits) == \
                SplitDot.dot_fused_i32(w, x, split_bits)


def test_fuse_128(rng):
    for split_bits in (4, 8, 16):
        for _ in range(5):
            w = _rand_ints(rng, 12)
            x = _rand_ints(rng, 12)
            partials = SplitDot.dot_split(w, x, 32, split_bits)
            assert _PySplitDot.fuse_128(partials, split_bits) == \
                SplitDot.fuse_128(partials, split_bits)


# ---- MSIntView ----

def test_msint_view_bitsplit(rng):
    for total_bits in (32, 64):
        for target in (4, 8, 16):
            if total_bits % target:
                continue
            for _ in range(5):
                v = _rand_ints(rng, 1, bits=total_bits)[0]
                assert _PyMSIntView.bitsplit(v, total_bits, target) == \
                    MSIntView.bitsplit(v, total_bits, target)


def test_msint_view_concat(rng):
    for _ in range(5):
        bits = [8, 8, 16]
        vals = _rand_ints(rng, len(bits), bits=16)
        assert _PyMSIntView.concat(vals, bits) == MSIntView.concat(vals, bits)
    # 有符号 concat
    for _ in range(5):
        bits = [4, 8, 12]
        vals = _rand_ints(rng, len(bits), bits=12)
        assert _PyMSIntView.concat_signed(vals, bits) == \
            MSIntView.concat_signed(vals, bits)


# ---- MultiScaleView ----

def test_multiscale_interpret_batch(rng):
    vals = _rand_ints(rng, 8)
    py = _PyMultiScaleView.interpret_batch(vals, 32)
    cpp = MultiScaleView.interpret_batch(vals, 32)
    assert set(py) == set(cpp)
    for b in py:
        assert py[b] == cpp[b]


# ---- LeveledSplitDot ----

def test_leveled_select_levels(rng):
    imp = _rand_ints(rng, 8)
    py = _PyLeveledSplitDot.select_levels(imp, 32, [16, 8, 4], [100, 1000])
    cpp = LeveledSplitDot.select_levels(imp, 32, [16, 8, 4], [100, 1000])
    assert py == cpp


def test_leveled_dot_split_leveled(rng):
    for _ in range(3):
        w = _rand_ints(rng, 8)
        x = _rand_ints(rng, 8)
        imp = _rand_ints(rng, 8, bits=12)
        py = _PyLeveledSplitDot.dot_split_leveled(
            w, x, imp, 32, [16, 8, 4], [100, 1000])
        cpp = LeveledSplitDot.dot_split_leveled(
            w, x, imp, 32, [16, 8, 4], [100, 1000])
        assert set(py) == set(cpp)
        for b in py:
            assert py[b] == cpp[b]


# ---- 批量解码（py vs cpp 函数）----

def test_batch_get_all_py_vs_cpp(rng):
    for bits in ([8, 8, 8, 8], [16, 8, 8], [4, 4, 4, 4, 4]):
        for _ in range(3):
            packed = [int(x) for x in rng.integers(0, 1 << 64, size=7, dtype=np.uint64)]
            arr = np.array(packed, dtype=np.uint64)
            py = np.asarray(_batch_get_all_py(bits, packed))
            cpp = np.asarray(_batch_get_all_cpp(bits, arr))
            assert cpp.shape == py.shape
            assert np.array_equal(cpp, py)


def test_batch_decode_to_float_py_vs_cpp(rng):
    for bits in ([8, 8], [8, 8, 8, 8]):
        for _ in range(3):
            packed = [int(x) for x in rng.integers(0, 1 << 64, size=9, dtype=np.uint64)]
            arr = np.array(packed, dtype=np.uint64)
            scale = 0.125
            py = np.asarray(_batch_decode_to_float_py(bits, packed, scale))
            cpp = np.asarray(_batch_decode_to_float_cpp(bits, arr, scale))
            assert py.shape == cpp.shape
            assert np.array_equal(py, cpp)   # 同公式 float 乘，逐位一致
