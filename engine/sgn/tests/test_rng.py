# test_rng.py - 随机性/可复现工具测试（基础设施 B3）
import numpy as np
import pytest

from engine.sgn.rng import (
    SeedBundle, make_seed_bundle, assert_reproducible,
    SR_CPP_OFFSET, ROAM_OFFSET, SR_OFFSET,
)


class _FakeAg:
    """模拟 sgn.autograd（记录 set_sr_seed 调用）。"""

    def __init__(self):
        self.sr_seed = None

    def set_sr_seed(self, v):
        self.sr_seed = v


def test_seed_bundle_deterministic():
    b1 = SeedBundle(7)
    b2 = SeedBundle(7)
    # 同 seed → 各 Generator 首个输出一致
    assert b1.init_rng.standard_normal(3) == pytest.approx(b2.init_rng.standard_normal(3))
    # 派生与手写约定一致（fresh default_rng(seed) 的首个输出）
    r = np.random.default_rng(7)
    assert SeedBundle(7).init_rng.standard_normal(1)[0] == r.standard_normal(1)[0]
    assert b1.sr_cpp_seed == SR_CPP_OFFSET + 7


def test_seed_bundle_offsets():
    b = SeedBundle(11)
    roam_ref = np.random.default_rng(11 + ROAM_OFFSET)
    sr_ref = np.random.default_rng(11 + SR_OFFSET)
    assert b.roam.standard_normal(2) == pytest.approx(roam_ref.standard_normal(2))
    assert b.sr.standard_normal(2) == pytest.approx(sr_ref.standard_normal(2))
    assert b.sr_cpp_seed == SR_CPP_OFFSET + 11


def test_seed_bundle_sets_cpp_sr():
    ag = _FakeAg()
    b = make_seed_bundle(7, ag=ag)
    assert ag.sr_seed == 1234 + 7


def test_seed_bundle_state_roundtrip():
    b = SeedBundle(7)
    b.init_rng.standard_normal(5)          # 消费一些状态
    st = b.state()
    b2 = SeedBundle(7)
    b2.init_rng.standard_normal(5)
    b2.restore(st)
    # 恢复后继续生成序列一致
    assert b.init_rng.standard_normal(4) == pytest.approx(b2.init_rng.standard_normal(4))


def test_assert_reproducible_pass():
    def task():
        b = SeedBundle(3)
        return [float(x) for x in b.init_rng.standard_normal(5)]
    outs = assert_reproducible(task, seed=3, repeats=2, label="t")
    assert outs[0] == outs[1]


def test_assert_reproducible_fail():
    def task():
        # 未固定随机源 → 每次不同
        return [float(x) for x in np.random.default_rng().standard_normal(5)]
    with pytest.raises(AssertionError):
        assert_reproducible(task, seed=3, repeats=2, label="t")


def test_assert_reproducible_repeats_guard():
    def task():
        return 1
    with pytest.raises(ValueError):
        assert_reproducible(task, seed=1, repeats=1)
