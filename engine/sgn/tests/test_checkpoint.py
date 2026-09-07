# test_checkpoint.py - 训练检查点保存/恢复测试（基础设施 B2）
import os

import numpy as np
import pytest

from engine.sgn.checkpoint import save_checkpoint, load_checkpoint
from engine.sgn.optimizer import SGD


def test_save_load_roundtrip(tmp_path):
    """参数 + 优化器 + step/seed/extra 往返完整。"""
    rng = np.random.default_rng(3)
    p = {"w": rng.standard_normal((6, 4)).astype(np.float32),
         "b": rng.standard_normal((3,)).astype(np.float32)}
    opt = SGD(p, lr=0.02, momentum=0.9, classical=True)
    opt.step({"w": np.ones_like(p["w"]), "b": np.ones_like(p["b"])})

    path = os.path.join(str(tmp_path), "ck.npz")
    save_checkpoint(path, params=p, optimizer=opt, step=42, seed=7, extra={"mode": "protocol"})

    ck = load_checkpoint(path)
    assert ck["step"] == 42 and ck["seed"] == 7
    assert ck["extra"] == {"mode": "protocol"}
    assert set(ck["params"]) == {"w", "b"}
    assert np.array_equal(ck["params"]["w"], p["w"])
    # 优化器状态恢复
    sd = ck["optimizer"]
    assert sd["optimizer"] == "SGD"
    assert sd["defaults"]["classical"] is True
    assert np.array_equal(sd["state"]["w"]["v"], opt._state["w"]["v"])


def test_resume_equals_continuous(tmp_path):
    """保存→恢复→续训 与 不间断训练 loss 逐位一致（params + optimizer 一起恢复）。"""
    rng = np.random.default_rng(5)
    p0 = rng.standard_normal((8,)).astype(np.float32)
    grads = [rng.standard_normal((8,)).astype(np.float32) for _ in range(6)]

    # 不间断：6 步
    p_a = {"w": p0.copy()}
    opt_a = SGD(p_a, lr=0.1, momentum=0.9)
    for g in grads:
        opt_a.step({"w": g})

    # 前 3 步 → save → 恢复 → 后 3 步
    p_b = {"w": p0.copy()}
    opt_b = SGD(p_b, lr=0.1, momentum=0.9)
    for g in grads[:3]:
        opt_b.step({"w": g})
    path = os.path.join(str(tmp_path), "resume.npz")
    save_checkpoint(path, params=p_b, optimizer=opt_b, step=3)

    ck = load_checkpoint(path)
    p_c = {k: v.copy() for k, v in ck["params"].items()}
    opt_c = SGD(p_c, lr=0.1, momentum=0.9)
    opt_c.load_state_dict(ck["optimizer"])
    assert ck["step"] == 3
    for g in grads[3:]:
        opt_c.step({"w": g})

    assert np.array_equal(p_a["w"], p_c["w"])


def test_resume_rng_state(tmp_path):
    """随机源状态保存/恢复：恢复后 Generator 序列不中断。"""
    rng_a = np.random.default_rng(11)
    _ = rng_a.standard_normal(5)
    state_a = rng_a.bit_generator.state

    path = os.path.join(str(tmp_path), "rng.npz")
    save_checkpoint(path, params={"w": np.zeros(1, np.float32)},
                    step=5, seed=11, rngs={"roam": rng_a})

    ck = load_checkpoint(path)
    rng_b = np.random.default_rng(11)
    _ = rng_b.standard_normal(5)
    rng_b.bit_generator.state = ck["rngs"]["roam"]
    # 两侧继续生成的序列一致
    assert np.array_equal(rng_a.standard_normal(7), rng_b.standard_normal(7))


def test_reject_foreign_npz(tmp_path):
    """非 SGN checkpoint 文件报错。"""
    path = os.path.join(str(tmp_path), "foreign.npz")
    np.savez(path, a=np.array([1.0]))
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_dotted_param_names_roundtrip(tmp_path):
    """参数名含点号（Sequential 子模块路径如 0.weight）时 checkpoint 往返完整。

    回归：load_checkpoint 曾用 k.split(".") 解包 opt.state.<name>.<field>，
    点号参数名会 split 出 5 段导致 ValueError（修复为 rsplit 只切最后一段）。
    """
    rng = np.random.default_rng(9)
    p = {"0.weight": rng.standard_normal((6, 4)).astype(np.float32),
         "0.bias": rng.standard_normal((4,)).astype(np.float32),
         "2.weight": rng.standard_normal((4, 6)).astype(np.float32),
         "2.bias": rng.standard_normal((4,)).astype(np.float32)}
    opt = SGD(p, lr=0.02, momentum=0.9, classical=True)
    opt.step({k: np.ones_like(v) for k, v in p.items()})

    path = os.path.join(str(tmp_path), "ck_dot.npz")
    save_checkpoint(path, params=p, optimizer=opt, step=10)

    ck = load_checkpoint(path)
    assert set(ck["params"]) == set(p)
    sd = ck["optimizer"]
    assert set(sd["state"]) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    for name in p:
        assert np.array_equal(sd["state"][name]["v"], opt._state[name]["v"])


def test_dotted_resume_equals_continuous(tmp_path):
    """点号参数名下：保存→恢复→续训 与 不间断训练逐位一致。"""
    rng = np.random.default_rng(13)
    p0 = {"0.weight": rng.standard_normal((8,)).astype(np.float32)}
    grads = [rng.standard_normal((8,)).astype(np.float32) for _ in range(6)]

    # 不间断：6 步
    p_a = {"0.weight": p0["0.weight"].copy()}
    opt_a = SGD(p_a, lr=0.1, momentum=0.9)
    for g in grads:
        opt_a.step({"0.weight": g})

    # 前 3 步 → save → 恢复 → 后 3 步
    p_b = {"0.weight": p0["0.weight"].copy()}
    opt_b = SGD(p_b, lr=0.1, momentum=0.9)
    for g in grads[:3]:
        opt_b.step({"0.weight": g})
    path = os.path.join(str(tmp_path), "resume_dot.npz")
    save_checkpoint(path, params=p_b, optimizer=opt_b, step=3)

    ck = load_checkpoint(path)
    p_c = {k: v.copy() for k, v in ck["params"].items()}
    opt_c = SGD(p_c, lr=0.1, momentum=0.9)
    opt_c.load_state_dict(ck["optimizer"])
    for g in grads[3:]:
        opt_c.step({"0.weight": g})

    assert np.array_equal(p_a["0.weight"], p_c["0.weight"])
