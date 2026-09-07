# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""validate_dispatch_guardrail.py - 内核通道验收：档位守门断言 + 三后端矩阵

背景：内部调研文档（conv2d backward im2col/col2im/transpose 优化调研） §13
通道验收轮（2026-08-19）：
  - db 端口收口 → conv2d 族 6/6 端口全实现
  - SSE2 L1 兼容后端注册（纯标量实现 → kBitExact，复用 ref_scalar 内核，
    验证"新增后端 = 注册，算子层零改动"承诺）
  - 档位守门：声明档位 == 实测档位
      kBitExact → 与 ref_scalar 锚点逐位全等（array_equal）
      kRounding → 记录实测 max_abs_diff（精度护照），断言在容差内

档位守门的动机（调研文档 §13）：防"静默精度塌陷"——某后端悄悄换实现、精度掉了
但没人察觉（对应华为诺亚方舟共享指数 SNR 过低 / 当前 HC8 退化为 v[0]-only 的失败
模式）。任一后端声明 kBitExact 而实测非逐位全等 → 本脚本测试失败。

运行（在 engine/sgn 下）：
  python autograd/validate_dispatch_guardrail.py
三后端 × test_phase5 正确性矩阵（另行执行）：
  SGN_KERNEL_BACKEND=scalar python autograd/test_phase5.py
  SGN_KERNEL_BACKEND=sse2   python autograd/test_phase5.py
  (default / x86_avx2)      python autograd/test_phase5.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# build 目录（sgn.cp314-win_amd64.pyd 所在位置）
_BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")

# kRounding 档位容差（绝对值；输出量级 O(0.1~10)，1e-3 对应相对 ~1e-4）
_ROUNDING_TOL = 1e-3


# ============================================================================
# 子进程模式：跑固定工作负载并保存结果（不同后端需独立进程——注册表在
# 进程生命周期内只解析一次，env 在首次调用前读取）
# ============================================================================
def _make_tensor(np_arr, requires_grad=False):
    import sgn
    t = sgn.autograd.Tensor.from_numpy(np.ascontiguousarray(np_arr, dtype=np.float32))
    t.requires_grad = requires_grad
    return t


def _run_workload() -> dict:
    """固定工作负载：conv2d fwd+bwd + matmul fwd+bwd（确定性生成输入，跨进程可比）。"""
    import sgn
    ag = sgn.autograd
    rng = np.random.default_rng(1234)

    # --- conv2d：B=4, 3->8, 16x16, s=1 p=1 ---
    X = rng.standard_normal((4, 3, 16, 16)).astype(np.float32) * 0.1
    W = rng.standard_normal((8, 3, 3, 3)).astype(np.float32) * 0.1
    b = rng.standard_normal((8,)).astype(np.float32) * 0.01
    dY = rng.standard_normal((4, 8, 16, 16)).astype(np.float32)
    X_t = _make_tensor(X, True)
    W_t = _make_tensor(W, True)
    b_t = _make_tensor(b, True)
    ag.start_recording()
    y = ag.conv2d(X_t, W_t, b_t, 1, 1)
    ag.stop_recording()
    y.backward(dY)

    # --- matmul：A(4,8) @ B(8,5) ---
    A = rng.standard_normal((4, 8)).astype(np.float32)
    B = rng.standard_normal((8, 5)).astype(np.float32)
    dC = rng.standard_normal((4, 5)).astype(np.float32)
    a_t = _make_tensor(A, True)
    bm_t = _make_tensor(B, True)
    ag.start_recording()
    c = ag.matmul(a_t, bm_t)
    ag.stop_recording()
    c.backward(dC)

    return {
        "backend": ag.kernel_backend(),
        "conv2d_y": y.to_numpy(),
        "conv2d_dX": X_t.grad,
        "conv2d_dW": W_t.grad,
        "conv2d_db": b_t.grad,
        "matmul_C": c.to_numpy(),
        "matmul_dA": a_t.grad,
        "matmul_dB": bm_t.grad,
    }


def _child(backend: str, out_npz: str) -> int:
    if backend != "default":
        os.environ["SGN_KERNEL_BACKEND"] = backend
    else:
        os.environ.pop("SGN_KERNEL_BACKEND", None)
    sys.path.insert(0, _BUILD)
    res = _run_workload()
    arrs = {k: v for k, v in res.items() if k != "backend"}
    np.savez(out_npz, **arrs)
    with open(out_npz + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(res["backend"], f)
    print(f"[child] backend={backend} -> {res['backend']}")
    return 0


# ============================================================================
# 驱动模式：三后端 × 固定工作负载 → 档位守门断言 + 精度护照
# ============================================================================
_BACKENDS = [
    ("scalar", {"matmul": "ref_scalar", "conv2d": "ref_scalar", "num_level": "bit_exact"}),
    ("sse2", {"matmul": "sse2", "conv2d": "sse2", "num_level": "bit_exact"}),
    ("default", {"matmul": "avx2", "conv2d": "avx2", "num_level": "rounding"}),
]


def _driver() -> int:
    collected = {}
    with tempfile.TemporaryDirectory(prefix="guardrail_") as td:
        for tag, _expect in _BACKENDS:
            out = os.path.join(td, tag + ".npz")
            env = dict(os.environ)
            env.pop("SGN_KERNEL_BACKEND", None)
            if tag != "default":
                env["SGN_KERNEL_BACKEND"] = tag
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--child", "--backend", tag, "--out", out],
                env=env, check=True)
            with np.load(out) as z:
                arrays = {k: z[k] for k in z.files}
            with open(out + ".meta.json", encoding="utf-8") as f:
                meta = json.load(f)
            collected[tag] = {"arrays": arrays, "meta": meta}

    anchor = collected["scalar"]["arrays"]
    failures = []

    # 断言 0：后端名与档位声明核对
    print("=" * 72)
    print("后端声明核对")
    print("=" * 72)
    for tag, expect in _BACKENDS:
        m = collected[tag]["meta"]
        ok_name = (expect["matmul"] in m["matmul"]["name"]) and (expect["conv2d"] in m["conv2d"]["name"])
        ok_level = (m["matmul"]["num_level"] == expect["num_level"]
                    and m["conv2d"]["num_level"] == expect["num_level"])
        status = "OK" if (ok_name and ok_level) else "FAIL"
        print(f"  [{status}] {tag:8s} matmul={m['matmul']}  conv2d={m['conv2d']}")
        if not (ok_name and ok_level):
            failures.append(f"backend name/level mismatch for {tag}")

    # 断言 1：kBitExact 后端（scalar 锚点 / sse2）——两两逐位全等
    print("=" * 72)
    print("档位守门 · kBitExact → 逐位全等（array_equal）")
    print("=" * 72)
    for key in sorted(anchor):
        bit_ok = np.array_equal(collected["sse2"]["arrays"][key], anchor[key])
        status = "OK" if bit_ok else "FAIL"
        print(f"  [{status}] sse2 vs scalar: {key}")
        if not bit_ok:
            failures.append(f"kBitExact violated: {key} differs between sse2 and scalar")

    # 断言 2：kRounding 后端（default/avx2）——精度护照（实测 max_abs_diff）+ 容差
    print("=" * 72)
    print("档位守门 · kRounding → 精度护照（max_abs_diff vs scalar 锚点）")
    print("=" * 72)
    print(f"  {'key':<12s} {'max_abs_diff':>14s}  {'<= tol(1e-3)':>12s}")
    for key in sorted(anchor):
        d = float(np.max(np.abs(collected["default"]["arrays"][key] - anchor[key])))
        status = "OK" if d <= _ROUNDING_TOL else "FAIL"
        print(f"  {key:<12s} {d:14.6e}  {status:>12s}")
        if d > _ROUNDING_TOL:
            failures.append(f"kRounding passport exceeded tol: {key} max_abs_diff={d:.3e}")

    print("=" * 72)
    if failures:
        print(f"[FAIL] 档位守门共 {len(failures)} 处不通过：")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[PASS] 档位守门全过：kBitExact 逐位全等、kRounding 容差内、后端名/档位声明一致。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="内核通道验收：档位守门断言 + 三后端矩阵")
    parser.add_argument("--child", action="store_true", help="子进程模式（由驱动自动调用）")
    parser.add_argument("--backend", choices=["scalar", "sse2", "default"], default="default")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if args.child:
        if not args.out:
            print("--child 需要 --out <npz 路径>")
            return 2
        return _child(args.backend, args.out)
    return _driver()


if __name__ == "__main__":
    sys.exit(main())
