#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""resnet8_cifar10_a1_longrun.py — ResNet-8 × CIFAR-10 全量长跑（A1 主线，单核后台）

背景（2026-09-08 用户指令"继续尝试"）
------------------------------------
R4（resnet8_a1_compare.py）是 10000 子集 × 600 步的短验证（无 test 评估）。
本脚本升级为**全量长跑**：50000 train / 10000 test，按 epoch 训练 + 逐 epoch
test 评估 + 逐 epoch 落盘（断点可判读），A1 主线（前向 Q8 STE + 反向 A1 SR）
与 FLOAT32 参照双臂。

模型/训练语义：**逐位复用** resnet8_a1_compare（import 复用 init_resnet8 /
resnet8_forward / make_bn_stats / quant8——零复制，改动零漂移）。

口径声明（预注册）
------------------
  - 单核：OMP_NUM_THREADS=1（import sgn 前设置；用户指定单核后台长跑）
  - 训练：SGD lr=0.01 无动量（与 R4 一致），B=128，epochs 默认 20
  - test 评估口径 = **训练式前向 @B=256**（batch 统计，与训练步 acc 同口径
    可比；engine batchnorm2d 会原地更新 running stats，故 eval 使用
    running-stats 深拷贝副本、eval 后还原——不污染训练流）
  - 判据（预注册，勿放松）：臂间 gap = |acc_float32 − acc_A1| ≤ 2.0 点
    （R4 / Stage 2.5 gap 口径延续）；负结果同等落档
  - 数据：torchvision CIFAR10(download=False)，本地 data/cifar-10-batches-py

运行
----
  单核后台（脱离会话，Start-Process 落盘日志，见 logs/resnet8_cifar10_a1_longrun/）
  产物：results.json（逐 epoch 曲线）+ latest_params_{arm}.npz（断点参数）
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # 单核（用户指定）——必须在 import sgn 前
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import time
import copy
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / ".." / "build"))   # pyd 直连（与 a1_compare 同款）
sys.path.insert(0, str(_HERE))

import sgn  # noqa: E402

ag = sgn.autograd
import resnet8_a1_compare as base  # noqa: E402  模型/训练语义单一来源

DATA_ROOT = base.DATA_ROOT
OUT_DIR = Path(__file__).resolve().parents[3] / "logs" / "resnet8_cifar10_a1_longrun"


def load_cifar10_split(train: bool):
    """torchvision CIFAR10 → (N,3,32,32) 归一化 float32 + int 标签。"""
    import torchvision
    ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=train, download=False)
    X = ds.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
    X = (X - mean) / std
    return X, np.array(ds.targets)


def to_onehot(y, K=10):
    oh = np.zeros((y.shape[0], K), np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def _make_tensor(a, rg=False):
    t = ag.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    t.requires_grad = rg
    return t


def eval_test(Xte, yte, p, quant_fwd, bn_stats, B=256):
    """训练式前向 @B=256（预注册口径）；eval 用 running-stats 副本对象，不污染训练流。"""
    bn_backup = {k: (copy.deepcopy(v[0].to_numpy()), copy.deepcopy(v[1].to_numpy()))
                 for k, v in bn_stats.items()}
    correct = 0
    n = Xte.shape[0]
    for s in range(0, n, B):
        xb = Xte[s:s + B]
        yb = yte[s:s + B]
        ag.clear()
        ag.start_recording()
        logits, _ = base.resnet8_forward(xb, p, quant_fwd, bn_stats)
        ag.stop_recording()
        pred = np.argmax(logits.to_numpy(), 1)
        correct += int(np.sum(pred == yb))
    # 还原 running stats（重建 Tensor 对象替换 dict 值——eval 的 batch 统计不进训练流）
    for k, (rm, rv) in bn_backup.items():
        bn_stats[k] = (base.make_tensor(rm, False), base.make_tensor(rv, False))
    return correct / n


def train_epoch(st, X, y_oh, p, bn_stats, B, lr, rng):
    """一个 epoch：固定顺序 shuffle 遍历（rng 确定性）。返回 (mean_loss, mean_acc)。"""
    quant_fwd = (st == "A1")
    ag.set_backward_strategy(getattr(ag.BackwardStrategy, st))
    N = X.shape[0]
    idx = rng.permutation(N)
    losses, accs = [], []
    for s in range(0, N, B):
        bi = idx[s:s + B]
        xb, yb = X[bi], y_oh[bi]
        ag.clear()
        ag.start_recording()
        logits, params = base.resnet8_forward(xb, p, quant_fwd, bn_stats)
        ag.stop_recording()
        logits_np = logits.to_numpy()
        grad = 2.0 * (logits_np - yb) / len(bi)
        logits.backward(grad)
        for key, t_ in params.items():
            p[key] = p[key] - lr * t_.grad
        losses.append(float(np.mean((logits_np - yb) ** 2)))
        accs.append(float(np.mean(np.argmax(logits_np, 1) == np.argmax(yb, 1))))
    return float(np.mean(losses)), float(np.mean(accs))


def run_arm(st, Xtr, ytr_oh, ytr, Xte, yte, epochs, B, lr, seed, out_dir):
    rng = np.random.default_rng(seed + 1)
    p = base.init_resnet8(seed)
    bn_stats = base.make_bn_stats(p)
    quant_fwd = (st == "A1")
    ag.set_backward_strategy(getattr(ag.BackwardStrategy, st))
    ag.set_sr_seed(1234 + seed)

    hist = {"arm": st, "epochs": [], "train_loss": [], "train_acc": [],
            "test_acc": [], "sec": []}
    t_arm0 = time.perf_counter()
    for ep in range(epochs):
        t0 = time.perf_counter()
        tl, ta = train_epoch(st, Xtr, ytr_oh, p, bn_stats, B, lr, rng)
        acc_te = eval_test(Xte, yte, p, quant_fwd, bn_stats)
        dt = time.perf_counter() - t0
        hist["epochs"].append(ep + 1)
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        hist["test_acc"].append(acc_te)
        hist["sec"].append(round(dt, 1))
        # 逐 epoch 落盘（断点可判读；参数 npz 覆盖最新）
        (out_dir / f"results_{st}.json").write_text(
            json.dumps(hist, ensure_ascii=False, indent=1), "utf-8")
        np.savez_compressed(out_dir / f"latest_params_{st}.npz",
                            **{k: v for k, v in p.items()})
        print(f"  [{st}] epoch {ep+1}/{epochs}  loss={tl:.4e}  train_acc={ta:.3f}  "
              f"test_acc={acc_te:.4f}  ({dt:.0f}s)", flush=True)
    hist["total_sec"] = round(time.perf_counter() - t_arm0, 1)
    (out_dir / f"results_{st}.json").write_text(
        json.dumps(hist, ensure_ascii=False, indent=1), "utf-8")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-subset", type=int, default=None,
                    help="smoke 用（如 2000）；默认全量 50000")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("#" * 76, flush=True)
    print(f"# ResNet-8 × CIFAR-10 全量长跑（A1 主线，单核 OMP=1）", flush=True)
    print(f"# epochs={a.epochs} B={a.batch} lr={a.lr} seed={a.seed} "
          f"train_subset={a.train_subset or '50000(全量)'}", flush=True)
    print("#" * 76, flush=True)

    Xtr, ytr = load_cifar10_split(True)
    Xte, yte = load_cifar10_split(False)
    if a.train_subset:
        Xtr, ytr = Xtr[:a.train_subset], ytr[:a.train_subset]
    ytr_oh = to_onehot(ytr)
    print(f"  train={Xtr.shape}  test={Xte.shape}", flush=True)

    t_all = time.perf_counter()
    hists = {}
    for st in ["FLOAT32", "A1"]:
        print(f"\n== 臂：{st} ==", flush=True)
        hists[st] = run_arm(st, Xtr, ytr_oh, ytr, Xte, yte,
                            a.epochs, a.batch, a.lr, a.seed, OUT_DIR)

    # 判读（预注册：gap ≤ 2.0 点）
    f32_final = float(np.mean(hists["FLOAT32"]["test_acc"][-3:]))
    a1_final = float(np.mean(hists["A1"]["test_acc"][-3:]))
    gap = abs(f32_final - a1_final)
    verdict = "GO" if gap <= 2.0 else "NO-GO"
    summary = {"f32_final3": f32_final, "a1_final3": a1_final, "gap_points": gap,
               "criterion": "gap <= 2.0 points", "verdict": verdict,
               "total_sec": round(time.perf_counter() - t_all, 1),
               "config": {"epochs": a.epochs, "batch": a.batch, "lr": a.lr,
                          "seed": a.seed, "train_subset": a.train_subset,
                          "omp_threads": 1}}
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print("\n" + "=" * 76, flush=True)
    print(f"  FLOAT32 final3 test acc = {f32_final:.4f}", flush=True)
    print(f"  A1      final3 test acc = {a1_final:.4f}", flush=True)
    print(f"  gap = {gap:.2f} 点（判据 ≤ 2.0）→ verdict: {verdict}", flush=True)
    print(f"  总耗时 {summary['total_sec']:.0f}s → {OUT_DIR}", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main()
