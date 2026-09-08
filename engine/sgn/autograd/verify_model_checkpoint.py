# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""verify_model_checkpoint.py - SGN 自训模型 npz 加载与推理验证（自包含）

用途：验证发布的 checkpoint（benchmark_mnist8_production_sgn_model.npz）
    1) 能被 numpy 加载（P7 教训：快照同步曾以文本模式损坏二进制——本脚本
       是 npz 完好性的终审门）；
    2) 权重可重建 ResNet-8 并支撑折叠 BN + STE 推理（权重文件的真实使用）；
    3) 子集测试精度与 meta_json 记录值一致（±1.5 点，子集采样波动容差）。

运行（手动或计划任务；结果落盘 JSON，自包含无人工介入）：
    python verify_model_checkpoint.py                # 默认主仓原件，2000 样本
    python verify_model_checkpoint.py --model <npz路径> --samples 10000
输出：verify_model_checkpoint_results.json（同目录）+ stdout 摘要
退出码：0 = 全部 PASS；1 = 任一 FAIL

注意：sgn 必须经 build 目录 raw pyd 直载（`import sgn`）——若经
`engine.sgn` 包导入，`engine/sgn/autograd/` 源码目录会以 namespace package
遮蔽 pyd 的 autograd 属性，set_backward_strategy 等函数将丢失。
"""

import os
import sys
import json
import time
import argparse

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
NPZ_CACHE = os.path.join(DATA_ROOT, "mnist8_bench_cache.npz")
MODEL_PATH = os.path.join(HERE, "..", "tests", "refs",
                          "benchmark_mnist8_production_sgn_model.npz")
RESULT_PATH = os.path.join(HERE, "verify_model_checkpoint_results.json")
MEAN, STD = 0.1307, 0.3081
BN_NAMES = ("c1", "s1a", "s1b", "s2a", "s2b", "s2sc", "s3a", "s3b", "s3sc")
BN_CHANNELS = {"c1": 16, "s1a": 16, "s1b": 16, "s2a": 32, "s2b": 32,
               "s2sc": 32, "s3a": 64, "s3b": 64, "s3sc": 64}
CONV_KEY = {"c1": ("c1w", "c1b"), "s1a": ("s1a_w", "s1a_b"),
            "s1b": ("s1b_w", "s1b_b"), "s2a": ("s2a_w", "s2a_b"),
            "s2b": ("s2b_w", "s2b_b"), "s2sc": ("s2sc_w", "s2sc_b"),
            "s3a": ("s3a_w", "s3a_b"), "s3b": ("s3b_w", "s3b_b"),
            "s3sc": ("s3sc_w", "s3sc_b")}


def load_test_arrays(samples):
    """自包含测试集加载（uint8 npz 缓存 → 归一化 float32；与基准同口径）。"""
    z = np.load(NPZ_CACHE)
    xte = ((z["x_test"].astype(np.float32) / 255.0 - np.float32(MEAN))
           / np.float32(STD))[:, None, :, :]
    yte = z["y_test"].astype(np.int64)
    if samples is not None and samples < xte.shape[0]:
        xte, yte = xte[:samples], yte[:samples]
    return np.ascontiguousarray(xte, dtype=np.float32), yte


def main(samples, model_path=None):
    model_path = model_path or MODEL_PATH
    t0 = time.perf_counter()
    checks = []

    def add(name, ok, value=""):
        checks.append({"name": name, "pass": bool(ok), "value": value})
        return bool(ok)

    # ---- 1) npz 加载（P7 终审门：BOM/乱码/central directory 损坏在此暴露）----
    try:
        z = np.load(model_path)
        meta = json.loads(z["meta_json"].tobytes().decode("utf-8"))
        add("npz_load", True, f"{len(z.keys())} keys")
    except Exception as e:
        add("npz_load", False, repr(e)[:200])
        _finish(checks, t0, model_path)
        return 1
    if not add("meta_json_parse", isinstance(meta, dict) and "test_acc_fold" in meta,
               meta.get("test_acc_fold")):
        _finish(checks, t0, model_path)
        return 1

    # 必需键齐全性
    required = (["c1w", "c1b", "s1a_w", "s1a_b", "s1b_w", "s1b_b",
                 "s2a_w", "s2a_b", "s2b_w", "s2b_b", "s2sc_w", "s2sc_b",
                 "s3a_w", "s3a_b", "s3b_w", "s3b_b", "s3sc_w", "s3sc_b",
                 "fcw", "fcb"]
                + [f"bn_{n}_g" for n in BN_NAMES]
                + [f"bn_{n}_b" for n in BN_NAMES]
                + [f"bn_{n}_rm" for n in BN_NAMES]
                + [f"bn_{n}_rv" for n in BN_NAMES])
    missing = [k for k in required if k not in z.keys()]
    add("required_keys", not missing,
        f"missing={missing}" if missing else "all present")

    # ---- 2) sgn raw pyd 直载（build 目录优先，防 namespace package 遮蔽）----
    sys.path.insert(0, os.path.join(HERE, "..", "build"))
    import sgn
    ag = sgn.autograd
    ag.set_backward_strategy(ag.BackwardStrategy.A1)   # STE 前向（部署语义）

    if not os.path.exists(NPZ_CACHE):
        import torchvision
        te = torchvision.datasets.MNIST(root=DATA_ROOT, train=False,
                                        download=False)
        np.savez_compressed(NPZ_CACHE, x_test=te.data.numpy().astype(np.uint8),
                            y_test=te.targets.numpy())
    Xte, yte = load_test_arrays(samples)

    # ---- 3) 折叠 BN + quant8 权重 + STE 推理（权重文件的真实使用）----
    p = {k: np.ascontiguousarray(z[k]) for k in required
         if not k.startswith("bn_") or k.endswith("_g") or k.endswith("_b")}
    bn_stats = {n: (z[f"bn_{n}_rm"], z[f"bn_{n}_rv"]) for n in BN_NAMES}
    fp = dict(p)
    for name in BN_NAMES:
        wname, bname = CONV_KEY[name]
        rm, rv = bn_stats[name]
        s = p[f"bn_{name}_g"] / np.sqrt(rv + np.float32(1e-5))
        fp[wname] = (p[wname] * s[:, None, None, None]).astype(np.float32)
        fp[bname] = ((p[bname] - rm) * s + p[f"bn_{name}_b"]).astype(np.float32)

    def quant8(x):
        amax = float(np.max(np.abs(x)))
        if amax == 0:
            return x * 0.0
        scale = amax / 127.0
        return np.clip(np.round(x / scale), -127.0, 127.0) * scale

    def folded_logits(x_np):
        def conv(x, wname, bname, s, pad):
            xt = x if isinstance(x, ag.Tensor) else \
                ag.Tensor.from_numpy(np.ascontiguousarray(x))
            return ag.conv2d(
                xt,
                ag.Tensor.from_numpy(np.ascontiguousarray(quant8(fp[wname]))),
                ag.Tensor.from_numpy(fp[bname]), s, pad)

        c1 = ag.relu(conv(x_np, "c1w", "c1b", 1, 1))
        s1 = ag.relu(conv(c1, "s1a_w", "s1a_b", 1, 1))
        s1 = conv(s1, "s1b_w", "s1b_b", 1, 1)
        h = ag.relu(ag.add(s1, c1))
        s2 = ag.relu(conv(h, "s2a_w", "s2a_b", 2, 1))
        s2 = conv(s2, "s2b_w", "s2b_b", 1, 1)
        sc2 = conv(h, "s2sc_w", "s2sc_b", 2, 0)
        h = ag.relu(ag.add(s2, sc2))
        s3 = ag.relu(conv(h, "s3a_w", "s3a_b", 2, 1))
        s3 = conv(s3, "s3b_w", "s3b_b", 1, 1)
        sc3 = conv(h, "s3sc_w", "s3sc_b", 2, 0)
        h = ag.relu(ag.add(s3, sc3))
        g = ag.avgpool2d(h, 7, 7)
        return ag.linear(ag.reshape(g, [x_np.shape[0], -1]),
                         ag.Tensor.from_numpy(np.ascontiguousarray(
                             quant8(fp["fcw"]))),
                         ag.Tensor.from_numpy(fp["fcb"])).to_numpy()

    B = 250
    correct, t1 = 0, time.perf_counter()
    for i in range(0, Xte.shape[0], B):
        out = folded_logits(Xte[i:i + B])
        correct += int((np.argmax(out, 1) == yte[i:i + B]).sum())
    acc = correct / Xte.shape[0]
    infer_sec = time.perf_counter() - t1
    expect = float(meta["test_acc_fold"])
    add("inference_acc", abs(acc - expect) <= 0.015,
        f"acc={acc:.4f} expect={expect:.4f} (|d|<=0.015, N={Xte.shape[0]})")

    _finish(checks, t0, {"test_acc": acc, "expect": expect,
                         "infer_sec": round(infer_sec, 2),
                         "n_samples": int(Xte.shape[0]),
                         "model_path": model_path, "meta": meta})
    return 0 if all(c["pass"] for c in checks) else 1


def _finish(checks, t0, model_path, extra=None):
    result = {"checks": checks,
              "all_pass": all(c["pass"] for c in checks),
              "elapsed_sec": round(time.perf_counter() - t0, 2),
              "model_path": model_path,
              "extra": extra or {},
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    for c in checks:
        print(f"{'PASS' if c['pass'] else 'FAIL'} {c['name']}: {c['value']}",
              flush=True)
    print(f"verdict: {'GO' if result['all_pass'] else 'FAIL'} "
          f"({result['elapsed_sec']}s) -> {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--model", default=MODEL_PATH,
                    help="npz 路径（默认主仓原件；可指 GitHub 下载副本）")
    args = ap.parse_args()
    sys.exit(main(args.samples, args.model))
