# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""benchmark_mnist8_production.py - SGN-A1 vs PyTorch float32 生产级速度基准

口径（用户拍板 2026-09-07，双臂）：
    负载  = 训练（端到端 epoch：批切取 + 张量拷贝 + 前向 + loss + 反向 + optimizer
            step）+ 推理（测试集全量前向），两负载都测
    模型  = ResNet-8 × MNIST 全量（60k train / 10k test；conv1 + 3 stage×2conv
            + 1×1 shortcut + GAP + fc，全 BN，~78k 参数，与 benchmark_mnist8.py
            同一架构）
    线程  = 4 核对齐（OMP_NUM_THREADS=4 + torch.set_num_threads(4)）
    分臂  = SGN-A1（前向 8bit STE + 反向 SR + numpy 侧 quant8 权重，A1 主线全语义）
            vs PyTorch float32（同 init / 同 batch 序 / 同超参 / 同 CE-mean）
    方法学 = R 次独立 run × E epoch，丢弃 epoch1（warmup），两级平均（run 内
            batch 均值 → run 间均值）；推理每 run 多 pass 丢首轮 warmup

推理形态裁决：SGN 侧 = BN 折叠（conv 权重吸收 eval BN）+ quant8 折叠权重 + STE
    前向（部署形态）；正确性由双检查门兜底——门 F1（torch 侧折叠公式 vs
    eval()，Δlogit ≤ 1e-3）+ 门 F2（SGN 折叠 acc vs 分离式 eval-BN 检查路径
    acc 差 ≤ 0.5 点，分离式路径不计入计时）。

进程隔离：--arm both 编排态按序起两个子进程（各自只 import 自己的框架），
    全程无 libomp/libiomp5 冲突，无需 KMP_DUPLICATE_LIB_OK 兜底（头部仍
    setdefault 一行作惯性防御）。计时 report-only，verdict 只 gate checks。

运行：
    python benchmark_mnist8_production.py --arm both --smoke   # 管线验证 ~2 分钟
    python benchmark_mnist8_production.py --arm both           # 全量（脱离会话跑）
输出：benchmark_mnist8_production_results.json（config/rows/checks/verdict 四键）
      + benchmark_mnist8_production_{sgn,torch}.json（臂部分结果）
"""

import os
import sys
import json
import time
import argparse
import subprocess

import numpy as np  # noqa: E402（arm 中立层：数据管道与统计均用）

# 线程口径先于任何框架导入设置（子进程由编排态显式注入同值）
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
NPZ_PATH = os.path.join(DATA_ROOT, "mnist8_bench_cache.npz")
MEAN, STD = 0.1307, 0.3081          # MNIST 归一化（双臂一致）
BN_NAMES = ("c1", "s1a", "s1b", "s2a", "s2b", "s2sc", "s3a", "s3b", "s3sc")
BN_CHANNELS = {"c1": 16, "s1a": 16, "s1b": 16, "s2a": 32, "s2b": 32,
               "s2sc": 32, "s3a": 64, "s3b": 64, "s3sc": 64}
CONV_KEY = {"c1": ("c1w", "c1b"), "s1a": ("s1a_w", "s1a_b"),
            "s1b": ("s1b_w", "s1b_b"), "s2a": ("s2a_w", "s2a_b"),
            "s2b": ("s2b_w", "s2b_b"), "s2sc": ("s2sc_w", "s2sc_b"),
            "s3a": ("s3a_w", "s3a_b"), "s3b": ("s3b_w", "s3b_b"),
            "s3sc": ("s3sc_w", "s3sc_b")}


def build_config(tier, smoke):
    if smoke:
        return {"R": 1, "E": 1, "B": 16, "B_infer": 250, "lr": 0.02,
                "train_subset": 512, "test_subset": 512, "infer_passes": 2,
                "acc_gate": None, "momentum": 0.9, "weight_decay": 0.0}
    if tier == "single":
        return {"R": 1, "E": 3, "B": 64, "B_infer": 250, "lr": 0.02,
                "train_subset": None, "test_subset": None, "infer_passes": 2,
                "acc_gate": 0.97, "momentum": 0.9, "weight_decay": 0.0}
    if tier == "aggressive":
        r = 5
    else:
        r = 3
    return {"R": r, "E": 3, "B": 64, "B_infer": 250, "lr": 0.02,
            "train_subset": None, "test_subset": None, "infer_passes": 4,
            "acc_gate": 0.97, "momentum": 0.9, "weight_decay": 0.0}


# ============================================================================
# 数据层：npz 缓存（uint8 像素 + int 标签；归一化在臂内加载时做，不计入计时）
# ============================================================================
def prepare_npz():
    """编排态一次性把 data/MNIST/raw 转 uint8 npz（双臂 numpy-only 读）。"""
    if os.path.exists(NPZ_PATH):
        return NPZ_PATH
    import torchvision
    tr = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=False)
    te = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False)
    np.savez_compressed(
        NPZ_PATH,
        x_train=tr.data.numpy().astype(np.uint8), y_train=tr.targets.numpy(),
        x_test=te.data.numpy().astype(np.uint8), y_test=te.targets.numpy())
    return NPZ_PATH


def load_arrays(subset_train=None, subset_test=None, seed=42):
    z = np.load(NPZ_PATH)
    mean = np.float32(MEAN)
    std = np.float32(STD)
    xtr = ((z["x_train"].astype(np.float32) / 255.0 - mean) / std)[:, None, :, :]
    xte = ((z["x_test"].astype(np.float32) / 255.0 - mean) / std)[:, None, :, :]
    ytr, yte = z["y_train"].astype(np.int64), z["y_test"].astype(np.int64)
    if subset_train is not None and subset_train < xtr.shape[0]:
        idx = np.random.default_rng(seed).permutation(xtr.shape[0])[:subset_train]
        xtr, ytr = xtr[idx], ytr[idx]
    if subset_test is not None and subset_test < xte.shape[0]:
        xte, yte = xte[:subset_test], yte[:subset_test]
    return np.ascontiguousarray(xtr, dtype=np.float32), ytr, \
        np.ascontiguousarray(xte, dtype=np.float32), yte


def epoch_batches(n, B, run_id):
    """permutation 批序（双臂同 run 同序；drop_last）。返回 list[np.ndarray]。"""
    perm = np.random.default_rng(1000 + run_id).permutation(n)
    nb = n // B
    return [perm[i * B:(i + 1) * B] for i in range(nb)]


# ============================================================================
# SGN 臂
# ============================================================================
def run_sgn_arm(cfg, out_path, export_model_path=None):
    import numpy as np
    import engine.sgn as sgn
    ag = sgn.autograd
    _ = export_model_path  # 签名占位（导出逻辑见函数尾 checkpoint 块）

    def make_tensor(a, rg=False):
        t = ag.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        t.requires_grad = rg
        return t

    def quant8(x):
        """HC8 对称 8bit 量化（A1 主线权重形态，照搬 resnet8_a1_compare）。"""
        amax = float(np.max(np.abs(x)))
        if amax == 0:
            return x * 0.0
        scale = amax / 127.0
        return np.clip(np.round(x / scale), -127.0, 127.0) * scale

    def init_resnet8_mnist(seed=7):
        rng = np.random.default_rng(seed)

        def kaiming(shape, fan_in):
            b = np.sqrt(6.0 / fan_in) * np.sqrt(2.0)
            return rng.uniform(-b, b, shape).astype(np.float32)

        p = {}
        p["c1w"] = kaiming((16, 1, 3, 3), 1 * 9)
        p["c1b"] = np.zeros(16, np.float32)
        p["s1a_w"] = kaiming((16, 16, 3, 3), 16 * 9)
        p["s1a_b"] = np.zeros(16, np.float32)
        p["s1b_w"] = kaiming((16, 16, 3, 3), 16 * 9)
        p["s1b_b"] = np.zeros(16, np.float32)
        p["s2a_w"] = kaiming((32, 16, 3, 3), 16 * 9)
        p["s2a_b"] = np.zeros(32, np.float32)
        p["s2b_w"] = kaiming((32, 32, 3, 3), 32 * 9)
        p["s2b_b"] = np.zeros(32, np.float32)
        p["s2sc_w"] = kaiming((32, 16, 1, 1), 16)
        p["s2sc_b"] = np.zeros(32, np.float32)
        p["s3a_w"] = kaiming((64, 32, 3, 3), 32 * 9)
        p["s3a_b"] = np.zeros(64, np.float32)
        p["s3b_w"] = kaiming((64, 64, 3, 3), 64 * 9)
        p["s3b_b"] = np.zeros(64, np.float32)
        p["s3sc_w"] = kaiming((64, 32, 1, 1), 32)
        p["s3sc_b"] = np.zeros(64, np.float32)
        p["fcw"] = kaiming((10, 64), 64)
        p["fcb"] = np.zeros(10, np.float32)
        for name, c in BN_CHANNELS.items():
            p[f"bn_{name}_g"] = np.ones(c, np.float32)
            p[f"bn_{name}_b"] = np.zeros(c, np.float32)
        return p

    def make_bn_stats():
        stats = {}
        for name in BN_NAMES:
            c = BN_CHANNELS[name]
            stats[name] = (make_tensor(np.zeros(c, np.float32)),
                           make_tensor(np.ones(c, np.float32)))
        return stats

    def resnet8_forward(x_np, p, quant_fwd, bn_stats, record):
        """A1 训练前向（record=True 走 tape；推理变体见独立函数）。"""
        params = {}

        def conv(x, wname, bname, s, pad):
            W = quant8(p[wname]) if quant_fwd else p[wname]
            wt, bt = make_tensor(W, True), make_tensor(p[bname], True)
            params[wname], params[bname] = wt, bt
            return ag.conv2d(x, wt, bt, s, pad)

        def bn(x, name):
            gt = make_tensor(p[f"bn_{name}_g"], True)
            bt = make_tensor(p[f"bn_{name}_b"], True)
            params[f"bn_{name}_g"], params[f"bn_{name}_b"] = gt, bt
            rm_t, rv_t = bn_stats[name]
            return ag.batchnorm2d(x, gt, bt, rm_t, rv_t, 0.1, 1e-5)

        B = x_np.shape[0]
        if record:
            ag.start_recording()
        try:
            c1 = ag.relu(bn(conv(make_tensor(x_np), "c1w", "c1b", 1, 1), "c1"))
            s1 = ag.relu(bn(conv(c1, "s1a_w", "s1a_b", 1, 1), "s1a"))
            s1 = bn(conv(s1, "s1b_w", "s1b_b", 1, 1), "s1b")
            h = ag.relu(ag.add(s1, c1))
            s2 = ag.relu(bn(conv(h, "s2a_w", "s2a_b", 2, 1), "s2a"))
            s2 = bn(conv(s2, "s2b_w", "s2b_b", 1, 1), "s2b")
            sc2 = bn(conv(h, "s2sc_w", "s2sc_b", 2, 0), "s2sc")
            h = ag.relu(ag.add(s2, sc2))
            s3 = ag.relu(bn(conv(h, "s3a_w", "s3a_b", 2, 1), "s3a"))
            s3 = bn(conv(s3, "s3b_w", "s3b_b", 1, 1), "s3b")
            sc3 = bn(conv(h, "s3sc_w", "s3sc_b", 2, 0), "s3sc")
            h = ag.relu(ag.add(s3, sc3))
            g = ag.avgpool2d(h, 7, 7)
            flat = ag.reshape(g, [B, -1])
            W = quant8(p["fcw"]) if quant_fwd else p["fcw"]
            wt, bt = make_tensor(W, True), make_tensor(p["fcb"], True)
            params["fcw"], params["fcb"] = wt, bt
            logits = ag.linear(flat, wt, bt)
        finally:
            if record:
                ag.stop_recording()
        return logits, params

    def fold_bn(p, bn_stats):
        """BN 折叠：W' = W·s, b' = (b−rm)·s + β，s = γ/√(rv+1e-5)。"""
        fp = dict(p)
        for name in BN_NAMES:
            wname, bname = CONV_KEY[name]
            rm = bn_stats[name][0].to_numpy()
            rv = bn_stats[name][1].to_numpy()
            s = p[f"bn_{name}_g"] / np.sqrt(rv + np.float32(1e-5))
            fp[wname] = (p[wname] * s[:, None, None, None]).astype(np.float32)
            fp[bname] = ((p[bname] - rm) * s + p[f"bn_{name}_b"]).astype(np.float32)
        return fp

    def infer_logits_folded(x_np, fp):
        """折叠 + quant8 权重 + STE 前向（部署形态；A1 策略下 conv 自动 STE）。"""
        B = x_np.shape[0]

        def conv(x, wname, bname, s, pad):
            return ag.conv2d(x, make_tensor(quant8(fp[wname])),
                             make_tensor(fp[bname]), s, pad)

        c1 = ag.relu(conv(make_tensor(x_np), "c1w", "c1b", 1, 1))
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
        logits = ag.linear(ag.reshape(g, [B, -1]),
                           make_tensor(quant8(fp["fcw"])), make_tensor(fp["fcb"]))
        return logits.to_numpy()

    def infer_logits_separated(x_np, p, bn_stats):
        """门 F2 检查专用（不计时）：STE conv ↔ numpy eval-BN 往返。"""
        def conv(x, wname, bname, s, pad):
            return ag.conv2d(x, make_tensor(quant8(p[wname])),
                             make_tensor(p[bname]), s, pad)

        def evbn(x_np2, name):
            rm = bn_stats[name][0].to_numpy()
            rv = bn_stats[name][1].to_numpy()
            s = p[f"bn_{name}_g"] / np.sqrt(rv + np.float32(1e-5))
            return ((x_np2 - rm[None, :, None, None]) * s[None, :, None, None]
                    + p[f"bn_{name}_b"][None, :, None, None]).astype(np.float32)

        c1 = evbn(conv(make_tensor(x_np), "c1w", "c1b", 1, 1).to_numpy(), "c1")
        c1 = np.maximum(c1, 0)
        s1 = evbn(conv(make_tensor(c1), "s1a_w", "s1a_b", 1, 1).to_numpy(), "s1a")
        s1 = np.maximum(s1, 0)
        s1 = evbn(conv(make_tensor(s1), "s1b_w", "s1b_b", 1, 1).to_numpy(), "s1b")
        h = np.maximum(s1 + c1, 0)
        s2 = evbn(conv(make_tensor(h), "s2a_w", "s2a_b", 2, 1).to_numpy(), "s2a")
        s2 = np.maximum(s2, 0)
        s2 = evbn(conv(make_tensor(s2), "s2b_w", "s2b_b", 1, 1).to_numpy(), "s2b")
        sc2 = evbn(conv(make_tensor(h), "s2sc_w", "s2sc_b", 2, 0).to_numpy(), "s2sc")
        h = np.maximum(s2 + sc2, 0)
        s3 = evbn(conv(make_tensor(h), "s3a_w", "s3a_b", 2, 1).to_numpy(), "s3a")
        s3 = np.maximum(s3, 0)
        s3 = evbn(conv(make_tensor(s3), "s3b_w", "s3b_b", 1, 1).to_numpy(), "s3b")
        sc3 = evbn(conv(make_tensor(h), "s3sc_w", "s3sc_b", 2, 0).to_numpy(), "s3sc")
        h = np.maximum(s3 + sc3, 0)
        g = h.mean(axis=(2, 3), keepdims=True)             # GAP（numpy 域等价）
        logits = g.reshape(h.shape[0], -1) @ p["fcw"].T + p["fcb"]
        return logits

    ag.set_backward_strategy(ag.BackwardStrategy.A1)
    ag.set_ste_quant_config(bits=8, clip_sigma=4.0)
    ag.set_quant_config(bits=16, clip_sigma=4.0)
    criterion = sgn.loss.CrossEntropyLoss()

    Xtr, ytr, Xte, yte = load_arrays(cfg["train_subset"], cfg["test_subset"])
    runs = []
    none_grad_count = 0
    for run_id in range(1, cfg["R"] + 1):
        p = init_resnet8_mnist(seed=7 + run_id)
        bn_stats = make_bn_stats()
        ag.set_sr_seed(1234 + run_id)
        opt = sgn.SGD(p, lr=cfg["lr"], momentum=cfg["momentum"],
                      weight_decay=cfg["weight_decay"])
        batches = epoch_batches(Xtr.shape[0], cfg["B"], run_id)
        epochs = []
        for ep in range(1, cfg["E"] + 1):
            loss_sum, correct, seen = 0.0, 0, 0
            t0 = time.perf_counter()
            for idx in batches:
                xb, yb = Xtr[idx], ytr[idx]
                logits, params = resnet8_forward(xb, p, True, bn_stats, True)
                logits_np = logits.to_numpy()
                loss, dY = criterion(logits_np, yb)
                logits.backward(dY.astype(np.float32))
                grads = {}
                for k, t_ in params.items():
                    if t_.grad is None:
                        none_grad_count += 1
                    else:
                        grads[k] = t_.grad
                opt.step(grads)
                loss_sum += float(loss) * len(idx)
                correct += int((np.argmax(logits_np, 1) == yb).sum())
                seen += len(idx)
            wall = time.perf_counter() - t0
            epochs.append({"epoch": ep, "train_loss": loss_sum / seen,
                           "train_acc": correct / seen, "batches": len(batches),
                           "epoch_sec": wall,
                           "ms_per_batch": wall / len(batches) * 1000.0})
            print(f"[sgn] run{run_id} epoch{ep}: loss={epochs[-1]['train_loss']:.4f} "
                  f"acc={epochs[-1]['train_acc']:.4f} "
                  f"{epochs[-1]['ms_per_batch']:.1f} ms/batch", flush=True)
        # 推理：折叠 + STE（计时 pass 丢首轮 warmup）
        fp = fold_bn(p, bn_stats)
        infer_batches = epoch_batches_pass(Xte.shape[0], cfg["B_infer"])
        pass_ms, acc_last = [], None
        for ps in range(cfg["infer_passes"]):
            correct, t0 = 0, time.perf_counter()
            for idx in infer_batches:
                out = infer_logits_folded(Xte[idx], fp)
                correct += int((np.argmax(out, 1) == yte[idx]).sum())
            pass_ms.append((time.perf_counter() - t0) / len(infer_batches) * 1000.0)
            acc_last = correct / (len(infer_batches) * cfg["B_infer"])
        acc_fold = acc_last
        # 门 F2：分离式检查路径（不计时）
        correct = 0
        for idx in infer_batches:
            out = infer_logits_separated(Xte[idx], p, bn_stats)
            correct += int((np.argmax(out, 1) == yte[idx]).sum())
        acc_sep = correct / (len(infer_batches) * cfg["B_infer"])
        counted = [e for e in epochs if e["epoch"] > 1] or epochs[:1]
        runs.append({
            "run_id": run_id,
            "epochs": epochs,
            "train_ms_per_batch_run_mean":
                float(np.mean([e["ms_per_batch"] for e in counted])),
            "epoch_sec_run_mean": float(np.mean([e["epoch_sec"] for e in counted])),
            "train_acc_last": epochs[-1]["train_acc"],
            "test_acc_fold": acc_fold, "test_acc_separated": acc_sep,
            "infer_ms_per_batch_run_mean": float(np.mean(pass_ms[1:]) or pass_ms[0]),
            "infer_pass_ms_raw": pass_ms,
        })
        _write_json(out_path, {"arm": "sgn", "runs": runs,
                               "env": _env_snapshot(None)})
        print(f"[sgn] run{run_id} done: test_acc_fold={acc_fold:.4f} "
              f"test_acc_sep={acc_sep:.4f}", flush=True)

    counted_tr = [r["train_ms_per_batch_run_mean"] for r in runs]
    counted_inf = [r["infer_ms_per_batch_run_mean"] for r in runs]
    summary = {
        "train_ms_per_batch": {"run_means": counted_tr,
                               "grand_mean": float(np.mean(counted_tr)),
                               "min": float(np.min(counted_tr)),
                               "max": float(np.max(counted_tr))},
        "infer_ms_per_batch": {"run_means": counted_inf,
                               "grand_mean": float(np.mean(counted_inf)),
                               "min": float(np.min(counted_inf)),
                               "max": float(np.max(counted_inf))},
        "epoch_sec_grand_mean":
            float(np.mean([r["epoch_sec_run_mean"] for r in runs])),
        "test_acc_fold_grand_mean":
            float(np.mean([r["test_acc_fold"] for r in runs])),
        "test_acc_separated_grand_mean":
            float(np.mean([r["test_acc_separated"] for r in runs])),
        "train_acc_grand_mean":
            float(np.mean([r["train_acc_last"] for r in runs])),
    }
    _write_json(out_path, {"arm": "sgn", "runs": runs, "summary": summary,
                           "env": _env_snapshot(None),
                           "none_grad_count": none_grad_count})
    # checkpoint 落盘（发布资产；MNIST 公开数据 + 自有代码，无权益障碍）
    if export_model_path:
        payload = dict(p)                       # 全部可训练权重（float32）
        for n in BN_NAMES:
            payload[f"bn_{n}_rm"] = bn_stats[n][0].to_numpy()
            payload[f"bn_{n}_rv"] = bn_stats[n][1].to_numpy()
        payload["meta_json"] = np.frombuffer(json.dumps({
            "model": "resnet8_mnist_a1(78k)",
            "strategy": "A1(ste8/sr16)", "epochs": cfg["E"], "B": cfg["B"],
            "lr": cfg["lr"], "momentum": cfg["momentum"],
            "seed_init": 7 + runs[-1]["run_id"],
            "test_acc_fold": runs[-1]["test_acc_fold"],
            "normalize": "(x/255-0.1307)/0.3081",
            "infer": "bn-folded quant8-w + STE",
        }, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(export_model_path, **payload)
        print(f"[sgn] checkpoint -> {export_model_path}", flush=True)


def epoch_batches_pass(n, B):
    perm = np.arange(n)
    nb = n // B
    return [perm[i * B:(i + 1) * B] for i in range(nb)]


def _env_snapshot(torch_threads):
    return {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "torch_set_num_threads": torch_threads}


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================================
# PyTorch 臂
# ============================================================================
def run_torch_arm(cfg, out_path):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(4)

    class ResNet8MNIST(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU())
            self.s1a = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU())
            self.s1b = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16))
            self.s2a = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
            self.s2b = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32))
            self.s2sc = nn.Sequential(nn.Conv2d(16, 32, 1, 2, 0), nn.BatchNorm2d(32))
            self.s3a = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
            self.s3b = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
            self.s3sc = nn.Sequential(nn.Conv2d(32, 64, 1, 2, 0), nn.BatchNorm2d(64))
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            c1 = self.c1(x)
            h = torch.relu(self.s1b(self.s1a(c1)) + c1)
            h = torch.relu(self.s2b(self.s2a(h)) + self.s2sc(h))
            h = torch.relu(self.s3b(self.s3a(h)) + self.s3sc(h))
            g = F.adaptive_avg_pool2d(h, 1)
            return self.fc(torch.flatten(g, 1))

    def load_state_from_numpy(model, p):
        m = {}
        for name in BN_NAMES:
            wname, bname = CONV_KEY[name]
            stem = {"c1": "c1", "s1a": "s1a", "s1b": "s1b", "s2a": "s2a",
                    "s2b": "s2b", "s2sc": "s2sc", "s3a": "s3a", "s3b": "s3b",
                    "s3sc": "s3sc"}[name]
            m[f"{stem}.0.weight"] = torch.tensor(p[wname])
            m[f"{stem}.0.bias"] = torch.tensor(p[bname])
            m[f"{stem}.1.weight"] = torch.tensor(p[f"bn_{name}_g"])
            m[f"{stem}.1.bias"] = torch.tensor(p[f"bn_{name}_b"])
        m["fc.weight"] = torch.tensor(p["fcw"])
        m["fc.bias"] = torch.tensor(p["fcb"])
        model.load_state_dict(m, strict=False)

    def fold_torch_bn(model):
        """门 F1 用：手工折叠 eval BN 到 conv 权重（float32）。"""
        import copy
        fm = copy.deepcopy(model).eval()
        for name in BN_NAMES:
            stem = name
            conv = dict(fm.named_modules())[f"{stem}.0"]
            bn = dict(fm.named_modules())[f"{stem}.1"]
            s = bn.weight.detach().numpy() / np.sqrt(
                bn.running_var.detach().numpy() + 1e-5)
            w = conv.weight.detach().numpy() * s[:, None, None, None]
            b = (conv.bias.detach().numpy() - bn.running_mean.detach().numpy()) \
                * s + bn.bias.detach().numpy()
            conv.weight.data = torch.tensor(w, dtype=torch.float32)
            conv.bias.data = torch.tensor(b, dtype=torch.float32)
            bn.weight.data = torch.ones_like(bn.weight.data)
            bn.bias.data = torch.zeros_like(bn.bias.data)
            bn.running_mean.data = torch.zeros_like(bn.running_mean.data)
            bn.running_var.data = torch.ones_like(bn.running_var.data)
        return fm

    Xtr, ytr, Xte, yte = load_arrays(cfg["train_subset"], cfg["test_subset"])
    Xtr_t = torch.tensor(Xtr)
    Xte_t = torch.tensor(Xte)
    runs = []
    for run_id in range(1, cfg["R"] + 1):
        model = ResNet8MNIST()
        load_state_from_numpy(model, _init_numpy_for_torch(run_id))
        opt = torch.optim.SGD(
            model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"])
        batches = epoch_batches(Xtr.shape[0], cfg["B"], run_id)
        epochs = []
        model.train()
        for ep in range(1, cfg["E"] + 1):
            loss_sum, correct, seen = 0.0, 0, 0
            t0 = time.perf_counter()
            for idx in batches:
                xb, yb = Xtr_t[idx], torch.tensor(ytr[idx])
                opt.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                loss_sum += float(loss) * len(idx)
                correct += int((logits.argmax(1).numpy() == yb.numpy()).sum())
                seen += len(idx)
            wall = time.perf_counter() - t0
            epochs.append({"epoch": ep, "train_loss": loss_sum / seen,
                           "train_acc": correct / seen, "batches": len(batches),
                           "epoch_sec": wall,
                           "ms_per_batch": wall / len(batches) * 1000.0})
            print(f"[torch] run{run_id} epoch{ep}: loss={epochs[-1]['train_loss']:.4f} "
                  f"acc={epochs[-1]['train_acc']:.4f} "
                  f"{epochs[-1]['ms_per_batch']:.1f} ms/batch", flush=True)
        # 推理：标准 eval()（计时 pass 丢首轮 warmup）+ 门 F1（折叠公式）
        model.eval()
        infer_batches = epoch_batches_pass(Xte.shape[0], cfg["B_infer"])
        pass_ms, acc_last = [], None
        with torch.no_grad():
            for ps in range(cfg["infer_passes"]):
                correct, t0 = 0, time.perf_counter()
                for idx in infer_batches:
                    out = model(Xte_t[idx])
                    correct += int((out.argmax(1).numpy() == yte[idx]).sum())
                pass_ms.append((time.perf_counter() - t0)
                               / len(infer_batches) * 1000.0)
                acc_last = correct / (len(infer_batches) * cfg["B_infer"])
            # 门 F1：手工折叠 eval BN → 与 eval() logits 对拍（float 域公式验证）
            fm = fold_torch_bn(model)
            batch0 = Xte_t[infer_batches[0]]
            dlogit = float((fm(batch0) - model(batch0)).abs().max())
        counted = [e for e in epochs if e["epoch"] > 1] or epochs[:1]
        runs.append({
            "run_id": run_id,
            "epochs": epochs,
            "train_ms_per_batch_run_mean":
                float(np.mean([e["ms_per_batch"] for e in counted])),
            "epoch_sec_run_mean": float(np.mean([e["epoch_sec"] for e in counted])),
            "train_acc_last": epochs[-1]["train_acc"],
            "test_acc": acc_last,
            "infer_ms_per_batch_run_mean": float(np.mean(pass_ms[1:]) or pass_ms[0]),
            "infer_pass_ms_raw": pass_ms,
            "bn_fold_max_dlogit": dlogit,
        })
        _write_json(out_path, {"arm": "torch", "runs": runs,
                               "env": _env_snapshot(torch.get_num_threads())})
        print(f"[torch] run{run_id} done: test_acc={acc_last:.4f} "
              f"fold_dlogit={dlogit:.2e}", flush=True)

    counted_tr = [r["train_ms_per_batch_run_mean"] for r in runs]
    counted_inf = [r["infer_ms_per_batch_run_mean"] for r in runs]
    summary = {
        "train_ms_per_batch": {"run_means": counted_tr,
                               "grand_mean": float(np.mean(counted_tr)),
                               "min": float(np.min(counted_tr)),
                               "max": float(np.max(counted_tr))},
        "infer_ms_per_batch": {"run_means": counted_inf,
                               "grand_mean": float(np.mean(counted_inf)),
                               "min": float(np.min(counted_inf)),
                               "max": float(np.max(counted_inf))},
        "epoch_sec_grand_mean":
            float(np.mean([r["epoch_sec_run_mean"] for r in runs])),
        "test_acc_grand_mean": float(np.mean([r["test_acc"] for r in runs])),
        "train_acc_grand_mean":
            float(np.mean([r["train_acc_last"] for r in runs])),
        "bn_fold_max_dlogit_max": max(r["bn_fold_max_dlogit"] for r in runs),
    }
    _write_json(out_path, {"arm": "torch", "runs": runs, "summary": summary,
                           "env": _env_snapshot(torch.get_num_threads())})


def _init_numpy_for_torch(run_id):
    """torch 臂起始权重 = 与 SGN 臂同源的 numpy dict（同 init seed）。"""
    rng = np.random.default_rng(7 + run_id)

    def kaiming(shape, fan_in):
        b = np.sqrt(6.0 / fan_in) * np.sqrt(2.0)
        return rng.uniform(-b, b, shape).astype(np.float32)

    p = {}
    p["c1w"] = kaiming((16, 1, 3, 3), 1 * 9)
    p["c1b"] = np.zeros(16, np.float32)
    p["s1a_w"] = kaiming((16, 16, 3, 3), 16 * 9)
    p["s1a_b"] = np.zeros(16, np.float32)
    p["s1b_w"] = kaiming((16, 16, 3, 3), 16 * 9)
    p["s1b_b"] = np.zeros(16, np.float32)
    p["s2a_w"] = kaiming((32, 16, 3, 3), 16 * 9)
    p["s2a_b"] = np.zeros(32, np.float32)
    p["s2b_w"] = kaiming((32, 32, 3, 3), 32 * 9)
    p["s2b_b"] = np.zeros(32, np.float32)
    p["s2sc_w"] = kaiming((32, 16, 1, 1), 16)
    p["s2sc_b"] = np.zeros(32, np.float32)
    p["s3a_w"] = kaiming((64, 32, 3, 3), 32 * 9)
    p["s3a_b"] = np.zeros(64, np.float32)
    p["s3b_w"] = kaiming((64, 64, 3, 3), 64 * 9)
    p["s3b_b"] = np.zeros(64, np.float32)
    p["s3sc_w"] = kaiming((64, 32, 1, 1), 32)
    p["s3sc_b"] = np.zeros(64, np.float32)
    p["fcw"] = kaiming((10, 64), 64)
    p["fcb"] = np.zeros(10, np.float32)
    for name, c in BN_CHANNELS.items():
        p[f"bn_{name}_g"] = np.ones(c, np.float32)
        p[f"bn_{name}_b"] = np.zeros(c, np.float32)
    return p


# ============================================================================
# PyTorch 原生参照臂（torch_native：生态惯用形态——DataLoader 管道 + 原生
# 配方；模型 = tests/refs/baseline/cnn_mnist.py 的 CNNMnist（~130K 参数，
# 无 BN）。定位 = 参照物，速度数字与 SGN 臂不可直接比，不进 speedup 主表。
# ============================================================================
def run_torch_native_arm(cfg, out_path):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(4)

    class CNNMnist(nn.Module):
        """原生 MNIST CNN（照搬 tests/refs/baseline/cnn_mnist.py，无 BN）。"""

        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 64)
            self.fc2 = nn.Linear(64, num_classes)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2, 2)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    Xtr, ytr, Xte, yte = load_arrays(cfg["train_subset"], cfg["test_subset"])
    runs = []
    for run_id in range(1, cfg["R"] + 1):
        model = CNNMnist()
        # 原生配方：SGD lr=0.01 momentum=0.9（cnn_mnist.py 默认；非对齐超参）
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(1000 + run_id)
        loader = DataLoader(TensorDataset(torch.tensor(Xtr),
                                          torch.tensor(ytr)),
                            batch_size=cfg["B"], shuffle=True, generator=g,
                            drop_last=True)
        epochs = []
        model.train()
        for ep in range(1, cfg["E"] + 1):
            loss_sum, correct, seen = 0.0, 0, 0
            t0 = time.perf_counter()
            for xb, yb in loader:                 # 原生管道：DataLoader 迭代
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                loss_sum += float(loss) * len(yb)
                correct += int((logits.argmax(1).numpy() == yb.numpy()).sum())
                seen += len(yb)
            wall = time.perf_counter() - t0
            epochs.append({"epoch": ep, "train_loss": loss_sum / seen,
                           "train_acc": correct / seen, "batches": len(loader),
                           "epoch_sec": wall,
                           "ms_per_batch": wall / len(loader) * 1000.0})
            print(f"[torch_native] run{run_id} epoch{ep}: "
                  f"loss={epochs[-1]['train_loss']:.4f} "
                  f"acc={epochs[-1]['train_acc']:.4f} "
                  f"{epochs[-1]['ms_per_batch']:.1f} ms/batch", flush=True)
        model.eval()
        te_loader = DataLoader(TensorDataset(torch.tensor(Xte),
                                             torch.tensor(yte)),
                               batch_size=cfg["B_infer"])
        pass_ms, acc_last = [], None
        with torch.no_grad():
            for ps in range(cfg["infer_passes"]):
                correct, t0 = 0, time.perf_counter()
                for xb, yb in te_loader:
                    out = model(xb)
                    correct += int((out.argmax(1).numpy() == yb.numpy()).sum())
                pass_ms.append((time.perf_counter() - t0) / len(te_loader)
                               * 1000.0)
                acc_last = correct / (len(te_loader) * cfg["B_infer"])
        counted = [e for e in epochs if e["epoch"] > 1] or epochs[:1]
        runs.append({
            "run_id": run_id, "epochs": epochs,
            "train_ms_per_batch_run_mean":
                float(np.mean([e["ms_per_batch"] for e in counted])),
            "epoch_sec_run_mean": float(np.mean([e["epoch_sec"] for e in counted])),
            "train_acc_last": epochs[-1]["train_acc"],
            "test_acc": acc_last,
            "infer_ms_per_batch_run_mean": float(np.mean(pass_ms[1:]) or pass_ms[0]),
        })
        _write_json(out_path, {"arm": "torch_native", "runs": runs,
                               "env": _env_snapshot(torch.get_num_threads())})
        print(f"[torch_native] run{run_id} done: test_acc={acc_last:.4f}",
              flush=True)

    counted_tr = [r["train_ms_per_batch_run_mean"] for r in runs]
    counted_inf = [r["infer_ms_per_batch_run_mean"] for r in runs]
    summary = {
        "train_ms_per_batch": {"run_means": counted_tr,
                               "grand_mean": float(np.mean(counted_tr)),
                               "min": float(np.min(counted_tr)),
                               "max": float(np.max(counted_tr))},
        "infer_ms_per_batch": {"run_means": counted_inf,
                               "grand_mean": float(np.mean(counted_inf)),
                               "min": float(np.min(counted_inf)),
                               "max": float(np.max(counted_inf))},
        "epoch_sec_grand_mean":
            float(np.mean([r["epoch_sec_run_mean"] for r in runs])),
        "test_acc_grand_mean": float(np.mean([r["test_acc"] for r in runs])),
        "note": "参照物：模型/管道与 SGN 臂不同（CNNMnist 130K + DataLoader），"
                "速度数字不可与 SGN 臂直接比",
    }
    _write_json(out_path, {"arm": "torch_native", "runs": runs,
                           "summary": summary,
                           "env": _env_snapshot(torch.get_num_threads())})


# ============================================================================
# 编排态与 checks
# ============================================================================
def orchestrate(cfg, tier, smoke, results_path):
    prepare_npz()
    env = {**os.environ, "OMP_NUM_THREADS": "4"}
    logs = {}
    arms = {}
    arm_order = ("sgn", "torch", "torch_native") if not smoke else ("sgn", "torch")
    for arm in arm_order:
        log_path = os.path.join(HERE, f"benchmark_mnist8_production_{arm}.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--arm", arm,
                 "--child", "--tier", tier] + (["--smoke"] if smoke else []),
                env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
            proc.wait()
        logs[arm] = {"returncode": proc.returncode, "log": log_path}
        arm_path = os.path.join(HERE, f"benchmark_mnist8_production_{arm}.json")
        arms[arm] = _read_json(arm_path) if os.path.exists(arm_path) else None
        print(f"[orchestrator] {arm}: returncode={proc.returncode}", flush=True)

    checks = []
    sgn_d, torch_d = arms.get("sgn"), arms.get("torch")
    native_d = arms.get("torch_native")

    def add(name, value, ok, threshold=""):
        checks.append({"name": name, "value": value, "threshold": threshold,
                       "pass": bool(ok)})

    # 精度门（smoke 下 report-only）
    gate = cfg["acc_gate"]
    for arm, d, key in (("sgn", sgn_d, "test_acc_fold_grand_mean"),
                        ("torch", torch_d, "test_acc_grand_mean")):
        if d and "summary" in d:
            acc = d["summary"][key]
            add(f"acc_gate_{arm}", acc,
                (acc >= gate) if gate is not None else True,
                f">={gate}" if gate else "report-only(smoke)")
        else:
            add(f"acc_gate_{arm}", "arm-missing", False)
    # 门 F1（torch 折叠公式）
    if torch_d and "summary" in torch_d:
        dl = torch_d["summary"]["bn_fold_max_dlogit_max"]
        add("bn_fold_formula_torch", dl, dl <= 1e-3, "<=1e-3")
    else:
        add("bn_fold_formula_torch", "arm-missing", False)
    # 门 F2（SGN 折叠 vs 分离式）
    if sgn_d and "summary" in sgn_d:
        dacc = abs(sgn_d["summary"]["test_acc_fold_grand_mean"]
                   - sgn_d["summary"]["test_acc_separated_grand_mean"])
        add("bn_fold_semantics_sgn", dacc, dacc <= 0.005, "|d|<=0.005")
    else:
        add("bn_fold_semantics_sgn", "arm-missing", False)
    # 批数 / 梯度完整性 / loss 有限 / 线程断言
    for arm, d in (("sgn", sgn_d), ("torch", torch_d)):
        if d and "runs" in d:
            nb = {r["epochs"][0]["batches"] for r in d["runs"]}
            add(f"batches_count_{arm}", sorted(nb), len(nb) == 1)
            ng = d.get("none_grad_count", 0 if arm == "torch" else -1)
            add(f"grads_complete_{arm}", ng, ng == 0 if arm == "sgn" else True)
            losses = [e["train_loss"] for r in d["runs"] for e in r["epochs"]]
            lo = bool(np.isfinite(np.array(losses, dtype=np.float64)).all())
            add(f"loss_finite_{arm}", "finite" if lo else "non-finite", lo)
        else:
            for nm in ("batches_count", "grads_complete", "loss_finite"):
                add(f"{nm}_{arm}", "arm-missing", False)
    omp_ok = all((a or {}).get("env", {}).get("OMP_NUM_THREADS") == "4"
                 for a in (sgn_d, torch_d) if a)
    add("threads_assert", {"omp": "4", "torch": 4}, omp_ok and
        bool(torch_d and torch_d.get("env", {}).get("torch_set_num_threads") == 4))

    speedup = None
    if sgn_d and torch_d and "summary" in sgn_d and "summary" in torch_d:
        speedup = {
            "torch_over_sgn_train":
                round(sgn_d["summary"]["train_ms_per_batch"]["grand_mean"]
                      / torch_d["summary"]["train_ms_per_batch"]["grand_mean"], 3),
            "torch_over_sgn_infer":
                round(sgn_d["summary"]["infer_ms_per_batch"]["grand_mean"]
                      / torch_d["summary"]["infer_ms_per_batch"]["grand_mean"], 3),
            "sgn_epoch_sec_grand_mean": sgn_d["summary"]["epoch_sec_grand_mean"],
            "torch_epoch_sec_grand_mean": torch_d["summary"]["epoch_sec_grand_mean"],
        }
    all_pass = all(c["pass"] for c in checks)
    both_done = all(a and "summary" in a for a in (sgn_d, torch_d))
    status = "GO" if (all_pass and both_done) else (
        "PARTIAL" if any(a for a in (sgn_d, torch_d)) else "FAIL")
    result = {
        "config": {
            "script": "benchmark_mnist8_production.py", "tier": tier,
            "smoke": smoke, "R": cfg["R"], "E": cfg["E"], "B": cfg["B"],
            "B_infer": cfg["B_infer"], "lr": cfg["lr"],
            "momentum": cfg["momentum"], "momentum_form": "pytorch(classical=False)",
            "weight_decay": cfg["weight_decay"],
            "sgn": {"strategy": "A1", "ste_bits": 8, "ste_clip_sigma": 4.0,
                    "sr_seed": "1234+run_id",
                    "quant8": "conv+fc weights per step, BN gamma/beta float"},
            "torch": {"precision": "float32"},
            "threads": {"OMP_NUM_THREADS": 4, "torch_set_num_threads": 4},
            "seeds": {"init": "7+run_id", "batch": "1000+run_id",
                      "sr": "1234+run_id"},
            "infer_form": "sgn=bn-folded quant8-w + STE (deployment); "
                          "torch=eval() float32",
            "data": {"npz": NPZ_PATH,
                     "normalize": "(x/255-0.1307)/0.3081"},
            "warmup": "epoch1 of each run + first infer pass discarded",
            "averaging": "level1=mean within run; level2=mean over runs",
            "timing_note": "manual in-memory batching both arms; per-batch "
                           "slice+tensor-copy+fwd+loss+bwd+opt timed; "
                           "preload/npz/fold/eval excluded; no DataLoader",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rows": {"sgn": sgn_d, "torch": torch_d, "torch_native": native_d,
                 "logs": logs},
        "checks": checks,
        "verdict": {"status": status, "speedup": speedup,
                    "note": "timing is report-only; verdict gates on checks "
                            "only; torch_native 为参照物不进 speedup 主表"},
    }
    _write_json(results_path, result)
    print(f"verdict: {status}  speedup: {speedup}", flush=True)
    print(f"results -> {results_path}", flush=True)
    return 0 if status in ("GO",) else 1


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sgn", "torch", "torch_native", "both"],
                    default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--export-model", action="store_true",
                    help="sgn 臂训练后保存 checkpoint npz（发布资产）")
    ap.add_argument("--tier", choices=["conservative", "aggressive", "single"],
                    default="conservative")
    args = ap.parse_args()
    cfg = build_config(args.tier, args.smoke)
    if args.arm == "both":
        suffix = "_smoke" if args.smoke else ""
        results = os.path.join(HERE, f"benchmark_mnist8_production{suffix}_results.json")
        sys.exit(orchestrate(cfg, args.tier, args.smoke, results))
    # 子进程：按 arm 只导入自己框架
    out_path = os.path.join(HERE, f"benchmark_mnist8_production_{args.arm}.json")
    if not os.path.exists(NPZ_PATH):
        # 允许单独跑子进程：先补 npz（需 torchvision，仅此一处）
        prepare_npz()
    if args.arm == "sgn":
        sys.path.insert(0, REPO_ROOT)
        # F3（2026-09-08）：导出直写 tests/refs/ 正式位（资产随仓，verify 脚本同源）
        exp = (os.path.join(REPO_ROOT, "engine", "sgn", "tests", "refs",
                            "benchmark_mnist8_production_sgn_model.npz")
               if args.export_model else None)
        run_sgn_arm(cfg, out_path, export_model_path=exp)
    elif args.arm == "torch_native":
        run_torch_native_arm(cfg, out_path)
    else:
        run_torch_arm(cfg, out_path)


if __name__ == "__main__":
    main()
