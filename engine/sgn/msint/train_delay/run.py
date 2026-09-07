# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint×Level 验证统一入口（基础设施 A2，2026-08-29）

语义归属：MSint×Level 方向（Level 旋钮层调度）验证实验的统一参数化 CLI。
替代 dose_sweep.py / real_verify.py 的手写 argv 转发，复用 A1 实验框架
（sgn.run_seeds / aggregate / export_*）做多 seed 聚合 + 导出。

用法（cd engine/sgn）：
    # dose：漫游段 8bit 剂量扫描（合成数据，free_anneal + p8 权重）
    python msint/train_delay/run.py dose --T 1500 --seeds 7,11 --p8 0.5,0.7,0.85,1.0
    # real：真实 CIFAR 验证（subset 子集）
    python msint/train_delay/run.py real --T 1000 --seeds 7 --subset 512
    # fixed：固定策略（合成数据）
    python msint/train_delay/run.py fixed --T 1500 --seeds 7 --strategy fixed8
    # 通用：--outdir 指定输出目录（默认 logs/）

输出：聚合表打印 + logs/train_delay_<mode>_<suffix>.csv/.json。
"""
from __future__ import annotations

import argparse
import os
import sys

_AUTODIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'autograd'))
_ENGINE = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _AUTODIR)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'build')))
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

import numpy as np  # noqa: E402

from engine.sgn.experiment import (  # noqa: E402
    make_resnet8_task, run_seeds, aggregate, export_json, export_csv,
)
import resnet8_free_roam as R  # noqa: E402


def _parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def _fmt_agg(agg: dict) -> str:
    """聚合 stats 打印行：metric mean±std（min/max）。"""
    parts = []
    for m, st in agg["stats"].items():
        parts.append(f"{m} {st['mean']:.4e}±{st['std']:.1e} "
                     f"[{st['min']:.2e},{st['max']:.2e}]")
    return "  ".join(parts)


def run_dose(args) -> int:
    X = R.make_lowfreq_fields(args.N)
    y = R.teacher_targets(X)
    seeds = [int(s) for s in _parse_list(args.seeds)]
    p8s = [float(p) for p in _parse_list(args.p8)]
    print(f"[dose] T={args.T} seeds={seeds} P[8bit]={p8s} N={args.N}")
    all_runs = {}
    for p8 in p8s:
        task = make_resnet8_task("free_anneal", X, y, args.T,
                                 mode="dose", p8=p8)
        runs = run_seeds(task, seeds, label=f"p8={p8}")
        agg = aggregate(runs)
        print(f"  p8={p8}: {_fmt_agg(agg)}")
        all_runs[p8] = agg
        export_json(runs, os.path.join(args.outdir, f"train_delay_dose_p{p8}.json"),
                    meta={"T": args.T, "strategy": "free_anneal", "p8": p8})
        export_csv(runs, os.path.join(args.outdir, f"train_delay_dose_p{p8}.csv"))
    # 汇总表：各 p8 的 last50 mean
    print("\n  dose 汇总（last50 mean，跨 seed）")
    print(f"  {'P[8bit]':>8}  {'mean':>12}  {'std':>12}  {'min':>12}  {'max':>12}")
    for p8 in p8s:
        st = all_runs[p8]["stats"]["last50"]
        print(f"  {p8:>8.2f}  {st['mean']:>12.4e}  {st['std']:>12.1e}  "
              f"{st['min']:>12.4e}  {st['max']:>12.4e}")
    return 0


def run_real(args) -> int:
    X, y = R.load_cifar_real(subset=args.subset)
    seeds = [int(s) for s in _parse_list(args.seeds)]
    print(f"[real] T={args.T} seeds={seeds} subset={args.subset} 数据{X.shape}")
    task = make_resnet8_task("free_anneal", X, y, args.T)
    runs = run_seeds(task, seeds, label="real")
    agg = aggregate(runs)
    print(f"  {_fmt_agg(agg)}")
    export_json(runs, os.path.join(args.outdir, "train_delay_real.json"),
                meta={"T": args.T, "subset": args.subset})
    export_csv(runs, os.path.join(args.outdir, "train_delay_real.csv"))
    return 0


def run_fixed(args) -> int:
    X = R.make_lowfreq_fields(args.N)
    y = R.teacher_targets(X)
    seeds = [int(s) for s in _parse_list(args.seeds)]
    print(f"[fixed] strategy={args.strategy} T={args.T} seeds={seeds} N={args.N}")
    task = make_resnet8_task(args.strategy, X, y, args.T)
    runs = run_seeds(task, seeds, label=args.strategy)
    agg = aggregate(runs)
    print(f"  {_fmt_agg(agg)}")
    export_json(runs, os.path.join(args.outdir, f"train_delay_{args.strategy}.json"),
                meta={"T": args.T, "strategy": args.strategy})
    export_csv(runs, os.path.join(args.outdir, f"train_delay_{args.strategy}.csv"))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="MSint×Level 验证统一入口")
    default_out = os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'logs'))
    sub = p.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("dose", help="漫游段 8bit 剂量扫描（合成数据）")
    d.add_argument("--T", type=int, default=1500)
    d.add_argument("--seeds", type=str, default="7,11")
    d.add_argument("--p8", type=str, default="0.5,0.7,0.85,1.0")
    d.add_argument("--N", type=int, default=128)
    d.add_argument("--outdir", type=str, default=default_out)

    r = sub.add_parser("real", help="真实 CIFAR 验证")
    r.add_argument("--T", type=int, default=1000)
    r.add_argument("--seeds", type=str, default="7")
    r.add_argument("--subset", type=int, default=512)
    r.add_argument("--outdir", type=str, default=default_out)

    f = sub.add_parser("fixed", help="固定策略（合成数据）")
    f.add_argument("--T", type=int, default=1500)
    f.add_argument("--seeds", type=str, default="7")
    f.add_argument("--strategy", type=str, default="fixed8")
    f.add_argument("--N", type=int, default=128)
    f.add_argument("--outdir", type=str, default=default_out)

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "dose":
        return run_dose(args)
    if args.mode == "real":
        return run_real(args)
    return run_fixed(args)


if __name__ == "__main__":
    sys.exit(main())
