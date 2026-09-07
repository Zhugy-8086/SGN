"""experiment.py - 训练实验框架（基础设施 A1，2026-08-29）

定位（infrastructure_roadmap_2026_08_29.md §4 A1）：
  为 MSint×Level 验证爬坡期的高频实验提供可复用实验壳——多 seed 参数化执行、
  结果聚合（跨 seed 统计）、JSON/CSV 导出、参照传递规范化。
  零侵入：不重构 resnet8_free_roam 的 mode 分发，仅在其上新增聚合层。

设计：
  - task_fn(seed, **kw) -> dict[str, float|int]：一次实验的指标（如 last50/min/roam8）
  - run_seeds(task_fn, seeds, **kw) -> dict[int, dict]：逐 seed 收集
  - aggregate(runs, metrics) -> dict：per-seed 表 + mean/std/min/max
  - export_json / export_csv：导出到 logs/
  - save_ref_json / load_ref_json：参照传递规范化（对齐 resnet8_free_roam 的
    logs/free_roam_ref.json 格式 {"T", "seeds", "<strategy>_last50": {seed: val}}）
  - make_resnet8_task(...)：resnet8_free_roam 快速接入（按需 import，避免硬依赖）
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

TaskFn = Callable[[int], Dict[str, Any]]


# ============================================================================
# 执行与聚合
# ============================================================================

def run_seeds(task_fn: TaskFn, seeds: Sequence[int],
              label: str = "task") -> Dict[int, Dict[str, Any]]:
    """逐 seed 执行 task_fn，收集结果为 {seed: metrics}。

    task_fn(seed) 返回 dict[str, float|int]（如 {"last50":..., "min":...}）。
    """
    runs: Dict[int, Dict[str, Any]] = {}
    for sd in seeds:
        runs[sd] = task_fn(sd)
        print(f"  [{label:14s} seed={sd:2d}] " + "  ".join(
            f"{k} {v:.4e}" if isinstance(v, float) else f"{k} {v}"
            for k, v in runs[sd].items()))
    return runs


def aggregate(runs: Dict[int, Dict[str, Any]],
              metrics: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """跨 seed 统计：per-seed 表 + 每指标 mean±std/min/max。

    返回：{"per_seed": {seed: metrics}, "stats": {metric: {mean,std,min,max}}}
    """
    if not runs:
        return {"per_seed": {}, "stats": {}}
    # 指标集合 = 显式给定 或 首个 seed 的键（数值型）
    if metrics is None:
        first = next(iter(runs.values()))
        metrics = [k for k, v in first.items() if isinstance(v, (int, float))]
    stats: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = np.array([float(r[m]) for r in runs.values()])
        stats[m] = {"mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max())}
    return {"per_seed": runs, "stats": stats, "metrics": list(metrics)}


# ============================================================================
# 导出
# ============================================================================

def export_json(runs: Dict[int, Dict[str, Any]], path: str,
                meta: Optional[Dict[str, Any]] = None) -> str:
    """导出 {seed: metrics} 为 JSON（可选 meta 如 T/seeds/strategy）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta or {}, "runs": {str(k): v for k, v in runs.items()}},
                  f, indent=2)
    return path


def export_csv(runs: Dict[int, Dict[str, Any]], path: str) -> str:
    """导出 per-seed 表为 CSV（行=seed，列=指标）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    metrics = [k for k, v in next(iter(runs.values())).items()
               if isinstance(v, (int, float))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + metrics)
        for sd, r in runs.items():
            w.writerow([sd] + [r[m] for m in metrics])
    return path


# ============================================================================
# 参照传递（protocol → attribution 规范化，对齐 free_roam_ref.json 口径）
# ============================================================================

def save_ref_json(path: str, T: int, seeds: Sequence[int],
                  last50: Dict[int, float], strategy: str = "free_anneal") -> str:
    """写参照 JSON（格式对齐 resnet8_free_roam 的 free_roam_ref.json）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"T": T, "seeds": list(seeds),
                   f"{strategy}_last50": {int(s): float(v) for s, v in last50.items()}},
                  f, indent=2)
    return path


def load_ref_json(path: str, strategy: str = "free_anneal") -> Dict[str, Any]:
    """读参照 JSON，返回 {"T", "seeds", "last50": {seed: val}}。"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    key = f"{strategy}_last50"
    return {
        "T": d.get("T"),
        "seeds": d.get("seeds"),
        "last50": {int(k): float(v) for k, v in d[key].items()} if key in d else {},
    }


# ============================================================================
# resnet8_free_roam 快速接入
# ============================================================================

def make_resnet8_task(strategy: str, X: Any, y: Any, T: int,
                      lr0: float = 0.02, momentum: float = 0.9,
                      decay_every: int = 500, mode: str = "fixed",
                      p8: Optional[float] = None) -> TaskFn:
    """构造 resnet8 训练任务：task_fn(seed) -> {"min","last50","conv","roam8"}。

    按需 import resnet8_free_roam（避免实验框架硬依赖训练脚本）。
    mode: fixed（固定策略）/ dose（漫游段 8bit 剂量，需 p8）。
    """
    import resnet8_free_roam as R

    def task(seed: int) -> Dict[str, Any]:
        if mode == "dose":
            assert p8 is not None, "dose mode 需 p8"
            R._P8 = p8
        losses, roam = R.train(strategy, X, y, T=T, lr0=lr0,
                               momentum=momentum, decay_every=decay_every, seed=seed)
        last = float(np.mean(losses[-50:]))
        return {
            "min": float(np.min(losses)),
            "last50": last,
            "conv": float(losses[max(0, T - 100)] - losses[-1]),
            "roam8": roam["n8"],
        }

    return task
