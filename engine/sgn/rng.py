"""rng.py - 随机性/可复现工具（基础设施 B3，2026-08-29）

定位（infrastructure_roadmap_2026_08_29.md §5 B3）：
  统一散落在训练脚本里的种子派生约定（如 resnet8_free_roam：init=default_rng(seed)、
  C++ SR=1234+seed、视图漫游=default_rng(seed+2)、权重 SR=default_rng(seed+3)），
  提供复现断言工具。

设计：
  - SeedBundle：一个 seed 派生出的全部随机源（确定性派生，同 seed 必同序列）。
      .init_rng   权重初始化 Generator（= default_rng(seed)）
      .roam       视图漫游 Generator（= default_rng(seed+2)，随机源分离约定见下）
      .sr         权重 SR 舍入 Generator（= default_rng(seed+3)）
      .sr_cpp_seed C++ SR seed（backward_strategy.h thread_local rng，= 1234+seed）
    make_seed_bundle(seed, ag=None) 构造；ag 非 None 时自动 ag.set_sr_seed。
  - assert_reproducible：同 seed 多次运行 task_fn()，断言返回逐位一致。

随机源分离约定（审查 2026-08-20，resnet8 沿用）：
  视图漫游选择与权重 SR 舍入不得共用随机流——SR 消耗量变化会无预警扰动漫游序列。
  故 roam/sr 用不同 seed 偏移派生，各持独立 Generator。

⚠️ fork/线程约束（文档化，非代码修复）：
  - C++ SR rng 是 thread_local（backward_strategy.h L159），同线程内由 sr_cpp_seed
    初始化；OpenMP 多线程下每线程独立，但共享全局 sr_global_seed——多线程训练
    需自行保证每线程语义（Level 决策层标量逻辑无此问题）。
  - 多进程 fork（如验证脚本后台并行）时，子进程须重新 make_seed_bundle 或显式
    重播种（numpy Generator 状态不会被 fork 隐式复制语义保证逐位一致）。
  - 本工具面向"同 seed 单进程内可复现"；跨进程/跨线程逐位复现需额外约定。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

# 种子派生偏移（与 resnet8_free_roam 等现有脚本保持一致，避免改序列破坏已有结果）
SR_CPP_OFFSET = 1234      # C++ SR seed 偏移（backward_strategy）
ROAM_OFFSET = 2           # 视图漫游 rng 偏移
SR_OFFSET = 3             # 权重 SR rng 偏移


class SeedBundle:
    """一个 seed 派生出的全部随机源（确定性）。"""

    def __init__(self, seed: int, ag=None):
        self.seed = seed
        self.init_rng = np.random.default_rng(seed)        # 权重初始化
        self.roam = np.random.default_rng(seed + ROAM_OFFSET)
        self.sr = np.random.default_rng(seed + SR_OFFSET)
        self.sr_cpp_seed = SR_CPP_OFFSET + seed
        if ag is not None:
            ag.set_sr_seed(self.sr_cpp_seed)

    def state(self) -> Dict[str, Any]:
        """全部 Generator 的 bit_generator.state（供 checkpoint 保存 rng 状态）。"""
        return {"init": self.init_rng.bit_generator.state,
                "roam": self.roam.bit_generator.state,
                "sr": self.sr.bit_generator.state,
                "sr_cpp_seed": self.sr_cpp_seed}

    def restore(self, state: Dict[str, Any], ag=None) -> "SeedBundle":
        """从 state 恢复（配合 checkpoint；返回 self 便于链式）。"""
        self.init_rng.bit_generator.state = state["init"]
        self.roam.bit_generator.state = state["roam"]
        self.sr.bit_generator.state = state["sr"]
        self.sr_cpp_seed = state["sr_cpp_seed"]
        if ag is not None:
            ag.set_sr_seed(self.sr_cpp_seed)
        return self


def make_seed_bundle(seed: int, ag=None) -> SeedBundle:
    """构造 seed 派生的随机源集合；ag 非 None 时设置 C++ SR seed。"""
    return SeedBundle(seed, ag=ag)


def assert_reproducible(task_fn: Callable[[], Any], seed: int,
                        repeats: int = 2, label: str = "task") -> List[Any]:
    """同 seed 多次运行 task_fn()，断言返回逐位一致。

    task_fn 内部应经 make_seed_bundle(seed) 派生随机源（或显式重播种），
    保证每次运行消费相同随机序列。返回各次结果列表。
    """
    if repeats < 2:
        raise ValueError("repeats 必须 >= 2")
    outs = [task_fn() for _ in range(repeats)]
    for i, o in enumerate(outs[1:], start=1):
        assert o == outs[0], (
            f"[{label}] 运行 {i} 与运行 0 结果不一致（同 seed={seed} 应逐位可复现）")
    return outs
