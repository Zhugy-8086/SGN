#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bug 修复验证测试 — 验证 Bug #1 / #2 / #3 的修复

测试项:
  [1] Bug #1: train_batch_multi_layer 中 _step_counter 递增
  [2] Bug #2: _verify 阈值降低后验证通过
  [3] Bug #3: _rebuild_for_dimension 单层模式重置 _step_counter
  [4] 端到端: 默认配置下训练 200 步，验证有产出
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.utils import C, box, hr

# ============================================================
# 测试框架
# ============================================================

_passed = 0
_failed = 0
_total = 4


def test_pass(name, detail=""):
    global _passed
    _passed += 1
    dots = "." * (48 - len(name))
    detail_str = f" {C.DIM}{detail}{C.RST}" if detail else ""
    print(f"  {C.GRN}✓{C.RST} {name}{dots}{C.GRN}PASS{C.RST}{detail_str}")


def test_fail(name, reason=""):
    global _failed
    _failed += 1
    print(f"  {C.RED}✗{C.RST} {name}{C.RED}FAIL{C.RST} — {reason}")


def run():
    """运行所有测试"""
    box("Bug 修复验证测试")
    print(f"  {C.DIM}共 {_total} 项测试{C.RST}\n")

    # ---- [1] Bug #1: _step_counter 递增 ----
    print(f"  {C.CYN}[1/{_total}]{C.RST} Bug #1: _step_counter 在批次训练中递增")
    try:
        from engine.core import SGNCore
        from engine.input import create_default_source
        from engine.config import ConfigRegistry, CONFIG

        # 临时设置多层级 + 批次模式
        ConfigRegistry.set("ENABLE_MULTI_LAYER_NEURON", True)
        ConfigRegistry.set("BATCH_TRAIN_ENABLED", True)

        core = SGNCore()
        source = create_default_source()
        batch = source.generate_batch(10, split='all')

        step_before = core._step_counter
        results = core.train_batch(batch)
        step_after = core._step_counter

        # 训练 10 个样本后，_step_counter 应该增加了 10
        if step_after == step_before + 10:
            test_pass("Bug #1: _step_counter 递增", f"{step_before} → {step_after}")
        else:
            test_fail("Bug #1: _step_counter 递增",
                      f"预期 {step_before}+10={step_before+10}，实际 {step_after}")
    except Exception as e:
        test_fail("Bug #1: _step_counter 递增", str(e))

    # ---- [2] Bug #2: _verify 阈值 ----
    print(f"  {C.CYN}[2/{_total}]{C.RST} Bug #2: _verify 降低阈值后通过率")
    try:
        from engine.core import SGNCore
        from engine.layers import extract_layers
        from engine.input import create_default_source

        ConfigRegistry.set("ENABLE_MULTI_LAYER_NEURON", True)
        core2 = SGNCore()
        source2 = create_default_source()
        batch2 = source2.generate_batch(50, split='all')

        verify_count = 0
        for intensity, label in batch2:
            layers, layer_count = extract_layers(intensity, d=core2.D, strategy=core2.layer_strategy)
            if core2._verify(intensity, layers, layer_count):
                verify_count += 1

        # 修复后，50 个样本中至少应有 30% 通过验证
        rate = verify_count / len(batch2) * 100
        if rate >= 30:
            test_pass("Bug #2: _verify 通过率", f"{rate:.0f}% ({verify_count}/{len(batch2)})")
        else:
            test_fail("Bug #2: _verify 通过率",
                      f"通过率 {rate:.0f}% 低于预期 30%")
    except Exception as e:
        test_fail("Bug #2: _verify 通过率", str(e))

    # ---- [3] Bug #3: _rebuild_for_dimension 重置 _step_counter ----
    print(f"  {C.CYN}[3/{_total}]{C.RST} Bug #3: _rebuild_for_dimension 重置 _step_counter")
    try:
        from engine.core import SGNCore
        from engine.config import ConfigRegistry

        # 单层模式
        ConfigRegistry.set("ENABLE_MULTI_LAYER_NEURON", False)
        ConfigRegistry.set("GRAPH_MODE", False)

        core3 = SGNCore()
        core3._step_counter = 999  # 模拟旧值

        # 触发维度重建
        core3._rebuild_for_dimension(64)

        if core3._step_counter == 0:
            test_pass("Bug #3: _step_counter 重置", "单层模式重建后归零")
        else:
            test_fail("Bug #3: _step_counter 重置",
                      f"重建后 _step_counter={core3._step_counter}，预期 0")
    except Exception as e:
        test_fail("Bug #3: _step_counter 重置", str(e))

    # ---- [4] 端到端: 200 步训练 ----
    print(f"  {C.CYN}[4/{_total}]{C.RST} 端到端: 默认配置训练 200 步")
    try:
        from engine.core import SGNCore
        from engine.config import ConfigRegistry
        from engine.input import create_default_source
        from app.training import run_training_loop

        ConfigRegistry.set("ENABLE_MULTI_LAYER_NEURON", True)
        ConfigRegistry.set("BATCH_TRAIN_ENABLED", True)
        ConfigRegistry.set("MAX_ITERATIONS", 200)
        ConfigRegistry.set("MODE", "compact")

        core4 = SGNCore()
        source4 = create_default_source()
        samples = source4.generate_batch(2000, split='all')

        # 自动模式训练 200 步（delay_ms=1 即自动模式，无交互停顿）
        history_before = len(core4.history)
        step = run_training_loop(core4, samples=samples, max_step=200, delay_ms=1, source=source4)
        history_after = len(core4.history)

        # 验证：训练后有历史记录，有胜出神经元
        if step >= 200 and history_after > history_before:
            # 检查是否有胜出神经元记录了非零匹配
            trained_neurons = sum(1 for n in core4.N if n["base"].index > 0)
            test_pass("端到端训练", f"{step}步, {history_after}条记录, {trained_neurons}个活跃神经元")
        else:
            test_fail("端到端训练",
                      f"步数={step}, 历史={history_before}→{history_after}")
    except Exception as e:
        test_fail("端到端训练", str(e))

    # ---- 汇总 ----
    hr()
    total = _passed + _failed
    if _failed == 0:
        print(f"  {C.GRN}全部通过！{C.RST} ({_passed}/{total})")
    else:
        print(f"  {C.YEL}通过: {_passed}/{total}  |  {C.RED}失败: {_failed}/{total}{C.RST}")
    print()
    return _failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)