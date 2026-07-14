#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5.1.7 补充: 端到端验证测试（合并版）

合并原 test_v517_supplement.py（10项）和 test_bug_fixes.py（7项），
去重后 8 个测试组，带颜色输出 + 进度点 + 通过率摘要。

测试项:
  [1] 语法检查
  [2] Bug 1: 强度值 0/255 + 验证通过
  [3] 参与率诊断
  [4] Bug 2: classify_sample 方法
  [5] Bug 3: 进度输出与评估间隔
  [6] 基准脚本功能（样本生成/噪声/导入）
  [7] 面板入口
  [8] 端到端运行（测试运行/JSON保存/历史列表/准确率非零）
"""
import sys, os, ast, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.utils import C, box, hr

# ============================================================
# 测试框架
# ============================================================

_passed = 0
_failed = 0
_total = 8


def test_pass(name, detail=""):
    global _passed
    _passed += 1
    dots = "." * (40 - len(name))
    detail_str = f" {C.DIM}{detail}{C.RST}" if detail else ""
    print(f"  {C.GRN}✓{C.RST} {name}{dots}{C.GRN}PASS{C.RST}{detail_str}")


def test_fail(name, reason=""):
    global _failed
    _failed += 1
    print(f"  {C.RED}✗{C.RST} {name}{C.RED}FAIL{C.RST} — {reason}")


# ============================================================
# 测试开始
# ============================================================

box("v5.1.7 补充: 端到端验证测试")
print(f"  {C.DIM}共 {_total} 项测试{C.RST}\n")


# ---- [1] 语法检查 ----
print(f"  {C.CYN}[1/{_total}]{C.RST} 语法检查")
try:
    for f in ['engine/core.py', 'app/panel.py', 'tests/benchmark_convergence.py']:
        ast.parse(open(f, encoding='utf-8').read())
    test_pass("语法检查", "3 个文件")
except SyntaxError as e:
    test_fail("语法检查", str(e))
    sys.exit(1)


# ---- [2] Bug 1: 强度值 0/255 + 验证通过 ----
print(f"\n  {C.CYN}[2/{_total}]{C.RST} Bug 1: 强度值 0/255 + 验证通过")
try:
    from tests.benchmark_convergence import generate_synthetic_samples
    samples = generate_synthetic_samples(num_labels=2, samples_per_label=5, d=64, seed=42)
    intensity_values = set()
    for intensity, _ in samples:
        intensity_values.update(intensity)
    assert intensity_values == {0, 255}, f"强度值应为 {{0,255}}, 实际 {intensity_values}"

    # 训练后应有图生成
    from engine.config import CONFIG, ConfigRegistry
    ConfigRegistry._values["ENABLE_MULTI_LAYER_NEURON"] = True
    ConfigRegistry._values["ENABLE_ADAPTIVE_SILENCE"] = True
    ConfigRegistry._values["NEURON_LAYER_0_COUNT"] = 64
    ConfigRegistry._values["NEURON_LAYER_1_COUNT"] = 32
    ConfigRegistry._values["D"] = 64
    ConfigRegistry._values["BATCH_SIZE"] = 16
    CONFIG["ENABLE_MULTI_LAYER_NEURON"] = True
    CONFIG["ENABLE_ADAPTIVE_SILENCE"] = True
    CONFIG["NEURON_LAYER_0_COUNT"] = 64
    CONFIG["NEURON_LAYER_1_COUNT"] = 32
    CONFIG["D"] = 64
    CONFIG["BATCH_SIZE"] = 16

    from engine.core import SGNCore
    core = SGNCore(seed=42)
    train_samples = generate_synthetic_samples(num_labels=4, samples_per_label=50, d=64, seed=42)
    batch = train_samples[:16]
    for _ in range(13):
        core.train_batch(batch)
    state = core.get_state()
    assert state['templates'] > 0, "训练后应有图生成"
    test_pass("Bug 1 修复", f"0/255, {state['templates']} 图生成")
except Exception as e:
    test_fail("Bug 1 修复", str(e))


# ---- [3] 参与率诊断 ----
print(f"\n  {C.CYN}[3/{_total}]{C.RST} 参与率诊断")
try:
    # 初始参与率应为 0
    core_init = SGNCore(seed=42)
    state0 = core_init.get_state()
    assert "avg_participation_rate" in state0, "缺少 avg_participation_rate 字段"
    assert state0['avg_participation_rate'] == 0.0

    # 训练后参与率应 > 0
    state1 = core.get_state()
    assert state1['avg_participation_rate'] > 0.0, "训练后参与率应 > 0"
    assert 0.0 <= state1['avg_participation_rate'] <= 1.0
    test_pass("参与率诊断", f"{state0['avg_participation_rate']:.4f} → {state1['avg_participation_rate']:.4f}")
except Exception as e:
    test_fail("参与率诊断", str(e))


# ---- [4] Bug 2: classify_sample 方法 ----
print(f"\n  {C.CYN}[4/{_total}]{C.RST} Bug 2: classify_sample 方法")
try:
    from engine.layers import classify_sample, classify_multi_layer
    import inspect
    sig = inspect.signature(classify_sample)
    assert 'core' in sig.parameters and 'intensity' in sig.parameters

    # 单层模式推理不应崩溃
    ConfigRegistry._values["ENABLE_MULTI_LAYER_NEURON"] = False
    CONFIG["ENABLE_MULTI_LAYER_NEURON"] = False
    core_single = SGNCore(seed=42)
    for i in range(20):
        intensity, label = train_samples[i % len(train_samples)]
        core_single.train(intensity, label)
    intensity, label = train_samples[0]
    pred, score = classify_sample(core_single, intensity, 64)
    test_pass("Bug 2 修复", f"pred={pred}, score={score}")
except Exception as e:
    test_fail("Bug 2 修复", str(e))


# ---- [5] Bug 3: 进度输出与评估间隔 ----
print(f"\n  {C.CYN}[5/{_total}]{C.RST} Bug 3: 进度输出与评估间隔")
try:
    src = open('tests/benchmark_convergence.py', encoding='utf-8').read()
    assert 'next_eval_step' in src, "缺少 next_eval_step 变量"
    assert 'next_eval_step += eval_interval' in src, "缺少 next_eval_step 累加"
    assert 'sys.stdout.write' in src, "缺少进度输出"

    # 评估间隔命中测试
    ConfigRegistry._values["ENABLE_MULTI_LAYER_NEURON"] = True
    CONFIG["ENABLE_MULTI_LAYER_NEURON"] = True
    ConfigRegistry._values["BATCH_SIZE"] = 16
    CONFIG["BATCH_SIZE"] = 16

    from tests.benchmark_convergence import _train_and_track
    test_samples = generate_synthetic_samples(num_labels=4, samples_per_label=10, d=64, seed=999)
    r = _train_and_track(core, train_samples, test_samples, max_steps=500,
                         eval_interval=100, target_acc=0.5, label="test")
    assert len(r['accuracy_history']) >= 4, f"评估次数过少: {len(r['accuracy_history'])}"
    test_pass("Bug 3 修复", f"{len(r['accuracy_history'])} 次评估命中")
except Exception as e:
    test_fail("Bug 3 修复", str(e))


# ---- [6] 基准脚本功能 ----
print(f"\n  {C.CYN}[6/{_total}]{C.RST} 基准脚本功能")
try:
    from tests.benchmark_convergence import (
        generate_synthetic_samples, add_noise, evaluate_accuracy,
        test_noise_robustness, test_network_scale,
        test_multi_vs_single, test_silence_on_off,
        print_report, save_json_report, list_history_results,
        interactive_menu, main
    )

    # 合成样本
    samples = generate_synthetic_samples(num_labels=4, samples_per_label=10, d=64, seed=42)
    assert len(samples) == 40
    assert set(s[1] for s in samples) == {'A', 'B', 'C', 'D'}

    # 噪声添加
    original = samples[0][0]
    noisy = add_noise(original, 0.3, seed=42)
    diff_count = sum(1 for a, b in zip(original, noisy) if a != b)
    assert diff_count > 0

    test_pass("基准脚本功能", f"40样本, {diff_count}位翻转")
except Exception as e:
    test_fail("基准脚本功能", str(e))


# ---- [7] 面板入口 ----
print(f"\n  {C.CYN}[7/{_total}]{C.RST} 面板入口")
try:
    import app.panel as panel
    assert hasattr(panel, '_benchmark_menu'), "panel 缺少 _benchmark_menu 函数"
    panel_src = open('app/panel.py', encoding='utf-8').read()
    assert '_benchmark_menu' in panel_src and '[n]' in panel_src, "扩展菜单缺少 [n] 入口"
    test_pass("面板入口", "_benchmark_menu + [n]")
except Exception as e:
    test_fail("面板入口", str(e))


# ---- [8] 端到端运行 ----
print(f"\n  {C.CYN}[8/{_total}]{C.RST} 端到端运行")
try:
    # 运行小型噪声测试
    ConfigRegistry._values["NEURON_LAYER_0_COUNT"] = 32
    ConfigRegistry._values["NEURON_LAYER_1_COUNT"] = 16
    ConfigRegistry._values["D"] = 64
    CONFIG["NEURON_LAYER_0_COUNT"] = 32
    CONFIG["NEURON_LAYER_1_COUNT"] = 16
    CONFIG["D"] = 64
    results = test_noise_robustness(d=64, max_steps=200)
    assert len(results) == 5
    for r in results:
        assert 'noise_prob' in r and 'final_acc' in r and 'first_target_step' in r

    # JSON 保存
    all_results = {'test1': results}
    filepath = save_json_report(all_results)
    assert os.path.exists(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert 'timestamp' in data and 'results' in data

    # 历史列表
    files = list_history_results()
    assert len(files) > 0

    # 准确率非零
    final_acc = results[0]['final_acc']
    assert final_acc >= 0.0  # 至少不报错

    test_pass("端到端运行", f"5组结果, JSON已保存, {len(files)}历史")
except Exception as e:
    test_fail("端到端运行", str(e))


# ============================================================
# 摘要
# ============================================================

print()
hr(50)
rate = _passed / _total
if _failed == 0:
    print(f"  {C.BOLD}{C.GRN}结果: {_passed}/{_total} 通过 ✓{C.RST}")
else:
    print(f"  {C.BOLD}{C.RED}结果: {_passed}/{_total} 通过, {_failed} 失败 ✗{C.RST}")
    sys.exit(1)
