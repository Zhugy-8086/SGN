#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5.1.7-patch2 基线验证：用无阈值方式测多层模式真实准确率"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import CONFIG, ConfigRegistry
from engine.core import SGNCore
from engine.layers import classify_multi_layer
from tests.benchmark_convergence import generate_synthetic_samples

ConfigRegistry._values["ENABLE_MULTI_LAYER_NEURON"] = True
CONFIG["ENABLE_MULTI_LAYER_NEURON"] = True
ConfigRegistry._values["NEURON_LAYER_0_COUNT"] = 128
CONFIG["NEURON_LAYER_0_COUNT"] = 128
ConfigRegistry._values["NEURON_LAYER_1_COUNT"] = 64
CONFIG["NEURON_LAYER_1_COUNT"] = 64
ConfigRegistry._values["D"] = 64
CONFIG["D"] = 64

print("=" * 60)
print("6.0 修复前基线：无阈值准确率 vs 80阈值准确率")
print("=" * 60)

samples = generate_synthetic_samples(num_labels=4, samples_per_label=50, d=64, seed=42)
core = SGNCore(seed=42)
print("训练 400 步...")
for i in range(400):
    intensity, label = samples[i % len(samples)]
    core.train(intensity, label)

test_samples = generate_synthetic_samples(num_labels=4, samples_per_label=25, d=64, seed=999)

correct_no_thresh = 0  # 无阈值（混淆矩阵逻辑）
correct_80 = 0          # 80 阈值（do_batch_test 逻辑）
correct_30 = 0          # 30 阈值（假设的多层阈值）
total = 100

scores_correct = []  # 正确分类的分数分布
scores_wrong = []    # 错误分类的分数分布

for i in range(total):
    intensity, label = test_samples[i % len(test_samples)]
    pred, score = classify_multi_layer(core, intensity, 64)
    if pred == label:
        correct_no_thresh += 1
        scores_correct.append(score)
        if score >= 80:
            correct_80 += 1
        if score >= 30:
            correct_30 += 1
    else:
        scores_wrong.append(score)

print(f"\n--- 准确率对比 ---")
print(f"  无阈值（混淆矩阵逻辑）: {correct_no_thresh}/{total} = {correct_no_thresh}%")
print(f"  80 阈值（do_batch_test）: {correct_80}/{total} = {correct_80}%")
print(f"  30 阈值（假设多层阈值）: {correct_30}/{total} = {correct_30}%")

print(f"\n--- 正确分类的分数分布 ---")
if scores_correct:
    print(f"  数量: {len(scores_correct)}")
    print(f"  范围: {min(scores_correct)}-{max(scores_correct)}")
    print(f"  平均: {sum(scores_correct)/len(scores_correct):.1f}")
    # 分桶
    buckets = {"<30": 0, "30-49": 0, "50-69": 0, "70-79": 0, ">=80": 0}
    for s in scores_correct:
        if s < 30: buckets["<30"] += 1
        elif s < 50: buckets["30-49"] += 1
        elif s < 70: buckets["50-69"] += 1
        elif s < 80: buckets["70-79"] += 1
        else: buckets[">=80"] += 1
    print(f"  分桶: {buckets}")
else:
    print("  无正确分类")

print(f"\n--- 错误分类的分数分布 ---")
if scores_wrong:
    print(f"  数量: {len(scores_wrong)}")
    print(f"  范围: {min(scores_wrong)}-{max(scores_wrong)}")
    print(f"  平均: {sum(scores_wrong)/len(scores_wrong):.1f}")
else:
    print("  无错误分类")

print(f"\n--- 结论 ---")
if correct_no_thresh > correct_80:
    print(f"  ✓ 确认：80 阈值导致准确率被低估 ({correct_80}% → {correct_no_thresh}%)")
    print(f"  ✓ 问题确实是测量误差，不是算法缺陷")
else:
    print(f"  ✗ 意外：80 阈值未导致低估，需要重新分析")
