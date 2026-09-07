# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""单步训练 timing 分析 — 定位 WEF+Triple SBE 训练的真正瓶颈

测量 conv2 层（B=32, C_in=32, C_out=64, 3x3）各部分耗时：
  前向: im2col + triple-rescale + 3x sbe_matmul + merge + x@w_eps
  反向: conv_transpose2d (grad_x) + weight grad
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "refs")))
# 2026-08-16 legacy 独立：删除指向 legacy/traditional/baseline/common 的死路径插入
#（该脚本实际仅依赖 refs/sbe_conv2d，common 从未被 import）

import numpy as np
import torch
torch.set_num_threads(8)

from sbe_conv2d import (
    sbe_matmul, sbe_matmul_wef_triple, quantize_weight_sbe,
    compute_w_epsilon, rescale_to_triple_int8_sbe,
)
import torch.nn.functional as F

# 模拟 conv2: B=32, C_in=32, H=W=15, C_out=64, 3x3
B, C_in, H, W = 32, 32, 15, 15
C_out, kh, kw = 64, 3, 3
stride, padding = 1, 1
K = C_in * kh * kw  # 288

x = torch.randn(B, C_in, H, W)
weight = torch.randn(C_out, C_in, kh, kw) * 0.1
bias = torch.zeros(C_out)

# im2col
x_col = F.unfold(x, kernel_size=(kh, kw), stride=stride, padding=padding)
x_col_2d = x_col.permute(0, 2, 1).reshape(-1, K)
m = x_col_2d.shape[0]  # 32*15*15=7200
n = C_out
groups = C_in
k_block = kh * kw

x_np = x_col_2d.detach().contiguous().numpy().astype(np.float32)
w_np = weight.permute(1, 2, 3, 0).detach().contiguous().numpy().astype(np.float32).reshape(K, n)
w_blocks = quantize_weight_sbe(w_np, groups, k_block, m, K, n)
w_eps = compute_w_epsilon(w_np, w_blocks, groups, k_block, K, n)

N = 20

def bench(name, fn):
    # warmup
    for _ in range(3):
        fn()
    t0 = time.time()
    for _ in range(N):
        fn()
    elapsed = (time.time() - t0) / N * 1000
    print(f"  {name:30s}: {elapsed:7.2f} ms")
    return elapsed

print(f"=== conv2 step timing (B={B}, m={m}, K={K}, n={n}, groups={groups}, k_block={k_block}) ===")
print(f"  torch threads: {torch.get_num_threads()}, cpu_count: {os.cpu_count()}")

# 前向各部分
print("\n--- Forward ---")
t_im2col = bench("im2col (unfold+reshape)", lambda: F.unfold(x, kernel_size=(kh, kw), stride=stride, padding=padding).permute(0, 2, 1).reshape(-1, K))
t_triple = bench("triple-int8 rescale", lambda: rescale_to_triple_int8_sbe(x_np))

Ch, Cm, Cl, sc = rescale_to_triple_int8_sbe(x_np)
t_3matmul = bench("3x sbe_matmul (C high/mid/low)", lambda: (sbe_matmul(Ch, w_blocks, groups, k_block, m, K, n), sbe_matmul(Cm, w_blocks, groups, k_block, m, K, n), sbe_matmul(Cl, w_blocks, groups, k_block, m, K, n)))

yh = sbe_matmul(Ch, w_blocks, groups, k_block, m, K, n)
ym = sbe_matmul(Cm, w_blocks, groups, k_block, m, K, n)
yl = sbe_matmul(Cl, w_blocks, groups, k_block, m, K, n)
def do_merge():
    y = np.empty(yh.shape, dtype=np.float64)
    np.multiply(yh, 65536.0, out=y, casting="unsafe")
    y += ym * 256.0
    y += yl
    y *= float(sc) / 65536.0
    return y
t_merge = bench("merge (float64)", do_merge)
t_weps = bench("x @ w_epsilon (numpy BLAS)", lambda: x_np @ w_eps)
t_fwd_total = bench("WEF+Triple total forward", lambda: sbe_matmul_wef_triple(x_np, w_blocks, w_eps, groups, k_block, m, K, n))

# 反向各部分
print("\n--- Backward (torch float32) ---")
grad_output_4d = torch.randn(B, C_out, H, W)
t_gx = bench("conv_transpose2d (grad_x)", lambda: F.conv_transpose2d(grad_output_4d, weight, stride=stride, padding=padding))

# weight grad: unfold(x) @ grad_output_2d
x_col_2d_g = x_col_2d.detach().clone()
grad_out_2d = grad_output_4d.permute(0, 2, 3, 1).reshape(-1, n)
t_gw = bench("weight grad (unfold@grad_out)", lambda: x_col_2d_g.T @ grad_out_2d)

# 汇总
fwd_sum = t_im2col + t_triple + t_3matmul + t_merge + t_weps
print(f"\n--- Summary (per step, conv2 only) ---")
print(f"  Forward parts sum:  {fwd_sum:7.2f} ms")
print(f"  Forward total:      {t_fwd_total:7.2f} ms")
print(f"  Backward (gx+gw):   {t_gx + t_gw:7.2f} ms")
print(f"  Step estimate:      {t_fwd_total + t_gx + t_gw:7.2f} ms")
print(f"  x1407 steps/epoch:  {(t_fwd_total + t_gx + t_gw) * 1407 / 1000:7.1f} s (conv2 only)")
print(f"\n  Note: conv1/conv3/fc1/fc2 layers add ~50-100% more time")
