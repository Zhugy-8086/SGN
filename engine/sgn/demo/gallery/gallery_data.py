# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""画廊演示 · 数据层：母本生成与视图提取

职责
----
1. 生成两块 int16 母本（4096 值 = 64×64 图像），缓存为 .npy 到 assets/
   - DEMO 渐变图（处处有规律：平滑渐变 + 正弦波，可压缩）
   - PI 噪声图（处处混沌：固定种子伪随机，不可压缩）
2. view(master, bits) —— 视图提取，纯位运算（取高 N 位），零副本
3. check_kernel_consistency() —— 与 sgn.MultiScaleView.interpret_batch 交叉验证，
   证明 numpy 位操作实现与内核语义一致

视图语义（全项目统一，见文件设计.md 易错点 V5）
----
   master 以 uint16 位模式存储（值域 0..65535），
   view(bits) = 取高 bits 位：master >> (16 - bits)，值域 [0, 2^bits)
   16→8/4/2/1 全部整除，无舍入（round 语义仅在非整除档位需要，此处不涉及）

用法
----
   cd engine/sgn/build
   python ../demo/gallery/gallery_data.py        # 生成 + 统计 + 内核一致性验证
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

# Windows 管道/重定向统一 UTF-8（避免 GBK 混编码，易错点 S1）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DEMO_PATH = os.path.join(_ASSETS, "demo_master.npy")
PI_PATH = os.path.join(_ASSETS, "pi_master.npy")

N = 4096                      # 母本长度（= 64×64 图像）
ROWS = COLS = 64
SEED = 0x20260820             # 固定种子：可复现（易错点 S2）
BITS_VALID = (1, 2, 4, 8, 16)


def gen_demo_master() -> np.ndarray:
    """DEMO 母本：同心圆环（降位宽时仍保留结构）。uint16[4096]。

    设计目标：降位宽后每个档位都有可辨识的图案，不会像纯渐变那样在1-bit变成两色块。
    圆环 = 多尺度结构：16bit 看到渐变，8bit 看到色阶，4bit 看到16级环带，
    2bit 看到4级环带，1bit 看到黑白交替环（仍可辨识为"同心圆"）。
    """
    yy, xx = np.mgrid[0:ROWS, 0:COLS].astype(np.float64)
    cx, cy = (COLS - 1) / 2.0, (ROWS - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # 径向频率：从中心到边缘约8个环（32像素半径 → 约8个环带）
    v = 0.5 + 0.5 * np.sin(r / (COLS / 2.0) * np.pi * 8)
    # 叠加渐变：左→右微弱渐变，增加层次感
    v = v * 0.85 + 0.15 * (xx / COLS)
    v = np.clip(v, 0.0, 1.0)
    return np.round(v * 65535.0).astype(np.uint16).reshape(-1)


def gen_pi_master() -> np.ndarray:
    """PI 噪声母本：处处混沌（固定种子伪随机）。uint16[1024]。
    命名沿用 demo_cube 的 PI_BITS 传统（π 的位模式 = 噪声）。"""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 65536, size=N, dtype=np.uint16)


def load_or_gen_masters() -> tuple[np.ndarray, np.ndarray]:
    """加载缓存母本；不存在则生成并保存（可复现，删除会自动重建）。"""
    os.makedirs(_ASSETS, exist_ok=True)
    if os.path.exists(DEMO_PATH) and os.path.exists(PI_PATH):
        return (np.load(DEMO_PATH), np.load(PI_PATH))
    demo = gen_demo_master()
    pi = gen_pi_master()
    np.save(DEMO_PATH, demo)
    np.save(PI_PATH, pi)
    return (demo, pi)


def view(master: np.ndarray, bits: int) -> np.ndarray:
    """视图提取：取高 bits 位（纯位运算，零副本）。返回 uint16 视图值。

    易错点 V1：此处不复制母本——返回的是移位后的新数组（位操作产物），
    母本本身不变；多视图共享母本这一事实由调用方持有母本引用体现。
    """
    assert bits in BITS_VALID, f"无效位宽 {bits}，仅支持 {BITS_VALID}"
    return (master.astype(np.uint32) >> (16 - bits)).astype(np.uint16)


def view_grid(master: np.ndarray, bits: int) -> np.ndarray:
    """视图值重排为 32×32 网格（渲染用）。"""
    return view(master, bits).reshape(ROWS, COLS)


def check_kernel_consistency(master: np.ndarray, bits: int) -> tuple[bool, str]:
    """与 sgn.MultiScaleView.interpret_batch 交叉验证。

    interpret_batch 语义 = 每值拆成 16/bits 个 parts（低位在前），
    reversed 后每行第一个 = 该值的最高 bits 位 part = 本模块 view() 的
    "取高 N 位"语义。两者应逐位一致，证明 numpy 位操作与内核一致。
    """
    try:
        import sgn
    except ImportError:
        return (False, "sgn 不可用，跳过内核验证")
    parts = sgn.MultiScaleView.interpret_batch(
        [int(x) for x in master.tolist()], 16, [bits])[bits]
    kernel_hi = np.array([list(reversed(p)) for p in parts], dtype=np.uint16)
    numpy_hi = view(master, bits)
    if np.array_equal(kernel_hi[:, 0], numpy_hi):
        return (True, f"bits={bits}: numpy 位操作 == sgn 内核最高位 part（逐位一致）")
    return (False, f"bits={bits}: 不一致！kernel={kernel_hi[:, 0].shape} numpy={numpy_hi.shape}")


def _main() -> int:
    demo, pi = load_or_gen_masters()
    print(f"DEMO 母本: {os.path.basename(DEMO_PATH)}  {demo.nbytes} B")
    print(f"PI   母本: {os.path.basename(PI_PATH)}  {pi.nbytes} B")
    for name, m in (("DEMO", demo), ("PI", pi)):
        print(f"  {name}: min={m.min()} max={m.max()} "
              f"unique={len(np.unique(m))}")
    print("视图统计（高 N 位值域上限）：")
    for b in BITS_VALID:
        v = view(m, b) if False else view(demo, b)
        print(f"  bits={b:>2}: 值域 [0,{2**b-1:<5}]  unique={len(np.unique(v))}")
    print("内核一致性验证：")
    ok_all = True
    for b in BITS_VALID:
        ok, msg = check_kernel_consistency(demo, b)
        ok_all = ok_all and ok
        print(f"  {msg}")
    for b in BITS_VALID:
        ok, msg = check_kernel_consistency(pi, b)
        ok_all = ok_all and ok
        print(f"  {msg}")
    print("验证结论:", "全部通过" if ok_all else "存在不一致！")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(_main())