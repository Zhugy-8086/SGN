# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint 3D 演示：一个整数 = 六种解读（双魔方）

叙事
----
  开场：双魔方自动旋转一圈，展示全部六个面（每个面 = 一种精度解读）
  之后：用户拖拽魔方，亲手切换解读
  左魔方 DEMO_NUM（有结构：处处规律）；右魔方 PI_BITS（噪声：处处混沌）
  六个面 = 32/16/8/4/2/1-bit 解读，面等大、格数 2→64 逐面翻倍
  → 每格大小 = 精度粒度（低精度 = 大块马赛克，高精度 = 细颗粒）

用法
----
  cd engine/sgn/build
  python ../demo/demo_cube.py              # 弹窗：开场动画后拖拽旋转
  python ../demo/demo_cube.py --save       # 存 GIF（动画）+ PNG（定格）不弹窗

依赖：sgn.MultiScaleView + matplotlib（GIF 需 Pillow）
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

plt.rcParams["font.family"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei",
                               "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import sgn

DEMO_NUM = 0x0123456789ABCDEF
PI_BITS = 0x400921FB54442D18          # float64(π) 的位模式

# 六面 = 六种解读。面等大（边长 2），格数 = 64/位宽 → 每格大小 = 精度粒度
# (center, u 轴(列), v 轴(行))，格子角 = center + (j/cols-0.5)*2u + (i/rows-0.5)*2v
FACES = {
    32: ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    16: ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    8:  ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    4:  ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    2:  ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    1:  ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
}
FACE_LAYOUT = {32: (1, 2), 16: (2, 2), 8: (2, 4), 4: (4, 4), 2: (4, 8), 1: (8, 8)}


def view_parts(value, n):
    """MultiScaleView 拆 parts（低位在前 → 反转成高位在前，MSB first）。"""
    parts = sgn.MultiScaleView.interpret_levels(value, 64, [n])[n]
    return list(reversed(parts))


def draw_face(ax, value, n, center, u_ax, v_ax, cmap_name):
    """在一个面上画 n-bit 解读的格子（每格 = 一个 part，颜色 = 数值）。"""
    rows, cols = FACE_LAYOUT[n]
    grid = np.array(view_parts(value, n), dtype=np.int64).reshape(rows, cols)
    vmax = (1 << n) - 1
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(0, vmax)
    center = np.array(center, float)
    u_ax = np.array(u_ax, float)
    v_ax = np.array(v_ax, float)
    polys, colors = [], []
    for i in range(rows):
        for j in range(cols):
            corners = []
            for di, dj in ((0, 0), (0, 1), (1, 1), (1, 0)):
                u = (j + dj) / cols - 0.5
                v = (i + di) / rows - 0.5
                corners.append(center + 2 * u * u_ax + 2 * v * v_ax)
            polys.append(corners)
            colors.append(cmap(norm(int(grid[i, j]))))
    pc = Poly3DCollection(polys, facecolors=colors, edgecolors="#2c2c2c",
                          linewidths=0.5)
    ax.add_collection3d(pc)
    # 面标签（半透明底小字，提示这是哪种解读）
    label_off = 1.18 * (center / np.linalg.norm(center))
    ax.text3D(*(center + label_off), f"{n}-bit",
              fontsize=8.5, color="#444444", ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                        alpha=0.85))
    return grid, vmax


def build_cube(ax, value, cx, cmap_name):
    """在 x=cx 处构建一个魔方（六面六解读）。"""
    for n, (center, u_ax, v_ax) in FACES.items():
        c = (cx + center[0], center[1], center[2])
        draw_face(ax, value, n, c, u_ax, v_ax, cmap_name)


def main():
    save = "--save" in sys.argv
    fig = plt.figure(figsize=(13.5, 9.2))
    fig.patch.set_facecolor("#f7f7f7")
    fig.suptitle("一个整数 = 六种解读", fontsize=19, fontweight="bold", y=0.97)
    fig.text(0.5, 0.905,
             "同一份 64bit 整数，按 32/16/8/4/2/1-bit 精度解读，得到六个不同的面孔。\n"
             "面等大、格数翻倍——每格大小就是精度粒度。拖拽转动魔方，亲手切换解读。",
             ha="center", fontsize=10.5, color="#444444")

    ax = fig.add_subplot(111, projection="3d", position=[0.03, 0.10, 0.94, 0.74])
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    for lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        lim(-3.2, 3.2)
    ax.view_init(elev=22, azim=30)

    build_cube(ax, DEMO_NUM, -1.7, "plasma")
    build_cube(ax, PI_BITS, 1.7, "gray")

    fig.text(0.275, 0.125, "DEMO_NUM  0x0123456789ABCDEF\n（可压缩：处处有规律）",
             fontsize=10.5, ha="center", color="#006600", fontweight="bold")
    fig.text(0.725, 0.125, "PI_BITS  0x400921FB54442D18\n（不可压缩：处处噪声）",
             fontsize=10.5, ha="center", color="#880000", fontweight="bold")
    fig.text(0.5, 0.030,
             "结论：结构来自数据，不来自解读——可压缩的数在任何精度下都有规律，噪声在哪都是噪声。\n"
             "MSint 让同一份整数存储，在任何解读之间零成本切换。",
             ha="center", fontsize=10.5, color="#333333")

    total = 110
    anim = None

    def update(frame):
        ax.view_init(elev=22, azim=30 + 360 * frame / total)
        return []

    if save:
        anim = FuncAnimation(fig, update, frames=total, interval=70, blit=False)
        outdir = os.path.dirname(os.path.abspath(__file__))
        gif = os.path.join(outdir, "demo_cube.gif")
        anim.save(gif, writer="pillow", fps=14, dpi=88)
        print(f"saved: {gif}")
        ax.view_init(elev=22, azim=45)
        fig.canvas.draw_idle()
        png = os.path.join(outdir, "demo_cube.png")
        fig.savefig(png, dpi=110)
        print(f"saved: {png}")
        return 0

    # 弹窗：开场自动转一圈 → 结束后可拖拽
    anim = FuncAnimation(fig, update, frames=total, interval=70, blit=False)
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())