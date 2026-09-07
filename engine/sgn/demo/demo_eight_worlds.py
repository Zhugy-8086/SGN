# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint 可视化演示：一个整数 = 八个世界（v2，清晰版）

叙事（自上而下，像望远镜逐级拉近）：
  顶部  —— 哲学句 + 两个数的定性对照（可压缩 vs 不可压缩）
  主体  —— 4 级变焦：8bit → 4bit → 2bit → 1bit 视图
          所有视图统一 8 列宽，行高 1→2→4→8（精度翻倍 = 网格翻倍）
          左列 DEMO_NUM：plasma 色 + 结构标注（处处有规律）
          右列 PI_BITS：灰色（处处噪声，无需标注）
  底部  —— 信息守恒结论：视图不创造信息，只暴露结构

核心哲学：表示是解读的函数——同一份 64bit 数据，换一种精度视图，
就是另一个世界。MSint 让同一份整数存储可以零成本切换到任意视图。

用法：
    cd engine/sgn/build
    python ../demo/demo_eight_worlds.py            # 弹窗（停留）
    python ../demo/demo_eight_worlds.py --save     # 保存 PNG 不弹窗

依赖：sgn.MultiScaleView（C++ 内核）+ matplotlib
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
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np

plt.rcParams["font.family"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei",
                               "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import sgn

# ============================================================================
# 数据
# ============================================================================
# 演示数：0x0123456789ABCDEF —— 每个精度视图下都有数学结构（自造）
# 对照数：π 的 IEEE 754 位模式 —— 每个精度视图下都是噪声
DEMO_NUM = 0x0123456789ABCDEF
PI_BITS = 0x400921FB54442D18          # float64(π) 的位模式
VIEWS = [8, 4, 2, 1]                  # 4 级变焦（从粗到细）

ROW_TITLES = {
    8: "世界 8-bit\n8 个字节\n每字节 +0x22 等差",
    4: "世界 4-bit\n16 个半字节\n0→F 计数器",
    2: "世界 2-bit\n32 个双位\n双计数器 (低2位, 高2位)",
    1: "世界 1-bit\n64 个位\n4 位二进制计数器",
}


def view_parts(value, n):
    """MultiScaleView 拆 parts（C++ 内核）。parts 低位在前 → 反转成高位在前。"""
    parts = sgn.MultiScaleView.interpret_levels(value, 64, [n])[n]
    return list(reversed(parts))


def view_grid(value, n):
    """视图 parts 排成 8 列网格（行数 = len/8 = 精度翻倍）。"""
    parts = view_parts(value, n)
    return np.array(parts, dtype=np.int64).reshape(len(parts) // 8, 8)


def draw_panel(ax, grid, n, show_text, cmap="plasma"):
    """画一个视图面板。show_text: 是否在格内显示数值（DEMO 侧 8/4bit 显示）。
    cmap: DEMO 侧 plasma（鲜艳=结构），PI 侧 gray（灰=无信息）。"""
    vmax = (1 << n) - 1
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax,
              interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_text:
        fs = 9 if n >= 4 else 7
        for r in range(grid.shape[0]):
            for c in range(8):
                p = int(grid[r, c])
                dark = p / vmax > 0.55
                ax.text(c, r, f"{p:x}" if n == 4 else f"{p:02x}", ha="center",
                        va="center", fontsize=fs, family="monospace",
                        fontweight="bold", color="white" if dark else "black")
    return vmax


# ============================================================================
# 主图
# ============================================================================
def main():
    save = "--save" in sys.argv
    fig = plt.figure(figsize=(17, 10.5))
    fig.patch.set_facecolor("#fafafa")

    # ---- 顶部：标题 + 哲学 + 两数对照 ----
    fig.suptitle("MSint 演示：一个整数 = 八个世界",
                 fontsize=20, fontweight="bold", color="#111111", y=0.965)
    fig.text(0.5, 0.925,
             "表示是解读的函数：同一份 64bit 整数，换一种精度视图，就是另一个世界。\n"
             "下面把同一个数按 8/4/2/1 bit 视图逐级放大（精度翻倍，网格翻倍）——",
             ha="center", fontsize=11.5, color="#444444")

    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 6.0, 6.0],
                          left=0.03, right=0.985, top=0.80, bottom=0.14,
                          wspace=0.05, hspace=0.0)
    hratio = [1, 2, 4, 8]
    col0 = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 0],
                                   height_ratios=hratio, hspace=0.10)
    col1 = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 1],
                                   height_ratios=hratio, hspace=0.10)
    col2 = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0, 2],
                                   height_ratios=hratio, hspace=0.10)

    # 列头
    col1_hdr = fig.add_axes([0.30, 0.825, 0.335, 0.05]); col1_hdr.axis("off")
    col1_hdr.text(0.0, 0.5, "DEMO_NUM  0x0123456789ABCDEF（可压缩：处处有结构）",
                  fontsize=12.5, fontweight="bold", color="#005500", va="center")
    col2_hdr = fig.add_axes([0.60, 0.825, 0.335, 0.05]); col2_hdr.axis("off")
    col2_hdr.text(0.0, 0.5, "PI_BITS  0x400921FB54442D18（不可压缩：处处噪声）",
                  fontsize=12.5, fontweight="bold", color="#880000", va="center")

    # 行标题（左缘，覆盖两列）+ 两个数据面板
    for i, n in enumerate(VIEWS):
        t = fig.add_subplot(col0[i])
        t.axis("off")
        t.text(0.02, 0.5, ROW_TITLES[n], fontsize=11.5, va="center", ha="left",
               color="#222222", linespacing=1.6)
        ax1 = fig.add_subplot(col1[i])
        ax2 = fig.add_subplot(col2[i])
        show = n >= 4
        draw_panel(ax1, view_grid(DEMO_NUM, n), n, show_text=show, cmap="plasma")
        draw_panel(ax2, view_grid(PI_BITS, n), n, show_text=False, cmap="gray")

    # 8bit 行的结构标注（DEMO 侧）
    ax8 = fig.add_subplot(col1[0])
    ax8.annotate("+0x22", xy=(7.5, 1.05), xytext=(5.2, 0.32),
                 fontsize=11, color="#006600", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#006600"))
    ax8.annotate("0x01 → 0xEF", xy=(1.5, -0.25), xytext=(0.5, -0.55),
                 fontsize=10, color="#444444", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#999999"))

    # 4bit 行的结构标注（DEMO 侧）
    ax4 = fig.add_subplot(col1[1])
    ax4.annotate("0 1 2 3 … F（每个半字节 +1）", xy=(7.2, 1.12),
                 xytext=(4.2, 0.42), fontsize=10, color="#006600",
                 arrowprops=dict(arrowstyle="->", color="#006600"))

    # 底部：信息守恒结论
    fig.text(0.5, 0.045,
             "结论：信息守恒——可压缩的数在任何精度视图下都可见结构，不可压缩的数在哪都是噪声。\n"
             "视图不创造信息，只暴露结构；而这 8 个世界来自同一份存储，MSint 可以在它们之间零成本切换。",
             ha="center", fontsize=11.5, color="#333333")

    if save:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "demo_eight_worlds.png")
        fig.savefig(out, dpi=130)
        print(f"saved: {out}")
        return 0
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())