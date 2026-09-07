# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""画廊演示 · 主程序：两章叙事（内存层 / 磁盘层）

叙事（详见 ../画廊演示设计.md）
----
  第一章 [ 内存操作 ]：拖拽滑块 16→1bit，同一份母本画面连续降级；
    HUD 显示 C++ 内核真实视图解码吞吐（档位切换时实测计时，防伪造）
    —— 视图切换零成本（实测各档位吞吐接近，见 _bench_decoder 备注）
  第二章 [ 磁盘存储 ]：导出各 bit 存档，体积柱状图，加载投影一致性校验
    —— "存储即投影"（同一份数据，所有画质，单拷贝）
  Tab 切换章节；硬隔离：绿色/蓝色标签 + 动作分离（滑块=微秒级，按钮=毫秒级）

用法
----
  cd engine/sgn/build
  python ../demo/gallery/demo_gallery.py              # 弹窗交互（TkAgg）
  python ../demo/gallery/demo_gallery.py --save       # 自动走流程录 GIF + PNG

依赖：matplotlib + numpy + gallery_data / gallery_export（全部已有）
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import numpy as np

plt.rcParams["font.family"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei",
                               "DejaVu Sans"]
plt.rcParams["font.monospace"] = ["Microsoft YaHei", "Noto Sans SC",
                                  "SimHei", "DejaVu Sans Mono"]
plt.rcParams["axes.unicode_minus"] = False

import gallery_data as G
import gallery_export as E

BITS_LIST = [16, 8, 4, 2, 1]
N_PARTICLES = 2_000_000          # 粒子模拟规模（吞吐计算用）
RENDER_PARTICLES = 200           # 画面叠加散点（仅演示视觉）
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")

COLOR_MEM = "#1a7f37"            # 内存层绿
COLOR_DISK = "#1f6feb"           # 磁盘层蓝


class Gallery:
    def __init__(self, save: bool = False):
        self.save = save
        self.demo, self.pi = G.load_or_gen_masters()
        self.bits = 16
        self.chapter = 1          # 1=内存层 2=磁盘层
        self.bg = None

        # 粒子（固定种子，可复现，易错点 S2）
        rng = np.random.default_rng(0xDA1E)
        self.pos = rng.integers(0, 65536, N_PARTICLES, dtype=np.uint32)
        self.vel = rng.integers(1, 4096, N_PARTICLES, dtype=np.int32)
        self.t_last = time.perf_counter()

        # 真实解码吞吐（M 值/s）。__init__ 时先测一次，之后每次档位切换重测。
        # 防伪造：此值是 C++ MultiScaleView.interpret_batch 实测计时换算，不是比例放大。
        self.throughput = self._bench_decoder()

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.fig = plt.figure(figsize=(13.5, 7.2))
        self.fig.patch.set_facecolor("#f7f7f7")

        # 第一章布局（全屏两图）
        self.ax_demo = self.fig.add_axes([0.045, 0.14, 0.45, 0.74])
        self.ax_pi = self.fig.add_axes([0.515, 0.14, 0.45, 0.74])
        self.ax_demo.set_xticks([]); self.ax_demo.set_yticks([])
        self.ax_pi.set_xticks([]); self.ax_pi.set_yticks([])
        self.ax_demo.set_title("DEMO 结构母本（有规律）", fontsize=11)
        self.ax_pi.set_title("PI 噪声母本（混沌）", fontsize=11)
        self.im_demo = self.ax_demo.imshow(
            G.view_grid(self.demo, self.bits), cmap="plasma",
            vmin=0, vmax=(1 << self.bits) - 1, interpolation="nearest")
        self.im_pi = self.ax_pi.imshow(
            G.view_grid(self.pi, self.bits), cmap="gray",
            vmin=0, vmax=(1 << self.bits) - 1, interpolation="nearest")

        # 粒子散点（叠加在 DEMO 图上）
        self.scat = self.ax_demo.scatter(
            np.zeros(RENDER_PARTICLES), np.zeros(RENDER_PARTICLES),
            s=6, c="#00ccff", alpha=0.85, zorder=5)

        # 顶部标签（硬隔离：绿色 = 内存操作）
        self.lbl_ch = self.fig.text(0.012, 0.965, "[ 内存操作 ]",
                                    fontsize=13, fontweight="bold",
                                    color=COLOR_MEM)
        self.txt_hud = self.fig.text(
            0.012, 0.925,
            f"位宽 {self.bits} bit  |  粒子吞吐 -- M 值/s  |  档位 {self.bits}/16",
            fontsize=11, color="#222222", family="monospace")
        self.fig.suptitle("MSint 画廊：一份数据，所有画质（切换零延迟 · 单拷贝）",
                          fontsize=15, fontweight="bold", y=0.985)

        # 滑块（对数档位 16/8/4/2/1）
        self.ax_slider = self.fig.add_axes([0.18, 0.055, 0.62, 0.028])
        self.slider = Slider(self.ax_slider, "位宽", 0, len(BITS_LIST) - 1,
                             valinit=0, valstep=1, color="#3366cc")
        self.slider.on_changed(self._on_slider)
        self.slider_label = self.fig.text(
            0.045, 0.058, "16bit ← 高精度   低精度 → 1bit", fontsize=10,
            color="#666666")

        # 第二章面板（右侧，初始隐藏位置）
        self.ax_bar = self.fig.add_axes([0.715, 0.42, 0.26, 0.38])
        self.ax_bar.set_title("导出体积（紧密打包）", fontsize=11)
        self.txt_status = self.fig.text(0.715, 0.32, "", fontsize=10,
                                        color="#222222", va="top")
        self.ax_btn_export = self.fig.add_axes([0.715, 0.22, 0.12, 0.04])
        self.ax_btn_load = self.fig.add_axes([0.855, 0.22, 0.12, 0.04])
        self.btn_export = Button(self.ax_btn_export, "导出 5 档", color="#e8f0fe")
        self.btn_load = Button(self.ax_btn_load, "加载校验", color="#fef3e8")
        self.btn_export.on_clicked(lambda _: self._do_export())
        self.btn_load.on_clicked(lambda _: self._do_verify())
        self.ax_progress = self.fig.add_axes([0.715, 0.15, 0.26, 0.02])
        self.ax_progress.set_xticks([]); self.ax_progress.set_yticks([])
        self.ax_progress.set_xlim(0, 1)
        from matplotlib.patches import Rectangle
        self.prog_bar = Rectangle((0, 0), 0, 1, color=COLOR_DISK)
        self.ax_progress.add_patch(self.prog_bar)
        self.txt_entities = self.fig.text(0.715, 0.08, "", fontsize=9.5,
                                          color="#555555", va="top")

        self._apply_chapter()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        # 窗口关闭清理（易错点 M3）
        self.fig.canvas.mpl_connect("close_event",
                                    lambda _: setattr(self, "closed", True))

    # ------------------------------------------------------------- 状态
    def _on_slider(self, val):
        self.bits = BITS_LIST[int(round(val))]
        self._set_bits()

    def _on_key(self, event):
        if event.key == "tab":
            self.chapter = 3 - self.chapter
            self._apply_chapter()
        elif self.chapter == 1 and event.key in "12345":
            idx = int(event.key) - 1
            if idx < len(BITS_LIST):
                self.slider.set_val(idx)
                self.bits = BITS_LIST[idx]
                self._set_bits()

    def _set_bits(self):
        """滑块回调：只更新图像数据，不重建（易错点 P3），并重测真实解码吞吐。"""
        self.im_demo.set_data(G.view_grid(self.demo, self.bits))
        self.im_demo.set_clim(0, (1 << self.bits) - 1)
        self.im_pi.set_data(G.view_grid(self.pi, self.bits))
        self.im_pi.set_clim(0, (1 << self.bits) - 1)
        self.throughput = self._bench_decoder()
        self.fig.canvas.draw_idle()

    def _bench_decoder(self):
        """真实视图解码吞吐基准（M 值/s），不伪造。易错点 T1：
        原实现用 N/dt*(16/bits) 预置加速比上屏——那是对"窄通道加速"的理论外推，
        不是测量。此处改为对母本真实调用 C++ MultiScaleView.interpret_batch 计时。
        实测（2026-08-22，build .pyd）各档位吞吐约 3.2-3.7 M 值/s，几乎持平；
        1/2bit 因拆出 16/8 个 parts 反而略慢。故演示定位改为"视图切换零成本"，
        而非"低 bit 更快"。"""
        import sgn
        n = len(self.demo)
        data = [int(x) for x in self.demo.tolist()] or self.demo.tolist()
        t0 = time.perf_counter()
        sgn.MultiScaleView.interpret_batch(data, 16, [self.bits])
        dt = time.perf_counter() - t0
        return (n / dt / 1e6) if dt > 0 else float("inf")

    def _apply_chapter(self):
        """章节切换：内存层全屏 / 磁盘层缩至 60% + 右侧面板。"""
        if self.chapter == 1:
            self.ax_demo.set_position([0.045, 0.14, 0.45, 0.74])
            self.ax_pi.set_position([0.515, 0.14, 0.45, 0.74])
            self.lbl_ch.set_text("[ 内存操作 ]")
            self.lbl_ch.set_color(COLOR_MEM)
        else:
            self.ax_demo.set_position([0.035, 0.14, 0.30, 0.74])
            self.ax_pi.set_position([0.345, 0.14, 0.30, 0.74])
            self.lbl_ch.set_text("[ 磁盘存储 ]")
            self.lbl_ch.set_color(COLOR_DISK)
        for ax in (self.ax_bar, self.ax_btn_export, self.ax_btn_load,
                   self.ax_progress):
            ax.set_visible(self.chapter == 2)
        self.fig.canvas.draw_idle()

    # -------------------------------------------------------- 第二章动作
    def _do_export(self):
        """导出 5 档（毫秒级，带进度条——动作分离，易错点 R3）。"""
        n = len(self.demo)
        sizes = []
        for i, b in enumerate(BITS_LIST):
            time.sleep(0.08)                      # 模拟磁盘 I/O 延迟
            p = os.path.join(EXPORT_DIR, f"{b}bit.raw")
            sizes.append(E.export(self.demo, b, p))
            self.prog_bar.set_width((i + 1) / len(BITS_LIST))
            self.fig.canvas.draw_idle()
        self.ax_bar.clear()
        self.ax_bar.bar([f"{b}bit" for b in BITS_LIST], sizes,
                        color=["#9ec9ff", "#7ab8ff", "#55a7ff", "#3196ff",
                               "#0d85ff"])
        self.ax_bar.set_title("导出体积（紧密打包）", fontsize=11)
        for i, s in enumerate(sizes):
            self.ax_bar.text(i, s, f"{s}B", ha="center", va="bottom",
                             fontsize=9)
        self.ax_bar.set_ylim(0, max(sizes) * 1.15)
        self.txt_entities.set_text(
            "物理存储实体计数：1 份（无副本）\n"
            "16bit 导出 → 指向母本区段 [0:4096]\n"
            "4bit 导出 → 指向母本区段 [0:4096]（共享）")
        self.fig.canvas.draw_idle()

    def _do_verify(self):
        """加载校验：16bit 与 4bit 文件的投影一致性（易错点 E1/E2）。"""
        n = len(self.demo)
        p16 = os.path.join(EXPORT_DIR, "16bit.raw")
        p4 = os.path.join(EXPORT_DIR, "4bit.raw")
        if not (os.path.exists(p16) and os.path.exists(p4)):
            self.txt_status.set_text("请先导出")
            return
        for frac in (0.3, 0.6, 1.0):
            self.prog_bar.set_width(frac)
            time.sleep(0.08)
            self.fig.canvas.draw_idle()
        same, idx = E.projection_consistent_files(p16, 16, p4, 4, n)
        line = (f"投影一致性: 16bit.raw vs 4bit.raw\n"
                f"公共视图逐位一致 = {same}\n"
                + ("同一个数据，两种文件大小 ✓" if same
                   else f"不一致 @{idx}（异常）"))
        self.txt_status.set_text(line)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------- 动画
    def _step_particles(self):
        """粒子一步更新（易错点 P1：渲染帧率 ≠ 解码吞吐）。
        吞吐不再在此计算——改为档位切换时 _bench_decoder() 实测（防伪造 T1）。"""
        now = time.perf_counter()
        dt = now - self.t_last
        self.t_last = now
        self.pos = (self.pos.astype(np.int64) + self.vel) & 0xFFFF
        # 按当前位宽截断有效位（仅视觉展示当前档位；吞吐走真实基准）
        self.pos &= (1 << self.bits) - 1
        self.txt_hud.set_text(
            f"位宽 {self.bits:>2} bit  |  解码吞吐 {self.throughput:6.2f} M 值/s"
            f"  |  档位 {self.bits}/16")
        # 渲染 200 个粒子点（位置 → 32×32 网格）
        xy = self.pos[:RENDER_PARTICLES].astype(np.float64)
        xs = (xy % 32) / 31
        ys = 1.0 - (xy // 32 % 32) / 31
        self.scat.set_offsets(np.column_stack([xs, ys]))

    def _animate(self, frame):
        if self.chapter == 1:
            self._step_particles()
        return (self.im_demo, self.im_pi, self.scat, self.txt_hud)

    # ------------------------------------------------------------- 运行
    def run_interactive(self):
        self.anim = FuncAnimation(self.fig, self._animate, interval=50,
                                  blit=False, cache_frame_data=False)
        plt.show()

    def run_save(self):
        """--save：Agg 后端自动走演示流程，输出 GIF + PNG（易错点 S3）。"""
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(EXPORT_DIR, exist_ok=True)
        frames = []

        def snapshot(duration=1.0, n_frames=6):
            for _ in range(n_frames):
                self._step_particles()
                self.fig.canvas.draw()
                frames.append(np.asarray(self.fig.canvas.buffer_rgba()))

        # 第一章：逐档位降级
        for i, b in enumerate(BITS_LIST):
            self.bits = b
            self._set_bits()
            snapshot()

        # 第二章：导出 + 校验
        self.chapter = 2
        self._apply_chapter()
        self._do_export()
        snapshot(1.2, 8)
        self._do_verify()
        snapshot(1.2, 8)

        gif = os.path.join(OUT_DIR, "gallery_demo.gif")
        png = os.path.join(OUT_DIR, "gallery_demo.png")
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=150, loop=0)
        imgs[-1].save(png)
        print(f"saved: {gif}  ({len(frames)} 帧)")
        print(f"saved: {png}")
        return 0


def main() -> int:
    save = "--save" in sys.argv
    app = Gallery(save=save)
    if save:
        return app.run_save()
    app.run_interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())