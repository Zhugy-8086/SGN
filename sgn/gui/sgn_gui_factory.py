#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sgn_gui_factory.py — 函数工厂 GUI 入口（支持命令行参数）

用法:
    # 直接运行（无参数，使用默认配置）
    python gui/sgn_gui_factory.py

    # 通过命令行控制启动配置
    python gui/sgn_gui_factory.py --grid 16 --char A --label A
    python gui/sgn_gui_factory.py --angle 30 --offset-x 2 --scale 0.8
    python gui/sgn_gui_factory.py --variants 200 --export dataset.csv
    python gui/sgn_gui_factory.py --dataset my_data.json --auto-train --train-steps 1000

    # 编程式调用
    python -c "from gui.sgn_gui_factory import run_factory; run_factory()"
    或在 main.py 交互菜单中按 [f] 或 [e] 启动 GUI

职责：
  - 将父目录加入 sys.path（使 gui/ 子模块能导入 SGN 核心）
  - 解析命令行参数，传递给 GUIFactory 控制启动配置
  - 提供 run_factory(core=None, cli_args=None) 统一入口
"""
from __future__ import annotations

import sys
import os
import argparse
from typing import Optional, Dict, Any

# ============================================================
# 将父目录加入 sys.path，使从 gui/ 子目录能导入 SGN 模块
# ============================================================
_gui_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_gui_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建函数工厂 GUI 的命令行参数解析器

    所有参数均为可选，用于调整 GUI 启动时的初始状态（非新建功能）。
    分为五组：画布/字符、变换参数、变体生成、训练控制、数据集。
    """
    parser = argparse.ArgumentParser(
        prog="sgn_gui_factory",
        description="SGN-Lite 函数工厂 GUI — 支持命令行参数控制启动配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 加载字符 'A' 到 16×16 画布
  python gui/sgn_gui_factory.py --char A --grid 16

  # 设置变换参数并生成 200 个变体
  python gui/sgn_gui_factory.py --angle 45 --variants 200 --export out.csv

  # 加载数据集并自动训练 1000 步
  python gui/sgn_gui_factory.py --dataset data.json --auto-train --train-steps 1000

  # 指定噪声类型和概率
  python gui/sgn_gui_factory.py --noise gaussian --noise-prob 0.2
""",
    )

    # ---- 画布与字符 ----
    g_canvas = parser.add_argument_group("画布与字符", "控制画布初始状态")
    g_canvas.add_argument(
        "--grid", type=int, default=None, metavar="N",
        help="初始网格大小 (8/16/32/64)，默认 8",
    )
    g_canvas.add_argument(
        "--char", type=str, default=None, metavar="C",
        help="启动时加载到画布的字符 (如 A, 5, @)",
    )
    g_canvas.add_argument(
        "--label", type=str, default=None, metavar="L",
        help="初始训练标签 (单个字符，如 A)",
    )

    # ---- 变换参数 ----
    g_transform = parser.add_argument_group("变换参数", "设置 TransformEngine 初始值")
    g_transform.add_argument(
        "--angle", type=float, default=None, metavar="DEG",
        help="初始旋转角度 (-180~180)，默认 0",
    )
    g_transform.add_argument(
        "--offset-x", type=int, default=None, metavar="N",
        help="初始 X 偏移量，默认 0",
    )
    g_transform.add_argument(
        "--offset-y", type=int, default=None, metavar="N",
        help="初始 Y 偏移量，默认 0",
    )
    g_transform.add_argument(
        "--scale", type=float, default=None, metavar="F",
        help="初始缩放因子 (0.5~1.5)，默认 1.0",
    )

    # ---- 变体生成 ----
    g_variant = parser.add_argument_group("变体生成", "批量生成变体并导出")
    g_variant.add_argument(
        "--variants", type=int, default=None, metavar="N",
        help="启动时生成 N 个变体到内存 (默认不生成)",
    )
    g_variant.add_argument(
        "--export", type=str, default=None, metavar="FILE",
        help="将生成的变体导出为 CSV/JSON 文件路径",
    )
    g_variant.add_argument(
        "--variant-angle", type=float, default=15.0, metavar="DEG",
        help="变体随机旋转范围 ±DEG (默认 15)",
    )
    g_variant.add_argument(
        "--variant-offset", type=int, default=6, metavar="N",
        help="变体随机偏移范围 ±N (默认 6)",
    )
    g_variant.add_argument(
        "--variant-scale", type=float, default=0.2, metavar="F",
        help="变体随机缩放范围 ±F (默认 0.2)",
    )

    # ---- 训练控制 ----
    g_train = parser.add_argument_group("训练控制", "自动训练设置")
    g_train.add_argument(
        "--auto-train", action="store_true",
        help="启动后自动开始训练",
    )
    g_train.add_argument(
        "--train-steps", type=int, default=None, metavar="N",
        help="训练目标步数 (默认使用滑块值)",
    )
    g_train.add_argument(
        "--train-speed", type=int, default=None, metavar="N",
        help="每帧训练步数 (1~50，默认 5)",
    )

    # ---- 噪声 ----
    g_noise = parser.add_argument_group("噪声", "训练噪声配置")
    g_noise.add_argument(
        "--noise", type=str, default=None, metavar="TYPE",
        choices=["composite", "gaussian", "salt_pepper", "block", "none"],
        help="噪声类型 (composite/gaussian/salt_pepper/block/none)",
    )
    g_noise.add_argument(
        "--noise-prob", type=float, default=None, metavar="P",
        help="噪声翻转概率 (0.0~1.0，默认 0.15)",
    )

    # ---- 数据集 ----
    g_dataset = parser.add_argument_group("数据集", "加载/保存训练集")
    g_dataset.add_argument(
        "--dataset", type=str, default=None, metavar="FILE",
        help="启动时加载的自定义训练集 JSON 文件路径",
    )

    return parser


def parse_args(argv=None) -> Dict[str, Any]:
    """解析命令行参数并返回字典

    Args:
        argv: 命令行参数列表，None 时使用 sys.argv[1:]

    Returns:
        参数字典，仅包含用户实际指定的参数（非默认值）
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = {}
    # 过滤：auto_train 仅在 True 时保留，variant_* 仅在非默认值时保留，其余非 None 时保留
    defaults = {"variant_angle": 15.0, "variant_offset": 6, "variant_scale": 0.2}
    for k, v in vars(args).items():
        if k == "auto_train":
            if v:
                result[k] = v
        elif k in defaults:
            if v != defaults[k]:
                result[k] = v
        elif v is not None:
            result[k] = v
    return result


def run_factory(core=None, cli_args: Optional[Dict[str, Any]] = None):
    """启动函数工厂 GUI

    Args:
        core: 可选，复用已有的 SGNCore 实例
        cli_args: 可选，命令行参数字典。为 None 时自动解析 sys.argv
                  传入空字典 {} 时跳过命令行解析（编程式调用）
    """
    from gui.bridge import _ensure_sgn
    if not _ensure_sgn():
        print("[错误] SGN 模块未就绪，请确保在项目目录中运行")
        return

    # 解析命令行参数：传入 None 时自动解析，传入 {} 时跳过
    if cli_args is None:
        cli_args = parse_args()

    if cli_args:
        print("[函数工厂] 命令行参数:")
        for k, v in cli_args.items():
            print(f"  --{k.replace('_', '-')} = {v}")

    print("[函数工厂] 启动 GUI...")
    print("  快捷键: 空格=训练/暂停  T=测试  R=随机变换  C=清空  ESC=退出")

    from gui.factory import GUIFactory
    app = GUIFactory(core=core, cli_args=cli_args)
    app.run()


if __name__ == "__main__":
    run_factory()
