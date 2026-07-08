#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sgn_gui_factory.py — 函数工厂 GUI 入口

用法:
    python gui/sgn_gui_factory.py              # 直接运行
    python -c "from gui.sgn_gui_factory import run_factory; run_factory()"
    或在 main.py 交互菜单中按 [f] 或 [e] 启动 GUI

职责：
  - 将父目录加入 sys.path（使 gui/ 子模块能导入 SGN 核心）
  - 提供 run_factory(core=None) 统一入口
"""
from __future__ import annotations

import sys
import os

# ============================================================
# 将父目录加入 sys.path，使从 gui/ 子目录能导入 SGN 模块
# ============================================================
_gui_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_gui_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def run_factory(core=None):
    """启动函数工厂 GUI

    Args:
        core: 可选，复用已有的 SGNCore 实例
    """
    from gui.bridge import _ensure_sgn
    if not _ensure_sgn():
        print("[错误] SGN 模块未就绪，请确保在项目目录中运行")
        return
    print("[函数工厂] 启动 GUI...")
    print("  快捷键: 空格=训练/暂停  T=测试  R=随机变换  C=清空  ESC=退出")
    from gui.factory import GUIFactory
    app = GUIFactory(core=core)
    app.run()


if __name__ == "__main__":
    run_factory()
