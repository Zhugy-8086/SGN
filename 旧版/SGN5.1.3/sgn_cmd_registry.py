#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 [zhugy-8086]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGN-Lite v5.0 命令注册聚合模块

集中导入所有拆分后的回调函数，并统一注册到 CommandRegistry。
由 main.py 或 sgn_interactive 最顶部显式导入触发注册。

【设计意图】
拆分后命令回调分散在多个模块中，此模块作为聚合层，避免：
1. 各模块自行注册导致的顺序混乱
2. 循环导入风险（此模块只导入函数，不导入会反向依赖的模块）
"""

from __future__ import annotations

from sgn_commands import Command, CommandRegistry


# ============================================================
# 命令回调定义
# ============================================================

def _cmd_continue(core=None, **kwargs):
    """[Enter] 继续训练"""
    return "continue"


def _cmd_quit(core=None, **kwargs):
    """[q] 退出"""
    return "quit"


def _cmd_reset(core=None, **kwargs):
    """[r] 重置网络"""
    from sgn_utils import C
    try:
        confirm = input(f"  {C.RED}确认重置? 所有进度将丢失 [y/N]: {C.RST}").strip().lower()
        if confirm in ("y", "yes"):
            return "reset"
        else:
            print(f"  {C.info('ℹ')} 取消重置")
    except (EOFError, KeyboardInterrupt):
        pass
    return None


def _cmd_auto(core=None, **kwargs):
    """[a] 切换到自动模式（训练期间可用）"""
    from sgn_utils import C
    try:
        d = input(f"  延时ms{C.DIM}[50]{C.RST}: ").strip()
        delay = int(d) if d else 50
        return ("auto", delay)
    except (EOFError, KeyboardInterrupt):
        return "quit"


def _cmd_save(core, **kwargs):
    """[o] 保存模型"""
    from sgn_persist import save_model
    from sgn_utils import C
    try:
        path = input(f"  保存路径{C.DIM}[sgn_model.json]{C.RST}: ").strip()
        save_model(core, path if path else None)
    except (EOFError, KeyboardInterrupt):
        pass


def _cmd_load(core, **kwargs):
    """[d] 加载模型"""
    from sgn_persist import load_model
    try:
        path = input("  加载路径: ").strip()
        if path:
            load_model(core, path)
    except (EOFError, KeyboardInterrupt):
        pass


def _cmd_mode_switch(core=None, **kwargs):
    """[k] 切换运行模式"""
    from sgn_config import ConfigRegistry, CONFIG
    from sgn_utils import C
    modes = {"full": "全记载", "compact": "精简", "blackbox": "黑箱"}
    current = CONFIG.get("MODE", "full")
    print(f"\n  当前模式: {C.val(modes.get(current, current))}")
    print(f"  {C.CYN}[1]{C.RST} 全记载  {C.CYN}[2]{C.RST} 精简  {C.CYN}[3]{C.RST} 黑箱")
    try:
        m = input("  选择: ").strip()
        new_mode = None
        if m == "1":
            new_mode = "full"
        elif m == "2":
            new_mode = "compact"
        elif m == "3":
            new_mode = "blackbox"
        if new_mode is not None:
            ok, msg = ConfigRegistry.set("MODE", new_mode)
            if ok:
                print(f"  {C.ok('✓')} 模式切换: {C.val(CONFIG['MODE'])}")
            else:
                print(f"  {C.err('✗')} 模式切换失败: {msg}")
    except (EOFError, KeyboardInterrupt):
        pass


def _cmd_inference(core, **kwargs):
    """[i] 推理测试"""
    from sgn_test import do_inference
    do_inference(core, source=kwargs.get("source"))


def _cmd_batch_test(core, **kwargs):
    """[t] 批量测试"""
    from sgn_test import do_batch_test
    test_samples = kwargs.get("test_samples") or kwargs.get("samples")
    do_batch_test(core, test_samples=test_samples)


def _cmd_stats(core, **kwargs):
    """[s] 统计信息"""
    from sgn_visual import do_stats
    do_stats(core)


def _cmd_gauge(core, **kwargs):
    """[g] 仪表盘"""
    from sgn_visual import do_gauge
    do_gauge(core)


def _cmd_visualize(core, **kwargs):
    """[u] 模板可视化"""
    from sgn_visual import do_visualize
    do_visualize(core)


def _cmd_heatmap(core, **kwargs):
    """[m] 热力图"""
    from sgn_visual import do_heatmap
    do_heatmap(core)


def _cmd_confusion(core, **kwargs):
    """[c] 混淆矩阵"""
    from sgn_test import do_confusion
    test_samples = kwargs.get("test_samples") or kwargs.get("samples")
    do_confusion(core, test_samples=test_samples)


def _cmd_noise_test(core, **kwargs):
    """[n] 噪声测试"""
    from sgn_test import do_noise_test
    test_samples = kwargs.get("test_samples") or kwargs.get("samples")
    source = kwargs.get("source")
    do_noise_test(core, test_samples=test_samples, source=source)


def _cmd_plot(core, **kwargs):
    """[p] 学习曲线"""
    from sgn_report import do_plot
    do_plot(core)


def _cmd_export(core, **kwargs):
    """[x] 导出报告"""
    from sgn_report import do_export
    do_export(core)


def _cmd_why_accuracy(core=None, **kwargs):
    """[w] 成功率疑问"""
    from sgn_help import do_why_accuracy
    do_why_accuracy()


def _cmd_help(core=None, **kwargs):
    """[h] 帮助手册"""
    from sgn_help import do_help
    do_help()


def _cmd_factory(core=None, **kwargs):
    """[f] 函数工厂 GUI"""
    try:
        from gui.sgn_gui_factory import run_factory
        run_factory(core=core)
    except ImportError as e:
        from sgn_utils import C
        print(f"  {C.err('✗')} 函数工厂模块未加载: {e}")
        print(f"  {C.DIM}  请确保 pygame 已安装: pip install pygame-ce{C.RST}")


def _cmd_back(core=None, **kwargs):
    """[b] 返回控制面板"""
    return "back"


def _cmd_cross_validate(core=None, **kwargs):
    """[v] 交叉验证"""
    from sgn_config import CONFIG
    from sgn_input import create_default_source
    from sgn_utils import run_cross_validation, C

    folds = CONFIG.get("CROSS_VALIDATE_FOLDS", 0)
    if folds <= 0:
        print(f"  {C.YEL}⚠ 交叉验证未启用，请在控制面板设置 CROSS_VALIDATE_FOLDS{C.RST}")
        return

    source = kwargs.get("source")
    if source is None or not hasattr(source, 'generate_batch'):
        source = create_default_source()
        print(f"  {C.YEL}⚠ 未提供有效 source，回退到默认内置字符{C.RST}")
    max_step = CONFIG.get("MAX_ITERATIONS", 640)
    results = run_cross_validation(source, max_step)
    return results


# ============================================================
# 统一注册函数
# ============================================================

def register_all_commands():
    """注册所有默认命令到 CommandRegistry

    由 main.py 或 sgn_interactive 在启动时显式调用。
    """
    defaults = [
        Command("",   "继续",    "训练",  _cmd_continue,    requires_trained=False, order=0),
        Command("v",  "交叉验证", "测试",  _cmd_cross_validate, requires_trained=False, order=4),
        Command("i",  "推理",    "测试",  _cmd_inference,   requires_trained=True,  order=0),
        Command("t",  "测试",    "测试",  _cmd_batch_test,  requires_trained=True,  order=1),
        Command("s",  "统计",    "测试",  _cmd_stats,       requires_trained=False, order=2),
        Command("g",  "仪表盘",  "测试",  _cmd_gauge,       requires_trained=False, order=3),
        Command("f",  "函数工厂", "系统",  _cmd_factory,     requires_trained=False, order=7),
        Command("c",  "混淆",    "可视化", _cmd_confusion,   requires_trained=True,  order=0),
        Command("u",  "可视化",  "可视化", _cmd_visualize,   requires_trained=True,  order=1),
        Command("m",  "热力图",  "可视化", _cmd_heatmap,     requires_trained=True,  order=2),
        Command("n",  "噪声",    "可视化", _cmd_noise_test,  requires_trained=True,  order=3),
        Command("p",  "曲线",    "可视化", _cmd_plot,        requires_trained=False, order=4),
        Command("x",  "导出",    "系统",  _cmd_export,      requires_trained=False, order=0),
        Command("w",  "成功率",  "系统",  _cmd_why_accuracy, requires_trained=False, order=1),
        Command("o",  "保存",    "系统",  _cmd_save,        requires_trained=False, order=2),
        Command("d",  "加载",    "系统",  _cmd_load,        requires_trained=False, order=3),
        Command("k",  "模式",    "系统",  _cmd_mode_switch, requires_trained=False, order=4),
        Command("h",  "帮助",    "系统",  _cmd_help,        requires_trained=False, order=5),
        Command("b",  "返回面板", "系统",  _cmd_back,       requires_trained=False, order=6),
        Command("a",  "自动",    "训练",  _cmd_auto,        requires_trained=False, order=1),
        Command("r",  "重置",    "训练",  _cmd_reset,       requires_trained=False, order=2),
        Command("q",  "退出",    "系统",  _cmd_quit,        requires_trained=False, order=99),
    ]
    for cmd in defaults:
        try:
            CommandRegistry.register(cmd, force=True)
        except KeyError:
            pass  # 重复注册时静默


# 模块导入时自动注册（向后兼容）
register_all_commands()
