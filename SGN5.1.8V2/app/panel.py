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
"""SGN-Lite v5.0 控制面板模块 —— 从 sgn_interactive 拆分

包含：控制面板主菜单、分类子菜单、扩展功能菜单、高级选项菜单
"""

from __future__ import annotations

import sys
from typing import Optional

from app.presets import PRESETS, apply_preset


def homepage(core=None):
    """主页 - 简化入口，模式选择 + 快速启动

    返回值:
      True  -> 开始训练
      None  -> 退出程序
      "back" -> 返回（不应发生，主页是最顶层）
    """
    from engine.config import ConfigRegistry, CONFIG, is_config_modified
    from engine.utils import C, box, hr

    while True:
        box("SGN-Lite v5.1.8")

        # 显示当前网络状态
        if core:
            st = core.get_state() if hasattr(core, 'get_state') else {}
            neurons = st.get('neurons', len(core.N)) if hasattr(core, 'N') else '?'
            templates = st.get('templates', len(core.templates)) if hasattr(core, 'templates') else '?'
            history_len = len(core.history) if hasattr(core, 'history') else '?'
            print(f"  当前网络: {C.val(neurons)}神经元 / {C.val(templates)}模板 / {C.val(history_len)}步")
        else:
            print(f"  当前网络: {C.val('未初始化')}")

        if is_config_modified():
            print(f"  {C.YEL}! 配置已修改{C.RST}")

        hr(46)

        # 模式选择
        print(f"  {C.BOLD}选择训练模式:{C.RST}")
        print(f"  {C.CYN}[1]{C.RST} 快速验证  {C.DIM}小网络(64), 500步, 快速跑通{C.RST}")
        print(f"  {C.CYN}[2]{C.RST} 标准训练  {C.DIM}中等网络(256), 2000步{C.RST}")
        print(f"  {C.CYN}[3]{C.RST} 精确训练  {C.DIM}大网络(1024), 10000步{C.RST}")
        print(f"  {C.CYN}[4]{C.RST} 自定义    {C.DIM}进入完整参数面板{C.RST}")
        hr(46)
        print(f"  {C.CYN}[i]{C.RST} 输入源  {C.CYN}[o]{C.RST} 保存配置  {C.CYN}[l]{C.RST} 加载配置")
        print(f"  {C.GRN}[Enter]{C.RST} 开始训练  {C.RED}[q]{C.RST} 退出")
        hr(46)

        try:
            choice = input("选择: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice == "q":
            return None

        if choice in ("1", "2", "3"):
            preset_name = {"1": "quick", "2": "standard", "3": "precise"}[choice]
            apply_preset(preset_name)
            print(f"\n  {C.DIM}按 Enter 开始训练...{C.RST}")
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            return True

        if choice == "4":
            cp_result = control_panel(core)
            if cp_result is None:
                return None
            continue

        if choice == "i":
            _input_source_menu(core)
            continue

        if choice == "o":
            from engine.config import save_config
            ok, msg = save_config()
            print(f"  {'!' if ok else 'x'} {'已保存: ' + msg if ok else msg}")
            continue

        if choice == "l":
            from engine.config import load_config
            ok, msg = load_config()
            print(f"  {'!' if ok else 'x'} {'已加载: ' + msg if ok else msg}")
            continue

        if choice == "":
            return True

        print(f"  {C.RED}未知命令: '{choice}'{C.RST}")


def control_panel(core=None):
    """控制面板 - v4.3 分层菜单（热键复用 + 职责分离）"""
    from engine.config import save_config, load_config, reset_config, ConfigRegistry, CONFIG, is_config_modified
    from engine.utils import C, box, hr

    categories = [
        ("1", "网络架构", "网络架构"),
        ("2", "学习参数", "学习参数"),
        ("3", "资源限制", "资源限制"),
        ("4", "噪声参数", "噪声参数"),
        ("5", "界面偏好", "界面偏好"),
        ("6", "高级选项", "高级选项"),
    ]

    while True:
        box("控制面板")
        if core:
            st = core.get_state() if hasattr(core, 'get_state') else {}
            neurons = st.get('neurons', len(core.N)) if hasattr(core, 'N') else '?'
            history = len(core.history) if hasattr(core, 'history') else '?'
            if getattr(core, 'graph_mode', False):
                graphs = st.get('templates', len(core.graphs)) if hasattr(core, 'graphs') else '?'
                total_nodes = sum(g.get_total_nodes() for g in core.graphs.values()) if hasattr(core, 'graphs') else 0
                print(f"  当前网络: {C.val(neurons)}神经元 / {C.val(graphs)}图 / {C.val(total_nodes)}节点 / {C.val(history)}步 {C.CYN}[图模式]{C.RST}")
            elif getattr(core, 'multi_layer_enabled', False):
                graphs = st.get('templates', len(core.graphs)) if hasattr(core, 'graphs') else '?'
                l0 = st.get('layer0_count', '?')
                l1 = st.get('layer1_count', '?')
                print(f"  当前网络: {C.val(neurons)}神经元(L0={l0}/L1={l1}) / {C.val(graphs)}图 / {C.val(history)}步 {C.CYN}[多层]{C.RST}")
            else:
                templates = st.get('templates', len(core.templates)) if hasattr(core, 'templates') else '?'
                print(f"  当前网络: {C.val(neurons)}神经元 / {C.val(templates)}模板 / {C.val(history)}步")
        print(f"  {C.DIM}修改架构参数（神经元/K值/种子）将重建网络并清空进度{C.RST}")
        if is_config_modified():
            print(f"\n  {C.YEL}⚠ 配置已修改{C.RST}")

        hr(46)
        for hk, name, cat in categories:
            extra = ""
            if cat == "噪声参数":
                nt = CONFIG.get("NOISE_TYPE", "composite")
                fp = CONFIG.get("FLIP_PROB", 0.1)
                nt_label = {"composite": "复合", "gaussian": "高斯", "salt_pepper": "椒盐", "block": "块遮挡"}.get(nt, nt)
                # 【v4.3-fix】动态显示当前噪声类型的关键参数
                if nt == "gaussian":
                    sigma = CONFIG.get("NOISE_SIGMA", 32.0)
                    extra = f"  ({nt_label} σ={sigma})"
                elif nt == "block":
                    bs = CONFIG.get("NOISE_BLOCK_SIZE", 2)
                    extra = f"  ({nt_label} size={bs} p={fp})"
                elif nt == "salt_pepper":
                    extra = f"  ({nt_label} p={fp})"
                else:
                    extra = f"  ({nt_label} p={fp})"
                # 五角星标记当前训练噪声配置
                extra += f" {C.GRN}★ 训练噪声{C.RST}"
            print(f"  {C.CYN}[{hk}]{C.RST} {name}{C.DIM}{extra}{C.RST}")
        print(f"  {C.CYN}[e]{C.RST} 扩展功能")
        print(f"  {C.CYN}[i]{C.RST} 输入源切换")
        hr(46)
        print(f"  {C.YEL}[s]{C.RST}保存配置  {C.YEL}[l]{C.RST}加载配置  {C.RED}[r]{C.RST}恢复默认")
        print(f"  {C.GRN}[Enter]{C.RST}开始训练  {C.RED}[q]{C.RST}退出")
        hr(46)

        try:
            choice = input("选择: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

        if choice == "":
            if is_config_modified():
                print(f"\n  {C.YEL}⚠ 配置已修改但未保存！{C.RST}")
                try:
                    confirm = input("  操作 [s=保存/Enter=不保存继续]: ").strip().lower()
                    if confirm == "s":
                        ok, msg = save_config()
                        print(f"  {'✓' if ok else '✗'} {'已保存: ' + msg if ok else msg}")
                except (EOFError, KeyboardInterrupt):
                    pass
            return True
        if choice == "q":
            return None
        if choice == "r":
            reset_config()
            print(f"  {C.ok('✓')} 已恢复默认配置")
            continue
        if choice == "s":
            ok, msg = save_config()
            print(f"  {'✓' if ok else '✗'} {'已保存: ' + msg if ok else msg}")
            continue
        if choice == "l":
            ok, msg = load_config()
            print(f"  {'✓' if ok else '✗'} {'已加载: ' + msg if ok else msg}")
            continue
        if choice == "e":
            _extension_menu(core)
            continue
        if choice == "i":
            _input_source_menu(core)
            continue
        if choice == "6":
            _show_category_menu("高级选项", core)
            continue

        for hk, name, cat in categories:
            if choice == hk:
                _show_category_menu(cat, core)
                break
        else:
            print(f"  {C.RED}未知命令: '{choice}'{C.RST}")


def _advanced_options_help():
    """高级选项帮助 - 说明各项作用与冲突关系"""
    from engine.utils import C, box, hr
    box("高级选项 帮助")
    print(f"  {C.BOLD}各项说明{C.RST}")
    print(f"  {C.CYN}[a]{C.RST} 数据集文件路径")
    print(f"      从 JSON/CSV 加载训练数据（仅输入源=file 时生效）")
    print(f"  {C.CYN}[b]{C.RST} 启用自适应随机静默（取代旧门控）")
    print(f"      每步 30%~80% 神经元随机静默，不参与判断/竞争/输出")
    print(f"      可调参数：SILENCE_MIN_RATIO / SILENCE_MAX_RATIO / SILENCE_TRIGGER_PROB")
    print(f"  {C.CYN}[c]{C.RST} 启用图模式（多图并行+层级递进）")
    print(f"      独立训练路径：神经元投影为图节点，多视图并行 + 反馈循环")
    print(f"  {C.CYN}[d]{C.RST} 启用多层神经元架构")
    print(f"      独立训练路径：Layer0 基础匹配 → Layer1 基于 L0 组合做概念判断")
    print(f"  {C.CYN}[e]{C.RST} 启用标签专精分片（路径分割）")
    print(f"      full 模式增强：神经元按标签专精，减少跨标签竞争")
    print(f"  {C.CYN}[g][h][i]{C.RST} 输入源相关")
    print(f"      也可通过主菜单 [i] 输入源切换 快捷设置")
    print()
    print(f"  {C.BOLD}冲突关系（训练路径分发优先级）{C.RST}")
    print(f"  {C.YEL}ENABLE_MULTI_LAYER_NEURON > ENABLE_GRAPH_MODE > full{C.RST}")
    print(f"  ─ 同时开 [c]图模式 和 [d]多层：只走 [d] 多层，[c] 被忽略")
    print(f"  ─ 同时开 [c]图模式 和 full增强([b][e])：走 [c] 图模式，增强项无效")
    print(f"  ─ 仅 [b][e] 开启 且 [c][d] 关闭：走 full + 增强项")
    print()
    print(f"  {C.BOLD}注意事项{C.RST}")
    print(f"  ⚠ 切换 [c][d] 或改 NEURON_LAYER_*_COUNT 会触发网络重建（清空进度）")
    print(f"  ⚠ [b] 静默机制与所有模式正交，full/图模式/多层模式均生效")
    print(f"  ⚠ 输入源参数 [g][h][i] 与路径开关独立，不冲突")
    hr(46)
    try:
        input(f"  {C.DIM}按 Enter 返回...{C.RST}")
    except (EOFError, KeyboardInterrupt):
        pass


def _show_category_menu(category, core=None):
    """Category 子菜单 - 热键 a-z 独立分配（当前页独占）"""
    from engine.config import ConfigRegistry, CONFIG
    from engine.utils import C, box, hr

    items = sorted(
        [item for item in ConfigRegistry._schema.values() if item.category == category and item.key != "D"],
        key=lambda x: x.key
    )
    if not items:
        print(f"  {C.DIM}该分类下无配置项{C.RST}")
        return

    hotkeys = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    menu = {}
    for i, item in enumerate(items):
        if i < len(hotkeys):
            menu[hotkeys[i]] = item

    # 高级选项分类提供帮助入口
    has_help = (category == "高级选项")

    while True:
        box(f"{category} 设置")
        for hk, item in menu.items():
            val = CONFIG.get(item.key)
            val_str = str(val) if not hasattr(val, 'level') else f"L{val.level}:I{val.index}"
            print(f"  {C.CYN}[{hk}]{C.RST} {item.description}: {C.val(val_str)}")
        if has_help:
            print(f"  {C.YEL}[?]{C.RST} 帮助（各项说明 + 冲突关系）")
        print(f"  {C.RED}[q]{C.RST} 返回主菜单")
        hr(46)

        try:
            choice = input("选择: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if choice == "q" or choice == "":
            break

        if has_help and choice == "?":
            _advanced_options_help()
            continue

        if choice in menu:
            item = menu[choice]
            key = item.key
            old = CONFIG[key]
            if key == "D":
                print(f"\n  {C.RED}⚠ D(输入维度) 已由窗口大小识别自动管理！{C.RST}")
                try:
                    confirm = input("  仍要修改? [y/N]: ").strip().lower()
                    if confirm not in ("y", "yes"):
                        continue
                except (EOFError, KeyboardInterrupt):
                    continue
            if key == "SEED":
                print(f"\n  {C.YEL}⚠ 随机种子决定每个神经元的初始掩码{C.RST}")
            try:
                new_str = input(f"  {item.description} {key}{C.DIM}[{old}]{C.RST}: ").strip()
                if not new_str:
                    continue
                if item.val_type == int:
                    new_val = int(new_str)
                elif item.val_type == float:
                    new_val = float(new_str)
                elif item.val_type == bool:
                    lo = new_str.lower()
                    if lo in ("true", "1", "yes", "y", "on"):
                        new_val = True
                    elif lo in ("false", "0", "no", "n", "off"):
                        new_val = False
                    else:
                        print(f"  {C.err('✗')} 无法识别 '{new_str}'，请输入 true/false（或 y/n）")
                        print(f"  {C.info('ℹ')} 保留原值: {key}={old}")
                        continue
                elif item.val_type == list:
                    new_val = [v.strip() for v in new_str.split(",") if v.strip()]
                elif hasattr(item.val_type, '__name__') and item.val_type.__name__ == 'DiscreteCoordinate':
                    new_val = new_str
                else:
                    new_val = new_str
                ok, msg = ConfigRegistry.set(key, new_val)
                if not ok:
                    print(f"  {C.err('✗')} {msg}")
                    print(f"  {C.info('ℹ')} 保留原值: {key}={old}")
                    continue
                print(f"  {C.ok('✓')} {key}={C.val(CONFIG[key])}")

                # ---- 神经元数量三方同步 ----
                # MAX_NEURONS [d] / NEURON_LAYER_0_COUNT [e] / NEURON_LAYER_1_COUNT [f]
                # 始终保持一致：L0 + L1 = MAX_NEURONS（比例约 2:1）
                if key == "MAX_NEURONS":
                    new_max = CONFIG["MAX_NEURONS"]
                    new_l0 = max(16, min(512, int(new_max * 0.67)))
                    new_l1 = max(8, min(256, int(new_max * 0.33)))
                    old_l0 = CONFIG.get("NEURON_LAYER_0_COUNT", 128)
                    old_l1 = CONFIG.get("NEURON_LAYER_1_COUNT", 64)
                    if new_l0 != old_l0:
                        ConfigRegistry.set("NEURON_LAYER_0_COUNT", new_l0)
                        print(f"  {C.info('ℹ')} 同步: Layer 0 {old_l0} → {C.val(new_l0)}")
                    if new_l1 != old_l1:
                        ConfigRegistry.set("NEURON_LAYER_1_COUNT", new_l1)
                        print(f"  {C.info('ℹ')} 同步: Layer 1 {old_l1} → {C.val(new_l1)}")

                elif key in ("NEURON_LAYER_0_COUNT", "NEURON_LAYER_1_COUNT"):
                    l0 = CONFIG.get("NEURON_LAYER_0_COUNT", 128)
                    l1 = CONFIG.get("NEURON_LAYER_1_COUNT", 64)
                    new_total = l0 + l1
                    old_total = CONFIG.get("MAX_NEURONS", 256)
                    if new_total != old_total:
                        ConfigRegistry.set("MAX_NEURONS", new_total)
                        print(f"  {C.info('ℹ')} 同步: MAX_NEURONS {old_total} → {C.val(new_total)} (L0+L1)")

                # 切换多层/单层模式时提示当前状态
                if key == "ENABLE_MULTI_LAYER_NEURON":
                    if CONFIG["ENABLE_MULTI_LAYER_NEURON"]:
                        l0 = CONFIG.get("NEURON_LAYER_0_COUNT", 128)
                        l1 = CONFIG.get("NEURON_LAYER_1_COUNT", 64)
                        print(f"  {C.info('ℹ')} 多层模式: Layer 0={l0} + Layer 1={l1} = {l0+l1} 神经元")
                        if CONFIG.get("ENABLE_GRAPH_MODE", False):
                            print(f"  {C.YEL}⚠ 图模式已开启但将被忽略（多层优先级更高）{C.RST}")
                    else:
                        print(f"  {C.info('ℹ')} 单层模式: MAX_NEURONS={CONFIG['MAX_NEURONS']} 神经元")

                # 开启图模式时检测多层冲突
                if key == "ENABLE_GRAPH_MODE" and CONFIG["ENABLE_GRAPH_MODE"]:
                    if CONFIG.get("ENABLE_MULTI_LAYER_NEURON", False):
                        print(f"  {C.YEL}⚠ 多层神经元已开启，图模式将被忽略！{C.RST}")
                        print(f"  {C.DIM}  优先级: 多层 > 图模式 > full{C.RST}")
                        print(f"  {C.DIM}  如需启用图模式，请先关闭 [d] 启用多层神经元架构{C.RST}")

                if item.requires_rebuild:
                    print(f"  {C.YEL}⚠ 架构参数已修改，返回主菜单后将自动重建网络{C.RST}")
                if msg:
                    print(f"  {C.YEL}⚠ {msg}{C.RST}")
            except (ValueError, TypeError):
                print(f"  {C.err('✗')} 无效输入，保留原值: {key}={old}")
            continue
        else:
            print(f"  {C.RED}未知命令: '{choice}'{C.RST}")


def _extension_menu(core):
    """扩展功能菜单 - 控制面板子菜单"""
    from app.backends import BackendRegistry
    from app.storage import StorageRegistry
    from engine.hooks import HookRegistry
    from app.persist import set_autosave_strategy
    from engine.config import CONFIG
    from engine.utils import C, box, hr

    while True:
        box("扩展功能")
        print(f"  {C.CYN}[o]{C.RST}保存模型    {C.CYN}[d]{C.RST}加载模型    {C.CYN}[x]{C.RST}导出报告")
        print(f"  {C.CYN}[w]{C.RST}成功率查询  {C.CYN}[b]{C.RST}图表后端    {C.CYN}[s]{C.RST}存储后端")
        print(f"  {C.CYN}[a]{C.RST}自动保存    {C.CYN}[h]{C.RST}钩子调试    {C.CYN}[v]{C.RST}噪声验证")
        print(f"  {C.CYN}[g]{C.RST}函数工厂GUI {C.CYN}[n]{C.RST}基准测试")
        print(f"  {C.RED}[q]{C.RST}返回主菜单")
        hr(46)
        try:
            choice = input("选择: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if choice == "q" or choice == "":
            break
        elif choice == "o":
            _cmd_save(core)
        elif choice == "d":
            _cmd_load(core)
        elif choice == "x":
            _cmd_export(core)
        elif choice == "w":
            _cmd_why_accuracy()
        elif choice == "b":
            _switch_chart_backend()
        elif choice == "s":
            _switch_storage_backend()
        elif choice == "a":
            _switch_autosave_strategy()
        elif choice == "h":
            _show_hook_debug()
        elif choice == "v":
            _run_noise_equivalent()
        elif choice == "g":
            _run_gui_factory(core)
        elif choice == "n":
            _benchmark_menu(core)
        else:
            print(f"  {C.RED}未知命令: '{choice}'{C.RST}")


def _input_source_menu(core):
    """输入源切换 - 输入源/公式/网格/文件路径切换"""
    from engine.config import ConfigRegistry, CONFIG
    from engine.utils import C, box, hr

    while True:
        box("输入源切换")
        src_type = CONFIG.get("INPUT_SOURCE_TYPE", "pattern")
        formula = CONFIG.get("VECTOR_FORMULA", "line")
        grid = CONFIG.get("VECTOR_GRID", 8)
        path = CONFIG.get("DATASET_PATH", "")

        print(f"  当前输入源: {C.val(src_type)}")
        if src_type == "vector":
            print(f"    公式: {C.val(formula)}  网格: {C.val(grid)}")
        elif src_type == "file":
            print(f"    文件: {C.val(path or '未设置')}")
        elif src_type == "pattern":
            print(f"    内置字符 (0-9A-Z)  网格: 8 (插件化)")
        print()
        print(f"  {C.CYN}[1]{C.RST} 内置字符模式 (8×8 插件化){C.GRN} ★{C.RST}" if src_type == "pattern" else f"  {C.CYN}[1]{C.RST} 内置字符模式 (8×8 插件化)")
        print(f"  {C.CYN}[2]{C.RST} 矢量直线 (8×8){C.GRN} ★{C.RST}" if src_type == "vector" and formula == "line" else f"  {C.CYN}[2]{C.RST} 矢量直线 (8×8)")
        print(f"  {C.CYN}[3]{C.RST} 矢量圆 (8×8){C.GRN} ★{C.RST}" if src_type == "vector" and formula == "circle" else f"  {C.CYN}[3]{C.RST} 矢量圆 (8×8)")
        print(f"  {C.CYN}[4]{C.RST} 矢量正弦 (8×8){C.GRN} ★{C.RST}" if src_type == "vector" and formula == "sine" else f"  {C.CYN}[4]{C.RST} 矢量正弦 (8×8)")
        print(f"  {C.CYN}[5]{C.RST} 混合矢量 (8×8){C.GRN} ★{C.RST}" if src_type == "vector" and formula == "mixed" else f"  {C.CYN}[5]{C.RST} 混合矢量 (8×8)")
        print(f"  {C.CYN}[7]{C.RST} 矢量拱门 (8×8){C.GRN} ★{C.RST}" if src_type == "vector" and formula == "arch" else f"  {C.CYN}[7]{C.RST} 矢量拱门 (8×8)")
        print(f"  {C.CYN}[8]{C.RST} 矢量叶片 (8×8){C.GRN} ★{C.RST}" if src_type == "vector" and formula == "leaf" else f"  {C.CYN}[8]{C.RST} 矢量叶片 (8×8)")
        print(f"  {C.CYN}[6]{C.RST} 从文件加载{C.GRN} ★{C.RST}" if src_type == "file" else f"  {C.CYN}[6]{C.RST} 从文件加载")
        print(f"  {C.CYN}[g]{C.RST} 网格大小: {C.val(grid)}")
        print(f"  {C.CYN}[f]{C.RST} 公式类型: {C.val(formula)}")
        print(f"  {C.CYN}[p]{C.RST} 文件路径: {C.val(path or '未设置')}")
        print(f"  {C.RED}[q]{C.RST} 返回主菜单")
        hr(46)

        try:
            choice = input("选择: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if choice == "q" or choice == "":
            break
        elif choice == "1":
            ConfigRegistry.set("INPUT_SOURCE_TYPE", "pattern")
            print(f"  {C.ok('✓')} 已切换到内置字符模式")
        elif choice == "2":
            ConfigRegistry.set("INPUT_SOURCE_TYPE", "vector")
            ConfigRegistry.set("VECTOR_FORMULA", "line")
            print(f"  {C.ok('✓')} 已切换到矢量直线 (网格: {grid})")
        elif choice == "3":
            ConfigRegistry.set("INPUT_SOURCE_TYPE", "vector")
            ConfigRegistry.set("VECTOR_FORMULA", "circle")
            print(f"  {C.ok('✓')} 已切换到矢量圆 (网格: {grid})")
        elif choice == "4":
            ConfigRegistry.set("INPUT_SOURCE_TYPE", "vector")
            ConfigRegistry.set("VECTOR_FORMULA", "sine")
            print(f"  {C.ok('✓')} 已切换到矢量正弦 (网格: {grid})")
        elif choice == "5":
            ConfigRegistry.set("INPUT_SOURCE_TYPE", "vector")
            ConfigRegistry.set("VECTOR_FORMULA", "mixed")
            print(f"  {C.ok('✓')} 已切换到混合矢量 (网格: {grid})")
        elif choice == "7":
            ConfigRegistry.set("INPUT_SOURCE_TYPE", "vector")
            ConfigRegistry.set("VECTOR_FORMULA", "arch")
            print(f"  {C.ok('✓')} 已切换到矢量拱门 (网格: {grid})")
        elif choice == "8":
            ConfigRegistry.set("INPUT_SOURCE_TYPE", "vector")
            ConfigRegistry.set("VECTOR_FORMULA", "leaf")
            print(f"  {C.ok('✓')} 已切换到矢量叶片 (网格: {grid})")
        elif choice == "6":
            ConfigRegistry.set("INPUT_SOURCE_TYPE", "file")
            p = input("  文件路径: ").strip()
            if p:
                ConfigRegistry.set("DATASET_PATH", p)
            print(f"  {C.ok('✓')} 已切换到文件输入")
        elif choice == "g":
            try:
                g = int(input("  网格大小 (8/16/32/64): ").strip())
                if g in (8, 16, 32, 64):
                    ConfigRegistry.set("VECTOR_GRID", g)
                    print(f"  {C.ok('✓')} 网格大小已设为 {g}")
                else:
                    print(f"  {C.err('✗')} 仅支持 8/16/32/64")
            except ValueError:
                print(f"  {C.err('✗')} 无效输入")
        elif choice == "f":
            print(f"  {C.CYN}[1]{C.RST} line  {C.CYN}[2]{C.RST} circle  {C.CYN}[3]{C.RST} sine  {C.CYN}[4]{C.RST} arch  {C.CYN}[5]{C.RST} leaf  {C.CYN}[6]{C.RST} mixed")
            try:
                fc = input("  选择: ").strip()
                fm = {"1": "line", "2": "circle", "3": "sine", "4": "arch", "5": "leaf", "6": "mixed"}
                if fc in fm:
                    ConfigRegistry.set("VECTOR_FORMULA", fm[fc])
                    print(f"  {C.ok('✓')} 公式已设为 {fm[fc]}")
                else:
                    print(f"  {C.err('✗')} 无效选择")
            except (EOFError, KeyboardInterrupt):
                pass
        elif choice == "p":
            try:
                p = input("  文件路径: ").strip()
                if p:
                    ConfigRegistry.set("DATASET_PATH", p)
                    print(f"  {C.ok('✓')} 文件路径已更新")
            except (EOFError, KeyboardInterrupt):
                pass
        else:
            print(f"  {C.RED}未知命令: '{choice}'{C.RST}")


# ============================================================
# 扩展菜单子功能
# ============================================================

def _cmd_save(core):
    """保存模型"""
    from app.persist import save_model
    from engine.utils import C
    try:
        path = input(f"  保存路径[sgn_model.json]: ").strip()
        save_model(core, path if path else None)
    except (EOFError, KeyboardInterrupt):
        pass


def _cmd_load(core):
    """加载模型"""
    from app.persist import load_model
    from engine.utils import C
    try:
        path = input("  加载路径: ").strip()
        if path:
            load_model(core, path)
    except (EOFError, KeyboardInterrupt):
        pass


def _cmd_export(core):
    """导出报告"""
    try:
        from app.report import do_export
        do_export(core)
    except ImportError:
        print("  导出模块未加载")


def _cmd_why_accuracy():
    """成功率疑问"""
    try:
        from app.help import do_why_accuracy
        do_why_accuracy()
    except ImportError:
        print("  帮助模块未加载")


def _switch_chart_backend():
    """切换图表后端"""
    from app.backends import BackendRegistry
    from engine.utils import C
    current = BackendRegistry.auto_select().name
    print(f"\n  当前后端: {C.val(current)}")
    for i, name in enumerate(BackendRegistry.list_backends(), 1):
        b = BackendRegistry.get(name)
        avail = "✓" if b.is_available() else "✗"
        marker = " <<<" if name == current else ""
        print(f"  {C.CYN}[{i}]{C.RST} {name} [{avail}]{marker}")
    try:
        c = input("  选择: ").strip()
        idx = int(c) - 1
        names = BackendRegistry.list_backends()
        if 0 <= idx < len(names):
            print(f"  {C.YEL}⚠ 图表后端选择需在调用时指定，当前仅作查看{C.RST}")
    except (ValueError, EOFError, KeyboardInterrupt):
        pass


def _switch_storage_backend():
    """切换存储后端"""
    from app.storage import StorageRegistry
    from engine.utils import C
    current = StorageRegistry._default
    print(f"\n  当前后端: {C.val(current)}")
    for i, name in enumerate(StorageRegistry.list_backends(), 1):
        marker = " <<<" if name == current else ""
        print(f"  {C.CYN}[{i}]{C.RST} {name}{marker}")
    try:
        c = input("  选择: ").strip()
        idx = int(c) - 1
        names = StorageRegistry.list_backends()
        if 0 <= idx < len(names):
            StorageRegistry._default = names[idx]
            print(f"  {C.ok('✓')} 默认存储后端已切换: {C.val(names[idx])}")
    except (ValueError, EOFError, KeyboardInterrupt):
        pass


def _switch_autosave_strategy():
    """切换自动保存策略"""
    from app.persist import set_autosave_strategy
    from app.storage import IntervalAutosave, DeltaAutosave
    from engine.config import CONFIG
    from engine.utils import C
    current = CONFIG.get("AUTOSAVE_STRATEGY", "interval")
    print(f"{C.CYN}[1]{C.RST} 间隔保存 (每N步){C.GRN} ★{C.RST}" if current == "interval" else f"{C.CYN}[1]{C.RST} 间隔保存 (每N步)")
    print(f"  {C.CYN}[2]{C.RST} 增量保存 (模板变化时){C.GRN} ★{C.RST}" if current == "delta" else f"  {C.CYN}[2]{C.RST} 增量保存 (模板变化时)")
    try:
        c = input("  选择: ").strip()
        if c == "1":
            try:
                n = input("  间隔步数[50]: ").strip()
                interval = int(n) if n else 50
                set_autosave_strategy(IntervalAutosave(interval=interval))
                print(f"  {C.ok('✓')} 已切换为间隔保存: {interval}步")
            except ValueError:
                pass
        elif c == "2":
            try:
                n = input("  最小间隔步数[50]: ").strip()
                min_interval = int(n) if n else 50
                set_autosave_strategy(DeltaAutosave(min_interval=min_interval))
                print(f"  {C.ok('✓')} 已切换为增量保存")
            except ValueError:
                pass
    except (EOFError, KeyboardInterrupt):
        pass


def _show_hook_debug():
    """钩子调试信息"""
    from engine.hooks import HookRegistry, get_hook_errors, clear_hook_errors
    from engine.utils import C
    events = HookRegistry.list_events()
    print(f"{C.BOLD}已注册事件 ({len(events)} 个){C.RST}")
    for evt in events:
        cnt = HookRegistry.count(evt)
        print(f"  {C.DIM}{evt}{C.RST}: {C.val(cnt)} 个回调")
    errors = get_hook_errors()
    if errors:
        print(f"{C.YEL}最近钩子错误 ({len(errors)} 条){C.RST}")
        for e in errors[-5:]:
            print(f"  {C.RED}{e['event']}: {e['exc_type']}{C.RST}")
        try:
            c = input(f"{C.DIM}输入 [c] 清空错误日志，Enter 返回{C.RST}: ").strip().lower()
            if c == "c":
                clear_hook_errors()
                print(f"  {C.GRN}✓ 错误日志已清空{C.RST}")
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        print(f"{C.GRN}✓ 无钩子错误记录{C.RST}")


def _run_noise_equivalent():
    """运行噪声等效验证"""
    from engine.utils import C
    print(f"{C.info('ℹ')} 正在运行噪声等效性验证...")
    try:
        from app.noise_equivalent import run_noise_equivalent
        run_noise_equivalent()
    except ImportError as e:
        print(f"  {C.err('✗')} 导入失败: {e}")
    except Exception as e:
        print(f"  {C.err('✗')} 运行失败: {e}")


def _run_gui_factory(core=None):
    """启动函数工厂 GUI"""
    from engine.utils import C
    try:
        from gui.sgn_gui_factory import run_factory
        run_factory(core=core)
    except ImportError as e:
        print(f"  {C.err('✗')} 函数工厂模块未加载: {e}")
        print(f"  {C.DIM}  请确保 pygame 已安装: pip install pygame-ce{C.RST}")


def _benchmark_menu(core):
    """v5.1.7 补充: 基准测试子菜单"""
    from engine.config import CONFIG
    from engine.utils import C, box, hr

    while True:
        box("基准测试")

        # 显示当前网络配置
        multi = CONFIG.get("ENABLE_MULTI_LAYER_NEURON", False)
        mode_str = "多层模式" if multi else "单层模式"
        if multi:
            l0 = CONFIG.get("NEURON_LAYER_0_COUNT", 0)
            l1 = CONFIG.get("NEURON_LAYER_1_COUNT", 0)
            neuron_count = l0 + l1
        else:
            neuron_count = CONFIG.get("MAX_NEURONS", 0)
        noise = CONFIG.get("NOISE_TYPE", "composite")
        silence = "开" if CONFIG.get("ENABLE_ADAPTIVE_SILENCE", True) else "关"

        print(f"  当前网络: {mode_str}  神经元: {neuron_count}  噪声: {noise}  静默: {silence}")
        hr(50)
        print(f"  {C.CYN}[1]{C.RST}噪声鲁棒性  (5级: 0.0~0.3)")
        print(f"  {C.CYN}[2]{C.RST}网络规模    (5档: 64~1024)")
        print(f"  {C.CYN}[3]{C.RST}多层vs单层  (2组对比)")
        print(f"  {C.CYN}[4]{C.RST}静默开vs关  (2组对比)")
        print(f"  {C.CYN}[a]{C.RST}全部运行    {C.CYN}[h]{C.RST}历史结果")
        print(f"  {C.RED}[q]{C.RST}返回扩展功能")
        hr(50)
        try:
            choice = input("选择: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if choice == "q" or choice == "":
            return

        # 委托给 benchmark_convergence 模块
        from tests.benchmark_convergence import (
            test_noise_robustness, test_network_scale,
            test_multi_vs_single, test_silence_on_off,
            print_report, save_json_report, show_history_detail,
        )

        all_results = {}
        if choice == "1":
            all_results['test1'] = test_noise_robustness()
        elif choice == "2":
            all_results['test2'] = test_network_scale()
        elif choice == "3":
            all_results['test3'] = test_multi_vs_single()
        elif choice == "4":
            all_results['test4'] = test_silence_on_off()
        elif choice == "a":
            all_results['test1'] = test_noise_robustness()
            all_results['test2'] = test_network_scale()
            all_results['test3'] = test_multi_vs_single()
            all_results['test4'] = test_silence_on_off()
        elif choice == "h":
            show_history_detail()
            continue
        else:
            print(f"  {C.RED}未知命令: '{choice}'{C.RST}")
            continue

        if all_results:
            print_report(all_results)
            save_json_report(all_results)
