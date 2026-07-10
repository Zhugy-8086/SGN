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
"""
SGN-Lite v5.0 Python PC平台验证脚本 — 插件化架构扩展版本
入口文件 (修正版 v3)

用法:
  python main.py                  交互式训练
  python main.py --auto 50        自动模式(50ms/步)
  python main.py --batch          批量训练后测试
  python main.py --test-only --import-model model.json   仅测试
  python main.py --resume         恢复中断的训练
  python main.py --no-color       禁用彩色输出
  python main.py --output log.txt 输出日志到文件
  python main.py --input-source file --dataset data.csv   从文件加载数据
  python main.py -h               查看完整帮助

v5.0 新特性:
  插件化架构: HookRegistry / ConfigRegistry / CommandRegistry
  输入管道抽象: InputSource / NoiseModel / FeatureExtractor
  可视化后端抽象: ChartBackend (matplotlib/ASCII/CSV)
  评估指标抽象: Metric (Accuracy/Confusion/NoiseRobustness)
  持久化策略抽象: StorageBackend / AutosaveStrategy
"""


# ============================================================
# 启动导入保护壳 —— 防止模块拆分/缺失导致的闪退
# ============================================================
_import_errors = []
try:
    from sgn_config import CONFIG, LABELS, D, DEFAULT_CONFIG, should_draw_grid
    from sgn_config import save_config as cfg_save, load_config as cfg_load
except ImportError as _e:
    _import_errors.append(("sgn_config", _e))

try:
    from sgn_utils import C, box, hr, progress_bar, gen_samples, extract_layers
except ImportError as _e:
    _import_errors.append(("sgn_utils", _e))

try:
    from sgn_core import SGNCore
except ImportError as _e:
    _import_errors.append(("sgn_core", _e))

try:
    from sgn_persist import save_model, load_model, autosave_check, check_resume
except ImportError as _e:
    _import_errors.append(("sgn_persist", _e))

try:
    from sgn_visual import do_stats, do_gauge, do_visualize, do_heatmap
except ImportError as _e:
    _import_errors.append(("sgn_visual", _e))

try:
    from sgn_interactive import control_panel, menu
except ImportError as _e:
    _import_errors.append(("sgn_interactive", _e))

try:
    from sgn_training import train_batch, header, out_step
except ImportError as _e:
    _import_errors.append(("sgn_training", _e))

try:
    from sgn_help import do_help
except ImportError as _e:
    _import_errors.append(("sgn_help", _e))

try:
    from sgn_layers import extract_layers
except ImportError as _e:
    _import_errors.append(("sgn_layers", _e))

try:
    from sgn_draw import draw_binary_grid
except ImportError as _e:
    _import_errors.append(("sgn_draw", _e))

try:
    from sgn_cmd_registry import register_all_commands
except ImportError as _e:
    _import_errors.append(("sgn_cmd_registry", _e))

try:
    from sgn_hooks import HookRegistry
except ImportError as _e:
    _import_errors.append(("sgn_hooks", _e))

try:
    from sgn_input import (
        create_default_source, create_file_source,
        DefaultCompositeNoise, PatternInputSource
    )
except ImportError as _e:
    _import_errors.append(("sgn_input", _e))

try:
    from sgn_test import do_batch_test, do_confusion, do_noise_test
except ImportError as _e:
    _import_errors.append(("sgn_test", _e))

try:
    from sgn_report import do_plot
except ImportError as _e:
    _import_errors.append(("sgn_report", _e))

try:
    from sgn_utils import compact_banner, compact_step, compact_summary
except ImportError as _e:
    _import_errors.append(("sgn_utils_compact", _e))

if _import_errors:
    print("\n" + "=" * 56)
    print("  [启动失败] 以下模块导入错误，程序无法启动")
    print("=" * 56)
    for mod, err in _import_errors:
        print(f"  • {mod}: {err}")
    print("\n  常见原因:")
    print("    1) 模块拆分后导入路径未更新")
    print("    2) 缺少依赖库（如 matplotlib）")
    print("    3) 文件被删除或命名错误")
    print("\n  请修复上述错误后重新启动。")
    input("\n  按回车退出...")
    import sys
    sys.exit(1)

import sys
import random
import argparse

from sgn_config import CONFIG, LABELS, D, DEFAULT_CONFIG, should_draw_grid
from sgn_config import save_config as cfg_save, load_config as cfg_load
from sgn_utils import (
    C, box, hr, detect_encoding_issue,
    set_log_file, close_log, log_print, clear_stdin_buffer
)
from sgn_core import SGNCore
from sgn_persist import save_model, load_model, autosave_check, check_resume
from sgn_visual import (
    do_stats, do_gauge, do_visualize, do_heatmap
)
from sgn_test import do_batch_test, do_confusion, do_noise_test
from sgn_report import do_plot
from sgn_utils import compact_banner, compact_step

from sgn_interactive import control_panel, menu
from sgn_training import train_batch, header, out_step
from sgn_help import do_help
from sgn_layers import extract_layers
from sgn_draw import draw_binary_grid
from sgn_cmd_registry import register_all_commands
from sgn_utils import gen_samples, extract_layers
from sgn_hooks import HookRegistry
from sgn_input import (
    create_default_source, create_file_source,
    DefaultCompositeNoise, PatternInputSource
)


def _format_duration(seconds):
    """格式化秒数为 HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


def build_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="SGN-Lite v5.0 Python PC平台验证脚本 (插件化架构扩展版本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                  交互式训练
  python main.py --auto 50        自动模式(50ms/步)
  python main.py --batch          批量模式
  python main.py --mode blackbox  黑箱模式(零输出+手动验证)
  python main.py --mode compact   精简模式(检查点摘要)
  python main.py --test-only --import-model model.json  仅测试
  python main.py --resume         恢复中断的训练
  python main.py --no-color       禁用彩色输出
  python main.py --output log.txt 输出日志到文件
  python main.py --input-source file --dataset data.csv  从CSV加载数据
  python main.py -h               查看完整帮助

v5.0 插件化架构: HookRegistry | ConfigRegistry | CommandRegistry |
                  InputSource | ChartBackend | Metric | StorageBackend
        """
    )
    parser.add_argument("--auto", type=int, metavar="MS",
                        help="自动模式，每MS毫秒显示一次")
    parser.add_argument("--batch", action="store_true",
                        help="批量模式: 训练后测试")
    parser.add_argument("--config", type=str,
                        help="指定配置文件路径")
    parser.add_argument("--no-color", action="store_true",
                        help="禁用ANSI颜色输出")
    parser.add_argument("--output", type=str, metavar="FILE",
                        help="指定日志/结果输出文件")
    parser.add_argument("--test-only", action="store_true",
                        help="仅做批量测试/混淆矩阵，跳过训练")
    parser.add_argument("--import-model", type=str, metavar="FILE",
                        help="启动时加载已保存的模型")
    parser.add_argument("--export-model", type=str, metavar="FILE",
                        help="训练结束后自动导出模型")
    parser.add_argument("--resume", action="store_true",
                        help="检测并恢复中断的训练")
    parser.add_argument("--mode", type=str, choices=["full", "compact", "blackbox"],
                        help="运行模式: full=全记载(默认), compact=精简, blackbox=黑箱")
    parser.add_argument("--input-source", type=str, default="pattern", choices=["pattern", "file"],
                        help="输入源: pattern=内置模式(默认), file=从文件加载")
    parser.add_argument("--dataset", type=str, metavar="FILE",
                        help="数据集文件路径（配合 --input-source file）")
    parser.add_argument("--vector-formula", type=str,
                        choices=["line", "circle", "sine", "arch", "leaf", "mixed"],
                        help="使用矢量公式生成输入（替代内置 PATTERNS）")
    parser.add_argument("--vector-grid", type=int, default=8,
                        help="矢量图形网格大小（8/16/32/64，默认8）")
    parser.add_argument("--test-split", type=float, default=None,
                        help="测试集留出比例（0.0-0.5，覆盖配置）")
    parser.add_argument("--validation-labels", type=str, default=None,
                        help="不参与训练的字符列表，逗号分隔（如 0,A,F）")
    parser.add_argument("--test-dataset", type=str, default=None,
                        help="独立测试集文件路径（CSV/JSON）")
    parser.add_argument("--cross-validate", action="store_true",
                        help="启用留一字符交叉验证（LOLO）")
    parser.add_argument("--cv-folds", type=int, default=16,
                        help="交叉验证折数（配合 --cross-validate 使用，默认16）")
    return parser


def welcome():
    """显示欢迎信息"""
    enc_hint = detect_encoding_issue()
    box("SGN-Lite v5.0 Python PC平台验证脚本 (插件化架构)")
    if enc_hint:
        print(enc_hint)
    # 显示模块结构
    print(f"\n  {C.DIM}模块结构:{C.RST}")
    print(f"    {C.BOLD}核心层 (Core):{C.RST}")
    print(f"      sgn_core.py      核心引擎(整数化) - {C.YEL}内部逻辑冻结{C.RST}")
    print(f"    {C.BOLD}扩展层 (Extension):{C.RST}")
    print(f"      sgn_hooks.py     事件总线/钩子系统")
    print(f"      sgn_config.py    配置注册表/动态UI")
    print(f"      sgn_commands.py  命令注册表")
    print(f"    {C.BOLD}策略层 (Strategy):{C.RST}")
    print(f"      sgn_input.py     输入管道/噪声模型/特征提取")
    print(f"      sgn_metrics.py   评估指标")
    print(f"      sgn_backends.py  图表后端(matplotlib/ASCII/CSV)")
    print(f"      sgn_storage.py   存储后端/自动保存策略")
    print(f"    {C.BOLD}应用层 (Application):{C.RST}")
    print(f"      sgn_utils.py     工具函数/颜色/日志")
    print(f"      sgn_visual.py    可视化/测试")
    print(f"      sgn_persist.py   模型持久化")
    print(f"      sgn_interactive.py 交互/训练循环")
    print(f"      main.py          入口文件")
    print(f"\n  {C.BOLD}{C.CYN}v5.0 实测调参指南{C.RST}")
    print(f"    神经元: 200~256 优于 64 (大网络才能活，小网络站在悬崖边)")
    print(f"    训练步数: 3000~5000 收益最大 (1万步后半段精炼期，边际递减)")
    print(f"    模板上限: 100~150 够用 (稳态约80个，500只是缓冲)")
    print(f"    随机种子: 35 实测优于 42 (不同种子=完全不同的初始探测器)")
    print(f"    识别率: 256N/3000步可达 80~87.5%% (远超旧认知的 75%% 极限)")
    print(f"    输入 [w] 查看详细成功率说明与调参策略")


def _run_training_step(core, intensity, label, step, old_template_count_ref):
    """执行单步训练 + 钩子 + 模板检测（公共逻辑）

    Returns: (info, new_template_count)
    """
    HookRegistry.emit("sgn:before_step", intensity=intensity, label=label, step=step, core=core)
    info = core.train(intensity, label)
    core.history.append(info)
    step += 1  # noqa: 传入的 step 已递增
    HookRegistry.emit("sgn:after_step", step=step, info=info, core=core)
    # 模板新增检测
    if len(core.templates) > old_template_count_ref[0]:
        new_sig = core.templates[-1][1] if core.templates else 0
        HookRegistry.emit("sgn:on_template_added",
                          label=info.get("label", "?"),
                          signature=new_sig,
                          template_count=len(core.templates))
        old_template_count_ref[0] = len(core.templates)
    return info


def _post_training_menu(core, samples, current_step, max_step, test_samples=None, source=None, is_paused=False):
    """训练结束后的交互菜单循环（公共逻辑）"""
    from sgn_interactive import menu
    while True:
        result = menu(core, samples, current_step, max_step, test_samples=test_samples, source=source)
        if result == "quit":
            return "quit"
        elif result == "back":
            # 【v4.3-fix】返回控制面板
            return "back"
        elif result == "reset":
            core = SGNCore(CONFIG["SEED"])
            log_print(f"\n  {C.info('ℹ')} 网络已重置，重新训练...")
            if is_paused:
                return "reset"
            return run_batch_mode(core, samples, max_step, test_samples=test_samples, source=source)
        elif result == "continue":
            if is_paused:
                return "continue"
            print(f"  {C.warn('⚠ 已达最大步数，使用 [r] 重置')}")
        elif isinstance(result, tuple) and result[0] == "auto":
            if is_paused:
                return result
            print(f"  {C.warn('⚠ 训练已完成，无法切换到自动')}")


def run_auto_mode(core, samples, max_step, delay_ms, test_samples=None, source=None):
    """自动模式 - 修正: 支持 MODE 配置"""
    import time as _time
    max_step = CONFIG.get("MAX_ITERATIONS", max_step)
    sparse = max(0, CONFIG.get("SPARSE_STEP", 0))
    mode = CONFIG.get("MODE", "full")
    step = len(core.history)
    train_start = _time.time()

    if step > 0:
        log_print(f"  {C.info('ℹ')} 从步 {step} 继续训练...")

    if mode == "blackbox":
        # 黑箱模式：全程零输出
        tc_ref = [len(core.templates)]
        while step < max_step:
            intensity, label = samples[step % len(samples)]
            info = _run_training_step(core, intensity, label, step, tc_ref)
            step += 1
            autosave_check(core, step)
        train_elapsed = _time.time() - train_start
        box("训练完成")
        print(f"  训练耗时: {C.val(_format_duration(train_elapsed))} ({step/train_elapsed:.0f} 步/秒)")
        do_batch_test(core, test_samples)
        do_noise_test(core, test_samples=test_samples, source=source)
        do_plot(core)
        result = _post_training_menu(core, samples, step, max_step, test_samples=test_samples, source=source)
        if result == "back":
            return "back"
        return

    # full 模式（默认）
    show_every = sparse if sparse > 0 else 1
    header()
    tc_ref = [len(core.templates)]
    try:
        while step < max_step:
            intensity, label = samples[step % len(samples)]
            info = _run_training_step(core, intensity, label, step, tc_ref)
            step += 1

            if show_every == 1 or step % show_every == 0 or step >= max_step:
                out_step(step, info)
                d = getattr(core, 'D', CONFIG.get("D", 16))
                layers, lc = extract_layers(intensity, d=d)
                if lc > 0:
                    _g2 = int(d**0.5)
                    if should_draw_grid(_g2):
                        draw_binary_grid(layers[0], grid_size=_g2)
                    else:
                        print(f"  {C.DIM}[{_g2}×{_g2} 网格，训练完成后再查看完整可视化]{C.RST}")
                sys.stdout.flush()
                import time
                time.sleep(delay_ms / 1000.0)
            else:
                import time
                time.sleep(delay_ms / 1000.0)

            autosave_check(core, step)
    except Exception as _e:
        _write_crash_log(_e)
        raise

    train_elapsed = _time.time() - train_start
    box("训练完成")
    print(f"  训练耗时: {C.val(_format_duration(train_elapsed))} ({step/train_elapsed:.0f} 步/秒)")
    do_batch_test(core, test_samples)
    do_noise_test(core, test_samples=test_samples, source=source)
    do_plot(core)
    result = _post_training_menu(core, samples, max_step, max_step, test_samples=test_samples, source=source)
    if result == "back":
        return "back"
    elif result == "quit":
        return "quit"


def run_batch_mode(core, samples, max_step, delay_ms=0, test_samples=None, source=None):
    """批量模式 - 修正: 支持 MODE 配置"""
    import time as _time
    max_step = CONFIG.get("MAX_ITERATIONS", max_step)
    mode = CONFIG.get("MODE", "full")
    log_print(f"\n  {C.BOLD}批量模式{C.RST}: {max_step} 步 (模式: {mode})")
    step = 0
    train_start = _time.time()

    if mode == "blackbox":
        tc_ref = [len(core.templates)]
        for i in range(max_step):
            intensity, label = samples[i % len(samples)]
            info = _run_training_step(core, intensity, label, step, tc_ref)
            step += 1
            autosave_check(core, step)
        train_elapsed = _time.time() - train_start
        box("训练完成")
        print(f"  训练耗时: {C.val(_format_duration(train_elapsed))} ({step/train_elapsed:.0f} 步/秒)")
        do_batch_test(core, test_samples)
        do_noise_test(core, test_samples=test_samples, source=source)
        do_plot(core)
        result = _post_training_menu(core, samples, step, max_step, test_samples=test_samples, source=source)
        if result == "back":
            return "back"
        return
    elif mode == "compact":
        compact_banner("精简批量")
        tc_ref = [len(core.templates)]
        compact_interval = max(10, CONFIG.get("COMPACT_INTERVAL", 100))
        for i in range(max_step):
            intensity, label = samples[i % len(samples)]
            info = _run_training_step(core, intensity, label, step, tc_ref)
            step += 1
            if (i + 1) % compact_interval == 0 or (i + 1) >= max_step:
                compact_step(i + 1, max_step, info)
                sys.stdout.flush()
            autosave_check(core, step)
        box("训练完成")
        do_batch_test(core, test_samples)
        do_noise_test(core, test_samples=test_samples, source=source)
        do_plot(core)
        result = _post_training_menu(core, samples, step, max_step, test_samples=test_samples, source=source)
        if result == "back":
            return "back"
        return

    # full 模式（默认）
    header()
    tc_ref = [len(core.templates)]
    try:
        for i in range(max_step):
            intensity, label = samples[i % len(samples)]
            info = _run_training_step(core, intensity, label, step, tc_ref)
            step += 1
            if (i + 1) % 50 == 0 or (i + 1) >= max_step:
                out_step(i + 1, info)
                sys.stdout.flush()
            autosave_check(core, step)
    except Exception as _e:
        _write_crash_log(_e)
        raise
    train_elapsed = _time.time() - train_start
    box("训练完成")
    print(f"  训练耗时: {C.val(_format_duration(train_elapsed))} ({step/train_elapsed:.0f} 步/秒)")
    do_batch_test(core, test_samples)
    do_noise_test(core, test_samples=test_samples, source=source)
    do_plot(core)
    result = _post_training_menu(core, samples, max_step, max_step, test_samples=test_samples, source=source)
    if result == "back":
        return "back"
    elif result == "quit":
        return "quit"


def run_test_only(core):
    """仅测试模式"""
    log_print(f"\n  {C.BOLD}仅测试模式 (跳过训练){C.RST}")
    if not core.templates:
        log_print(f"  {C.warn('⚠')} 无模板，请先训练或导入模型")
        do_help()
        return
    # 生成测试样本
    test_samples = []
    for lbl in LABELS:
        test_samples.extend(gen_samples(lbl, 40, 0.0))
    do_batch_test(core, test_samples)
    do_confusion(core, test_samples)
    do_noise_test(core)
    do_plot(core)


def _maybe_rebuild_core(core, label=""):
    """检查配置是否修改，若修改了架构参数则重建网络

    Returns: 新的 core 对象（若重建）或原对象
    """
    from sgn_config import is_config_modified, mark_config_synced, ConfigRegistry
    if not is_config_modified():
        return core

    # 检查是否有架构参数被修改
    modified_arch = False
    modified_keys = []
    for key in ConfigRegistry._values:
        item = ConfigRegistry.get_schema(key)
        if item and item.requires_rebuild and ConfigRegistry._values[key] != item.default:
            modified_keys.append(key)
            modified_arch = True

    if modified_arch:
        seed = CONFIG.get("SEED", 42)
        log_print(f"\n  {C.info('ℹ')} 检测到架构参数变更 {modified_keys}，正在重建网络...")
        log_print(f"  {C.DIM}   随机种子: {C.val(seed)} (神经元初始模板由种子决定){C.RST}")
        core = SGNCore(seed)
        mark_config_synced()
        log_print(f"  {C.ok('✓')} 网络已重建: {len(core.N)} 神经元 / 种子={C.val(seed)}")
    else:
        # 非架构参数修改，只需同步标记
        mark_config_synced()

    return core


def apply_cli_args(args):
    """命令行参数 → ConfigRegistry 统一入口

    所有命令行参数必须通过此函数进入系统，享受：
      - 类型校验
      - 范围校验
      - 双向同步（浮点↔定点数）
      - 配置变更钩子（提醒/重建检测）
    """
    from sgn_config import ConfigRegistry
    from sgn_utils import C, log_print

    # 映射：命令行属性名 → (配置键, 转换函数, 是否必需)
    cli_mappings = {
        "mode":           ("MODE", str, False),
        "auto":           ("AUTO_DELAY_MS", int, False),
        "test_split":     ("TEST_SPLIT", float, False),
        "input_source":   ("INPUT_SOURCE_TYPE", str, False),
        "vector_formula": ("VECTOR_FORMULA", str, False),
        "vector_grid":    ("VECTOR_GRID", int, False),
    }
    # 注意：DiscreteCoordinate 参数通过控制面板或配置文件设置，
    # 命令行只设置简单类型（str/int/bool/float）

    for attr, (key, transform, required) in cli_mappings.items():
        val = getattr(args, attr, None)
        if val is not None:
            try:
                typed = transform(val)
            except (ValueError, TypeError) as e:
                log_print(f"  {C.err('✗')} 命令行参数 --{attr}={val} 类型转换失败: {e}")
                continue
            ok, msg = ConfigRegistry.set(key, typed)
            if not ok:
                log_print(f"  {C.err('✗')} 命令行参数 {key}={typed} 无效: {msg}")
            elif msg:
                log_print(f"  {C.YEL}⚠ {msg}{C.RST}")

    # --dataset 特殊处理
    if args.dataset:
        ok, msg = ConfigRegistry.set("DATASET_PATH", args.dataset)
        if not ok:
            log_print(f"  {C.err('✗')} 命令行参数 DATASET_PATH={args.dataset} 无效: {msg}")

    # --vector-formula 隐式设置输入源为 vector
    if args.vector_formula:
        # [v5.1 alias compatibility] catear auto-mapped to arch
        if args.vector_formula == "catear":
            args.vector_formula = "arch"
            log_print(f"  {C.info('i')} 'catear' auto-mapped to 'arch' (new name)")
        ok, msg = ConfigRegistry.set("VECTOR_FORMULA", args.vector_formula)
        if ok:
            ConfigRegistry.set("INPUT_SOURCE_TYPE", "vector")
        else:
            log_print(f"  {C.err('✗')} 命令行参数 VECTOR_FORMULA={args.vector_formula} 无效: {msg}")

    # --validation-labels 特殊处理
    if args.validation_labels is not None:
        val = [x.strip() for x in args.validation_labels.split(",") if x.strip()]
        ok, msg = ConfigRegistry.set("VALIDATION_LABELS", val)
        if not ok:
            log_print(f"  {C.err('✗')} 命令行参数 VALIDATION_LABELS={val} 无效: {msg}")

    # --cross-validate 特殊处理
    if args.cross_validate:
        folds = getattr(args, 'cv_folds', 16)
        ok, msg = ConfigRegistry.set("CROSS_VALIDATE_FOLDS", folds)
        if not ok:
            log_print(f"  {C.err('✗')} 命令行参数 CROSS_VALIDATE_FOLDS={folds} 无效: {msg}")

    # --no-color 特殊处理：映射为 COLOR_OUTPUT=False
    if args.no_color:
        ok, msg = ConfigRegistry.set("COLOR_OUTPUT", False)
        if ok:
            log_print(f"  {C.info('ℹ')} 彩色输出已禁用")


def _prepare_samples(args):
    """根据当前 CONFIG 生成输入源和分层样本

    关键行为：
      - pattern 模式固定 4×4（内置字符硬编码 16 像素），不受 VECTOR_GRID 影响
      - vector 模式读取 VECTOR_GRID（默认 8），支持 8/16/32/64
      - file 模式按数据集实际维度
      - 噪声模型由 CONFIG['NOISE_TYPE'] 驱动（create_noise_model 工厂）
    """
    from sgn_input import (
        create_default_source, create_file_source,
        create_vector_source, create_mixed_vector_source
    )

    source_type = CONFIG.get("INPUT_SOURCE_TYPE", "pattern")
    grid = CONFIG.get("VECTOR_GRID", 8)

    if source_type == "file":
        path = CONFIG.get("DATASET_PATH", args.dataset if hasattr(args, 'dataset') and args.dataset else None)
        if path:
            try:
                source = create_file_source(path)
            except Exception as e:
                print(f"  {C.err('X')} File load failed: {e}")
                print(f"  {C.info('i')} Auto fallback to built-in character mode")
                source = create_default_source()
        else:
            print(f"  {C.warn('⚠')} 未设置文件路径，回退到内置字符")
            source = create_default_source()
    elif source_type == "vector":
        formula = CONFIG.get("VECTOR_FORMULA", "line")
        if formula == "mixed":
            source = create_mixed_vector_source(grid_size=grid)
        else:
            source = create_vector_source(formula, grid_size=grid)
    else:
        # pattern 模式：始终 4×4，不扩展。内置字符是硬编码 16 像素，
        # 若需大网格请切换到 vector 模式或 file 模式。
        source = create_default_source()

    val_labels = set(CONFIG.get("VALIDATION_LABELS", []))
    test_split = CONFIG.get("TEST_SPLIT", 0.0)
    all_samples = source.generate_batch(2000, split='all')
    train_samples, test_samples = _stratified_split(all_samples, test_split, val_labels)

    # 噪声配置确认输出（帮助用户验证控制面板修改是否生效）
    noise_type = CONFIG.get("NOISE_TYPE", "composite")
    noise_prob = CONFIG.get("FLIP_PROB", 0.15)
    nt_label = {"composite": "复合", "gaussian": "高斯", "salt_pepper": "椒盐", "block": "块遮挡"}.get(noise_type, noise_type)
    log_print(f"  {C.info('ℹ')} 训练噪声配置: {C.val(nt_label)} (p={noise_prob}) | 输入源: {C.val(source_type)} | 网格: {C.val(getattr(source, 'grid_size', 4))}")

    return source, train_samples, test_samples


def _stratified_split(samples, test_split, val_labels):
    """分层留出：确保每个标签按比例拆分

    Args:
        samples: [(intensity, label), ...] 全部样本
        test_split: 留出比例（0.0-0.5）
        val_labels: 强制留出的字符列表

    Returns:
        (train_samples, test_samples)
    """
    from collections import defaultdict
    import random
    by_label = defaultdict(list)
    for s in samples:
        by_label[s[1]].append(s)

    train, test = [], []
    for lbl, lst in by_label.items():
        # 【v4.3-fix】每个标签内部先打乱，避免有序取样偏差
        random.shuffle(lst)
        if lbl in val_labels:
            # 强制留出：全部进 test
            test.extend(lst)
        else:
            # 按比例拆分，至少保留 1 个测试样本
            n_test = max(1, int(len(lst) * test_split)) if test_split > 0 else 0
            test.extend(lst[:n_test])
            train.extend(lst[n_test:])
    return train, test


def _write_crash_log(exc):
    """将异常堆栈写入日志文件，避免 IDE/无 TTY 环境下错误信息被吞"""
    import traceback, time
    ts = time.strftime("%Y%m%d_%H%M%S")
    crash_path = f"sgn_crash_{ts}.log"
    try:
        with open(crash_path, "w", encoding="utf-8") as f:
            f.write(f"SGN-Lite v5.0 崩溃日志\n时间: {ts}\n\n")
            f.write(f"异常类型: {type(exc).__name__}\n")
            f.write(f"异常信息: {exc}\n\n")
            traceback.print_exc(file=f)
        print(f"\n  [致命错误] 日志已保存: {crash_path}")
    except Exception:
        pass

def main():
    """主入口函数"""
    # v5.0 拆分后显式注册所有命令
    register_all_commands()
    parser = build_parser()
    args = parser.parse_args()

    # 1. 应用命令行参数（统一网关）
    apply_cli_args(args)

    # 2. 加载配置文件（如果指定）
    if args.config:
        ok, msg = cfg_load(args.config)
        if ok:
            log_print(f"  {C.ok('✓')} 配置已加载: {C.val(msg)}")
        else:
            log_print(f"  {C.err('✗')} 配置加载失败: {msg}")

    # 3. 设置日志文件
    if args.output:
        set_log_file(args.output)

    # 4. 欢迎信息
    welcome()

    # 5. 恢复中断训练（如果指定）
    resume_path = None
    if args.resume:
        resume_path = check_resume()

    # 6. 创建核心引擎
    seed = CONFIG.get("SEED", 42)
    core = SGNCore(seed)

    # 7. 导入模型（如果指定）
    if args.import_model:
        if load_model(core, args.import_model):
            log_print(f"  {C.info('ℹ')} 已加载模型，跳过训练阶段")
        else:
            log_print(f"  {C.warn('⚠')} 模型加载失败，将从头训练")

    # 8. 恢复自动保存（如果找到）
    if resume_path:
        load_model(core, resume_path)

    # 9. 初始化占位（控制面板切换后会重建）
    source = None
    train_samples = []
    test_samples = []

    # 10. 交叉验证模式
    if args.cross_validate:
        from sgn_utils import run_cross_validation
        log_print(f"\n  {C.BOLD}留一字符交叉验证模式{C.RST}")
        run_cross_validation(source, CONFIG.get("MAX_ITERATIONS", 640))
        close_log()
        return

    # 12. 仅测试模式
    if args.test_only:
        run_test_only(core)
        if args.export_model:
            save_model(core, args.export_model)
        close_log()
        return

    # 13. 主循环：支持多次进出控制面板（层级导航）
    while True:
        # 【fix】根据当前 CONFIG 重建 source 和样本（控制面板切换输入源后生效）
        source, train_samples, test_samples = _prepare_samples(args)

        # 检查是否需要重建网络（控制面板修改了架构参数）
        core = _maybe_rebuild_core(core)

        # 【fix】自动模式切换后跳过控制面板，直接进入训练
        if args.auto is None:
            # 显示控制面板，接收返回值
            cp_result = control_panel(core)
            if cp_result is None:
                # 【fix】用户按 q 退出控制面板，直接退出程序
                break

            # ========== 关键修复：控制面板返回后立即重新生成样本 ==========
            source, train_samples, test_samples = _prepare_samples(args)
            core = _maybe_rebuild_core(core)

        # 获取当前运行参数
        delay_ms = CONFIG.get("AUTO_DELAY_MS", 0)
        max_step = CONFIG.get("MAX_ITERATIONS", 640)

        result = None
        if args.batch:
            result = run_batch_mode(core, train_samples, max_step, delay_ms, test_samples=test_samples, source=source)
        elif args.auto is not None:
            result = run_auto_mode(core, train_samples, max_step, args.auto, test_samples=test_samples, source=source)
        else:
            # 交互模式
            try:
                current_step = train_batch(core, train_samples, max_step, delay_ms=0, test_samples=test_samples, source=source)
            except Exception as _e:
                import traceback, time
                ts = time.strftime("%Y%m%d_%H%M%S")
                crash_path = f"sgn_crash_{ts}.log"
                with open(crash_path, "w", encoding="utf-8") as _f:
                    _f.write(f"SGN-Lite v5.0 训练崩溃日志\n时间: {ts}\n\n")
                    _f.write(f"异常类型: {type(_e).__name__}\n")
                    _f.write(f"异常信息: {_e}\n\n")
                    traceback.print_exc(file=_f)
                print(f"\n  [致命错误] 训练过程崩溃，日志已保存: {crash_path}")
                traceback.print_exc()
                try:
                    input("\n  按回车退出...")
                except:
                    pass
                raise

            # 【fix】train_batch 返回整数步数，进入训练后菜单
            if isinstance(current_step, int):
                if current_step >= max_step:
                    result = _post_training_menu(core, train_samples, current_step, max_step,
                                                 test_samples=test_samples, source=source, is_paused=False)
                else:
                    result = _post_training_menu(core, train_samples, current_step, max_step,
                                                 test_samples=test_samples, source=source, is_paused=True)

        if result == "back":
            continue
        elif result == "reset":
            seed = CONFIG.get("SEED", 42)
            core = SGNCore(seed)
            continue
        elif result == "quit":
            break
        elif isinstance(result, tuple) and result[0] == "auto":
            args.auto = result[1]
            continue
        elif result == "continue":
            continue
        else:
            break

    # 14. 导出模型（如果指定）
    if args.export_model:
        save_model(core, args.export_model)

    close_log()
    log_print(f"\n  {C.DIM}再见!{C.RST}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  用户中断")
        import sys
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n  [致命错误] {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            input("\n  按回车退出...")
        except (EOFError, KeyboardInterrupt):
            pass
