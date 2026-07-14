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
"""SGN-Lite v5.0 工具模块 - 颜色输出/环境检测/日志/输入处理"""

import sys
import os
import re
from engine.config import CONFIG  # 统一导入，避免函数内重复

# ============================================================
# 终端环境检测
# ============================================================

def detect_color_support():
    """检测终端是否支持ANSI颜色，返回布尔值"""
    # 优先读取 CONFIG（命令行 --no-color 通过 ConfigRegistry 设置）
    try:
        from engine.config import CONFIG
        if CONFIG.get("COLOR_OUTPUT") is False:
            return False
    except (ImportError, KeyError):
        pass
    if "--no-color" in sys.argv:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            hStdOut = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode)):
                ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                if not (mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING):
                    kernel32.SetConsoleMode(hStdOut, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
                return True
        except Exception:
            pass
        if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM") == "vscode":
            return True
        term = os.environ.get("TERM", "")
        return "xterm" in term or "ansi" in term
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def detect_encoding_issue():
    """检测Windows编码问题，返回提示字符串或None"""
    if sys.platform == "win32":
        import locale
        cp = locale.getpreferredencoding().lower()
        # 检查环境变量，若已设置UTF-8则跳过提示
        if os.environ.get("PYTHONIOENCODING", "").lower() == "utf-8":
            return None
        if cp not in ("utf-8", "utf_8", "65001"):
            return (
                f"\n  [提示] 当前终端编码为 {cp}，中文可能显示乱码。\n"
                f"         建议使用 PowerShell 或 Windows Terminal，\n"
                f"         或在 cmd 中执行: chcp 65001\n"
                f"         或设置环境变量: set PYTHONIOENCODING=utf-8\n"
            )
    return None


def clear_stdin_buffer():
    """清空输入缓冲区，避免残留回车跳过菜单

    【v4.3-fix】Windows 11 IDE/伪终端环境下，msvcrt 虽可导入但控制台句柄
    可能无效，导致底层访问违规（segfault）。添加最外层 except 防护，并
    限制最大清空次数避免无限循环。
    """
    try:
        import msvcrt
        # 防御：最多清空 100 个字符，避免无限循环；
        # 若 kbhit() 异常（伪终端句柄无效），直接跳到外层 except
        for _ in range(100):
            if msvcrt.kbhit():
                msvcrt.getch()
            else:
                break
    except ImportError:
        try:
            import termios, select
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, new)
            try:  # 【fix】确保无论是否异常都恢复终端
                while select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSANOW, old)
        except Exception:
            pass
    except Exception:
        # 【v4.3-fix】Windows 下 msvcrt 可能可用但控制台句柄无效
        # （IDE 内置终端、伪终端、远程 SSH 等），此时静默跳过清空。
        pass


# ============================================================
# 颜色输出类
# ============================================================

USE_COLOR = detect_color_support()

class C:
    """颜色输出类 - 不支持颜色时自动降级为空字符串"""
    RST  = "\033[0m"  if USE_COLOR else ""
    BOLD = "\033[1m" if USE_COLOR else ""
    DIM  = "\033[2m"  if USE_COLOR else ""
    RED  = "\033[31m" if USE_COLOR else ""
    GRN  = "\033[32m" if USE_COLOR else ""
    YEL  = "\033[33m" if USE_COLOR else ""
    BLU  = "\033[34m" if USE_COLOR else ""
    MAG  = "\033[35m" if USE_COLOR else ""
    CYN  = "\033[36m" if USE_COLOR else ""
    WHT  = "\033[37m" if USE_COLOR else ""

    @classmethod
    def refresh_colors(cls):
        """根据 CONFIG.COLOR_SCHEME 刷新网格高亮色"""
        if not USE_COLOR:
            return
        try:
            from engine.config import CONFIG
            scheme = CONFIG.get("COLOR_SCHEME", "green")
        except ImportError:
            return
        _color_map = {
            "green": "\033[32m",
            "cyan": "\033[36m",
            "yellow": "\033[33m",
            "white": "\033[37m",
        }
        cls.GRN = _color_map.get(scheme, "\033[32m")

    @staticmethod
    def title(t):  return f"{C.BOLD}{C.CYN}{t}{C.RST}"
    @staticmethod
    def ok(t):     return f"{C.GRN}{t}{C.RST}"
    @staticmethod
    def err(t):    return f"{C.RED}{t}{C.RST}"
    @staticmethod
    def warn(t):   return f"{C.YEL}{t}{C.RST}"
    @staticmethod
    def info(t):   return f"{C.BLU}{t}{C.RST}"
    @staticmethod
    def val(t):    return f"{C.BOLD}{C.WHT}{t}{C.RST}"
    @staticmethod
    def strip_ansi(text):
        return re.sub(r'\033\[[0-9;]*m', '', text)


def deprecated(msg="此函数将在 v4.3 中移除，请使用 sgn_input 中的新类"):
    """弃用装饰器"""
    import warnings as _warnings
    def decorator(func):
        def wrapper(*args, **kwargs):
            _warnings.warn(f"{func.__name__} 已弃用: {msg}", DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# ============================================================
# 输出格式化工具
# ============================================================

def box(t):
    print(f"\n{C.CYN}╔{'═'*46}╗{C.RST}\n{C.CYN}║{C.RST} {C.BOLD}{t:^44}{C.RST} {C.CYN}║{C.RST}\n{C.CYN}╚{'═'*46}╝{C.RST}")

def hr(n=46, c="─", col=C.DIM):
    print(f"  {col}{c*n}{C.RST}")

def mini_bar(v, max_v, w):
    return f"{C.GRN}{'█'*int(v/max_v*w)}{C.RST}" if max_v else ""

def progress_bar(cur, tot, w=20):
    p = cur/tot if tot else 0
    filled = int(p * w)
    return f"{'█'*filled}{'░'*(w-filled)} {p*100:.0f}%"


# ============================================================
# 日志系统
# ============================================================

_log_file = None

def set_log_file(path):
    """设置日志输出文件"""
    global _log_file
    path = os.path.normpath(path)
    try:
        _log_file = open(path, "w", encoding="utf-8")
        return True
    except Exception as e:
        print(f"  {C.warn('⚠')} 无法创建日志文件: {e}")
        return False

def log_print(text="", end="\n"):
    """同时输出到终端和日志文件"""
    clean = C.strip_ansi(text)
    if USE_COLOR:
        print(text, end=end)
    else:
        print(clean, end=end)
    if _log_file:
        _log_file.write(clean + end)
        _log_file.flush()

def close_log():
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None


# ============================================================
# 层处理工具 (供核心和可视化共用)
# ============================================================

# 【v4.3 拆分】popcount 已移至 sgn_layers.py


# 【v4.3 拆分】match_bits 已移至 sgn_layers.py


# 【v4.3 拆分】extract_layers 已移至 sgn_layers.py


# ============================================================
# 模板匹配提取（v4.3 新增：消除 6 处重复逻辑）
# ============================================================

# 【v4.3 拆分】classify_sample 已移至 sgn_layers.py



# 【v4.3 拆分】combine_layers 已移至 sgn_layers.py


@deprecated("此函数将在 v4.3 中移除，请使用 sgn_input.PatternInputSource")
def gen_samples(label, count, noise_prob=0.15):
    """生成带复合噪声的训练样本（废弃兼容层）

    内部转发到 sgn_input.PatternInputSource + DefaultCompositeNoise，
    确保新旧代码行为一致，避免双轨制分叉。
    """
    from engine.input import PatternInputSource, DefaultCompositeNoise
    from engine.config import PATTERNS

    noise = DefaultCompositeNoise(noise_prob)
    # 创建仅含该标签的 PatternInputSource
    source = PatternInputSource(
        {label: PATTERNS[label]},
        noise,
        labels=[label]
    )
    return source.generate_batch(count, split='train')



# ============================================================
# 模式输出辅助
# ============================================================

def compact_banner(mode_name):
    """显示模式横幅"""
    print(f"\n  {C.BOLD}{C.CYN}[{mode_name}模式]{C.RST}")

def compact_step(step, max_step, info):
    """精简模式单行输出"""
    v_mark = f"{C.GRN}✓{C.RST}" if info["V"] else f"{C.RED}✗{C.RST}"
    if info.get("multi_layer"):
        # Bug #8 修复：多层模式精简输出 L0/L1 + 图节点（旧代码打印占位 base=0/lck=0）
        l0 = info.get("layer0_active", 0)
        l1 = info.get("layer1_active", 0)
        gn = info.get("graph_nodes", 0)
        print(f"  [{step:>5}/{max_step}] {v_mark} L={info['label']} match={info['match']:>3} "
              f"L0={l0:<3} L1={l1:<3} G={gn:<3} tpl={info['templates']:>3}")
        return
    base_val = info['base']
    if isinstance(base_val, int):
        base_str = f"{base_val:.3f}"
    elif hasattr(base_val, 'index'):
        base_str = f"{base_val.index / base_val.scale:.3f}"
    else:
        base_str = f"{base_val:.3f}"
    print(f"  [{step:>5}/{max_step}] {v_mark} L={info['label']} match={info['match']:>3} "
          f"base={base_str} act={info['active']:>3} lck={info['locked']:>3} tpl={info['templates']:>3}")

def compact_summary(core, step):
    """精简模式训练完成摘要"""
    v = sum(1 for x in core.history if x["V"])
    pct = v / step * 100 if step > 0 else 0
    st = core.get_state()
    box(f"训练完成 ({step}步)")
    print(f"  校验通过: {C.GRN}{v}{C.RST}/{step} ({pct:.1f}%)")
    print(f"  活跃:{C.GRN}{st['active']}{C.RST} 锁定:{C.YEL}{st['locked']}{C.RST} "
          f"模板:{C.CYN}{st['templates']}{C.RST}/{CONFIG['MAX_TEMPLATES']}")

def blackbox_banner():
    """黑箱模式启动提示"""
    box("黑箱模式")
    print(f"  {C.DIM}训练阶段全程无输出，结束后进入手动验证{C.RST}")
    print(f"  {C.DIM}这是SGN最严谨的阶段：训练与推理无分界{C.RST}")

def blackbox_complete(step):
    """黑箱训练完成提示"""
    box(f"黑箱训练完成 ({step}步)")
    print(f"  {C.BOLD}网络已内化 {step} 步训练经验{C.RST}")
    print(f"  {C.DIM}接下来进入手动验证阶段，检验网络真实能力{C.RST}")


# ============================================================
# v4.2 兼容别名 - 指向新的 InputSource 类
# ============================================================

def _get_default_extractor():
    """获取默认层提取器"""
    try:
        from engine.input import DefaultLayerExtractor
        return DefaultLayerExtractor()
    except ImportError:
        return None


def extract_layers_v2(intensity, layer_max=4, d=16):
    """v4.2 兼容版本：使用 DefaultLayerExtractor"""
    extractor = _get_default_extractor()
    if extractor:
        return extractor.extract(intensity)
    # 降级到旧实现
    return extract_layers(intensity, layer_max, d)


# ============================================================
# 弃用警告装饰器
# ============================================================



# ============================================================
# v4.3 转发层 —— 从 sgn_layers 导入核心算法，保持向后兼容
# ============================================================

from engine.layers import popcount, match_bits, extract_layers, combine_layers, classify_sample


def _classify_unified(core, intensity, d=None):
    """统一分类入口 - 自动适配模板模式和图模式"""
    if d is None:
        d = getattr(core, 'D', 16)
    if getattr(core, 'graph_mode', False):
        from graph.graph_match import classify_with_graph
        return classify_with_graph(core, intensity, d)
    else:
        return classify_sample(core, intensity, d)

# v4.2 注意: extract_layers 已内部接入 DefaultLayerExtractor，
# 核心引擎调用时自动走新抽象。此处不再标记 @deprecated，
# 避免高频路径触发 DeprecationWarning 污染日志。
# 旧函数的完全移除计划在 v4.3 中进行。

# ============================================================
# 交叉验证（v4.3 从 main.py 移入，消除循环导入）
# ============================================================

def run_cross_validation(source, max_step):
    """留一字符交叉验证（LOLO）

    每折：
    1. 留出一个字符完全不参与训练
    2. 用剩余字符训练 max_step 步
    3. 对留出字符做零样本测试
    4. 记录结果，销毁网络
    """
    from engine.config import LABELS, PATTERNS, CONFIG
    from engine.input import PatternInputSource, DefaultCompositeNoise
    from engine.core import SGNCore
    import random
    from engine.utils import log_print, C

    # 【fix】从传入的 source 获取标签，而非硬编码 LABELS
    if hasattr(source, 'patterns') and source.patterns:
        labels = list(source.patterns.keys())
        patterns = source.patterns
    elif hasattr(source, '_samples') and source._samples:
        labels = sorted(set(lb for _, lb in source._samples))
        patterns = None
    else:
        labels = LABELS
        patterns = PATTERNS

    folds = CONFIG.get("CROSS_VALIDATE_FOLDS", 16)
    if folds <= 0:
        folds = len(labels)

    results = []
    for held_out in labels[:folds]:
        log_print(f"\n  {C.CYN}交叉验证折: 留出 [{held_out}]{C.RST}")

        # 1. 重建网络（必须全新，不能复用旧模板）
        core = SGNCore(CONFIG.get("SEED", 42))

        # 2. 生成训练样本（排除 held_out）
        if patterns is not None:
            train_source = PatternInputSource(
                patterns,
                DefaultCompositeNoise(CONFIG.get("FLIP_PROB", 0.15)),
                labels=[l for l in labels if l != held_out],
                validation_labels=[held_out]
            )
        else:
            # 对于无 patterns 的源（如矢量/文件），用 source 生成后过滤
            all_samples = source.generate_batch(max_step, split='all')
            train_samples = [(s, l) for s, l in all_samples if l != held_out]
            test_samples = [(s, l) for s, l in all_samples if l == held_out][:40]
            random.shuffle(train_samples)
            # 直接训练，跳过 PatternInputSource 包装
            for intensity, label in train_samples:
                core.train(intensity, label)
            # 测试
            correct = 0
            total = 0
            for intensity, label in test_samples:
                pred_lb, best_s = _classify_unified(core, intensity)
                if pred_lb == label and best_s >= 80:
                    correct += 1
                total += 1
            acc = (correct / total * 100) if total else 0
            results.append({
                "held_out": held_out,
                "accuracy": acc,
                "templates": len(core.templates)
            })
            log_print(f"  {C.GRN}✓{C.RST} [{held_out}] 零样本准确率: {C.val(f'{acc:.1f}%')} (模板数: {len(core.templates)})")
            continue

        train_samples = train_source.generate_batch(max_step, split='train')
        random.shuffle(train_samples)

        # 3. 训练
        for intensity, label in train_samples:
            core.train(intensity, label)

        # 4. 测试留出字符（零样本）
        test_samples = train_source.generate_batch(40, split='test')
        correct = 0
        total = 0
        for intensity, label in test_samples:
            pred_lb, best_s = _classify_unified(core, intensity)
            if pred_lb == label and best_s >= 80:
                correct += 1
            total += 1
        acc = (correct / total * 100) if total else 0

        results.append({
            "held_out": held_out,
            "accuracy": acc,
            "templates": len(core.templates)
        })
        log_print(f"  {C.GRN}✓{C.RST} [{held_out}] 零样本准确率: {C.val(f'{acc:.1f}%')} (模板数: {len(core.templates)})")

    # 汇总
    avg_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0
    log_print(f"\n{C.BOLD}  LOLO 平均泛化准确率: {C.GRN}{avg_acc:.1f}%{C.RST}")
    return results


# ============================================================
# 颜色方案变更钩子
# ============================================================

def _on_color_scheme_change(key, old, new):
    if key == "COLOR_SCHEME":
        C.refresh_colors()

try:
    from engine.hooks import HookRegistry
    HookRegistry.register("sgn:on_config_changed", _on_color_scheme_change, weak=False)
except ImportError:
    pass

