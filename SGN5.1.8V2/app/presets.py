#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SGN-Lite 内置预设配置 + 预设应用辅助函数

CLI --preset 和主页 homepage() 共用此模块，避免重复定义。
"""

from __future__ import annotations


# ============================================================
# 内置预设配置
# ============================================================
PRESETS = {
    "quick":    {"MAX_NEURONS": 64,   "MAX_ITERATIONS": 500,   "FLIP_PROB": 0.15, "TOP_K": 4, "NOISE_TYPE": "composite"},
    "standard": {"MAX_NEURONS": 256,  "MAX_ITERATIONS": 2000,  "FLIP_PROB": 0.10, "TOP_K": 6, "NOISE_TYPE": "composite"},
    "precise":  {"MAX_NEURONS": 1024, "MAX_ITERATIONS": 10000, "FLIP_PROB": 0.05, "TOP_K": 8, "NOISE_TYPE": "composite"},
}


def apply_preset(preset_name, verbose=True):
    """应用预设配置到 ConfigRegistry

    Args:
        preset_name: 预设名称 (quick/standard/precise)
        verbose: 是否打印应用结果

    Returns:
        (ok, msg) — 应用是否成功
    """
    from engine.config import ConfigRegistry

    preset_data = PRESETS.get(preset_name)
    if not preset_data:
        return False, f"未知预设: {preset_name}"

    if verbose:
        from engine.utils import C, log_print
        log_print(f"  {C.info('i')} 应用预设: {C.val(preset_name)}")

    errors = []
    for key, val in preset_data.items():
        ok, msg = ConfigRegistry.set(key, val)
        if not ok:
            errors.append(f"{key}={val}: {msg}")
            if verbose:
                from engine.utils import C, log_print
                log_print(f"    {C.err('x')} {key} = {val} ({msg})")
        elif verbose:
            from engine.utils import C, log_print
            log_print(f"    {C.ok('!')} {key} = {C.val(val)}")

    if errors:
        return False, "; ".join(errors)
    return True, f"预设 {preset_name} 已应用"
