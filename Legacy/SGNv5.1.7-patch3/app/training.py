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
"""SGN-Lite v5.0 训练循环模块 —— 从 sgn_interactive 拆分

包含：训练循环、输出格式、模式辅助函数
"""

from __future__ import annotations

import sys
import time
from typing import List, Tuple, Optional


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


def header():
    """训练输出表头"""
    from engine.utils import C, hr
    headers = ["步数", "校验", "字符", "层", "匹配", "速度", "活跃", "锁定", "模板"]
    widths = [6, 4, 4, 3, 5, 6, 5, 5, 5]
    print(f"  {C.BOLD}{' '.join(h.rjust(w) for h, w in zip(headers, widths))}{C.RST}")
    hr(46)


def out_step(step, info):
    """单步训练输出"""
    from engine.utils import C

    v = f"{C.GRN}✓{C.RST}" if info["V"] else f"{C.RED}✗{C.RST}"

    if info.get("graph_mode"):
        # 图模式输出
        graph_nodes = info.get("graph_nodes", 0)
        label = info.get("label", "?")
        match_val = info.get("match", 0)
        print(f"  {step:>5} {v:>3}  {label:>3}  {'G':>2}   {match_val:>4}  {'':>5}  {graph_nodes:>4}  {'':>4}   {info.get('templates', 0):>4}")
    elif info.get("multi_layer"):
        # Bug #8 修复：多层模式输出 L0/L1 激活数 + 图节点数（旧代码当真实值打印占位 0）
        label = info.get("label", "?")
        match_val = info.get("match", 0)
        l0 = info.get("layer0_active", 0)
        l1 = info.get("layer1_active", 0)
        gn = info.get("graph_nodes", 0)
        print(f"  {step:>5} {v:>3}  {label:>3}  {'M':>2}   {match_val:>4}  L0={l0:<3} L1={l1:<3}  G={gn:<3}  {info.get('templates', 0):>4}")
    else:
        base_val = info.get('base', 0)
        if isinstance(base_val, int):
            base_str = f"{base_val:>5}"
        elif hasattr(base_val, 'index'):
            base_str = f"{base_val.index:>5}"
        else:
            base_str = f"{base_val:>5.3f}"
        print(f"  {step:>5} {v:>3}  {info['label']:>3}  {info['layer_count']:>2}   "
              f"{info['match']:>4}  {base_str}  {info['active']:>4}  "
              f"{info['locked']:>4}   {info['templates']:>4}")
    for w in info.get("warnings", []):
        try:
            from engine.utils import C
            print(f"  {C.RED}⚠ {w}{C.RST}")
        except ImportError:
            print(f"  ⚠ {w}")


def run_training_loop(core, samples, max_step, delay_ms=0, test_samples=None, source=None):
    """训练循环 - v5.1.6 分批次训练

    v4.3 拆分后由 main.py 调用
    v5.1.6: 重构为分批次训练（BATCH_TRAIN_ENABLED），由 core.train_batch 批次投喂；
            关闭批次时回退到逐样本 core.train（向后兼容）。
    """
    from engine.config import mark_config_synced, is_config_modified, CONFIG
    from engine.hooks import HookRegistry
    from app.persist import autosave_check
    from engine.utils import C, box
    import time as _time
    import random as _random

    train_start = _time.time()

    max_step = CONFIG.get("MAX_ITERATIONS", max_step)
    step = len(core.history)
    if step >= max_step:
        print(f"\n  {C.YEL}⚠ 当前已训练 {step} 步，MAX_ITERATIONS={max_step}{C.RST}")
        print(f"  {C.info('ℹ')} 如需继续训练，请在控制面板调大训练步数 [0]")
        return step

    auto = delay_ms > 0
    mode = CONFIG.get("MODE", "full")
    compact_interval = max(10, CONFIG.get("COMPACT_INTERVAL", 100))

    # v5.1.6 分批次训练参数
    batch_enabled = CONFIG.get("BATCH_TRAIN_ENABLED", True)
    batch_size = CONFIG.get("BATCH_SIZE", 32)
    do_shuffle = CONFIG.get("BATCH_SHUFFLE", True)
    seed = CONFIG.get("BATCH_SHUFFLE_SEED", 0)

    if is_config_modified():
        print(f"\n  {C.YEL}⚠ 检测到配置变更{C.RST}")
        print(f"  {C.info('ℹ')} 架构参数需重建网络才生效")
        print(f"  {C.info('ℹ')} 非架构参数已立即生效")

    if not auto:
        if mode == "blackbox":
            from engine.utils import blackbox_banner
            try:
                blackbox_banner()
            except Exception:
                print("\n=== 黑箱模式 ===")
            try:
                c = input(f"  {C.DIM}黑箱训练 {max_step - step} 步，按Enter开始，q退出{C.RST}\n> ").strip()
                if c.lower() == "q":
                    return step
            except (EOFError, KeyboardInterrupt):
                return step
        elif mode == "compact":
            from engine.utils import compact_banner
            try:
                compact_banner("精简")
            except Exception:
                print("\n[精简模式]")
            print(f"  准备训练: {max_step - step} 步 (按Enter开始, q退出)")
            try:
                c = input("> ").strip()
                if c.lower() == "q":
                    return step
            except (EOFError, KeyboardInterrupt):
                return step
        else:
            print(f"\n  {C.info('ℹ')} 准备训练: {max_step - step} 步 (按Enter开始, q退出)")
            try:
                c = input("> ").strip()
                if c.lower() == "q":
                    return step
            except (EOFError, KeyboardInterrupt):
                return step

    if mode != "blackbox":
        header()

    old_template_count = len(core.templates)
    old_graph_count = len(core.graphs) if hasattr(core, 'graphs') else 0

    # ---- v5.1.6: 单步后处理（批次/单样本共用）----
    def _post_step(intensity, info):
        """处理一步训练后的钩子/输出/自动保存（批次与单样本共用）"""
        nonlocal step, old_template_count, old_graph_count
        HookRegistry.emit("sgn:after_step", step=step, info=info, core=core)

        if len(core.templates) > old_template_count:
            new_sig = core.templates[-1][1] if core.templates else 0
            HookRegistry.emit("sgn:on_template_added",
                              label=info.get("label", "?"),
                              signature=new_sig,
                              template_count=len(core.templates))
            old_template_count = len(core.templates)
        # 图模式：检测图数变化
        if hasattr(core, 'graphs') and len(core.graphs) > old_graph_count:
            old_graph_count = len(core.graphs)

        cur_mode = CONFIG.get("MODE", "full")
        if cur_mode == "full":
            out_step(step, info)
            sys.stdout.flush()
            if auto:
                time.sleep(delay_ms / 1000.0)

        elif cur_mode == "compact":
            if step % compact_interval == 0 or step >= max_step:
                from engine.utils import compact_step
                try:
                    compact_step(step, max_step, info)
                except ImportError:
                    print(f"  [{step:>5}/{max_step}] match={info['match']:>3} templates={info['templates']:>3}")
                sys.stdout.flush()
            if auto:
                time.sleep(delay_ms / 1000.0)
                if step % compact_interval == 0:
                    sys.stdout.flush()

        else:  # blackbox：零输出、无延迟，保证竞争纯粹性
            pass

        autosave_check(core, step)

    # ---- 训练主循环 ----
    if batch_enabled and len(samples) > 0:
        # v5.1.6 分批次训练：构建样本池 → 每轮 shuffle → 分批投喂 core.train_batch
        rng = _random.Random(seed) if seed else _random.Random()
        sample_pool = core.build_sample_pool(samples)
        if not sample_pool:
            sample_pool = list(samples)

        while step < max_step:
            if do_shuffle:
                rng.shuffle(sample_pool)
            for batch_start in range(0, len(sample_pool), batch_size):
                if step >= max_step:
                    break
                batch = sample_pool[batch_start:batch_start + batch_size]
                # 批次内每样本 before_step 钩子
                for s in batch:
                    HookRegistry.emit("sgn:before_step", intensity=s[0], label=s[1], step=step, core=core)
                batch_infos = core.train_batch(batch)
                for s, info in zip(batch, batch_infos):
                    core.history.append(info)
                    step += 1
                    _post_step(s[0], info)
                    # 非自动模式：到达 max_step 或每20步暂停 → 提前返回（不打印完成摘要）
                    if not auto and step >= max_step:
                        return step
                    if not auto and mode == "full" and (step % 20 == 0):
                        return step
                    if step >= max_step:
                        break  # 自动模式：退出批次迭代，进入完成摘要
                if step >= max_step:
                    break
    else:
        # 单样本训练（向后兼容：BATCH_TRAIN_ENABLED=False 或无样本）
        while step < max_step:
            mode = CONFIG.get("MODE", "full")
            intensity, label = samples[step % len(samples)]
            HookRegistry.emit("sgn:before_step", intensity=intensity, label=label, step=step, core=core)
            info = core.train(intensity, label)
            core.history.append(info)
            step += 1
            _post_step(intensity, info)

            if not auto and step >= max_step:
                return step
            if not auto and mode == "full" and (step % 20 == 0):
                return step

    # 训练完成处理
    train_elapsed = _time.time() - train_start
    if mode == "blackbox":
        from engine.utils import blackbox_complete
        try:
            blackbox_complete(step)
        except ImportError:
            print(f"\n=== 黑箱训练完成 ({step}步) ===")
        print(f"  训练耗时: {C.val(_format_duration(train_elapsed))} ({step/train_elapsed:.0f} 步/秒)")
    elif mode == "compact":
        from engine.utils import compact_summary
        try:
            compact_summary(core, step)
        except ImportError:
            print(f"\n=== 训练完成 ({step}步) ===")
        print(f"  训练耗时: {C.val(_format_duration(train_elapsed))} ({step/train_elapsed:.0f} 步/秒)")
    else:
        box("训练完成")
        v = sum(1 for x in core.history if x["V"])
        t = step
        pct = v / t * 100 if t > 0 else 0
        st = core.get_state()
        print(f"  总步数: {C.val(t)}")
        print(f"  训练耗时: {C.val(_format_duration(train_elapsed))} ({step/train_elapsed:.0f} 步/秒)")
        print(f"  校验通过: {C.GRN}{v}{C.RST}/{t} ({C.GRN}{pct:.1f}%{C.RST})")
        if getattr(core, 'graph_mode', False):
            total_nodes = sum(g.get_total_nodes() for g in core.graphs.values()) if hasattr(core, 'graphs') else 0
            print(f"  活跃:{C.GRN}{st['active']}{C.RST} 图:{C.CYN}{st['templates']}{C.RST} 节点:{C.val(total_nodes)} {C.CYN}[图模式]{C.RST}")
        else:
            print(f"  活跃:{C.GRN}{st['active']}{C.RST} 锁定:{C.YEL}{st['locked']}{C.RST} 模板:{C.CYN}{st['templates']}{C.RST}/{CONFIG['MAX_TEMPLATES']}")

    return step
