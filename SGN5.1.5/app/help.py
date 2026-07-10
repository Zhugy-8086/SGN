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
"""SGN-Lite v5.0 帮助手册模块 —— 从 sgn_interactive 拆分

纯文本输出，不依赖核心引擎。
"""

from __future__ import annotations


def do_help():
    """显示帮助手册"""
    from engine.utils import C, box, hr

    box("SGN-Lite v5.0 帮助手册")
    print(f"""{C.BOLD}SGN-Lite v5.0 完全离散神经网络系统{C.RST}

  核心流程:
  输入 → 二值化提取(≤4层) → 整数popcount匹配 → 整数响应排序
       → 三层校验(交叉相乘/稀疏保护) → 整数赫布学习
       → 分级模板合并(OR≥85% / AND≥80%)

  {C.BOLD}训练流程{C.RST}
  控制面板(一级) → [Enter] 开始训练 → 训练菜单(二级)
    → [Enter] 继续训练  /  [a] 切换自动模式  /  [q] 返回上级
  每20步自动暂停显示菜单，训练完成后进入完成菜单
  按 [a] 输入延迟(ms)后直接进入自动训练，不返回控制面板

  {C.BOLD}交互命令（训练菜单）{C.RST}
  [Enter] 继续训练  [a] 自动模式    [i] 推理测试    [t] 批量测试
  [s] 统计        [g] 仪表盘      [c] 混淆矩阵    [u] 模板可视化
  [m] 热力图      [n] 噪声测试    [p] 学习曲线    [x] 导出报告
  [v] 交叉验证    [o] 保存模型    [d] 加载模型    [w] 成功率疑问
  [b] 返回面板    [r] 重置网络    [h] 帮助        [q] 退出

  {C.BOLD}常用组合{C.RST}
  训练 → [t]测试 → [n]噪声 → [o]保存 → [q]退出
  [d]加载 → [c]混淆 → [n]噪声 → [x]导出报告
  控制面板 → [6]高级选项 → [2]矢量直线 → [Enter]训练


  {C.BOLD}控制面板（一级菜单）{C.RST}
  [1] 网络架构    [2] 学习参数    [3] 资源限制
  [4] 噪声参数    [5] 界面偏好    [6] 高级选项
  [e] 扩展功能    [s] 保存配置    [l] 加载配置
  [r] 恢复默认    [Enter] 开始训练  [q] 退出

  {C.BOLD}高级选项（输入源切换）{C.RST}
  [1] 内置字符模式 (8×8 插件化 0-9A-Z)
  [2] 矢量直线 (8×8)    [3] 矢量圆 (8×8)
  [4] 矢量正弦 (8×8)    [5] 混合矢量 (8×8)
  [6] 从文件加载 (CSV/JSON)    [g] 网格大小 (8/16/32/64)

  {C.BOLD}扩展功能{C.RST}
  [o] 保存模型    [d] 加载模型    [x] 导出报告
  [w] 成功率查询  [b] 图表后端    [s] 存储后端
  [a] 自动保存    [h] 钩子调试    [v] 噪声验证
  {C.BOLD}运行模式{C.RST}
  [k] 切换模式 — 全记载(full) / 精简(compact) / 黑箱(blackbox)
  全记载: 每步显示完整信息（默认）
  精简:   仅显示检查点摘要，大幅压缩输出
  黑箱:   训练全程零输出，结束后进入手动验证阶段
         体现SGN"训练与推理无严格分界"的教学本质

  {C.BOLD}黑箱模式设计意图{C.RST}
  SGN 的核心竞争机制依赖于神经元的响应速度（时间量纲）：
    响应速度 = 基础速度 + 匹配度×γ + 鼓励
  输出到屏幕、打印日志、甚至 sleep 都会引入不可控的时间延迟，
  扭曲真实的速度竞争关系，导致学习目标偏离。

  黑箱模式的特点：
  • 训练全程零输出（无任何 print、不绘制网格、不调用 time.sleep）
  • 保证训练过程的时间流纯净，让网络仅依赖"信号处理速度"学习
  • 这是 SGN 教学级严谨模式，体现了"训练与推理无严格分界"的本质
  • 训练完成后自动进入手动验证阶段，由用户判断网络真实能力

  注意：黑箱模式下无进度反馈是预期行为，不是 bug。

  控制面板 [0-9] 调整参数, [s/l] 保存/加载配置, [r] 恢复默认

  {C.BOLD}命令行参数{C.RST}
  python main.py --auto 50          自动模式(50ms/步)
  python main.py --batch            批量模式
  python main.py --test-only        仅测试(需配合 --import-model)
  python main.py --import-model f   加载已保存模型
  python main.py --export-model f   训练后导出模型
  python main.py --no-color         禁用彩色输出
  python main.py --output log.txt   输出日志到文件
  python main.py --resume           自动恢复中断的训练
  python main.py --config cfg.json  指定配置文件
  python main.py --mode blackbox    黑箱模式
  python main.py --mode compact     精简模式
  python main.py --input-source file --dataset data.csv  从CSV加载
  python main.py --vector-formula line --vector-grid 8   矢量直线8×8
  python main.py --vector-formula arch --vector-grid 16    矢量拱门16×16
  python main.py --vector-formula leaf --vector-grid 16    矢量叶片16×16
  {C.DIM}注: 'catear' 继续作为 'arch' 的别名支持 (旧配置/CLI自动映射){C.RST}
""")
    hr(46)
    print(f"\n  {C.BOLD}{C.CYN}常见问题快速修复{C.RST}\n")
    fixes = [
        ("看到乱码/方块", "使用 PowerShell/Windows Terminal，或执行: chcp 65001\n"
                           "           或添加 --no-color 禁用颜色输出"),
        ("matplotlib 报错", "安装命令: pip install matplotlib\n"
                            "           脚本会自动降级为ASCII曲线显示"),
        ("训练速度太慢", "使用自动模式: python main.py --auto 10\n"
                         "           或调大跳步显示(控制面板 [5] 界面偏好)"),
        ("想继续之前训练", "启动时使用: python main.py --resume\n"
                           "           或手动 [d] 加载模型文件"),
        ("颜色显示错乱", "添加 --no-color 参数强制禁用ANSI颜色\n"
                         "           脚本会自动检测终端支持情况"),
        ("训练步数设置", "控制面板 [3] 资源限制 → MAX_ITERATIONS"),
        ("Windows下路径问题", "使用正斜杠或双反斜杠: sgn_model.json\n"
                              "           脚本已使用 os.path 处理路径"),
        ("参数修改后没生效", "修改架构参数(神经元数/K值/种子)后返回控制面板再开始\n"
                             "           控制面板 [r] 是恢复默认配置，不是重置网络！"),
        ("切换矢量模式无效", "控制面板 [6] 高级选项 → 选择矢量模式 → [Enter]返回\n"
                              "           必须返回控制面板再开始训练才会生效"),
        ("按 [a] 又回到控制面板", "这是已修复的 121 结构问题，请更新到最新 main.py\n"
                                  "           更新后按 [a] 会直接开始自动训练"),
        ("训练后闪退无日志", "检查 sgn_crash_*.log 文件，或添加 --output log.txt\n"
                             "           确保所有 .py 文件无缩进错误(Tab/空格混用)"),
    ]
    for i, (problem, fix) in enumerate(fixes, 1):
        print(f"  {C.CYN}{i}.{C.RST} {C.BOLD}{problem}{C.RST}")
        print(f"     {C.GRN}→{C.RST} {fix}")
        print()
    hr(46)
    print(f"\n  {C.DIM}v5.0 改进: 模块拆分/职责分离/消除循环导入风险{C.RST}")
    print(f"\n  {C.DIM}按 Enter 返回菜单...{C.RST}")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def do_why_accuracy():
    """显示成功率与硬件约束的精简说明"""
    from engine.utils import C, box, hr

    box("成功率与硬件约束")
    hr(46)
    print(f"\n  {C.DIM}按 Enter 返回菜单...{C.RST}")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
