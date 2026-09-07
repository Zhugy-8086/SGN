# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""活跃测试的只读参考实现集合（refs）。

背景：
  legacy/traditional 下的 HC 树统一框架（hc_tree_unified.py）、SBE
  卷积参考实现（sbe_conv2d.py / hc_conv2d.py）、BitsAllocator 数值
  验证参考（stage_2_7_consolidation/exploration/exp34 / exp40）是
  SGN Lite 时代的 Python 参考实现，被活跃测试（engine/sgn/tests/）
  用作验证 C++ 扩展的数学基准。

  这些文件当年随 legacy 剥离被一并隔离，导致活跃测试引用
  `SGN_ROOT/traditional/...` 产生坏链。这里将**真正被活跃测试需要**
  的参考文件复制为本目录内容，活跃测试改为引用本目录，从而：
    - 活跃路径零引用 legacy/
    - legacy/traditional 原始文件保持冻结只读

2026-08-16 更新（legacy 彻底独立，可整体打包 7z）：
  - baseline/   ：PyTorch baseline 参考（cnn6_cifar10 等），自 legacy/traditional/baseline/ 复制
  - stage_2_4/  ：stage_2_4 层参考实现（core/sgn_layers、sgn_loss 等），自 legacy/traditional/stage_2_4_independent/ 复制
  - stage_1_3/  ：stage_1_3 层参考实现（hc_adapter、hc_matmul），自 legacy/traditional/stage_1_3_int_path/ 复制
  - data/       ：测试所需数据（.pt 权重、MNIST 原始 IDX），自 legacy 复制；
                  数据缺失时可用 scripts/download_test_data.py 恢复（联网下载）
  活跃层对 legacy/ 的引用已清零；legacy/ 仅剩冻结只读原始件。

  本目录为只读参考基准，不参与打包分发（见 pyproject packages）。
"""
