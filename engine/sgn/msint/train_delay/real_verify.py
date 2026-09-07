# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint×Level 旋钮调度验证 · 真实数据安全/迁移入口（薄包装）

语义归属：本脚本属 **MSint×Level 方向**（Level 旋钮层调度在真实数据上的
安全性 + static≈free 迁移性），为 Level 调度"编译期静态规划"提供真实背书。

实现：复用 `autograd/resnet8_free_roam.py` 的 `real` mode（协议训练循环 +
CIFAR 加载在 autograd，因依赖 A1/SR 反向链路）；本目录仅作 MSint 方向执行入口。

用法（cd engine/sgn）：
    python msint/train_delay/real_verify.py [T] [seeds_csv] [subset]
    例：python msint/train_delay/real_verify.py 1000 7 512
"""
from __future__ import annotations

import os
import sys

_AUTODIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'autograd'))
sys.path.insert(0, _AUTODIR)
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'build')))

import resnet8_free_roam as R  # noqa: E402


def main() -> int:
    T = sys.argv[1] if len(sys.argv) > 1 else "1000"
    seeds = sys.argv[2] if len(sys.argv) > 2 else "7"
    subset = sys.argv[3] if len(sys.argv) > 3 else "512"
    old = sys.argv[:]
    sys.argv = ["resnet8_free_roam.py", T, seeds, "real", subset]
    try:
        return R.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    sys.exit(main())