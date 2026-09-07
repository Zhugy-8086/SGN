# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""MSint×Level 旋钮调度验证 · 剂量响应入口（薄包装）

语义归属：本脚本属 **MSint×Level 方向**（Level 旋钮层调度的剂量-收敛行为），
是 R6 单拷贝自由漫游协议（HC 系列）在 MSint 方向上的调度机理验证。

实现：复用 `autograd/resnet8_free_roam.py` 的 `dose` mode（协议训练循环在
autograd，因依赖 A1/SR 反向链路）；本目录仅作 MSint 方向的执行入口与语义归置。

用法（cd engine/sgn）：
    python msint/train_delay/dose_sweep.py [T] [seeds_csv] [P8_list_csv]
    例：python msint/train_delay/dose_sweep.py 1500 7,11 0.5,0.7,0.85,1.0
"""
from __future__ import annotations

import os
import sys

# 依赖 autograd 的协议训练循环（A1/SR 反向链路所在），构建路径
_AUTODIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'autograd'))
sys.path.insert(0, _AUTODIR)
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'build')))

import resnet8_free_roam as R  # noqa: E402


def main() -> int:
    # resnet8_free_roam.main 直接读 sys.argv[1..4]，构造对应 argv 后转交。
    T = sys.argv[1] if len(sys.argv) > 1 else "1500"
    seeds = sys.argv[2] if len(sys.argv) > 2 else "7,11"
    p8 = sys.argv[3] if len(sys.argv) > 3 else "0.5,0.7,0.85,1.0"
    old = sys.argv[:]
    sys.argv = ["resnet8_free_roam.py", T, seeds, "dose", p8]
    try:
        return R.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    sys.exit(main())