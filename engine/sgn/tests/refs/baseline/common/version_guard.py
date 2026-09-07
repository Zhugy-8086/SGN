#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pysgn_net C 扩展版本守卫（副本）

本文件是 traditional/common/version_guard.py 的副本，放置在 baseline/common/ 的原因：
    train_cnn_template.py 的 sys.path 顺序为
        [traditional/common/, traditional/baseline/, traditional/, ...]
    当 Python 查找 `common` 包时，会优先命中 traditional/baseline/common/（有 __init__.py，
    常规包），而该目录原本只有 data_loader.py / metrics.py / evaluation.py / plotting.py，
    没有 version_guard.py，导致 `from common.version_guard import` 失败。
    将 version_guard.py 复制到 baseline/common/ 后，baseline/common/ 同时拥有
    data_loader / metrics / version_guard，train_cnn_template.py 的所有 `from common.* import`
    均能从同一包内解析成功。

正本：traditional/common/version_guard.py
副本：traditional/baseline/common/version_guard.py（本文件）
同步规则：修改正本后需同步更新本副本。
"""

from __future__ import annotations
import sys
import re
from typing import Tuple, Optional


# ===== 最低版本要求（按场景分级） =====

# 训练最低要求：必须有 set_omp_threads + sbe_rescale_to_triple_c
# 缺失会导致多线程调度失效 + Triple 缩放走 Python 单线程
MIN_VERSION_FOR_TRAINING = "1.9.0"

# 最低有意义的版本：有 C 扩展（仅 float matmul，无 OpenMP 优化）
MIN_VERSION_BASIC = "1.4.0"


# ===== 关键 API 清单（用于功能性检测） =====

# v1.9.0-triple-c 必须有的 API
REQUIRED_APIS_V19 = (
    "set_omp_threads",
    "get_omp_threads",
    "sbe_rescale_to_triple",
    "sbe_matmul",
)


def parse_version(version_str: str) -> Tuple[int, ...]:
    """解析版本字符串为可比较的元组"""
    if "-" in version_str:
        num_part, suffix_part = version_str.split("-", 1)
        suffix = tuple(suffix_part.split("-"))
    else:
        num_part = version_str
        suffix = ()

    nums = tuple(int(x) for x in num_part.split("."))
    return nums + suffix


def _get_pysgn_net():
    """获取 pysgn_net 模块（通过 Clang 编译的 sgn 模块）。"""
    try:
        import engine.sgn as _sgn
        return _sgn._native.hc8_net
    except (ImportError, AttributeError):
        return None


def get_pysgn_version() -> Optional[str]:
    """获取当前安装的 pysgn_net 版本"""
    pysgn_net = _get_pysgn_net()
    if pysgn_net is None:
        return None
    return getattr(pysgn_net, "__version__", "unknown")


def check_pysgn_apis() -> Tuple[bool, list]:
    """检查 pysgn_net 是否提供 v1.9.0 必需的 API"""
    pysgn_net = _get_pysgn_net()
    if pysgn_net is None:
        return False, list(REQUIRED_APIS_V19)

    missing = [api for api in REQUIRED_APIS_V19 if not hasattr(pysgn_net, api)]
    return len(missing) == 0, missing


def check_pysgn_version(
    min_version: str = MIN_VERSION_FOR_TRAINING,
    strict: bool = True,
    require_apis: bool = True,
) -> bool:
    """检查 pysgn_net C 扩展版本与 API"""
    version = get_pysgn_version()

    if version is None:
        msg = (
            "[version_guard] 严重警告：pysgn_net 未安装！\n"
            "  训练将走 Python float fallback，速度极慢。\n"
            "  请编译统一扩展：cmake -B engine/sgn/build -S engine/sgn && "
            "cmake --build engine/sgn/build（见 COMPILER_TOOLCHAIN.md）"
        )
        if strict:
            print(msg, file=sys.stderr)
            raise SystemExit(1)
        print(msg, file=sys.stderr)
        return False

    try:
        cur_ver = parse_version(version)
        min_ver = parse_version(min_version)
    except (ValueError, AttributeError) as e:
        print(f"[version_guard] 版本解析失败: {e}", file=sys.stderr)
        return False

    nums_cur = tuple(x for x in cur_ver if isinstance(x, int))
    nums_min = tuple(x for x in min_ver if isinstance(x, int))
    version_ok = nums_cur >= nums_min

    apis_ok = True
    missing_apis = []
    if require_apis:
        apis_ok, missing_apis = check_pysgn_apis()

    if version_ok and apis_ok:
        print(f"[version_guard] ✓ pysgn_net 版本 {version} >= {min_version}，API 齐全")
        return True

    problems = []
    if not version_ok:
        problems.append(f"版本过低: {version} < {min_version}")
    if not apis_ok:
        problems.append(f"缺失 API: {missing_apis}")

    msg = (
        "[version_guard] 严重警告：pysgn_net C 扩展不达标！\n"
        f"  当前版本: {version}\n"
        f"  最低要求: {min_version}\n"
        f"  问题: {'; '.join(problems)}\n"
        "\n"
        "  修复：\n"
        "    cmake -B engine/sgn/build -S engine/sgn && cmake --build engine/sgn/build\n"
        "    py -c \"import sgn; print(sgn.version())\"  # 验证"
    )

    if strict:
        print(msg, file=sys.stderr)
        raise SystemExit(1)
    print(msg, file=sys.stderr)
    return False


def get_environment_summary() -> str:
    """返回环境摘要字符串"""
    version = get_pysgn_version()
    apis_ok, missing = check_pysgn_apis()

    lines = [
        f"pysgn_net version: {version or 'NOT INSTALLED'}",
        f"Required APIs present: {apis_ok}",
    ]
    if not apis_ok:
        lines.append(f"Missing APIs: {missing}")

    if version is not None:
        try:
            pysgn_net = _get_pysgn_net()
            if pysgn_net is not None and hasattr(pysgn_net, "get_omp_threads"):
                omp_threads = pysgn_net.get_omp_threads()
                lines.append(f"OpenMP max threads: {omp_threads}")
        except Exception:
            pass

    import os
    cpu_count = os.cpu_count() or 4
    lines.append(f"CPU count: {cpu_count}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("pysgn_net C 扩展版本检查")
    print("=" * 60)
    print()
    print(get_environment_summary())
    print()

    ok = check_pysgn_version(strict=False)
    if ok:
        print("\n✓ 版本检查通过，可以开始训练")
        sys.exit(0)
    else:
        print("\n✗ 版本检查未通过，请修复后再训练", file=sys.stderr)
        sys.exit(1)
