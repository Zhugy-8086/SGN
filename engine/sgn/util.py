# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""sgn.util — 工具箱

提供轻量级实用工具：
  - ensure_built(): 检查 .pyd 是否最新构建
  - Timer: 简单计时器（上下文管理器）
  - describe(): 打印参数/张量详细信息
"""

from __future__ import annotations

import os
import time
from datetime import datetime


def ensure_built() -> bool:
    """检查 sgn.*.pyd 是否存在且所有源文件不新于 .pyd。

    Returns:
        True 如果 .pyd 存在且所有源文件均为最新（无需重新编译）。
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    _build_dir = os.path.join(_here, "build")

    if not os.path.isdir(_build_dir):
        print(f"ERROR: build 目录不存在: {_build_dir}")
        print("请先编译: cmake -B build && cmake --build build")
        return False

    pyds = [f for f in os.listdir(_build_dir)
            if f.startswith("sgn.") and f.endswith(".pyd")]
    if not pyds:
        print(f"ERROR: 未找到 sgn.*.pyd 在 {_build_dir}")
        print("请先编译: cmake --build build")
        return False

    pyd_path = os.path.join(_build_dir, pyds[0])
    pyd_mtime = os.path.getmtime(pyd_path)
    mtime = datetime.fromtimestamp(pyd_mtime)

    print(f"sgn.pyd: {pyd_path}")
    print(f"  Build time: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Size: {os.path.getsize(pyd_path) / 1024:.1f} KB")

    # 检查源文件是否比 .pyd 新
    src_dirs = [
        os.path.join(_here, "autograd"),
        os.path.join(_here, "hc"),
        os.path.join(_here, "common"),
        os.path.join(_here, "level"),
        os.path.join(_here, "msint"),
    ]

    stale = []
    for src_dir in src_dirs:
        if not os.path.isdir(src_dir):
            continue
        for root, _, files in os.walk(src_dir):
            for f in files:
                if f.endswith(('.cpp', '.h', '.c')):
                    src_path = os.path.join(root, f)
                    if os.path.getmtime(src_path) > pyd_mtime:
                        rel = os.path.relpath(src_path, _here)
                        stale.append(rel)

    if stale:
        print("\nWARNING: 以下源文件比 .pyd 新，建议重新编译:")
        for s in stale[:10]:  # 最多显示 10 个
            print(f"  {s}")
        if len(stale) > 10:
            print(f"  ... 还有 {len(stale) - 10} 个文件")
        return False

    print("  All source files are up-to-date.")
    return True


class _BuildLock:
    """跨进程构建锁（安全审计 2026-08-16 M2）。

    防止两个 Python 进程同时 import 触发并发 cmake 构建损坏 build/。
    Windows 用 msvcrt.locking，POSIX 用 fcntl.flock；同目录 lockfile。
    拿不到锁时阻塞等待（对端构建完成后重新检测 .pyd 新鲜度即可）。
    """

    def __init__(self, lock_path: str):
        self._lock_path = lock_path
        self._fh = None

    def __enter__(self):
        import time
        self._fh = open(self._lock_path, "a+")
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.5)  # 对端持有锁，等待重试

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
        return False


def auto_build(verbose: bool = True) -> bool:
    """自动编译 sgn.*.pyd（如果不存在或源文件过时）。

    在 import sgn 时自动调用，无需用户手动编译。
    仅在开发场景（源码目录存在 CMakeLists.txt）下生效；
    已 pip install 的场景跳过（.pyd 已在包目录中）。

    Returns:
        True 如果 .pyd 已就绪（无需编译或编译成功）。
    """
    import subprocess

    _here = os.path.dirname(os.path.abspath(__file__))
    _build_dir = os.path.join(_here, "build")
    _cmake_file = os.path.join(_here, "CMakeLists.txt")

    # 非开发场景（无 CMakeLists.txt），跳过
    if not os.path.isfile(_cmake_file):
        if verbose:
            print("[sgn] 非开发环境，跳过自动编译")
        return True

    # 检查是否需要编译
    need_build = False
    if not os.path.isdir(_build_dir):
        need_build = True
        if verbose:
            print("[sgn] build 目录不存在，自动配置 cmake ...")
    else:
        pyds = [f for f in os.listdir(_build_dir)
                if f.startswith("sgn.") and f.endswith(".pyd")]
        if not pyds:
            need_build = True
            if verbose:
                print("[sgn] .pyd 不存在，自动编译 ...")
        else:
            pyd_mtime = os.path.getmtime(os.path.join(_build_dir, pyds[0]))
            # 快速检查：CMakeLists.txt 是否比 .pyd 新
            if os.path.getmtime(_cmake_file) > pyd_mtime:
                need_build = True
                if verbose:
                    print("[sgn] CMakeLists.txt 已更新，自动重新编译 ...")
            else:
                # 检查源文件
                for src_dir_name in ["autograd", "hc", "common", "level", "msint"]:
                    src_dir = os.path.join(_here, src_dir_name)
                    if not os.path.isdir(src_dir):
                        continue
                    for root, _, files in os.walk(src_dir):
                        for f in files:
                            if f.endswith(('.cpp', '.h', '.c')):
                                if os.path.getmtime(os.path.join(root, f)) > pyd_mtime:
                                    need_build = True
                                    break
                        if need_build:
                            break
                    if need_build:
                        break
                if need_build and verbose:
                    print("[sgn] 源文件已更新，自动重新编译 ...")

    if not need_build:
        return True

    # 安全审计 2026-08-16 M2：并发 import 触发的并发构建用文件锁串行化
    #（两进程同时 cmake 配置/构建可能损坏 build/ 目录）。
    # 拿到锁后无需重检测——cmake --build 幂等，up-to-date 时为 no-op
    os.makedirs(_build_dir, exist_ok=True)
    with _BuildLock(os.path.join(_build_dir, ".build.lock")):
        # 执行 cmake 配置（如果需要）
        if not os.path.isfile(os.path.join(_build_dir, "CMakeCache.txt")):
            try:
                subprocess.run(
                    ["cmake", "-B", "build", "-S", "."],
                    cwd=_here, check=True,
                    capture_output=not verbose,
                )
            except subprocess.CalledProcessError as e:
                print(f"[sgn] cmake 配置失败: {e}")
                if e.stderr:
                    print(f"  stderr: {e.stderr.decode(errors='replace')[:500]}")
                return False
            except FileNotFoundError:
                print("[sgn] 未找到 cmake，请安装 CMake 后手动编译")
                return False

        # 执行 cmake 编译
        try:
            subprocess.run(
                ["cmake", "--build", "build", "--config", "Release"],
                cwd=_here, check=True,
                capture_output=not verbose,
            )
            if verbose:
                print("[sgn] 编译成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[sgn] 编译失败: {e}")
            if e.stderr:
                print(f"  stderr: {e.stderr.decode(errors='replace')[:500]}")
            return False
        except FileNotFoundError:
            print("[sgn] 未找到 cmake，请安装 CMake 后手动编译")
            return False


class Timer:
    """简单计时器，支持上下文管理器。

    Usage:
        with sgn.util.Timer("forward"):
            y = model.forward(x)

        t = sgn.util.Timer("training")
        t.start()
        # ... work ...
        t.stop()
        print(t.elapsed_ms)
    """

    def __init__(self, label: str = "Timer"):
        self.label = label
        self._start: float | None = None
        self._elapsed: float = 0.0

    def start(self) -> Timer:
        """开始计时。"""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """停止计时，返回耗时（秒）。"""
        if self._start is not None:
            self._elapsed = time.perf_counter() - self._start
            self._start = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        """已计时间（秒）。"""
        return self._elapsed

    @property
    def elapsed_ms(self) -> float:
        """已计时间（毫秒）。"""
        return self._elapsed * 1000.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        print(f"[{self.label}] {self.elapsed_ms:.2f} ms")

    def __repr__(self):
        return f"Timer(label='{self.label}', elapsed={self.elapsed_ms:.2f}ms)"


def describe(param) -> None:
    """打印参数/张量的详细信息（shape、numel、dtype、requires_grad）。

    支持 sgn.nn.Parameter、sgn.autograd.Tensor、sgn.nn.Buffer 等类型。

    Usage:
        sgn.util.describe(model.fc1_w)
        # Tensor(shape=(128, 784), numel=100352, dtype=float32, requires_grad=True)
    """
    # 尝试获取 shape（Parameter 有 .shape 属性，Tensor 有 .shape 属性）
    try:
        shape = tuple(param.shape)  # pybind11 返回 list，转 tuple
    except (AttributeError, TypeError):
        try:
            shape = tuple(param.shape())
        except (AttributeError, TypeError):
            shape = "?"

    # 尝试获取 numel
    try:
        numel = param.numel
    except AttributeError:
        try:
            numel = param.numel()
        except AttributeError:
            numel = "?"

    # 尝试获取 requires_grad
    try:
        rg = param.requires_grad
    except AttributeError:
        try:
            rg = param.requires_grad()
        except AttributeError:
            rg = "?"

    # 类型名
    type_name = type(param).__name__

    print(f"{type_name}(shape={shape}, numel={numel}, dtype=float32, requires_grad={rg})")


def count_parameters(model) -> int:
    """返回模型的总可学习参数量（含子模块）。

    Args:
        model: sgn.nn.Module 实例

    Returns:
        总参数量（int）

    Usage:
        total = sgn.util.count_parameters(model)
        print(f"Total parameters: {total}")
    """
    total = 0
    for _, p in model.named_parameters():
        total += p.numel
    return total


def summary(model, input_shape: list | tuple | None = None) -> str:
    """打印网络结构摘要：每层名称、类型、输出形状、参数量。

    Args:
        model: sgn.nn.Module 实例
        input_shape: 输入张量形状（可选，仅用于显示）

    Returns:
        格式化的摘要字符串

    Usage:
        print(sgn.util.summary(model))
        print(sgn.util.summary(model, input_shape=(1, 784)))
    """
    lines = []
    total_params = 0

    # 标题
    lines.append("=" * 72)
    lines.append(f"{'Layer':<32} {'Type':<16} {'Params':>10}")
    lines.append("=" * 72)

    if input_shape is not None:
        lines.append(f"  Input shape: {list(input_shape)}")

    def _format_row(name, type_name, params):
        return f"  {name:<30} {type_name:<16} {params:>10,}"

    # 递归收集所有层
    def _collect(module, prefix=""):
        nonlocal total_params

        # 收集直接参数
        direct_params = 0
        for n, p in module.named_parameters():
            if "." not in n:
                direct_params += p.numel

        if direct_params > 0:
            type_name = type(module).__name__
            lines.append(_format_row(prefix or type_name, type_name, direct_params))
            total_params += direct_params

        # 收集子模块
        for child in module.children():
            child_name = type(child).__name__
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            _collect(child, child_prefix)

    _collect(model)

    lines.append("=" * 72)
    lines.append(f"  Total params: {total_params:,}")
    if total_params > 0:
        lines.append(f"  Model size: {total_params * 4 / 1024:.1f} KB (float32)")
    lines.append("=" * 72)

    return "\n".join(lines)