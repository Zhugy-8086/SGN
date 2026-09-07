# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""SGN Stage 3.0 统一 C++ 抽象层 Python 包入口

为 C++ 扩展类型（ValueSpec/UnitValue/PrecisionBudget）补充：
  - 魔术方法：__hash__ / __eq__ / __repr__
  - 序列化：to_dict / from_dict / to_json / from_json（JSON 兼容，含 level_f/level_b）

设计说明：
  - C++ pybind11 绑定位于编译后的 sgn.cp314-win_amd64.pyd
  - 本文件用 importlib 直接加载 .pyd，避免与包名 sgn 冲突
  - 加载后通过 monkey-patch 为类型补充 Python 端方法
  - 序列化保留 level_f/level_b 字段（Stage 3.0.4 双向控制接口预留）

Usage:
    from engine.sgn import ValueSpec, UnitValue, PrecisionBudget, ScaleFn
    spec = ValueSpec(16)
    d = spec.to_dict()  # {'bits': 16, 'scale': 'MAX'}
    spec2 = ValueSpec.from_dict(d)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any

# ============================================================
# 加载编译后的 C++ 扩展（sgn.cp314-win_amd64.pyd）
# ============================================================
# 注意：包名和 .pyd 名都是 sgn，不能用 import sgn（会递归）
# 解决方案：用 importlib.util 直接从文件路径加载

_BUILD_DIR = os.path.join(os.path.dirname(__file__), "build")


def _find_pyd() -> str:
    """在 build/ 目录或包目录查找 sgn.cp*.pyd

    查找顺序：
      1. build/ 目录（开发模式）
      2. 包目录自身（pip install 后 .pyd 被复制到包目录）
      3. 触发自动编译（开发模式 + cmake 可用）
    """
    # 安全审计 2026-08-16 L2：按当前 Python 版本过滤（cp{major}{minor}），
    # 避免多版本 .pyd 共存时 listdir 顺序不定加载错误 ABI 崩溃
    _ver_prefix = f"sgn.cp{sys.version_info.major}{sys.version_info.minor}-"

    def _scan_dir(dirpath: str) -> str | None:
        if not os.path.isdir(dirpath):
            return None
        for name in os.listdir(dirpath):
            if name.startswith(_ver_prefix) and name.endswith(".pyd"):
                return os.path.join(dirpath, name)
        return None

    # 优先查找 build/ 目录
    found = _scan_dir(_BUILD_DIR)
    if found:
        return found

    # pip install 后 .pyd 在包目录中
    _pkg_dir = os.path.dirname(__file__)
    found = _scan_dir(_pkg_dir)
    if found:
        return found

    # 尝试自动编译
    # 安全审计 2026-08-16 L3：auto_build 失败原因不静默吞没——附加到
    # ImportError，便于定位构建断因（cmake 缺失/编译错误等）
    _build_err = None
    try:
        from .util import auto_build
        if auto_build(verbose=True):
            found = _scan_dir(_BUILD_DIR)
            if found:
                return found
    except Exception as e:  # noqa: BLE001 — 任何构建失败都不能阻断 ImportError 语义
        _build_err = e

    msg = (
        f"未找到 {_ver_prefix}*.pyd（Python {sys.version_info.major}.{sys.version_info.minor}）。\n"
        f"  已检查: {_BUILD_DIR}\n"
        f"  已检查: {_pkg_dir}\n"
        f"  请先编译: cmake -B build -S . && cmake --build build"
    )
    if _build_err is not None:
        msg += f"\n  自动编译失败: {type(_build_err).__name__}: {_build_err}"
    raise ImportError(msg)


_pyd_path = _find_pyd()


def _load_native_module():
    """加载 sgn.*.pyd，避免与包名 sgn 冲突

    .pyd 的 PyInit 函数名为 PyInit_sgn（由 PYBIND11_MODULE(sgn, m) 定义），
    因此必须用 "sgn" 名字加载。通过 importlib.util 直接从 .pyd 文件路径加载，
    加载后重命名为 _sgn_native 避免覆盖包。
    """
    import os as _os

    # 确保 .pyd 所在目录在 DLL 搜索路径中（Windows）
    _pyd_dir = _os.path.dirname(_pyd_path)
    if _os.name == "nt":
        _os.add_dll_directory(_pyd_dir)

    # 使用 importlib.util 直接从 .pyd 文件路径加载
    # 模块名必须为 "sgn" 以匹配 PyInit_sgn
    spec = importlib.util.spec_from_file_location("sgn", _pyd_path)
    if spec is None:
        raise ImportError(
            f"无法为 .pyd 创建模块 spec。\n"
            f"  .pyd 路径: {_pyd_path}\n"
            f"  请确认文件存在且为有效的 Python 扩展模块"
        )
    _native = importlib.util.module_from_spec(spec)
    # 将 .pyd 模块注册到别名 _sgn_native，避免覆盖 sgn 包
    sys.modules["_sgn_native"] = _native
    spec.loader.exec_module(_native)
    return _native


_native = _load_native_module()

# 从 native 模块导出 C++ 类型
ScaleFn = _native.ScaleFn
ValueSpec = _native.ValueSpec
UnitValue = _native.UnitValue
PrecisionBudget = _native.PrecisionBudget
LayerCost = _native.LayerCost
Col2im = _native.Col2im
col2im_add = _native.col2im_add
version = _native.version
# 包版本（普通用户可见）：import sgn; sgn.__version__ 即可查看。
# 与 C++ native 的 version()（Stage 3.0 placeholder）区分开。
__version__ = "0.10.1"
compiler_info = _native.compiler_info
add = _native.add
test_avx_vnni = _native.test_avx_vnni

# Stage 3.0.5: MSint C++ 扩展类型
SlotSpec = _native.SlotSpec
PackedBackend = _native.PackedBackend
MSIntView = _native.MSIntView
SplitDot = _native.SplitDot
MultiScaleView = _native.MultiScaleView
PrecisionSelector = _native.PrecisionSelector
# LeveledSplitDot 为新组合类：用 getattr 容错，避免旧 .pyd 未重编译时包导入崩溃
LeveledSplitDot = getattr(_native, "LeveledSplitDot", None)
batch_get_all = _native.batch_get_all
batch_decode_to_float = _native.batch_decode_to_float
batch_decode_to_float_into = _native.batch_decode_to_float_into
# 安全审计 2026-08-16 A2-7：补导出，模式 A（import engine.sgn as sgn）下
# test_unit_value 调用 sgn.batch_to_float / sgn.batch_from_float 可访问
batch_to_float = _native.batch_to_float
batch_from_float = _native.batch_from_float

# Stage 3.0.6: HC8 余积存储 C++ 扩展类型
HC8Coproduct = _native.HC8Coproduct
HC8CoproductArray = _native.HC8CoproductArray
HC8KalmanFilter = _native.HC8KalmanFilter

# 从 native 模块导出子模块（供 models.py 等 Python 模块使用）
autograd = _native.autograd
nn = _native.nn
# mkern/nested 嵌套量化原语子模块（nested_quant 设计档 §二.4；不 re-export 会在
# engine 路径优先时被包遮蔽——2026-09-05 Phase 1 接线发现）
mkern_nested = getattr(_native, "mkern_nested", None)
# mkern/simd 整型点积原语子模块（层 2 LeveledState 状态消费计算域入口；
# 同上不 re-export 会被包遮蔽——2026-09-07 层 2 Phase 2a 接线）
mkern_simd = getattr(_native, "mkern_simd", None)

# 优化器封装（基础设施 B1，2026-08-29）
from .optimizer import Optimizer as Optimizer
from .optimizer import SGD as SGD
from .optimizer import Adam as Adam
from .optimizer import AdamW as AdamW

# 训练检查点保存/恢复（基础设施 B2，2026-08-29）
from .checkpoint import save_checkpoint as save_checkpoint
from .checkpoint import load_checkpoint as load_checkpoint

# 训练实验框架（基础设施 A1，2026-08-29）：多 seed 执行 + 聚合 + 导出 + 参照传递
from .experiment import run_seeds as run_seeds
from .experiment import aggregate as aggregate
from .experiment import export_json as export_json
from .experiment import export_csv as export_csv
from .experiment import save_ref_json as save_ref_json
from .experiment import load_ref_json as load_ref_json

# 随机性/可复现工具（基础设施 B3，2026-08-29）：统一种子派生 + 复现断言
from .rng import SeedBundle as SeedBundle
from .rng import make_seed_bundle as make_seed_bundle
from .rng import assert_reproducible as assert_reproducible

# HC 子模块统一 re-export（2026-08-16 A2-7：模式 A 测试经 engine.sgn 访问
# 这些子模块；原模式 B 直接 import .pyd 顶层即可见，统一入口后需在此暴露）
# getattr 容错，避免旧 .pyd 未重编译时包导入崩溃（与 LeveledSplitDot 一致）
col2im_c = getattr(_native, "col2im_c", None)
hc8_net = getattr(_native, "hc8_net", None)
hc16 = getattr(_native, "hc16", None)
hc16ms = getattr(_native, "hc16ms", None)
hc4 = getattr(_native, "hc4", None)


# ============================================================
# record_scope 上下文管理器 — 封装 autograd 磁带控制
# ============================================================

class _RecordScope:
    """封装 start_recording()/stop_recording() 的上下文管理器。

    用法：
        with ag.record_scope():
            y = model.forward([x])
            loss = ag.cross_entropy_loss(y, target)
        loss.backward()

        # 清空旧磁带后再开始：
        with ag.record_scope(clear=True):
            ...
    """

    def __init__(self, clear: bool = False):
        self._clear = clear

    def __enter__(self):
        if self._clear:
            autograd.clear()
        autograd.start_recording()

    def __exit__(self, exc_type, exc_val, exc_tb):
        autograd.stop_recording()
        return False  # 不抑制异常


# 注入到 autograd 子模块
autograd.record_scope = _RecordScope


# ============================================================
# Module.__repr__ 美化 — 模型结构树形打印
# ============================================================

def _module_repr(self, _depth: int = 0) -> str:
    """美化 Module 的 __repr__，打印参数和子模块树形结构。

    Example:
        MLP(
          (fc1_w): Parameter([128, 784])
          (fc1_b): Parameter([128])
          (fc2_w): Parameter([64, 128])
          (fc2_b): Parameter([64])
          (fc3_w): Parameter([10, 64])
          (fc3_b): Parameter([10])
        )
    """
    # 安全审计 2026-08-16 L8：递归深度限制——循环引用的子模块图
    # 会导致 RecursionError，深度 > 10 时截断为 "..."
    if _depth > 10:
        return f"{type(self).__name__}(...)"
    lines = [f"{type(self).__name__}("]

    # 直接参数（不含子模块的，通过 '.' 判断）
    for name, p in self.named_parameters():
        if '.' not in name:
            shape = [str(d) for d in p.shape]
            lines.append(f"  ({name}): Parameter([{', '.join(shape)}])")

    # 直接缓冲区
    for name, b in self.named_buffers():
        if '.' not in name:
            shape = [str(d) for d in b.shape]
            lines.append(f"  ({name}): Buffer([{', '.join(shape)}])")

    # 子模块
    for child in self.children():
        child_lines = _module_repr(child, _depth + 1).split('\n')
        for line in child_lines:
            lines.append(f"  {line}")

    lines.append(")")
    return "\n".join(lines)


nn.Module.__repr__ = _module_repr


# ============================================================
# ScaleFn 序列化辅助
# ============================================================

_SCALE_FN_NAMES = {
    ScaleFn.MAX: "MAX",
    ScaleFn.RMS: "RMS",
    ScaleFn.L2: "L2",
    ScaleFn.P95: "P95",
}

_SCALE_FN_VALUES = {v: k for k, v in _SCALE_FN_NAMES.items()}


def _scale_fn_to_str(fn: ScaleFn) -> str:
    """ScaleFn 枚举转字符串"""
    return _SCALE_FN_NAMES.get(fn, "MAX")


def _scale_fn_from_str(s: str) -> ScaleFn:
    """字符串转 ScaleFn 枚举"""
    if s not in _SCALE_FN_VALUES:
        raise ValueError(f"未知 ScaleFn: {s}，可选: {list(_SCALE_FN_VALUES.keys())}")
    return _SCALE_FN_VALUES[s]


# ============================================================
# ValueSpec 补充魔术方法 + 序列化
# ============================================================

def _value_spec_hash(self: ValueSpec) -> int:
    return hash((self.bits, self.scale))


def _value_spec_to_dict(self: ValueSpec) -> dict:
    """序列化为 dict（JSON 兼容）"""
    return {"bits": int(self.bits), "scale": _scale_fn_to_str(self.scale)}


def _value_spec_from_dict(cls: type, d: dict) -> ValueSpec:
    """从 dict 反序列化"""
    return cls(bits=d["bits"], scale=_scale_fn_from_str(d["scale"]))


def _value_spec_to_json(self: ValueSpec) -> str:
    """序列化为 JSON 字符串"""
    return json.dumps(self.to_dict())


def _value_spec_from_json(cls: type, s: str) -> ValueSpec:
    """从 JSON 字符串反序列化"""
    return cls.from_dict(json.loads(s))


# 注入到 ValueSpec 类
ValueSpec.__hash__ = _value_spec_hash
ValueSpec.to_dict = _value_spec_to_dict
ValueSpec.from_dict = classmethod(_value_spec_from_dict)
ValueSpec.to_json = _value_spec_to_json
ValueSpec.from_json = classmethod(_value_spec_from_json)


# ============================================================
# UnitValue 补充魔术方法 + 序列化
# ============================================================

def _unit_value_hash(self: UnitValue) -> int:
    return hash((self.raw, self.spec))


def _unit_value_eq(self: UnitValue, other: object) -> bool:
    if not isinstance(other, UnitValue):
        return NotImplemented
    return self.raw == other.raw and self.spec == other.spec


def _unit_value_repr(self: UnitValue) -> str:
    return f"UnitValue(raw={self.raw}, spec={self.spec})"


def _unit_value_to_dict(self: UnitValue) -> dict:
    """序列化为 dict（JSON 兼容）"""
    return {"raw": int(self.raw), "spec": self.spec.to_dict()}


def _unit_value_from_dict(cls: type, d: dict) -> UnitValue:
    """从 dict 反序列化"""
    return cls(raw=d["raw"], spec=ValueSpec.from_dict(d["spec"]))


def _unit_value_to_json(self: UnitValue) -> str:
    """序列化为 JSON 字符串"""
    return json.dumps(self.to_dict())


def _unit_value_from_json(cls: type, s: str) -> UnitValue:
    """从 JSON 字符串反序列化"""
    return cls.from_dict(json.loads(s))


# 注入到 UnitValue 类
UnitValue.__hash__ = _unit_value_hash
UnitValue.__eq__ = _unit_value_eq
UnitValue.__repr__ = _unit_value_repr
UnitValue.to_dict = _unit_value_to_dict
UnitValue.from_dict = classmethod(_unit_value_from_dict)
UnitValue.to_json = _unit_value_to_json
UnitValue.from_json = classmethod(_unit_value_from_json)


# ============================================================
# PrecisionBudget 补充魔术方法 + 序列化
# ============================================================

def _precision_budget_hash(self: PrecisionBudget) -> int:
    # 按 total_bits + layers 成本哈希
    layer_hash = tuple((lc.c, lc.b_min, lc.b_max) for lc in self.layers)
    return hash((self.total_bits, layer_hash))


def _precision_budget_repr(self: PrecisionBudget) -> str:
    lf = f", level_f={self.level_f}" if self.has_level_f else ""
    lb = f", level_b={self.level_b}" if self.has_level_b else ""
    return f"PrecisionBudget(total_bits={self.total_bits}, layers={len(self.layers)}{lf}{lb})"


def _precision_budget_to_dict(self: PrecisionBudget) -> dict:
    """序列化为 dict（JSON 兼容，含 level_f/level_b 字段）

    保留 level_f/level_b 预留字段（Stage 3.0.4 双向控制接口）
    """
    return {
        "total_bits": int(self.total_bits),
        "layers": [
            {"c": float(lc.c), "b_min": int(lc.b_min), "b_max": int(lc.b_max)}
            for lc in self.layers
        ],
        "level_f": self.level_f.to_dict() if self.has_level_f else None,
        "level_b": self.level_b.to_dict() if self.has_level_b else None,
    }


def _precision_budget_from_dict(cls: type, d: dict) -> PrecisionBudget:
    """从 dict 反序列化（含 level_f/level_b 字段）"""
    layers = [
        LayerCost(c=lc["c"], b_min=lc["b_min"], b_max=lc["b_max"])
        for lc in d["layers"]
    ]
    pb = cls(d["total_bits"], layers)
    if d.get("level_f") is not None:
        pb.set_level_f(ValueSpec.from_dict(d["level_f"]))
    if d.get("level_b") is not None:
        pb.set_level_b(ValueSpec.from_dict(d["level_b"]))
    return pb


def _precision_budget_to_json(self: PrecisionBudget) -> str:
    """序列化为 JSON 字符串"""
    return json.dumps(self.to_dict())


def _precision_budget_from_json(cls: type, s: str) -> PrecisionBudget:
    """从 JSON 字符串反序列化"""
    return cls.from_dict(json.loads(s))


# 注入到 PrecisionBudget 类
PrecisionBudget.__hash__ = _precision_budget_hash
PrecisionBudget.__repr__ = _precision_budget_repr
PrecisionBudget.to_dict = _precision_budget_to_dict
PrecisionBudget.from_dict = classmethod(_precision_budget_from_dict)
PrecisionBudget.to_json = _precision_budget_to_json
PrecisionBudget.from_json = classmethod(_precision_budget_from_json)


# ============================================================
# 公开 API
# ============================================================

__all__ = [
    "ScaleFn",
    "ValueSpec",
    "UnitValue",
    "PrecisionBudget",
    "LayerCost",
    "Col2im",
    "col2im_add",
    "version",
    "compiler_info",
    "add",
    "test_avx_vnni",
    "diagnose",
    "test",
    "util",
    "optim",
    "set_verbosity",
    "get_verbosity",
]


# ============================================================
# 全局 verbosity 控制
# ============================================================

_VERBOSITY = 1  # 0=quiet, 1=normal, 2=debug


def set_verbosity(level: int) -> None:
    """设置全局日志级别。

    Args:
        level: 0=quiet（静默），1=normal（默认），2=debug（详细调试信息）
    """
    global _VERBOSITY
    if level not in (0, 1, 2):
        raise ValueError(f"verbosity 必须在 0-2 之间，收到: {level}")
    _VERBOSITY = level


def get_verbosity() -> int:
    """返回当前全局日志级别。

    Returns:
        0=quiet, 1=normal, 2=debug
    """
    return _VERBOSITY


# ============================================================
# 诊断工具
# ============================================================

def diagnose() -> str:
    """一键诊断：版本、编译器、CPU 特性、子模块状态。

    Usage:
        >>> import sgn
        >>> print(sgn.diagnose())

    Example output:
        SGN version: Stage 3.0 placeholder
        Build: 2026-08-06 05:48
        Compiler: Clang 22.1.8
        Python: 3.14.0
        CPU: AVX2 ✓  AVX-VNNI ✓
        Submodules: col2im_c ✓  hc8_net ✓  hc16 ✓  hc16ms ✓  hc4 ✓  autograd ✓  nn ✓

    注：首行为 C++ native version() 占位符输出；包版本见 sgn.__version__。
    （安全审计 2026-08-16 docstring 修正：原示例硬编码 "SGN v2.0.0" 与实际不符）
    """
    import datetime
    import platform

    lines = []
    lines.append(f"SGN version: {version()}")

    # 构建时间（从 .pyd 文件修改时间推断）
    if os.path.exists(_pyd_path):
        mtime = os.path.getmtime(_pyd_path)
        build_time = datetime.datetime.fromtimestamp(mtime)
        lines.append(f"Build: {build_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        lines.append("Build: unknown")

    # 编译器
    lines.append(f"Compiler: {compiler_info()}")

    # Python 版本
    lines.append(f"Python: {platform.python_version()}")

    # CPU 特性
    avx2_ok = "✓" if _check_avx2() else "✗"
    avx_vnni_ok = "✓" if _check_avx_vnni() else "✗"
    lines.append(f"CPU: AVX2 {avx2_ok}  AVX-VNNI {avx_vnni_ok}")

    # 子模块状态
    submodules = [
        ("col2im_c", "col2im_c"),
        ("hc8_net", "hc8_net"),
        ("hc16", "hc16"),
        ("hc16ms", "hc16ms"),
        ("hc4", "hc4"),
        ("autograd", "autograd"),
        ("nn", "nn"),
    ]
    status_parts = []
    for name, attr in submodules:
        ok = "✓" if hasattr(_native, attr) else "✗"
        status_parts.append(f"{name} {ok}")
    lines.append("Submodules: " + "  ".join(status_parts))

    return "\n".join(lines)


def _check_avx2() -> bool:
    """检测 AVX2 是否可用。

    .pyd 以 -mavx2 编译；register_hc8_net 在 import 时已用 cpuid 显式检测
    （非 AVX2 CPU 上 import 直接抛 RuntimeError），故此处能执行到即表示支持
    （安全审计 2026-08-16 L6，检测移至 C++ 侧 I2 统一实现）。
    """
    return True


def _check_avx_vnni() -> bool:
    """检测 AVX-VNNI 是否可用（通过 test_avx_vnni() 验证）。"""
    try:
        result = test_avx_vnni()
        return result == 8  # dpbusd: 0 + 1*2 + 1*2 + 1*2 + 1*2 = 8 per lane
    except Exception:
        return False


def test() -> bool:
    """快速自检：验证核心功能是否正常。

    检查项：
      1. 版本号
      2. col2im_c 子模块
      3. HC 编码子模块（hc8_net / hc16 / hc16ms / hc4）
      4. autograd / nn 子模块
      5. AVX-VNNI 指令
      6. 基础算子（add / col2im_add 签名）

    Returns:
        True 如果全部通过。

    Usage:
        >>> import sgn
        >>> sgn.test()
        [PASS] version: Stage 3.0 placeholder
        [PASS] col2im_c submodule
        [PASS] hc8_net submodule
        [PASS] hc16 submodule
        [PASS] hc16ms submodule
        [PASS] hc4 submodule
        [PASS] autograd submodule
        [PASS] nn submodule
        [PASS] AVX-VNNI
        [PASS] add(1, 2) = 3
        ---
        All 10 tests passed.
    """
    results = []

    def _check(desc, fn):
        try:
            fn()
            results.append((desc, True, None))
            print(f"[PASS] {desc}")
        except Exception as e:
            results.append((desc, False, str(e)))
            print(f"[FAIL] {desc}: {e}")

    _check("version", lambda: None if version() else None)
    _check("col2im_c submodule", lambda: _native.col2im_c)
    _check("hc8_net submodule", lambda: _native.hc8_net)
    _check("hc16 submodule", lambda: _native.hc16)
    _check("hc16ms submodule", lambda: _native.hc16ms)
    _check("hc4 submodule", lambda: _native.hc4)
    _check("autograd submodule", lambda: _native.autograd)
    _check("nn submodule", lambda: _native.nn)
    _check("AVX-VNNI", lambda: None if _check_avx_vnni() else (_ for _ in ()).throw(RuntimeError("AVX-VNNI not available")))
    _check("add(1, 2) = 3", lambda: None if add(1, 2) == 3 else (_ for _ in ()).throw(AssertionError(f"add(1, 2) = {add(1, 2)}")))

    all_pass = all(r[1] for r in results)
    n_pass = sum(1 for r in results if r[1])
    print("---")
    if all_pass:
        print(f"All {len(results)} tests passed.")
    else:
        print(f"{n_pass}/{len(results)} tests passed, {len(results) - n_pass} failed.")
    return all_pass


# ============================================================
# 工具箱（util）
# ============================================================
from . import util

# ============================================================
# 优化器（optim）
# ============================================================
from . import optim

# ============================================================
# 损失函数（loss）— 可插拔损失 + 诊断器
# ============================================================
from . import loss

# ============================================================
# 日志工具（logger）— 梯度/张量统计日志器
# ============================================================
from . import logger


# ============================================================
# 向后兼容：pysgn_* 模块别名
# ============================================================
# 合并后，pysgn_net/hc16/hc16ms/hc4_pshufb 不再是独立 .pyd，
# 而是合并到 sgn 模块的子模块中（placeholder.cpp 通过 m.def_submodule 注册）。
# 这里通过 sys.modules 注入兼容模块，使旧代码的 import pysgn_* 继续工作。
#
# _native 是上方 _load_native_module() 返回的 C++ sgn 模块（即任务说明中的
# "sgn 模块"），其 hc8_net/hc16/hc16ms/hc4 子模块属性即原 pysgn_* 的等价实现。
#
# 安全审计 2026-08-16 决策项 2：新增 SGN_DISABLE_PYSGN_COMPAT 环境变量开关——
# 置 1 时跳过别名注入（用于验证/剔除对旧别名的依赖）；默认保持注入以便
# 旧代码与兼容性测试继续工作。别名计划在依赖全部迁移后移除。

def _setup_compat_aliases():
    """为合并前的 pysgn_* 模块创建向后兼容别名。"""
    # 安全审计 2026-08-16 L4：原 except 静默吞掉旧 .pyd 情况——
    # 补 warning 日志（.pyd 过旧需重编译的可感知提示）
    # 决策项 2：SGN_DISABLE_PYSGN_COMPAT=1 时跳过别名注入
    if os.environ.get("SGN_DISABLE_PYSGN_COMPAT", "") == "1":
        return
    import logging as _logging
    _log = _logging.getLogger(__name__)
    try:
        # pysgn_net → sgn.hc8_net
        if 'pysgn_net' not in sys.modules:
            sys.modules['pysgn_net'] = _native.hc8_net

        # pysgn_hc16 → sgn.hc16
        if 'pysgn_hc16' not in sys.modules:
            sys.modules['pysgn_hc16'] = _native.hc16

        # pysgn_hc16ms → sgn.hc16ms
        if 'pysgn_hc16ms' not in sys.modules:
            sys.modules['pysgn_hc16ms'] = _native.hc16ms

        # pysgn_hc4_pshufb → sgn.hc4
        if 'pysgn_hc4_pshufb' not in sys.modules:
            sys.modules['pysgn_hc4_pshufb'] = _native.hc4
    except AttributeError as e:
        # .pyd 为合并前的旧版本，hc8_net 等子模块尚未注册，跳过别名注入
        _log.warning(
            "pysgn_* 兼容别名注入失败（.pyd 可能是合并前的旧版本，"
            "建议重新 cmake --build）: %s", e
        )


_setup_compat_aliases()

# ============================================================
# 稳定模式预定义模型（sgn.models.MLP, sgn.models.CNN4）
# ============================================================
from . import models

# ============================================================
# 标准层 Module 子类 — 注入 sgn.nn 命名空间
# ============================================================
# nn_layers.py 提供 Linear / Conv2d / ReLU / MaxPool2d / BatchNorm2d / Sequential
# 这些类继承 sgn.nn.Module，在 __init__ 中自动创建和注册参数
# 注入到 sgn.nn 后，用户可以用 sgn.nn.Linear(784, 256) 等语法
#
# 安全审计 2026-08-16 L5/A4-1：本注入依赖导入顺序——必须先加载 .pyd
# （上方 _load_native_module）再 monkey-patch。若 .pyd 未就绪本模块
# import 已失败，故实际风险有限；彻底解法（层定义移 C++ 侧或 lazy
# loading）列为后续架构项，暂以本注释固化约束。

from . import nn_layers

nn.Linear = nn_layers.Linear
nn.Conv2d = nn_layers.Conv2d
nn.ReLU = nn_layers.ReLU
nn.MaxPool2d = nn_layers.MaxPool2d
nn.BatchNorm2d = nn_layers.BatchNorm2d
nn.Sequential = nn_layers.Sequential
nn.Dropout = nn_layers.Dropout
nn.LayerNorm = nn_layers.LayerNorm
