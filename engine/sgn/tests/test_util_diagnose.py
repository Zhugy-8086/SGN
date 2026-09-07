# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""test_util_diagnose.py — 验证 sgn.diagnose / sgn.test / sgn.util / Module.__repr__

测试覆盖：
  1. sgn.diagnose() — 诊断信息完整性
  2. sgn.test() — 10 项自检全部通过
  3. sgn.util.ensure_built() — .pyd 存在且最新
  4. sgn.util.Timer — 上下文管理器 + 手动启停
  5. sgn.util.describe() — 打印参数信息
  6. Module.__repr__ — MLP 和 CNN4 树形结构

运行方式：
    cd <repo>
    python engine/sgn/tests/test_util_diagnose.py
"""

from __future__ import annotations

import os
import sys
import io

# 路径设置：添加项目根目录（engine 包的父目录）到 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _PROJ_ROOT)

# 用相对导入 engine.sgn 代替顶层 `import sgn`，避免 pytest 全量收集时
# 与已加载的 engine.sgn 包产生 sys.modules 冲突（顶层 `import sgn` 会
# 解析到别名 _sgn_native 或产生 INTERNALERROR）。
import engine.sgn as sgn

# ============================================================
# 测试辅助
# ============================================================
_results = []

# 脚本式测试：仅当以 `python test_util_diagnose.py` 直接运行时执行断言。
# 被 pytest 收集（import）时短路，避免模块级 sys.exit(1) 触发 INTERNALERROR。
if __name__ == '__main__':
    def test(desc: str):
        """装饰器风格的测试注册（手动调用）"""
        def decorator(fn):
            try:
                fn()
                _results.append((desc, True, None))
                print(f"  [PASS] {desc}")
            except Exception as e:
                _results.append((desc, False, str(e)))
                print(f"  [FAIL] {desc}: {e}")
            return fn
        return decorator
else:
    def test(desc: str):
        """pytest 收集路径：不执行，仅占位（脚本式测试直接运行才执行）。"""
        def decorator(fn):
            return fn
        return decorator


# ============================================================
# 1. sgn.diagnose() — 诊断信息完整性
# ============================================================
@test("diagnose: 返回字符串且非空")
def test_diagnose_not_empty():
    output = sgn.diagnose()
    assert isinstance(output, str), f"应为 str，实际为 {type(output)}"
    assert len(output) > 0, "diagnose 输出为空"


@test("diagnose: 包含版本号")
def test_diagnose_has_version():
    output = sgn.diagnose()
    assert "version" in output.lower(), f"缺少 version 字段: {output[:80]}"


@test("diagnose: 包含编译器信息")
def test_diagnose_has_compiler():
    output = sgn.diagnose()
    assert "Clang" in output or "MSVC" in output or "GCC" in output, \
        f"缺少编译器信息: {output[:80]}"


@test("diagnose: 包含所有 7 个子模块状态")
def test_diagnose_has_submodules():
    output = sgn.diagnose()
    for name in ["col2im_c", "hc8_net", "hc16", "hc16ms", "hc4", "autograd", "nn"]:
        assert name in output, f"缺少子模块 {name}"


@test("diagnose: CPU 特性行存在")
def test_diagnose_has_cpu():
    output = sgn.diagnose()
    assert "AVX2" in output, "缺少 AVX2 状态"
    assert "AVX-VNNI" in output, "缺少 AVX-VNNI 状态"


# ============================================================
# 2. sgn.test() — 10 项自检
# ============================================================
@test("sgn.test(): 返回 True（全部通过）")
def test_sgn_test_all_pass():
    # 捕获输出，避免混入测试日志
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = sgn.test()
    finally:
        sys.stdout = old_stdout
    assert result is True, f"sgn.test() 返回 {result}，期望 True"


@test("sgn.test(): 返回值类型为 bool")
def test_sgn_test_returns_bool():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = sgn.test()
    finally:
        sys.stdout = old_stdout
    assert isinstance(result, bool), f"返回类型应为 bool，实际为 {type(result)}"


# ============================================================
# 3. sgn.util.ensure_built() — .pyd 存在且最新
# ============================================================
@test("ensure_built: 返回 True（.pyd 存在且最新）")
def test_ensure_built():
    result = sgn.util.ensure_built()
    assert result is True, f"ensure_built() 返回 {result}，期望 True"


# ============================================================
# 4. sgn.util.Timer — 计时器
# ============================================================
@test("Timer: 上下文管理器正常计时")
def test_timer_context():
    with sgn.util.Timer("ctx_test") as t:
        _ = sum(range(100000))
    assert t.elapsed > 0, "耗时应为正数"
    assert t.elapsed_ms > 0, "毫秒耗时应为正数"


@test("Timer: 手动启停正常")
def test_timer_manual():
    t = sgn.util.Timer("manual_test")
    t.start()
    _ = sum(range(100000))
    elapsed = t.stop()
    assert elapsed > 0, "停表返回值应为正数"
    assert t.elapsed_ms > 0, "毫秒耗时应为正数"


@test("Timer: __repr__ 包含 label 和 elapsed")
def test_timer_repr():
    t = sgn.util.Timer("repr_test")
    t.start()
    t.stop()
    r = repr(t)
    assert "repr_test" in r, f"__repr__ 缺少 label: {r}"
    assert "ms" in r, f"__repr__ 缺少时间单位: {r}"


# ============================================================
# 5. sgn.util.describe() — 打印参数信息
# ============================================================
@test("describe: Parameter 正确显示 shape")
def test_describe_parameter():
    model = sgn.models.MLP(input_size=784, hidden1=32, hidden2=16, num_classes=10)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        sgn.util.describe(model.fc1_w)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    assert "Parameter" in output, f"应为 Parameter: {output}"
    assert "32, 784" in output or "32,784" in output, f"shape 不对: {output}"


@test("describe: 显示 requires_grad")
def test_describe_requires_grad():
    model = sgn.models.MLP(input_size=784, hidden1=32, hidden2=16)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        sgn.util.describe(model.fc1_w)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    assert "requires_grad=True" in output, f"缺少 requires_grad: {output}"


# ============================================================
# 6. Module.__repr__ — 树形结构
# ============================================================
@test("__repr__: MLP 显示类名")
def test_repr_mlp_classname():
    model = sgn.models.MLP(input_size=784, hidden1=32, hidden2=16, num_classes=10)
    r = repr(model)
    assert r.startswith("MLP("), f"repr 应以 MLP( 开头: {r[:40]}"


@test("__repr__: MLP 包含全部 6 个参数")
def test_repr_mlp_all_params():
    model = sgn.models.MLP(input_size=784, hidden1=32, hidden2=16, num_classes=10)
    r = repr(model)
    for name in ["fc1_w", "fc1_b", "fc2_w", "fc2_b", "fc3_w", "fc3_b"]:
        assert f"({name}):" in r, f"缺少参数 {name}"


@test("__repr__: MLP 参数形状正确")
def test_repr_mlp_shapes():
    model = sgn.models.MLP(input_size=784, hidden1=32, hidden2=16, num_classes=10)
    r = repr(model)
    expected_shapes = {
        "fc1_w": "32, 784",
        "fc1_b": "32",
        "fc2_w": "16, 32",
        "fc2_b": "16",
        "fc3_w": "10, 16",
        "fc3_b": "10",
    }
    for name, shape_str in expected_shapes.items():
        assert f"[{shape_str}]" in r, f"{name} 形状应为 [{shape_str}]: {r}"


@test("__repr__: CNN4 显示类名和参数")
def test_repr_cnn4():
    model = sgn.models.CNN4(in_channels=3, img_size=32, num_classes=10)
    r = repr(model)
    assert r.startswith("CNN4("), f"repr 应以 CNN4( 开头: {r[:40]}"
    for name in ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b"]:
        assert f"({name}):" in r, f"缺少参数 {name}"


# ============================================================
# 7. 边缘情况
# ============================================================
@test("diagnose: 可重复调用不报错")
def test_diagnose_idempotent():
    a = sgn.diagnose()
    b = sgn.diagnose()
    assert a == b, "两次 diagnose 输出应一致"


@test("compiler_info: 返回 Clang 字符串")
def test_compiler_info():
    info = sgn.compiler_info()
    assert "Clang" in info, f"编译器应为 Clang: {info}"
    assert "22.1.8" in info, f"版本应为 22.1.8: {info}"


# ============================================================
# 汇总（仅直接运行时执行）
# ============================================================
if __name__ == '__main__':
    n_pass = sum(1 for r in _results if r[1])
    n_total = len(_results)
    print(f"\n{'='*60}")
    print(f"Tests: {n_pass}/{n_total} passed")
    if n_pass == n_total:
        print("All tests passed!")
    else:
        print(f"FAILURES ({n_total - n_pass}):")
        for desc, ok, err in _results:
            if not ok:
                print(f"  - {desc}: {err}")
        sys.exit(1)