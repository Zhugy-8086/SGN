# test_leveled_state.py - 层 2 接线验收测试（W5–W7，Phase 2a）
#
# 预注册验收钩子（docs/ltc_cfc_dynamic_quantization/
# 层2接线设计_LeveledState状态布局_2026_09_06.md §七）：
#   W5 floor 域内定理：任意 int32 码字（含 ±顶码）floor 读法恒在符号域内；
#     RTN 读法（view_codes）顶码 +2^(b−1) 超域复现（B4a 例外回归锚）
#   W6 零点修正恒等式：dot8(q_u8, w) − 128·Σw == Σ q_floor·w 逐位（int64 域，
#     含 K > 2^18 分块域）；Q4 仿射解包路径（unpacked = state_s4 + 8，
#     零点 8）同构逐位；dot4_packed 4×4 往返
#   W7 分组点积正确性：native dot8 路径 == numpy 参考路径（逐位，含组边界
#     情形）；端到端值重建界 |2^24·u·(dot+修正) − Σ h·w| ≤ step·Σ|w|·(1+2⁻²⁴)
#   随迁：LeveledStateBatch 守卫（零样本 u=0 / N≥2 陷阱）+ Q8 half-cell 界
#
# 运行：SGN 根目录 pytest engine/sgn/tests/test_leveled_state.py

import numpy as np
import pytest

import engine.sgn as sgn_pkg
from engine.sgn.leveled_state import LeveledStateBatch

_mk_nested = sgn_pkg.mkern_nested
_mk_simd = sgn_pkg.mkern_simd


def _ref_rtn_code(I, m):
    """规格 d 的整数域 RTN（half-to-even），返回商 q（不左移回去）。"""
    q = I >> m
    if m > 0:
        r = I - (q << m)
        half = 1 << (m - 1)
        if r > half or (r == half and (q & 1)):
            q += 1
    return q


# ---------------------------------------------------------------- W5
@pytest.mark.parametrize("level,m", [(4, 28), (8, 24), (16, 16)])
def test_w5_floor_domain_theorem(level, m):
    """floor 读法（算术右移）对全 int32 域恒在 b 位符号域内；RTN 顶码超一格。"""
    rng = np.random.default_rng(level * 7 + 5)
    codes = rng.integers(-2**31, 2**31, size=4096, dtype=np.int64)
    codes[0] = 2**31 - 1        # 正顶码（D3 满幅口径下 max 元素必达）
    codes[1] = -2**31           # 负顶码
    lo, hi = -(2 ** (level - 1)), 2 ** (level - 1) - 1
    floor_codes = codes >> m
    assert floor_codes.min() >= lo and floor_codes.max() <= hi
    assert (codes[0] >> m) == hi          # 正顶 → floor 恰在域顶
    assert (codes[1] >> m) == lo          # 负顶 → floor 恰在域底
    # RTN 读法：正顶码上取整超域（+2^(b−1)），B4a 例外的系统性复现
    rtn = np.array(_mk_nested.nested_view_codes([int(codes[0])], level))
    assert rtn[0] == hi + 1


def test_w5_random_codes_floor_vs_rtn_gap():
    """floor 与 RTN 读法差 ≤ 1 格且方向非负（RTN ≥ floor），方差同量级。"""
    rng = np.random.default_rng(11)
    codes = rng.integers(-2**31, 2**31, size=8192, dtype=np.int64)
    for m in (24, 28):
        floor_c = codes >> m
        rtn_c = np.array([_ref_rtn_code(int(c), m) for c in codes[:512]])
        gap = rtn_c - floor_c[:512]
        assert (gap >= 0).all() and (gap <= 1).all()


# ---------------------------------------------------------------- W6
@pytest.mark.parametrize("k", [1, 7, 255, 4096, 2**18, 2**18 + 5])
def test_w6_zeropoint_identity_dot8(k):
    """dot8(q_u8, w) − 128·Σw == Σ q_floor·w 逐位（int64 域，含 2^18 分块域）。"""
    rng = np.random.default_rng(k % 97 + 1)
    codes = rng.integers(-2**31, 2**31, size=k, dtype=np.int64)
    w = rng.integers(-127, 128, size=k, dtype=np.int8)
    q_u8 = ((codes >> 24) + 128).astype(np.uint8)
    got = _mk_simd.dot8(q_u8.tolist(), w.tolist()) - 128 * int(w.astype(np.int64).sum())
    ref = int((codes >> 24).astype(np.int64) @ w.astype(np.int64))
    assert got == ref


def test_w6_q4_affine_unpack_identity():
    """Q4 仿射解包路径：状态平面 = 偏置二进制 q_u4 = q_s4 + 8（非 two's
    complement——实现期修正，见设计文档 §三.4）→ unpack_nibble_u →
    dot8 − 8·Σw == Σ q_s4·w（逐位）。two's-complement 读法经 unpack_nibble_s
    精确恢复，一并钉死两种编码的语义差。"""
    rng = np.random.default_rng(23)
    k = 4096
    q_s4 = rng.integers(-8, 8, size=k, dtype=np.int64)
    w = rng.integers(-127, 128, size=k, dtype=np.int8)
    # 偏置二进制打包（低 nibble 在前，与 simd unpack_nibble_u 契约一致）
    u4 = (q_s4 + 8).astype(np.uint8)
    packed = ((u4[1::2] << 4) | u4[0::2]).astype(np.uint8)
    unpacked = np.array(_mk_simd.unpack_nibble_u(packed.tolist()), dtype=np.int64)
    assert (unpacked == q_s4 + 8).all()          # 仿射形式恰为 state_s4 + 8
    got = _mk_simd.dot8(unpacked.tolist(), w.tolist()) - 8 * int(w.astype(np.int64).sum())
    ref = int(q_s4 @ w.astype(np.int64))
    assert got == ref
    # 对照：two's-complement nibble（q mod 16）经 unpack_nibble_s 精确恢复 q
    twos = (q_s4 & 0xF).astype(np.uint8)
    packed_t = ((twos[1::2] << 4) | twos[0::2]).astype(np.uint8)
    s_back = np.array(_mk_simd.unpack_nibble_s(packed_t.tolist()), dtype=np.int64)
    assert (s_back == q_s4).all()


def test_w6_dot4_packed_roundtrip():
    """dot4_packed 4×4 往返：a∈[0,15]、b∈[−8,7] 打包布局逐位 == numpy 参考和。"""
    rng = np.random.default_rng(31)
    k = 8192
    a = rng.integers(0, 16, size=k, dtype=np.int64)
    b = rng.integers(-8, 8, size=k, dtype=np.int64)
    a_p = ((a[1::2] << 4) | a[0::2]).astype(np.uint8)
    b_p = ((b[1::2] & 0xF) << 4 | (b[0::2] & 0xF)).astype(np.uint8).view(np.int8)
    got = _mk_simd.dot4_packed(a_p.tolist(), b_p.tolist())
    ref = int(a @ b)
    assert got == ref


# ---------------------------------------------------------------- W7
def _random_groups(rng, n):
    """随机逐神经元档位分组（保证两组都可能为空/单元素的边界也被覆盖由调用方构造）。"""
    mask = rng.random(n) < 0.5
    return {"q8": mask, "q4": ~mask}


def _ref_dot_state(h, codes, w8_groups, groups):
    """numpy 独立参考：Σ_j q_floor_j·w_ij（Q8 组 floor 平面 + Q4 组 nibble）。"""
    n, b = h.shape
    out = np.zeros((b, w8_groups["q8"].shape[0]), dtype=np.int64)
    for j in range(b):
        qf8 = codes[j] >> 24
        q4 = codes[j] >> 28
        out[j] += (qf8[groups["q8"]].astype(np.int64)
                   @ w8_groups["q8"][:, groups["q8"]].T.astype(np.int64))
        out[j] += (q4[groups["q4"]].astype(np.int64)
                   @ w8_groups["q4"][:, groups["q4"]].T.astype(np.int64))
    return out


def test_w7_grouped_dot_native_vs_reference():
    """native dot8 路径 == numpy 参考路径逐位；组边界情形（全 Q8/全 Q4/单元素组）。"""
    rng = np.random.default_rng(41)
    n, b = 64, 3
    h = rng.normal(0, 0.5, size=(n, b)).astype(np.float32)
    ls = LeveledStateBatch(h, seeds=99)

    scenarios = []
    mask = _random_groups(rng, n)["q8"]
    scenarios.append({"q8": mask, "q4": ~mask})                       # 随机混合
    scenarios.append({"q8": np.ones(n, dtype=bool), "q4": np.zeros(n, dtype=bool)})   # 全 Q8
    scenarios.append({"q8": np.zeros(n, dtype=bool), "q4": np.ones(n, dtype=bool)})   # 全 Q4
    lone = np.zeros(n, dtype=bool); lone[7] = True
    scenarios.append({"q8": lone, "q4": ~lone})                       # 单元素 Q8 组
    lone4 = np.zeros(n, dtype=bool); lone4[13] = True
    scenarios.append({"q8": ~lone4, "q4": lone4})                     # 单元素 Q4 组

    for groups in scenarios:
        w8 = rng.integers(-127, 128, size=(10, n)).astype(np.int8)
        w4 = rng.integers(-127, 128, size=(10, n)).astype(np.int8)
        w = {"q8": w8, "q4": w4}
        got = ls.dot_state_weight(w, groups, native=_mk_simd)
        ref = _ref_dot_state(h, ls.codes, w, groups)
        assert (got == ref).all()


def test_w7_end_to_end_value_bound():
    """端到端值重建界：|step·(dot+零点修正的恢复) − Σ h·w| ≤ step·Σ|w|·(1+2⁻²⁴)。"""
    rng = np.random.default_rng(43)
    n, b = 64, 3
    h = rng.normal(0, 0.5, size=(n, b)).astype(np.float32)
    ls = LeveledStateBatch(h, seeds=99)
    step = ls.us.astype(np.float64) * 2.0 ** 24          # Q8 格距（每样本）
    groups = {"q8": np.ones(n, dtype=bool), "q4": np.zeros(n, dtype=bool)}
    w = rng.integers(-127, 128, size=(10, n)).astype(np.int8)
    w_ref = w.astype(np.float64)

    dot = ls.dot_state_weight({"q8": w, "q4": w[:, :0]}, groups, native=_mk_simd)
    for j in range(b):
        recon = dot[j].astype(np.float64) * step[j]      # Σ q_floor·w·step（码字域值）
        target = w_ref @ h[:, j].astype(np.float64)      # Σ h·w
        bound = step[j] * np.abs(w_ref).sum(axis=1) * (1.0 + 2.0 ** -24)
        assert (np.abs(recon - target) <= bound).all()


def test_w7_leveled_state_guards_and_half_cell():
    """守卫：零样本 u=0 全零码字；Q8 视图（RTN）half-cell 界。"""
    rng = np.random.default_rng(47)
    n, b = 32, 3
    h = rng.normal(0, 0.5, size=(n, b)).astype(np.float32)
    h[:, 0] = 0.0                                        # 零样本
    ls = LeveledStateBatch(h, seeds=13)
    assert ls.us[0] == 0.0 and (ls.codes[0] == 0).all()
    with pytest.raises(AssertionError):
        LeveledStateBatch(h[:1], seeds=1)                # N=1 单元素陷阱
    for j in (1, 2):
        deq = np.asarray(
            _mk_nested.nested_dequant(ls.codes[j].tolist(), float(ls.us[j]), 8),
            dtype=np.float64)
        scale = float(ls.us[j]) * 2.0 ** 24
        err = np.abs(deq - h[:, j].astype(np.float64)) / scale
        assert err.max() <= 0.5 + 1e-6                   # RTN half-cell 界


# ----------------------------------------------- W9 / 产品化推理循环
def _tiny_params(rng, n=8, in_dim=4, n_cls=3):
    """玩具 LTC 参数（层 2 设计 §五数据流的端到端最小形态）。"""
    return {
        "W_h": rng.normal(0, 0.2, size=(n, n)).astype(np.float32),
        "W_x": rng.normal(0, 0.3, size=(n, in_dim)).astype(np.float32),
        "b": rng.normal(0, 0.1, size=(n,)).astype(np.float32),
        "W_y": rng.normal(0, 0.4, size=(n_cls, n)).astype(np.float32),
        "b_y": rng.normal(0, 0.1, size=(n_cls,)).astype(np.float32),
        "dts": np.exp(np.linspace(np.log(0.05), np.log(0.6), n)).astype(np.float32),
    }


def test_w9_leveled_ltc_determinism_and_accounting():
    """产品化推理循环：确定性（同 seed 同 logits）+ W9 计数埋点精确。"""
    from engine.sgn.leveled_state import LeveledLTCInference
    rng = np.random.default_rng(53)
    p = _tiny_params(rng)
    x = rng.normal(0, 1, size=(6, 4, 20)).astype(np.float32)
    is_q8 = rng.random(8) < 0.5

    inf1 = LeveledLTCInference(p["W_h"], p["W_x"], p["b"], p["W_y"], p["b_y"],
                               p["dts"], is_q8=is_q8, weight_mode="float-w")
    lg1 = inf1.forward_chunk(x)
    acc1 = dict(inf1.accounting)
    inf2 = LeveledLTCInference(p["W_h"], p["W_x"], p["b"], p["W_y"], p["b_y"],
                               p["dts"], is_q8=is_q8, weight_mode="float-w")
    lg2 = inf2.forward_chunk(x)
    assert (lg1 == lg2).all()                            # 确定性（逐位）
    T, _, B = x.shape
    assert acc1["steps"] == T and acc1["samples"] == T * B
    assert acc1["quant_calls"] == T * B
    assert acc1["master_code_bytes"] == T * B * 8 * 4
    assert acc1["f32_rw_bytes"] == T * B * 8 * 4 * 2


def test_w9_leveled_ltc_first_step_quantize_bound():
    """t=0 携带态为零码字（W_h@h 贡献为零）→ 首步量化重建值落在
    h_pre 的 SR 邻域（每元素 |err| ≤ 1 个 Q8 步长）。"""
    from engine.sgn.leveled_state import LeveledLTCInference
    rng = np.random.default_rng(59)
    p = _tiny_params(rng)
    x0 = rng.normal(0, 1, size=(4,)).astype(np.float32)
    inf = LeveledLTCInference(p["W_h"], p["W_x"], p["b"], p["W_y"], p["b_y"],
                              p["dts"], weight_mode="float-w")
    z_ref = inf.W_x @ x0.astype(np.float64) + inf.b      # 携带态 0 → W_h@h = 0
    h_pre_ref = inf.dts * np.tanh(z_ref)
    mk = _mk_nested
    h_f32 = h_pre_ref.astype(np.float32)
    u = float(np.float32(np.abs(h_f32).max() * 2.0 ** -31))
    code = np.asarray(mk.nested_quant_i32(h_f32.tolist(), u, 7000), dtype=np.int64)
    h_recon = (code >> 24) * (2.0 ** 24) * u
    step = u * 2.0 ** 24
    assert (np.abs(h_recon - h_f32) <= step * (1.0 + 2.0 ** -24)).all()


def test_w9_int8_weight_quantization_bound():
    """int8-w：W̃ 与 W 的逐元素差 ≤ 0.5·s_row（对称量化的 RTN 界）。"""
    from engine.sgn.leveled_state import LeveledLTCInference
    rng = np.random.default_rng(61)
    p = _tiny_params(rng)
    inf = LeveledLTCInference(p["W_h"], p["W_x"], p["b"], p["W_y"], p["b_y"],
                              p["dts"], weight_mode="int8-w")
    w_tilde = inf.W_h_q * inf.s_row[:, None]
    err = np.abs(w_tilde - p["W_h"])
    assert (err <= 0.5 * inf.s_row[:, None] + 1e-6).all()


# -------------------------------- NP1 numpy 零拷贝热路径（2026-09-07 遗留项收口）
# 契约：_np 与 list 版调同一 C++ 内核 → 逐位一致；入侧 1-D C-contiguous +
# 精确 dtype 严格校验（不符 ValueError，不静默转换——裸 caster 实证会静默宽容）。
@pytest.mark.parametrize("k", [1, 7, 255, 4096, 2**18, 2**18 + 5])
def test_np1_simd_bitwise_parity(k):
    """simd 5 个 _np 函数 == list 版逐位（同内核），dtype/shape 契约钉死。"""
    rng = np.random.default_rng(k % 89 + 2)
    codes = rng.integers(-2**31, 2**31, size=k, dtype=np.int64)
    w = rng.integers(-127, 128, size=k, dtype=np.int8)
    q_u8 = ((codes >> 24) + 128).astype(np.uint8)
    # dot8 / dot4：np 版（指针直读）== list 版逐位
    assert _mk_simd.dot8_np(q_u8, w) == \
        _mk_simd.dot8(q_u8.tolist(), w.tolist())
    assert _mk_simd.dot4_np(q_u8, w) == \
        _mk_simd.dot4(q_u8.tolist(), w.tolist())
    # dot4_packed：打包 nibble（K = 2×len 隐式）
    a = rng.integers(0, 16, size=2 * k, dtype=np.int64)
    b = rng.integers(-8, 8, size=2 * k, dtype=np.int64)
    a_p = ((a[1::2] << 4) | a[0::2]).astype(np.uint8)
    b_p = ((b[1::2] & 0xF) << 4 | (b[0::2] & 0xF)).astype(np.uint8).view(np.int8)
    assert _mk_simd.dot4_packed_np(a_p, b_p) == \
        _mk_simd.dot4_packed(a_p.tolist(), b_p.tolist())
    # unpack：值 + dtype + shape（u8[2K] / int8[2K]）
    u4 = (rng.integers(0, 16, size=2 * k, dtype=np.int64)).astype(np.uint8)
    packed = ((u4[1::2] << 4) | u4[0::2]).astype(np.uint8)
    up_np = _mk_simd.unpack_nibble_u_np(packed)
    up_ls = _mk_simd.unpack_nibble_u(packed.tolist())
    assert up_np.dtype == np.uint8 and up_np.shape == (2 * k,)
    assert (up_np == np.asarray(up_ls)).all()
    twos = (rng.integers(-8, 8, size=2 * k, dtype=np.int64) & 0xF).astype(np.uint8)
    packed_t = ((twos[1::2] << 4) | twos[0::2]).astype(np.uint8)
    sp_np = _mk_simd.unpack_nibble_s_np(packed_t)
    sp_ls = _mk_simd.unpack_nibble_s(packed_t.tolist())
    assert sp_np.dtype == np.int8 and sp_np.shape == (2 * k,)
    assert (sp_np == np.asarray(sp_ls)).all()


def test_np1_nested_bitwise_parity():
    """nested _np == list 版逐位：quant 同 seed 多 u 码字逐位；dequant float32
    用 view(uint32) 位模式比较（list 版 Python float = f32 精确拓宽）。"""
    rng = np.random.default_rng(71)
    n = 257
    for u in (1e-8, 2.0 ** -31, 0.003):
        h = rng.normal(0, 0.5, size=n).astype(np.float32)
        h[0] = 0.0                                        # 吸收态
        seed = 4242
        q_np = _mk_nested.nested_quant_i32_np(h, u, seed)
        q_ls = np.asarray(
            _mk_nested.nested_quant_i32(h.tolist(), u, seed), dtype=np.int64)
        assert q_np.dtype == np.int64 and (q_np == q_ls).all()
        d_np = _mk_nested.nested_dequant_np(q_np, u, 8)
        d_ls = _mk_nested.nested_dequant(q_ls.tolist(), u, 8)
        assert d_np.dtype == np.float32
        assert (d_np.view(np.uint32) ==
                np.asarray(d_ls, dtype=np.float32).view(np.uint32)).all()


def test_np1_strict_contract_rejection():
    """零拷贝严格契约：错 dtype / 非连续切片 / 2-D / Python list 一律 ValueError
    （裸 caster 会静默转换/拷贝——本测试把显式报错钉死为回归锚）。"""
    ok_u8 = np.zeros(8, dtype=np.uint8)
    ok_i8 = np.zeros(8, dtype=np.int8)
    ok_f32 = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError):
        _mk_simd.dot8_np(ok_u8.astype(np.int32), ok_i8)          # 错 dtype
    with pytest.raises(ValueError):
        _mk_simd.dot8_np(ok_u8, ok_i8.astype(np.int64))          # 错 dtype（b 侧）
    with pytest.raises(ValueError):
        col = np.zeros((8, 8), dtype=np.uint8)[:, 0]             # 非连续列切片
        _mk_simd.dot8_np(col, ok_i8)
    with pytest.raises(ValueError):
        _mk_simd.dot8_np(ok_u8.reshape(2, 4), ok_i8.reshape(2, 4))  # 2-D
    with pytest.raises(ValueError):
        _mk_simd.dot8_np(ok_u8.tolist(), ok_i8.tolist())         # Python list
    with pytest.raises(ValueError):
        _mk_nested.nested_quant_i32_np(ok_f32.astype(np.float64), 1e-8, 1)  # 错 dtype
    with pytest.raises(ValueError):
        _mk_nested.nested_quant_i32_np(ok_f32.reshape(2, 4), 1e-8, 1)       # 2-D
    with pytest.raises(ValueError):
        _mk_nested.nested_quant_i32_np(ok_f32.tolist(), 1e-8, 1)  # Python list
    codes = np.zeros(8, dtype=np.int64)
    with pytest.raises(ValueError):
        _mk_nested.nested_dequant_np(codes, 1e-8, 3)             # level 校验保留
    with pytest.raises(ValueError):
        _mk_nested.nested_dequant_np(codes.reshape(2, 4), 1e-8, 8)          # 2-D


def test_np1_forward_chunk_np_vs_list_path_bitwise():
    """端到端双路径对拍：屏蔽 _np（monkeypatch 置 None 强制 list 回退）后
    forward_chunk logits 逐位相等，accounting 亦相等——数值语义不变的 pin。"""
    from engine.sgn.leveled_state import LeveledLTCInference
    rng = np.random.default_rng(73)
    p = _tiny_params(rng)
    x = rng.normal(0, 1, size=(6, 4, 20)).astype(np.float32)
    is_q8 = rng.random(8) < 0.5                                  # 混合组（Q4 肢也走热路径）
    inf_np = LeveledLTCInference(p["W_h"], p["W_x"], p["b"], p["W_y"], p["b_y"],
                                 p["dts"], is_q8=is_q8, weight_mode="int8-w")
    lg_np = inf_np.forward_chunk(x)
    acc_np = dict(inf_np.accounting)
    monkey_targets = [
        (_mk_nested, "nested_quant_i32_np"),
        (_mk_nested, "nested_dequant_np"),
        (_mk_simd, "dot8_np"),
        (_mk_simd, "unpack_nibble_u_np"),
    ]
    for mod, name in monkey_targets:
        assert getattr(mod, name, None) is not None              # _np 在位才有屏蔽意义
    import contextlib
    @contextlib.contextmanager
    def _np_disabled():
        saved = [(mod, name, getattr(mod, name, None)) for mod, name in monkey_targets]
        try:
            for mod, name in monkey_targets:
                setattr(mod, name, None)                         # None = 缺失 → 回退
            yield
        finally:
            for mod, name, val in saved:
                if val is not None:
                    setattr(mod, name, val)
    with _np_disabled():
        inf_ls = LeveledLTCInference(p["W_h"], p["W_x"], p["b"], p["W_y"], p["b_y"],
                                     p["dts"], is_q8=is_q8, weight_mode="int8-w")
        lg_ls = inf_ls.forward_chunk(x)
        acc_ls = dict(inf_ls.accounting)
    assert (lg_np == lg_ls).all()                                # logits 逐位
    assert acc_np == acc_ls                                      # W9 埋点不受影响


def test_np1_dot_state_weight_path_parity():
    """dot_state_weight native _np 路径 == 屏蔽 _np 后的 list 回退路径
    （(B, n_out) int64 逐位；混合组覆盖 Q8+Q4 双肢）。"""
    rng = np.random.default_rng(79)
    n, b = 64, 3
    h = rng.normal(0, 0.5, size=(n, b)).astype(np.float32)
    ls = LeveledStateBatch(h, seeds=99)
    groups = {"q8": _random_groups(rng, n)["q8"], "q4": None}
    groups["q4"] = ~groups["q8"]
    w = {"q8": rng.integers(-127, 128, size=(10, n)).astype(np.int8),
         "q4": rng.integers(-127, 128, size=(10, n)).astype(np.int8)}
    got_np = ls.dot_state_weight(w, groups, native=_mk_simd)
    saved = getattr(_mk_simd, "dot8_np", None)
    saved_u = getattr(_mk_simd, "unpack_nibble_u_np", None)
    try:
        _mk_simd.dot8_np = None
        _mk_simd.unpack_nibble_u_np = None
        got_ls = ls.dot_state_weight(w, groups, native=_mk_simd)
    finally:
        _mk_simd.dot8_np = saved
        _mk_simd.unpack_nibble_u_np = saved_u
    assert (got_np == got_ls).all()
