# leveled_state.py - LeveledState 状态布局（层 2 接线 Phase 2a，设计文档 §四/§五）
#
# 设计依据：docs/ltc_cfc_dynamic_quantization/层2接线设计_LeveledState状态布局_2026_09_06.md
#   - 每样本 u_b = max|h|·2⁻³¹（D3 满幅口径，同源冻结；u 与 max 同一函数产出）
#   - 两级存储：母版 = int32 码字（本骨架的操作层，训练/校准/分发期），
#     运行态 = floor 平面（Q8 u8 1B / Q4 nibble 0.5B，带宽账主体）；
#     档位只影响消费读法，切档零重量化（前缀性质）
#   - 点积消费读法 = floor 字节平面（算术右移，域内定理：Q8 [−128,127]、
#     Q4 [−8,7]——RTN 读法顶码 +2^(b−1) 超域的系统性解，设计文档 §三）
#   - 零点修正恒等式：Σ h·w = 2^24·u·(dot8(q_u8, w) − 128·Σw)
#
# 方向状态：开环推理路径（层 2），无梯度消费（floor 读法梯度禁令不变），
#           训练走层 1 配方（state_precision_recipe.py，torch STE）。
#
# 边界（2026-09-07 零拷贝收口后）：
#   - 热路径走 sgn.mkern_nested/nested_quant_i32_np、nested_dequant_np 与
#     sgn.mkern_simd/dot8_np、unpack_nibble_u_np（1-D C-contiguous 精确 dtype
#     严格契约，不符 ValueError）；_np 缺失（getattr=None，monkeypatch 友好）
#     回退 list 版——两路径同 C++ 内核逐位一致（tests/test_leveled_state.py）
#   - forward_chunk 主存储 h_code_t (B,N) 行主（逐样本行 = 连续零拷贝切片），
#     读侧 h_code = h_code_t.T 视图——数值语义不变（RNG 流按缓冲下标计数，
#     转置不改列内元素顺序）
#   - 每样本循环调用 mkern_nested（B 组 Python 循环）——批量化属于系统账批次

import numpy as np

LEVELS = (4, 8, 16, 32)
_ZERO_POINT_Q8 = 128
_ZERO_POINT_Q4 = 8


def _native():
    import sys
    for name in ("sgn", "engine.sgn"):
        mod = sys.modules.get(name)
        mk = getattr(mod, "mkern_nested", None) if mod is not None else None
        if mk is not None:
            return mk
    import sgn as _sgn
    return _sgn.mkern_nested


class LeveledStateBatch:
    """B 个样本的 LeveledState 组（LTC 状态 (N,B) 的码字布局）。

    属性（每组）：code int64[N]（嵌套码字）、u float32（满幅格距）、
    max_abs（u 的同源来源）。零样本守卫：u=0 → 全零码字。
    """

    def __init__(self, h, seeds):
        """h: (N,B) float 状态；seeds: int 或长度 B 序列（代次种子）。"""
        h = np.asarray(h, dtype=np.float32)
        assert h.ndim == 2, "LeveledStateBatch 吃 (N,B) 状态"
        n, b = h.shape
        assert n >= 2, "量化对象须 ≥2 元素（单元素 u 取自身幅值陷阱，同 nested_view）"
        self.n, self.b = int(n), int(b)
        seeds = np.broadcast_to(np.asarray(seeds, dtype=np.uint64), (b,))
        self.codes = np.zeros((self.b, self.n), dtype=np.int64)
        self.us = np.zeros(self.b, dtype=np.float32)
        self.max_abs = np.zeros(self.b, dtype=np.float32)
        mk = _native()
        q_np = getattr(mk, "nested_quant_i32_np", None)   # None 视为缺失 → 回退 list 版
        for j in range(b):
            col = np.ascontiguousarray(h[:, j])
            m = float(np.max(np.abs(col))) if self.n else 0.0
            self.max_abs[j] = m
            u = float(np.float32(m * 2.0 ** -31))       # D3 同源：u 与 max 同函数产出
            self.us[j] = u
            if u <= 0.0:
                continue                                 # 零样本 → 全零码字
            if q_np is not None:
                self.codes[j] = q_np(col, u, int(seeds[j]))
            else:
                self.codes[j] = np.asarray(
                    mk.nested_quant_i32(col.tolist(), u, int(seeds[j])),
                    dtype=np.int64)

    # ---- floor 平面提取（域内定理：设计文档 §三.2；纯整数移位）----
    def q8_u8_plane(self):
        """Q8 消费平面：q_u8 = (code >> 24) + 128 ∈ [0,255]（无回绕）。
        返回 (B,N) uint8。"""
        return ((self.codes >> 24) + _ZERO_POINT_Q8).astype(np.uint8)

    def q4_nibble_s(self):
        """Q4 有符号读法：q_s4 = (code >> 28) ∈ [−8,7]。
        返回 (B,N) int64（符号域参考读法）。"""
        return self.codes >> 28

    def q4_u4_plane(self):
        """Q4 消费平面（偏置二进制）：q_u4 = (code >> 28) + 8 ∈ [0,15]。
        返回 (B,N) uint8。与 Q8 同构（零点 = 2^(b−1)）：two's-complement
        nibble（q mod 16）≠ 仿射形式（q+8）——消费平面必须用后者，
        设计文档 §三.4（2026-09-07 实现期修正）。"""
        return ((self.codes >> 28) + 8).astype(np.uint8)

    def q4_u4_packed(self):
        """Q4 消费平面的 nibble 打包（低 nibble 在前：byte[j] = elem[2j+1]<<4 |
        elem[2j]，与 simd unpack_nibble_u 契约一致）。返回 (B, ceil(N/2)) uint8，
        存储 0.5 B/元素（带宽账主体）。"""
        u4 = self.q4_u4_plane()
        pad = u4.shape[1] % 2
        if pad:
            u4 = np.concatenate([u4, np.zeros((u4.shape[0], 1), dtype=np.uint8)], axis=1)
        return (u4[:, 1::2] << 4) | u4[:, 0::2]

    # ---- 消费点积（参考实现；native 适配位见 NativeDots）----
    def dot_state_weight(self, w_int8_by_group, groups, native=None):
        """分组点积：Σ_j h_j·w_ij 的码字域参考值（int64，未乘 2^(32−b)·u）。

        w_int8_by_group: {"q8": int8[n_out,N], "q4": int8[n_out,N]}——W_h 全宽
                         权重（行 = 输出神经元），组内列由 groups 选择
        groups: {"q8": bool[N], "q4": bool[N]}——h_j 的逐神经元档位
        native: 含 dot8/unpack_nibble_u 的模块适配位（传 sgn.mkern_simd；
                None 走 numpy 参考实现——两路径逐位一致由 W7 钉死）
        返回 (B, n_out) int64：Σ_j q_floor_j·w_ij，零点修正已含
        （Q8 组：dot8 − 128·Σw；Q4 组：native = unpack+dot8 − 8·Σw /
        numpy = 全字节精确 matmul——两式均 == Σ q_s·w）；量纲恢复
        2^(32−b)·u 与 floor 偏差折算由调用方按设计文档 §五.3 处理。
        """
        w8 = w_int8_by_group["q8"]
        w4 = w_int8_by_group["q4"]
        n_out = w8.shape[0]
        out = np.zeros((self.b, n_out), dtype=np.int64)
        q8 = self.q8_u8_plane()
        q4_u4 = self.q4_u4_plane()
        k8 = groups["q8"]
        k4 = groups["q4"]
        # _np 零拷贝探测（None 视为缺失 → 回退 list 版，monkeypatch 友好）
        dot8_np = (getattr(native, "dot8_np", None)
                   if native is not None else None)
        unpack_np = (getattr(native, "unpack_nibble_u_np", None)
                     if native is not None else None)
        for j in range(self.b):
            if k8.any():
                if native is not None:
                    a8 = q8[j][k8]     # bool 掩码索引 → 新 C-contiguous 数组（严格契约满足）
                    for i in range(n_out):
                        w_row = w8[i][k8]
                        dot = (dot8_np(a8, w_row) if dot8_np is not None
                               else native.dot8(a8, w_row))
                        out[j, i] += dot \
                            - _ZERO_POINT_Q8 * int(w_row.astype(np.int64).sum())
                else:
                    out[j] += (q8[j][k8].astype(np.int64)
                               @ w8[:, k8].T.astype(np.int64)) \
                        - _ZERO_POINT_Q8 * w8[:, k8].sum(axis=1)
            if k4.any():
                if native is not None:
                    packed = self.q4_u4_packed()[j]
                    if unpack_np is not None:
                        unpacked = unpack_np(packed)     # numpy u8[2K]
                    else:
                        unpacked = np.asarray(
                            native.unpack_nibble_u(packed.tolist()), dtype=np.uint8)
                    unpacked = unpacked[: self.n]           # 去掉补位
                    a4 = unpacked[k4]                       # bool 掩码 → C-contiguous
                    for i in range(n_out):
                        w_row = w4[i][k4]
                        dot = (dot8_np(a4, w_row) if dot8_np is not None
                               else native.dot8(a4, w_row))
                        out[j, i] += dot \
                            - _ZERO_POINT_Q4 * int(w_row.astype(np.int64).sum())
                else:
                    out[j] += (q4_u4[j][k4].astype(np.int64)
                               @ w4[:, k4].T.astype(np.int64)) \
                        - _ZERO_POINT_Q4 * w4[:, k4].sum(axis=1)
        return out


def selfcheck(seed=7, n=64, b=3):
    """骨架自检（毫秒级）：域内定理 / 零点恒等式 / 零样本守卫 / 值重建。"""
    rng = np.random.default_rng(seed)
    mk = _native()
    # 1) 域内定理：任意码字（含顶码）floor 读法恒在符号域内
    codes = rng.integers(-2**31, 2**31, size=(n,), dtype=np.int64)
    codes[0] = 2**31 - 1                                  # 正顶码（RTN 超、floor 域内）
    codes[1] = -2**31                                     # 负顶码
    assert (codes >> 24).min() >= -128 and (codes >> 24).max() <= 127
    assert (codes >> 28).min() >= -8 and (codes >> 28).max() <= 7
    assert (codes[0] >> 24) == 127 and (codes[1] >> 24) == -128
    # 2) RTN 顶码超域复现（B4a 例外的回归锚）
    rtn_top = mk.nested_view_codes([int(codes[0])], 8)[0]
    assert rtn_top == 128
    # 3) 零点恒等式：dot(q_u8,w) − 128·Σw == Σ q_floor·w（逐位）
    w = rng.integers(-127, 128, size=(n,)).astype(np.int64)
    q_u8 = (codes >> 24).astype(np.int64) + 128
    q_floor = (codes >> 24).astype(np.int64)
    assert ((q_u8 @ w - 128 * w.sum()) == (q_floor @ w)).all()
    # 4) LeveledStateBatch 值重建：dequant(q8 视图) ≈ RTN 值；零样本守卫
    h = rng.normal(0, 0.5, size=(n, b)).astype(np.float32)
    h[:, 0] = 0.0                                          # 零样本
    ls = LeveledStateBatch(h, seeds=11)
    assert ls.us[0] == 0.0 and (ls.codes[0] == 0).all()
    j = 1
    deq = np.asarray(mk.nested_dequant(ls.codes[j].tolist(), float(ls.us[j]), 8),
                     dtype=np.float64)
    scale = float(ls.us[j]) * 2.0 ** 24
    err = np.abs(deq - h[:, j]) / max(scale, 1e-30)
    assert err.max() <= 0.5 + 1e-6, f"Q8 视图 RTN half-cell 界破 {err.max()}"
    print("selfcheck: 域内定理 / RTN 顶码回归 / 零点恒等式 / 零样本守卫 / "
          "Q8 half-cell 界 —— PASS")


# ---- 端到端 LTC 平面推理（层 2 设计 §五数据流产品化；V-L2W8 验收语义）----
_LEVEL_STEP = {8: 2.0 ** 24, 4: 2.0 ** 28}
_LEVEL_ZP = {8: 128, 4: 8}


class LeveledLTCInference:
    """LTC 单元的平面推理循环（码字即携带态，层 2 设计 §五）。

    每步：重建携带码字 → W_h@h 分组整数点积（零点/量纲恢复）→ float 侧
    （W_x@x_t + b）→ tanh 动力学 → 每样本重量化换代（u_b 同源重算）→
    池化新携带态。W9 计数埋点内建（accounting 属性）。

    参数：
      W_h (N,N) float32；W_x (N,in)；b (N)；W_y (10,N)；b_y (10)；dts (N,)
      is_q8 (N,) bool（None = 全 Q8）
      weight_mode: "int8-w"（W_h per-row 对称 int8 量化——部署形态，
                   V-L2W8 门控实测 Δ≤0.19 点）/ "float-w"（归因形态）
      reconstruct: "floor"（平面重建）/ "rtn"（dequant 视图重建——精度更优
                   （V-L2W8 report-only），每步多 dequant 调用；仅全 Q8 支持）
    """

    def __init__(self, W_h, W_x, b, W_y, b_y, dts, is_q8=None,
                 weight_mode="int8-w", reconstruct="floor", seed_base=7000):
        self.W_h = np.asarray(W_h, dtype=np.float32)
        self.W_x = np.asarray(W_x, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.W_y = np.asarray(W_y, dtype=np.float64)
        self.b_y = np.asarray(b_y, dtype=np.float64)
        self.dts = np.asarray(dts, dtype=np.float64)
        self.n = self.W_h.shape[0]
        self.is_q8 = (np.ones(self.n, dtype=bool) if is_q8 is None
                      else np.asarray(is_q8, dtype=bool))
        self.k8 = np.where(self.is_q8)[0]
        self.k4 = np.where(~self.is_q8)[0]
        assert reconstruct in ("floor", "rtn")
        assert reconstruct == "floor" or self.k4.size == 0, \
            "rtn 重建暂仅全 Q8 支持"
        self.reconstruct = reconstruct
        self.weight_mode = weight_mode
        self.seed_base = int(seed_base)
        if weight_mode == "int8-w":
            self.s_row = np.abs(self.W_h).max(axis=1) / 127.0
            self.s_row[self.s_row == 0] = 1.0
            self.W_h_q = np.round(self.W_h / self.s_row[:, None]).astype(np.int64)
        else:
            self.s_row = np.ones(self.n)
            self.W_h_q = None
        # W9 计数埋点（系统账解析账口径，层 2 设计 §九）
        self.accounting = {"steps": 0, "samples": 0, "quant_calls": 0,
                           "master_code_bytes": 0, "plane_rw_bytes": 0,
                           "f32_rw_bytes": 0}

    # -- 携带码字 → 状态值 --
    def _reconstruct(self, h_code, u):
        h = np.zeros((self.n, h_code.shape[1]))
        if self.reconstruct == "floor":
            if self.k8.size:
                h[self.k8] = (h_code[self.k8] >> 24) * _LEVEL_STEP[8] * u[None, :]
            if self.k4.size:
                h[self.k4] = (h_code[self.k4] >> 28) * _LEVEL_STEP[4] * u[None, :]
            return h
        mk = _native()
        dq_np = getattr(mk, "nested_dequant_np", None)
        hc_t = np.ascontiguousarray(h_code.T)   # (B,N) 行主；h_code 本为 h_code_t.T 视图时 no-op
        for s in range(h_code.shape[1]):
            if u[s] > 0:
                if dq_np is not None:
                    h[:, s] = dq_np(hc_t[s], float(u[s]), 8)   # f32→f64 拓宽精确
                else:
                    h[:, s] = np.asarray(
                        mk.nested_dequant(hc_t[s].tolist(), float(u[s]), 8),
                        dtype=np.float64)
        return h

    def _whh_contrib(self, h_code, u):
        """W_h@h 分组整数点积（列组 = 状态神经元 = 收缩维；numpy 整数路径，
        与 native dot8 逐位等价——W6/W7 钉死）。"""
        contrib = np.zeros((self.n, h_code.shape[1]))
        if self.k8.size:
            q8 = h_code[self.k8] >> 24
            if self.weight_mode == "int8-w":
                Wq = self.W_h_q[:, self.k8]
                dot = Wq @ q8
                contrib += dot * _LEVEL_STEP[8] * (self.s_row[:, None] * u[None, :])
            else:
                contrib += (self.W_h[:, self.k8].astype(np.float64)
                            @ q8.astype(np.float64)) * _LEVEL_STEP[8] * u[None, :]
        if self.k4.size:
            q4 = h_code[self.k4] >> 28
            if self.weight_mode == "int8-w":
                Wq = self.W_h_q[:, self.k4]
                dot = Wq @ q4
                contrib += dot * _LEVEL_STEP[4] * (self.s_row[:, None] * u[None, :])
            else:
                contrib += (self.W_h[:, self.k4].astype(np.float64)
                            @ q4.astype(np.float64)) * _LEVEL_STEP[4] * u[None, :]
        return contrib

    def forward_chunk(self, xb, seed_off=0):
        """单 chunk 前向。xb (T, in_dim, B) float32；返回 (B, n_classes) logits。"""
        T, _, B = xb.shape
        mk = _native()
        q_np = getattr(mk, "nested_quant_i32_np", None)   # None 视为缺失 → 回退 list 版
        # 主存储 (B,N) 行主：逐样本行 = 连续零拷贝切片；读侧用 .T 视图（切片语义不变）
        h_code_t = np.zeros((B, self.n), dtype=np.int64)
        h_code = h_code_t.T
        u = np.zeros(B)
        hb = np.zeros((self.n, B))
        acc = self.accounting
        for t in range(T):
            acc["steps"] += 1
            acc["samples"] += B
            acc["quant_calls"] += B
            acc["master_code_bytes"] += self.n * 4 * B
            bytes_per = (1.0 if self.k4.size == 0 else
                         (self.k8.size + 0.5 * self.k4.size) / self.n)
            acc["plane_rw_bytes"] += int(round(2 * self.n * bytes_per * B))
            acc["f32_rw_bytes"] += 2 * self.n * 4 * B
            h = self._reconstruct(h_code, u)
            z = self._whh_contrib(h_code, u) \
                + self.W_x @ xb[t].astype(np.float64) + self.b[:, None]
            h_pre = h + self.dts[:, None] * (-h + np.tanh(z))
            h_f32 = h_pre.astype(np.float32)
            u = np.abs(h_f32).max(axis=0).astype(np.float32) * np.float32(2.0 ** -31)
            h_f32_t = np.ascontiguousarray(h_f32.T)   # 每步一次 (B,N) 连续化；转置不改元素序
            seed_t = self.seed_base + seed_off + t    # 同步各样本共用同一 seed（冻结语义）
            for s in range(B):
                if u[s] > 0:
                    if q_np is not None:
                        h_code_t[s] = q_np(h_f32_t[s], float(u[s]), seed_t)
                    else:
                        h_code_t[s] = np.asarray(
                            mk.nested_quant_i32(h_f32_t[s].tolist(), float(u[s]),
                                                seed_t),
                            dtype=np.int64)
                else:
                    h_code_t[s] = 0
            hb += self._reconstruct(h_code, u)
        return (self.W_y @ (hb / T) + self.b_y[:, None]).T   # (B, n_classes)

    def accuracy(self, x, y, bs=1000):
        """x (T, in, S) float32 numpy；返回测试准确率。"""
        correct, start = 0, 0
        while start < x.shape[2]:
            xb = x[:, :, start:start + bs]
            logits = self.forward_chunk(xb)
            correct += int((logits.argmax(axis=1) == y[start:start + xb.shape[2]]).sum())
            start += xb.shape[2]
        return correct / x.shape[2]


if __name__ == "__main__":
    selfcheck()
