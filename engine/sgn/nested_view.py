# nested_view.py - 嵌套量化消费方薄层（Phase 1 接线，nested_quant 设计档 §五.5）
#
# 职责（nested_quant_Level接线设计_2026_09_05.md §五 草案的正式化）：
#   把 sgn.mkern_nested 原语包成调度器友好的视图对象：
#     - u 的同源约束（D3 已拍板）：u 在构造时由 max|h|·2⁻³¹ 一次性计算并冻结，
#       消费方不可传入自己的 u（执行记录 §十.4：u 与 max 必须同源、float32 口径）
#     - D1 吸附规则（已拍板）：调度器连续 bits → 嵌套档 {4,8,16,32}
#     - 任意档位序列的视图取用（零重采样：视图 = (code, level) 的纯函数）
#   不做：调度器内部逻辑、训练循环、有损打包存储（Phase 2，见 storage_bits 注）。
#
# 已知边界（如实声明，见测试 test_nested_view.py）：
#   - 零张量：u=0 违反原语前置条件 → 构造时特判为全零码字（视图恒 0）
#   - 单元素张量：u 取自身幅值 → 所有视图 ≈ 原值（S0-A 单元素恒等陷阱的嵌套
#     形态）——量化对象须 ≥2 元素，调用方负责
#   - n=0 合法（空视图，dequant 返回空）

import numpy as np


class NestedLevelView:
    """调度器 ↔ 嵌套原语的接线薄层（Phase 1：吸附式，D1 已拍板）。

    一次 nested_quant 产出码字，任意档位视图即取（切换零重采样噪声，
    棘轮探针 §十一：档位轨迹由调度器决定，嵌套视图降低其噪声税）。

    属性：
      n     : 元素数
      u     : 最细格距（float32，max|h|·2⁻³¹ 满幅口径；零张量为 0.0）
      seed  : 量化种子（同 seed 同输入同码字，kBitExact）
      code  : int64 ndarray，嵌套 int32 码字（零张量为全 0）
      max_abs: 构造时冻结的 max|h|（u 的同源来源）
    """

    LEVELS = (4, 8, 16, 32)

    def __init__(self, h, seed: int = 0):
        h = np.asarray(h, dtype=np.float32).ravel()
        self.n = int(h.size)
        self.seed = int(seed)
        self.max_abs = float(np.max(np.abs(h))) if self.n else 0.0
        # D3 满幅口径（同源冻结）：u = max|h|·2⁻³¹，按原语签名 float32 化
        self.u = float(np.float32(self.max_abs * 2.0 ** -31))
        if self.u <= 0.0:
            # 零张量守卫：u>0 是原语前置条件；零向量各档视图恒 0，直接给零码字
            self.code = np.zeros(self.n, dtype=np.int64)
        else:
            self.code = np.asarray(
                sgn_call_quant(h, self.u, self.seed), dtype=np.int64)

    # ---- D1 吸附规则（拍板 2026-09-05：b<6→4 / 6-11→8 / 12-23→16 / ≥24→32）----
    @classmethod
    def snap(cls, bits: int) -> int:
        b = int(bits)
        if b < 6:
            return 4
        if b < 12:
            return 8
        if b < 24:
            return 16
        return 32

    # ---- 视图取用（零重采样：结果 = (code, level) 的纯函数）----
    def dequant(self, level: int):
        """level 档视图（level ∈ {4,8,16,32}，其他值 ValueError——比 C++ 契约
        的无操作更严格）。零张量视图恒 0。"""
        if level not in self.LEVELS:
            raise ValueError(f"level must be one of {self.LEVELS}, got {level!r}")
        if self.u <= 0.0:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(
            call_dequant(self.code, self.u, level), dtype=np.float32)

    def dequant_bits(self, bits: int):
        """调度器 bits 口径的视图 = dequant(snap(bits))（D1 吸附后取档）。"""
        return self.dequant(self.snap(bits))

    def view_codes(self, level: int):
        """level 档视图整数码（b bit 有符号：RTN 商，int64 ndarray）。

        dequant 值 = view_codes · 2^(32−level) · u。喂 MSint 点积域用：
        Q8 → int8/dot8、Q16 → int16/pair 载体、Q4 → int4/dot4（S3 衔接评估
        B3/B4：嵌套 Q16 视图码 ∈ int16，直接过 pair 载体恒等式）。
        零张量返回全 0。"""
        if level not in self.LEVELS:
            raise ValueError(f"level must be one of {self.LEVELS}, got {level!r}")
        if self.u <= 0.0:
            return np.zeros(self.n, dtype=np.int64)
        return np.asarray(call_view_codes(self.code, level), dtype=np.int64)

    # ---- 存储记账（Phase 1 只记账不打包，见下注）----
    def storage_bits(self, level: int) -> int:
        """level 档视图的信息量下界 = level + 1 bit（b 档 + RTN 决策位）。

        注（Phase 2 挂起项）：按档有损打包存储时，截码重建的视图在 RTN 边界
        case（余数恰半）可与全码视图差一格——Phase 1 不实现有损打包，消费方
        持全码（int32），此方法仅作压缩率记账。
        """
        if level not in self.LEVELS:
            raise ValueError(f"level must be one of {self.LEVELS}, got {level!r}")
        return int(level) + 1


# ---- 原语调用（延迟定位 mkern_nested：兼容 sgn.* 与 engine.sgn.* 两种导入身份）----

def _native():
    import sys
    for name in ("sgn", "engine.sgn"):
        mod = sys.modules.get(name)
        mk = getattr(mod, "mkern_nested", None) if mod is not None else None
        if mk is not None:
            return mk
    import sgn as _sgn          # 兜底：engine 在 sys.path 的顶层导入场景
    return _sgn.mkern_nested


def sgn_call_quant(h, u, seed):
    return _native().nested_quant_i32(h.tolist(), u, seed)


def call_dequant(code, u, level):
    return _native().nested_dequant(code.tolist(), u, level)


def call_view_codes(code, level):
    return _native().nested_view_codes(code.tolist(), level)
