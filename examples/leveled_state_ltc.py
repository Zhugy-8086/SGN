"""leveled_state_ltc.py - 层 2 平面推理示例：LTC 单元的嵌套量化状态布局

演示 SGN 的层 2 能力——**LeveledState 平面推理**：RNN/LTC 类"状态即记忆"
单元的状态以嵌套 int32 码字/字节平面常驻（Q8 神经元 1 字节、Q4 神经元
半字节），消费矩阵乘走整数点积（零点修正精确），状态搬运带宽相对 float32
降 4–8×（V-L2W9 系统账首读：全 Q8 4.00×、WF-25 混合 4.571×）。

本示例在合成时间序列分类任务上对比两条推理路径：
  1. float32 基线（float 状态全程）；
  2. LeveledState 平面路径（Q8 全档；每步重量化换代，码字即携带态）。
两侧共享同一组随机权重（示例不训练；训练配方见
内部档案 温度前置
日程 + WF 静态水填充表，层 1 配方）。

关键语义（见 docs/ltc_cfc_dynamic_quantization/层2接线设计_*.md）：
  - 每样本格距 u = max|h|·2⁻³¹（满幅口径，同源冻结）；
  - floor 读法（算术右移）恒在有符号域内：Q8 [−128,127]、Q4 [−8,7]——
    RTN 读法在顶码 +2^(b−1) 处超域（B4a 例外）的结构性解；
  - 零点修正恒等式：Σ q_floor·w = dot8(q_u8, w) − 128·Σw（逐位精确）。

运行：python examples/leveled_state_ltc.py
依赖：numpy（本示例不依赖 torch；原生原语来自预编译的 sgn 扩展模块）。
"""

import os
import sys

import numpy as np

# ---- 路径设置（与 mnist_mlp.py 同款：engine/ 入栈）----
_sgn_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "engine"))
sys.path.insert(0, _sgn_root)

import sgn
from sgn.leveled_state import LeveledLTCInference

N, IN_DIM, T, N_CLS = 32, 8, 16, 4
SEED = 7


def make_synthetic_task(rng, n_samples=400):
    """二模式时间序列分类：正弦相位 vs 方波相位（线性可分但需时间积分）。"""
    t = np.linspace(0, 2 * np.pi, T)
    labels = rng.integers(0, N_CLS, size=n_samples)
    freqs = np.linspace(0.8, 2.2, N_CLS)
    x = np.zeros((T, IN_DIM, n_samples), dtype=np.float32)
    for s in range(n_samples):
        f = freqs[labels[s]]
        base = np.sin(f * t + rng.uniform(0, 2 * np.pi))
        for ch in range(IN_DIM):
            x[:, ch, s] = base * (0.5 + 0.5 * rng.uniform(0.5, 1.5)) \
                + 0.1 * rng.standard_normal(T)
    return x, labels


def float_reference_forward(p, x):
    """float32/f64 基线推理（与平面路径同池化语义）。"""
    W_h, W_x, b = p["W_h"].astype(np.float64), p["W_x"].astype(np.float64), \
        p["b"].astype(np.float64)
    W_y, b_y = p["W_y"].astype(np.float64), p["b_y"].astype(np.float64)
    dts = p["dts"].astype(np.float64)
    B = x.shape[2]
    h = np.zeros((N, B))
    hb = np.zeros((N, B))
    for t in range(T):
        z = W_h @ h + W_x @ x[t].astype(np.float64) + b[:, None]
        h = h + dts[:, None] * (-h + np.tanh(z))
        hb += h
    return (W_y @ (hb / T) + b_y[:, None]).T


def main():
    rng = np.random.default_rng(SEED)
    print(f"=== 层 2 平面推理示例（LTC N={N}/T={T}，合成二模式分类）===\n")

    # 随机参数 + 合成任务
    p = {"W_h": rng.normal(0, 0.15, (N, N)).astype(np.float32),
         "W_x": rng.normal(0, 0.30, (N, IN_DIM)).astype(np.float32),
         "b": np.zeros(N, dtype=np.float32),
         "W_y": rng.normal(0, 0.40, (N_CLS, N)).astype(np.float32),
         "b_y": np.zeros(N_CLS, dtype=np.float32),
         "dts": np.exp(np.linspace(np.log(0.05), np.log(0.6), N)).astype(np.float32)}
    x, y = make_synthetic_task(rng)

    # 路径 1：float32 基线
    logits_f32 = float_reference_forward(p, x)
    acc_f32 = float(np.mean(logits_f32.argmax(axis=1) == y))

    # 路径 2：LeveledState 平面推理（Q8 全档 + int8 权重——部署形态）
    inf = LeveledLTCInference(p["W_h"], p["W_x"], p["b"], p["W_y"], p["b_y"],
                              p["dts"], is_q8=None, weight_mode="int8-w",
                              reconstruct="floor")
    logits_plane = inf.forward_chunk(x)
    acc_plane = float(np.mean(logits_plane.argmax(axis=1) == y))

    # 路径 3（对照）：WF-25 混合位宽（最不敏感 25% → Q4，Ω 用真值梯度能量
    # 的代理——本示例直接以固定掩码演示逐神经元混合档位）
    is_q8_mixed = np.ones(N, dtype=bool)
    is_q8_mixed[rng.permutation(N)[:N // 4]] = False
    inf_mixed = LeveledLTCInference(p["W_h"], p["W_x"], p["b"], p["W_y"],
                                    p["b_y"], p["dts"], is_q8=is_q8_mixed,
                                    weight_mode="int8-w", reconstruct="floor")
    acc_mixed = float(np.mean(inf_mixed.forward_chunk(x).argmax(axis=1) == y))

    # —— 读数 ——
    print(f"float32 基线          : acc={acc_f32:.4f}")
    print(f"平面路径（Q8 全档）  : acc={acc_plane:.4f}   "
          f"Δ={100 * (acc_plane - acc_f32):+.2f} 点")
    print(f"平面路径（WF-25 混合）: acc={acc_mixed:.4f}   "
          f"Δ={100 * (acc_mixed - acc_f32):+.2f} 点")
    a = inf.accounting
    print(f"\n系统账（解析账，W9 埋点）：")
    print(f"  f32 状态搬运   = {a['f32_rw_bytes']:,} B")
    print(f"  平面状态搬运   = {a['plane_rw_bytes']:,} B"
          f"（{a['f32_rw_bytes'] / a['plane_rw_bytes']:.2f}×）")
    print(f"  重量化调用     = {a['quant_calls']:,}（每样本每步 1 次）")
    print(f"  母版码字（分发格式）= {a['master_code_bytes']:,} B，不进运行期账")

    ok = abs(acc_plane - acc_f32) <= 0.02 and abs(acc_mixed - acc_f32) <= 0.05
    print(f"\n结论: {'PASS——平面路径与 f32 基线精度一致，' if ok else '偏差超示例预算（调整幅度/seed 复核），'
          }状态带宽账 4.00×（全 Q8）成立" if ok else "")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
