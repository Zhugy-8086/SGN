"""HC8 分层坐标矩阵乘实现

阶段 1.3（介入深度 3：全整数路径）的核心运算组件。

设计目标：
  - 在 HC8 坐标空间完成前向矩阵乘，不还原为浮点
  - 实现 int8 × int8 → int16 累加的整数路径
  - 提供整数 ReLU 和定点 Softmax 近似

为什么自实现：
  - HC 库（engine/hc/）只提供标量运算（加/减/比较/移位）
  - 没有矩阵乘 API，必须自行实现
  - pysgn 未编译，无法调用 C 实现

实现思路：
  - HC8 量化值存在 v[0]（[1, 255]，偏移 128 后表示 [-127, 127]）
  - 矩阵乘 C = A @ B：
    - A: m×k HC8 矩阵（权重，已量化）
    - B: k×n HC8 矩阵（激活，已量化）
    - C: m×n HC8 矩阵（输出，需重新量化）
  - 整数累加：sum(a_q * b_q) 在 int32 空间完成
  - 反量化后用 from_double 重新编码到 HC8

性能说明：
  - 纯 Python 实现，比 PyTorch 浮点矩阵乘慢 100-1000 倍
  - 阶段 1.3 的目标是验证精度，不是性能
  - 后续阶段可用 numpy 向量化或 C 扩展优化
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# 将 stage_1_3 目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hc_adapter import HC8, HC8WeightSchema


# ============================================================
# HC8 矩阵表示
# ============================================================

class HC8Matrix:
    """HC8 矩阵：二维 HC8 元素列表 + 量化 scale + 偏移

    Attributes:
        data: 行优先二维列表，data[i][j] 是 HC8 元素
        rows: 行数
        cols: 列数
        scale: 量化 scale（float → int 时的缩放因子）
        offset: 量化偏移（对称量化偏移 128）
    """

    def __init__(
        self,
        data: List[List[HC8]],
        scale: float,
        offset: int = 128,
    ):
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        self.data = data
        self.scale = scale
        self.offset = offset

    def get_int(self, i: int, j: int) -> int:
        """获取 (i, j) 位置的量化整数值（已去偏移）"""
        return self.data[i][j].v[0] - self.offset

    def get_uint(self, i: int, j: int) -> int:
        """获取 (i, j) 位置的无符号量化值（含偏移）"""
        return self.data[i][j].v[0]

    def __repr__(self) -> str:
        return f"HC8Matrix({self.rows}x{self.cols}, scale={self.scale:.6e}, offset={self.offset})"


# ============================================================
# tensor ↔ HC8Matrix 互转
# ============================================================

def tensor_to_hc8_matrix(
    tensor: torch.Tensor,
    schema: Optional[HC8WeightSchema] = None,
) -> HC8Matrix:
    """float 2D tensor → HC8Matrix

    Args:
        tensor: 2D float tensor
        schema: 量化方案（None 表示新建）

    Returns:
        HC8Matrix
    """
    if tensor.dim() != 2:
        raise ValueError(f"tensor_to_hc8_matrix 需要 2D tensor，得到 {tensor.dim()}D")
    if schema is None:
        schema = HC8WeightSchema()

    q_u, scale = schema.quantize(tensor)
    q_u_list = q_u.tolist()

    data = []
    for i in range(q_u.size(0)):
        row = []
        for j in range(q_u.size(1)):
            # v[0] 存量化值，v[1..5] = 0
            row.append(HC8([int(q_u_list[i][j]) & 0xFF, 0, 0, 0, 0, 0]))
        data.append(row)

    return HC8Matrix(data, scale=scale, offset=schema.offset)


def hc8_matrix_to_tensor(
    matrix: HC8Matrix,
    schema: Optional[HC8WeightSchema] = None,
) -> torch.Tensor:
    """HC8Matrix → float 2D tensor

    Args:
        matrix: HC8Matrix
        schema: 量化方案（None 表示新建）

    Returns:
        2D float tensor
    """
    if schema is None:
        schema = HC8WeightSchema()

    q_u = torch.zeros((matrix.rows, matrix.cols), dtype=torch.int32)
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            q_u[i, j] = matrix.get_uint(i, j)

    return schema.dequantize(q_u, matrix.scale)


# ============================================================
# HC8 整数矩阵乘
# ============================================================

def hc_matmul(a: HC8Matrix, b: HC8Matrix) -> HC8Matrix:
    """HC8 矩阵乘：C = A @ B

    整数路径：
      1. 从 HC8 提取量化整数（去偏移）：a_q, b_q ∈ [-127, 127]
      2. 在 int32 空间累加：c_acc = sum(a_q * b_q)
      3. 反量化累加结果：c_float = c_acc * a.scale * b.scale
      4. 重新量化到 HC8：c_hc8 = HC8Matrix(c_float)

    Args:
        a: m×k HC8Matrix
        b: k×n HC8Matrix

    Returns:
        m×n HC8Matrix
    """
    if a.cols != b.rows:
        raise ValueError(
            f"矩阵乘维度不匹配: A is {a.rows}x{a.cols}, B is {b.rows}x{b.cols}"
        )

    m, k, n = a.rows, a.cols, b.cols

    # 步骤 1+2: 整数累加（纯整数路径）
    # c_acc[i][j] = sum_{l} (a_q[i][l] * b_q[l][j])
    c_acc = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0
            for l in range(k):
                a_q = a.get_int(i, l)  # 去偏移后的有符号整数
                b_q = b.get_int(l, j)
                acc += a_q * b_q
            c_acc[i][j] = acc

    # 步骤 3: 反量化累加结果
    # c_float = c_acc * a.scale * b.scale
    out_scale = a.scale * b.scale
    c_float_2d = [[c_acc[i][j] * out_scale for j in range(n)] for i in range(m)]

    # 步骤 4: 重新量化到 HC8
    # 用 torch.tensor 做量化（复用 schema.quantize 的 clamp 逻辑）
    c_tensor = torch.tensor(c_float_2d, dtype=torch.float32)
    schema = HC8WeightSchema()
    q_u, new_scale = schema.quantize(c_tensor)
    q_u_list = q_u.tolist()

    data = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(HC8([int(q_u_list[i][j]) & 0xFF, 0, 0, 0, 0, 0]))
        data.append(row)

    return HC8Matrix(data, scale=new_scale, offset=schema.offset)


def hc_matmul_fast(a: HC8Matrix, b: HC8Matrix) -> HC8Matrix:
    """HC8 矩阵乘（向量化版本，用 PyTorch tensor 做整数累加）

    与 hc_matmul 等价但更快（对大矩阵约 10-50 倍）。
    仍然在整数空间累加，只是用 PyTorch 的 tensor 操作替代 Python 循环。

    v1.3 修复 BUG-3：原版从 HC8Matrix 提取数据到 tensor 仍是 Python 双重循环，
    对于 784×128 大矩阵会成为瓶颈。改为 list comprehension 一次性构造 tensor，
    真正发挥 PyTorch tensor 矩阵乘的加速效果。

    Args:
        a: m×k HC8Matrix
        b: k×n HC8Matrix

    Returns:
        m×n HC8Matrix
    """
    if a.cols != b.rows:
        raise ValueError(
            f"矩阵乘维度不匹配: A is {a.rows}x{a.cols}, B is {b.rows}x{b.cols}"
        )

    m, k, n = a.rows, a.cols, b.cols

    # v1.3 修复 BUG-3：用 list comprehension 一次性提取 v[0] 字节，再批量减偏移
    # 原版用双重循环 a_q[i, j] = a.get_int(i, j) 慢 10-50 倍
    a_q_u = torch.tensor(
        [[a.data[i][j].v[0] for j in range(k)] for i in range(m)],
        dtype=torch.int32,
    )
    b_q_u = torch.tensor(
        [[b.data[i][j].v[0] for j in range(n)] for i in range(k)],
        dtype=torch.int32,
    )
    # 去偏移得到有符号量化整数
    a_q = a_q_u - a.offset
    b_q = b_q_u - b.offset

    # 整数矩阵乘（int32 累加，不会溢出）
    c_acc = a_q @ b_q  # m×n int32

    # 反量化 + 重新量化
    out_scale = a.scale * b.scale
    c_float = c_acc.float() * out_scale

    schema = HC8WeightSchema()
    q_u, new_scale = schema.quantize(c_float)
    q_u_list = q_u.tolist()

    # 输出构造也可以用 list comprehension 加速（但相对矩阵乘本身收益较小）
    data = [
        [HC8([int(q_u_list[i][j]) & 0xFF, 0, 0, 0, 0, 0]) for j in range(n)]
        for i in range(m)
    ]

    return HC8Matrix(data, scale=new_scale, offset=schema.offset)


# ============================================================
# HC8 整数 ReLU
# ============================================================

def hc_relu(x: HC8Matrix) -> HC8Matrix:
    """整数 ReLU：负值变零

    HC8 是无符号偏移表示，q_u = q + 128：
      - q < 0 → q_u < 128 → ReLU 后应为 0 → q_u = 128
      - q >= 0 → q_u >= 128 → ReLU 后不变

    Args:
        x: HC8Matrix

    Returns:
        ReLU 后的 HC8Matrix（新对象，scale/offset 不变）
    """
    data = []
    for i in range(x.rows):
        row = []
        for j in range(x.cols):
            q_u = x.get_uint(i, j)
            # q_u < 128 表示负值，ReLU 后为 0（即 q=0, q_u=128）
            new_q_u = q_u if q_u >= x.offset else x.offset
            row.append(HC8([new_q_u & 0xFF, 0, 0, 0, 0, 0]))
        data.append(row)

    return HC8Matrix(data, scale=x.scale, offset=x.offset)


# ============================================================
# HC8 定点 Softmax 近似
# ============================================================

def hc_softmax(x: HC8Matrix, dim: int = -1, precision_bits: int = 12) -> HC8Matrix:
    """定点 Softmax 近似

    实现：
      1. 从 HC8 提取量化值并反量化到 float
      2. 用浮点计算 softmax（数学正确性）
      3. 重新量化到 HC8

    为什么不用纯整数 softmax：
      - exp() 在整数空间难以精确近似
      - 行业做法（如 TFLite）也是用查找表 + 浮点中间值
      - 阶段 1.3 的目标是验证矩阵乘的整数路径，softmax 用浮点是可接受的

    Args:
        x: HC8Matrix
        dim: softmax 沿哪个维度（仅支持 -1 或 1，即行 softmax）
        precision_bits: 保留参数（当前未使用，留给未来定点实现）

    Returns:
        Softmax 后的 HC8Matrix
    """
    if dim not in (-1, 1):
        raise ValueError(f"hc_softmax 仅支持 dim=-1 或 1，得到 {dim}")

    # 反量化到 float
    x_float = hc8_matrix_to_tensor(x)

    # 沿最后一维做 softmax
    x_float = x_float.float()
    x_max = x_float.max(dim=1, keepdim=True).values
    exp_x = torch.exp(x_float - x_max)
    sum_exp = exp_x.sum(dim=1, keepdim=True)
    softmax_float = exp_x / sum_exp

    # 重新量化到 HC8
    # softmax 输出在 [0, 1]，需要单独的 scale（不能用原 x.scale）
    schema = HC8WeightSchema()
    q_u, new_scale = schema.quantize(softmax_float)
    q_u_list = q_u.tolist()

    data = []
    for i in range(x.rows):
        row = []
        for j in range(x.cols):
            row.append(HC8([int(q_u_list[i][j]) & 0xFF, 0, 0, 0, 0, 0]))
        data.append(row)

    return HC8Matrix(data, scale=new_scale, offset=schema.offset)


# ============================================================
# HC8 整数前向传播：完整 MLP 一层
# ============================================================

def hc_linear(
    x: HC8Matrix,
    weight: HC8Matrix,  # shape: (out_features, in_features)
    bias: Optional[HC8Matrix] = None,  # shape: (1, out_features) 或 (out_features, 1) 或 (1, 1)
) -> HC8Matrix:
    """HC8 整数线性层：Y = X @ W^T + b

    Args:
        x: (batch, in_features) HC8Matrix
        weight: (out_features, in_features) HC8Matrix
        bias: (1, out_features) HC8Matrix，None 表示无偏置
            兼容 (n, 1) 列向量形状（自动转置为行向量）

    Returns:
        (batch, out_features) HC8Matrix

    v1.3 修复 BUG-2：
        - 去掉无意义的 if/else 分支（两分支同代码）
        - bias 反量化用 b.v[0] 替代 b.to_float()，语义明确
          （bias 的 HC8 只用 v[0] 存量化值，v[1..5]=0，to_float() 结果虽正确但语义不清）
        - 支持 (n, 1) 列向量 bias 形状

    v1.3.5 修复 ISSUE-1：
        - 入口增加 shape 断言，避免调用方误传 (in_features, out_features) 时静默出错
        - 期望 weight 形状 (out_features, in_features)，与 PyTorch nn.Linear 约定一致
        - x.cols (in_features) 必须等于 weight.cols (in_features)
        - 若调用方误传 (in_features, out_features)，断言会明确报错而非静默错误
    """
    # v1.3.5 ISSUE-1: 显式 shape 断言，避免静默语义错误
    if x.cols != weight.cols:
        raise ValueError(
            f"hc_linear shape 不匹配: x 是 ({x.rows}, {x.cols}), "
            f"weight 应为 (out_features, {x.cols}) 即 (out_features, in_features), "
            f"实际 weight 是 ({weight.rows}, {weight.cols}). "
            f"若你想传入 (in_features, out_features)，请先转置后再调用。"
        )

    # W^T: (in_features, out_features)
    # X @ W^T: (batch, out_features)
    # 用 PyTorch tensor 转置 HC8Matrix
    w_data_t = [[weight.data[r][c] for r in range(weight.rows)] for c in range(weight.cols)]
    weight_t = HC8Matrix(w_data_t, scale=weight.scale, offset=weight.offset)

    y = hc_matmul_fast(x, weight_t)

    if bias is not None:
        # 统一 bias 为行向量 (1, out_features)
        # 兼容 (1, n) 和 (n, 1) 两种形状
        if bias.rows == 1:
            bias_row = bias.data[0]
        elif bias.cols == 1:
            # 列向量 → 转为行向量
            bias_row = [bias.data[i][0] for i in range(bias.rows)]
        else:
            # 多行多列的 bias 取第一行（不常见，向后兼容）
            bias_row = bias.data[0]

        # 反量化 bias 加到 y 上
        # v1.3 修复 BUG-2：用 b.v[0] 而非 b.to_float()，语义明确
        # bias 的 HC8 只用 v[0] 存量化值（q_u），v[1..5]=0
        # q = q_u - offset, bias_float = q * scale
        y_float = hc8_matrix_to_tensor(y)
        bias_float = torch.tensor(
            [[b.v[0] - bias.offset for b in bias_row]] * y.rows,
            dtype=torch.float32,
        ) * bias.scale
        y_with_bias = y_float + bias_float

        # 重新量化
        schema = HC8WeightSchema()
        q_u, new_scale = schema.quantize(y_with_bias)
        q_u_list = q_u.tolist()

        data = []
        for i in range(y.rows):
            row = []
            for j in range(y.cols):
                row.append(HC8([int(q_u_list[i][j]) & 0xFF, 0, 0, 0, 0, 0]))
            data.append(row)
        y = HC8Matrix(data, scale=new_scale, offset=schema.offset)

    return y


# ============================================================
# 自检函数
# ============================================================

def _self_check() -> None:
    """模块加载时的自检（仅打印信息，不抛异常）"""
    # 1. 小矩阵乘验证
    # A = [[1.0, 2.0], [3.0, 4.0]]
    # B = [[5.0, 6.0], [7.0, 8.0]]
    # C = A @ B = [[19, 22], [43, 50]]
    a_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b_tensor = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    expected = a_tensor @ b_tensor  # [[19, 22], [43, 50]]

    a_hc = tensor_to_hc8_matrix(a_tensor)
    b_hc = tensor_to_hc8_matrix(b_tensor)
    c_hc = hc_matmul(a_hc, b_hc)
    c_back = hc8_matrix_to_tensor(c_hc)

    mse = ((c_back - expected) ** 2).mean().item()
    # v1.3.5 修复 ISSUE-2：收紧 MSE 阈值 5.0 → 0.5
    # 原阈值 5.0 等于"只要不是纯随机噪声都通过"，无法抓住量化逻辑错误
    # 新阈值 0.5：8-bit 量化在 2x2 小矩阵上的实际 MSE 通常在 0.01~0.5 之间
    # 若逻辑写错（如量化溢出、反量化偏移错），MSE 通常 > 5，新阈值能明确抓住
    assert mse < 0.5, f"矩阵乘 MSE 过大：{mse} (expected ~{expected.tolist()}, got ~{c_back.tolist()})"

    # 2. ReLU 验证
    # [[1.0, -2.0], [-3.0, 4.0]] → [[1.0, 0.0], [0.0, 4.0]]
    x_tensor = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])
    x_hc = tensor_to_hc8_matrix(x_tensor)
    r_hc = hc_relu(x_hc)
    r_back = hc8_matrix_to_tensor(r_hc)

    # 验证非负位置保留，负位置变零
    assert r_back[0, 0].item() > 0.5, f"ReLU(1.0) 应为正，得到 {r_back[0, 0]}"
    assert abs(r_back[0, 1].item()) < 0.5, f"ReLU(-2.0) 应为 0，得到 {r_back[0, 1]}"
    assert abs(r_back[1, 0].item()) < 0.5, f"ReLU(-3.0) 应为 0，得到 {r_back[1, 0]}"
    assert r_back[1, 1].item() > 3.5, f"ReLU(4.0) 应为正，得到 {r_back[1, 1]}"

    # 3. Softmax 验证（每行和为 1）
    s_tensor = torch.tensor([[1.0, 2.0, 3.0]])
    s_hc = tensor_to_hc8_matrix(s_tensor)
    softmax_hc = hc_softmax(s_hc)
    softmax_back = hc8_matrix_to_tensor(softmax_hc)
    row_sum = softmax_back.sum(dim=1).item()
    assert abs(row_sum - 1.0) < 0.1, f"Softmax 行和应为 1，得到 {row_sum}"

    # 4. hc_matmul_fast 与 hc_matmul 等价性
    # v1.3.5 ISSUE-2: 同步收紧 fast 版本阈值 5.0 → 0.5
    c_hc_fast = hc_matmul_fast(a_hc, b_hc)
    c_back_fast = hc8_matrix_to_tensor(c_hc_fast)
    mse_fast = ((c_back_fast - expected) ** 2).mean().item()
    assert mse_fast < 0.5, f"hc_matmul_fast MSE 过大：{mse_fast}"

    print(f"[hc_matmul self-check] OK.")
    print(f"  matmul MSE (slow): {mse:.6e}")
    print(f"  matmul MSE (fast): {mse_fast:.6e}")
    print(f"  ReLU([[1,-2],[-3,4]]) ≈ {r_back.tolist()}")
    print(f"  Softmax([[1,2,3]]) row sum = {row_sum:.4f}")


if __name__ == "__main__":
    _self_check()
